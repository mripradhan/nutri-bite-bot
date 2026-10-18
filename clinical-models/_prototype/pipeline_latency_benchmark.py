"""
Deployment-footprint benchmark (Reviewer 1 item 4, Reviewer 2 item 14).

Measures, on the machine it runs on and without any network call:
  * artifact size on disk (four TabNet networks + preprocessing + manifest)
  * resident memory after import, after loading the models, and at peak while benchmarking
  * per-patient latency (p50/p95/p99, ms) for each stage:
      - inference_no_explain   preprocessing + monotonic maps + 4 TabNet forward passes
      - inference_with_explain Model1Predictor.predict (adds TabNet attention masks)
      - engine_end_to_end      PortionControlModel.get_recommendations (risk + budget + portions
                               + IFCT lookup + substitutes), the work behind /api/recommend
      - api_recommend          Flask test client POST /api/recommend (only with --with-app)

Outputs artifacts/models/reports/footprint.json and footprint_table.tex.
Run in a clean process so the memory numbers are not inflated by other work:

    python deployment_footprint.py [--runs 1000] [--warmup 50] [--threads N] [--with-app]

For a low-resource proxy, run it inside a constrained container, for example
    docker run --cpus=1 --memory=1g ...   and record the result next to the unconstrained one.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent.parent   # lives in clinical-models/_mine/
MODEL_DIR = ROOT / "artifacts" / "models"
OUT_DIR = MODEL_DIR / "reports"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same illustrative profile and ingredients as usecase_report.py (not a MIMIC record).
PATIENT = {
    "age": 58, "sex_male": 1, "has_htn": 1, "has_dm": 0, "has_ckd": 1,
    "serum_sodium": 140, "serum_potassium": 5.1, "creatinine": 1.9, "egfr": 45,
    "hba1c": 5.2, "fbs": 85, "sbp": 162, "dbp": 96, "bmi": 26,
}
INGREDIENTS = [
    "Green gram dal (Moong dal)", "Paneer (Cottage cheese)", "Egg, whole, boiled", "Rice, milled (white)",
    "Banana, ripe", "Apple", "Pineapple (Ananas)", "Bottle gourd (Lauki)",
]

PROC = psutil.Process(os.getpid())


def rss_mb() -> float:
    return PROC.memory_info().rss / 1024 ** 2


def artifact_bytes() -> dict:
    files = [p for p in MODEL_DIR.rglob("*") if p.is_file() and "reports" not in p.relative_to(MODEL_DIR).parts]
    nets = sum(p.stat().st_size for p in files if "tabnet" in p.relative_to(MODEL_DIR).parts)
    meta = sum(p.stat().st_size for p in files if p.name in {"preprocessing.json", "manifest.json"})
    return {"tabnet_networks_mb": nets / 1024 ** 2, "metadata_mb": meta / 1024 ** 2,
            "total_mb": sum(p.stat().st_size for p in files) / 1024 ** 2, "n_files": len(files)}


def time_ms(fn, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    samples = np.empty(runs)
    peak = rss_mb()
    for i in range(runs):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1000
        if i % 50 == 0:
            peak = max(peak, rss_mb())
    return {"runs": runs, "mean_ms": float(samples.mean()), "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)), "p99_ms": float(np.percentile(samples, 99)),
            "max_ms": float(samples.max()), "peak_rss_mb": peak}


def hardware() -> dict:
    return {"cpu": platform.processor() or platform.machine(), "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1),
            "os": f"{platform.system()} {platform.release()}", "python": platform.python_version()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads (default: library default)")
    ap.add_argument("--with-app", action="store_true", help="also time the Flask /api/recommend route")
    args = ap.parse_args()

    mem = {"process_start_mb": rss_mb()}
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    from model1_artifacts import TARGETS, Model1Predictor
    from train_model2 import PortionControlModel
    mem["after_imports_mb"] = rss_mb()

    with contextlib.redirect_stdout(io.StringIO()):
        engine = PortionControlModel()
    predictor: Model1Predictor = engine.model1.predictor
    mem["after_model_load_mb"] = rss_mb()

    row = np.array([[PATIENT[f] for f in predictor.feature_names]], dtype=float)
    stages = {
        "inference_no_explain": lambda: [predictor.predict_proba_matrix(row, t) for t in TARGETS],
        "inference_with_explain": lambda: predictor.predict(PATIENT),
        "engine_end_to_end": lambda: engine.get_recommendations(PATIENT, INGREDIENTS, include_substitutes=True,
                                                                 use_ledger=False),
    }
    if args.with_app:
        sys.path.insert(0, str(ROOT))
        with contextlib.redirect_stdout(io.StringIO()):
            import app as deployed
        client = deployed.app.test_client()
        stages["api_recommend"] = lambda: client.post(
            "/api/recommend", json={"patient": PATIENT, "ingredients": INGREDIENTS})

    latency = {}
    for name, fn in stages.items():
        with contextlib.redirect_stdout(io.StringIO()):
            latency[name] = time_ms(fn, args.runs, args.warmup)
        print(f"{name:26s} p50 {latency[name]['p50_ms']:8.2f} ms   p95 {latency[name]['p95_ms']:8.2f} ms")
    mem["peak_rss_mb"] = max(v["peak_rss_mb"] for v in latency.values())

    report = {"hardware": hardware(), "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
              "artifacts": artifact_bytes(), "memory_mb": mem, "latency": latency,
              "notes": "single patient per call, CPU only, no network calls (Roboflow/Groq/Supabase excluded); "
                       "memory is process RSS, which includes the Python/torch runtime"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "footprint.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = [(k.replace("_", r"\_"), f"{v['p50_ms']:.1f}", f"{v['p95_ms']:.1f}", f"{v['p99_ms']:.1f}")
            for k, v in latency.items()]
    tex = ["% generated by clinical-models/deployment_footprint.py -- do not edit by hand",
           r"\begin{tabular}{lrrr}", r"\hline", r"Stage & p50 (ms) & p95 (ms) & p99 (ms) \\", r"\hline"]
    tex += [" & ".join(r) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}"]
    (OUT_DIR / "footprint_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    a = report["artifacts"]
    print(f"\nartifacts: networks {a['tabnet_networks_mb']:.2f} MB, metadata {a['metadata_mb']:.3f} MB, "
          f"total {a['total_mb']:.2f} MB")
    print(f"RSS: after imports {mem['after_imports_mb']:.0f} MB, after load {mem['after_model_load_mb']:.0f} MB, "
          f"peak {mem['peak_rss_mb']:.0f} MB")
    print(f"wrote footprint.json / footprint_table.tex to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
