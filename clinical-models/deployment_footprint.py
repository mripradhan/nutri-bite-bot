"""
Measure the deployed Model 1 footprint on CPU: artifact size, process memory, and
single-patient inference latency (Reviewer 1, point 10: low-resource deployment).

Runs the same code path the Flask API uses (model1_artifacts.Model1Predictor on CPU).

    python deployment_footprint.py --repeats 200
"""
import argparse
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PATIENT = {
    "age": 58, "sex_male": 1, "has_htn": 1, "has_dm": 0, "has_ckd": 1,
    "serum_sodium": 140, "serum_potassium": 5.1, "creatinine": 1.9, "egfr": 45,
    "hba1c": 5.2, "fbs": 85, "sbp": 162, "dbp": 96, "bmi": 26,
}


def dir_bytes(path: Path, patterns) -> int:
    return sum(f.stat().st_size for p in patterns for f in path.glob(p) if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "models")
    args = ap.parse_args()

    # single-threaded, as in the container
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    import torch
    torch.set_num_threads(1)
    from model1_artifacts import TARGETS, Model1Predictor

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    predictor = Model1Predictor(args.model_dir)
    load_s = time.perf_counter() - t0

    predictor.predict(PATIENT)  # warm-up (lazy allocations)
    times = []
    for _ in range(args.repeats):
        t = time.perf_counter()
        predictor.predict(PATIENT)
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times)
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux

    weights = dir_bytes(args.model_dir / "tabnet", ["*/network.pt", "*/model_params.json"])
    prep = (args.model_dir / "preprocessing.json").stat().st_size
    ifct = (ROOT / "clinical-models" / "ifct_database.csv").stat().st_size

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": "cpu (1 thread)",
        "n_targets": len(TARGETS),
        "artifact_bytes": {
            "tabnet_weights_all_targets": weights,
            "preprocessing_json": prep,
            "ifct_database_csv": ifct,
            "total_model_artifacts": weights + prep,
        },
        "artifact_mb": {
            "tabnet_weights_all_targets": round(weights / 1e6, 2),
            "preprocessing_json": round(prep / 1e6, 2),
            "ifct_database_csv": round(ifct / 1e6, 3),
            "total_model_artifacts": round((weights + prep) / 1e6, 2),
        },
        "model_load_seconds": round(load_s, 2),
        "peak_process_rss_mb": round(peak_rss_mb, 1),
        "rss_growth_during_load_mb": round((peak_rss_mb - rss_before / 1024), 1),
        "inference_ms_per_patient_all_4_targets": {
            "n": int(args.repeats),
            "mean": round(float(times.mean()), 2),
            "median": round(float(np.median(times)), 2),
            "p95": round(float(np.percentile(times, 95)), 2),
            "min": round(float(times.min()), 2),
            "max": round(float(times.max()), 2),
        },
    }
    out_path = args.model_dir / "reports" / "deployment_footprint.json"
    out_path.write_text(json.dumps(out, indent=2))

    inf = out["inference_ms_per_patient_all_4_targets"]
    print(f"artifacts: {out['artifact_mb']['total_model_artifacts']} MB "
          f"(weights {out['artifact_mb']['tabnet_weights_all_targets']} MB for {len(TARGETS)} targets)")
    print(f"IFCT database: {out['artifact_mb']['ifct_database_csv']} MB")
    print(f"model load: {out['model_load_seconds']} s | peak process RSS: {out['peak_process_rss_mb']} MB")
    print(f"inference (4 targets + attributions): mean {inf['mean']} ms, median {inf['median']} ms, p95 {inf['p95']} ms")
    print(f"  ✓ Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
