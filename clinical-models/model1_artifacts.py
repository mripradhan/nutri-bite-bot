"""
Portable Model 1 artifacts: one format written by train_model1.py and read by
train_model2.py and app.py, so training and deployment cannot drift apart.

Layout (artifacts/models/):
    manifest.json                       provenance, config, metrics
    preprocessing.json                  feature order, imputer medians, scaler params,
                                        per-target isotonic (monotonic) maps
    tabnet/<target>/model_params.json   TabNet architecture
    tabnet/<target>/network.pt          TabNet weights

No joblib/sklearn objects are stored, so loading does not depend on the
sklearn version. Every file is required: a missing artifact is an error.
"""
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

TARGETS = ["sodium_sensitivity", "potassium_sensitivity", "protein_restriction", "carb_sensitivity"]
LABELS = ["low", "moderate", "high"]
ARTIFACT_SCHEMA_VERSION = 2


def _tabnet_dir(model_dir: Path, target: str) -> Path:
    return Path(model_dir) / "tabnet" / target


def save_tabnet(model, model_dir: Path, target: str) -> None:
    out = _tabnet_dir(model_dir, target)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = model.save_model(str(Path(tmp) / target))
        with zipfile.ZipFile(zpath) as zf:
            for name in ("model_params.json", "network.pt"):
                (out / name).write_bytes(zf.read(name))


def load_tabnet(model_dir: Path, target: str):
    from pytorch_tabnet.tab_model import TabNetClassifier

    src = _tabnet_dir(model_dir, target)
    parts = [src / "model_params.json", src / "network.pt"]
    for p in parts:
        if not p.exists():
            raise FileNotFoundError(f"Missing Model 1 artifact: {p}")
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / f"{target}.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            for p in parts:
                zf.write(p, p.name)
        model = TabNetClassifier(device_name="cpu")
        model.load_model(str(zpath))
    return model


def full_proba(model, X: np.ndarray) -> np.ndarray:
    """TabNet only emits columns for classes seen in training; place them in the 3-class layout."""
    proba = model.predict_proba(X)
    full = np.zeros((len(X), len(LABELS)))
    for col, cls in model.preds_mapper.items():
        full[:, int(cls)] = proba[:, int(col)]
    return full


def apply_monotonic(X: np.ndarray, feature_names: List[str], maps: Dict[str, Dict[str, List[float]]]) -> np.ndarray:
    """Equivalent to sklearn IsotonicRegression(out_of_bounds='clip').transform per feature."""
    X = X.copy()
    for f, m in maps.items():
        i = feature_names.index(f)
        X[:, i] = np.interp(X[:, i], m["x"], m["y"])
    return X


class Model1Predictor:
    """Loads a complete Model 1 artifact set and produces per-target risk outputs."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        manifest_path = self.model_dir / "manifest.json"
        prep_path = self.model_dir / "preprocessing.json"
        for p in (manifest_path, prep_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing Model 1 artifact: {p}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact schema {self.manifest.get('schema_version')}")
        prep = json.loads(prep_path.read_text())
        self.feature_names: List[str] = prep["feature_names"]
        self.medians = np.asarray(prep["imputer_median"], dtype=float)
        self.mean = np.asarray(prep["scaler_mean"], dtype=float)
        self.scale = np.asarray(prep["scaler_scale"], dtype=float)
        self.monotonic: Dict[str, Dict] = prep["monotonic"]
        missing = [t for t in TARGETS if t not in self.monotonic]
        if missing:
            raise ValueError(f"preprocessing.json has no monotonic map for: {missing}")
        self.models = {t: load_tabnet(self.model_dir, t) for t in TARGETS}

    def predict_proba_matrix(self, X_raw: np.ndarray, target: str) -> np.ndarray:
        """Batch class probabilities for raw feature rows (columns in self.feature_names order, NaN = missing)."""
        X = np.where(np.isnan(X_raw), self.medians, X_raw)
        X = apply_monotonic((X - self.mean) / self.scale, self.feature_names, self.monotonic[target])
        return full_proba(self.models[target], X.astype(np.float32))

    def _scaled(self, patient: Dict[str, Any]) -> np.ndarray:
        x = np.array([np.nan if patient.get(f) is None else float(patient[f]) for f in self.feature_names])
        x = np.where(np.isnan(x), self.medians, x)
        return ((x - self.mean) / self.scale)[None, :]

    def predict(self, patient: Dict[str, Any]) -> Dict[str, dict]:
        base = self._scaled(patient)
        out = {}
        for target, model in self.models.items():
            X = apply_monotonic(base, self.feature_names, self.monotonic[target]).astype(np.float32)
            proba = full_proba(model, X)[0]
            proba = proba / proba.sum()
            pred = int(np.argmax(proba))
            explain, _ = model.explain(X)
            attr = explain[0]
            attr = attr / attr.sum() if attr.sum() > 0 else np.zeros_like(attr)
            ranked = sorted(zip(self.feature_names, attr), key=lambda kv: kv[1], reverse=True)[:5]
            out[target] = {
                "label": LABELS[pred],
                "severity_score": round(float(np.dot(proba, [0.0, 1.0, 2.0])), 4),
                "confidence": round(float(proba.max()), 4),
                "proba": {lbl: round(float(p), 4) for lbl, p in zip(LABELS, proba)},
                "feature_attribution": {f: round(float(w), 4) for f, w in ranked},
            }
        return out
