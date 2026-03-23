"""
Minimal FastAPI inference endpoint for the IDS model.

Usage:
    uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict   – classify a single network flow
    GET  /health    – liveness check
    GET  /classes   – list known attack classes
"""

import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config

app = FastAPI(title="IDS Inference API", version="1.0.0")

_model = None
_label_mapping = None
_feature_names = None


@app.on_event("startup")
def _load_artifacts():
    global _model, _label_mapping, _feature_names

    model_path = config.DEFAULT_MODEL
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path} – train first.")

    _model = joblib.load(model_path)

    lm_path = config.DEFAULT_LABEL_MAPPING
    if lm_path.exists():
        _label_mapping = joblib.load(lm_path)
    else:
        _label_mapping = {0: 'BENIGN', 1: 'ATTACK'}

    fn_path = config.DEFAULT_FEATURE_NAMES
    if fn_path.exists():
        _feature_names = joblib.load(fn_path)


class FlowFeatures(BaseModel):
    """Network flow features as a flat dict of feature_name → float."""
    features: dict[str, float]


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


@app.post("/predict", response_model=PredictionResponse)
async def predict(flow: FlowFeatures):
    if _model is None:
        raise HTTPException(503, detail="Model not loaded")

    df = pd.DataFrame([flow.features])

    if _feature_names is not None:
        missing = set(_feature_names) - set(df.columns)
        if missing:
            raise HTTPException(422, detail=f"Missing features: {sorted(missing)}")
        df = df[_feature_names]

    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    pred = int(_model.predict(df)[0])
    proba = _model.predict_proba(df)[0]
    label = _label_mapping.get(pred, str(pred))
    prob_dict = {
        str(_label_mapping.get(i, i)): round(float(p), 6)
        for i, p in enumerate(proba)
    }

    return PredictionResponse(
        prediction=label,
        confidence=round(float(proba.max()), 6),
        probabilities=prob_dict,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/classes")
async def classes():
    if _label_mapping is None:
        raise HTTPException(503, detail="Model not loaded")
    return {"classes": list(_label_mapping.values())}
