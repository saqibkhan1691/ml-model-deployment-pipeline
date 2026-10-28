from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model.joblib"
FEATURES_PATH = ROOT / "feature_names.json"

app = FastAPI(
    title="Breast Cancer Classification API",
    description="Academic demo API for a Kaggle breast-cancer classification model. Not for medical use.",
    version="1.0.0",
)

# Fail clearly if training artifacts have not been generated yet.
if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    raise RuntimeError(
        "Model artifacts missing. Run train_model.ipynb first and ensure "
        "model.joblib and feature_names.json are in the project root."
    )

model = joblib.load(MODEL_PATH)
feature_names = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Exactly the 30 feature names and numeric values expected by the trained model.",
        examples=[{"radius_mean": 14.0, "texture_mean": 20.0}],
    )


@app.get("/")
def home():
    return {"message": "Breast Cancer Classification API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "feature_count": len(feature_names)}


@app.post("/predict")
def predict(request: PredictionRequest):
    supplied = request.features
    missing = sorted(set(feature_names) - set(supplied))
    extra = sorted(set(supplied) - set(feature_names))

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": extra},
        )

    try:
        row = pd.DataFrame([[float(supplied[name]) for name in feature_names]], columns=feature_names)
        prediction = str(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        probability_by_class = {
            str(label): round(float(prob), 6)
            for label, prob in zip(model.classes_, probabilities)
        }
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid numeric feature value: {exc}") from exc

    label_names = {"B": "benign", "M": "malignant"}
    return {
        "prediction": prediction,
        "prediction_label": label_names.get(prediction, prediction),
        "probabilities": probability_by_class,
        "disclaimer": "Academic demonstration only; not a medical diagnosis.",
    }
