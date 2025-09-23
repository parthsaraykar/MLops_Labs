# src/main.py
from pathlib import Path
from typing import Optional, Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, confloat

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "penguin_clf.joblib"

app = FastAPI(title="Penguin Classifier API", version="1.0")

class PenguinFeatures(BaseModel):
    # Make fields optional so FastAPI won't 422 if one is missing
    bill_length_mm: Optional[confloat(gt=0)] = Field(None, example=43.2)
    bill_depth_mm:  Optional[confloat(gt=0)] = Field(None, example=17.1)
    flipper_length_mm: Optional[confloat(gt=0)] = Field(None, example=197)
    body_mass_g: Optional[confloat(gt=0)] = Field(None, example=4200)
    sex: Optional[Literal["male","female"]] = "male"

class PredictResponse(BaseModel):
    species: Literal["Adelie","Chinstrap","Gentoo"]
    proba: float

@app.get("/")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict(x: PenguinFeatures):
    if not MODEL_PATH.exists():
        raise HTTPException(500, "Model not trained. Run `python src/train.py` or POST /train first.")

    # Load the full pipeline (preprocessor + model)
    model = joblib.load(MODEL_PATH)

    # Build a DataFrame with the *original training columns*
    df = pd.DataFrame([{
        "bill_length_mm": x.bill_length_mm,
        "bill_depth_mm": x.bill_depth_mm,
        "flipper_length_mm": x.flipper_length_mm,
        "body_mass_g": x.body_mass_g,
        "sex": x.sex
    }])

    # The pipeline will impute missing, one-hot encode 'sex', and predict
    proba_vec = model.predict_proba(df)[0]
    pred = model.predict(df)[0]
    return {"species": pred, "proba": float(round(proba_vec.max(), 4))}
