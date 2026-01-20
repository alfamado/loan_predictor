from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

from app.schemas import LoanRequest
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(title="Loan Approval API")

app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = Path("app/model/loan_pipeline_CatBoost.joblib")
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve frontend HTML."""
    return FileResponse("static/index.html")

@app.post("/predict")
def predict(data: LoanRequest):
    try:
        df = pd.DataFrame([data.dict()])
        df = df[[
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "Property_Area"
        ]]


        cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
        for col in cat_cols:
            df[col] = df[col].fillna("Missing").astype(str)
        df["Credit_History"] = df["Credit_History"].fillna(-1).astype(str)

        pred = pipeline.predict(df)[0]

        pred = str(pred)
        proba = pipeline.predict_proba(df)[0]

        return {
            "Loan_Status": "Approved" if pred == "Y" else "Rejected",
            "Confidence": round(float(proba.max()) * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}