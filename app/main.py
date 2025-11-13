from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

from app.schemas import LoanRequest
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Loan Approval API")

app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = Path("app/model/loan_pipeline.joblib")
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve frontend HTML."""
    return FileResponse("static/index.html")

@app.post("/predict")
def predict_loan(input_data: LoanRequest):
    try:
        # Convert input to DataFrame (CRITICAL!)
        input_df = pd.DataFrame([input_data.dict()])

        # Ensure column order matches training (optional but safe)
        expected_columns = [
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "Property_Area"
        ]
        input_df = input_df[expected_columns]

        # Predicting
        prediction = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df)[0].tolist()
        confidence = probability[list(pipeline.classes_).index(prediction)]

        # pred_idx = list(pipeline.classes_).index(prediction)
        # confidence = probability[pred_idx]

        return {
            "Loan_Status": "Approved" if prediction[0] == "Y" else "Rejected",
            "Confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))