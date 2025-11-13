# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# from .schemas import LoanRequest

# # Initializing app
# app = FastAPI(title="Loan Approval Prediction API")

# # Loading fitted pipeline
# pipeline = joblib.load(r"C:\Users\NCC200\Desktop\loan_api\app\model\loan_pipeline.joblib")

# # Prediction endpoint
# @app.post("/predict")
# def predict_loan_status(request: LoanRequest):
#     data = pd.DataFrame([request.dict()])
#     prediction = pipeline.predict(data)
#     probability = pipeline.predict_proba(data)[0][1]

#     result = "Approved" if prediction[0] == 1 else "Rejected"
#     return {
#         "Loan_Status": result,
#         "Approval_Probability": round(float(probability), 3)
#     }

# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import joblib
# from schemas import LoanRequest

# app = FastAPI(title="Loan Approval Prediction API")

# # Loading trained pipeline
# pipeline = joblib.load(r"C:\Users\NCC200\Desktop\loan_api\app\model\loan_pipeline.joblib")

# @app.post("/predict")
# def predict_loan(data: LoanRequest):
#     # Convert input to DataFrame
#     df = pd.DataFrame([data.dict()])
    
#     # Predict using the pipeline (no manual preprocessing needed)
#     prediction = pipeline.predict(df)[0]
#     probability = pipeline.predict_proba(df)[0, 1]
    
#     return {"prediction": prediction, "probability": float(probability)}

# from fastapi import FastAPI
# import pandas as pd
# from app.schemas import LoanApplication
# from app.model.train_model import load_model

# app = FastAPI(title="Loan Approval Prediction API")

# # Load the trained model
# model = load_model()

# @app.get("/")
# def home():
#     return {"message": "Loan Approval API is running "}

# @app.post("/predict")
# def predict_loan_status(application: LoanApplication):
#     # Convert request to DataFrame
#     data = pd.DataFrame([application.dict()])

#     # Make prediction
#     prediction = model.predict(data)[0]
#     result = "Approved" if prediction == 1 else "Rejected"

#     return {"Loan_Status": result}

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

from app.schemas import LoanRequest
import joblib
import pandas as pd

app = FastAPI(title="Loan Approval API")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve frontend automatically
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
# templates = Jinja2Templates(directory=FRONTEND_DIR)

# Load trained pipeline
pipeline = joblib.load("app/model/loan_pipeline.joblib")
if not pipeline.exists():
    raise RuntimeError(f"Model not found at {pipeline}")
pipeline = joblib.load(pipeline)

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