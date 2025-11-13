from pydantic import BaseModel

class LoanRequest(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    Property_Area: str