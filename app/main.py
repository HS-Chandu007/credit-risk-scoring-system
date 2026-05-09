import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from app.schemas import credit_risk
from app.preprocessing import preprocess_input
from app.pipeline import CreditRiskPipeline


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Production-ready ensemble ML system for credit default risk prediction.",
    version="1.0.0"
)

model = CreditRiskPipeline()
    
templates = Jinja2Templates(directory="app/templates")

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )
    
@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(

    RevolvingUtilizationOfUnsecuredLines: float = Form(...),
    age: int = Form(...),
    NumberOfTime30_59DaysPastDueNotWorse: int = Form(...),
    DebtRatio: float = Form(...),
    MonthlyIncome: float = Form(...),
    NumberOfOpenCreditLinesAndLoans: int = Form(...),
    NumberOfTimes90DaysLate: int = Form(...),
    NumberRealEstateLoansOrLines: int = Form(...),
    NumberOfTime60_89DaysPastDueNotWorse: int = Form(...),
    NumberOfDependents: int = Form(...)
):

    data = {
        "RevolvingUtilizationOfUnsecuredLines": RevolvingUtilizationOfUnsecuredLines,
        "age": age,
        "NumberOfTime30_59DaysPastDueNotWorse": NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio": DebtRatio,
        "MonthlyIncome": MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines": NumberRealEstateLoansOrLines,
        "NumberOfTime60_89DaysPastDueNotWorse": NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents": NumberOfDependents
    }

    df = pd.DataFrame([data])

    processed_df = preprocess_input(df)

    probability = model.predict_proba(processed_df)[0]
    prediction = model.predict(processed_df)[0]

    if probability < 0.1:
        risk_level = "Very Low Risk"

    elif probability < 0.3:
        risk_level = "Low Risk"

    elif probability < 0.5:
        risk_level = "Moderate Risk"

    elif probability < 0.7:
        risk_level = "High Risk"

    else:
        risk_level = "Very High Risk"

    if prediction == 1:
        predicted_label = "Elevated Credit Risk Detected"

    else:
        predicted_label = "Low Credit Risk"

    return f"""
    <div class="result-card">

        <h2>{risk_level}</h2>

        <p><strong>Default Probability:</strong>
        {round(float(probability) * 100, 2)}%</p>

        <p><strong>Prediction:</strong>
        {predicted_label}</p>

    </div>
    """
    