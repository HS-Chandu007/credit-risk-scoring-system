import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30_59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60_89DaysPastDueNotWorse',
    'NumberOfDependents',
    'TotalLatePayments',
    'SevereLateRatio',
    'HighUtilization',
    'LowUtilization',
    'IncomeToDebt',
    'DebtPerPerson',
    'IncomePerDependent',
    'LoanDensity',
    'RealEstateRatio',
    'HasLatePayment',
    'HighDebt',
    'LowIncome'
]

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["TotalLatePayments"] = (
    df["NumberOfTime30_59DaysPastDueNotWorse"] +
    df["NumberOfTime60_89DaysPastDueNotWorse"] +
    df["NumberOfTimes90DaysLate"]
    )

    df["SevereLateRatio"] = df["NumberOfTimes90DaysLate"] / (df["TotalLatePayments"] + 1)

    df["HighUtilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.8).astype(int)
    df["LowUtilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] < 0.3).astype(int)

    df["IncomeToDebt"] = df["MonthlyIncome"] / (df["DebtRatio"] + 1)
    df["DebtPerPerson"] = df["DebtRatio"] / (df["NumberOfDependents"] + 1)

    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    df["LoanDensity"] = df["NumberOfOpenCreditLinesAndLoans"] / (df["age"] + 1)
    df["RealEstateRatio"] = df["NumberRealEstateLoansOrLines"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)

    df["HasLatePayment"] = (df["TotalLatePayments"] > 0).astype(int)
    df["HighDebt"] = (df["DebtRatio"] > 1).astype(int)
    df["LowIncome"] = (df["MonthlyIncome"] < df["MonthlyIncome"].median()).astype(int)
    
    df = df[FEATURE_COLUMNS]

    return df