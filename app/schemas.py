from pydantic import BaseModel, Field

class credit_risk(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ge=0,
        description="Percentage of unsecured credit utilization."
    )

    age: int = Field(
        ge=18,
        le=110,
        description="Age of applicant."
    )

    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ge=0,
        description="Number of 30-59 day late payments."
    )

    DebtRatio: float = Field(
        ge=0,
        description="Debt ratio of borrower."
    )

    MonthlyIncome: float = Field(
        ge=0,
        description="Monthly income."
    )

    NumberOfOpenCreditLinesAndLoans: int = Field(
        ge=0,
        description="Open loans and credit lines."
    )

    NumberOfTimes90DaysLate: int = Field(
        ge=0,
        description="Number of 90+ day late payments."
    )

    NumberRealEstateLoansOrLines: int = Field(
        ge=0,
        description="Real estate loans or credit lines."
    )

    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ge=0,
        description="Number of 60-89 day late payments."
    )

    NumberOfDependents: int = Field(
        ge=0,
        description="Financial dependents."
    )