from pydantic import BaseModel, Field, conint, confloat, ConfigDict


class LoanApplicationRequest(BaseModel):
    """
    Input payload for a single loan applicant.
    
    """

    model_config = ConfigDict(populate_by_name=True)

    RevolvingUtilizationOfUnsecuredLines: confloat(ge=0.0) = Field(
        ..., description="Credit card balance divided by total limit"
    )
    age: conint(ge=18, le=120) = Field(
        ..., description="Applicant age in years"
    )
    NumberOfTime30_59DaysPastDueNotWorse: conint(ge=0) = Field(
        ..., alias="NumberOfTime30-59DaysPastDueNotWorse",
        description="Times 30-59 days past due"
    )
    DebtRatio: confloat(ge=0.0) = Field(
        ..., description="Monthly debt payments / gross income"
    )
    MonthlyIncome: confloat(ge=0.0) = Field(
        ..., description="Monthly gross income in USD"
    )
    NumberOfOpenCreditLinesAndLoans: conint(ge=0) = Field(
        ..., description="Open loans and lines of credit"
    )
    NumberOfTimes90DaysLate: conint(ge=0) = Field(
        ..., description="Times 90+ days past due"
    )
    NumberRealEstateLoansOrLines: conint(ge=0) = Field(
        ..., description="Mortgage and equity lines"
    )
    NumberOfTime60_89DaysPastDueNotWorse: conint(ge=0) = Field(
        ..., alias="NumberOfTime60-89DaysPastDueNotWorse",
        description="Times 60-89 days past due"
    )
    NumberOfDependents: conint(ge=0) = Field(
        0, description="Dependents excluding the borrower"
    )


class PredictionResponse(BaseModel):
    default_probability: float    # raw model output, 0.0–1.0
    decision_threshold: float
    prediction: str               # "Approved" or "Flagged for Default Risk"
    risk_score_percentage: float  # default_probability * 100