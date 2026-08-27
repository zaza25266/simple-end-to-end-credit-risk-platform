import pandera as pa
from pandera import Column, Check

# Define strict Pandera schema for the Give Me Some Credit raw dataset
raw_data_schema = pa.DataFrameSchema(
    {
    "SeriousDlqin2yrs": Column(int, Check.isin([0, 1]), nullable=False),
    "RevolvingUtilizationOfUnsecuredLines": Column(float, Check.ge(0), nullable=False),
    "age": Column(int, Check.between(0, 120), nullable=False),
    "NumberOfTime30-59DaysPastDueNotWorse": Column(int, Check.ge(0), nullable=False),
    "DebtRatio": Column(float, Check.ge(0), nullable=False),
    "MonthlyIncome": Column(float, Check.ge(0), nullable=True),
    "NumberOfOpenCreditLinesAndLoans": Column(int, Check.ge(0), nullable=False),
    "NumberOfTimes90DaysLate": Column(int, Check.ge(0), nullable=False),
    "NumberRealEstateLoansOrLines": Column(int, Check.ge(0), nullable=False),
    "NumberOfTime60-89DaysPastDueNotWorse": Column(int, Check.ge(0), nullable=False),
    "NumberOfDependents": Column(float, Check.ge(0), nullable=True),
    },
    strict=False,  # Allows extra columns if any exist without failing immediately
    coerce=True    # Automatically cast types if safe
)

def validate_raw_data(df):
    """
    Validates the raw dataframe against the Pandera schema.
    Raises SchemaError if validation fails.
    """
    return raw_data_schema.validate(df, lazy=True)