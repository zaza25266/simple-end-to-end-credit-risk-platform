import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import load_params
from src.data.validation import validate_raw_data


def load_and_process_data():
    """
    Loads raw data, validates via Pandera, and returns a stratified
    train/test split of RAW features plus the target vector.

    """
    params = load_params()

    raw_path = params["data"]["raw_path"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]
    target_col = params["project"]["target_column"]

    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)

    print("Running Pandera schema validation...")
    validated_df = validate_raw_data(df)

    print("Splitting into features (X) and target (y)...")
    X = validated_df.drop(columns=[target_col])
    y = validated_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Raw data loaded and split successfully. Transformer will be applied in train.py.")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_and_process_data()