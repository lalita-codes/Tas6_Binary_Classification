import pandas as pd
import joblib
import os

# Load model + feature names
model = joblib.load("models/model.pkl")
feature_columns = joblib.load("models/features.pkl")


def predict_single(input_data: list):
    """
    Predict single transaction
    """

    # Safety check
    if len(input_data) != len(feature_columns):
        raise ValueError(
            f"Expected {len(feature_columns)} features, got {len(input_data)}"
        )

    df = pd.DataFrame([input_data], columns=feature_columns)
    prediction = model.predict(df)

    return prediction[0]


def predict_from_csv(file_path, output_path="outputs/predictions_new.csv"):
    """
    Predict multiple rows from CSV
    """

    df = pd.read_csv(file_path)

    predictions = model.predict(df)

    df["prediction"] = predictions

    os.makedirs("outputs", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path}")

    return df


# ---------------- TEST ----------------
if __name__ == "__main__":

    # CORRECT sample size = 30 features
    sample = [0] * 28 + [100, 0]   # 28 V features + Amount + Time

    print("Prediction:", predict_single(sample))