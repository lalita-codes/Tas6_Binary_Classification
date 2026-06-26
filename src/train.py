from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from src.logs import get_logger

logger = get_logger()


def train_model(df, target_column, config):

    # =========================
    # Split features and target
    # =========================
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # =========================
    # Train-test split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info(f"Training samples: {X_train.shape}")
    logger.info(f"Test samples: {X_test.shape}")

    # =========================
    # Feature Scaling
    # =========================
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================
    # Select model from config
    # =========================
    model_type = config["model"]["model_type"]   #  Fixed
    print("Model from CONFIG:",model_type)


    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=config["models"]["logistic_regression"]["max_iter"],
            class_weight=config["models"]["logistic_regression"]["class_weight"]
        )
        model_file = "models/logistic.pkl"

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config["models"]["random_forest"]["n_estimators"],
            max_depth=config["models"]["random_forest"]["max_depth"],
            random_state=config["models"]["random_forest"]["random_state"],
            class_weight=config["models"]["random_forest"]["class_weight"]
        )
        model_file = "models/random_forest.pkl"

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # =========================
    # Train model
    # =========================
    model.fit(X_train_scaled, y_train)

    # =========================
    # Save model + scaler
    # =========================
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, model_file)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(X.columns.tolist(), "models/features.pkl")

    logger.info(f"{model_type} model, scaler and features saved successfully")

    return model, X_train_scaled, X_test_scaled, y_train, y_test