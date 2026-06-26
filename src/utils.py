# Used to save the model predictions into csv file

import pandas as pd

def save_predictions(X_test, y_test, y_pred, path="outputs/predictions.csv"):
    df = pd.DataFrame(X_test.copy())
    df["actual"] = y_test
    df["predicted"] = y_pred

    df.to_csv(path, index=False)