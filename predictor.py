# UFC fight outcome predictor using logistic regression
# This is a simple demonstration script. It does not guarantee accuracy or profit.
# The script expects a CSV dataset with the following columns:
#   - fighter_a_stats (numerical features for fighter A)
#   - fighter_b_stats (numerical features for fighter B)
#   - result (1 if fighter A wins, 0 if fighter B wins)
#
# Usage:
#   python predictor.py train data.csv
#   python predictor.py predict data.csv
#
# For `train`, the script trains a logistic regression model and saves it to disk as `model.joblib`.
# For `predict`, the script loads the model and outputs predictions for each row in the dataset.
#
# Note: This is a toy example. Real-world betting involves risk, and this script does not guarantee success.

import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

MODEL_FILE = Path("model.joblib")


def load_data(csv_path: str):
    """Load dataset from a CSV file."""
    df = pd.read_csv(csv_path)
    features = df.drop(columns=["result"])
    target = df["result"]
    return features, target


def train_model(csv_path: str):
    X, y = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    dump(model, MODEL_FILE)
    print(f"Model trained. Accuracy on test set: {accuracy:.2f}")


def predict(csv_path: str):
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    model = load(MODEL_FILE)
    X, _ = load_data(csv_path)
    predictions = model.predict_proba(X)[:, 1]
    for i, p in enumerate(predictions):
        print(f"Fight {i+1}: probability fighter A wins = {p:.2f}")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in {"train", "predict"}:
        print("Usage: python predictor.py [train|predict] <data.csv>")
        sys.exit(1)

    command, csv_path = sys.argv[1], sys.argv[2]

    if command == "train":
        train_model(csv_path)
    elif command == "predict":
        predict(csv_path)


if __name__ == "__main__":
    main()
