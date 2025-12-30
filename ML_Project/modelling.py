import pandas as pd
import mlflow
import mlflow.sklearn
import os
import argparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

# --- Tambahkan ini untuk menangkap parameter dari MLProject ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="SpamEmail_preprocessing")
args = parser.parse_args()

DATA_DIR = args.data_dir

# 1. LOAD DATA
print(f"Memuat data dari folder: {DATA_DIR}")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

# 2. TRAINING & AUTOLOGGING
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="MLProject_CI_Run"):
    model = MultinomialNB()
    print("Sedang melatih model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
    
    # Log metrik secara manual untuk memastikan tercatat di MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)