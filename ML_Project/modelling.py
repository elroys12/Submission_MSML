import pandas as pd
import mlflow
import mlflow.sklearn
import os
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. LOAD DATA
# Pastikan folder ini ada di dalam folder MLProject Anda di GitHub
DATA_DIR = "SpamEmail_preprocessing"

print("Memuat data dari CSV...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

# 2. TRAINING & AUTOLOGGING
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="CI_Training_Run"):
    # Gunakan alpha dari parameter atau default 1.0
    alpha_param = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    
    model = MultinomialNB(alpha=alpha_param)
    print(f"Sedang melatih model dengan alpha={alpha_param}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrik
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("-" * 30)
    print(f" Accuracy  : {acc:.4f}")
    print(f" F1-score  : {f1:.4f}")
    print("-" * 30)
    
    # Log manual tambahan jika diperlukan
    mlflow.log_param("alpha", alpha_param)
    mlflow.log_metric("accuracy", acc)