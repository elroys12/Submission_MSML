import argparse
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score


def main(data_dir):

    # ❌ JANGAN set tracking_uri
    # ❌ JANGAN set_experiment
    # ❌ JANGAN start_run (parent sudah dibuat oleh MLflow Projects)

    X_train = joblib.load(os.path.join(data_dir, "X_train.pkl"))
    X_test  = joblib.load(os.path.join(data_dir, "X_test.pkl"))
    y_train = joblib.load(os.path.join(data_dir, "y_train.pkl"))
    y_test  = joblib.load(os.path.join(data_dir, "y_test.pkl"))

    alpha_list = [0.1, 0.5, 1.0]

    best_f1 = 0
    best_model = None
    best_alpha = None

    mlflow.log_param("model", "MultinomialNB")

    for alpha in alpha_list:
        with mlflow.start_run(run_name=f"alpha_{alpha}", nested=True):

            mlflow.log_param("alpha", alpha)

            model = MultinomialNB(alpha=alpha)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metric("f1", f1)

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_alpha = alpha

    mlflow.log_param("best_alpha", best_alpha)
    mlflow.log_metric("best_f1", best_f1)
    mlflow.sklearn.log_model(best_model, artifact_path="best_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.data_dir)
