import argparse
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

def main(data_dir):
    X_train = joblib.load(os.path.join(data_dir, "X_train.pkl"))
    X_test  = joblib.load(os.path.join(data_dir, "X_test.pkl"))
    y_train = joblib.load(os.path.join(data_dir, "y_train.pkl"))
    y_test  = joblib.load(os.path.join(data_dir, "y_test.pkl"))

    alpha_list = [0.1, 0.5, 1.0]
    best_f1 = 0
    best_model = None
    best_alpha = None

    # Ini memastikan semua log masuk ke satu Run ID yang sama
    with mlflow.start_run(run_name="MultinomialNB_Tuning") as parent_run:
        mlflow.log_param("model", "MultinomialNB")

        for alpha in alpha_list:
            # Nested run untuk menyimpan tiap percobaan alpha
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

        # LOG MODEL TERBAIK (Di luar loop alpha, tapi di dalam parent run)
        if best_model is not None:
            mlflow.log_param("best_alpha", best_alpha)
            mlflow.log_metric("best_f1", best_f1)
            
            # membuat folder 'best_model' di tab Artifacts
            mlflow.sklearn.log_model(
                sk_model=best_model, 
                artifact_path="best_model",
                registered_model_name="NB_Classifier_Model"
            )
            print(f"Berhasil! Model terbaik (alpha={best_alpha}) telah disimpan.")
            print(f"Run ID: {parent_run.info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.data_dir)