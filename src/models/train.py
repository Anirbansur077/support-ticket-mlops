import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier


def load_data():
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    return train, val


def build_features(train, val):
    train = train.copy()
    val = val.copy()
    train["text"] = train["subject"].fillna("") + " " + train["body"].fillna("")
    val["text"] = val["subject"].fillna("") + " " + val["body"].fillna("")

    tfidf = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    X_train_text = tfidf.fit_transform(train["text"])
    X_val_text = tfidf.transform(val["text"])

    cat_cols = ["type", "queue"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_train_cat = encoder.fit_transform(train[cat_cols])
    X_val_cat = encoder.transform(val[cat_cols])

    tag_cols = [f"tag_{i}" for i in range(1, 9)]
    train_tag_lists = train[tag_cols].apply(lambda row: [t for t in row if pd.notna(t)], axis=1)
    val_tag_lists = val[tag_cols].apply(lambda row: [t for t in row if pd.notna(t)], axis=1)

    mlb_full = MultiLabelBinarizer()
    train_tags_full = mlb_full.fit_transform(train_tag_lists)
    freq = np.asarray(train_tags_full).sum(axis=0)
    tag_freq = pd.Series(freq, index=mlb_full.classes_).sort_values(ascending=False)
    top_tags = tag_freq.head(100).index.tolist()

    mlb = MultiLabelBinarizer(classes=top_tags)
    X_train_tags = mlb.fit_transform(train_tag_lists)
    X_val_tags = mlb.transform(val_tag_lists)

    X_train = hstack([X_train_text, X_train_cat, csr_matrix(X_train_tags)])
    X_val = hstack([X_val_text, X_val_cat, csr_matrix(X_val_tags)])

    artifacts = {"tfidf": tfidf, "encoder": encoder, "mlb": mlb, "top_tags": top_tags}
    return X_train, X_val, artifacts


def train_and_evaluate(X_train, y_train, X_val, y_val):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    metrics = {
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
        "f1": f1_score(y_val, preds),
        "pr_auc": average_precision_score(y_val, probs),
        "roc_auc": roc_auc_score(y_val, probs),
    }
    return model, metrics


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
    mlflow.set_experiment("ticket-escalation-prediction")

    train, val = load_data()
    X_train, X_val, artifacts = build_features(train, val)

    y_train = train["escalated"]
    y_val = val["escalated"]

    with mlflow.start_run(run_name="automated_retrain"):
        model, metrics = train_and_evaluate(X_train, y_train, X_val, y_val)

        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 400)
        mlflow.log_param("max_depth", 8)
        mlflow.log_param("learning_rate", 0.05)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.xgboost.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id

        print(f"Run ID: {run_id}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")

    return run_id, metrics["pr_auc"]


def promote_if_better(run_id, new_pr_auc, model_name="ticket-escalation-model", metric_threshold=0.0):
    from mlflow import MlflowClient

    client = MlflowClient()

    try:
        current_prod = client.get_model_version_by_alias(model_name, "production")
        current_run = client.get_run(current_prod.run_id)
        current_pr_auc = current_run.data.metrics.get("pr_auc", 0.0)
        print(f"Current production PR-AUC: {current_pr_auc:.4f}")
    except Exception:
        print("No current production model found — this will become the first one.")
        current_pr_auc = 0.0

    print(f"New model PR-AUC: {new_pr_auc:.4f}")

    if new_pr_auc > current_pr_auc + metric_threshold:
        result = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
        client.set_registered_model_alias(model_name, "production", result.version)
        print(f"PROMOTED: new model (v{result.version}) is better ({new_pr_auc:.4f} > {current_pr_auc:.4f})")
        return True
    else:
        print(f"NOT PROMOTED: new model ({new_pr_auc:.4f}) does not beat production ({current_pr_auc:.4f})")
        return False


if __name__ == "__main__":
    run_id, pr_auc = main()
    promote_if_better(run_id, pr_auc)