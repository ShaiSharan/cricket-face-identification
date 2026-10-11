import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

EMBEDDINGS_FILE = "../data/embeddings/player_embeddings.csv"
MODEL_OUT = "../data/embeddings/classifier_v2.joblib"
LABEL_ENCODER_OUT = "../data/embeddings/label_encoder_v2.joblib"
CONFIDENCE_THRESHOLD = 0.35


def load_data():
    df = pd.read_csv(EMBEDDINGS_FILE)
    embedding_cols = [c for c in df.columns if c.startswith("e")]
    X = df[embedding_cols].to_numpy(dtype=np.float64)
    y = df["player"].astype(str).to_numpy()
    return X, y, df


def train_and_evaluate():
    X, y, df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # class_weight="balanced" makes the model pay more attention to
    # under-represented players instead of treating every photo equally
    clf = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42, C=0.5)
    clf.fit(X_train, y_train_enc)

    y_pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall test accuracy: {accuracy * 100:.1f}%")
    print("\nPer-player report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    probs = clf.predict_proba(X_test)
    max_probs = probs.max(axis=1)
    confident_mask = max_probs >= CONFIDENCE_THRESHOLD
    confident_accuracy = accuracy_score(
        np.array(y_test)[confident_mask], y_pred[confident_mask]
    )
    print(f"\nWith confidence threshold {CONFIDENCE_THRESHOLD}:")
    print(f"  {confident_mask.sum()}/{len(y_test)} confident, accuracy {confident_accuracy*100:.1f}%")

    joblib.dump(clf, MODEL_OUT)
    joblib.dump(le, LABEL_ENCODER_OUT)
    print(f"\nModel saved to {MODEL_OUT}")


if __name__ == "__main__":
    train_and_evaluate()