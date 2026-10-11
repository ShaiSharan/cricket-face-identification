import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

EMBEDDINGS_FILE = "../data/embeddings/player_embeddings.csv"
MODEL_FILE = "../data/embeddings/classifier_v2.joblib"
LABEL_ENCODER_FILE = "../data/embeddings/label_encoder_v2.joblib"
CONFIDENCE_THRESHOLD = 0.30


def load_data():
    df = pd.read_csv(EMBEDDINGS_FILE)
    embedding_cols = [c for c in df.columns if c.startswith("e")]
    X = df[embedding_cols].to_numpy(dtype=np.float64)
    y = df["player"].astype(str).to_numpy()
    return X, y


def main():
    X, y = load_data()
    clf = joblib.load(MODEL_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    y_pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    probs = clf.predict_proba(X_test)
    max_probs = probs.max(axis=1)

    correct_mask = (np.array(y_test) == y_pred)

    print("Confidence stats for CORRECT predictions:")
    print(f"  min={max_probs[correct_mask].min():.3f}  "
          f"mean={max_probs[correct_mask].mean():.3f}  "
          f"max={max_probs[correct_mask].max():.3f}")

    print("\nConfidence stats for INCORRECT predictions:")
    if (~correct_mask).sum() > 0:
        print(f"  min={max_probs[~correct_mask].min():.3f}  "
              f"mean={max_probs[~correct_mask].mean():.3f}  "
              f"max={max_probs[~correct_mask].max():.3f}")

    print("\nThreshold sweep:")
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        mask = max_probs >= t
        if mask.sum() == 0:
            continue
        acc = (np.array(y_test)[mask] == y_pred[mask]).mean()
        print(f"  threshold={t:.2f} -> {mask.sum()}/{len(y_test)} covered, accuracy={acc*100:.1f}%")


if __name__ == "__main__":
    main()