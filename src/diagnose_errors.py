import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

EMBEDDINGS_FILE = "../data/embeddings/player_embeddings.csv"
MODEL_FILE = "../data/embeddings/classifier.joblib"
LABEL_ENCODER_FILE = "../data/embeddings/label_encoder.joblib"


def load_data():
    df = pd.read_csv(EMBEDDINGS_FILE)
    embedding_cols = [c for c in df.columns if c.startswith("e")]
    X = df[embedding_cols].to_numpy(dtype=np.float64)
    y = df["player"].astype(str).to_numpy()
    return X, y, df


def main():
    X, y, df = load_data()
    clf = joblib.load(MODEL_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=42
    )

    y_test_enc = le.transform(y_test)
    y_pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    # How many training images does each player actually have?
    print("=== Training images per player ===")
    print(pd.Series(y_train).value_counts())

    print("\n=== Misclassified test images ===")
    test_df = df.loc[idx_test].reset_index(drop=True)
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            print(f"  TRUE: {y_test[i]:20s} | PREDICTED: {y_pred[i]:20s} | file: {test_df.iloc[i]['filename']}")


if __name__ == "__main__":
    main()