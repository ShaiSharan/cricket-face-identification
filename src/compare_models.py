import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, normalize

EMBEDDINGS_FILE = "../data/embeddings/player_embeddings.csv"


def load_data():
    df = pd.read_csv(EMBEDDINGS_FILE)
    embedding_cols = [c for c in df.columns if c.startswith("e")]
    X = df[embedding_cols].to_numpy(dtype=np.float64)
    y = df["player"].astype(str).to_numpy()
    return X, y


def main():
    X, y = load_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Normalize embeddings so cosine-like distance works well for kNN
    X_norm = normalize(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "SVM (linear, balanced)": SVC(kernel="linear", class_weight="balanced", random_state=42),
        "SVM (rbf, balanced)": SVC(kernel="rbf", class_weight="balanced", random_state=42),
        "kNN (k=3, cosine via normalized+euclidean)": KNeighborsClassifier(n_neighbors=3, metric="euclidean"),
        "kNN (k=5, cosine via normalized+euclidean)": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
    }

    print("Comparing models with 5-fold stratified cross-validation:\n")
    results = {}
    for name, model in models.items():
        data = X_norm if "kNN" in name else X
        scores = cross_val_score(model, data, y_enc, cv=cv, scoring="accuracy")
        results[name] = scores.mean()
        print(f"{name:45s} -> {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")

    best = max(results, key=results.get)
    print(f"\nBest approach: {best} ({results[best]*100:.1f}% average accuracy)")


if __name__ == "__main__":
    main()