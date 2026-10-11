import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Paths
IMAGES_DIR = "../data/raw_images/images"
EMBEDDINGS_OUT = "../data/embeddings/player_embeddings.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# MTCNN detects and crops the face from a photo
mtcnn = MTCNN(image_size=160, margin=20, device=device, post_process=True)

# InceptionResnetV1 (pretrained on VGGFace2) turns a face crop into a 512-number embedding
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def get_embedding(image_path):
    """
    Detect the face in an image and return its 512-dim embedding.
    Returns None if no face is detected (so we can skip bad images).
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  Could not open {image_path}: {e}")
        return None

    face_tensor = mtcnn(img)
    if face_tensor is None:
        return None  # no face detected in this image

    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0).to(device))

    return embedding.cpu().numpy().flatten()


def build_all_embeddings():
    rows = []
    player_folders = sorted(os.listdir(IMAGES_DIR))

    for player in player_folders:
        player_path = os.path.join(IMAGES_DIR, player)
        if not os.path.isdir(player_path):
            continue

        image_files = [f for f in os.listdir(player_path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"Processing {player}: {len(image_files)} images")

        success_count = 0
        for filename in image_files:
            image_path = os.path.join(player_path, filename)
            embedding = get_embedding(image_path)

            if embedding is not None:
                row = {"player": player, "filename": filename}
                for i, val in enumerate(embedding):
                    row[f"e{i}"] = val
                rows.append(row)
                success_count += 1

        print(f"  -> {success_count}/{len(image_files)} faces detected successfully")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(EMBEDDINGS_OUT), exist_ok=True)
    df.to_csv(EMBEDDINGS_OUT, index=False)
    print(f"\nSaved {len(df)} embeddings to {EMBEDDINGS_OUT}")
    print(f"Players included: {df['player'].nunique()}")


if __name__ == "__main__":
    build_all_embeddings()