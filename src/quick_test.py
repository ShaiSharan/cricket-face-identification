import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

MODEL_FILE = "../data/embeddings/classifier_v2.joblib"
LABEL_ENCODER_FILE = "../data/embeddings/label_encoder_v2.joblib"
CONFIDENCE_THRESHOLD = 0.30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=20, device=device, post_process=True)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

clf = joblib.load(MODEL_FILE)
le = joblib.load(LABEL_ENCODER_FILE)


def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    face_tensor = mtcnn(img)

    if face_tensor is None:
        print("No face detected in this image.")
        return

    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()

    probs = clf.predict_proba(embedding)[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    print(f"\nPredictions for {image_path}:")
    for idx in top3_idx:
        name = le.inverse_transform([idx])[0]
        print(f"  {name:20s} — {probs[idx]*100:.1f}%")

    best_idx = top3_idx[0]
    best_prob = probs[best_idx]
    best_name = le.inverse_transform([best_idx])[0]

    if best_prob < CONFIDENCE_THRESHOLD:
        title = f"UNKNOWN / not confident\n(closest guess: {best_name}, {best_prob*100:.1f}%)"
    else:
        title = f"Predicted: {best_name}\nConfidence: {best_prob*100:.1f}%"

    plt.figure(figsize=(5, 6))
    plt.imshow(img)
    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("../data/quick_test_result.png", dpi=150)
    print(f"\nResult image saved to data/quick_test_result.png")
    plt.show()


if __name__ == "__main__":
    test_image_path = "../data/raw_images/images/mohammed_shami/" + __import__("os").listdir("../data/raw_images/images/mohammed_shami")[0]
    predict_image(test_image_path)