import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from player_info import PLAYER_INFO

st.set_page_config(page_title="Cricket Player Face ID", layout="wide", page_icon="🏏")

MODEL_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings", "classifier_v2.joblib")
LABEL_ENCODER_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings", "label_encoder_v2.joblib")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_images", "images")
CONFIDENCE_THRESHOLD = 0.45


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn_single = MTCNN(image_size=160, margin=20, device=device, post_process=True)
    mtcnn_multi = MTCNN(image_size=160, margin=20, device=device, post_process=True, keep_all=True)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    clf = joblib.load(MODEL_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)
    return mtcnn_single, mtcnn_multi, resnet, clf, le, device


mtcnn_single, mtcnn_multi, resnet, clf, le, device = load_models()


def pretty_name(raw_name):
    return raw_name.replace("_", " ").replace(".", "").title()


def identify_face(face_crop_tensor):
    """Run one cropped face through the embedder + classifier."""
    with torch.no_grad():
        embedding = resnet(face_crop_tensor.unsqueeze(0).to(device)).cpu().numpy()
    probs = clf.predict_proba(embedding)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    best_idx = top3_idx[0]
    best_prob = probs[best_idx]
    best_raw = le.inverse_transform([best_idx])[0]
    return best_raw, best_prob, top3_idx, probs


def show_player_card(raw_name, confidence):
    info = PLAYER_INFO.get(raw_name, {})
    name = pretty_name(raw_name)
    sample_folder = os.path.join(IMAGES_DIR, raw_name)
    sample_img = None
    if os.path.isdir(sample_folder):
        files = [f for f in os.listdir(sample_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            sample_img = os.path.join(sample_folder, files[0])

    card_col1, card_col2 = st.columns([1, 2])
    with card_col1:
        if sample_img:
            st.image(sample_img, caption="Reference photo", use_container_width=True)
    with card_col2:
        st.markdown(f"### {name}")
        if info.get("nickname"):
            st.caption(f"\"{info['nickname']}\"")
        st.write(f"**Role:** {info.get('role', 'Unknown')}")
        st.write(f"**Batting style:** {info.get('style', 'Unknown')}")
        st.progress(float(confidence), text=f"Confidence: {confidence*100:.1f}%")


# --- Sidebar: player gallery ---
st.sidebar.header("🏏 Players in the Database")
st.sidebar.caption("This system can recognize these 15 players.")
for raw_name in sorted(PLAYER_INFO.keys()):
    folder = os.path.join(IMAGES_DIR, raw_name)
    if os.path.isdir(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            with st.sidebar.expander(pretty_name(raw_name)):
                st.image(os.path.join(folder, files[0]), use_container_width=True)
                st.caption(PLAYER_INFO[raw_name].get("role", ""))

st.title("🏏 Indian Cricketer Face Identification")
st.caption("Upload a photo, batch of photos, a group photo, or use your webcam — the system identifies known players and flags unknowns.")

mode = st.radio(
    "Choose input mode:",
    ["📷 Upload Photo", "📁 Batch Upload (multiple photos)", "👥 Group Photo (multiple faces)", "🎥 Webcam"],
    horizontal=True,
)

# ================= SINGLE PHOTO MODE =================
if mode == "📷 Upload Photo":
    uploaded_file = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Uploaded photo", use_container_width=True)

        with col2:
            with st.spinner("Detecting face and identifying player..."):
                face_tensor = mtcnn_single(img)

                if face_tensor is None:
                    st.error("⚠️ No face detected. Try a clearer, front-facing photo.")
                else:
                    best_raw, best_prob, top3_idx, probs = identify_face(face_tensor)

                    if best_prob < CONFIDENCE_THRESHOLD:
                        st.warning(f"🤷 **Unknown / Not confident**\n\nClosest guess: "
                                   f"*{pretty_name(best_raw)}* at {best_prob*100:.1f}% — "
                                   f"below the threshold for a reliable match.")
                    else:
                        st.success("✅ Match found!")
                        show_player_card(best_raw, best_prob)

                    st.markdown("---")
                    st.markdown("**Top 3 matches:**")
                    for idx in top3_idx:
                        name = pretty_name(le.inverse_transform([idx])[0])
                        st.write(name)
                        st.progress(float(probs[idx]))
    else:
        st.info("👆 Upload a photo to get started.")

# ================= BATCH UPLOAD MODE =================
elif mode == "📁 Batch Upload (multiple photos)":
    st.caption("Upload several individual photos at once — each one will be identified separately.")
    uploaded_files = st.file_uploader(
        "Upload multiple photos (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch",
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} photo(s) uploaded.** Processing...")

        results = []
        progress_bar = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):
            img = Image.open(uploaded_file).convert("RGB")
            face_tensor = mtcnn_single(img)

            if face_tensor is None:
                results.append({
                    "filename": uploaded_file.name,
                    "image": img,
                    "label": "No face detected",
                    "confidence": None,
                })
            else:
                best_raw, best_prob, _, _ = identify_face(face_tensor)
                if best_prob < CONFIDENCE_THRESHOLD:
                    label = f"Unknown (closest: {pretty_name(best_raw)})"
                else:
                    label = pretty_name(best_raw)
                results.append({
                    "filename": uploaded_file.name,
                    "image": img,
                    "label": label,
                    "confidence": best_prob,
                })

            progress_bar.progress((i + 1) / len(uploaded_files))

        st.success(f"Done! Processed {len(results)} photos.")
        st.markdown("---")

        cols_per_row = 3
        for i in range(0, len(results), cols_per_row):
            row_results = results[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, result in zip(cols, row_results):
                with col:
                    st.image(result["image"], use_container_width=True)
                    if result["confidence"] is not None:
                        if "Unknown" in result["label"]:
                            st.warning(f"**{result['label']}**")
                        else:
                            st.success(f"**{result['label']}**\n\n{result['confidence']*100:.1f}% confidence")
                    else:
                        st.error(f"**{result['label']}**")
                    st.caption(result["filename"])

        st.markdown("---")
        st.markdown("### Summary")
        summary_rows = [
            {"File": r["filename"], "Prediction": r["label"],
             "Confidence": f"{r['confidence']*100:.1f}%" if r["confidence"] is not None else "N/A"}
            for r in results
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.info("👆 Upload multiple photos — each will be processed and identified individually.")

# ================= GROUP PHOTO MODE =================
elif mode == "👥 Group Photo (multiple faces)":
    st.caption("Upload a photo with multiple people — every detected face will be labeled.")
    uploaded_file = st.file_uploader("Upload a group photo (JPG/PNG)", type=["jpg", "jpeg", "png"], key="group")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Detecting and identifying all faces..."):
            boxes, _ = mtcnn_multi.detect(img)
            face_tensors = mtcnn_multi(img)

            if boxes is None or face_tensors is None:
                st.error("⚠️ No faces detected in this image.")
            else:
                draw_img = img.copy()
                draw = ImageDraw.Draw(draw_img)
                results = []

                for i, box in enumerate(boxes):
                    face_tensor = face_tensors[i]
                    best_raw, best_prob, _, _ = identify_face(face_tensor)

                    if best_prob < CONFIDENCE_THRESHOLD:
                        label = "Unknown"
                        color = "red"
                    else:
                        label = pretty_name(best_raw)
                        color = "lime"

                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw.text((x1, max(0, y1 - 18)), f"{label} ({best_prob*100:.0f}%)", fill=color)
                    results.append((label, best_prob))

                st.image(draw_img, caption=f"{len(boxes)} face(s) detected", use_container_width=True)

                st.markdown("### Detected people:")
                for label, prob in results:
                    st.write(f"- **{label}** ({prob*100:.1f}% confidence)")

# ================= WEBCAM MODE =================
else:
    st.caption("Take a photo with your webcam to identify.")
    camera_image = st.camera_input("Take a photo")

    if camera_image:
        img = Image.open(camera_image).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Captured photo", use_container_width=True)

        with col2:
            with st.spinner("Identifying..."):
                face_tensor = mtcnn_single(img)
                if face_tensor is None:
                    st.error("⚠️ No face detected.")
                else:
                    best_raw, best_prob, top3_idx, probs = identify_face(face_tensor)
                    if best_prob < CONFIDENCE_THRESHOLD:
                        st.warning(f"🤷 Unknown — closest guess: {pretty_name(best_raw)} ({best_prob*100:.1f}%)")
                    else:
                        st.success("✅ Match found!")
                        show_player_card(best_raw, best_prob)