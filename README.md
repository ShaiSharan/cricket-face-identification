# Indian Cricketer Face Identification

This is my second internship project — a computer vision system that identifies Indian cricketers from photos using face recognition. Instead of training a CNN from scratch like most basic versions of this project do, I used face embeddings (pretrained FaceNet) and built a classifier on top, with a strong focus on actually testing whether the system knows what it doesn't know — not just whether it gets the easy cases right.

## The problem I was solving

Given a photo, identify which Indian cricketer it is, trained on a set of known players. The obvious way to do this is train a CNN on labeled photos and call it done. I wanted to go further and build something that actually behaves responsibly when it sees someone it doesn't recognize, instead of confidently guessing wrong.

## What I focused on

- **Face embeddings instead of training from scratch.** I used a pretrained FaceNet model (InceptionResnetV1, trained on VGGFace2) to convert each face into a 512-number "fingerprint," then trained a simple SVM classifier on top of those fingerprints. This works well even with a modest dataset, since the heavy lifting (understanding what a face even is) was already learned on millions of faces.
- **"Unknown person" rejection.** Most basic face-ID projects will confidently label *any* face as one of the trained people. Mine checks a confidence threshold and says "unknown" when it isn't sure — and I actually tested this on real outsider photos, not just assumed it works (see below).
- **Multiple real test modes**, not just a single demo screenshot: single photo, batch upload (many photos at once), group photo detection (multiple faces in one image with bounding boxes), and live webcam capture.
- **Proper evaluation**, not just eyeballing results — a stratified train/test split, per-player accuracy breakdown, and model comparison across different algorithms.

## Dataset

15 Indian cricketers, ~560 usable face images total (after face detection filtered out a small number of unusable photos), from a Kaggle dataset organized as one folder per player: Virat Kohli, MS Dhoni, Rohit Sharma, Hardik Pandya, Jasprit Bumrah, Mohammed Shami, Kedar Jadhav, Ravindra Jadeja, Shikhar Dhawan, Kuldeep Yadav, K.L. Rahul, Vijay Shankar, Yuzvendra Chahal, Dinesh Karthik, Bhuvneshwar Kumar.

## How well does it actually work?

**Overall accuracy: 92%** on a held-out test set (112 photos the model never saw during training), using a proper stratified 80/20 split so every player is fairly represented in both sets.

With a calibrated confidence threshold, accuracy on confident predictions rises to **96%**.

### Per-player accuracy isn't uniform — and I think that's worth reporting honestly

8 of the 15 players hit 100% test accuracy. Two players — Mohammed Shami and Kuldeep Yadav — sat around 77-78% F1-score, and Vijay Shankar was the weakest at 60%. I diagnosed this rather than just reporting it: I found Kuldeep Yadav's embeddings were acting like a "magnet," pulling in misclassifications from other players (low precision), while Vijay Shankar's own photos were the ones getting scattered to other labels (low recall). I tried class-balancing the SVM, which helped Kuldeep Yadav's F1 go from 0.70 to 0.78, but didn't fix Vijay Shankar — which tells me the issue is more likely inconsistent photo quality in his specific training set than something fixable purely through algorithm tuning. I also compared SVM (linear and RBF) against k-NN with cross-validation, and all approaches landed within about 1% of each other, confirming the bottleneck isn't the algorithm choice.

### I actually tested the "unknown" rejection on a real outsider — and found a real bug

Early on, I set the confidence threshold at 0.30, calibrated only using known players' test photos. When I uploaded a real photo of Sanju Samson (not in the training set), the system confidently (and wrongly) identified him as Vijay Shankar at 35% confidence. That's a real failure case I caught through testing, not through metrics.

I fixed it by raising the threshold to 0.45, then re-tested: Sanju Samson now correctly gets flagged as "Unknown," and known players (including the weaker ones) still get recognized correctly on most photos. I also validated this further with a blurry, real group photo containing a mix of trained players (Dhoni, Rohit Sharma, Kohli, Kuldeep Yadav, Shikhar Dhawan, Ravindra Jadeja — all correctly identified) and non-trained players (Suresh Raina, Rachin Ravindra, Ashwin, Rahane — all correctly flagged as unknown).

## Features

- Upload a single photo and get the player identification with confidence, top-3 guesses, and a player info card (role, batting style, nickname)
- Batch upload — process many individual photos at once, see results in a grid plus a summary table
- Group photo mode — detects every face in a photo and draws labeled bounding boxes (green for known players, red for unknown)
- Live webcam capture mode
- Sidebar gallery browsing all 15 trained players
- Confidence-based "unknown" rejection, calibrated and validated against real outsider photos

## How it's built

1. **`build_embeddings.py`** — goes through every player's photos, detects the face (MTCNN) and computes a 512-dimension embedding (InceptionResnetV1 pretrained on VGGFace2) for each one
2. **`train_classifier.py` / `train_classifier_v2.py`** — trains an SVM classifier on the embeddings with a stratified train/test split, reports per-player accuracy
3. **`compare_models.py`** — cross-validates SVM (linear/RBF) and k-NN to confirm the best approach empirically rather than assuming
4. **`calibrate_threshold.py`** — analyzes confidence scores for correct vs incorrect predictions to pick a real, evidence-backed confidence threshold
5. **`app/dashboard.py`** — the Streamlit app tying it all together with four input modes

## Tech stack

Python, facenet-pytorch (MTCNN for face detection, InceptionResnetV1 for embeddings), scikit-learn (SVM classifier, evaluation), PyTorch, Streamlit, PIL/Pillow, pandas, numpy.

## How to run it
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Build the embeddings and train the model (only needed once, or if the dataset changes):
cd src
python build_embeddings.py
python train_classifier_v2.py

Run the app:
streamlit run app/dashboard.py


## Folder structure

- `app/dashboard.py` — the Streamlit web app
- `src/build_embeddings.py` — face detection + embedding generation
- `src/train_classifier_v2.py` — classifier training with class balancing
- `src/compare_models.py` — algorithm comparison via cross-validation
- `src/calibrate_threshold.py` — confidence threshold analysis
- `src/diagnose_errors.py` — misclassification diagnostics
- `src/player_info.py` — player metadata (role, batting style, nickname)
- `data/raw_images/images/` — the dataset, one folder per player
- `data/embeddings/` — computed embeddings, trained classifier, label encoder

## Where this falls short (being honest about it)

- The dataset has 15 players with 17-50 photos each — fine for this project, but a production system would need more photos per player, especially for the weaker-performing ones
- Vijay Shankar's recognition accuracy (60% F1) remains a real weak point I wasn't able to fully resolve through algorithm tuning — more/better training photos would likely be the actual fix
- The confidence threshold (0.45) is a real tradeoff — it's tuned to catch outsiders without rejecting too many correct matches, but it's not perfect; a different outsider photo might still occasionally slip through, or a legitimate player's harder photo might get flagged as unknown
- Face detection (MTCNN) can fail silently on very blurry, extreme-angle, or very small faces in group photos — those get skipped rather than guessed, which is the safer failure mode but means not every face in a crowded photo gets labeled
- This only recognizes photos, not live video streams frame-by-frame (the webcam mode captures a still photo, not continuous tracking)

With more time, I'd want to add more training photos specifically for Vijay Shankar and Kuldeep Yadav, try a few more outsider validation photos to stress-test the threshold further, and potentially add the ability to register a new player from just a few photos without retraining the whole classifier (since the embedding approach makes this technically straightforward).

## Notes

Built as part of an internship training project. The thing I focused on most wasn't raw accuracy — it was making sure the system is honest about its own uncertainty, and actually testing that claim against real photos instead of just assuming it works.