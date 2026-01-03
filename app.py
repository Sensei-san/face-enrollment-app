import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import zipfile
import requests
import json
import tempfile
import re
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "dataset"
BACKEND_UPLOAD_URL = "http://localhost:8000/upload"  # change if needed
UPLOAD_TIMEOUT = 30  # seconds

POSES = [
    ("front", "Look straight at the camera"),
    ("left", "Turn your head slightly LEFT"),
    ("right", "Turn your head slightly RIGHT"),
    ("up", "Tilt your head UP"),
    ("down", "Tilt your head DOWN"),
]

# -----------------------------
# HELPERS
# -----------------------------

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def overlay_face_outline(image_pil):
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape
    cv2.ellipse(
        img,
        (w // 2, h // 2),
        (int(w * 0.25), int(h * 0.35)),
        0,
        0,
        360,
        (0, 255, 0),
        2,
    )
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def validate_single_face(image_pil):
    gray = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return len(faces) == 1


def sanitize_filename(s: str) -> str:
    # remove characters that could be used to traverse or cause FS issues
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s.strip())


def zip_employee_data(folder_path, zip_path, metadata, poses):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # only include pose files (avoid including zip or other unrelated files)
        for pose_name, _ in poses:
            file_path = os.path.join(folder_path, f"{pose_name}.jpg")
            if os.path.exists(file_path):
                zipf.write(file_path, arcname=os.path.basename(file_path))
        # write metadata.json
        zipf.writestr("metadata.json", json.dumps(metadata, indent=2))


# UI
# -----------------------------

st.set_page_config(page_title="Face Enrollment", layout="centered")
st.title("Employee Face Enrollment")

# Page state
if "page" not in st.session_state:
    st.session_state.page = "intro"

# -----------------------------
# PAGE 1 â€” INTRO / CONSENT
# -----------------------------

if st.session_state.page == "intro":
    name = st.text_input("Employee Name")
    emp_id = st.text_input("Employee ID")

    consent = st.checkbox(
        "I understand that images are collected for attendance system training purposes"
    )

    if st.button("âž¡ Start Face Enrollment"):
        if not (name and emp_id and consent):
            st.error("Please fill all fields and provide consent.")
        else:
            # sanitize inputs for file system safety
            st.session_state.name = sanitize_filename(name)
            st.session_state.emp_id = sanitize_filename(emp_id)
            st.session_state.page = "capture"
            # set timestamp when enrollment begins
            st.session_state.timestamp = datetime.utcnow().isoformat() + "Z"
            st.rerun()

    st.stop()

name = st.session_state.name
emp_id = st.session_state.emp_id

save_dir = os.path.join(DATASET_DIR, f"{emp_id}_{name}")

# Prevent overwriting existing IDs
if os.path.exists(save_dir) and "step" not in st.session_state:
    st.error("âŒ Employee ID already exists. Enrollment is locked.")
    st.stop()

os.makedirs(save_dir, exist_ok=True)

if "step" not in st.session_state:
    st.session_state.step = 0

# Progress bar (clamped between 0 and 1)
progress_val = min(max(st.session_state.step / len(POSES), 0.0), 1.0)
st.progress(progress_val)

enrollment_complete = st.session_state.step >= len(POSES)

# -----------------------------
# PAGE 2 â€” CAPTURE FLOW
# -----------------------------

if not enrollment_complete:
    pose_name, instruction = POSES[st.session_state.step]

    st.subheader(f"Step {st.session_state.step + 1}: {pose_name.upper()}")
    st.info(instruction)

    photo = st.camera_input("Align your face within the outline")

    if photo:
        img = Image.open(photo)

        if not validate_single_face(img):
            st.error("âŒ Exactly one face must be visible. Please retake.")
            st.stop()

        preview = overlay_face_outline(img)
        st.image(preview, caption="Check alignment")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("ðŸ” Retake"):
                st.rerun()
        with col2:
            # single accept button (fixed duplicated block)
            if st.button("âœ… Accept & Continue"):
                try:
                    img.save(os.path.join(save_dir, f"{pose_name}.jpg"))
                except Exception as e:
                    st.error(f"Failed to save image: {e}")
                    st.stop()
                st.session_state.step += 1
                st.rerun()

# -----------------------------
# PAGE 3 â€” FINAL SCREEN + UPLOAD
# -----------------------------

else:
    st.success("âœ… Enrollment complete")

    zip_name = f"{emp_id}_{name}.zip"
    zip_path = os.path.join(DATASET_DIR, zip_name)

    if not os.path.exists(zip_path):
        metadata = {
            "employee_id": emp_id,
            "name": name,
            "poses": [p[0] for p in POSES],
            "timestamp": st.session_state.get("timestamp", ""),
            "source": "streamlit-face-enrollment-v1",
        }
        # create zip deterministically
        zip_employee_data(save_dir, zip_path, metadata, POSES)

    st.subheader("Submit Your Data")

    upload_btn = st.button("ðŸ“¤ Upload Data", disabled=not enrollment_complete)

    if upload_btn:
        try:
            with open(zip_path, "rb") as f:
                files = {"file": (zip_name, f, "application/zip")}
                data = {"employee_id": emp_id, "name": name}
                response = requests.post(
                    BACKEND_UPLOAD_URL, files=files, data=data, timeout=UPLOAD_TIMEOUT
                )

            if response.status_code == 200:
                st.success("âœ… Data uploaded successfully")
            else:
                # try to show backend JSON message if present
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                st.error(f"âŒ Upload failed ({response.status_code}): {detail}")
        except requests.RequestException as e:
            st.error(f"âŒ Upload request failed: {e}")
