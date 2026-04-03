import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

YOLO_WEIGHTS    = r"C:\Workspace\Github projects\Raju Sir Project\Project file\working_backup\runs\detect\train\weights\best.pt"
TROCR_MODEL_DIR = r"C:\Workspace\Github projects\Raju Sir Project\Project file\trocr_alpr_final"


@st.cache_resource
def load_yolo():
    return YOLO(YOLO_WEIGHTS)


@st.cache_resource
def load_trocr():
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_DIR)
    model     = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_DIR)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


yolo_model                    = load_yolo()
processor, trocr_model, device = load_trocr()


def preprocess_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """
    Minimal preprocessing for TrOCR.

    TrOCR is a Vision Transformer — it expects a natural RGB image.
    Binarization, CLAHE, morphology and sharpening all hurt because:
      - They destroy the grayscale information the ViT encoder relies on
      - Adaptive threshold makes B/8, 0/O harder to separate (stroke breakage)
      - Unsharp mask adds ringing artifacts around characters

    We only do two things that genuinely help:
      1. Upscale — TrOCR works best when character height >= 32px
      2. Deskew  — straighten obvious camera tilt (only if angle > 2°)
    """

    # ── 1. Upscale small plates ───────────────────────────────────────────────
    h, w = plate_bgr.shape[:2]
    target_h = 128
    if h < target_h:
        scale     = target_h / h
        plate_bgr = cv2.resize(
            plate_bgr,
            (int(w * scale), target_h),
            interpolation=cv2.INTER_CUBIC,
        )

    # ── 2. Deskew ─────────────────────────────────────────────────────────────
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        angle   = cv2.minAreaRect(largest)[2]
        if 2 < abs(angle) < 45:
            rh, rw = plate_bgr.shape[:2]
            M = cv2.getRotationMatrix2D((rw // 2, rh // 2), angle, 1.0)
            plate_bgr = cv2.warpAffine(
                plate_bgr, M, (rw, rh),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

    # ── 3. BGR → RGB for TrOCR ───────────────────────────────────────────────
    return cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)


def run_trocr(plate_bgr: np.ndarray) -> str:
    processed_rgb = preprocess_plate(plate_bgr)
    plate_pil     = Image.fromarray(processed_rgb)
    pixel_values  = processor(images=plate_pil, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("License Plate Recognition")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image  = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    results        = yolo_model(img_np)
    detected_texts = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            plate = img_np[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            st.image(plate, caption=f"Detected Plate #{i + 1}", width=300)

            plate_bgr = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
            text      = run_trocr(plate_bgr)

            if text:
                detected_texts.append(text)
                st.success(f"Detected Plate Number: **{text}**")
            else:
                st.warning(f"Plate #{i + 1} detected but OCR returned empty.")

    if not detected_texts:
        st.warning("No license plate detected.")