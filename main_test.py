import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import re

YOLO_WEIGHTS    = r"C:\Workspace\Github projects\Raju Sir Project\Project file\working_backup\runs\detect\train\weights\best.pt"
TROCR_MODEL_DIR = r"C:\Workspace\Github projects\Raju Sir Project\Project file\trocr_alpr_final"

@st.cache_resource
def load_yolo():
    return YOLO(YOLO_WEIGHTS)

@st.cache_resource
def load_trocr():
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_DIR)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

yolo_model = load_yolo()
processor, trocr_model, device = load_trocr()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_padded_crop(img_np: np.ndarray, box: np.ndarray, pad_px: int = 12) -> np.ndarray:
    h, w = img_np.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(w, x2 + pad_px)
    y2 = min(h, y2 + pad_px)
    return img_np[y1:y2, x1:x2]


def is_valid_indian_plate(text: str) -> bool:
    cleaned = re.sub(r'[\s.]', '', text).upper()
    return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$', cleaned))


def normalise_plate(text: str) -> str:
    return re.sub(r'[\s.]', '', text).upper()


def preprocess_plate(plate_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Preprocessing pipeline tuned for TrOCR (a vision transformer).
    Key difference from CRNN pipelines: we do NOT binarize.
    TrOCR was pretrained on real photographs so it works best on
    enhanced grayscale — binarization actively hurts it.
    Pipeline: upscale → deskew → grayscale → CLAHE → denoise → sharpen → RGB
    """
    steps = {}

    # ── 1. Upscale ────────────────────────────────────────────────────────────
    h, w = plate_bgr.shape[:2]
    target_h = 128
    if h < target_h:
        scale = target_h / h
        plate_bgr = cv2.resize(
            plate_bgr, (int(w * scale), target_h),
            interpolation=cv2.INTER_CUBIC
        )
    steps["1_upscaled"] = plate_bgr.copy()

    # ── 2. Deskew ─────────────────────────────────────────────────────────────
    gray_for_warp = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh_warp = cv2.threshold(
        gray_for_warp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh_warp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        angle = rect[2]
        if 2 < abs(angle) < 45:
            rh, rw = plate_bgr.shape[:2]
            M = cv2.getRotationMatrix2D((rw // 2, rh // 2), angle, 1.0)
            plate_bgr = cv2.warpAffine(
                plate_bgr, M, (rw, rh),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
    steps["2_deskewed"] = plate_bgr.copy()

    # ── 3. Grayscale ──────────────────────────────────────────────────────────
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    steps["3_gray"] = gray.copy()

    # ── 4. CLAHE ──────────────────────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    steps["4_clahe"] = gray.copy()

    # ── 5. Denoise ────────────────────────────────────────────────────────────
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    steps["5_denoised"] = gray.copy()

    # ── 6. Sharpen (unsharp mask) ─────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=1.0)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    steps["6_sharpened"] = gray.copy()

    # ── 7. Convert to RGB for TrOCR ───────────────────────────────────────────
    # No binarization — TrOCR needs the continuous-tone grayscale image.
    # Convert single-channel → 3-channel RGB so the processor doesn't choke.
    final_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    steps["7_final_rgb"] = final_rgb.copy()

    return final_rgb, steps


def run_trocr(plate_bgr: np.ndarray) -> tuple[str, dict]:
    processed_rgb, debug_steps = preprocess_plate(plate_bgr)

    plate_pil = Image.fromarray(processed_rgb).convert("RGB")
    pixel_values = processor(images=plate_pil, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = trocr_model.generate(
            pixel_values,
            max_new_tokens=24,   # prevents early truncation on long/dot-format plates
            num_beams=4,         # beam search: more accurate than greedy decoding
            early_stopping=True,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip(), debug_steps


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.title("License Plate Recognition")
show_debug = st.sidebar.checkbox("Show preprocessing steps", value=False)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = yolo_model(img_np)
    detected_texts = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes):
            plate = get_padded_crop(img_np, box, pad_px=12)

            if plate.size == 0:
                continue

            plate_bgr = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)

            st.image(plate, caption=f"Detected Plate #{i+1}", width=300)

            text, debug_steps = run_trocr(plate_bgr)

            if show_debug:
                with st.expander(f"Preprocessing pipeline — Plate #{i+1}"):
                    labels = {
                        "1_upscaled":  "① Upscaled",
                        "2_deskewed":  "② Deskewed",
                        "3_gray":      "③ Grayscale",
                        "4_clahe":     "④ CLAHE",
                        "5_denoised":  "⑤ Denoised",
                        "6_sharpened": "⑥ Sharpened",
                        "7_final_rgb": "⑦ Final (fed to TrOCR)",
                    }
                    cols = st.columns(3)
                    for j, (key, label) in enumerate(labels.items()):
                        img_step = debug_steps[key]
                        if len(img_step.shape) == 2:
                            img_step = cv2.cvtColor(img_step, cv2.COLOR_GRAY2RGB)
                        elif img_step.shape[2] == 3:
                            img_step = cv2.cvtColor(img_step, cv2.COLOR_BGR2RGB)
                        cols[j % 3].image(img_step, caption=label, use_container_width=True)

            if text:
                normalised = normalise_plate(text)
                detected_texts.append(normalised)

                if is_valid_indian_plate(text):
                    st.success(f"Detected Plate Number: **{normalised}**")
                else:
                    st.warning(
                        f"Detected Plate Number: **{normalised}**  \n"
                        f"⚠️ Doesn't match expected Indian plate format — may be inaccurate."
                    )
            else:
                st.error(f"Plate #{i+1}: OCR returned empty output.")

    if not detected_texts:
        st.warning("No license plate detected.")