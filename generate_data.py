"""
Synthetic Indian License Plate Generator for TrOCR Training
============================================================
Generates 15,000+ diverse plate images + labels.csv

Run:
    pip install pillow opencv-python numpy
    python generate_plates.py

Output:
    plate_dataset/
        images/          <- .jpg plate images
        labels.csv       <- filename, label, raw_text
"""

import csv
import os
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("plate_dataset")
IMAGES_DIR  = OUTPUT_DIR / "images"
LABELS_FILE = OUTPUT_DIR / "labels.csv"
N_IMAGES    = 50000
RANDOM_SEED = 99

# ── Fonts — update these paths for your Windows machine ──────────────────────
# Drop any .ttf files you want into a "fonts/" folder next to this script.
# The script will auto-discover all .ttf files in that folder.
# If none found, it falls back to built-in fonts (ugly but functional).
FONTS_DIR = Path("fonts")

FALLBACK_FONTS = [
    r"C:\Windows\Fonts\arialbd.ttf",      # Arial Bold
    r"C:\Windows\Fonts\calibrib.ttf",     # Calibri Bold
    r"C:\Windows\Fonts\verdanab.ttf",     # Verdana Bold
    r"C:\Windows\Fonts\couriernew.ttf",   # Courier New
    r"C:\Windows\Fonts\impact.ttf",       # Impact
    r"C:\Windows\Fonts\trebucbd.ttf",     # Trebuchet Bold
    r"C:\Windows\Fonts\consolab.ttf",     # Consolas Bold
]

def discover_fonts():
    fonts = []
    if FONTS_DIR.exists():
        fonts = list(FONTS_DIR.glob("*.ttf")) + list(FONTS_DIR.glob("*.otf"))
    if not fonts:
        fonts = [Path(f) for f in FALLBACK_FONTS if Path(f).exists()]
    if not fonts:
        print("WARNING: No fonts found. Using PIL default (low quality).")
    return [str(f) for f in fonts]

ALL_FONTS = discover_fonts()


def load_font(size: int) -> ImageFont.FreeTypeFont:
    if ALL_FONTS:
        path = random.choice(ALL_FONTS)
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Indian state codes ────────────────────────────────────────────────────────
STATE_CODES = [
    "AN","AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HP","HR",
    "JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL","OD","PB",
    "PY","RJ","SK","TN","TR","TS","UK","UP","UT","WB",
]
DIGITS  = string.digits
LETTERS = string.ascii_uppercase
CONFUSABLE = list("0O1IB8S5Z2G6QD")   # chars models often confuse


# ── Separator styles ──────────────────────────────────────────────────────────
# Each is (display_separator, included_in_label)
SEPARATORS = [
    ("",    False),   # no separator         MH12AB1234
    (" ",   False),   # space                MH 12 AB 1234
    ("-",   False),   # dash                 MH-12-AB-1234
    (".",   False),   # dot                  MH.12.AB.1234
]


# ── Color schemes ─────────────────────────────────────────────────────────────
# (bg_color, text_color, border_color, description)
COLOR_SCHEMES = [
    # White — private (most common)
    ((255, 255, 255), (0,   0,   0  ), (0,   0,   0  ), "white"),
    ((248, 248, 248), (10,  10,  10 ), (40,  40,  40 ), "white_slightly_off"),
    ((235, 235, 230), (0,   0,   0  ), (0,   0,   0  ), "cream"),
    # Yellow — commercial / taxi
    ((255, 220, 0  ), (0,   0,   0  ), (0,   0,   0  ), "yellow"),
    ((240, 200, 10 ), (20,  20,  20 ), (0,   0,   0  ), "yellow_dark"),
    ((255, 235, 50 ), (0,   0,   0  ), (80,  60,  0  ), "yellow_bright"),
    # Green — electric vehicles
    ((0,   150, 60 ), (255, 255, 255), (0,   80,  30 ), "green"),
    ((20,  130, 50 ), (240, 240, 240), (0,   60,  20 ), "green_dark"),
    ((0,   180, 80 ), (255, 255, 255), (0,   100, 40 ), "green_bright"),
    # Blue — diplomatic
    ((0,   0,   180), (255, 255, 255), (0,   0,   100), "blue_diplomatic"),
    ((0,   30,  150), (220, 220, 255), (0,   0,   80 ), "blue_dark"),
    # Black — very old / special plates
    ((20,  20,  20 ), (255, 255, 255), (255, 255, 255), "black"),
    # Red — government vehicles (rare)
    ((180, 0,   0  ), (255, 255, 255), (120, 0,   0  ), "red_govt"),
    # Faded / weathered white
    ((220, 215, 205), (40,  40,  40 ), (80,  80,  80 ), "faded_white"),
    ((200, 195, 185), (60,  60,  60 ), (100, 100, 100), "very_faded"),
]


# ── Plate text generators ─────────────────────────────────────────────────────

def rs():  return random.choice(STATE_CODES)
def rd(n): return "".join(random.choices(DIGITS, k=n))
def rl(n):
    pool = LETTERS + "".join(CONFUSABLE)
    return "".join(random.choices(pool, k=n)).upper()
def rone(): return random.choice(LETTERS)


def apply_separator(parts: list, sep: str) -> str:
    return sep.join(parts)


def gen_standard(sep=""):
    parts = [rs(), rd(2), rl(2), rd(4)]
    return apply_separator(parts, sep)

def gen_bh(sep=""):
    yr = str(random.randint(21, 25))
    parts = [yr, "BH", rd(4), rone()]
    return apply_separator(parts, sep)

def gen_diplomatic_short(sep=""):
    parts = [rd(2).zfill(2), random.choice(["CD","CC"]), rd(4)]
    return apply_separator(parts, sep)

def gen_diplomatic_long(sep=""):
    # e.g. 01CC1A0001
    num1  = rd(2).zfill(2)
    mixed = random.choice([
        rd(1) + rone(),
        rone() + rd(1),
        rd(1) + rl(2),
        rl(2) + rd(1),
    ])
    num2 = rd(4).zfill(4)
    parts = [num1, "CC", mixed, num2]
    return apply_separator(parts, sep)

def gen_temp(sep=""):
    parts = [rs(), rd(2), "TEMP", rd(4)]
    return apply_separator(parts, sep)

def gen_old_format(sep=""):
    parts = [rs(), rd(2), rl(random.randint(1, 3)), rd(random.randint(3, 4))]
    return apply_separator(parts, sep)

def gen_long_series(sep=""):
    # 3-letter series: MH12ABC1234
    parts = [rs(), rd(2), rl(3), rd(4)]
    return apply_separator(parts, sep)

def gen_single_letter(sep=""):
    # MH12A1234 — single letter series
    parts = [rs(), rd(2), rone(), rd(4)]
    return apply_separator(parts, sep)

def gen_electric(sep=""):
    # Same as standard but will be rendered on green plate
    parts = [rs(), rd(2), rl(2), rd(4)]
    return apply_separator(parts, sep)


GENERATORS = [
    (gen_standard,          30),
    (gen_bh,                10),
    (gen_diplomatic_short,   6),
    (gen_diplomatic_long,   10),   # high weight to fix embassy plate problem
    (gen_temp,               8),
    (gen_old_format,         7),
    (gen_long_series,        8),
    (gen_single_letter,      6),
    (gen_electric,           5),   # same text, different color (handled below)
    (gen_standard,          10),   # extra standard with different separator
]


def generate_plate_text():
    funcs, weights = zip(*GENERATORS)
    fn = random.choices(funcs, weights=weights)[0]
    sep_display, _ = random.choice(SEPARATORS)
    # Pass separator to generators that accept it
    try:
        text = fn(sep=sep_display)
    except TypeError:
        text = fn()
    label = text.replace(" ", "").replace("-", "").replace(".", "")
    return text, label


# ── Layout modes ──────────────────────────────────────────────────────────────

def render_single_row(draw, text, w, h, font, txt_col):
    """Standard single-row plate text."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (w - tw) // 2
    y = (h - th) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=txt_col)


def render_two_row(draw, text, w, h, font_size, txt_col):
    """Split plate text across two rows (common on trucks/buses)."""
    mid = len(text) // 2
    top_text = text[:mid]
    bot_text = text[mid:]
    font = load_font(font_size - 6)
    for i, row_text in enumerate([top_text, bot_text]):
        bbox = draw.textbbox((0, 0), row_text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (w - tw) // 2
        row_h = h // 2
        y = i * row_h + (row_h - th) // 2 - bbox[1]
        draw.text((x, y), row_text, font=font, fill=txt_col)


# ── Plate renderer ────────────────────────────────────────────────────────────

def render_plate(text: str, label: str) -> np.ndarray:
    scheme = random.choice(COLOR_SCHEMES)
    bg_col, txt_col, border_col, scheme_name = scheme

    # Force green bg for electric plates
    if label.startswith(tuple(STATE_CODES)) and random.random() < 0.12:
        bg_col     = (0, 150, 60)
        txt_col    = (255, 255, 255)
        border_col = (0, 80, 30)

    # Plate dimensions — vary aspect ratio
    w = random.randint(460, 580)
    h = random.randint(100, 140)

    img  = Image.new("RGB", (w, h), bg_col)
    draw = ImageDraw.Draw(img)

    # Border
    bw = random.randint(3, 8)
    draw.rectangle([bw, bw, w - bw, h - bw], outline=border_col, width=bw)

    # Inner border (some plates have double border)
    if random.random() < 0.3:
        draw.rectangle(
            [bw + 3, bw + 3, w - bw - 3, h - bw - 3],
            outline=border_col, width=1
        )

    # Font size
    font_size = random.randint(40, 60)
    font = load_font(font_size)

    # Shrink if text too wide
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    while tw > w - 30 and font_size > 20:
        font_size -= 2
        font = load_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]

    # Layout: 85% single row, 15% two-row
    use_two_row = (random.random() < 0.15 and len(text) >= 8)
    if use_two_row:
        render_two_row(draw, text, w, h, font_size, txt_col)
    else:
        render_single_row(draw, text, w, h, font, txt_col)

    # Watermark text (state transport dept branding on real plates)
    if random.random() < 0.45:
        wm_size = random.randint(7, 13)
        wm_font = load_font(wm_size)
        wm_text = random.choice([
            "TRANSPORT DEPT", "GOVT OF INDIA", "VAHAN.NIC.IN",
            "MV DEPARTMENT", "STATE TRANSPORT", "PARIVAHAN.GOV.IN",
            "HIGH SECURITY", "HSRP", "IND",
        ])
        # Make watermark nearly invisible against the bg
        wm_alpha = random.randint(15, 45)
        wm_col = tuple(
            max(0, c - wm_alpha) if bg_col[0] > 128
            else min(255, c + wm_alpha)
            for c in bg_col
        )
        wm_bbox = draw.textbbox((0, 0), wm_text, font=wm_font)
        wm_x = (w - (wm_bbox[2] - wm_bbox[0])) // 2
        draw.text((wm_x, h - wm_size - 4), wm_text, font=wm_font, fill=wm_col)

    # Ashoka chakra / emblem dot
    if random.random() < 0.35:
        dot_r = random.randint(3, 8)
        dot_x = random.randint(8, 25)
        dot_y = h // 2
        draw.ellipse(
            [dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r],
            fill=txt_col
        )

    # Screw holes (real plates have mounting screws)
    if random.random() < 0.4:
        for sx in [20, w - 20]:
            sy = h // 2
            r = random.randint(3, 5)
            draw.ellipse([sx-r, sy-r, sx+r, sy+r], fill=border_col)

    return np.array(img)


# ── Augmentations ─────────────────────────────────────────────────────────────

def gaussian_noise(img, sigma=None):
    if sigma is None: sigma = random.uniform(3, 25)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def motion_blur(img):
    size = random.choice([3, 5, 7, 9, 11])
    k = np.zeros((size, size))
    if random.random() < 0.8:
        k[size//2, :] = 1.0 / size          # horizontal
    else:
        k[:, size//2] = 1.0 / size          # vertical
    return cv2.filter2D(img, -1, k)

def brightness_contrast(img):
    alpha = random.uniform(0.5, 1.5)
    beta  = random.randint(-60, 60)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def low_light(img):
    """Simulate night / underground parking lighting."""
    factor = random.uniform(0.15, 0.45)
    dark = (img.astype(np.float32) * factor).astype(np.uint8)
    # Add slight warm/cool tint
    tint = np.zeros_like(dark, dtype=np.float32)
    ch = random.randint(0, 2)
    tint[:, :, ch] = random.uniform(5, 20)
    return np.clip(dark.astype(np.float32) + tint, 0, 255).astype(np.uint8)

def perspective_warp(img):
    h, w = img.shape[:2]
    m = random.randint(2, 18)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([
        [random.randint(0,m), random.randint(0,m)],
        [w-random.randint(0,m), random.randint(0,m)],
        [w-random.randint(0,m), h-random.randint(0,m)],
        [random.randint(0,m), h-random.randint(0,m)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def rotation(img):
    h, w = img.shape[:2]
    angle = random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def jpeg_compress(img):
    q = random.randint(30, 90)
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def shadow(img):
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    x1 = random.randint(0, w//2)
    x2 = random.randint(w//2, w)
    s  = random.uniform(0.25, 0.65)
    mask[:, x1:x2] = s
    mask = cv2.GaussianBlur(mask, (61, 61), 0)
    return (img.astype(np.float32) * mask[:,:,np.newaxis]).clip(0,255).astype(np.uint8)

def gaussian_blur(img):
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), 0)

def dirt_scratches(img):
    """Add random lines / dots simulating dirt on plate."""
    out = img.copy()
    h, w = out.shape[:2]
    n_marks = random.randint(1, 6)
    for _ in range(n_marks):
        if random.random() < 0.5:
            # scratch line
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            col = tuple(int(c) for c in (np.array(img[y1 % h, x1 % w]) * random.uniform(0.3, 0.8)))
            cv2.line(out, (x1, y1), (x2, y2), col, random.randint(1, 2))
        else:
            # dirt blob
            cx, cy = random.randint(0, w), random.randint(0, h)
            r = random.randint(2, 8)
            col = (random.randint(30, 100),) * 3
            cv2.circle(out, (cx, cy), r, col, -1)
    return out

def overexpose(img):
    """Blown-out highlights — harsh sunlight."""
    factor = random.uniform(1.4, 2.2)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def augment(img: np.ndarray) -> np.ndarray:
    # Decide augmentation intensity level
    level = random.choice(["clean", "mild", "moderate", "heavy"])

    if level == "clean":
        # ~10% images nearly clean — model needs to see clean plates too
        if random.random() < 0.3:
            img = jpeg_compress(img)
        return img

    augmentations = [
        (gaussian_noise,    0.60),
        (motion_blur,       0.30),
        (brightness_contrast, 0.65),
        (perspective_warp,  0.50),
        (rotation,          0.40),
        (jpeg_compress,     0.55),
        (shadow,            0.30),
        (gaussian_blur,     0.25),
        (dirt_scratches,    0.40),
    ]

    if level == "heavy":
        # One in ~8 images gets low-light treatment
        if random.random() < 0.5:
            img = low_light(img)
        else:
            img = overexpose(img)
        augmentations = [(fn, min(p + 0.2, 1.0)) for fn, p in augmentations]

    for fn, prob in augmentations:
        if random.random() < prob:
            img = fn(img)

    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fonts found: {len(ALL_FONTS)}")
    for f in ALL_FONTS:
        print(f"  {f}")

    records   = []
    seen_keys = set()

    print(f"\nGenerating {N_IMAGES} images → {OUTPUT_DIR}/")

    for i in range(N_IMAGES):
        # Unique plate text
        for _ in range(30):
            text, label = generate_plate_text()
            if label not in seen_keys:
                seen_keys.add(label)
                break

        plate = render_plate(text, label)
        aug   = augment(plate)

        fname = f"plate_{i:05d}.jpg"
        fpath = IMAGES_DIR / fname
        cv2.imwrite(
            str(fpath),
            cv2.cvtColor(aug, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(82, 97)]
        )

        records.append({
            "filename": fname,
            "label":    label,      # clean text — what TrOCR should output
            "raw_text": text,       # text as displayed (may have separators)
        })

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{N_IMAGES}")

    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "raw_text"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nDone! {N_IMAGES} images, labels at {LABELS_FILE}")
    print(f"Total dataset size: {sum(os.path.getsize(IMAGES_DIR/r['filename']) for r in records) / 1e6:.1f} MB")

    # Format breakdown
    fmt_counts = {}
    for r in records:
        l = r["label"]
        if "TEMP" in l:       k = "TEMP"
        elif "BH" in l:       k = "BH series"
        elif "CC" in l or "CD" in l: k = "Diplomatic"
        else:                 k = "Standard"
        fmt_counts[k] = fmt_counts.get(k, 0) + 1

    print("\nFormat distribution:")
    for k, v in sorted(fmt_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v:5d}  ({v/N_IMAGES*100:.1f}%)")

    print("\nSample output:")
    for r in random.sample(records, 15):
        print(f"  {r['filename']}  raw={r['raw_text']:20s}  label={r['label']}")


if __name__ == "__main__":
    main()