import os
import uuid
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import cv2
import numpy as np


INPUT_DIR = "input_pictures"
OUTPUT_DIR = "output_pictures"


def ensure_directories():
    """
    Ensure input_pictures and output_pictures exist.
    """
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_base64_to_image(b64_str: str, folder: str) -> tuple[str, str]:
    """
    Saves a Base64 image to a folder with a random UUID filename.
    Returns the full path to the saved file.
    """
    image_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(image_bytes))

    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(folder, filename)

    img.save(path)
    return (path , filename)


def save_image_object(img: Image.Image, folder: str) -> str:
    """
    Saves a PIL Image to the given folder with a random UUID name.
    """
    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(folder, filename)
    img.save(path)
    return path


def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def add_hello_text_to_image(image: Image.Image) -> Image.Image:
    """
    Simple demo effect: write HELLO THERE.
    You said currently: no processing — so this stays optional.
    """
    img = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    print("[DEBUG] Adding text to image")
    draw.text((10, 10), "HELLO THERE", fill=(255, 0, 0), font=font)
    return img


def crop_image_to_output(image_path: str, output_path: str ,  x1: int, y1: int, x2: int, y2: int) -> str:
    """
    Crops the region (x1, y1, x2, y2) from the image at image_path
    and saves it into OUTPUT_DIR with a random UUID name.

    Returns: path to the cropped file.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    w, h = img.size

    # Clamp coords
    x1_c = max(0, min(x1, w))
    y1_c = max(0, min(y1, h))
    x2_c = max(0, min(x2, w))
    y2_c = max(0, min(y2, h))

    if x2_c <= x1_c or y2_c <= y1_c:
        raise ValueError(
            f"Invalid crop rectangle: {(x1, y1, x2, y2)} → {(x1_c, y1_c, x2_c, y2_c)}"
        )

    cropped = img.crop((x1_c, y1_c, x2_c, y2_c))

    filename = f"{output_path}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)
    cropped.save(out_path)

    return out_path

def preprocess_sticker_image(input_path: str) -> str:
    """
    Preprocess JPG images gently, without introducing noise.
    Uses the Green channel (best SNR), mild contrast boost, mild sharpening.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Image not found: {input_path}")

    # Read as BGR (OpenCV default)
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Failed to load image.")

    # ---------------------------------------
    # 1) Extract green channel (best signal)
    # ---------------------------------------
    green = img[:, :, 1]  # channel G

    # ---------------------------------------
    # 2) Gentle CLAHE (reduced clipLimit)
    # ---------------------------------------
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # ---------------------------------------
    # 3) Mild unsharp mask (no noise boost)
    # ---------------------------------------
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 1.5)
    sharpened = cv2.addWeighted(enhanced, 1.3, blurred, -0.3, 0)

    # ---------------------------------------
    # 4) Final normalization
    # ---------------------------------------
    final = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

    # ---------------------------------------
    # 5) Save
    # ---------------------------------------
    os.makedirs("preprocessed", exist_ok=True)

    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    out_path = f"preprocessed/{name}_prep.jpg"

    cv2.imwrite(out_path, final)

    print(f"[INFO] Preprocessed image saved to {out_path}")
    return out_path