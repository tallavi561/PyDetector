import os
import uuid
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import List, Tuple

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

def image_preprocess_clahe(input_path: str, filename: str) -> tuple[str, Image.Image]:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the image at input_path and saves a processed copy.
    
    Returns:
      processed_path (str): path to the enhanced image.
      pil_image (PIL.Image): PIL image object of the enhanced image.
    """

    # Load image
    img = cv2.imread(input_path)

    if img is None:
        raise ValueError(f"[ERROR] Failed to read saved image: {input_path}")

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Apply CLAHE on only the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Merge back and convert to BGR
    lab_clahe = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Save processed image
    processed_path = os.path.join("input_pictures", f"processed_{filename}")
    cv2.imwrite(processed_path, img_clahe)

    print(f"[INFO] Saved CLAHE processed image: {processed_path}")

    # Convert to PIL for possible further use
    pil_img = Image.fromarray(cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB))

    return processed_path, pil_img

def draw_boxes_and_save(
    image_path: str,
    output_path: str,
    boxes: List[Tuple[int, int, int, int]],
    line_width: int = 4
) -> str:
    """
    Draws red bounding boxes on an image and saves the result.

    Args:
        image_path: Path to the source image
        output_path: Path (without extension) for the output image
        boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
        line_width: Thickness of rectangle borders

    Returns:
        Path to the saved image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    draw = ImageDraw.Draw(img)

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        # Clamp coordinates
        x1_c = max(0, min(x1, w))
        y1_c = max(0, min(y1, h))
        x2_c = max(0, min(x2, w))
        y2_c = max(0, min(y2, h))

        if x2_c <= x1_c or y2_c <= y1_c:
            print(f"[WARN] Skipping invalid box #{idx}: {(x1, y1, x2, y2)}")
            continue

        draw.rectangle(
            [(x1_c, y1_c), (x2_c, y2_c)],
            outline="red",
            width=line_width
        )

    output_file = f"{output_path}"
    img.save(output_file)

    return output_file