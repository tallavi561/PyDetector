import os
from typing import List, Optional, Tuple
import uuid
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pydetector.modules.classes import Barcode, Point


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



def draw_boxes_and_save(
    image_path: str,
    output_path: str,
    boxes: list[tuple[int, int, int, int]],
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

def draw_rotated_barcodes_on_image(
    image_path: str,
    barcodes: list[Barcode],
    output_path: str
) -> None:
    print(f"[INFO] Found {len(barcodes)} barcodes")
    img = cv2.imread(image_path)
    for i, bc in enumerate(barcodes):
        pts = np.array(bc.points, dtype=np.float32)

        # IMPORTANT: minAreaRect expects contour shape (N,1,2)
        contour = pts.reshape(-1, 1, 2)

        rect = cv2.minAreaRect(contour)   # ((cx,cy),(w,h),angle)
        box = cv2.boxPoints(rect)          # 4x2
        box = box.astype(np.int32)

        # Draw rotated rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

        # Optional: draw index near center
        cx, cy = rect[0]
        cv2.putText(
            img,
            f"{i}",
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imwrite(output_path, img)
    print(f"[INFO] Saved debug image to: {output_path}")


def get_pixels_inside_barcode(
    image_path: str,
    barcode: Barcode,
    image: np.ndarray 
) -> List[int]:
    polygon = np.array(barcode.points, dtype=np.int32)
    
    # מציאת הריבוע החוסם
    x, y, w, h = cv2.boundingRect(polygon)
    
    # הגנה מפני חריגה מגבולות התמונה (חשוב מאוד!)
    img_h, img_w = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    
    # חיתוך ה-ROI מהתמונה המקורית
    roi = image[y1:y2, x1:x2]
    
    # יצירת המסיכה בדיוק בגודל של ה-ROI שחתכנו
    mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
    
    # הזזת נקודות הפוליגון שיתאימו ל-ROI החדש
    # אנחנו מחסירים את x1 ו-y1 כדי שהפוליגון יהיה יחסי לפינה השמאלית של ה-ROI
    shifted_polygon = polygon - [x1, y1]
    
    cv2.fillPoly(mask_roi, [shifted_polygon], 255)
    
    # כעת הגדלים חייבים להתאים
    return roi[mask_roi == 255].tolist()
    # """
    # Returns all pixels inside the barcode polygon.

    # Output format:
    #     [ (pixel_value), ... ]

    # Assumptions:
    # - Image is MONO (grayscale)
    # - Barcode.points defines a polygon (at least 3 points)
    # """

    # # --- Load grayscale image ---


    # height, width = image.shape

    # # --- Create empty mask ---
    # mask = np.zeros((height, width), dtype=np.uint8)

    # # --- Prepare polygon ---
    # polygon = np.array(barcode.points, dtype=np.int32)

    # # --- Fill polygon on mask ---
    # cv2.fillPoly(mask, [polygon], 255)

    # # --- Extract pixels ---
    # pixels: List[int] = []

    # ys, xs = np.where(mask == 255)
    # for x, y in zip(xs, ys):
    #     pixels.append((int(image[y, x])))

    # return pixels

def save_brightness_distribution(
    pixel_values: List[int],
    output_path: str,
    step_size: int = 5,
    threshold: Optional[int] = None
) -> None:
    """
    Saves a bar-chart showing brightness distribution.
    Optionally draws a vertical line for a given threshold.

    Args:
        pixel_values: list of grayscale values (0–255)
        output_path: path to save the chart image
        step_size: size of each bin (e.g. 5, 10, 20)
        threshold: optional grayscale threshold to mark on the chart
    """

    if not pixel_values:
        raise ValueError("pixel_values list is empty")

    if step_size <= 0:
        raise ValueError("step_size must be positive")

    # --- Prepare bins ---
    max_val = 256
    bins = list(range(0, max_val + step_size, step_size))
    hist, bin_edges = np.histogram(pixel_values, bins=bins)

    # --- Prepare labels ---
    labels = [
        f"{bin_edges[i]}-{bin_edges[i+1]-1}"
        for i in range(len(bin_edges) - 1)
    ]

    # --- Plot ---
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(hist)), hist)

    plt.title("Brightness Distribution")
    plt.xlabel("Brightness Range")
    plt.ylabel("Pixel Count")

    plt.xticks(range(len(labels)), labels, rotation=90)

    # --- Draw threshold line (if provided) ---
    if threshold is not None:
        if not (0 <= threshold <= 255):
            raise ValueError("threshold must be in range 0..255")

        # Find which bin contains the threshold
        bin_index = threshold // step_size

        plt.axvline(
            x=bin_index,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {threshold}"
        )

        plt.legend()

    plt.tight_layout()

    # --- Ensure directory exists ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Save ---
    plt.savefig(output_path)
    plt.close()

# def save_brightness_distribution(
#     pixel_values: List[int],
#     output_path: str,
#     step_size: int = 5
# ) -> None:
#     """
#     Saves a bar-chart showing brightness distribution.

#     Each bar represents a range:
#         [0-step_size), [step_size-2*step_size), ...

#     Args:
#         pixel_values: list of grayscale values (0–255)
#         step_size: size of each bin (e.g. 5, 10, 20)
#         output_path: path to save the chart image
#     """

#     if not pixel_values:
#         raise ValueError("pixel_values list is empty")

#     if step_size <= 0:
#         raise ValueError("step_size must be positive")

#     # --- Prepare bins ---
#     max_val = 256
#     bins = list(range(0, max_val + step_size, step_size))

#     hist, bin_edges = np.histogram(pixel_values, bins=bins)

#     # --- Prepare labels ---
#     labels = [
#         f"{bin_edges[i]}-{bin_edges[i+1]-1}"
#         for i in range(len(bin_edges) - 1)
#     ]

#     # --- Plot ---
#     plt.figure(figsize=(14, 6))
#     plt.bar(labels, hist)

#     plt.title("Brightness Distribution")
#     plt.xlabel("Brightness Range")
#     plt.ylabel("Pixel Count")

#     plt.xticks(rotation=90)
#     plt.tight_layout()

#     # --- Ensure directory exists ---
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     # --- Save ---
#     plt.savefig(output_path)
#     plt.close()