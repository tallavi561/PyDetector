import math
from typing import Literal
import cv2
import os
import numpy as np
from typing import List, Tuple
import xml.etree.ElementTree as ET
import os
from pydetector.bl.detect import detector

from pydetector.utils.image_utils import draw_boxes_and_save

ImageType = Literal["GRAYSCALE", "RGB", "RGBA", "UNKNOWN"]


def print_image_info(image_path: str) -> ImageType:
    """
    Prints detailed information about an image:
    - Representation (GRAYSCALE / RGB / RGBA)
    - Width & Height
    - Number of channels
    - Data type
    - Pixel value range

    :param image_path: Path to JPG image
    :return: Image representation type
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image: np.ndarray | None = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    height: int
    width: int

    print("Image information:")
    print("-" * 30)

    # Shape & size
    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape

    print(f"Resolution        : {width} x {height}")
    print(f"Channels          : {channels}")
    print(f"Data type         : {image.dtype}")
    print(f"Pixel value range : [{image.min()} , {image.max()}]")

    # Representation
    if channels == 1:
        image_type: ImageType = "GRAYSCALE"
        print("Representation    : GRAYSCALE (mono)")

    elif channels == 3:
        image_type = "RGB"
        print("Representation    : RGB (BGR order in OpenCV)")

    elif channels == 4:
        image_type = "RGBA"
        print("Representation    : RGBA (with alpha channel)")

    else:
        image_type = "UNKNOWN"
        print("Representation    : UNKNOWN")

    return image_type

def extract_boxes_from_xml(xml_path: str) -> List[Tuple[int, int, int, int]]:
    """
    Parses a SICK XML file and extracts bounding boxes of detected barcodes only
    (from <symbol> elements), corrected using image origin.

    Returns:
        List of bounding boxes as (x1, y1, x2, y2) in IMAGE coordinates.
    """

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # --- Read image size ---
    size_node = root.find(".//imageinfo/size")
    if size_node is None:
        raise ValueError("Image size not found in XML.")

    image_width = int(size_node.attrib["width"])
    image_height = int(size_node.attrib["length"])

    # --- Read origin ---
    origin_node = root.find(".//imageinfo/origin")
    origin_x = int(origin_node.attrib.get("x", 0)) if origin_node is not None else 0
    origin_y = int(origin_node.attrib.get("y", 0)) if origin_node is not None else 0

    boxes: List[Tuple[int, int, int, int]] = []

    def polygon_to_bbox(coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return min(xs), min(ys), max(xs), max(ys)

    def clamp_box(
        x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[int, int, int, int] | None:
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        if x1 >= x2 or y1 >= y2:
            return None

        return x1, y1, x2, y2

    # --- Parse BARCODE SYMBOLS ONLY ---
    for symbol in root.findall(".//symbol"):
        coords = []

        for c in symbol.findall(".//coordinate"):
            x = int(float(c.attrib["x"])) - origin_x
            y = int(float(c.attrib["y"])) - origin_y
            coords.append((x, y))

        if len(coords) < 4:
            continue

        bbox = polygon_to_bbox(coords)
        clamped = clamp_box(*bbox)

        if clamped:
            boxes.append(clamped)

    return boxes

Box = Tuple[float, float, float, float]
def point_inside(px: float, py: float, box: Box) -> bool:
    """בדיקה האם נקודה נמצאת בתוך תיבה (כולל הקצוות)."""
    bx1, by1, bx2, by2 = box
    return bx1 <= px <= bx2 and by1 <= py <= by2

def box_containment_score_v2(
    outer_box: Box,
    inner_box: Box
) -> int:
    """
    מחשב עד כמה inner_box מוכלת בתוך outer_box באופן מדויק.

    הקוד משתמש במדד 'שטח החפיפה חלקי שטח התיבה הפנימית' (Area Containment Ratio).
    הציונים נגזרים מהיחס הזה, למעט בדיקות קצה מיוחדות.

    ארגומנטים:
        outer_box (x1, y1, x2, y2)
        inner_box (x1, y1, x2, y2)

    מחזיר:
        100 - הכלה מלאה (100% שטח פנימי בפנים)
         90 - לפחות 90% משטח התיבה הפנימית בפנים
         60 - לפחות 60% משטח התיבה הפנימית בפנים
         50 - נקודת מרכז התיבה הפנימית בפנים
          0 - הכלה לא משמעותית
    """

    ox1, oy1, ox2, oy2 = outer_box
    ix1, iy1, ix2, iy2 = inner_box
    
    # ודא שקואורדינטות תקינות (x1 < x2, y1 < y2)
    if ix1 >= ix2 or iy1 >= iy2 or ox1 >= ox2 or oy1 >= oy2:
        # אם התיבה הפנימית היא נקודה (x1=x2 ו-y1=y2), נטפל בה לאחר מכן.
        if ix1 == ix2 and iy1 == iy2:
            pass # עובר לבדיקת מרכז/הכלה מלאה
        else:
            return 0
    
    # 1. חישוב שטח התיבה הפנימית (A_inner)
    inner_width = ix2 - ix1
    inner_height = iy2 - iy1
    A_inner = inner_width * inner_height

    # אם התיבה הפנימית היא נקודה (שטח 0), בודקים הכלה מלאה או מרכז
    if A_inner == 0.0:
        if point_inside(ix1, iy1, outer_box):
            return 100 # נקודה בתוך קופסה נחשבת הכלה מלאה
        else:
            return 0

    # 2. חישוב שטח החפיפה (A_overlap)
    
    # מציאת קואורדינטות תיבת החפיפה (Intersection Box)
    x_I1 = max(ox1, ix1)
    y_I1 = max(oy1, iy1)
    x_I2 = min(ox2, ix2)
    y_I2 = min(oy2, iy2)

    # רוחב וגובה החפיפה (מוודאים שלא שלילי)
    overlap_width = max(0.0, x_I2 - x_I1)
    overlap_height = max(0.0, y_I2 - y_I1)
    
    A_overlap = overlap_width * overlap_height

    # 3. חישוב יחס הכיסוי (Containment Ratio)
    # Area_Cont_Ratio = A_overlap / A_inner
    
    # זהו המדד המדויק לכמה משטח התיבה הפנימית נמצא בפועל בתוך התיבה החיצונית.
    containment_ratio = A_overlap / A_inner

    # 4. החזרת הציונים לפי יחס הכיסוי (המדויק)

    # --- 1. Full containment (100%) ---
    # קרוב ל-1.0 בשל שימוש ב-float, אך אם החישוב מדויק, הוא צריך להיות 1.0.
    if containment_ratio >= 1.0 - 1e-9: 
        return 100

    # --- 2. 90% Area inside ---
    if containment_ratio >= 0.9:
        return 90

    # --- 3. 60% Area inside ---
    if containment_ratio >= 0.6:
        return 60

    # --- 4. Center point inside ---
    
    # אם לא הגענו ל-60% כיסוי, נבדוק את נקודת המרכז (זהו מדד חלש יותר)
    cx = (ix1 + ix2) / 2.0
    cy = (iy1 + iy2) / 2.0

    if point_inside(cx, cy, outer_box):
        return 50

    # --- 5. No meaningful containment ---
    return 0


def remove_small_boxes(
      boxes: List[Tuple[int, int, int, int]],
      min_width: int,
      min_height: int
      ) -> List[Tuple[int, int, int, int]]:
      """
      Removes boxes smaller than specified width and height.
      
      Args:
            boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            min_width: Minimum width threshold
            min_height: Minimum height threshold
      Returns:
            Filtered list of boxes
      """
      filtered_boxes = []
      for box in boxes:
          x1, y1, x2, y2 = box
          width = x2 - x1
          height = y2 - y1
          if width >= min_width and height >= min_height:
              filtered_boxes.append(box)
      return filtered_boxes

def bright_pixel_ratio_in_box_from_path(
    image_path: str,
    box: Tuple[int, int, int, int],
    brightness_threshold: int
) -> float:
    """
    Calculates the percentage of pixels inside a bounding box
    whose brightness is above a given threshold.

    Assumes GRAYSCALE image.

    Args:
        image_path: Path to grayscale image
        box: (x1, y1, x2, y2)
        brightness_threshold: Threshold in range [0–255]

    Returns:
        Percentage (0–100) of bright pixels inside the box
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # --- Load image as grayscale ---
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    x1, y1, x2, y2 = box
    h, w = image.shape

    # --- Clamp box to image bounds ---
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x1 >= x2 or y1 >= y2:
        return 0.0

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    bright_pixels = roi > brightness_threshold
    return (bright_pixels.sum() / roi.size) * 100.0

def mark_relevant_boxes_from_xml(
    xml_path: str,
    image_path: str,
    output_path: str
) -> str:
    """
    Extracts relevant bounding boxes from XML and draws them on the image.

    Args:
        xml_path: Path to the XML file
        image_path: Path to the source JPG image
        output_path: Output path (without extension)

    Returns:
        Path to the saved image
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(f"image.shape: {image.shape}")
    boxes = extract_boxes_from_xml(xml_path)
    print(f"[INFO] Extracted {len(boxes)} relevant boxes from XML.")
    if not boxes:
        raise ValueError("No relevant bounding boxes found in XML.")

    draw_boxes_and_save(
        image_path=image_path,
        output_path=output_path,
        boxes=boxes
    )

    print(f"[INFO] Saved marked image with boxes to: {output_path}")
    decetions: dict =  detector.detect(image_path, conf_threshold=0.002, save_outputs=False)
    print(f"[INFO] Detector found {len(decetions.get('objects', []))} objects.")
    draw_boxes_and_save(
        image_path=image_path,
        output_path=output_path.replace(".jpg", "_all_detected.jpg"),
        boxes=[(obj["X1"], obj["Y1"], obj["X2"], obj["Y2"]) for obj in decetions.get("objects", [])]
    )
    filter_boxes = []
    for obj in decetions.get("objects", []):
      x1, y1, x2, y2 = (obj["X1"], obj["Y1"], obj["X2"], obj["Y2"])
      for box in boxes:
          if box_containment_score_v2((x1, y1, x2, y2), box) >= 50:
            print(f"[DEBUG]: the width is {x2 - x1}, the height is {y2 - y1}")
            filter_boxes.append((obj["X1"], obj["Y1"], obj["X2"], obj["Y2"]))
    print(f"[INFO] After filtering, {len(filter_boxes)} boxes remain.")
    
    without_small_boxes = remove_small_boxes(filter_boxes, min_width=400, min_height=400)
    draw_boxes_and_save(
            image_path=image_path,
            output_path=output_path.replace(".jpg", "_detected_no_small.jpg"),
            boxes=without_small_boxes)
    bright_boxes = []
    for box in without_small_boxes:
        print(f"[DEBUG] Remaining box after size filter: {box}")
        percentage = bright_pixel_ratio_in_box_from_path(
            image_path,
            box,
            brightness_threshold=190
        )
        print(f"[DEBUG] Bright pixel ratio in box {box}: {percentage:.2f}% in image {image_path} width-height: {abs(box[0] - box[2])} X {abs(box[1] - box[3])}")
        if percentage > 0:
            bright_boxes.append(box)
    print(f"[INFO] After brightness filtering, {len(bright_boxes)} boxes remain.")

    draw_boxes_and_save(
            image_path=image_path,
            output_path=output_path.replace(".jpg", "_detected_bright_boxes.jpg"),
            boxes=bright_boxes
      )
    return output_path