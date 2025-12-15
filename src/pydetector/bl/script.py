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

Box = Tuple[int, int, int, int]
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
    box: Box,
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

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_brightness_bins(
    image_path: str,
    bin_size: int = 10
) -> str:
    """
    Counts how many pixels fall into fixed brightness ranges:
    0–9, 10–19, ..., 250–255
    and saves a bar-chart visualization.

    Returns:
        Path to the saved image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    pixels = image.flatten()

    # --- Define bins ---
    bins = list(range(0, 256 + bin_size, bin_size))
    counts, edges = np.histogram(pixels, bins=bins)

    # --- Prepare labels ---
    labels = [
        f"{edges[i]}–{edges[i+1]-1}"
        for i in range(len(edges) - 1)
    ]

    # --- Plot ---
    plt.figure(figsize=(14, 6))
    plt.bar(labels, counts)
    plt.xticks(rotation=90)
    plt.xlabel("Brightness range")
    plt.ylabel("Pixel count")
    plt.title("Brightness Distribution (Bucketed)")
    plt.tight_layout()

    # --- Save ---
    base, _ = os.path.splitext(image_path)
    output_path = f"{base}_brightness_bins.png"
    plt.savefig(output_path)
    plt.close()

    print(f"[INFO] Brightness bins saved to: {output_path}")

    return output_path




# Mapping from barcode index to list of box indices it is contained in
BARCODE_TO_BOXES_INDEXES = dict[int, list[int]]
BARCODE_TO_BOXES = dict[int, list[Box]]
def box_containment(barcode_boxes: list[Box], detection_box: list[Box]) -> BARCODE_TO_BOXES_INDEXES:
    """
    Builds a mapping from each barcode box to the list of detection boxes that contain it.
    """
    barcode_to_box: BARCODE_TO_BOXES_INDEXES = dict()
    for barcode_idx, barcode_box in enumerate(barcode_boxes):
        containing_boxes = []
        for d_idx, d_box in enumerate(detection_box):
            score = box_containment_score_v2(d_box, barcode_box)
            if score >= 50:
                containing_boxes.append(d_idx)
        barcode_to_box[barcode_idx] = containing_boxes
    return barcode_to_box

def mark_relevant_boxes_from_xml(
    xml_path: str,
    image_path: str,
    output_path: str,
    MIN_BOX_WIDTH: int = 400,
    MIN_BOX_HEIGHT: int = 400
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
    p = save_brightness_bins(image_path, 10)
    print(p)
    FINAL_BOXES: List[Box] = []

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(f"image.shape: {image.shape}")
    barcode_boxes = extract_boxes_from_xml(xml_path)
    print(f"[INFO] Extracted {len(barcode_boxes)} relevant boxes from XML.")
    if not barcode_boxes:
        raise ValueError("No relevant bounding boxes found in XML.")

    draw_boxes_and_save(
        image_path=image_path,
        output_path=output_path,
        boxes=barcode_boxes
    )

    print(f"[INFO] Saved marked image with boxes to: {output_path}")
    decetions_dict: dict =  detector.detect(image_path, conf_threshold=0.002, save_outputs=False)
    print(f"[INFO] Detector found {len(decetions_dict.get('objects', []))} objects.")
    
    detections_boxes: list[Box] = [(obj["X1"], obj["Y1"], obj["X2"], obj["Y2"]) for obj in decetions_dict.get("objects", [])]

    draw_boxes_and_save(
        image_path=image_path,
        output_path=output_path.replace(".jpg", "_all_detected1.jpg"),
        boxes=detections_boxes
    )

    barcode_to_containing_boxes_indexes: BARCODE_TO_BOXES_INDEXES = box_containment(barcode_boxes, detections_boxes)
    barcode_to_containing_boxes : BARCODE_TO_BOXES = dict()
    for barcode_idx, box_indexes in barcode_to_containing_boxes_indexes.items():
        barcode_to_containing_boxes[barcode_idx] = [detections_boxes[i] for i in box_indexes]
    
    # if only one box contains a barcode, it definitely contains it
    for barcode_idx, containing_boxes in barcode_to_containing_boxes.items():
        if len(containing_boxes) == 1:
            box = containing_boxes[0]
            if box not in FINAL_BOXES:
                FINAL_BOXES.append(box)


    print(f"[INFO] {len(FINAL_BOXES)} detection boxes definitely contain unique barcode boxes.")
    draw_boxes_and_save(
        image_path=image_path,
        output_path=output_path.replace(".jpg", "_definitely_exist_boxes2.jpg"),
        boxes=FINAL_BOXES
    )


    for barcode_idx, containing_boxes in barcode_to_containing_boxes_indexes.items():
        print(f"[DEBUG] Barcode box {barcode_idx} is contained in detection boxes: {containing_boxes}")
        filtered_boxes = []
        for box_idx in containing_boxes:
            box = detections_boxes[box_idx]
            x1, y1, x2, y2 = box
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            # NEED TO FIX IT! 
            if width >= MIN_BOX_WIDTH or height >= MIN_BOX_HEIGHT:
                filtered_boxes.append(box_idx)
        # remove small boxes from containing_boxes, and ansure at least one box remains
        if filtered_boxes:
            barcode_to_containing_boxes_indexes[barcode_idx] = filtered_boxes 
            # containing_boxes[:] = filtered_boxes

    
    # debug print after size filtering
    without_small_boxes = []
    for barcode_idx, containing_boxes in barcode_to_containing_boxes_indexes.items():
        print(f"[DEBUG] After size filtering, Barcode box {barcode_idx} is contained in detection boxes: {containing_boxes}")
        for box_idx in containing_boxes:
            box = detections_boxes[box_idx]
            if box not in without_small_boxes:
                without_small_boxes.append(box)

    draw_boxes_and_save(
            image_path=image_path,
            output_path=output_path.replace(".jpg", "_detected_no_small3.jpg"),
            boxes=without_small_boxes)
    """
    We assume that the box with the highest bright pixel ratio is the correct one for each barcode.
    But we don't actuallt know the correct bright threshold, so we start with 200 and lower it until we find exact one box for each barcode.
    """

    for barcode_idx, containing_boxes in barcode_to_containing_boxes.items():
        print(f"[DEBUG] Processing barcode {barcode_idx} with {containing_boxes} containing boxes.")   
        BRIGHTNESS_THRESHOLD = 160
        STEP_SIZE = 10
        RATIO = 80.0
        CONTINURE = True
        if len(containing_boxes) == 1:
            continue
        containing_boxes.sort(key=lambda box: bright_pixel_ratio_in_box_from_path(
            image_path=image_path, box=box, brightness_threshold=BRIGHTNESS_THRESHOLD),
            reverse=True
            )
        # now we want to save each box s.t the ratio is in the name of the file
        for box in containing_boxes:
            ratio = bright_pixel_ratio_in_box_from_path(
                image_path=image_path,
                box=box,
                brightness_threshold=BRIGHTNESS_THRESHOLD
            )
            print(f"[DEBUG] Box {box} has bright pixel ratio: {ratio:.2f}% at threshold {BRIGHTNESS_THRESHOLD}.")
            draw_boxes_and_save(
                image_path=image_path,
                output_path=output_path.replace(".jpg", f"_barcode{barcode_idx}_box_{box[0]}_{box[1]}_{box[2]}_{box[3]}_ratio_{int(ratio)}.jpg"),
                boxes=[box]
            )
        
        barcode_to_containing_boxes[barcode_idx] = [containing_boxes[0]]
    print(f"[INFO] Selected boxes after brightness evaluation.")
    print(f"barcode_to_containing_boxes: {barcode_to_containing_boxes}")
    for barcode_idx, containing_boxes in barcode_to_containing_boxes.items():
        for box in containing_boxes:
            if box not in FINAL_BOXES:
                FINAL_BOXES.append(box)
        # while CONTINURE:
        #     if len(containing_boxes) == 1:
        #         print(f"[INFO] Only one box contains barcode {barcode_idx}, selecting it.")
        #         if containing_boxes[0] not in FINAL_BOXES:
        #             FINAL_BOXES.append(containing_boxes[0])
        #         CONTINURE = False
        #         continue
        #     print(f"[DEBUG] Evaluating bright pixel ratios for barcode {barcode_idx} with brightness threshold {BRIGHTNESS_THRESHOLD}.")
        #     # here, the index indicates the box in containing_boxes
        #     box_index_to_bright_ratio: dict[int, float] = dict()
        #     for box_id, box in enumerate(containing_boxes):
        #         ratio = bright_pixel_ratio_in_box_from_path(
        #             image_path=image_path,
        #             box=box,
        #             brightness_threshold=BRIGHTNESS_THRESHOLD
        #         )

        #         box_index_to_bright_ratio[box_id] = ratio
        #     print(f"[DEBUG] Bright pixel ratios for barcode {barcode_idx}: {box_index_to_bright_ratio}")
        #     # find how many boxes are above the ratio
        #     boxes_above_ratio: list[int] = [box_id for box_id, ratio in box_index_to_bright_ratio.items() if ratio >= RATIO]
        #     if len(boxes_above_ratio) == 1:
        #         print(f"[INFO] Found unique bright box for barcode {barcode_idx} with brightness threshold {BRIGHTNESS_THRESHOLD}.")
        #         box = containing_boxes[boxes_above_ratio[0]]
        #         if box not in FINAL_BOXES:
        #             FINAL_BOXES.append(box)
        #         CONTINURE = False
        #     else:
        #         print(f"[DEBUG] Could not find unique bright box for barcode {barcode_idx} with brightness threshold {BRIGHTNESS_THRESHOLD}. Boxes above ratio: {boxes_above_ratio}")
        #         BRIGHTNESS_THRESHOLD -= STEP_SIZE
        #         RATIO -= 5.0
        #         if BRIGHTNESS_THRESHOLD < 50:
        #             print(f"[WARNING] Reached minimum brightness threshold for barcode {barcode_idx} without finding unique bright box.")
        #             # give up
        #             CONTINURE = False
        #             print(f"[WARNING] Could not determine unique bright box for barcode {barcode_idx}.")
    


    draw_boxes_and_save(
            image_path=image_path,
            output_path=output_path.replace(".jpg", "_detected_bright_boxes4.jpg"),
            boxes=FINAL_BOXES
      )
    return output_path