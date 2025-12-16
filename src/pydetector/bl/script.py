import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

import cv2
import numpy as np


Point = Tuple[int, int]
BarcodeResult = Dict[str, object]


# ------------------------------------------------------------
# XML parsing
# ------------------------------------------------------------

def get_barcode_results(files_path: str, xml_file_name: str) -> List[BarcodeResult]:
    """
    Parses SICK camera XML and extracts barcode polygons,
    applying origin correction.

    Returns:
    [
        {
            "points": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)],
            "angle": float
        },
        ...
    ]
    """

    xml_path = os.path.join(files_path, xml_file_name)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # --- Extract origin (CRITICAL) ---
    imageinfo = root.find(".//imageinfo")
    if imageinfo is None:
        raise ValueError("imageinfo not found in XML")

    origin_node = imageinfo.find("origin")
    if origin_node is None:
        raise ValueError("origin not found in XML")

    ox = int(origin_node.attrib.get("x", 0))
    oy = int(origin_node.attrib.get("y", 0))

    results: List[BarcodeResult] = []

    # --- Iterate over barcodes ---
    for symbol in root.findall(".//symbol"):
        if symbol.attrib.get("type") != "C128":
            continue

        angle_node = symbol.find("angle")
        position_node = symbol.find("position")

        if angle_node is None or position_node is None:
            continue

        angle = float(angle_node.text)

        points: List[Point] = []
        for coord in position_node.findall("coordinate"):
            x_xml = int(coord.attrib["x"])
            y_xml = int(coord.attrib["y"])

            # Apply origin correction
            x_img = x_xml - ox
            y_img = y_xml - oy

            points.append((x_img, y_img))

        if len(points) != 4:
            continue

        results.append({
            "points": points,
            "angle": angle
        })

    return results


# ------------------------------------------------------------
# Drawing (MONO, black rectangles)
# ------------------------------------------------------------

def draw_rotated_barcodes_mono_black(
    image_path: str,
    barcodes: List[BarcodeResult],
    output_path: str | None = None
) -> str:
    """
    Draw rotated barcode rectangles as BLACK lines on a MONO image.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = image.shape[:2]

    for barcode in barcodes:
        points: List[Point] = barcode["points"]

        if len(points) != 4:
            continue

        # Optional safety clipping
        clipped_points = [
            (max(0, min(w - 1, x)), max(0, min(h - 1, y)))
            for x, y in points
        ]

        pts = np.array(clipped_points, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            image,
            [pts],
            isClosed=True,
            color=0,      # BLACK in MONO
            thickness=4
        )

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_debug_black{ext}"

    cv2.imwrite(output_path, image)
    return output_path


# ------------------------------------------------------------
# Main processing
# ------------------------------------------------------------

def process_script(
    files_path: str,
    file_name: str,
    image_type: str = ".jpg",
    xml_type: str = ".xml"
) -> None:
    """
    Full pipeline:
    XML → barcode polygons → draw on MONO image
    """

    xml_file = file_name + xml_type
    image_file = file_name + image_type

    barcodes = get_barcode_results(files_path, xml_file)

    print(f"[INFO] Found {len(barcodes)} barcodes")
    for i, b in enumerate(barcodes):
        print(f"  #{i}: angle={b['angle']} points={b['points']}")

    image_path = os.path.join(files_path, image_file)

    output_image_path = draw_rotated_barcodes_mono_black(
        image_path=image_path,
        barcodes=barcodes
    )

    print(f"[INFO] Debug image saved to: {output_image_path}")
