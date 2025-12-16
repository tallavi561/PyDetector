import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional

import cv2
import numpy as np

from pydetector.utils.image_utils import draw_boxes_and_save



Point = Tuple[int, int]

@dataclass(frozen=True)
class Barcode:
    points_coordinates: List[Point]   # 4 points polygon
    angle: float

@dataclass
class StickerRegion:
    barcode_index: int
    pixels: Set[Point]
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
def get_barcode_results(files_path: str, xml_file_name: str) -> List[Barcode]:
    xml_path = os.path.join(files_path, xml_file_name)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    imageinfo = root.find(".//imageinfo")
    if imageinfo is None:
        raise ValueError("imageinfo not found")

    origin_node = imageinfo.find("origin")
    if origin_node is None:
        raise ValueError("origin not found")

    ox = int(origin_node.attrib.get("x", 0))
    oy = int(origin_node.attrib.get("y", 0))

    barcodes: List[Barcode] = []

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
            x = int(coord.attrib["x"]) - ox
            y = int(coord.attrib["y"]) - oy
            points.append((x, y))

        if len(points) == 4:
            barcodes.append(Barcode(points_coordinates=points, angle=angle))

    return barcodes

def sample_white_pixels_near_barcode(
    image: np.ndarray,
    barcode_points: List[Point],
    white_threshold: int,
    min_pixels: int = 10
) -> List[int]:

    mask = np.zeros(image.shape, dtype=np.uint8)
    pts = np.array(barcode_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    ys, xs = np.where(mask == 255)
    values = image[ys, xs]

    white_values = values[values >= white_threshold]

    if len(white_values) < min_pixels:
        return []

    return white_values.tolist()

def compute_median_tone(values: List[int]) -> int:
    return int(np.median(values))


def region_grow(
    image: np.ndarray,
    seed: list[Point],
    median_tone: int,
    delta: int,
    neighbors_distance_limit: int
) -> Set[Point]:

    h, w = image.shape
    stack = seed.copy()
    visited: Set[Point] = set()
    region: Set[Point] = set()
    neighbors_distances = [i for i in range(1, neighbors_distance_limit)]
    neighbors_offsets = [
        (dx, dy)
        for dist in neighbors_distances
        for dx in range(-dist, dist + 1)
        for dy in range(-dist, dist + 1)
        if abs(dx) + abs(dy) == dist
    ]
    print(f"neighbors_offsets: {neighbors_offsets} len of seed: {len(seed)}")
    debug_counter = 0
    while stack:
        debug_counter +=1
        x, y = stack.pop()

        if (x, y) in visited:

            continue
        if not (0 <= x < w and 0 <= y < h):
            continue

        visited.add((x, y))
        pixel = int(image[y, x])
      #   print(f"debug_counter: {debug_counter}, pixel: {pixel}, median_tone: {median_tone}, delta: {delta}")
        if pixel + delta< median_tone :
            continue

        region.add((x, y))

        for dx, dy in neighbors_offsets:
            stack.append((x + dx, y + dy))
    print(f"debug_counter: {debug_counter}")
    return region

def bounding_box_from_pixels(pixels: Set[Point]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    return min(xs), min(ys), max(xs), max(ys)

def extract_sticker_for_barcode(
    image: np.ndarray,
    barcode: Barcode,
    barcode_index: int,
    white_threshold: int,
    delta: int
) -> Optional[StickerRegion]:
    
    samples = sample_white_pixels_near_barcode(
        image=image,
        barcode_points=barcode.points_coordinates,
        white_threshold=white_threshold
    )

    if len(samples) < 30:
        samples = sample_white_pixels_near_barcode(
            image=image,
            barcode_points=barcode.points_coordinates,
            white_threshold=int(white_threshold * 0.7),
        )
    if len(samples) < 30:
        samples = sample_white_pixels_near_barcode(
            image=image,
            barcode_points=barcode.points_coordinates,
            white_threshold=int(white_threshold * 0.5),
        )

    if not samples:
        print(f"[WARN] Not enough white pixels near barcode #{barcode_index}")
        return None
    else:
        print(f"[INFO] Sampled {len(samples)} white pixels near barcode #{barcode_index}")
    mt = compute_median_tone(samples)

#     seed_x = int(np.mean([p[0] for p in barcode.points_coordinates]))
#     seed_y = int(np.mean([p[1] for p in barcode.points_coordinates]))
#     seed = (seed_x, seed_y)
#     seed_x_sample = random.sample()
    print(f"[INFO] Median tone for barcode #{barcode_index}: {mt}")
    count_samples = min(10, len(barcode.points_coordinates))
    print(f"sample {count_samples} from barcode points coordinates - {len(barcode.points_coordinates)} points")
    seed: list[Point] = random.sample(barcode.points_coordinates, count_samples)
    neighbors_distance_limit = 3
    if  len(barcode.points_coordinates) > 1000000:
        neighbors_distance_limit = 2
    region = region_grow(
        image=image,
        seed=seed,
        median_tone=mt,
        delta=delta,
        neighbors_distance_limit=neighbors_distance_limit
    )

    if not region:
        print(f"[WARN] Region growing failed for barcode #{barcode_index}")
        return None
    print(f"[INFO] Region growing found {len(region)} pixels for barcode #{barcode_index}")
    bbox = bounding_box_from_pixels(region)

    return StickerRegion(
        barcode_index=barcode_index,
        pixels=region,
        bounding_box=bbox
    )


# ------------------------------------------------------------
# Drawing (MONO, black rectangles)
# ------------------------------------------------------------
def draw_rotated_barcodes_mono_black(
    image_path: str,
    barcodes: List[Barcode],
    output_path: Optional[str] = None
) -> str:
    """
    Draw rotated barcode rectangles as BLACK lines on a MONO image.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = image.shape[:2]

    for barcode in barcodes:
        if len(barcode.points_coordinates) != 4:
            continue

        clipped_points = [
            (max(0, min(w - 1, x)), max(0, min(h - 1, y)))
            for x, y in barcode.points_coordinates
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
def process_image_with_barcodes(
    files_path: str,
    file_name: str,
    white_threshold: int = 90,
    delta: int = 30
) -> List[StickerRegion]:

    image_path = os.path.join(files_path, file_name + ".jpg")
    xml_path = file_name + ".xml"

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to load image")

    barcodes: List[Barcode]= get_barcode_results(files_path, xml_path)
    stickers: List[StickerRegion] = []
    print(f"barcodes: {len(barcodes)}")
    for i, barcode in enumerate(barcodes):
        sticker = extract_sticker_for_barcode(
            image=image,
            barcode=barcode,
            barcode_index=i,
            white_threshold=white_threshold,
            delta=delta
        )

        if sticker:
            print(f"[INFO] Extracted sticker for barcode #{i} with {len(sticker.pixels)} pixels")
            stickers.append(sticker)
        else:
            print(f"[WARN] No sticker extracted for barcode #{i}")
    print(f"stickers: {len(stickers)}")
#     print(f"stickers boxes: {stickers}")
    image_with_stickers = draw_boxes_and_save(
        image_path=image_path,
            output_path=os.path.join(files_path, file_name + "_stickers.jpg"),
            boxes=[s.bounding_box for s in stickers]
    )
    return stickers
