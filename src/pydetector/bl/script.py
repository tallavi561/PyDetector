import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional

import cv2
import numpy as np

from pydetector.utils.image_utils import draw_boxes_and_save

Point = Tuple[int, int]


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class Barcode:
    points_coordinates: List[Point]   # 4 points polygon
    angle: float


@dataclass
class StickerRegion:
    barcode_index: int
    pixels: Set[Point]
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2


# -----------------------------
# XML parsing
# -----------------------------
def get_barcode_results(files_path: str, xml_file_name: str) -> List[Barcode]:
    """
    Parse SICK-like XML and return C128 barcode polygons in image coordinates.
    """
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


# -----------------------------
# Helpers: geometry & masks
# -----------------------------
def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1c = max(0, min(w - 1, x1))
    y1c = max(0, min(h - 1, y1))
    x2c = max(0, min(w - 1, x2))
    y2c = max(0, min(h - 1, y2))
    if x2c < x1c:
        x1c, x2c = x2c, x1c
    if y2c < y1c:
        y1c, y2c = y2c, y1c
    return x1c, y1c, x2c, y2c


def bbox_from_points(points: List[Point]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_mask(h: int, w: int, polygon: List[Point]) -> np.ndarray:
    """
    Return uint8 mask (0/255) for the polygon.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def save_debug_image(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def save_debug_mask(path: str, mask: np.ndarray) -> None:
    """
    Save mask as visible grayscale: 0..255.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if mask.dtype != np.uint8:
        m = mask.astype(np.uint8)
    else:
        m = mask
    cv2.imwrite(path, m)


# -----------------------------
# Step 1: Smart threshold learning
# -----------------------------
def sample_pixels_near_barcode(
    image: np.ndarray,
    barcode_polygon: List[Point],
    ring_thickness: int = 25,
    min_pixels: int = 200
) -> List[int]:
    """
    Sample pixels in a ring around the barcode polygon (dilate - original).
    This tries to capture the sticker/label vicinity without using a fixed ROI.
    """
    h, w = image.shape[:2]

    base = polygon_mask(h, w, barcode_polygon)

    # Dilate polygon to create a "nearby region"
    k = max(3, ring_thickness)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(base, kernel, iterations=1)

    ring = cv2.subtract(dil, base)  # around polygon
    ys, xs = np.where(ring > 0)
    if len(xs) < min_pixels:
        # Fallback: use dilated area itself if ring too small
        ys, xs = np.where(dil > 0)

    if len(xs) == 0:
        return []

    vals = image[ys, xs]
    return vals.astype(np.int32).tolist()


def compute_smart_threshold_from_samples(
    samples: List[int],
    base_white_threshold: int = 90
) -> int:
    """
    Compute a "smart" brightness threshold from samples.
    Uses robust percentiles so it adapts to different exposures.
    """
    if not samples:
        return base_white_threshold

    arr = np.array(samples, dtype=np.float32)

    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    p90 = float(np.percentile(arr, 90))

    # Heuristic: threshold close to upper-middle, but not too aggressive.
    # If contrast is low (p90 ~ p50), keep threshold near p75.
    spread = max(1.0, (p90 - p50))
    thr = p50 + 0.65 * spread  # tuned for "bright sticker background"

    # Clamp to sane range + never below a base minimal threshold
    thr_i = int(max(base_white_threshold, min(250, round(thr))))

    return thr_i


def learn_global_threshold(
    image: np.ndarray,
    barcodes: List[Barcode],
    base_white_threshold: int,
    debug_dir: str
) -> int:
    """
    Learn one threshold for the whole image by aggregating samples near all barcodes.
    """
    all_samples: List[int] = []
    for i, bc in enumerate(barcodes):
        s = sample_pixels_near_barcode(image, bc.points_coordinates, ring_thickness=25)
        if s:
            # Keep bounded number of samples per barcode for speed
            if len(s) > 4000:
                s = random.sample(s, 4000)
            all_samples.extend(s)

    thr = compute_smart_threshold_from_samples(all_samples, base_white_threshold=base_white_threshold)

    # Debug: histogram-like visualization
    try:
        hist = np.zeros((200, 256), dtype=np.uint8)
        if all_samples:
            arr = np.array(all_samples, dtype=np.int32)
            bins = np.bincount(np.clip(arr, 0, 255), minlength=256)
            bins = bins / (bins.max() + 1e-9)
            for x in range(256):
                h = int(bins[x] * 199)
                hist[199 - h:199, x] = 255
            # mark threshold line (set to mid-gray for visibility)
            hist[:, thr] = 150
        save_debug_image(os.path.join(debug_dir, "00_threshold_hist.png"), hist)
    except Exception as e:
        print(f"[DEBUG] Failed to save threshold histogram: {e}")

    print(f"[INFO] Learned global brightness threshold = {thr}")
    return thr


# -----------------------------
# Step 2: Candidate pixels mask
# -----------------------------
def build_candidate_mask(image: np.ndarray, thr: int) -> np.ndarray:
    """
    Return binary candidate mask (uint8: 0/255) where pixel >= thr.
    """
    cand = (image >= thr).astype(np.uint8) * 255
    return cand


# -----------------------------
# Step 3: Barcode-guided "radius wash" growth (efficient)
# -----------------------------
def grow_near_barcode_by_dilation(
    candidate_mask: np.ndarray,
    barcode: Barcode,
    max_radius: int,
    step_radius: int,
    debug_dir: str,
    barcode_index: int
) -> np.ndarray:
    """
    "Smart radius wash":
    Start from candidate pixels near the barcode polygon,
    then iteratively expand by dilating the current region and intersecting with candidates.

    This implements:
    - radius 0: pixels near barcode
    - radius r: pixels that are close to already accepted pixels
    - stop: max_radius reached or no new pixels added
    """
    h, w = candidate_mask.shape[:2]

    # Work in ROI to keep it fast
    bx1, by1, bx2, by2 = bbox_from_points(barcode.points_coordinates)
    x1, y1, x2, y2 = clamp_bbox(bx1 - max_radius, by1 - max_radius, bx2 + max_radius, by2 + max_radius, w, h)

    cand_roi = candidate_mask[y1:y2 + 1, x1:x2 + 1]

    # Seed area: a small dilation of the barcode polygon, intersect with candidates
    base_poly = polygon_mask(h, w, barcode.points_coordinates)[y1:y2 + 1, x1:x2 + 1]
    seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, step_radius * 2 + 1),) * 2)
    seed_area = cv2.dilate(base_poly, seed_kernel, iterations=1)

    region = cv2.bitwise_and(cand_roi, seed_area)

    # Early debug
    save_debug_mask(os.path.join(debug_dir, f"10_bc_{barcode_index:03d}_cand_roi.png"), cand_roi)
    save_debug_mask(os.path.join(debug_dir, f"11_bc_{barcode_index:03d}_seed_area.png"), seed_area)
    save_debug_mask(os.path.join(debug_dir, f"12_bc_{barcode_index:03d}_region_r0.png"), region)

    if cv2.countNonZero(region) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Iterative "radius wash"
    remaining = max_radius
    it = 0
    while remaining > 0:
        it += 1
        dil = cv2.dilate(region, seed_kernel, iterations=1)
        new_region = cv2.bitwise_and(dil, cand_roi)

        # If no growth, stop
        if cv2.countNonZero(new_region) == cv2.countNonZero(region):
            break

        region = new_region
        remaining -= step_radius

        if it <= 6:  # keep debug limited
            save_debug_mask(os.path.join(debug_dir, f"13_bc_{barcode_index:03d}_region_it{it:02d}.png"), region)

    # Put ROI back into full image mask
    out = np.zeros((h, w), dtype=np.uint8)
    out[y1:y2 + 1, x1:x2 + 1] = region
    return out


# -----------------------------
# Step 4: KMeans assignment (K = num barcodes) with smart init
# -----------------------------
def barcode_centers(barcodes: List[Barcode]) -> np.ndarray:
    """
    Return Nx2 float32 centers of each barcode polygon.
    """
    centers = []
    for bc in barcodes:
        xs = [p[0] for p in bc.points_coordinates]
        ys = [p[1] for p in bc.points_coordinates]
        centers.append([float(np.mean(xs)), float(np.mean(ys))])
    return np.array(centers, dtype=np.float32)


def kmeans_assign_pixels_to_barcodes(
    union_mask: np.ndarray,
    barcodes: List[Barcode],
    max_points: int,
    debug_dir: str
) -> List[np.ndarray]:
    """
    Run KMeans on coordinates of union_mask pixels.
    K = number of barcodes.
    Initialize centers using barcode centers (smart).
    Returns a list of K binary masks (uint8 0/255), one per cluster.
    """
    h, w = union_mask.shape[:2]
    k = len(barcodes)
    if k == 0:
        return []

    ys, xs = np.where(union_mask > 0)
    n = len(xs)
    if n == 0:
        return [np.zeros((h, w), dtype=np.uint8) for _ in range(k)]

    # Subsample if too many points (speed)
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        xs_s = xs[idx]
        ys_s = ys[idx]
    else:
        xs_s = xs
        ys_s = ys

    data = np.stack([xs_s, ys_s], axis=1).astype(np.float32)  # Nx2

    # Smart initialization: barcode centers
    init = barcode_centers(barcodes)
    if init.shape[0] != k:
        init = init[:k, :]

    # OpenCV KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.5)
    flags = cv2.KMEANS_USE_INITIAL_LABELS

    # Create initial labels by nearest init center (so we can use USE_INITIAL_LABELS)
    # This keeps clusters aligned with barcodes from the start.
    d2 = ((data[:, None, :] - init[None, :, :]) ** 2).sum(axis=2)  # NxK
    labels0 = np.argmin(d2, axis=1).astype(np.int32).reshape((-1, 1))

    compactness, labels, centers = cv2.kmeans(
        data=data,
        K=k,
        bestLabels=labels0,
        criteria=criteria,
        attempts=1,
        flags=flags
    )

    print(f"[INFO] KMeans compactness: {compactness:.2f}, points used: {len(data)}")

    # Build masks from ALL union points, assigned to nearest kmeans center (fast, deterministic)
    centers = centers.astype(np.float32)
    all_pts = np.stack([xs, ys], axis=1).astype(np.float32)  # Nx2
    d2_all = ((all_pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    lab_all = np.argmin(d2_all, axis=1)

    cluster_masks: List[np.ndarray] = []
    for ci in range(k):
        m = np.zeros((h, w), dtype=np.uint8)
        sel = (lab_all == ci)
        m[ys[sel], xs[sel]] = 255
        cluster_masks.append(m)

        if ci < 8:
            save_debug_mask(os.path.join(debug_dir, f"30_cluster_{ci:03d}.png"), m)

    # Debug: draw centers
    centers_vis = np.zeros((h, w), dtype=np.uint8)
    for (cx, cy) in centers:
        ix = int(max(0, min(w - 1, round(cx))))
        iy = int(max(0, min(h - 1, round(cy))))
        cv2.circle(centers_vis, (ix, iy), 10, 255, thickness=-1)
    save_debug_mask(os.path.join(debug_dir, "31_kmeans_centers.png"), centers_vis)

    return cluster_masks


# -----------------------------
# Post: bounding boxes & cleanup
# -----------------------------
def bounding_box_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def remove_too_small_or_too_big_boxes(
    boxes: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    min_area: int = 400,
    max_area_ratio: float = 0.90
) -> List[Tuple[int, int, int, int]]:
    """
    Remove boxes that are too small or cover too much of the image.
    """
    h, w = image_shape[:2]
    img_area = float(h * w)
    out = []
    for (x1, y1, x2, y2) in boxes:
        bw = max(0, x2 - x1 + 1)
        bh = max(0, y2 - y1 + 1)
        area = bw * bh
        if area < min_area:
            continue
        if area >= max_area_ratio * img_area:
            continue
        out.append((x1, y1, x2, y2))
    return out


# -----------------------------
# Main API
# -----------------------------
def process_image_with_barcodes(
    files_path: str,
    file_name: str,
    base_white_threshold: int = 90,
    max_radius: int = 220,
    step_radius: int = 12,
    kmeans_max_points: int = 120_000
) -> List[StickerRegion]:
    """
    Pipeline:
    1) Load image (grayscale)
    2) Parse barcodes from XML
    3) Learn global smart threshold from barcode vicinity
    4) Candidate mask = pixels >= threshold
    5) For each barcode: grow region by iterative dilation within max_radius
    6) Union all regions, then KMeans into K clusters (K = number of barcodes)
    7) Convert clusters into 1 bounding box per barcode

    Returns: list[StickerRegion] in barcode order (same length as barcodes),
    with empty regions where nothing found.
    """
    image_path = os.path.join(files_path, file_name + ".jpg")
    xml_name = file_name + ".xml"

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = image.shape[:2]
    debug_dir = os.path.join(files_path, f"{file_name}_DEBUG")
    os.makedirs(debug_dir, exist_ok=True)

    print(f"[INFO] Image loaded: {image_path} | shape={image.shape}")

    barcodes = get_barcode_results(files_path, xml_name)
    print(f"[INFO] Barcodes found: {len(barcodes)}")

    if len(barcodes) == 0:
        return []

    # Debug: save original image
    save_debug_image(os.path.join(debug_dir, "00_original.png"), image)

    # Learn threshold
    thr = learn_global_threshold(
        image=image,
        barcodes=barcodes,
        base_white_threshold=base_white_threshold,
        debug_dir=debug_dir
    )

    # Candidate mask
    cand = build_candidate_mask(image, thr)
    save_debug_mask(os.path.join(debug_dir, "01_candidate_mask.png"), cand)

    # Per-barcode smart growth
    per_barcode_masks: List[np.ndarray] = []
    union = np.zeros((h, w), dtype=np.uint8)

    for i, bc in enumerate(barcodes):
        m = grow_near_barcode_by_dilation(
            candidate_mask=cand,
            barcode=bc,
            max_radius=max_radius,
            step_radius=step_radius,
            debug_dir=debug_dir,
            barcode_index=i
        )
        per_barcode_masks.append(m)
        union = cv2.bitwise_or(union, m)

    save_debug_mask(os.path.join(debug_dir, "20_union_before_kmeans.png"), union)

    # KMeans split into K clusters (K = number of barcodes)
    cluster_masks = kmeans_assign_pixels_to_barcodes(
        union_mask=union,
        barcodes=barcodes,
        max_points=kmeans_max_points,
        debug_dir=debug_dir
    )

    # Convert clusters to StickerRegion list
    regions: List[StickerRegion] = []
    boxes: List[Tuple[int, int, int, int]] = []

    for i, m in enumerate(cluster_masks):
        bbox = bounding_box_from_mask(m)
        if bbox is None:
            regions.append(StickerRegion(barcode_index=i, pixels=set(), bounding_box=(0, 0, 0, 0)))
            continue

        # Optional cleanup: remove tiny speckles via opening (light)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        m2 = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        bbox2 = bounding_box_from_mask(m2)
        if bbox2 is not None:
            bbox = bbox2
            m = m2

        # Convert mask pixels to a set (only if you really need it; can be heavy)
        ys, xs = np.where(m > 0)
        pix_set: Set[Point] = set(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))

        regions.append(StickerRegion(barcode_index=i, pixels=pix_set, bounding_box=bbox))
        boxes.append(bbox)

    # Filter extreme boxes (too small / too big)
    filtered_boxes = remove_too_small_or_too_big_boxes(
        boxes=boxes,
        image_shape=image.shape,
        min_area=500,
        max_area_ratio=0.90
    )

    # Debug: final boxes image
    try:
        _ = draw_boxes_and_save(
            image_path=image_path,
            output_path=os.path.join(files_path, file_name + "_stickers.jpg"),
            boxes=filtered_boxes
        )
    except Exception as e:
        print(f"[WARN] draw_boxes_and_save failed: {e}")
    
    print(f"[INFO] Regions produced: {len(regions)} | final boxes after filter: {len(filtered_boxes)}")
    return regions
