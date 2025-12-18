from dataclasses import dataclass
from pydetector.modules.classes import POINTS_DIRECTION, POINTS_DIRECTION_LIST, Barcode, Point, StepPosition, Vector2D
from typing import List, Literal
import numpy as np

def calculate_vector_by_points_and_length(source_point: Point, dest_point: Point, length: float) -> Vector2D:
    from math import sqrt
    dx = dest_point[0] - source_point[0]
    dy = dest_point[1] - source_point[1]
    dist = sqrt(dx * dx + dy * dy)
    if dist == 0:
        return Vector2D(0.0, 0.0)
    scale = length / dist
    return Vector2D(dx * scale, dy * scale)




"""
The seconde respone is the indexes points that need to be change
"""
def get_changes_vectors_by_vector_and_position(barcode: Barcode, stepPosition: StepPosition, vector_length: float=10.0) -> tuple[list[Vector2D], POINTS_DIRECTION_LIST]:

      # in index 0 there is the vector that need to be added to the barcode-point in index 0, and so on.
      vector_for_barcode_points: list[Vector2D] = [
            Vector2D(0.0, 0.0),
            Vector2D(0.0, 0.0),
            Vector2D(0.0, 0.0),
            Vector2D(0.0, 0.0)
      ]
      source_point = (0,0)
      dest_point = (0,0)
      points_direction: POINTS_DIRECTION_LIST = [
            POINTS_DIRECTION(-1, -1),
            POINTS_DIRECTION(-1, -1),
      ]

      match stepPosition:
            case StepPosition.FORWARD:
                  source_point = barcode.points[0]
                  dest_point = barcode.points[1]
                  vector = calculate_vector_by_points_and_length(source_point, dest_point, vector_length)
                  vector_for_barcode_points[1] = vector
                  vector_for_barcode_points[2] = vector
                  points_direction[0].fromPointIndex = 0
                  points_direction[0].toPointIndex = 1
                  points_direction[1].fromPointIndex = 3
                  points_direction[1].toPointIndex = 2
            case StepPosition.BACK:
                  source_point = barcode.points[1]
                  dest_point = barcode.points[0]
                  vector = calculate_vector_by_points_and_length(source_point, dest_point, vector_length)
                  vector_for_barcode_points[0] = vector
                  vector_for_barcode_points[3] = vector
                  points_direction[0].fromPointIndex = 1
                  points_direction[0].toPointIndex = 0
                  points_direction[1].fromPointIndex = 2
                  points_direction[1].toPointIndex = 3
            case StepPosition.RIGHT:
                  source_point = barcode.points[1]
                  dest_point = barcode.points[2]
                  vector = calculate_vector_by_points_and_length(source_point, dest_point, vector_length)
                  vector_for_barcode_points[2] = vector
                  vector_for_barcode_points[3] = vector
                  points_direction[0].fromPointIndex = 1
                  points_direction[0].toPointIndex = 2
                  points_direction[1].fromPointIndex = 0
                  points_direction[1].toPointIndex = 3
            case StepPosition.LEFT:
                  source_point = barcode.points[2]
                  dest_point = barcode.points[1]
                  vector = calculate_vector_by_points_and_length(source_point, dest_point, vector_length)
                  vector_for_barcode_points[0] = vector
                  vector_for_barcode_points[1] = vector
                  points_direction[0].fromPointIndex = 2
                  points_direction[0].toPointIndex = 1
                  points_direction[1].fromPointIndex = 3
                  points_direction[1].toPointIndex = 0
      return vector_for_barcode_points, points_direction


def is_zero_vectror(vector: Vector2D)->bool:
      return vector.dx == 0 and vector.dy ==0


def find_bimodal_threshold(
    pixel_values: List[int],
    method: Literal["otsu", "kmeans"] = "otsu",
    kmeans_iters: int = 30,
    kmeans_seed: int = 0
) -> int:
    """
    Finds a threshold that separates two 'hills' (bimodal distribution) in grayscale values.

    Args:
        pixel_values: list of grayscale values (0..255)
        method: "otsu" (recommended default) or "kmeans"
        kmeans_iters: iterations for 1D kmeans
        kmeans_seed: random seed for kmeans init

    Returns:
        threshold in [0..255]
    """
    if not pixel_values:
        raise ValueError("pixel_values is empty")

    vals = np.asarray(pixel_values, dtype=np.uint8)

    if method == "otsu":
        # Build histogram
        hist = np.bincount(vals, minlength=256).astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 0

        prob = hist / total
        omega = np.cumsum(prob)                 # class probabilities
        mu = np.cumsum(prob * np.arange(256))   # class means * probs
        mu_t = mu[-1]

        # Between-class variance: sigma_b^2 = (mu_t*omega - mu)^2 / (omega*(1-omega))
        denom = omega * (1.0 - omega)
        denom[denom == 0] = np.nan
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom

        t = int(np.nanargmax(sigma_b2))
        return t

    if method == "kmeans":
        # 1D k-means with k=2
        x = vals.astype(np.float64)

        rng = np.random.default_rng(kmeans_seed)
        c1, c2 = rng.choice(x, size=2, replace=False)

        for _ in range(kmeans_iters):
            # Assign
            d1 = np.abs(x - c1)
            d2 = np.abs(x - c2)
            g1 = x[d1 <= d2]
            g2 = x[d1 > d2]

            # Update (avoid empty cluster)
            new_c1 = g1.mean() if g1.size else c1
            new_c2 = g2.mean() if g2.size else c2

            if np.isclose(new_c1, c1) and np.isclose(new_c2, c2):
                break

            c1, c2 = new_c1, new_c2

        low, high = (c1, c2) if c1 <= c2 else (c2, c1)
        threshold = int(round((low + high) / 2.0))
        return max(0, min(255, threshold))

    raise ValueError(f"Unknown method: {method}")