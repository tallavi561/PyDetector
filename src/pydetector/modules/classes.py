from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

Box = Tuple[int, int, int, int]  # x1,y1,x2,y2
Point = Tuple[int, int]

@dataclass(frozen=True)
class Vector2D:
    dx: float
    dy: float

SOURCE_TO_DEST_POINTS = tuple[Point, Point]

# ============================================================
# Models
# ============================================================
@dataclass(frozen=True)
class Barcode:
    corners_points: List[Point]
    angle: float  # degrees


@dataclass
class StickerResult:
    barcode_index: int
    bbox: Box


class StepPosition(Enum):
    FORWARD = "FORWARD" # Brcode points[0] to points[1]
    BACK = "BACK" # Brcode points[1] to points[0]
    RIGHT = "RIGHT" # Barcode points[0] to points[3]
    LEFT = "LEFT" # Barcode points[3] to points[0]


@dataclass
class POINTS_DIRECTION:
      fromPointIndex: int
      toPointIndex: int

POINTS_DIRECTION_LIST = list[POINTS_DIRECTION]