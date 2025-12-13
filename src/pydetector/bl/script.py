from typing import Literal
import cv2
import os
import numpy as np


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
