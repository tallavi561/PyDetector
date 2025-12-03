# my_server/bl/detect_service.py
import base64
import io
from PIL import Image

from pydetector.bl.detect import detector
from pydetector.utils.image_utils import (
    save_base64_to_image,
    crop_image_to_output
)


def process_base64_detection(b64_img: str, save_crops: bool = False):
    """
    Handles detection pipeline:
    1. Decode base64 and save image
    2. Run YOLO detection
    3. Optionally crop detections
    4. Return data for response
    """

    # 1. Save input image
    input_path, filename = save_base64_to_image(b64_img, "input_pictures")
    print(f"[INFO] Saved input image: {input_path}, filename: {filename}")

    # 2. Run detection
    detection_result = detector.detect(input_path)
    boxes = detection_result.get("objects", [])
    print(f"[INFO] Detected {len(boxes)} objects")

    cropped_paths = []
    if save_crops:
        for idx, obj in enumerate(boxes, start=1):
            x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]

            output_filename = (
                f"{filename}_{idx}_conf{int(obj['confidence']*100)}%.jpg"
            )

            crop_path = crop_image_to_output(
                input_path, output_filename, x1, y1, x2, y2
            )

            cropped_paths.append(crop_path)
            print(f"[INFO] Saved crop: {crop_path}")

    return {
        "filename": filename,
        "detections": boxes,
        "crops": cropped_paths
    }
