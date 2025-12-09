# my_server/bl/detect_service.py
import base64
import io
from PIL import Image

from pydetector.bl.detect import detector
from pydetector.utils.image_utils import (
    image_preprocess_clahe,
    save_base64_to_image,
    crop_image_to_output
)
import cv2


def process_base64_detection(b64_img: str, save_crops: bool):
    """
    Handles detection pipeline:
    1. Decode base64 and save image
    2. Run YOLO detection
    3. Optionally crop detections
    4. Return data for response
    """

    # 0. Save input image
    input_path, filename = save_base64_to_image(b64_img, "input_pictures")
    print(f"[INFO] Saved input image: {input_path}, filename: {filename}")

    # 1. pre-process image (if needed)
    processed_path, pil_img = image_preprocess_clahe(input_path, filename)

    # 2. Run detection
    detection_result = detector.detect(processed_path, conf_threshold=0.06, save_outputs=save_crops)
    boxes = detection_result.get("objects", [])
    print(f"[INFO] Detected {len(boxes)} objects")


    return {
        "filename": filename,
        "detections": boxes,
    }
