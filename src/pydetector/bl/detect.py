import os
from ultralytics import YOLO
import cv2
import numpy as np
import onnx
from PIL import Image

from pydetector.utils.image_utils import crop_image_to_output

model_version = "12"
model_size = "x"


class YoloObjectDetector:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        # Load YOLO model (Ultralytics)
        self.model = YOLO(model_path)
        print(f"[INFO] Loaded YOLO model (Ultralytics) from: {model_path}")

        # Detect if the model is ONNX or PT
        self.is_onnx = model_path.lower().endswith(".onnx")

        if self.is_onnx:
            print("[INFO] Loading ONNX graph metadata...")
            self.onnx_model = onnx.load(model_path)

            print("Model outputs:")
            for o in self.onnx_model.graph.output:
                print("•", o.name)

            print("\nModel output shapes:")
            for o in self.onnx_model.graph.output:
                shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
                print("•", o.name, "=>", shape)

            print("[INFO] ONNX model loaded successfully.\n")
        else:
            print("[INFO] PT model detected — skipping ONNX metadata.\n")

    def detect(self, image_path: str, conf_threshold: float = 0.01, save_outputs: bool = False) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Failed to load image (invalid or corrupted): {image_path}")

        # YOLO inference (identical for PT and ONNX when using Ultralytics)
        results = self.model(image_path, conf=conf_threshold, verbose=False)[0]

        detections = []
        index = 0

        for box in results.boxes:
            index += 1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            if save_outputs:
                crop_image_to_output(
                    image_path,
                    f"crop_outputs/crop_{index}_inputname+{image_path[-10:]}_cls_name+{cls_name}_conf+{int(conf*100)}.jpg",
                    int(x1), int(y1), int(x2), int(y2)
                )

            detections.append({
                "X1": int(x1),
                "Y1": int(y1),
                "X2": int(x2),
                "Y2": int(y2),
                "confidence": conf,
                "class_name": cls_name
            })

            print(f" • Detected {cls_name} (ID: {cls_id} | Conf: {conf:.4f})")

        return {"objects": detections}


# detector = YoloObjectDetector(f"models/yolov{model_version}{model_size}.onnx")
detector = YoloObjectDetector(f"models/yolo{model_version}{model_size}.pt")
