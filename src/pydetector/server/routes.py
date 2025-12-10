import base64
import io
from flask import request, jsonify
from PIL import Image

from pydetector.bl.message_handler import process_base64_detection


def register_routes(app):
    @app.route("/api/v1/detectBase64Image", methods=["POST"])
    def detect_base64_image():
        try:
            data = request.get_json()

            if not data or "image" not in data:
                return jsonify({"error": "Missing 'image' field"}), 400

            # save_crops = data.get("saveCrops", False)

            result = process_base64_detection(
                data["image"],
                save_crops=True
            )

            return jsonify({
                "message": "success",
                "detections": result["detections"],
                # "crops": result["crops"]
            })

        except Exception as e:
            print("[ERROR]", str(e))
            return jsonify({"error": str(e)}), 500
