import os
from flask import Flask
from pydetector.utils.models_downloader import export_model_to_onnx
from ultralytics import YOLO
from pydetector.server.routes import register_routes
from pydetector.utils.image_utils import ensure_directories
# from pydetector.bl.detect import model_version, model_size


def create_app():
    app = Flask(__name__)


    # export_model_to_onnx(model_version, model_size)
    ensure_directories()
    register_routes(app)
    return app



