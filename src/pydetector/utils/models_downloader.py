import os
from ultralytics import YOLO

def download_model_if_missing(model_path, model_name):
    if not os.path.exists(model_path):
        print(f"[INFO] Downloading {model_name} to {model_path} ...")
        try:
            model = YOLO(model_name)   # זה מוריד מהאינטרנט
            model.save(model_path)     # וזה שומר לתיקיית models
            print(f"[SUCCESS] Downloaded to {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download {model_name}: {e}")
            return False
    else:
        print(f"[INFO] Model already exists: {model_path}")
    return True


def export_model_to_onnx(version, size):
    model_name = f"yolov{version}{size}.pt"
    model_path = os.path.join("models", model_name)
    onnx_path = os.path.join("models", f"yolov{version}{size}.onnx")

    os.makedirs("models", exist_ok=True)
    if not download_model_if_missing(model_path, model_name):
        return

    if os.path.exists(onnx_path):
        print(f"[INFO] ONNX already exists: {onnx_path}")
        return

    print(f"[INFO] Loading model from {model_path}...")
    model = YOLO(model_path)
    for o in model.graph.output:
        print(o.name)
    try:
        print(f"[INFO] Exporting to ONNX...")
        result = model.export(
            format="onnx",
            imgsz=640,
            opset=17,
            dynamic=True,
            simplify=True,
        )

        os.rename(result, onnx_path)

        print(f"[SUCCESS] ONNX exported to {onnx_path}")

    except Exception as e:
        print(f"[ERROR] Failed exporting ONNX: {e}")


# import os
# from ultralytics import YOLO

# def export_model(version="12", size="l"):
#     model_name = f"yolov{version}{size}.pt"
#     onnx_path = f"models/yolov{version}{size}.onnx"

#     os.makedirs("models", exist_ok=True)

#     print(f"[INFO] Loading model {model_name} ...")
#     model = YOLO(model_name)  # Ultralytics יוריד אוטומטית אם לא קיים

#     # אם ה-ONNX כבר קיים - אין צורך לייצא מחדש
#     if os.path.exists(onnx_path):
#         print(f"[INFO] ONNX already exists: {onnx_path}")
#         return

#     print(f"[INFO] Exporting {model_name} to ONNX...")
#     result_paths = model.export(
#         format="onnx",
#         imgsz=640,
#         opset=17,
#         dynamic=True,
#         simplify=True,
#         outdir="models",
#     )

#     print(f"[SUCCESS] ONNX exported: {onnx_path}")
