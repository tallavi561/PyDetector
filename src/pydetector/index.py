from pydetector.bl.script import process_image_with_barcodes
from pydetector.server.app import create_app

# images_path = "new-pictures/"
# files_names = [
#       "a", "b", "c", "d"]
# for file_name in files_names:
#     process_image_with_barcodes(
#         images_path,
#         file_name=file_name)
print("Starting PyDetector...")
def main():
    app = create_app()
    app.run(host="0.0.0.0", port=5000)