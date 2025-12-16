from pydetector.bl.script import process_image_with_barcodes
from pydetector.server.app import create_app

path = "new-pictures/"
files_names = ["a",  "b" ,"c",  "d"]
for fn in files_names:
    file_name = fn
    files_path = path
    print(f"Processing file: {file_name}")
    process_image_with_barcodes(
        files_path=files_path,
        file_name=file_name
    )
      

print("Starting PyDetector...")
def main():
    app = create_app()
    app.run(host="0.0.0.0", port=5000)