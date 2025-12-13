from pydetector.bl.script import print_image_info
from pydetector.server.app import create_app


print("Strarting script...")
print_image_info("new-pictures/a.jpg")

print("Starting PyDetector...")
def main():
    app = create_app()
    app.run(host="0.0.0.0", port=5000)