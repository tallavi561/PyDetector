from pydetector.bl.script import mark_relevant_boxes_from_xml, print_image_info
from pydetector.server.app import create_app


print("Strarting script...")
# print_image_info("new-pictures/a.jpg")
mark_relevant_boxes_from_xml("new-pictures/a.xml","new-pictures/a.jpg",  "new-pictures/a_marked.jpg")
mark_relevant_boxes_from_xml("new-pictures/b.xml","new-pictures/b.jpg",  "new-pictures/b_marked.jpg")
mark_relevant_boxes_from_xml("new-pictures/c.xml","new-pictures/c.jpg",  "new-pictures/c_marked.jpg")
print("Starting PyDetector...")
def main():
    app = create_app()
    app.run(host="0.0.0.0", port=5000)