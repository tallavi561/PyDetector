from typing import List
from pydetector.modules.classes import Barcode, Point
import xml.etree.ElementTree as ET

# ============================================================
# XML parsing
# ============================================================


def load_barcodes(xml_path: str) -> List[Barcode]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    origin = root.find(".//imageinfo/origin")
    ox = int(origin.attrib.get("x", 0))
    oy = int(origin.attrib.get("y", 0))

    barcodes: List[Barcode] = []

    for sym in root.findall(".//symbol"):
        if sym.attrib.get("type") != "C128":
            continue

        angle = float(sym.find("angle").text)
        pts: List[Point] = []

        for c in sym.find("position").findall("coordinate"):
            x = int(c.attrib["x"]) - ox
            y = int(c.attrib["y"]) - oy
            pts.append((x, y))

        if len(pts) == 4:
            barcodes.append(Barcode(points=pts, angle=angle))

    return barcodes