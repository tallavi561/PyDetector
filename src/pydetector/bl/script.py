import os
from typing import List, Tuple
import cv2
import numpy as np
from pydetector.modules.classes import Barcode, Point, StepPosition, StickerResult, Vector2D
from pydetector.utils.image_utils import draw_rotated_barcodes_on_image, get_pixels_inside_barcode, save_brightness_distribution
from pydetector.utils.math import calculate_vector_by_points_and_length, find_bimodal_threshold, get_changes_vectors_by_vector_and_position
from pydetector.utils.xml_utils import load_barcodes

positions: list[StepPosition] = [StepPosition.FORWARD, StepPosition.BACK, StepPosition.RIGHT, StepPosition.LEFT]
def deep_copy_barcode(barcode : Barcode)->Barcode:
     copy_points: List[Point] = []
     for barcode_points in barcode.points:
          copy_point: Point = (barcode_points[0], barcode_points[1])
          copy_points.append(copy_point)
     
     copy_barcode:Barcode = Barcode(points=copy_points, angle=barcode.angle)
     return copy_barcode


      




"""
The first Barcode that returned is the expansion of the input barcode.
The seconde, is the expansion itself.
"""
def expand_barcode_bbox_by_position(barcode: Barcode,
                        stepPosition: StepPosition,
                        expantion_pixels: int = 100
                        ) -> tuple[Barcode, Barcode]:
    copy_expanded_barcode = deep_copy_barcode(barcode)
    expantion_vectors, points_directions = get_changes_vectors_by_vector_and_position(barcode, stepPosition, expantion_pixels)
#     print(f"barcode: {barcode} go stepPosition: {stepPosition}")
    for index in range(len(barcode.points)):
         copy_expanded_barcode.points[index] = (int(barcode.points[index][0] + expantion_vectors[index].dx),
                  int(barcode.points[index][1] + expantion_vectors[index].dy))
    
    only_expantion_barcode = deep_copy_barcode(copy_expanded_barcode)
    for points_direction in points_directions:
        only_expantion_barcode.points[points_direction.fromPointIndex] = barcode.points[points_direction.toPointIndex]
#     print(f"barcode: {barcode}")
    return copy_expanded_barcode, only_expantion_barcode

"""
The seconde return value, is the new barcode - if True, with the step, else- without
"""
def is_new_step_in_sticker(files_path: str,
      file_name: str,
      barcode: Barcode,
      stepPosition: StepPosition,
      expantion_pixels: int=120,
      barcode_index_debug: int=0,
      debug_index:int = 0,
      picture_type: str=".jpg")-> tuple[bool, Barcode]:
    image_path = os.path.join(files_path, f"{file_name}{picture_type}")
    expanded_barcode, expansion_only =  expand_barcode_bbox_by_position(barcode,
                                    stepPosition,
                                    expantion_pixels=expantion_pixels)
#     draw_rotated_barcodes_on_image(image_path,
#                                    [expanded_barcode],
#                                    output_path=os.path.join(files_path, f"{file_name}-bc-{barcode_index_debug}{stepPosition}{debug_index}_barcodes_after_step{picture_type}"))
#     draw_rotated_barcodes_on_image(image_path,
#                                    [expansion_only],
#                                    output_path=os.path.join(files_path, f"{file_name}-br-{barcode_index_debug}{stepPosition}{debug_index}_expansion_only{picture_type}"))        
    
    pixels_values_in_current_area = get_pixels_inside_barcode(image_path, barcode)
    current_threshold = find_bimodal_threshold(pixels_values_in_current_area)
    print(f"current_threshold: {current_threshold}")
    pixels_values_in_currnet_area_grather_than_T = [pixel for pixel in pixels_values_in_current_area if pixel >= current_threshold]
    brightness_in_current_area = len(pixels_values_in_currnet_area_grather_than_T)/len(pixels_values_in_current_area)
    print(f"pixels_values_in_currnet_area_grather_than_T: {brightness_in_current_area}")
    # save_brightness_distribution(pixels_values_in_current_area, files_path + file_name+"-br-"+str(barcode_index_debug)+str(stepPosition)+str(debug_index) + f"_histogram_current.jpg", step_size=5, threshold=current_threshold)
    pixels_values_in_new_area = get_pixels_inside_barcode(image_path, expansion_only)
    new_area_threshold = find_bimodal_threshold(pixels_values_in_new_area)
    print(f"new_area_threshold: {new_area_threshold}")
    # save_brightness_distribution(pixels_values_in_new_area, files_path + file_name+"-br-"+str(barcode_index_debug)+str(stepPosition)+str(debug_index) + f"_histogram_new.jpg", step_size=5, threshold=current_threshold)
    pixels_values_in_new_area_grather_than_T = [pixel for pixel in pixels_values_in_new_area if pixel >= current_threshold]
    brightness_in_new_area = len(pixels_values_in_new_area_grather_than_T)/len(pixels_values_in_new_area)
    print(f"pixels_values_in_new_area_grather_than_T: {brightness_in_new_area}")
    if  brightness_in_new_area >= max(brightness_in_current_area- 0.2, 0.55):
        return (True , expanded_barcode)
    else:
        return (False , barcode)


# ============================================================
# Main pipeline
# ============================================================
def process_image(files_path: str,
      file_name: str,
      picture_type: str=".jpg",

      ) -> List[StickerResult]:
    xml_path = os.path.join(files_path, f"{file_name}.xml")
    image_path = os.path.join(files_path, f"{file_name}{picture_type}")
    barcodes: list[Barcode] = load_barcodes(xml_path)
#     print(f"[INFO] Found {len(barcodes)} barcodes in XML: {barcodes}")
    draw_rotated_barcodes_on_image(image_path,
                                   barcodes,
                                   output_path=os.path.join(files_path, f"{file_name}_barcodes{picture_type}"))
    for barcode_index in range(len(barcodes)):
      for step_direction in [StepPosition.BACK, StepPosition.LEFT, StepPosition.FORWARD, StepPosition.RIGHT]:
            for index in range(30):
                  step_in_sticker, barcode =  is_new_step_in_sticker(files_path=files_path,
                        file_name=file_name,
                        barcode=barcodes[barcode_index],
                        barcode_index_debug=barcode_index,
                        stepPosition= step_direction,
                        debug_index=index)
                  if not step_in_sticker:
                        break
                  barcodes[barcode_index] = barcode
      print(f"Frist loop was finished")
      for step_direction in [StepPosition.BACK, StepPosition.LEFT, StepPosition.FORWARD, StepPosition.RIGHT]:
            for index in range(40, 70):
                  step_in_sticker, barcode =  is_new_step_in_sticker(files_path=files_path,
                        file_name=file_name,
                        barcode=barcodes[barcode_index],
                        barcode_index_debug=barcode_index,
                        stepPosition= step_direction,
                        debug_index=index)
                  if not step_in_sticker:
                        break
                  barcodes[barcode_index] = barcode
    draw_rotated_barcodes_on_image(files_path + file_name + picture_type,
      barcodes=barcodes,
      output_path=files_path + "FINAL" +file_name + picture_type
    )
    A = "<" * 10
    B = ">" * 10
    AB  = f"{A}{B}\n" * 5
    print(f"{AB}")
    return []

# ============================================================
# Entry helper (your wrapper)
# ============================================================
def process_image_with_barcodes(files_path: str, file_name: str) -> None:
    image_path = os.path.join(files_path, f"{file_name}.jpg")
    xml_path = os.path.join(files_path, f"{file_name}.xml")

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return
    if not os.path.exists(xml_path):
        print(f"[ERROR] XML not found: {xml_path}")
        return

    print(f"[INFO] Processing image: {image_path}")

    results = process_image(files_path, file_name)
    boxes = [r.bbox for r in results]

