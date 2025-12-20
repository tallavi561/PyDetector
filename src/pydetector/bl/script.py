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
     for barcode_points in barcode.corners_points:
          copy_point: Point = (barcode_points[0], barcode_points[1])
          copy_points.append(copy_point)
     
     copy_barcode:Barcode = Barcode(corners_points=copy_points, angle=barcode.angle)
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
    for index in range(len(barcode.corners_points)):
         copy_expanded_barcode.corners_points[index] = (int(barcode.corners_points[index][0] + expantion_vectors[index].dx),
                  int(barcode.corners_points[index][1] + expantion_vectors[index].dy))
    
    only_expantion_barcode = deep_copy_barcode(copy_expanded_barcode)
    for points_direction in points_directions:
        only_expantion_barcode.corners_points[points_direction.fromPointIndex] = barcode.corners_points[points_direction.toPointIndex]
#     print(f"barcode: {barcode}")
    return copy_expanded_barcode, only_expantion_barcode

def is_corners_bright(expanded_barcode: Barcode,
                      image_path: str,
                      rectangle_length: int,
                      brightness_threshold:int,
                      brightness_percentage:int):
     pass
     

"""
The seconde return value, is the new barcode - if True, with the step, else- without
"""
def is_new_step_in_sticker(files_path: str,
      file_name: str,
      barcode: Barcode,
      stepPosition: StepPosition,
      image: np.ndarray,
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
    image_h, image_w = image.shape
    for expansion_only_points in expansion_only.corners_points:
         if expansion_only_points[0] < 0 or expansion_only_points[0] > image_w:
              return (False , barcode)
         if expansion_only_points[1] < 0 or expansion_only_points[1] > image_h:
              return (False , barcode)
    pixels_values_in_current_area = get_pixels_inside_barcode(image_path, barcode, image)
    current_threshold = find_bimodal_threshold(pixels_values_in_current_area)
#     print(f"current_threshold: {current_threshold}")
    pixels_values_in_currnet_area_grather_than_T = [pixel for pixel in pixels_values_in_current_area if pixel >= current_threshold]
    brightness_in_current_area = len(pixels_values_in_currnet_area_grather_than_T)/len(pixels_values_in_current_area)
#     print(f"pixels_values_in_currnet_area_grather_than_T: {brightness_in_current_area}")
    # save_brightness_distribution(pixels_values_in_current_area, files_path + file_name+"-br-"+str(barcode_index_debug)+str(stepPosition)+str(debug_index) + f"_histogram_current.jpg", step_size=5, threshold=current_threshold)
    print("<><><><>")
    print(f"{barcode=}")
    print(f"{expansion_only=}")
    pixels_values_in_new_area = get_pixels_inside_barcode(image_path, expansion_only, image)
    # new_area_threshold = find_bimodal_threshold(pixels_values_in_new_area)
#     print(f"new_area_threshold: {new_area_threshold}")
    # save_brightness_distribution(pixels_values_in_new_area, files_path + file_name+"-br-"+str(barcode_index_debug)+str(stepPosition)+str(debug_index) + f"_histogram_new.jpg", step_size=5, threshold=current_threshold)
    pixels_values_in_new_area_grather_than_T = [pixel for pixel in pixels_values_in_new_area if pixel >= current_threshold]
    brightness_in_new_area = len(pixels_values_in_new_area_grather_than_T)/len(pixels_values_in_new_area)
#     print(f"pixels_values_in_new_area_grather_than_T: {brightness_in_new_area}")
    if  brightness_in_new_area >= max(brightness_in_current_area- 0.2, 0.55):
        return (True , expanded_barcode)
    else:
        return (False , barcode)

def expentions_for_loops(
          files_path: str,
          file_name: str,
          image: np.ndarray,
          barcodes:List[Barcode],
          steps_directions: list[StepPosition],
          debug_indexes: tuple[int, int]
      ):
    # stepPosition remover
    directions_not_in_sticker: dict[StepPosition, int] = {sp: 0 for sp in steps_directions}

    for barcode_index in range(len(barcodes)):
      steps_directions_copy = [sp for sp in steps_directions]
      for index in range(debug_indexes[0], debug_indexes[1]):
            print(f"debug_indexes: {index} from {debug_indexes[0]} to {debug_indexes[1]}")
            for step_direction in steps_directions_copy:
                  print(f"step_direction: {step_direction}")
                  expantion_pixels = 120
                  print(f"expantion_pixels: {expantion_pixels}")
                  step_in_sticker, barcode =  is_new_step_in_sticker(
                        files_path=files_path,
                        file_name=file_name,
                        image=image,
                        barcode=barcodes[barcode_index],
                        barcode_index_debug=barcode_index,
                        stepPosition= step_direction,
                        expantion_pixels=expantion_pixels,
                        debug_index=index)
                  if not step_in_sticker:

                        expantion_pixels = 60
                        print(f" if not step_in_sticker -> expantion_pixels: {expantion_pixels}")
                        step_in_sticker, barcode =  is_new_step_in_sticker(
                        files_path=files_path,
                        file_name=file_name,
                        image=image,
                        barcode=barcodes[barcode_index],
                        barcode_index_debug=barcode_index,
                        stepPosition= step_direction,
                        expantion_pixels=expantion_pixels,
                        debug_index=index)
                  if step_in_sticker:
                        directions_not_in_sticker[step_direction] = 0
                        barcodes[barcode_index] = barcode
                  else:
                       directions_not_in_sticker[step_direction] +=1
                       if directions_not_in_sticker[step_direction] >= 10:
                            steps_directions_copy.remove(step_direction)
                            print("not step_in_sticker, removed")
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
    steps_directions = [StepPosition.LEFT, StepPosition.FORWARD, StepPosition.RIGHT, StepPosition.BACK, ]
    image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    expentions_for_loops(
         files_path=files_path,
         file_name=file_name,
         barcodes=barcodes,
         image=image,
         steps_directions=steps_directions,
        debug_indexes=(0, 4)
    )
    print(f"Frist loop was finished")
    expentions_for_loops(
         files_path=files_path,
         file_name=file_name,
         barcodes=barcodes,
        image=image,
         steps_directions=steps_directions,
        debug_indexes=(10, 14)
    )
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

