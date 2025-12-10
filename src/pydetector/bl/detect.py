import cv2
import numpy as np
import os

class OpenCVDetector:
    """
    מחלקה לזיהוי Bounding Boxes של מדבקות לבנות בתמונה באמצעות OpenCV.
    כולל יכולות Debug משופרות.
    """

    def __init__(self):
        print("[INFO] OpenCV clean detector initialized with DEBUG capabilities.")

    def _print_debug_stats(self, step_name: str, image: np.ndarray):
        """הדפסת נתונים סטטיסטיים בסיסיים על התמונה לאחר שלב עיבוד."""
        if image is None or image.size == 0:
            print(f"  [DEBUG] {step_name}: Image is None or empty.")
            return
        
        # אם התמונה היא בגווני אפור, חשב ממוצע וסטיית תקן
        if len(image.shape) == 2 or image.shape[2] == 1:
            mean_val = np.mean(image)
            std_dev = np.std(image)
            print(f"  [DEBUG] {step_name}: Mean Brightness={mean_val:.2f}, Std Dev={std_dev:.2f}, Max Val={np.max(image)}")
        # אם התמונה צבעונית, הדפס רק את המידות
        else:
             print(f"  [DEBUG] {step_name}: Shape={image.shape}")


    def detect(self, 
               image_path: str,
               save_outputs: bool = False,
               min_contour_area: int = 10000, 
               max_contour_area: int = 400000,
               adaptive_thresh_block_size: int = 11,
               adaptive_thresh_C: int = 2
            ) -> dict:
        
        print(f"\n[START] Processing image: {image_path}")
        print(f"[DEBUG] parameters: min_contour_area={min_contour_area}, max_contour_area={max_contour_area} ")
        print(f" // adaptive_thresh_block_size={adaptive_thresh_block_size}, adaptive_thresh_C={adaptive_thresh_C} ")
        # 1. קריאת התמונה
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[ERROR] Could not load image from path: {image_path}")
            return {"detections": [], "message": "error: failed to load image"}
        
        self._print_debug_stats("Initial Load", image)

        # 2. עיבוד מקדים (גווני אפור)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._print_debug_stats("Grayscale Conversion", gray)
        
        # 3. סף אדפטיבי (Adaptive Thresholding)
        try:
            block_size = adaptive_thresh_block_size if adaptive_thresh_block_size % 2 != 0 else adaptive_thresh_block_size + 1
            
            thresh = cv2.adaptiveThreshold(
                gray, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV,  # אנו מצפים למצוא מדבקות בהירות (לבן=255)
                block_size,
                adaptive_thresh_C
            )
            # אם תצטרך לבדוק מקרה הפוך (THRESH_BINARY_INV), שנה כאן:
            # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, adaptive_thresh_C)
            
            self._print_debug_stats("Adaptive Threshold", thresh)

        except Exception as e:
            print(f"[ERROR] Adaptive Thresholding failed: {e}")
            return {"detections": [], "message": "error: thresholding failed"}
        
        # 4. זיהוי קונטורים
        contours, hierarchy = cv2.findContours(
            thresh, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # --- הוסף את הקטע הזה מיד אחרי ששלב findContours הסתיים ---
        print(f"[DEBUG] Found {len(contours)} initial contours BEFORE filtering. Area breakdown:")
        large_contour_areas = []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1000: # הדפסת שטחים מעל 1000 פיקסלים (כדי לא להציף ברעש קטן)
                large_contour_areas.append(area)

        if large_contour_areas:
            print(f"  [DEBUG] Large Contour Areas Found: {large_contour_areas}")
        else:
            print("  [DEBUG] No contours found with area > 1000. Check Thresholding (Step 3).")
        # -----------------------------------------------------------------
        
        print(f"[DEBUG] Found {len(contours)} initial contours BEFORE filtering.")
        
        detections = []
        output_image = image.copy()
        
        # 5. עיבוד וסינון הקונטורים
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # בדיקת סינון לפי גודל
            if min_contour_area < area < max_contour_area:
                
                # מציאת ה-Bounding Box המקיף
                x, y, w, h = cv2.boundingRect(contour)
                
                # סינון משני: יחס גובה-רוחב (AspectRatio)
                # אם המדבקות הן בדרך כלל מלבניות, נוכל להוסיף סינון זה:
                # aspect_ratio = float(w) / h
                # if 0.5 < aspect_ratio < 3.0: # טווח לדוגמה
                
                detection = {
                    "X1": int(x), "Y1": int(y), "X2": int(x + w), "Y2": int(y + h),
                    "class_name": "label",
                    "confidence": 1.0 
                }
                detections.append(detection)
                
                if save_outputs:
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print(f"[INFO] Detected {len(detections)} final objects AFTER filtering.")

        # 6. שמירת פלט חזותי
        if save_outputs:
            cv2.imwrite("detected_labels_output.jpg", output_image)
            cv2.imwrite("threshold_mask_output.jpg", thresh)
            print("[INFO] Output images saved.")

        return {
            "detections": detections,
            "message": "success"
        }

detector = OpenCVDetector()
        