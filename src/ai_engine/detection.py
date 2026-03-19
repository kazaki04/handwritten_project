"""
This file will handle the text detection part using PaddleOCR.
"""
import os
import cv2
from paddleocr import PaddleOCR

class TextDetector:
    def __init__(self, use_angle_cls=True, lang='en'):
        """
        Initializes the TextDetector with PaddleOCR.
        """
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    def detect(self, image_path):
        """
        Detects text in an image and returns bounding boxes.
        :param image_path: Path to the image file.
        :return: A list of bounding boxes for detected text regions.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        result = self.ocr.ocr(image_path, cls=True, det=True, rec=False)
        return result

    def crop_text_regions(self, image_path):
        """
        Detects text and crops the regions from the image.
        :param image_path: Path to the image file.
        :return: A list of cropped image regions (as numpy arrays).
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image from {image_path}")

        detection_results = self.detect(image_path)
        cropped_images = []

        if detection_results:
            for line in detection_results:
                box = line[0]
                # The box is a list of 4 points (x, y)
                # We can use these points to crop the image
                # For simplicity, we'll use the min/max x/y coordinates to create a rectangle
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                cropped = img[y_min:y_max, x_min:x_max]
                cropped_images.append(cropped)
        
        return cropped_images
