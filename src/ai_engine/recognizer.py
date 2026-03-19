"""
recognizer.py
This file is updated to use VietOCR for text recognition.
"""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

IMG_HEIGHT = 32
IMG_WIDTH  = 160

# ──────────────────────────── Preprocessing ────────────────────────────

def preprocess_image(
    image_path: str,
    img_height: int = IMG_HEIGHT,
    img_width:  int = IMG_WIDTH,
) -> Image.Image:
    """Load image and convert to PIL Image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    # VietOCR's predictor expects a PIL Image
    return Image.fromarray(img)


# ──────────────────────────── Inference Engine ────────────────────────────

class OCREngine:
    """Load VietOCR model and run inference."""

    def __init__(
        self,
        model_path: str | None = None, # model_path is handled by VietOCR config
        device: str = "cpu",
        beam_width: int = 10, # beam_width can be set in config
    ):
        # Configuration for VietOCR
        config = Cfg.load_config_from_name('vgg_transformer') # Or other model config
        config['weights'] = model_path or 'https://github.com/pbcquoc/vietocr/releases/download/v0.3.1/vgg_transformer.pth' # Default weights if none provided
        config['device'] = device
        config['predictor']['beamsearch'] = True if beam_width > 1 else False
        config['predictor']['beam_width'] = beam_width

        self.detector = Predictor(config)
        self.model_info = {
            "model_path": config['weights'],
            "loaded":     True,
        }
        print(f"[OCREngine] Loaded VietOCR with weights from: {config['weights']}")


    def predict(self, image, use_beam_search: bool = True) -> str:
        """Predict text from image. Accepts file path or PIL Image."""
        if isinstance(image, str):
            pil_img = preprocess_image(image)
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            raise ValueError("predict expects either a file path string or a PIL Image")
        # VietOCR's predict method takes a PIL image
        text = self.detector.predict(pil_img)
        return text
