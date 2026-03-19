"""
dataset.py - Dataset loader cho OCR.
Đọc label file, load ảnh, tiền xử lý và trả về (image_tensor, label_indices).
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.vocab import text_to_indices


# ======================== Tiền xử lý ảnh ========================

IMG_HEIGHT = 32
IMG_WIDTH = 128


def preprocess_image(image_path: str) -> torch.Tensor:
    """Pipeline xử lý ảnh:
    image -> grayscale -> resize (32x128) -> normalize -> tensor

    Returns:
        Tensor shape (1, IMG_HEIGHT, IMG_WIDTH) - 1 channel grayscale.
    """
    # Đọc ảnh bằng OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    # Resize giữ tỉ lệ, padding phần còn lại bằng trắng (255)
    h, w = img.shape
    ratio = min(IMG_WIDTH / w, IMG_HEIGHT / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding về đúng kích thước (32 x 128)
    canvas = np.full((IMG_HEIGHT, IMG_WIDTH), 255, dtype=np.uint8)
    canvas[:new_h, :new_w] = img

    # Normalize về [0, 1]
    img_float = canvas.astype(np.float32) / 255.0

    # Chuyển thành tensor (1, H, W)
    tensor = torch.from_numpy(img_float).unsqueeze(0)
    return tensor


# ======================== Dataset Class ========================

class HandwrittenDataset(Dataset):
    """Dataset cho OCR chữ viết tay.
    
    Label file format (mỗi dòng):
        <relative_image_path>\t<text_label>
    
    Ví dụ:
        train_data/1/1.jpg\tKHÁI QUÁT VỀ BIỂN ĐẢO VIỆT NAM...
    """

    def __init__(self, label_file: str, data_root: str, max_label_length: int = 256):
        """
        Args:
            label_file: Đường dẫn tới file train_labels.txt hoặc val_labels.txt.
            data_root:  Thư mục gốc chứa dữ liệu (data/).
            max_label_length: Giới hạn độ dài label (ký tự).
        """
        self.data_root = data_root
        self.max_label_length = max_label_length
        self.samples = []  # list of (image_path, label_text)

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', maxsplit=1)
                if len(parts) != 2:
                    continue
                img_rel_path, label_text = parts
                img_abs_path = os.path.join(data_root, img_rel_path)
                if os.path.exists(img_abs_path):
                    self.samples.append((img_abs_path, label_text))

        print(f"[Dataset] Đã load {len(self.samples)} mẫu từ {label_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_text = self.samples[idx]

        # Tiền xử lý ảnh
        image = preprocess_image(img_path)

        # Chuyển label text -> index sequence
        label_indices = text_to_indices(label_text)
        if len(label_indices) > self.max_label_length:
            label_indices = label_indices[:self.max_label_length]

        return image, torch.tensor(label_indices, dtype=torch.long)


def collate_fn(batch):
    """
    Hàm tùy chỉnh để xử lý batch trong DataLoader.
    - Resize ảnh về cùng kích thước.
    - Nối các label thành một tensor duy nhất.
    """
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]  # Giữ labels là một list các tensor
    label_lengths = [len(label) for label in labels]

    # Xử lý ảnh
    processed_images = torch.stack(images, 0)

    # Xử lý labels và lengths
    all_labels = torch.cat(labels)
    all_label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return processed_images, all_labels, all_label_lengths
