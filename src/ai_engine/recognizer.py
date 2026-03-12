"""
recognizer.py - Model OCR + Inference cho chữ viết tay tiếng Việt.

Kiến trúc: CNN backbone -> Feature map -> BiLSTM -> Linear -> CTC Loss
"""

import os
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.vocab import NUM_CLASSES, decode_ctc
from src.ai_engine.dataset import preprocess_image


# ======================== CNN + BiLSTM + CTC Model ========================

class HandwrittenOCRModel(nn.Module):
    """Model OCR: CNN backbone → BiLSTM → Linear → CTC.

    Input:  (batch, 1, 32, 128)  - ảnh grayscale
    Output: (seq_len, batch, num_classes) - log probabilities cho CTC
    """

    def __init__(self, num_classes: int = NUM_CLASSES, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # ---- CNN Backbone ----
        # Trích xuất feature map từ ảnh
        self.cnn = nn.Sequential(
            # Block 1: (1, 32, 128) -> (64, 16, 64)
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (64, 16, 64) -> (128, 8, 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (128, 8, 32) -> (256, 4, 32)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # chỉ giảm chiều cao

            # Block 4: (256, 4, 32) -> (512, 2, 32)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # chỉ giảm chiều cao

            # Block 5: (512, 2, 32) -> (512, 1, 32)
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ---- BiLSTM ----
        # Input: (seq_len=32, batch, 512)
        # Output: (seq_len, batch, hidden_size * 2)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=False,
        )

        # ---- Linear projection ----
        # (seq_len, batch, hidden_size*2) -> (seq_len, batch, num_classes)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 32, 128) - ảnh đầu vào
        Returns:
            log_probs: (seq_len, batch, num_classes) - cho CTC loss
        """
        # CNN: trích xuất features
        conv_out = self.cnn(x)  # (batch, 512, 1, W')

        # Squeeze chiều cao, permute thành (W', batch, channels) cho RNN
        conv_out = conv_out.squeeze(2)        # (batch, 512, W')
        conv_out = conv_out.permute(2, 0, 1)  # (W', batch, 512)

        # BiLSTM
        rnn_out, _ = self.rnn(conv_out)  # (seq_len, batch, hidden*2)

        # Linear projection
        output = self.fc(rnn_out)  # (seq_len, batch, num_classes)

        # Log softmax cho CTC loss
        log_probs = torch.nn.functional.log_softmax(output, dim=2)
        return log_probs


# ======================== Inference Engine ========================

class OCREngine:
    """Engine nhận diện chữ viết tay tiếng Việt.
    Load model đã train và thực hiện inference."""

    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = HandwrittenOCRModel().to(self.device)

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[OCREngine] Đã load model từ {model_path}")
        else:
            print("[OCREngine] Chưa có model được train. Sử dụng model với trọng số ngẫu nhiên.")

        self.model.eval()

    def predict(self, image_path: str) -> str:
        """Nhận diện chữ viết tay từ ảnh.

        Args:
            image_path: Đường dẫn tới ảnh cần nhận diện.
        Returns:
            text: Chuỗi văn bản nhận diện được.
        """
        # Tiền xử lý ảnh
        image_tensor = preprocess_image(image_path)                # (1, H, W)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)   # (1, 1, H, W)

        # Forward pass
        with torch.no_grad():
            log_probs = self.model(image_tensor)  # (seq_len, 1, num_classes)

        # Greedy decoding: lấy index có xác suất cao nhất tại mỗi timestep
        preds = log_probs.squeeze(1).argmax(dim=1)  # (seq_len,)
        pred_indices = preds.cpu().tolist()

        # Giải mã CTC
        text = decode_ctc(pred_indices)
        return text