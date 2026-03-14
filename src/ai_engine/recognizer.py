"""
recognizer.py
Improved CRNN OCR: ResNet-style CNN + 3-layer BiLSTM + Self-Attention + CTC.
- Beam search decoder for best inference quality.
- Greedy decoder for fast validation during training.
"""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.vocab import NUM_CLASSES, decode_ctc, indices_to_text


IMG_HEIGHT = 32
IMG_WIDTH  = 160   # increased from 128 -> gives T=40 timesteps (more room for CTC)


# ──────────────────────────── Preprocessing ────────────────────────────

def preprocess_image(
    image_path: str,
    img_height: int = IMG_HEIGHT,
    img_width:  int = IMG_WIDTH,
) -> torch.Tensor:
    """grayscale → aspect-ratio-preserving resize+pad → normalize [-1,1] → tensor (1,H,W)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    h, w = img.shape
    scale  = min(img_width / w, img_height / h)
    new_w  = max(1, int(w * scale))
    new_h  = max(1, int(h * scale))
    img    = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas           = np.full((img_height, img_width), 255, dtype=np.uint8)
    canvas[:new_h, :new_w] = img

    arr = canvas.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5               # [-1, 1]
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


# ──────────────────────────── CNN Backbone ────────────────────────────

class ResidualBlock(nn.Module):
    """Basic residual block with optional channel projection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.shortcut(x))


# ──────────────────────────── CRNN Model ────────────────────────────

class CRNNModel(nn.Module):
    """
    ResNet CNN  +  3-layer BiLSTM(512)  +  Multi-head Self-Attention  +  CTC

    Input  : (B, 1, 32, 160)
    Output : (T=40, B, num_classes)  – log-softmax probabilities for CTC Loss
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden_size: int = 512,
        rnn_layers:  int = 3,
        dropout:     float = 0.5,
    ):
        super().__init__()

        # ── ResNet-style CNN ──────────────────────────────────────────
        # (B,1,32,160)
        self.cnn = nn.Sequential(
            # Stem
            nn.Conv2d(1, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 1  → (B, 64, 16, 80)
            ResidualBlock(64, 64),
            nn.MaxPool2d(2, 2),

            # Block 2  → (B, 128, 8, 40)
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2),

            # Block 3  → (B, 256, 4, 40)
            ResidualBlock(128, 256),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Block 4  → (B, 512, 2, 40)
            ResidualBlock(256, 512),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Squeeze height to 1  → (B, 512, 1, 40)
            nn.Conv2d(512, 512, (2, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── 3-layer BiLSTM ────────────────────────────────────────────
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=False,
        )

        # ── Multi-head Self-Attention ─────────────────────────────────
        rnn_out_size = hidden_size * 2          # 1024
        self.attn = nn.MultiheadAttention(
            embed_dim=rnn_out_size,
            num_heads=8,
            dropout=0.1,
            batch_first=False,
        )
        self.attn_norm = nn.LayerNorm(rnn_out_size)

        # ── Classifier ───────────────────────────────────────────────
        self.drop = nn.Dropout(p=dropout)
        self.fc   = nn.Linear(rnn_out_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        feat = self.cnn(x)               # (B, 512, 1, T)
        feat = feat.squeeze(2)           # (B, 512, T)
        feat = feat.permute(2, 0, 1)     # (T, B, 512)

        # BiLSTM
        seq, _ = self.rnn(feat)          # (T, B, 1024)

        # Self-attention with residual
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.attn_norm(seq + attn_out)   # (T, B, 1024)

        # Classifier
        logits = self.fc(self.drop(seq))       # (T, B, C)
        return torch.log_softmax(logits, dim=2)


# ──────────────────────────── Decoders ────────────────────────────

def greedy_ctc_decode(log_probs: torch.Tensor) -> List[str]:
    """Fast greedy CTC decoder.  log_probs: (T, N, C)."""
    pred_ids = log_probs.argmax(dim=2).permute(1, 0).cpu().tolist()
    return [decode_ctc(ids) for ids in pred_ids]


def beam_search_ctc_decode(
    log_probs: torch.Tensor,
    beam_width: int = 10,
) -> List[str]:
    """
    Standard CTC beam search decoder.

    Each beam stores (p_blank, p_no_blank):
      p_blank   – cumulative prob that the prefix ends with a blank at step t
      p_no_blank – cumulative prob that it ends with a real character

    Args:
        log_probs  : (T, N, C)
        beam_width : number of beams to keep
    Returns:
        List of decoded strings, one per sample.
    """
    T, N, C = log_probs.shape
    probs = torch.exp(log_probs).cpu().numpy()   # (T, N, C)

    results: List[str] = []
    for b in range(N):
        p = probs[:, b, :]   # (T, C)

        # beams: prefix_tuple → [p_blank, p_no_blank]
        beams: dict = {(): [1.0, 0.0]}

        for t in range(T):
            new_beams: dict = {}

            for prefix, (p_b, p_nb) in beams.items():
                total = p_b + p_nb

                # ── extend with BLANK (index 0) ──
                nb = total * float(p[t, 0])
                entry = new_beams.setdefault(prefix, [0.0, 0.0])
                entry[0] += nb

                # ── extend with each non-blank character ──
                for c in range(1, C):
                    cp = float(p[t, c])
                    if cp < 1e-9:
                        continue

                    new_prefix = prefix + (c,)

                    # repeated last char: only a preceding blank can extend
                    if prefix and prefix[-1] == c:
                        nb_c = p_b * cp
                    else:
                        nb_c = total * cp

                    entry = new_beams.setdefault(new_prefix, [0.0, 0.0])
                    entry[1] += nb_c

            # Prune to top beam_width
            beams = dict(
                sorted(
                    new_beams.items(),
                    key=lambda kv: kv[1][0] + kv[1][1],
                    reverse=True,
                )[:beam_width]
            )

        best_prefix = max(beams, key=lambda k: beams[k][0] + beams[k][1])
        results.append(indices_to_text(list(best_prefix)))

    return results


# ──────────────────────────── Inference Engine ────────────────────────────

class OCREngine:
    """Load trained CRNN model and run inference (greedy or beam search)."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        beam_width: int = 10,
    ):
        self.device     = torch.device(device)
        self.beam_width = beam_width
        self.model      = CRNNModel(num_classes=NUM_CLASSES).to(self.device)
        self.model_info = {
            "model_path": model_path,
            "epoch":      None,
            "val_cer":    None,
            "val_wer":    None,
            "loaded":     False,
        }

        if model_path and os.path.exists(model_path):
            ckpt       = torch.load(model_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            self.model_info["loaded"]     = True
            self.model_info["model_path"] = model_path
            if isinstance(ckpt, dict):
                self.model_info["epoch"]   = ckpt.get("epoch")
                self.model_info["val_cer"] = ckpt.get("val_cer")
                self.model_info["val_wer"] = ckpt.get("val_wer")
            cer = self.model_info["val_cer"]
            cer_str = f"{cer:.4f}" if cer is not None else "N/A"
            print(
                f"[OCREngine] Loaded: {os.path.basename(model_path)} | "
                f"epoch={self.model_info['epoch']} | CER={cer_str}"
            )
        else:
            print("[OCREngine] Warning: weights not found – using random initialisation.")

        self.model.eval()

    def predict(self, image_path: str, use_beam_search: bool = True) -> str:
        """Predict text from image. Uses beam search by default for best accuracy."""
        img = preprocess_image(image_path).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_probs = self.model(img)   # (T, 1, C)

        if use_beam_search and self.beam_width > 1:
            return beam_search_ctc_decode(log_probs, beam_width=self.beam_width)[0]
        return greedy_ctc_decode(log_probs)[0]
