"""
train.py
Improved training pipeline for Handwritten OCR – CRNN + CTC.
- ResNet CNN + 3-layer BiLSTM(512) + Self-Attention
- AdamW + CosineAnnealingLR (with linear warmup)
- Advanced augmentation: elastic distortion, perspective, blur, rotation, affine, brightness, noise
- Width=160 → T=40 timesteps → max_target_length=38
- epochs=200, batch=32, lr=3e-4
- Greedy decode during training; beam search in OCREngine.predict()
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ai_engine.recognizer import CRNNModel, greedy_ctc_decode
from src.utils.metrics import compute_cer, compute_wer
from src.utils.vocab import NUM_CLASSES, indices_to_text, text_to_indices


# ─────────────────────────── Config ───────────────────────────

@dataclass
class TrainConfig:
    project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    img_height: int   = 32
    img_width:  int   = 160   # T=40 timesteps → allows up to 38 chars per sample

    epochs:         int   = 200
    batch_size:     int   = 32
    learning_rate:  float = 3e-4
    weight_decay:   float = 1e-5
    warmup_epochs:  int   = 10
    num_workers:    int   = 0

    hidden_size: int   = 512
    rnn_layers:  int   = 3
    dropout:     float = 0.5

    clip_grad_norm: float = 5.0

    max_target_length: int = 38   # safe margin below T=40
    min_target_length: int = 1

    print_shape_every: int = 100  # print CTC shapes every N batches

    @property
    def data_dir(self) -> str:
        return os.path.join(self.project_root, "data")

    @property
    def train_labels(self) -> str:
        return os.path.join(self.data_dir, "train_labels.txt")

    @property
    def val_labels(self) -> str:
        return os.path.join(self.data_dir, "val_labels.txt")

    @property
    def model_path(self) -> str:
        return os.path.join(self.project_root, "models", "ocr_model.pth")

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────── Augmentation helpers ───────────────────────────

def elastic_distortion(img: np.ndarray, alpha: float = 30.0, sigma: float = 4.0) -> np.ndarray:
    """Elastic deformation: smooth random displacement field applied to image."""
    h, w = img.shape
    dx = np.random.uniform(-1, 1, (h, w)).astype(np.float32) * alpha
    dy = np.random.uniform(-1, 1, (h, w)).astype(np.float32) * alpha
    k = int(sigma * 3) | 1          # ensure odd kernel
    dx = cv2.GaussianBlur(dx, (k, k), sigma)
    dy = cv2.GaussianBlur(dy, (k, k), sigma)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(xs + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(ys + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def random_perspective(img: np.ndarray, distort: float = 0.05) -> np.ndarray:
    """4-corner random perspective warp."""
    h, w = img.shape
    d = distort
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    rnd = lambda lim: np.random.uniform(-lim, lim)
    dst = np.float32([
        [rnd(d * w), rnd(d * h)],
        [w - 1 + rnd(d * w), rnd(d * h)],
        [w - 1 + rnd(d * w), h - 1 + rnd(d * h)],
        [rnd(d * w), h - 1 + rnd(d * h)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def augment_image(img: np.ndarray) -> np.ndarray:
    """Full augmentation pipeline for handwritten OCR images (applied on raw grayscale)."""
    # Random rotation ±6°
    if random.random() < 0.6:
        angle = random.uniform(-6.0, 6.0)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    # Elastic distortion (simulates handwriting irregularity)
    if random.random() < 0.5:
        img = elastic_distortion(img, alpha=random.uniform(15, 40), sigma=random.uniform(3, 5))

    # Random perspective warp
    if random.random() < 0.4:
        img = random_perspective(img, distort=random.uniform(0.02, 0.06))

    # Affine shear
    if random.random() < 0.5:
        h, w = img.shape
        shear_x = random.uniform(-0.05 * w, 0.05 * w)
        shear_y = random.uniform(-0.1 * h, 0.1 * h)
        src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
        dst_pts = np.float32([
            [shear_x, shear_y],
            [w - 1 + shear_x, -shear_y],
            [-shear_x, h - 1 + shear_y],
        ])
        M = cv2.getAffineTransform(src_pts, dst_pts)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    # Gaussian blur
    if random.random() < 0.4:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Brightness / contrast jitter
    if random.random() < 0.7:
        alpha = random.uniform(0.75, 1.25)
        beta  = random.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(3, 12), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


# ─────────────────────────── Dataset ───────────────────────────

def _parse_label_line(line: str) -> Tuple[str, str] | None:
    """Parse 'path|text' or 'path\\ttext'."""
    line = line.strip()
    if not line:
        return None
    sep = "|" if "|" in line else "\t" if "\t" in line else None
    if sep is None:
        return None
    parts = line.split(sep, 1)
    if len(parts) != 2:
        return None
    p, t = parts[0].strip(), parts[1].strip()
    return (p, t) if p and t else None


class OCRDataset(Dataset):
    """
    Reads label file, loads images with augmentation+preprocessing,
    returns (image_tensor, label_indices, raw_text).

    Labels longer than max_target_length are truncated (NOT dropped) so
    every image participates in training.
    """

    def __init__(
        self,
        label_file: str,
        data_root: str,
        img_height: int,
        img_width: int,
        max_target_length: int,
        min_target_length: int,
        augment: bool = False,
    ):
        self.data_root         = data_root
        self.img_height        = img_height
        self.img_width         = img_width
        self.max_target_length = max_target_length
        self.min_target_length = min_target_length
        self.augment           = augment

        self.samples: List[Tuple[str, List[int], str]] = []
        n_total = n_truncated = n_short = n_oov = 0

        with open(label_file, "r", encoding="utf-8") as f:
            for raw in f:
                n_total += 1
                parsed = _parse_label_line(raw)
                if not parsed:
                    continue
                rel_path, text = parsed
                abs_path = os.path.join(data_root, rel_path)
                if not os.path.exists(abs_path):
                    continue

                ids = text_to_indices(text)
                if not ids:
                    n_oov += 1
                    continue
                if len(ids) < min_target_length:
                    n_short += 1
                    continue
                if len(ids) > max_target_length:
                    ids  = ids[:max_target_length]
                    text = indices_to_text(ids)
                    n_truncated += 1

                self.samples.append((abs_path, ids, text))

        print(f"[Dataset] {os.path.basename(label_file)}: "
              f"{len(self.samples)} samples  "
              f"(truncated={n_truncated}, skipped short={n_short}, OOV={n_oov})")

        if not self.samples:
            raise RuntimeError(
                "No usable samples. Check label format (path|text or path\\ttext) "
                "and ensure images exist."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        abs_path, ids, text = self.samples[idx]

        img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {abs_path}")

        # Augment on raw image before resize
        if self.augment:
            img = augment_image(img)

        # Aspect-ratio-preserving resize + white padding
        h, w    = img.shape
        scale   = min(self.img_width / w, self.img_height / h)
        new_w   = max(1, int(w * scale))
        new_h   = max(1, int(h * scale))
        img     = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas  = np.full((self.img_height, self.img_width), 255, dtype=np.uint8)
        canvas[:new_h, :new_w] = img

        # Normalize [-1, 1]
        arr = canvas.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).unsqueeze(0)   # (1, H, W)

        return tensor, torch.tensor(ids, dtype=torch.long), text


def collate_fn(batch):
    images      = torch.stack([x[0] for x in batch], dim=0)
    labels_list = [x[1] for x in batch]
    raw_texts   = [x[2] for x in batch]
    target_lengths = torch.tensor([len(t) for t in labels_list], dtype=torch.long)
    targets        = torch.cat(labels_list, dim=0)
    return images, targets, target_lengths, raw_texts


def _split_targets(targets: torch.Tensor, lengths: torch.Tensor) -> List[List[int]]:
    out, offset = [], 0
    for l in lengths.tolist():
        out.append(targets[offset: offset + l].tolist())
        offset += l
    return out


# ─────────────────────────── Scheduler ───────────────────────────

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup then CosineAnnealingLR."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        return 1.0

    warmup  = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=1e-6,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# ─────────────────────────── Validation ───────────────────────────

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> Tuple[float, float, float, str, str]:
    model.eval()
    total_loss = total_cer = total_wer = 0.0
    n_samples = n_batches = 0
    sample_pred = sample_gt = ""

    with torch.no_grad():
        for images, targets, target_lengths, raw_texts in loader:
            images  = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            log_probs = model(images)               # (T, N, C)
            T, N, C   = log_probs.shape
            input_lengths = torch.full((N,), T, dtype=torch.long, device=device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += float(loss.item())
            n_batches  += 1

            preds = greedy_ctc_decode(log_probs)
            for pred, gt in zip(preds, raw_texts):
                total_cer += compute_cer(pred, gt)
                total_wer += compute_wer(pred, gt)
                n_samples += 1

            if not sample_pred:
                sample_pred = preds[0]
                sample_gt   = raw_texts[0]

    return (
        total_loss / max(n_batches, 1),
        total_cer  / max(n_samples, 1),
        total_wer  / max(n_samples, 1),
        sample_pred,
        sample_gt,
    )


# ─────────────────────────── Training loop ───────────────────────────

def train() -> None:
    cfg    = TrainConfig()
    device = torch.device(cfg.device)

    print("=" * 70)
    print("  Handwritten OCR Training  |  ResNet CNN + BiLSTM + Attention + CTC")
    print("=" * 70)
    print(f"  Device        : {cfg.device}")
    print(f"  Image size    : {cfg.img_height} × {cfg.img_width}")
    print(f"  Epochs        : {cfg.epochs}  (warmup {cfg.warmup_epochs})")
    print(f"  Batch size    : {cfg.batch_size}")
    print(f"  LR            : {cfg.learning_rate}  wd={cfg.weight_decay}")
    print(f"  BiLSTM        : {cfg.rnn_layers}-layer hidden={cfg.hidden_size}")
    print(f"  Max target    : {cfg.max_target_length} chars")
    print(f"  Num classes   : {NUM_CLASSES}")
    print("=" * 70)

    # ── Datasets ──
    train_ds = OCRDataset(
        cfg.train_labels, cfg.data_dir,
        cfg.img_height, cfg.img_width,
        cfg.max_target_length, cfg.min_target_length,
        augment=True,
    )
    val_ds = OCRDataset(
        cfg.val_labels, cfg.data_dir,
        cfg.img_height, cfg.img_width,
        cfg.max_target_length, cfg.min_target_length,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=(cfg.device == "cuda"), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=(cfg.device == "cuda"), drop_last=False,
    )

    # ── Model ──
    model  = CRNNModel(NUM_CLASSES, cfg.hidden_size, cfg.rnn_layers, cfg.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable params: {n_params:,}")

    # ── Loss / Optimiser / Scheduler ──
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs)

    os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
    best_cer = math.inf

    print("\nStarting training…\n")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_loss  = 0.0
        n_batches = 0
        t0        = time.time()

        for step, (images, targets, target_lengths, raw_texts) in enumerate(train_loader, 1):
            images  = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            log_probs = model(images)              # (T, N, C)
            T, N, C   = log_probs.shape
            input_lengths = torch.full((N,), T, dtype=torch.long, device=device)

            # Debug: print shapes periodically
            if step % cfg.print_shape_every == 1:
                print(
                    f"  [debug e{epoch} s{step}] "
                    f"log_probs={tuple(log_probs.shape)}  "
                    f"input_len={T}  "
                    f"target_len min/max={int(target_lengths.min())}/{int(target_lengths.max())}"
                )

            # Safety: remove samples whose target > T (should be rare after truncation)
            bad = target_lengths > T
            if bad.any():
                keep = (~bad).nonzero(as_tuple=True)[0]
                if keep.numel() == 0:
                    continue
                images = images[keep]
                seqs   = _split_targets(targets, target_lengths)
                seqs   = [seqs[i] for i in keep.tolist()]
                target_lengths = torch.tensor([len(s) for s in seqs],
                                              dtype=torch.long, device=device)
                targets   = torch.tensor([c for s in seqs for c in s],
                                         dtype=torch.long, device=device)
                log_probs = model(images)
                T, N, C   = log_probs.shape
                input_lengths = torch.full((N,), T, dtype=torch.long, device=device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()

            run_loss  += float(loss.item())
            n_batches += 1

        scheduler.step()

        avg_loss = run_loss / max(n_batches, 1)
        val_loss, val_cer, val_wer, spred, sgt = validate(
            model, val_loader, criterion, device
        )
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"Train={avg_loss:.4f}  Val={val_loss:.4f}  "
            f"CER={val_cer:.4f}  WER={val_wer:.4f}  "
            f"LR={current_lr:.2e}  t={elapsed:.0f}s"
        )
        print(f"  GT  : {sgt[:80]}")
        print(f"  Pred: {spred[:80]}")

        # Save best checkpoint (by CER)
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":            val_loss,
                    "val_cer":             val_cer,
                    "val_wer":             val_wer,
                },
                cfg.model_path,
            )
            print(f"  ✓ Saved best model  CER={val_cer:.4f}  → {cfg.model_path}")

    print(f"\nTraining complete.  Best Val CER = {best_cer:.4f}")


if __name__ == "__main__":
    train()
