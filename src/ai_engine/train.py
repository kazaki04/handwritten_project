"""
train.py - Script huấn luyện model OCR chữ viết tay tiếng Việt.

Pipeline: load dataset -> dataloader -> training loop -> validation -> save model.
Kiến trúc model: CNN + BiLSTM + CTC Loss.
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Đảm bảo import đúng từ project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ai_engine.dataset import HandwrittenDataset, collate_fn
from src.ai_engine.recognizer import HandwrittenOCRModel
from src.utils.vocab import NUM_CLASSES, decode_ctc, text_to_indices
from src.utils.metrics import compute_cer, compute_wer


# ======================== Cấu hình ========================

class TrainConfig:
    """Tham số huấn luyện."""
    # Đường dẫn
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    TRAIN_LABEL = os.path.join(DATA_DIR, 'train_labels.txt')
    VAL_LABEL = os.path.join(DATA_DIR, 'val_labels.txt')
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'ocr_model.pth')

    # Siêu tham số
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_WORKERS = 0          # Windows thường cần 0
    MAX_LABEL_LENGTH = 512

    # Thiết bị
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ======================== Hàm validation ========================

def validate(model, val_loader, criterion, device):
    """Chạy validation, tính loss trung bình và CER/WER.

    Returns:
        avg_loss, avg_cer, avg_wer
    """
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    num_samples = 0
    num_batches = 0

    with torch.no_grad():
        for images, labels, label_lengths in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            log_probs = model(images)  # (seq_len, batch, num_classes)
            seq_len = log_probs.size(0)
            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

            # CTC Loss
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            total_loss += loss.item()
            num_batches += 1

            # Decode và tính CER/WER
            preds = log_probs.argmax(dim=2)  # (seq_len, batch)
            preds = preds.permute(1, 0)       # (batch, seq_len)

            # Tách label cho từng sample
            offset = 0
            for i in range(batch_size):
                length = label_lengths[i].item()
                gt_indices = labels[offset:offset + length].cpu().tolist()
                offset += length

                pred_indices = preds[i].cpu().tolist()
                pred_text = decode_ctc(pred_indices)
                gt_text = decode_ctc(gt_indices)

                total_cer += compute_cer(pred_text, gt_text)
                total_wer += compute_wer(pred_text, gt_text)
                num_samples += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_cer = total_cer / max(num_samples, 1)
    avg_wer = total_wer / max(num_samples, 1)

    return avg_loss, avg_cer, avg_wer


# ======================== Training loop ========================

def train():
    """Hàm huấn luyện chính."""
    cfg = TrainConfig()

    print("=" * 60)
    print("  HUẤN LUYỆN MODEL OCR CHỮ VIẾT TAY TIẾNG VIỆT")
    print("  Kiến trúc: CNN + BiLSTM + CTC")
    print("=" * 60)
    print(f"  Device       : {cfg.DEVICE}")
    print(f"  Epochs       : {cfg.EPOCHS}")
    print(f"  Batch size   : {cfg.BATCH_SIZE}")
    print(f"  Learning rate: {cfg.LEARNING_RATE}")
    print(f"  Num classes  : {NUM_CLASSES}")
    print("=" * 60)

    device = torch.device(cfg.DEVICE)

    # ---- Load dataset ----
    print("\n[1/4] Đang load dataset...")
    train_dataset = HandwrittenDataset(
        label_file=cfg.TRAIN_LABEL,
        data_root=cfg.DATA_DIR,
        max_label_length=cfg.MAX_LABEL_LENGTH,
    )
    val_dataset = HandwrittenDataset(
        label_file=cfg.VAL_LABEL,
        data_root=cfg.DATA_DIR,
        max_label_length=cfg.MAX_LABEL_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ---- Khởi tạo model ----
    print("\n[2/4] Đang khởi tạo model...")
    model = HandwrittenOCRModel(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Tổng tham số: {total_params:,}")

    # ---- Loss & Optimizer ----
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # ---- Training ----
    print("\n[3/4] Bắt đầu huấn luyện...\n")
    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(cfg.MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, labels, label_lengths) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            log_probs = model(images)  # (seq_len, batch, num_classes)
            seq_len = log_probs.size(0)
            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

            # Bỏ qua batch nếu label dài hơn seq_len
            if label_lengths.max().item() > seq_len:
                continue

            # CTC Loss
            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - start_time

        # Validation
        val_loss, val_cer, val_wer = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # In kết quả
        print(
            f"Epoch {epoch:3d}/{cfg.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"CER: {val_cer:.4f} | "
            f"WER: {val_wer:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Lưu model tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_cer': val_cer,
                'val_wer': val_wer,
            }, cfg.MODEL_SAVE_PATH)
            print(f"  -> Đã lưu model tốt nhất tại: {cfg.MODEL_SAVE_PATH}")

    # ---- Kết thúc ----
    print("\n[4/4] Huấn luyện hoàn tất!")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Model lưu tại: {cfg.MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train()