"""
vocab.py - Bộ ký tự tiếng Việt cho OCR
Chuyển đổi text <-> index sequence phục vụ CTC Loss.
"""


# Bảng ký tự tiếng Việt đầy đủ (chữ thường + chữ hoa + số + dấu câu + khoảng trắng)
_LOWERCASE = list("abcdefghijklmnopqrstuvwxyz")
_UPPERCASE = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Nguyên âm có dấu - chữ thường
_VIET_LOWER = list(
    "àáảãạăắằẳẵặâấầẩẫậ"
    "èéẻẽẹêếềểễệ"
    "ìíỉĩị"
    "òóỏõọôốồổỗộơớờởỡợ"
    "ùúủũụưứừửữự"
    "ỳýỷỹỵ"
    "đ"
)

# Nguyên âm có dấu - chữ hoa
_VIET_UPPER = list(
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ"
    "ÈÉẺẼẸÊẾỀỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
    "ÙÚỦŨỤƯỨỪỬỮỰ"
    "ỲÝỶỸỴ"
    "Đ"
)

_DIGITS = list("0123456789")
_PUNCTUATION = list(".,;:!?'\"-()[]{}/@#$%^&*+=<>~`\\_|")

# Ký tự đặc biệt
BLANK_TOKEN = "<blank>"   # CTC blank (index 0)
SPACE_TOKEN = " "

# Tổng hợp bảng ký tự
CHARACTERS = (
    [BLANK_TOKEN]            # index 0 dành cho CTC blank
    + [SPACE_TOKEN]          # index 1 dành cho khoảng trắng
    + _LOWERCASE
    + _UPPERCASE
    + _VIET_LOWER
    + _VIET_UPPER
    + _DIGITS
    + _PUNCTUATION
)

# Mapping hai chiều
CHAR_TO_IDX = {ch: idx for idx, ch in enumerate(CHARACTERS)}
IDX_TO_CHAR = {idx: ch for idx, ch in enumerate(CHARACTERS)}

NUM_CLASSES = len(CHARACTERS)   # Tổng số lớp (bao gồm blank)


def text_to_indices(text: str) -> list[int]:
    """Chuyển chuỗi text thành danh sách index.
    Ký tự không có trong bảng sẽ bị bỏ qua (skip)."""
    return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]


def indices_to_text(indices: list[int]) -> str:
    """Chuyển danh sách index thành chuỗi text.
    Bỏ qua blank token (index 0)."""
    return "".join(IDX_TO_CHAR[idx] for idx in indices if idx != 0)


def decode_ctc(raw_indices: list[int]) -> str:
    """Giải mã output CTC (loại bỏ blank và ký tự lặp liền kề)."""
    result = []
    prev = -1
    for idx in raw_indices:
        if idx != 0 and idx != prev:  # bỏ blank và ký tự lặp
            result.append(idx)
        prev = idx
    return indices_to_text(result)
