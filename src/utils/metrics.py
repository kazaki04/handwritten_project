"""
metrics.py - Tính toán Character Error Rate (CER) và Word Error Rate (WER).
"""


def _edit_distance(s1: str, s2: str) -> int:
    """Tính khoảng cách Levenshtein giữa hai chuỗi."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(prediction: str, ground_truth: str) -> float:
    """Character Error Rate = edit_distance(pred, gt) / len(gt)."""
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    return _edit_distance(prediction, ground_truth) / len(ground_truth)


def compute_wer(prediction: str, ground_truth: str) -> float:
    """Word Error Rate = edit_distance trên mức từ / số từ gt."""
    pred_words = prediction.split()
    gt_words = ground_truth.split()
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return _edit_distance_list(pred_words, gt_words) / len(gt_words)


def _edit_distance_list(s1: list, s2: list) -> int:
    """Levenshtein distance cho danh sách phần tử (dùng cho WER)."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]
