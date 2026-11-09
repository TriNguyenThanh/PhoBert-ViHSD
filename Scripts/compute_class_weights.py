import argparse
import os
import numpy as np
import pandas as pd
import torch


# Hàm tính toán vector trọng số lớp dựa trên 3 phương pháp khác nhau cho bài toán ViHSD
def compute_class_weights(train_csv_path, method="icf", beta=0.999):
    """Tính toán vector trọng số phân bố lớp (Class Weights) cho tập dữ liệu huấn luyện.

    Các phương pháp hỗ trợ:
    1. 'icf' (Inverse Class Frequency):
       w_c = N / (K * N_c)
       Chuẩn Scikit-learn, cân bằng tuyệt đối đóng góp gradient giữa các lớp.

    2. 'smoothed' (Smoothed Inverse Frequency):
       w_c = sqrt(N / N_c), sau đó chuẩn hóa mean(w) = 1.0
       Làm mịn trọng số nhằm hạn chế việc lớp thiểu số nhận trọng số quá lớn gây suy giảm Precision.

    3. 'effective' (Effective Number of Samples - Cui et al., 2019):
       E_c = (1 - beta^N_c) / (1 - beta)
       w_c = 1 / E_c, sau đó chuẩn hóa sum(w) = K
       Dựa trên lý thuyết thông tin biên và thể tích đặc trưng hiệu dụng.

    Tham số:
        train_csv_path: Đường dẫn tệp CSV dữ liệu huấn luyện (chứa cột 'label_id').
        method: Phương pháp tính toán ('icf', 'smoothed', 'effective').
        beta: Siêu tham số cho phương pháp mẫu hiệu dụng (mặc định 0.999).

    Trả về:
        torch.Tensor chứa vector trọng số kích thước (K,) kiểu float32.
    """
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu tại: {train_csv_path}")

    df = pd.read_csv(train_csv_path)
    if "label_id" not in df.columns:
        raise ValueError("Tệp dữ liệu không chứa cột 'label_id'")

    labels = df["label_id"].dropna().astype(int).values
    total_samples = len(labels)
    classes, counts = np.unique(labels, return_counts=True)
    num_classes = len(classes)

    counts_float = counts.astype(np.float64)

    if method == "icf":
        # Phương pháp nghịch đảo tần suất lớp tiêu chuẩn
        weights = total_samples / (num_classes * counts_float)
    elif method == "smoothed":
        # Phương pháp làm mịn căn bậc hai
        raw_weights = np.sqrt(total_samples / counts_float)
        weights = raw_weights / np.mean(raw_weights)
    elif method == "effective":
        # Phương pháp số lượng mẫu hiệu dụng
        effective_num = (1.0 - np.power(beta, counts_float)) / (1.0 - beta)
        raw_weights = 1.0 / effective_num
        weights = (raw_weights / np.sum(raw_weights)) * num_classes
    else:
        raise ValueError(
            f"Phương pháp không hợp lệ: '{method}'. Chỉ chấp nhận 'icf', 'smoothed', 'effective'."
        )

    return torch.tensor(weights, dtype=torch.float32), classes, counts


def main():
    parser = argparse.ArgumentParser(
        description="Tính toán vector trọng số lớp (Class Weights) cho tập dữ liệu ViHSD."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/train.csv",
        help="Đường dẫn đến tệp train.csv (mặc định: data/train.csv)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["icf", "smoothed", "effective", "all"],
        default="all",
        help="Phương pháp tính toán trọng số (mặc định: all)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.999,
        help="Siêu tham số beta cho phương pháp effective (mặc định: 0.999)",
    )

    args = parser.parse_args()
    class_names = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}

    methods_to_run = ["icf", "smoothed", "effective"] if args.method == "all" else [args.method]

    print("=" * 70)
    print("TÍNH TOÁN VECTOR TRỌNG SỐ LỚP CHO TẬP DỮ LIỆU VIETNAMESE HATE SPEECH (ViHSD)")
    print("=" * 70)
    print(f"Tệp dữ liệu: {args.data_path}")

    for m in methods_to_run:
        weights_tensor, classes, counts = compute_class_weights(
            args.data_path, method=m, beta=args.beta
        )
        print(f"\n--- Phương pháp: {m.upper()} ---")
        for cls_id, count, w in zip(classes, counts, weights_tensor.tolist()):
            pct = (count / np.sum(counts)) * 100
            print(
                f"  Lớp {cls_id} ({class_names.get(cls_id, 'UNKNOWN'):9s}): {count:5d} mẫu ({pct:5.2f}%) -> Trọng số: {w:.4f}"
            )
        print(f"  Vector PyTorch: torch.tensor({[round(w, 4) for w in weights_tensor.tolist()]}, dtype=torch.float32)")

    print("=" * 70)


if __name__ == "__main__":
    main()
