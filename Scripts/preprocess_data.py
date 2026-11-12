import argparse
import os
import sys
import pandas as pd

# Thêm thư mục gốc vào sys.path để import mô-đun Pipeline khi chạy độc lập
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Pipeline.preprocessing import ViHSDPreprocessor


# Phân tích và thiết lập các đối số dòng lệnh phục vụ tiền xử lý tập dữ liệu
def parse_args():
    parser = argparse.ArgumentParser(
        description="Kịch bản tiền xử lý và chuẩn hóa tập dữ liệu ViHSD theo Checkpoint 1."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Thư mục chứa các tập tin train.csv, dev.csv, test.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Thư mục xuất các tập tin dữ liệu đã xử lý",
    )
    parser.add_argument(
        "--teencode_path",
        type=str,
        default="data/teencode.csv",
        help="Đường dẫn đến tập tin từ điển teencode chuẩn hóa",
    )
    parser.add_argument(
        "--emoji_path",
        type=str,
        default="data/emojis.csv",
        help="Đường dẫn đến tập tin từ điển emoji chuẩn hóa",
    )
    parser.add_argument(
        "--lowercased",
        action="store_true",
        default=False,
        help="Chuyển toàn bộ văn bản về chữ thường (mặc định giữ nguyên chữ hoa để phục vụ PhoBERT)",
    )
    parser.add_argument(
        "--clean_leakage",
        action="store_true",
        default=True,
        help="Khử trùng lặp nội bộ và triệt tiêu rò rỉ dữ liệu (Data Leakage) từ Train sang Dev/Test",
    )
    return parser.parse_args()


# Đọc tập tin dữ liệu thô, thực thi chuẩn hóa văn bản và lưu trữ tệp kết quả sau xử lý
def process_and_save(preprocessor, input_path, output_path, lowercased=False):
    print(f"Đọc tập tin: {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)

    texts, labels = preprocessor.transform_dataset(
        df, text_col="free_text", label_col="label_id", lowercased=lowercased
    )

    df_processed = pd.DataFrame({"free_text": texts, "label_id": labels})
    processed_count = len(df_processed)

    df_processed.to_csv(output_path, index=False, encoding="utf-8")
    print(
        f"Đã xử lý và lưu vào: {output_path} (Ban đầu: {initial_count}, Hợp lệ: {processed_count}, Loại bỏ: {initial_count - processed_count})"
    )
    return df_processed


# Hàm điều phối chính thực thi tiền xử lý tuần tự trên cả ba phân vùng train, dev, test
def main():
    args = parse_args()

    preprocessor = ViHSDPreprocessor(
        teencode_path=args.teencode_path, emoji_path=args.emoji_path
    )

    os.makedirs(args.output_dir, exist_ok=True)

    train_file = os.path.join(args.data_dir, "train.csv")
    dev_file = os.path.join(args.data_dir, "dev.csv")
    test_file = os.path.join(args.data_dir, "test.csv")

    dev_df = pd.read_csv(dev_file) if os.path.exists(dev_file) else None
    test_df = pd.read_csv(test_file) if os.path.exists(test_file) else None

    splits = ["train", "dev", "test"]
    for split in splits:
        input_file = os.path.join(args.data_dir, f"{split}.csv")
        output_file = os.path.join(args.output_dir, f"{split}_processed.csv")

        if not os.path.exists(input_file):
            print(f"Cảnh báo: Không tìm thấy tập tin {input_file}")
            continue

        if split == "train" and args.clean_leakage and (dev_df is not None or test_df is not None):
            print("Áp dụng khử trùng lặp và triệt tiêu rò rỉ dữ liệu (Data Leakage) cho tập Train...")
            raw_train = pd.read_csv(input_file)
            cleaned_train = preprocessor.clean_and_deduplicate(
                raw_train, dev_df=dev_df, test_df=test_df, verbose=True
            )
            initial_count = len(raw_train)
            texts, labels = preprocessor.transform_dataset(
                cleaned_train, text_col="free_text", label_col="label_id", lowercased=args.lowercased
            )
            df_processed = pd.DataFrame({"free_text": texts, "label_id": labels})
            df_processed.to_csv(output_file, index=False, encoding="utf-8")
            print(
                f"Đã xử lý và lưu vào: {output_file} (Ban đầu: {initial_count}, Sạch: {len(cleaned_train)}, Sau biến đổi: {len(df_processed)})"
            )
        else:
            process_and_save(
                preprocessor, input_file, output_file, lowercased=args.lowercased
            )


if __name__ == "__main__":
    main()
