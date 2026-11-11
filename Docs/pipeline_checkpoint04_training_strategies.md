# Đặc Tả Pipeline Checkpoint 04: Tối Ưu Hóa Chiến Lược Huấn Luyện (Advanced Training Strategies)

Tài liệu này đặc tả chi tiết kiến trúc tối ưu hóa phân tầng tốc độ học, mở rộng chiều dài chuỗi ngữ cảnh, cơ chế lập lịch Cosine Annealing with Warmup và kỹ thuật dừng sớm cho Checkpoint 04 trong bài toán phân loại phát ngôn thù ghét (ViHSD).

---

## 1. Mục Tiêu Kỹ Thuật

- Khắc phục các rào cản trong cấu hình huấn luyện của Baseline gốc:
  - Giới hạn chuỗi 100 tokens gây cắt cụt ngữ cảnh bình luận dài.
  - Số lượng epoch quá ngắn (2 epochs) khiến mô hình chưa hội tụ tối đa.
  - Tốc độ học đồng nhất giữa các tầng RoBERTa tiền huấn luyện và Classification Head mới ngẫu nhiên.
- Triển khai các giải pháp huấn luyện nâng cao:
  - Mở rộng chiều dài chuỗi lên `max_length = 256` (ngưỡng tối đa của PhoBERT).
  - Tăng số epoch huấn luyện lên 8 epochs kết hợp cơ chế dừng sớm (Early Stopping với patience = 2) theo dõi trực tiếp chỉ số `eval_macro_f1`.
  - Phân tầng tốc độ học (Differential Learning Rates): giữ tốc độ học nhỏ ($2 \times 10^{-5}$) cho backbone RoBERTa và tốc độ học lớn hơn ($1 \times 10^{-4}$) cho Classification Head.
  - Lịch trình Cosine Annealing with Warmup (10% warmup steps) giúp ổn định gradient trong các bước đầu.
  - Tích lũy gradient (batch size 16, gradient_accumulation_steps = 2 $\to$ effective batch size = 32).
  - Tự động nạp lại trọng số tốt nhất (`load_best_model_at_end = True`).
- Xuất đầy đủ 4 tệp Mandatory Artifacts cùng biểu đồ Learning Curves vào `outputs/checkpoint04/`.

---

## 2. Kiến Trúc Luồng Huấn Luyện (Training Pipeline Architecture)

```
[Dữ liệu thô: train.csv, dev.csv, test.csv]
                     |
                     v
  Mã hóa PhoBERT Tokenizer với max_length = 256
                     |
                     v
  Đóng gói BuildDataset (train_dataset, dev_dataset, test_dataset)
                     |
                     v
  Cấu hình DifferentialTrainer
  - Nhóm 1: model.roberta (lr = 2e-5, weight_decay = 0.01 / 0.0)
  - Nhóm 2: model.classifier (lr = 1e-4, weight_decay = 0.01 / 0.0)
  - Lịch trình: get_cosine_schedule_with_warmup (warmup = 10%)
                     |
                     v
  Huấn luyện tối đa 8 Epochs + EarlyStoppingCallback(patience = 2)
                     |
                     v
  Phục hồi trọng số tối ưu nhất theo eval_macro_f1
                     |
                     v
  Đánh giá trên tập test & Xuất Learning Curves + Artifacts vào outputs/checkpoint04/
```

---

## 3. Cấu Hình Siêu Tham Số

| Tham số | Giá trị | Giải thích |
| :--- | :---: | :--- |
| **Mô hình nền tảng** | `vinai/phobert-base` | Kiến trúc Transformer tiếng Việt tiêu chuẩn. |
| **Độ dài chuỗi (max_length)** | **256** | Bao phủ trọn vẹn bình luận dài. |
| **Số epochs tối đa** | **8** | Cho phép mô hình hội tụ sâu. |
| **Early Stopping patience** | **2 epochs** | Ngăn ngừa quá khớp trên tập dev. |
| **Thước đo chọn mô hình** | `eval_macro_f1` | Căn chỉnh trực tiếp theo Macro-F1. |
| **Tốc độ học backbone** | $2 \times 10^{-5}$ | Bảo toàn tri thức tiền huấn luyện. |
| **Tốc độ học head** | $1 \times 10^{-4}$ | Học nhanh đặc trưng phân loại mới. |
| **Lịch trình LR** | Cosine with Warmup | Suy giảm mượt theo hàm cosine. |
| **Batch size hiệu dụng** | **32** | $16 \times 2$ qua tích lũy gradient. |

---

## 4. Danh Mục Tệp Kết Quả Bắt Buộc (Mandatory Artifacts)

Mọi lần chạy của notebook `Notebook/checkpoint04_training_strategies.ipynb` đều lưu các kết quả sau vào `outputs/checkpoint04/`:
1. `predictions.csv`: Cột `id`, `text`, `ground_truth`, `predicted_label`.
2. `confusion_matrix_checkpoint04.png` & `confusion_matrix.csv`: Biểu đồ nhiệt Seaborn và dữ liệu ma trận nhầm lẫn.
3. `learning_curves_checkpoint04.png`: Biểu đồ trực quan tiến trình Loss và Macro-F1.
4. `evaluation_scores.csv`: Điểm số tóm tắt chi tiết các độ đo.
5. `classification_report.csv`: Báo cáo chi tiết từng lớp với giá trị support nguyên.
6. `benchmark_comparison.csv`: Bảng đối sánh trực tiếp với Baseline gốc và các checkpoint trước.
