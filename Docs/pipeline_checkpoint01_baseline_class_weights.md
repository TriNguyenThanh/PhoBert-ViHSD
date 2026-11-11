# Đặc Tả Pipeline Checkpoint 01: Baseline Kết Hợp Cân Bằng Lớp Tĩnh (Smoothed Class Weights)

Tài liệu này đặc tả chi tiết kiến trúc luồng dữ liệu, công thức tính toán trọng số tĩnh Smoothed Class Weights, phương thức tích hợp vào hàm mất mát Weighted Cross-Entropy và quy chuẩn thực nghiệm cho Checkpoint 01 trong bài toán phân loại phát ngôn thù ghét tiếng Việt (ViHSD).

---

## 1. Mục Tiêu Kỹ Thuật

- Tái lập mô hình đường cơ sở (Baseline) theo công bố khoa học của Lưu et al. (2021) trên mô hình `vinai/phobert-base`.
- Giữ nguyên toàn bộ cấu hình huấn luyện của Baseline: 2 epochs, learning rate $2 \times 10^{-5}$, max_length = 100, batch size = 16, không thực hiện tiền xử lý dữ liệu sâu.
- Bổ sung cơ chế phạt tĩnh thông qua vector trọng số lớp làm mịn căn bậc hai (Smoothed Inverse Class Frequency):
  $$w_c = \sqrt{\frac{N}{N_c}} \bigg/ \left( \frac{1}{K} \sum_{j=0}^{K-1} \sqrt{\frac{N}{N_j}} \right)$$
- Kiểm soát tỷ lệ trọng số thiểu số / đa số ở mức lý tưởng (~3.52 lần), ngăn chặn hiện tượng trừng phạt quá mức dẫn đến sụp đổ Precision của lớp đa số (CLEAN).
- Xuất đầy đủ 4 tệp Mandatory Artifacts theo quy chuẩn học thuật vào `outputs/checkpoint01/`.

---

## 2. Kiến Trúc Luồng Xử Lý (Pipeline Architecture)

```
[Dữ liệu thô: train.csv (24.048 mẫu), dev.csv (2.672 mẫu), test.csv (6.680 mẫu)]
                                      |
                                      v
                     Làm sạch dòng rỗng & Trích xuất (X, y)
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
        Tokenize qua FastBPE (max_len=100)    Tính toán Vector Smoothed CW
                     |                                 |
                     v                                 v
               BuildDataset                    alpha = [0.47, 1.66, 1.32]
                     |                                 |
                     +----------------+----------------+
                                      |
                                      v
            WeightedLossTrainer (CrossEntropyLoss(weight=alpha))
                                      |
                                      v
                 Huấn luyện 2 Epochs trên vinai/phobert-base
                                      |
                                      v
               Đánh giá độc lập trên tập test.csv & Xuất Artifacts
```

---

## 3. Cấu Hình Siêu Tham Số

| Tham số | Giá trị | Giải thích |
| :--- | :---: | :--- |
| **Mô hình nền tảng** | `vinai/phobert-base` | Kiến trúc RoBERTa-base tiền huấn luyện cho tiếng Việt. |
| **Độ dài chuỗi (max_length)** | 100 | Đồng bộ cấu hình Baseline gốc. |
| **Số epochs** | 2 | Đồng bộ cấu hình Baseline gốc. |
| **Batch size** | 16 | Kích thước mini-batch chuẩn. |
| **Learning rate** | $2 \times 10^{-5}$ | Tốc độ học tối ưu của AdamW trên Transformer. |
| **Warmup steps** | 500 | Khởi động tuyến tính theo baseline. |
| **Weight decay** | 0.01 | Kiểm soát độ phức tạp trọng số. |
| **Hàm mất mát** | Weighted Cross-Entropy | Áp dụng trọng số làm mịn $\alpha$. |
| **Tỷ lệ trọng số $\alpha$** | `[0.4735, 1.6657, 1.3242]` | Tỷ lệ thiểu số/đa số: $\approx 3.52$. |

---

## 4. Danh Mục Tệp Kết Quả Bắt Buộc (Mandatory Artifacts)

Mọi lần chạy của notebook `Notebook/checkpoint01_baseline_class_weights.ipynb` đều lưu các kết quả sau vào `outputs/checkpoint01/`:
1. `predictions.csv`: Cột `id`, `text`, `ground_truth`, `predicted_label`.
2. `confusion_matrix_checkpoint01.png` & `confusion_matrix.csv`: Biểu đồ nhiệt Seaborn và dữ liệu ma trận nhầm lẫn.
3. `evaluation_scores.csv`: Tổng hợp các chỉ số Macro-F1, Micro-F1, Accuracy, Precision, Recall và F1 từng lớp.
4. `classification_report.csv`: Báo cáo chi tiết từng lớp với giá trị support nguyên.
5. `benchmark_comparison.csv`: Bảng đối sánh trực tiếp với Baseline gốc.
