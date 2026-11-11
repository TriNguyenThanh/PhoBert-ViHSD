# Đặc Tả Pipeline Checkpoint 02: Baseline Kết Hợp Cân Bằng Lớp Động (Focal Loss, gamma = 1.5)

Tài liệu này đặc tả chi tiết cơ sở toán học, thiết kế mô-đun hàm mất mát Focal Loss, kỹ thuật tối ưu hóa số học và phương thức tích hợp vào luồng huấn luyện cho Checkpoint 02 trong bài toán phân loại phát ngôn thù ghét tiếng Việt (ViHSD).

---

## 1. Mục Tiêu Kỹ Thuật

- Đánh giá khả năng tự động triệt tiêu gradient từ các mẫu dễ phân loại (CLEAN chiếm ~82.7%) bằng hàm mất mát Focal Loss trên cấu hình Baseline gốc của Lưu et al. (2021).
- Cô lập hoàn toàn tác động của cơ chế điều chế mẫu khó bằng cách thiết lập $\alpha = \text{None}$ (hoặc $\alpha = 1$), không sử dụng vector trọng số lớp tĩnh.
- Tận dụng hệ số điều chế $(1 - p_t)^\gamma$ với tham số tập trung tối ưu $\gamma = 1.5$:
  $$\mathcal{L}_{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t), \quad \gamma = 1.5$$
- Đảm bảo tính ổn định số học trong PyTorch bằng cách tính $p_t = \exp(-\mathcal{L}_{CE})$ qua `F.cross_entropy` với thủ thuật log-sum-exp.
- Xuất đầy đủ 4 tệp Mandatory Artifacts theo quy chuẩn vào thư mục `outputs/checkpoint02/`.

---

## 2. Kiến Trúc Luồng Xử Lý (Pipeline Architecture)

```
[Dữ liệu thô: train.csv, dev.csv, test.csv]
                     |
                     v
  Làm sạch dòng rỗng & Mã hóa FastBPE (max_length = 100)
                     |
                     v
  Đóng gói BuildDataset (train_dataset, dev_dataset, test_dataset)
                     |
                     v
  Khởi tạo mô hình vinai/phobert-base (num_labels = 3)
                     |
                     v
  FocalLossTrainer (ghi đè compute_loss với FocalLoss(gamma=1.5))
                     |
                     v
  Huấn luyện 2 Epochs theo cấu hình Baseline gốc
                     |
                     v
  Dự đoán trên test_dataset & Xuất Artifacts vào outputs/checkpoint02/
```

---

## 3. Cấu Hình Siêu Tham Số

| Tham số | Giá trị | Giải thích |
| :--- | :---: | :--- |
| **Mô hình nền tảng** | `vinai/phobert-base` | Kiến trúc Transformer tiếng Việt tiêu chuẩn. |
| **Độ dài chuỗi (max_length)** | 100 | Đồng bộ cấu hình Baseline gốc. |
| **Số epochs** | 2 | Đồng bộ cấu hình Baseline gốc. |
| **Batch size** | 16 | Kích thước mini-batch chuẩn. |
| **Learning rate** | $2 \times 10^{-5}$ | Tốc độ học tối ưu của AdamW. |
| **Warmup steps** | 500 | Khởi động tuyến tính theo baseline. |
| **Weight decay** | 0.01 | Suy giảm trọng số. |
| **Hàm mất mát** | Focal Loss | Điều chế mẫu khó theo hàm mũ. |
| **Tham số tập trung $\gamma$** | 1.5 | Triệt tiêu gradient mẫu dễ, bảo vệ gradient mẫu khó. |
| **Trọng số tĩnh $\alpha$** | None | Cô lập tác động của $\gamma$. |

---

## 4. Danh Mục Tệp Kết Quả Bắt Buộc (Mandatory Artifacts)

Mọi lần chạy của notebook `Notebook/checkpoint02_baseline_focal_loss.ipynb` đều lưu các kết quả sau vào `outputs/checkpoint02/`:
1. `predictions.csv`: Cột `id`, `text`, `ground_truth`, `predicted_label`.
2. `confusion_matrix_checkpoint02.png` & `confusion_matrix.csv`: Biểu đồ nhiệt Seaborn và dữ liệu ma trận nhầm lẫn.
3. `evaluation_scores.csv`: Điểm số tóm tắt chi tiết các độ đo.
4. `classification_report.csv`: Báo cáo chi tiết từng lớp với giá trị support nguyên.
5. `benchmark_comparison.csv`: Bảng đối sánh trực tiếp với Baseline gốc và Checkpoint 01.
