# Đặc Tả Pipeline Checkpoint 05: Tích Hợp Toàn Diện Các Giải Pháp Nâng Cấp (Combined Enhancements)

Tài liệu này đặc tả chi tiết kiến trúc tích hợp đồng thời 4 giải pháp tối ưu hóa độc lập (CP01 + CP02 + CP03 + CP04), cơ chế hoạt động của hàm mất mát Weighted Focal Loss và phương thức đánh giá trên mô hình `vinai/phobert-base` cho Checkpoint 05 trong bài toán phân loại phát ngôn thù ghét (ViHSD).

---

## 1. Mục Tiêu Kỹ Thuật

- Tổng hợp và cộng hưởng toàn bộ các ưu thế kỹ thuật đã được kiểm chứng từ các mốc thực nghiệm trước:
  1. **Tiền xử lý mạng xã hội chuyên sâu (CP03):** Áp dụng `ViHSDPreprocessor` (chuẩn hóa teencode 412 quy tắc, ánh xạ emoji 2 pha, thu gọn ký tự, tách từ ghép PyVi, bảo tồn đại từ nhân xưng và từ phủ định).
  2. **Chiến lược huấn luyện nâng cao (CP04):** Mở rộng chuỗi `max_length = 256`, phân tầng tốc độ học ($2 \times 10^{-5}$ roberta / $1 \times 10^{-4}$ classifier), Cosine Annealing with Warmup, 8 epochs, batch size hiệu dụng 32, Early Stopping với patience = 2 dựa trên `eval_macro_f1`.
  3. **Hàm mất mát Weighted Focal Loss (CP01 + CP02):** Kết hợp vector trọng số làm mịn căn bậc hai $\alpha$ (Smoothed Class Weights) và tham số điều chế mẫu khó $\gamma = 1.5$:
     $$\mathcal{L}_{WFL} = -\alpha_{y_i} (1 - p_{i, y_i})^\gamma \log(p_{i, y_i})$$
- Kiểm chứng hiệu ứng tương hỗ giữa việc làm sạch biểu diễn ngôn ngữ, cân bằng động học tối ưu hóa và chống phạt quá mức.
- Xuất đầy đủ 4 tệp Mandatory Artifacts cùng biểu đồ Learning Curves vào `outputs/checkpoint05/`.

---

## 2. Kiến Trúc Luồng Tích Hợp (Combined Architecture)

```
[Dữ liệu thô: train.csv, dev.csv, test.csv]
                     |
                     v
   Tiền xử lý với ViHSDPreprocessor (CP03)
                     |
                     v
   Mã hóa PhoBERT Tokenizer (max_length = 256, use_fast = False)
                     |
                     v
   Tính toán Vector trọng số làm mịn căn bậc hai alpha (CP01)
                     |
                     v
   CombinedOptimizedTrainer
   - Loss Function: WeightedFocalLoss(alpha=active_class_weights, gamma=1.5)
   - Optimizer: Differential AdamW (lr_roberta=2e-5, lr_head=1e-4)
   - Scheduler: Cosine Annealing with 10% Warmup
   - Callbacks: EarlyStoppingCallback(early_stopping_patience = 2)
                     |
                     v
   Huấn luyện 8 Epochs trên vinai/phobert-base & Lưu Best Checkpoint
                     |
                     v
   Đánh giá trên tập test & Xuất Learning Curves + Artifacts vào outputs/checkpoint05/
```

---

## 3. Cấu Hình Siêu Tham Số

| Tham số | Giá trị | Giải thích |
| :--- | :---: | :--- |
| **Mô hình nền tảng** | `vinai/phobert-base` | Kiến trúc Transformer tiếng Việt v1. |
| **Tiền xử lý văn bản** | `ViHSDPreprocessor` | Tự chứa: Emoji 2 pha + Teencode 412 + PyVi. |
| **Độ dài chuỗi (max_length)** | **256** | Bao phủ ngữ cảnh bình luận trọn vẹn. |
| **Số epochs tối đa** | **8** | Huấn luyện sâu với dừng sớm. |
| **Hàm mất mát** | **Weighted Focal Loss** | Kết hợp $\alpha$ Smoothed CW và $\gamma = 1.5$. |
| **Tốc độ học phân tầng** | $2 \times 10^{-5}$ / $1 \times 10^{-4}$ | Backbone RoBERTa / Classification Head. |
| **Lịch trình LR** | Cosine with Warmup | 10% bước khởi động tuyến tính. |
| **Batch size hiệu dụng** | **32** | Mini-batch 16 tích lũy 2 bước gradient. |

---

## 4. Danh Mục Tệp Kết Quả Bắt Buộc (Mandatory Artifacts)

Mọi lần chạy của notebook `Notebook/checkpoint05_combined_enhancements.ipynb` đều lưu các kết quả sau vào `outputs/checkpoint05/`:
1. `predictions.csv`: Cột `id`, `text`, `ground_truth`, `predicted_label`.
2. `confusion_matrix_checkpoint05.png` & `confusion_matrix.csv`: Biểu đồ nhiệt Seaborn và dữ liệu ma trận nhầm lẫn.
3. `learning_curves_checkpoint05.png`: Đồ thị tiến trình hội tụ Loss và Macro-F1.
4. `evaluation_scores.csv`: Điểm số tóm tắt chi tiết các độ đo.
5. `classification_report.csv`: Báo cáo chi tiết từng lớp với giá trị support nguyên.
6. `benchmark_comparison.csv`: Bảng đối sánh trực tiếp toàn bộ 5 mốc thực nghiệm.
