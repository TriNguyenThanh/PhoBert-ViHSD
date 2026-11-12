# Đặc Tả Pipeline Checkpoint 06: Hiện Đại Hóa Kiến Trúc Mô Hình (PhoBERT-base-v2 Modernization)

Tài liệu này đặc tả chi tiết cơ sở khoa học, quy trình chuyển đổi kiến trúc nền tảng sang `vinai/phobert-base-v2`, phương thức kế thừa trọn vẹn pipeline tối ưu hóa từ Checkpoint 05 và phân tích đối sánh năng lực biểu diễn ngôn ngữ cho Checkpoint 06 trong bài toán phân loại phát ngôn thù ghét (ViHSD).

---

## 1. Mục Tiêu Kỹ Thuật

- Đánh giá tác động độc lập của việc nâng cấp nền tảng biểu diễn ngôn ngữ tiền huấn luyện (Backbone Modernization) từ `vinai/phobert-base` thế hệ đầu sang `vinai/phobert-base-v2` thế hệ mới.
- Ưu thế vượt trội của `vinai/phobert-base-v2`:
  - Tiền huấn luyện trên 20GB văn bản tiếng Việt hiện đại (gấp hơn 5 lần so với tập dữ liệu 4GB của phiên bản v1).
  - Khả năng biểu diễn chính xác hơn các biến thể từ vựng, tiếng lóng mạng xã hội và cấu trúc câu phức tạp.
  - Vốn từ vựng FastBPE được tối ưu hóa tốt hơn, giảm thiểu hiện tượng phân mảnh token.
- Kế thừa 100% các thành phần tối ưu hóa từ Checkpoint 05:
  - `ViHSDPreprocessor` tiền xử lý chuyên sâu mạng xã hội tiếng Việt.
  - Hàm mất mát Weighted Focal Loss ($\alpha$ Smoothed Class Weights, $\gamma = 1.5$).
  - Chiến lược huấn luyện nâng cao (`max_length = 256`, differential learning rates, Cosine Annealing, 8 epochs, Early Stopping).
- Xuất đầy đủ 4 tệp Mandatory Artifacts cùng biểu đồ Learning Curves vào `outputs/checkpoint06/`.

---

## 2. So Sánh Kiến Trúc: PhoBERT-base v1 vs PhoBERT-base-v2

| Đặc tính kỹ thuật | PhoBERT-base (v1) | PhoBERT-base-v2 (v2) | Ý nghĩa học thuật |
| :--- | :---: | :---: | :--- |
| **Dung lượng tập tiền huấn luyện** | ~4 GB văn bản | **~20 GB văn bản** | Khái quát hóa ngôn ngữ mạng xã hội phong phú hơn. |
| **Nguồn dữ liệu huấn luyện** | Báo chí, Wikipedia | Báo chí, MXH, Web corpora | Phản ánh ngôn ngữ giao tiếp đời thực tự nhiên hơn. |
| **Số tầng Transformer** | 12 layers | 12 layers | Giữ nguyên độ phức tạp tham số (~135M). |
| **Kích thước tầng ẩn ($d$)** | 768 | 768 | Chi phí tính toán tương đương, suy luận nhanh chóng. |
| **Tokenizer** | FastBPE (use_fast=False) | FastBPE (use_fast=False) | Tương thích hoàn toàn với quy trình tách từ PyVi. |

---

## 3. Kiến Trúc Luồng Huấn Luyện (Modernized Pipeline Architecture)

```
[Dữ liệu thô: train.csv, dev.csv, test.csv]
                     |
                     v
   Tiền xử lý với ViHSDPreprocessor (CP03)
                     |
                     v
   Mã hóa bằng vinai/phobert-base-v2 Tokenizer (max_length = 256)
                     |
                     v
   Tính toán Vector trọng số làm mịn căn bậc hai alpha (CP01)
                     |
                     v
   CombinedOptimizedTrainer
   - Mô hình: AutoModelForSequenceClassification('vinai/phobert-base-v2')
   - Hàm mất mát: WeightedFocalLoss(alpha=active_class_weights, gamma=1.5)
   - Optimizer: Differential AdamW (lr_roberta=1.5e-5, lr_head=5e-5)
   - Scheduler: Cosine Annealing with Warmup (10%)
   - Callbacks: EarlyStoppingCallback(patience = 2)
                     |
                     v
   Huấn luyện tối đa 8 Epochs & Lưu Checkpoint tối ưu nhất theo eval_macro_f1
                     |
                     v
   Đánh giá trên tập test & Xuất Learning Curves + Artifacts vào outputs/checkpoint06/
```

---

## 4. Cấu Hình Siêu Tham Số

| Tham số | Giá trị | Giải thích |
| :--- | :---: | :--- |
| **Mô hình nền tảng** | **`vinai/phobert-base-v2`** | Kiến trúc Transformer tiếng Việt thế hệ 2 hiện đại (20GB ngữ liệu). |
| **Tiền xử lý văn bản** | `ViHSDPreprocessor` | Tự chứa: Emoji 2 pha + Teencode 412 + PyVi. |
| **Độ dài chuỗi (max_length)** | **256** | Ngữ cảnh bình luận trọn vẹn. |
| **Số epochs tối đa** | **8** | Huấn luyện sâu với dừng sớm (patience = 2). |
| **Hàm mất mát** | **Weighted Focal Loss** | Kết hợp $\alpha$ Smoothed CW và $\gamma = 1.5$. |
| **Tốc độ học phân tầng** | **$1.5 \times 10^{-5}$ / $5.0 \times 10^{-5}$** | RoBERTa v2 Backbone / Classification Head (tỷ lệ cân bằng 3.3x chống gradient shock). |
| **Lịch trình LR** | Cosine with Warmup | 10% bước khởi động tuyến tính (warmup_ratio = 0.10). |
| **Batch size hiệu dụng** | **32** | Mini-batch 16 tích lũy 2 bước gradient. |

*Ghi chú về tinh chỉnh Learning Rate:* Không gian biểu diễn của PhoBERT-base-v2 được tiền huấn luyện sâu trên 20GB ngữ liệu. Việc hạ `head_lr` từ $1 \times 10^{-4}$ xuống $5 \times 10^{-5}$ và tinh chỉnh `backbone_lr` về $1.5 \times 10^{-5}$ giúp loại bỏ hiện tượng gradient shock từ classification head ngẫu nhiên lên các tầng trên cùng của backbone, ngăn ngừa việc mô hình tự khóa quá sớm vào lớp đa số CLEAN, bảo toàn và nâng cao Recall của lớp HATE và OFFENSIVE.

---

## 5. Danh Mục Tệp Kết Quả Bắt Buộc (Mandatory Artifacts)

Mọi lần chạy của notebook `Notebook/checkpoint06_phobert_base_v2.ipynb` đều lưu các kết quả sau vào `outputs/checkpoint06/`:
1. `predictions.csv`: Cột `id`, `text`, `ground_truth`, `predicted_label`.
2. `confusion_matrix_checkpoint06.png` & `confusion_matrix.csv`: Biểu đồ nhiệt Seaborn và dữ liệu ma trận nhầm lẫn.
3. `learning_curves_checkpoint06.png`: Đồ thị tiến trình hội tụ Loss và Macro-F1.
4. `evaluation_scores.csv`: Điểm số tóm tắt chi tiết các độ đo.
5. `classification_report.csv`: Báo cáo chi tiết từng lớp với giá trị support nguyên.
6. `benchmark_comparison.csv`: Bảng đối sánh trực tiếp toàn bộ 6 mốc thực nghiệm từ Baseline đến PhoBERT-base-v2.
