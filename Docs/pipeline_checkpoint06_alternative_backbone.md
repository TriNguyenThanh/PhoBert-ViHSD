# Đặc Tả Pipeline Checkpoint 6: Khảo Sát Kiến Trúc Mô Hình Nền Tảng Mới (Alternative Backbone Architecture)

Tài liệu này đặc tả chi tiết cơ sở lý thuyết, kiến trúc mô hình, phương án tối ưu hóa tài nguyên phần cứng và quy trình đánh giá thực nghiệm cho Checkpoint 6 trong bài toán phân loại phát ngôn thù ghét tiếng Việt (ViHSD).

---

## 1. Mục Tiêu Kỹ Thuật

- Kiểm chứng giả thuyết về **năng lực biểu diễn của mô hình nền tảng (Foundation Model Capacity)**: Liệu việc mở rộng số lượng tham số và chiều sâu kiến trúc từ `vinai/phobert-base` (135M tham số) lên `vinai/phobert-large` (370M tham số) có cải thiện đáng kể khả năng phân biệt các dạng phát ngôn thù ghét tinh vi, ẩn ý, châm biếm (sarcasm) và đa tầng ngữ nghĩa hay không.
- Kế thừa toàn bộ chuỗi cải tiến đã được kiểm chứng từ các Checkpoint trước:
  - Chuỗi tiền xử lý dữ liệu chuẩn hóa (Checkpoint 1): Teencode, Emoji 2 pha, phân đoạn từ PyVi, bảo tồn đại từ nhân xưng và từ phủ định.
  - Chiến lược huấn luyện nâng cao (Checkpoint 2): Độ dài chuỗi `max_length = 256`, cơ chế Cosine Annealing Scheduler, lưu checkpoint tốt nhất theo `eval_macro_f1` kết hợp Early Stopping.
  - Hàm mất mát giải quyết mất cân bằng lớp (Checkpoint 3 / Checkpoint 5): Tích hợp Focal Loss ($\gamma = 2.0$) hoặc Weighted Focal Loss nhằm bảo vệ gradient của các lớp thiểu số.
- Thiết kế giải pháp kiểm soát bộ nhớ GPU (VRAM Management) chống tràn bộ nhớ (CUDA Out-Of-Memory) khi huấn luyện mô hình quy mô lớn trên nền tảng Google Colab GPU (16GB VRAM).

---

## 2. So Sánh Kiến Trúc: PhoBERT-base vs PhoBERT-large

| Đặc Tính Kỹ Thuật | PhoBERT-base (`vinai/phobert-base`) | PhoBERT-large (`vinai/phobert-large`) | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| **Tổng số tham số (Parameters)** | ~135 triệu (135M) | **~370 triệu (370M)** | Khả năng ghi nhớ và khái quát hóa ngữ cảnh tăng ~2.74 lần. |
| **Số tầng Transformer (Layers)** | 12 tầng | **24 tầng** | Trích xuất các tầng đặc trưng trừu tượng sâu hơn. |
| **Kích thước tầng ẩn (Hidden size $d$)** | 768 | **1024** | Không gian biểu diễn vector dày đặc và phong phú hơn. |
| **Kích thước tầng FFN (Intermediate)** | 3072 | **4096** | Nâng cao năng lực biến đổi phi tuyến cục bộ. |
| **Số đầu tự chú ý (Attention Heads)** | 12 đầu | **16 đầu** | Nắm bắt đồng thời nhiều mối liên hệ ngữ pháp và ngữ nghĩa chéo. |
| **Vị trí nhúng tối đa (Max Position)** | 256 | **256** | Đồng bộ trọn vẹn với ngưỡng cắt ngữ cảnh `max_length = 256`. |
| **Từ điển & Thuật toán Tokenizer** | BPE (FastBPE + PyVi) | BPE (FastBPE + PyVi) | **Hoàn toàn tương thích**: Tái sử dụng 100% pipeline tiền xử lý từ CP1. |

---

## 3. Kiến Trúc Luồng Xử Lý (Pipeline Architecture)

```
[Tập dữ liệu chuẩn hóa ViHSD: train_X, dev_X, test_X (Checkpoint 1)]
                                  |
                                  v
[Mã hóa với Tokenizer: vinai/phobert-large (use_fast=False, max_length=256)]
                                  |
                                  v
       [PyTorch BuildDataset: input_ids, attention_mask, labels]
                                  |
                                  v
  [Khởi tạo Mô hình vinai/phobert-large (24 layers, 1024 hidden, 3 nhãn)]
                                  |
                                  v
       [Phân tầng Tốc độ học: Backbone lr = 1e-5 | Head lr = 1e-4]
                                  |
                                  v
  [Cấu hình Batch Size: per_device_batch_size = 8, grad_accum = 4 (Effective BS = 32)]
                                  |
                                  v
[Hàm Mất Mát: Focal Loss (gamma = 2.0) triệt tiêu mẫu CLEAN dễ, tập trung mẫu khó]
                                  |
                                  v
   [Huấn luyện 8 Epochs + Mixed Precision fp16 + Cosine LR Scheduler + Warmup 10%]
                                  |
                                  v
    [Đánh giá sau mỗi Epoch trên Dev Set -> Nạp Best Checkpoint theo eval_macro_f1]
                                  |
                                  v
[Đánh giá độc lập trên Test Set (6680 mẫu) & Đối sánh hiệu năng với PhoBERT-base]
                                  |
                                  v
     [Xuất 5 tệp kết quả chuẩn hóa vào outputs/checkpoint06/]
```

---

## 4. Giải Pháp Kiểm Soát Tài Nguyên và Tối Ưu Hóa Bộ Nhớ (VRAM Optimization)

### 4.1. Thách thức bộ nhớ của mô hình Large
Mô hình `vinai/phobert-large` sở hữu 370M tham số. Với `max_length = 256`, ma trận Self-Attention kích thước $(256 \times 256)$ nhân với 16 heads qua 24 layers tạo ra áp lực bộ nhớ kích hoạt (activation memory) rất lớn trong quá trình lan truyền xuôi và ngược.
Nếu giữ nguyên `per_device_train_batch_size = 16` như ở Checkpoint 2-5, mức tiêu thụ bộ nhớ sẽ vượt quá 16GB VRAM của card đồ họa Nvidia T4 hoặc V100 trên Google Colab, dẫn đến lỗi `CUDA Out of Memory`.

### 4.2. Chiến lược điều chỉnh Batch và Tích lũy Gradient
Để duy trì đúng động học cập nhật trọng số của batch hiệu dụng 32 mà không bị tràn bộ nhớ:
$$\text{Effective Batch Size} = \text{per\_device\_train\_batch\_size} \times \text{gradient\_accumulation\_steps} = 8 \times 4 = 32$$

- `per_device_train_batch_size = 8`: Giảm tải 50% lượng activation trên mỗi lượt forward.
- `gradient_accumulation_steps = 4`: Tích lũy gradient qua 4 bước mini-batch trước khi gọi `optimizer.step()` và `optimizer.zero_grad()`.
- `fp16 = True`: Lưu trữ trọng số và kích hoạt dưới định dạng bán chính xác 16-bit, giảm 50% dung lượng VRAM.

### 4.3. Cân chỉnh Tốc độ học phân tầng cho Kiến trúc sâu (24 tầng)
Mô hình 24 tầng rất nhạy cảm với tốc độ học lớn. Nếu sử dụng `backbone_lr = 2e-5`, các tầng biểu diễn phía dưới dễ bị xáo trộn:
- **Tốc độ học Backbone RoBERTa Large:** $\text{lr}_{\text{backbone}} = 1 \times 10^{-5}$ (bảo toàn tri thức biểu diễn sâu).
- **Tốc độ học Classification Head:** $\text{lr}_{\text{head}} = 1 \times 10^{-4}$ (thúc đẩy hội tụ nhanh cho tầng phân loại ngẫu nhiên).
- **Weight Decay:** $\lambda = 0.01$ loại trừ các tham số bias và LayerNorm.

---

## 5. Quy Trình Kiểm Thử và Tiêu Chuẩn Nghiệm Thu

1. **Khởi tạo và kiểm tra tính toàn vẹn:**
   - Notebook `Notebook/checkpoint06_alternative_model.ipynb` hoàn toàn tự chứa (100% self-contained), có thể thực thi độc lập trên Google Colab.
   - Quá trình tải tokenizer và mô hình `vinai/phobert-large` diễn ra ổn định, ánh xạ đúng 3 nhãn phân loại (`num_labels = 3`).
   - Kiểm tra tính ổn định số học của hàm mất mát, không xuất hiện `NaN` hay `Inf`.

2. **Tiêu chuẩn nghiệm thu trên tập Test (`test.csv`):**
   - Đánh giá độc lập trên toàn bộ 6,680 mẫu kiểm tra.
   - So sánh trực tiếp chỉ số `Macro-F1`, `F1-Clean`, `F1-Offensive`, `F1-Hate` so với `PhoBERT-base` từ các Checkpoint trước.
   - Phân tích chi tiết ma trận nhầm lẫn: Kiểm tra khả năng nhận diện các mẫu ngôn ngữ thù ghét ẩn ý và các mẫu ranh giới phức tạp giữa OFFENSIVE và HATE.
   - Xuất đầy đủ 4 tệp kết quả vào `outputs/checkpoint06/`:
     1. `predictions.csv`
     2. `confusion_matrix_checkpoint06.png` & `confusion_matrix.csv`
     3. `evaluation_scores.csv`
     4. `classification_report.csv`
