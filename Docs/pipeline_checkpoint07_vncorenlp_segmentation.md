# Đặc Tả Pipeline Checkpoint 07: Tích Hợp Toàn Diện Kết Hợp Phân Đoạn Từ VnCoreNLP (VnCoreNLP Segmentation)

Tài liệu này đặc tả chi tiết cơ sở khoa học, cơ chế khảo sát bóc tách thành phần (Ablation Study) giữa công cụ phân đoạn từ `pyvi` và `py_vncorenlp`, quy trình tích hợp mô hình tách từ `RDRsegmenter` vào lớp `ViHSDPreprocessor` và phương thức đối sánh hiệu năng trên nền tảng Checkpoint 05 cho bài toán phân loại phát ngôn thù ghét (ViHSD).

---

## 1. Mục Tiêu Kỹ Thuật

- **Thực hiện khảo sát bóc tách thành phần (Ablation Study) về công cụ phân đoạn từ tiếng Việt:**
  - Trong công bố nền tảng của PhoBERT (*"PhoBERT: Pre-trained language models for Vietnamese"*, Nguyen & Nguyen, Findings of EMNLP 2020), toàn bộ kho ngữ liệu tiền huấn luyện khổng lồ đều được phân đoạn từ bằng công cụ `RDRsegmenter` thuộc bộ công cụ `VnCoreNLP` (VinAI Research).
  - Tại Checkpoint 05, thư viện `pyvi` được sử dụng như một giải pháp phân đoạn từ nhẹ, viết bằng Python thuần túy. Tuy nhiên, sự khác biệt trong ranh giới tách từ ghép (word boundaries) giữa PyVi và VnCoreNLP có thể dẫn đến hiện tượng lệch phân phối từ vựng (vocabulary mismatch) khi đưa vào bộ mã hóa FastBPE của PhoBERT.
  - Tại Checkpoint 07, toàn bộ pipeline tối ưu của Checkpoint 05 được giữ nguyên 100%, chỉ thay thế duy nhất công cụ tách từ từ `pyvi` sang `py_vncorenlp` (RDRsegmenter). Thực nghiệm này nhằm trả lời câu hỏi nghiên cứu: *Liệu việc sử dụng đúng bộ tách từ gốc chính thức của PhoBERT có giúp cải thiện chất lượng biểu diễn ngữ nghĩa và nâng cao chỉ số Macro-F1 so với giải pháp PyVi hay không?*
- **Kế thừa toàn diện pipeline tối ưu từ Checkpoint 05:**
  - Mô hình nền tảng: `vinai/phobert-base`.
  - Bộ tiền xử lý `ViHSDPreprocessor`: chuẩn hóa teencode 412 quy tắc, ánh xạ emoji 2 pha, thu gọn ký tự kéo dài, bảo tồn đại từ nhân xưng và từ phủ định.
  - Hàm mất mát **Weighted Focal Loss**: kết hợp vector trọng số làm mịn căn bậc hai $\alpha$ (Smoothed Class Weights) và tham số điều chế mẫu khó $\gamma = 1.5$.
  - Chiến lược huấn luyện nâng cao: độ dài chuỗi `max_length = 256`, phân tầng tốc độ học ($2 \times 10^{-5}$ cho RoBERTa / $1 \times 10^{-4}$ cho Classification Head), lịch trình Cosine Annealing with Warmup, 8 epochs, batch size hiệu dụng 32 (16 tích lũy 2 bước), Early Stopping với patience = 2 theo `eval_macro_f1`.
- Xuất đầy đủ 4 tệp Mandatory Artifacts cùng biểu đồ Learning Curves vào `outputs/checkpoint07/`.

---

## 2. So Sánh Bản Chất: PyVi vs VnCoreNLP (RDRsegmenter)

| Đặc tính kỹ thuật | PyVi (`ViTokenizer`) | VnCoreNLP (`py_vncorenlp`) | Tác động học thuật trong nghiên cứu |
| :--- | :---: | :---: | :--- |
| **Thuật toán cốt lõi** | Maximum Matching & Từ điển heuristics | **Ripple Down Rules (RDR)** | VnCoreNLP xử lý đa nghĩa và ngữ cảnh ranh giới từ phức tạp chính xác hơn. |
| **Độ chính xác tách từ (F1)** | ~96.0% trên VLSP | **~97.9% trên VLSP** | Giảm thiểu lỗi tách rời các từ ghép mang sắc thái thù ghét/xúc phạm. |
| **Tương thích với PhoBERT** | Công cụ bên thứ ba (xấp xỉ) | **Công cụ gốc chính thức của PhoBERT** | Đồng bộ 100% với không gian từ vựng FastBPE của tác giả PhoBERT. |
| **Phụ thuộc môi trường** | Python thuần túy, không cần Java | **Yêu cầu môi trường Java 1.8+** | Cần khởi tạo JVM qua wrapper `py_vncorenlp` trên Google Colab. |
| **Tốc độ suy luận** | Rất nhanh | Nhanh khi chỉ nạp annotator `wseg` | Cấu hình `annotators=["wseg"]` để tối ưu hóa thời gian tiền xử lý. |

---

## 3. Kiến Trúc Luồng Huấn Luyện (Checkpoint 07 Architecture)

```
[Dữ liệu thô: train.csv, dev.csv, test.csv]
                     |
                     v
   Khởi tạo py_vncorenlp.VnCoreNLP(annotators=["wseg"])
                     |
                     v
   Tiền xử lý với ViHSDPreprocessor (Segmenter = VnCoreNLP RDRsegmenter)
   (Làm sạch ký tự -> Emoji 2 pha -> Teencode 412 -> Thu gọn ký tự -> Tách từ VnCoreNLP)
                     |
                     v
   Mã hóa bằng vinai/phobert-base Tokenizer (max_length = 256, use_fast = False)
                     |
                     v
   Tính toán Vector trọng số làm mịn căn bậc hai alpha (Smoothed Class Weights)
                     |
                     v
   CombinedOptimizedTrainer
   - Mô hình: AutoModelForSequenceClassification('vinai/phobert-base')
   - Hàm mất mát: WeightedFocalLoss(alpha=active_class_weights, gamma=1.5)
   - Optimizer: Differential AdamW (lr_roberta=2e-5, lr_head=1e-4)
   - Scheduler: Cosine Annealing with Warmup (10% warmup)
   - Callbacks: EarlyStoppingCallback(patience = 2)
                     |
                     v
   Huấn luyện tối đa 8 Epochs & Phục hồi Checkpoint tối ưu nhất theo eval_macro_f1
                     |
                     v
   Đánh giá trên tập test & Xuất Learning Curves + Artifacts vào outputs/checkpoint07/
```

---

## 4. Cấu Hình Siêu Tham Số

| Tham số | Giá trị | Giải thích |
| :--- | :---: | :--- |
| **Mô hình nền tảng** | `vinai/phobert-base` | Transformer tiếng Việt tiêu chuẩn (v1). |
| **Công cụ phân đoạn từ** | **`py_vncorenlp` (RDRsegmenter)** | Bộ tách từ gốc chính thức của PhoBERT. |
| **Tiền xử lý văn bản** | `ViHSDPreprocessor` | Tự chứa: Emoji 2 pha + Teencode 412 + VnCoreNLP. |
| **Độ dài chuỗi (max_length)** | **256** | Ngưỡng nhúng tối đa của PhoBERT. |
| **Số epochs tối đa** | **8** | Huấn luyện sâu với dừng sớm (patience = 2). |
| **Hàm mất mát** | **Weighted Focal Loss** | Kết hợp $\alpha$ Smoothed CW và $\gamma = 1.5$. |
| **Tốc độ học phân tầng** | $2 \times 10^{-5}$ / $1 \times 10^{-4}$ | Backbone RoBERTa / Classification Head. |
| **Lịch trình LR** | Cosine with Warmup | 10% bước khởi động tuyến tính. |
| **Batch size hiệu dụng** | **32** | Mini-batch 16 tích lũy 2 bước gradient. |

---

## 5. Danh Mục Tệp Kết Quả Bắt Buộc (Mandatory Artifacts)

Mọi lần chạy của notebook `Notebook/checkpoint07_vncorenlp_segmentation.ipynb` đều lưu các kết quả sau vào `outputs/checkpoint07/`:
1. `predictions.csv`: Cột `id`, `text`, `ground_truth`, `predicted_label`.
2. `confusion_matrix_checkpoint07.png` & `confusion_matrix.csv`: Biểu đồ nhiệt Seaborn và dữ liệu ma trận nhầm lẫn.
3. `learning_curves_checkpoint07.png`: Đồ thị tiến trình hội tụ Loss và Macro-F1.
4. `evaluation_scores.csv`: Điểm số tóm tắt chi tiết các độ đo.
5. `classification_report.csv`: Báo cáo chi tiết từng lớp với giá trị support nguyên.
6. `benchmark_comparison.csv`: Bảng đối sánh trực tiếp toàn bộ các mốc thực nghiệm, đặc biệt làm nổi bật tương quan giữa PyVi (CP05) và VnCoreNLP (CP07).
