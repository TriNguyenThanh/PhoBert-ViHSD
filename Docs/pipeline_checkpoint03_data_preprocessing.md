# Đặc Tả Pipeline Checkpoint 03: Tiền Xử Lý và Chuẩn Hóa Dữ Liệu Mạng Xã Hội (Data Preprocessing)

Tài liệu này đặc tả chi tiết thiết kế lớp xử lý chuyên biệt `ViHSDPreprocessor`, quy tắc chuẩn hóa ngôn ngữ mạng xã hội tiếng Việt, cơ chế bảo tồn đại từ / từ phủ định và quy chuẩn tích hợp vào Checkpoint 03 trong bài toán phân loại phát ngôn thù ghét (ViHSD).

---

## 1. Mục Tiêu Kỹ Thuật

- Khắc phục các hạn chế nghiêm trọng trong tiền xử lý của nghiên cứu gốc (loại bỏ từ dừng bừa bãi và dùng deEmojify tiếng Anh gây nhiễu).
- Xây dựng lớp tiền xử lý độc lập `ViHSDPreprocessor` tự chứa (100% self-contained) trong notebook:
  - Làm sạch URL, thẻ điều khiển, chuẩn hóa Unicode dựng sẵn (NFC).
  - Ánh xạ emoji 2 pha: đệm khoảng trắng bảo vệ từ tiếng Việt trước PyVi và chuyển đổi thành từ tố ngữ nghĩa theo từ điển `emojis.csv`.
  - Chuẩn hóa viết tắt, tiếng lóng, ký tự né bộ lọc theo từ điển 412 quy tắc (`teencode.csv`).
  - Thu gọn ký tự kéo dài cảm xúc (`haizzzz` $\to$ `haiz`).
  - Phân đoạn từ tiếng Việt tự động bằng thư viện `pyvi` (`ViTokenizer.tokenize`).
  - **Bảo tồn tuyệt đối** đại từ nhân xưng (`mày`, `tao`, `nó`, `thằng`, `con`) và từ phủ định (`không`, `chẳng`, `chả`).
- Giữ nguyên cấu hình huấn luyện Baseline (2 epochs, lr = $2 \times 10^{-5}$, max_length = 100, Cross-Entropy tiêu chuẩn) nhằm cô lập chính xác mức tăng trưởng của chỉ số Macro-F1 do tiền xử lý mang lại.
- Xuất đầy đủ 4 tệp Mandatory Artifacts vào thư mục `outputs/checkpoint03/`.

---

## 2. Kiến Trúc Luồng Xử Lý (Pipeline Architecture)

```
[Dữ liệu thô: train.csv, dev.csv, test.csv]
                     |
                     v
             ViHSDPreprocessor
  (Làm sạch ký tự -> Emoji 2 pha -> Teencode 412 -> Thu gọn ký tự -> Tách từ PyVi)
                     |
                     v
   Kiểm tra toàn vẹn sau xử lý (Post-pipeline Assertion Checks)
                     |
                     v
   Mã hóa PhoBERT Tokenizer (max_length = 100, use_fast = False)
                     |
                     v
   Huấn luyện 2 Epochs với Cross-Entropy Loss tiêu chuẩn
                     |
                     v
   Đánh giá trên tập test & Xuất Artifacts vào outputs/checkpoint03/
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
| **Hàm mất mát** | Cross-Entropy tiêu chuẩn | Cô lập tác động của tiền xử lý dữ liệu. |
| **Bộ tiền xử lý** | `ViHSDPreprocessor` | Tự chứa, tích hợp emoji + teencode + PyVi. |

---

## 4. Danh Mục Tệp Kết Quả Bắt Buộc (Mandatory Artifacts)

Mọi lần chạy của notebook `Notebook/checkpoint03_data_preprocessing.ipynb` đều lưu các kết quả sau vào `outputs/checkpoint03/`:
1. `predictions.csv`: Cột `id`, `text`, `ground_truth`, `predicted_label`.
2. `confusion_matrix_checkpoint03.png` & `confusion_matrix.csv`: Biểu đồ nhiệt Seaborn và dữ liệu ma trận nhầm lẫn.
3. `evaluation_scores.csv`: Điểm số tóm tắt chi tiết các độ đo.
4. `classification_report.csv`: Báo cáo chi tiết từng lớp với giá trị support nguyên.
5. `benchmark_comparison.csv`: Bảng đối sánh trực tiếp với Baseline gốc, CP01 và CP02.
