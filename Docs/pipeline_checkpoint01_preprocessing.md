# Đặc Tả Pipeline Checkpoint 1: Tiền Xử Lý và Chuẩn Hóa Dữ Liệu

Tài liệu này đặc tả quy trình và các bước thực hiện tiền xử lý dữ liệu cho Checkpoint 1 trong bài toán phân loại phát ngôn thù ghét tiếng Việt (ViHSD). Mọi cơ sở lý thuyết và phân tích học thuật chi tiết được lưu trữ tại [Docs/roadmap.md](Docs/roadmap.md).

---

## 1. Mục Tiêu Kỹ Thuật

- Chuẩn hóa ngôn ngữ mạng xã hội (teencode, tiếng lóng, biến thể né kiểm duyệt, ký tự kéo dài).
- Bảo tồn thông tin đối tượng công kích (đại từ nhân xưng) và sắc thái phân cực cảm xúc (từ phủ định, emoji).
- Phân đoạn từ tiếng Việt tương thích với mô hình ngôn ngữ `vinai/phobert-base`.

---

## 2. Kiến Trúc Luồng Xử Lý (Pipeline Architecture)

```
[Dữ liệu thô: train.csv, dev.csv, test.csv]
                     |
                     v
  (Bước 1) Kiểm tra tính toàn vẹn và lọc chuỗi rỗng
                     |
                     v
  (Bước 2) Làm sạch URL, Email, HTML tags và chuẩn hóa khoảng trắng
                     |
                     v
  (Bước 3) Bảo tồn và chuẩn hóa sắc thái cảm xúc (Emoji 2 pha)
                     |
                     v
  (Bước 4) Chuẩn hóa Teencode & Từ lóng qua Từ điển (916 cặp từ)
                     |
                     v
  (Bước 5) Thu gọn ký tự lặp kéo dài (Lengthening Reduction)
                     |
                     v
  (Bước 6) Giữ nguyên Stopwords, đại từ nhân xưng và từ phủ định
                     |
                     v
  (Bước 7) Phân đoạn từ tiếng Việt (PyVi Word Segmentation)
                     |
                     v
[Dữ liệu đã chuẩn hóa: train_X, dev_X, test_X, nhãn y tương ứng]
```

---

## 3. Đặc Tả Dữ Liệu Đầu Vào và Đầu Ra

### 3.1. Dữ liệu đầu vào
- Tập huấn luyện: `data/train.csv` (24.048 mẫu).
- Tập kiểm định: `data/dev.csv` (2.672 mẫu).
- Tập kiểm tra: `data/test.csv` (6.680 mẫu).
- Cột dữ liệu: `free_text` (văn bản bình luận), `label_id` (nhãn 0, 1, 2).
- Tài nguyên chuẩn hóa: `data/teencode.csv` (916 cặp từ viết tắt, tiếng lóng) và `data/emojis.csv` (từ điển ánh xạ sắc thái emoji sang từ tố tiếng Việt).

### 3.2. Dữ liệu và Artifacts đầu ra
- Mảng văn bản đã phân đoạn từ: `train_X`, `dev_X`, `test_X`.
- Mảng nhãn nguyên tương ứng: `train_y`, `dev_y`, `test_y`.
- Tệp dự đoán chi tiết: `outputs/checkpoint01/predictions.csv` (các cột: `id`, `text`, `ground_truth`, `predicted_label`).
- Ma trận nhầm lẫn:
  - Biểu đồ nhiệt: `outputs/checkpoint01/confusion_matrix_checkpoint01.png`.
  - Tệp bảng tính dữ liệu: `outputs/checkpoint01/confusion_matrix.csv`.
- Tệp bảng điểm đánh giá:
  - Điểm số tổng hợp: `outputs/checkpoint01/evaluation_scores.csv` (danh sách đầy đủ các chỉ số cốt lõi: `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `micro_f1`, cùng precision/recall/f1 riêng cho từng lớp `clean`, `offensive`, `hate`).
  - Báo cáo phân loại chi tiết: `outputs/checkpoint01/classification_report.csv` (bảng chuẩn hóa: `class`, `precision`, `recall`, `f1_score`, `support` theo từng nhãn và các hàng accuracy, macro avg, weighted avg).

---

## 4. Các Bước Thực Hiện Tiền Xử Lý

### Bước 1: Kiểm tra tính toàn vẹn và xử lý chuỗi rỗng
1. Kiểm tra và loại bỏ các bản ghi chứa giá trị thiếu (`NaN`, `None`) trong cột `free_text`.
2. Lọc bỏ đồng thời cả văn bản $x$ và nhãn $y$ tương ứng nếu chuỗi rỗng sau làm sạch.

### Bước 2: Làm sạch URL, Email và thẻ HTML
1. Xóa đường dẫn web bằng biểu thức chính quy:
   ```python
   text = re.sub(r'https?://\S+|www\.\S+', '', text)
   ```
2. Xóa địa chỉ email và thẻ HTML:
   ```python
   text = re.sub(r'\S+@\S+', '', text)
   text = re.sub(r'<.*?>', '', text)
   ```
3. Chuẩn hóa nhiều khoảng trắng liên tiếp về một khoảng trắng đơn.

### Bước 3: Bảo tồn và chuẩn hóa sắc thái cảm xúc (Emoji)
1. **Bước 3.1 (Đệm khoảng trắng - Pad Emoji):** Thêm khoảng trắng trước và sau tất cả các ký tự emoji bằng biểu thức chính quy để tách rời emoji khỏi từ tố tiếng Việt liền kề, ngăn chặn lỗi dính từ khi tách từ bằng PyVi:
   ```python
   text = re.sub(r'(\s*)([\U00010000-\U0010ffff☀-➿])(\s*)', r' \2 ', text)
   ```
2. **Bước 3.2 (Ánh xạ ngữ nghĩa qua từ điển - Semantic Mapping):** Thay thế các emoji phổ biến mang sắc thái cảm xúc thành từ ngữ tiếng Việt tương thích với vốn từ vựng của PhoBERT:
   - Nhóm mỉa mai, chế giễu: `🙂` $\to$ `mặt_cười_mỉa_mai`, `😂`/`🤣` $\to$ `cười_ra_nước_mắt`.
   - Nhóm phẫn nộ, căm ghét: `😡`/`🤬` $\to$ `tức_giận`, `👿` $\to$ `ác_quỷ`.
   - Nhóm thô tục, xúc phạm: `🖕` $\to$ `ngón_tay_thối`, `🔞` $\to$ `khiêu_dâm`.
3. **Bước 3.3 (Bảo lưu Unicode dự phòng):** Giữ nguyên các ký tự emoji hiếm nằm ngoài bảng ánh xạ (đã được đệm khoảng trắng ở Bước 3.1) để hạn chế mất mát thông tin.

### Bước 4: Chuẩn hóa Teencode và tiếng lóng qua từ điển
1. Nạp bảng ánh xạ 916 cặp từ từ tập tin `data/teencode.csv`.
2. Áp dụng thay thế toàn bộ từ (word boundary) theo biểu thức chính quy:
   ```python
   pattern = re.compile(r'\b(' + '|'.join(map(re.escape, teencode_dict.keys())) + r')\b')
   text = pattern.sub(lambda m: teencode_dict[m.group(0)], text)
   ```
3. Chuẩn hóa đồng nhất các biến thể né kiểm duyệt (`vcl`, `vkl`, `vlon` $\to$ `vcl`; `đm`, `dmm`, `clmm` $\to$ `đm`; `dell`, `éo` $\to$ `đéo`).

### Bước 5: Thu gọn ký tự lặp kéo dài (Lengthening Reduction)
1. Rút gọn các ký tự chữ cái lặp lại liên tiếp từ 3 lần trở lên về 1 ký tự:
   ```python
   text = re.sub(r'([a-zA-Z])\1{2,}', r'\1', text)
   ```

### Bước 6: Giữ nguyên Stopwords, đại từ nhân xưng và từ phủ định
1. Tuyệt đối không áp dụng danh sách từ dừng (`vietnamese-stopwords.txt`).
2. Giữ nguyên toàn bộ đại từ nhân xưng (`mày`, `tao`, `nó`, `hắn`, `thằng`, `con`, `bọn`) và từ phủ định (`không`, `chẳng`, `chả`).

### Bước 7: Phân đoạn từ tiếng Việt (Word Segmentation)
1. Áp dụng công cụ tách từ PyVi trên toàn bộ chuỗi văn bản:
   ```python
   from pyvi import ViTokenizer
   text = ViTokenizer.tokenize(text)
   ```
2. Kiểm tra các từ ghép được nối thống nhất bằng ký tự gạch dưới (`_`).

---

## 5. Cấu Trúc Module Mã Nguồn

Pipeline được đóng gói tại [Pipeline/preprocessing.py](Pipeline/preprocessing.py) và script thực thi hàng loạt tại [Scripts/preprocess_data.py](Scripts/preprocess_data.py):

```python
class ViHSDPreprocessor:
    def __init__(self, teencode_path="data/teencode.csv", emoji_path="data/emojis.csv"):
        self.teencode_dict = self.load_teencode(teencode_path)
        self.emoji_dict = self.load_emoji_mapping(emoji_path)

    def load_teencode(self, path):
        pass

    def load_emoji_mapping(self, path):
        pass

    def clean_special_characters(self, text):
        pass

    def pad_and_map_emojis(self, text):
        pass

    def normalize_teencode(self, text):
        pass

    def reduce_lengthening(self, text):
        pass

    def segment_words(self, text):
        pass

    def transform_text(self, text):
        # Thực hiện tuần tự Bước 2 -> Bước 3 -> Bước 4 -> Bước 5 -> Bước 6 -> Bước 7
        pass

    def transform_dataset(self, df):
        pass
```

---

## 6. Quy Trình Kiểm Thử và Tiêu Chuẩn Nghiệm Thu

1. **Kiểm tra tính toàn vẹn (Assertion Checks):**
   - Không xuất hiện giá trị rỗng hoặc `NaN`: `assert sum(pd.isna(train_X)) == 0`.
   - Kích thước tập đặc trưng và tập nhãn khớp nhau: `assert len(train_X) == len(train_y)`.
2. **Nghiệm thu trên mô hình cơ sở:**
   - Thực thi notebook `Notebook/checkpoint01_data_preprocessing.ipynb`.
   - Huấn luyện theo cấu hình baseline (PhoBERT-base, `max_length = 100`, 2 epochs, CE loss tiêu chuẩn).
   - Xuất báo cáo đầy đủ: **Accuracy**, **Macro-Precision**, **Macro-Recall**, **Macro-F1**, **Micro-F1**, cùng Precision, Recall và F1 theo từng nhãn.
   - Chỉ số **Macro-F1 trên tập Test** phải ghi nhận mức tăng so với baseline ban đầu.
   - Xuất đầy đủ 5 tệp kết quả chuẩn hóa vào thư mục `outputs/checkpoint01/`.
