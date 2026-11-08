# Đề Xuất Các Hướng Cải Tiến Mô Hình Phát Hiện Phát Ngôn Thù Ghét (ViHSD)

Tài liệu này tổng hợp phân tích học thuật từ bài báo nghiên cứu _"A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts"_ (Lưu et al., 2021), rà soát hiện trạng thực nghiệm trong notebook `Paper/ViHSD_monolingual_Transformers.ipynb`, và đề xuất các phương án cải tiến mô hình toàn diện theo 4 trục kỹ thuật chính.

---

## I. Tổng quan Hiện trạng và Phân tích Điểm nghẽn

### 1. Đặc thù bài toán và tập dữ liệu ViHSD

Tập dữ liệu ViHSD gồm 33.400 bình luận thu thập từ Facebook và YouTube, được gán một trong ba nhãn:

- **0 - CLEAN:** Bình luận sạch, không mang tính thù ghét hay xúc phạm.
- **1 - OFFENSIVE:** Bình luận chứa từ ngữ thô tục, xúc phạm nhưng không nhắm cụ thể vào một cá nhân, nhóm người hoặc đối tượng đặc định.
- **2 - HATE:** Phát ngôn thù ghét nhắm trực diện vào cá nhân, tổ chức, tôn giáo, giới tính hoặc nguồn gốc xuất thân.

### 2. Hai điểm nghẽn kỹ thuật cốt lõi

1. **Mất cân bằng lớp nghiêm trọng (Severe Class Imbalance):**
   - Phân bố tập dữ liệu: CLEAN (~82.7%), HATE (~10.6%), OFFENSIVE (~6.7%).
   - Hệ quả: Khi huấn luyện bằng hàm mất mát tiêu chuẩn (Cross-Entropy Loss), mô hình bị chi phối hoàn toàn bởi lớp đa số. Kết quả thực nghiệm cho thấy Accuracy đạt mức cao (~86.8%), tuy nhiên Macro-F1 chỉ đạt mốc thấp (~60% - 62%). Ma trận nhầm lẫn (Confusion Matrix) phản ánh việc phần lớn các mẫu OFFENSIVE và HATE bị phân loại nhầm thành CLEAN.
2. **Đặc thù ngôn ngữ mạng xã hội không chuẩn tắc (Social Media Noise):**
   - Sử dụng dày đặc từ viết tắt, tiếng lóng, teencode đa dạng.
   - Sử dụng từ lóng thay thế nhằm vượt qua bộ lọc kiểm duyệt (ví dụ: _coin card, dell, éo, cc_).
   - Kéo dài ký tự để biểu đạt cảm xúc (_giiiiii, ừmmmmm, được lắmmmm_).
   - Sử dụng ngôn từ mang nghĩa bóng, châm biếm, mỉa mai (sarcasm).

---

## II. Các Trục Cải Tiến Chi Tiết

### 1. Trục Tiền xử lý và Chuẩn hóa Dữ liệu (Text Preprocessing & Normalization)

- **Loại bỏ việc lọc Stopwords:**
  - _Hạn chế hiện tại:_ Pipeline trong notebook sử dụng tập tin `vietnamese-stopwords.txt` để loại bỏ các từ dừng thông thường.
  - _Giải pháp cải tiến:_ Giữ nguyên toàn bộ cấu trúc văn bản. Trong tác vụ phát hiện phát ngôn thù ghét, đại từ nhân xưng (_mày, tao, nó, thằng, con, lũ, bọn_) và từ phủ định/nhấn mạnh (_không, chẳng, chả, quá, cực_) mang giá trị ngữ nghĩa quyết định trong việc xác định đối tượng công kích (target identification) và thái độ phát ngôn. Việc xóa stopwords phá vỡ cơ chế tự chú ý (Self-Attention) của Transformer và làm mất thông tin then chốt.
- **Tích hợp Từ điển Teencode chuẩn hóa (`data/teencode.csv`):**
  - _Hạn chế hiện tại:_ Pipeline gốc chưa có cơ chế chuẩn hóa viết tắt, dẫn đến việc các từ lóng thô tục bị tách rời hoặc xem là từ ngoài vốn từ (OOV).
  - _Giải pháp cải tiến:_ Sử dụng từ điển 916 cặp từ đã hợp nhất từ hai nguồn để thay thế từ viết tắt và từ né kiểm duyệt về dạng chuẩn trước khi phân đoạn từ.
- **Thu gọn ký tự lặp kéo dài (Lengthening Reduction):**
  - _Giải pháp cải tiến:_ Chuẩn hóa các từ có nguyên âm hoặc phụ âm lặp lại bất thường về tối đa 1 hoặc 2 ký tự (ví dụ: _ngonnnnn_ chuyển thành _ngon_, _dmmmm_ chuyển thành _dmm_).
- **Chuẩn hóa quy trình tách từ tiếng Việt (Word Segmentation):**
  - _Giải pháp cải tiến:_ Đối với kiến trúc PhoBERT, đầu vào bắt buộc phải được phân đoạn từ thành dạng từ ghép nối bằng dấu gạch dưới (`học_sinh`, `sinh_viên`). Cần áp dụng PyVi đồng bộ trên toàn bộ tập train, dev và test.
- **Sửa lỗi logic làm sạch chuỗi rỗng:**
  - _Hạn chế hiện tại:_ Hàm `np.delete` trên kiểu danh sách (list) không làm biến đổi danh sách gốc.
  - _Giải pháp cải tiến:_ Lọc chuỗi rỗng thông qua list comprehension hoặc phương thức chuyên dụng của DataFrame để đảm bảo tính nhất quán giữa dữ liệu đặc trưng $X$ và nhãn $y$.

---

### 2. Trục Xử lý Mất cân bằng Lớp (Handling Class Imbalance)

- **Hàm mất mát Cross-Entropy có trọng số (Weighted Cross-Entropy Loss):**
  - Thiết lập vector trọng số nghịch đảo với tần suất xuất hiện của từng lớp:
    $$w_c = \frac{N}{K \cdot N_c}$$
    Trong đó $N$ là tổng số mẫu, $K=3$ là số lớp, và $N_c$ là số mẫu thuộc lớp $c$.
  - Trọng số này phạt mô hình nặng hơn khi dự đoán sai lớp thiểu số (HATE, OFFENSIVE), buộc mô hình chú trọng vào các mẫu khó.
- **Focal Loss:**
  - Áp dụng Focal Loss nhằm điều chỉnh giảm trọng số đóng góp của các mẫu dễ phân loại (chủ yếu là lớp CLEAN) và tập trung gradient vào các mẫu khó:
    $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
    Với $\gamma \in [1.5, 2.5]$ là tham số điều tiết độ tập trung và $\alpha_t$ là trọng số cân bằng lớp.

---

### 3. Trục Kiến trúc Mô hình (Model Architecture)

- **Nâng cấp Mô hình nền tảng (Backbone Upgrades):**
  - Thử nghiệm `vinai/phobert-large` (24 layers, 1024 hidden size) để nâng cao khả năng khái quát hóa và nắm bắt ngữ cảnh phức tạp so với `vinai/phobert-base`.
- **Thiết kế Đầu phân loại nâng cao (Custom Classification Head):**
  - _Hạn chế hiện tại:_ Chỉ sử dụng biểu diễn của token `<s>` đưa qua một tầng tuyến tính đơn lẻ.
  - _Giải pháp cải tiến:_
    - **Multi-layer Pooling:** Trích xuất và kết hợp biểu diễn từ 4 tầng ẩn cuối cùng của Transformer (Mean-pooling hoặc Concatenation) để giữ lại cả đặc trưng ngữ pháp tầng thấp lẫn đặc trưng ngữ nghĩa tầng cao.
    - **Transformer + BiLSTM Head:** Đưa chuỗi hidden states của câu qua tầng BiLSTM hai chiều kèm Attention Pooling nhằm nắm bắt tốt hơn các phụ thuộc ngữ cảnh cục bộ giữa các cụm từ xúc phạm.

---

### 4. Trục Chiến lược Huấn luyện và Đánh giá (Training & Optimization)

- **Cơ chế Checkpoint tối ưu hóa theo Macro-F1:**
  - Cấu hình Trainer đánh giá định kỳ sau mỗi epoch (`evaluation_strategy="epoch"`).
  - Tự động lưu và nạp lại checkpoint đạt điểm Macro-F1 cao nhất trên tập validation (`load_best_model_at_end=True`, `metric_for_best_model="eval_macro_f1"`), khắc phục tình trạng dừng ở checkpoint cuối cùng bị overfitting.
- **Tối ưu hóa số epochs và Early Stopping:**
  - Nâng số epochs huấn luyện lên 8 epochs.
  - Kích hoạt `EarlyStoppingCallback` (dừng nếu sau 2 - 3 epochs chỉ số Macro-F1 không cải thiện) để tiết kiệm tài nguyên tính toán và chống quá khớp.
- **Phân tầng tốc độ học (Differential Learning Rate):**
  - Thiết lập tốc độ học nhỏ ($1.5 \times 10^{-5}$ đến $2 \times 10^{-5}$) cho các trọng số pre-trained của Transformer để bảo tồn tri thức ngôn ngữ sẵn có.
  - Thiết lập tốc độ học lớn hơn ($1 \times 10^{-4}$) cho Classification Head mới khởi tạo.
  - Sử dụng Cosine Annealing Learning Rate Scheduler để mô hình hội tụ mượt mà hơn.
- **Mở rộng độ dài chuỗi tối đa (`max_length = 256`):**
  - Nâng ngưỡng cắt chuỗi từ 100 lên 256 nhằm tận dụng tối đa giới hạn vị trí nhúng của PhoBERT (`max_position_embeddings = 256`).
  - Hạn chế mất mát ngữ cảnh đối với các bình luận mạng xã hội dài, kết hợp kỹ thuật Mixed Precision (`fp16=True`) và Gradient Accumulation để kiểm soát bộ nhớ GPU.
