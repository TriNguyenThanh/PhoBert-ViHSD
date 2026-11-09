# Kế Hoạch Nghiên Cứu và Lộ Trình Thực Nghiệm (Research Roadmap)

Dự án: Nâng cao hiệu năng mô hình phát hiện phát ngôn thù ghét trên mạng xã hội tiếng Việt (ViHSD - Vietnamese Hate Speech Detection).

Tài liệu này xác lập lộ trình kỹ thuật và các mốc thực nghiệm (Checkpoints) có tính kế thừa, hướng đến việc giải quyết hai thách thức cốt lõi: đặc thù nhiễu ngôn ngữ mạng xã hội và hiện tượng mất cân bằng lớp nghiêm trọng trong tập dữ liệu ViHSD.

---

## Tổng Quan Bài Toán và Mục Tiêu Đánh Giá

- **Đặc tả tác vụ:** Phân loại văn bản 3 lớp trên bình luận mạng xã hội tiếng Việt:
  - Nhãn `0` - CLEAN: Bình luận sạch, không chứa nội dung độc hại (~82.7% tập dữ liệu).
  - Nhãn `1` - OFFENSIVE: Bình luận chứa từ ngữ thô tục, xúc phạm nhưng không nhắm vào đối tượng cụ thể (~6.7% tập dữ liệu).
  - Nhãn `2` - HATE: Phát ngôn thù ghét trực diện nhắm vào cá nhân, tổ chức, nhóm người hoặc cộng đồng (~10.6% tập dữ liệu).
- **Thước đo đánh giá chính:** **Macro-F1 Score**. Do tỷ lệ phân bố giữa lớp đa số (CLEAN) và lớp thiểu số (OFFENSIVE, HATE) có sự chênh lệch lớn, chỉ số Accuracy thông thường không phản ánh chính xác năng lực phát hiện của mô hình.
- **Mô hình đường cơ sở (Baseline):** Kiến trúc `vinai/phobert-base` được huấn luyện theo cấu hình trong nghiên cứu gốc (Lưu et al., 2021) với hàm mất mát Cross-Entropy tiêu chuẩn.

---

## Chi Tiết Các Checkpoint Kỹ Thuật

```
+-----------------------------------------------------------------------+
| Checkpoint 1: Tiền xử lý và Chuẩn hóa Dữ liệu (Preprocessing)         |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| Checkpoint 2: Tối ưu Chiến lược Huấn luyện (Training Strategy)        |
+-----------------------------------------------------------------------+
                                   |
       +---------------------------+---------------------------+
       |                                                       |
       v                                                       v
+-----------------------------+         +-------------------------------+
| Checkpoint 3: Focal Loss    |         | Checkpoint 4: Class Weight    |
+-----------------------------+         +-------------------------------+
       |                                                       |
       +---------------------------+---------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| Checkpoint 5: Kết hợp Focal Loss và Class Weight (Combined Strategy)  |
+-----------------------------------------------------------------------+
```

---

### Checkpoint 1: Triển Khai Các Phương Pháp Tiền Xử Lý Dữ Liệu

Mục tiêu của Checkpoint 1 là khắc phục các hạn chế trong pipeline tiền xử lý gốc, làm sạch nhiễu ngôn ngữ mạng xã hội và bảo toàn tối đa thông tin ngữ nghĩa phục vụ việc phát hiện đối tượng công kích.

#### 1. Các nội dung triển khai chi tiết

1. **Bảo tồn toàn bộ cấu trúc câu và từ dừng (Stopwords Retention):**
   - Không áp dụng tập tin từ dừng (`vietnamese-stopwords.txt`).
   - Giữ nguyên các đại từ nhân xưng (`mày`, `tao`, `nó`, `thằng`, `con`, `bọn`, `chúng nó`) vì đây là dấu hiệu phân biệt trực tiếp giữa hành vi xúc phạm vu vơ (OFFENSIVE) và thù ghét nhắm mục tiêu (HATE).
   - Giữ nguyên từ phủ định và từ nhấn mạnh (`không`, `chẳng`, `chả`, `rất`, `cực`) nhằm bảo toàn cấu trúc phân cực ngữ nghĩa cho cơ chế Self-Attention của Transformer.

2. **Chuẩn hóa từ viết tắt, tiếng lóng và biến thể né kiểm duyệt (Teencode Normalization):**
   - Tích hợp từ điển teencode hợp nhất 916 mục từ chuẩn (`data/teencode.csv` và `data/teencode.json`).
   - Thay thế các biến thể viết tắt và ký tự né bộ lọc (ví dụ: `vcl`, `cc`, `đm`, `dmm`, `dell`, `éo`) về dạng từ tố ngữ nghĩa đồng nhất trước khi đưa vào bộ tách từ.

3. **Thu gọn ký tự kéo dài biểu đạt cảm xúc (Lengthening Reduction):**
   - Nhận diện và quy đổi các chuỗi ký tự lặp bất thường từ 3 lần trở lên về độ dài tiêu chuẩn (tối đa 1 hoặc 2 ký tự, ví dụ: `ngonnnnn` thành `ngon`, `chánnnnnn` thành `chán`).

4. **Bảo tồn và chuẩn hóa sắc thái cảm xúc (Vấn đề xử lý Emoji):**
   - **Nghịch lý trong nghiên cứu cơ sở:**
     - Trong bài báo gốc (*Lưu et al., 2021, Mục 5*), nhóm tác giả khẳng định emoji là đặc trưng phân cực then chốt giúp nhận diện sắc thái tiêu cực hoặc công kích. Tuy nhiên, trong mã nguồn notebook gốc (`Paper/ViHSD_monolingual_Transformers.ipynb`), hàm `deEmojify` lại loại bỏ toàn bộ emoji bằng biểu thức chính quy.
     - Hậu quả: Mô hình cơ sở bị tước đoạt hoàn toàn các tín hiệu cảm xúc phân cực, dẫn đến việc bỏ sót các phát ngôn thù ghét tinh vi.
   - **Đặc thù thực tế trên tập dữ liệu ViHSD:**
     - Thống kê cho thấy hơn 11% bình luận trong ViHSD có chứa emoji (tập Train: 11.38%, tập Dev: 11.04%, tập Test: 11.66%).
     - Trong các mẫu OFFENSIVE và HATE, emoji thường đóng vai trò biểu đạt thái độ mỉa mai, châm biếm (sarcasm) như `😂`, `🤣`, `😏`, thái độ công kích thụ động kinh điển trên mạng xã hội Việt Nam như biểu tượng mặt cười đơ `🙂`, hoặc ác cảm cực đoan như `😡`, `🤬`, `🖕`. Khi loại bỏ emoji, các câu như *"Lan Hoa mấy đứa suốt ngày bảo người ta đạo này nọ cả đời chỉ núp dưới bóng của người khác 🙂"* sẽ mất hoàn toàn tính mỉa mai công kích.
   - **Hai rào cản kỹ thuật cốt lõi:**
     1. *Lỗi dính từ với PyVi:* Khi người dùng viết liền không khoảng cách (`ngu🙂`, `chó😂😂`), công cụ tách từ PyVi sẽ ghép sai emoji thành từ ghép lạ dạng `ngu_🙂`.
     2. *Lỗi ngoài vốn từ (Out-of-Vocabulary - OOV) của PhoBERT:* PhoBERT được tiền huấn luyện trên báo chí chính thống (VnExpress, Dân Trí) và Wikipedia nên từ điển FastBPE 64K không chứa mã Unicode của hầu hết các emoji. Khi đưa trực tiếp vào PhoBERT Tokenizer, emoji bị phân mảnh thành nhiều byte-token rời rạc hoặc chuyển thành `<unk>`, làm loãng độ dài chuỗi và triệt tiêu ngữ nghĩa.
   - **Giải pháp kỹ thuật (Quy trình 2 pha):**
     1. *Pha 1 (Đệm khoảng trắng):* Thêm khoảng trắng trước và sau emoji để bảo vệ hình thái từ tiếng Việt trước khi đưa qua PyVi.
     2. *Pha 2 (Ánh xạ ngữ nghĩa - Semantic Text Mapping):* Ánh xạ các emoji phổ biến thành các từ tố tiếng Việt có sẵn trong từ điển PhoBERT (ví dụ: `🙂` $\to$ `mặt_cười_mỉa_mai`, `😂` $\to$ `cười_ra_nước_mắt`, `😡` $\to$ `tức_giận`, `🖕` $\to$ `ngón_tay_thối`). Giải pháp này tương thích 100% với vốn từ vựng của PhoBERT mà không cần mở rộng kiến trúc embedding. Đối với emoji hiếm nằm ngoài từ điển ánh xạ, giữ lại dạng Unicode có khoảng cách để hạn chế mất mát thông tin.

5. **Đồng bộ hóa quy trình phân đoạn từ tiếng Việt (Word Segmentation):**
   - Áp dụng công cụ tách từ PyVi (`ViTokenizer`) một cách nhất quán trên toàn bộ các tập dữ liệu `train.csv`, `dev.csv` và `test.csv`.
   - Đảm bảo các từ ghép tiếng Việt được liên kết bằng ký tự gạch dưới (`_`) đúng theo quy cách đầu vào của PhoBERT.

6. **Làm sạch chuỗi rỗng và kiểm tra tính nhất quán:**
   - Xử lý triệt để các mẫu chỉ chứa khoảng trắng hoặc ký tự đặc biệt sau chuẩn hóa.
   - Đảm bảo tính nhất quán tuyệt đối về chỉ số giữa mảng đặc trưng $X$ và nhãn $y$.

#### 2. Tiêu chí nghiệm thu Checkpoint 1

- Tạo lập module tiền xử lý độc lập trong `Pipeline/preprocessing.py` hoặc `Scripts/preprocess_data.py`.
- Xuất dữ liệu đã xử lý và kiểm tra độ vẹn toàn (không còn giá trị rỗng, nhãn đúng định dạng số nguyên).
- Huấn luyện mô hình cơ sở trên dữ liệu mới; đánh giá sự cải thiện của Macro-F1 so với dữ liệu thô ban đầu.

---

### Checkpoint 2: Thay Đổi Chiến Lược Huấn Luyện

Mục tiêu của Checkpoint 2 là tái cấu trúc toàn diện quy trình huấn luyện và tối ưu hóa siêu tham số, đảm bảo mô hình hội tụ ổn định và được căn chỉnh trực tiếp theo thước đo Macro-F1 thay vì Loss hay Accuracy.

#### 1. Các nội dung triển khai chi tiết

1. **Cơ chế lưu Checkpoint tối ưu theo Macro-F1:**
   - Cấu hình tiến trình kiểm thử định kỳ sau mỗi epoch trên tập validation: `eval_strategy="epoch"`, `save_strategy="epoch"`.
   - Kích hoạt cơ chế nạp lại trọng số tối ưu nhất: `load_best_model_at_end=True`, `metric_for_best_model="eval_macro_f1"`, `greater_is_better=True`.
   - Khắc phục tình trạng mô hình bị suy giảm hiệu năng ở các epoch cuối do overfitting vào lớp đa số.

2. **Gia tăng số lượng epoch và kích hoạt Early Stopping:**
   - Tăng số epoch huấn luyện tối đa lên 8 đến 10 epochs.
   - Tích hợp `EarlyStoppingCallback` với ngưỡng kiên nhẫn (`patience = 2` hoặc `3` epochs dựa trên chỉ số `eval_macro_f1`), tự động dừng khi mô hình không còn cải thiện để tối ưu tài nguyên tính toán.

3. **Phân tầng tốc độ học (Differential Learning Rates):**
   - Áp dụng tốc độ học nhỏ ($1.5 \times 10^{-5}$ đến $2 \times 10^{-5}$) cho các tầng biểu diễn tiền huấn luyện của PhoBERT nhằm duy trì tri thức ngôn ngữ chung.
   - Áp dụng tốc độ học lớn hơn ($1 \times 10^{-4}$) cho tầng phân loại (Classification Head) mới được khởi tạo ngẫu nhiên.

4. **Lịch trình suy giảm tốc độ học (Learning Rate Scheduler):**
   - Sử dụng Cosine Annealing Scheduler kèm giai đoạn khởi động tuyến tính (Linear Warmup chiếm 10% tổng số bước huấn luyện) giúp ổn định gradient trong các bước cập nhật đầu tiên.

5. **Mở rộng độ dài chuỗi tối đa (`max_length = 256`):**
   - Nâng ngưỡng cắt chuỗi từ `max_length = 100` (ở notebook cơ sở) lên `max_length = 256`.
   - Khai thác tối đa giới hạn vị trí nhúng ngữ cảnh của kiến trúc `vinai/phobert-base` (`max_position_embeddings = 256`).
   - Khắc phục hiện tượng cắt cụt (truncation) các bình luận dài trên mạng xã hội, đảm bảo không bỏ sót các mệnh đề thù ghét hoặc đối tượng công kích thường xuất hiện ở phần sau của câu.

6. **Tối ưu hóa tài nguyên phần cứng khi mở rộng ngữ cảnh:**
   - Do độ phức tạp tính toán và tiêu thụ bộ nhớ của cơ chế Self-Attention tăng theo bình phương độ dài chuỗi ($O(L^2)$), kích hoạt chế độ Mixed Precision Training (`fp16=True`) để tăng tốc xử lý và tiết kiệm VRAM.
   - Thiết lập kích thước batch (`per_device_train_batch_size = 16`) kết hợp kỹ thuật tích lũy gradient (`gradient_accumulation_steps = 2`) để duy trì kích thước batch hiệu dụng ổn định mà không vượt quá giới hạn bộ nhớ GPU (CUDA Out-of-Memory).

#### 2. Tiêu chí nghiệm thu Checkpoint 2

- Thiết lập module cấu hình huấn luyện chuẩn trong `Pipeline/train.py`.
- Trích xuất biểu đồ hội tụ (Learning Curves) thể hiện rõ giá trị Loss và Macro-F1 qua từng epoch.
- Xác định điểm dừng tối ưu trên tập dev, ghi nhận mức tăng trưởng Macro-F1 so với mô hình từ Checkpoint 1.

---

### Checkpoint 3: Xử Lý Mất Cân Bằng Lớp Bằng Focal Loss

Mục tiêu của Checkpoint 3 là triệt tiêu hiện tượng lớp đa số (CLEAN) áp đảo gradient bằng cách hạ thấp trọng số đóng góp của các mẫu dễ phân loại và tập trung vào các mẫu khó thuộc lớp thiểu số.

#### 1. Các nội dung triển khai chi tiết

1. **Cơ sở toán học của Focal Loss:**
   Hàm mất mát Focal Loss chuẩn được định nghĩa:
   $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   Trong đó:
   - $p_t$ là xác suất dự đoán của mô hình đối với nhãn thực tế.
   - $(1 - p_t)^\gamma$ là hệ số điều chế (Modulating Factor). Khi một mẫu được phân loại với độ tin cậy cao ($p_t \to 1$), hệ số này tiến về 0, làm giảm đáng kể giá trị tổn thất.
   - $\gamma \ge 0$ là tham số điều chỉnh độ tập trung (Focusing Parameter). Khi $\gamma = 0$, hàm trở về Cross-Entropy tiêu chuẩn.
   - Ở Checkpoint này, thiết lập $\alpha_t = 1$ để đánh giá độc lập tác động của cơ chế điều chế mẫu khó mà không bị chi phối bởi trọng số lớp tĩnh.

2. **Cài đặt kỹ thuật trong PyTorch:**
   - Xây dựng lớp `FocalLoss(torch.nn.Module)` hỗ trợ tính toán gradient trên Tensor nhiều chiều và tương thích với hàm softmax 3 lớp.
   - Tùy biến lớp `Trainer` của HuggingFace thông qua việc ghi đè (override) phương thức `compute_loss(model, inputs, return_outputs=False)`.

3. **Khảo sát siêu tham số:**
   - Thực nghiệm quét lưới (Grid Search) đối với tham số $\gamma \in \{1.0, 1.5, 2.0, 2.5\}$.
   - Phân tích sự dịch chuyển của hàm tổn thất và mức độ hội tụ của từng lớp.

#### 2. Tiêu chí nghiệm thu Checkpoint 3

- Module `Pipeline/losses.py` chứa định nghĩa hoàn chỉnh của `FocalLoss`.
- Bảng thực nghiệm so sánh các giá trị $\gamma$ khác nhau.
- Ma trận nhầm lẫn (Confusion Matrix) phản ánh sự suy giảm số lượng mẫu OFFENSIVE và HATE bị phân loại nhầm sang CLEAN.

---

### Checkpoint 4: Xử Lý Mất Cân Bằng Lớp Bằng Class Weight

Mục tiêu của Checkpoint 4 là điều chỉnh độ dốc gradient thông qua việc gán trọng số phạt tĩnh tỉ lệ nghịch với tần suất xuất hiện của từng lớp trong tập huấn luyện.

#### 1. Các nội dung triển khai chi tiết

1. **Phương pháp tính toán Vector trọng số:**
   - **Nghịch đảo tần suất lớp (Inverse Class Frequency - ICF):**
     $$w_c = \frac{N}{K \cdot N_c}$$
     Trong đó:
     - $N$ là tổng số mẫu trong tập huấn luyện.
     - $K = 3$ là số lớp phân loại.
     - $N_c$ là số mẫu thuộc lớp $c$.
   - **Làm mịn căn bậc hai (Smoothed Inverse Frequency):**
     $$w_c = \sqrt{\frac{N}{N_c}}$$
     Phương pháp này tránh tình trạng trọng số của lớp thiểu số quá lớn gây bất ổn định gradient.
   - **Số lượng mẫu hiệu dụng (Effective Number of Samples - Cui et al., 2019):**
     $$E_{N_c} = \frac{1 - \beta^{N_c}}{1 - \beta}, \quad w_c \propto \frac{1}{E_{N_c}}$$
     Với $\beta \in [0.99, 0.9999]$, mô phỏng sự suy giảm thông tin biên khi số lượng mẫu tăng lên.

2. **Cài đặt kỹ thuật:**
   - Tính toán vector trọng số chuẩn hóa $W = [w_0, w_1, w_2]$ từ tập `train.csv`.
   - Đưa vector trọng số vào hàm mất mát Cross-Entropy chuẩn của PyTorch:
     $$\mathcal{L}_{WCE} = -\sum_{c=0}^{2} w_c y_c \log(\hat{y}_c)$$
     thông qua `torch.nn.CrossEntropyLoss(weight=class_weights)`.

3. **Đánh giá sự cân bằng Precision và Recall:**
   - Phân tích chi tiết sự đánh đổi giữa Recall của lớp OFFENSIVE/HATE và Precision của lớp CLEAN.
   - Xác định phương thức gán trọng số tối ưu nhất cho tập dữ liệu ViHSD.

#### 2. Tiêu chí nghiệm thu Checkpoint 4

- Script tính toán trọng số tự động dựa trên dữ liệu đầu vào trong `Scripts/compute_class_weights.py`.
- Tích hợp thành công Class Weight vào quá trình huấn luyện của HuggingFace `Trainer`.
- Bảng dữ liệu đối sánh hiệu quả giữa các công thức tính trọng số khác nhau trên tập validation và test.

---

### Checkpoint 5: Xử Lý Mất Cân Bằng Lớp Bằng Cả Hai Phương Pháp

Mục tiêu của Checkpoint 5 là tổng hợp đồng thời hai cơ chế: phân bổ trọng số phạt theo phân bố lớp (Class Weight) và triệt tiêu đóng góp của các mẫu dễ phân loại (Focal Loss) nhằm đạt được năng lực phân loại tối ưu trên toàn bộ các nhãn.

#### 1. Các nội dung triển khai chi tiết

1. **Thiết lập toán học của Weighted Focal Loss:**
   Hàm mất mát kết hợp được biểu diễn dưới dạng:
   $$FL_{combined}(p_t) = -\alpha_c (1 - p_t)^\gamma \log(p_t)$$
   Trong đó:
   - $\alpha_c$ là vector trọng số lớp tĩnh thu được từ phương pháp tối ưu nhất ở Checkpoint 4 ($c \in \{0, 1, 2\}$).
   - $\gamma$ là tham số điều chế mẫu khó được xác định từ Checkpoint 3 (thông thường $\gamma \in [1.5, 2.0]$).

2. **Cân chỉnh tương tác giữa tham số:**
   - Khi kết hợp cả $\alpha_c$ và $(1 - p_t)^\gamma$, hàm mất mát có thể phạt rất nặng các trường hợp dự đoán sai nhãn thiểu số.
   - Tiến hành chuẩn hóa vector $\alpha_c$ về khoảng giá trị thích hợp (tổng bằng 1 hoặc trung bình bằng 1) để tránh làm bão hòa gradient và gây suy giảm đột ngột chỉ số Precision của lớp CLEAN.

3. **Nghiên cứu bóc tách thành phần (Ablation Study):**
   - Tiến hành thực nghiệm so sánh trên cùng một tập dữ liệu tiền xử lý (Checkpoint 1) và cùng một chiến lược huấn luyện (Checkpoint 2).
   - So sánh 4 biến thể hàm mất mát:
     1. Standard Cross-Entropy.
     2. Focal Loss thuần túy ($\gamma > 0, \alpha = 1$).
     3. Weighted Cross-Entropy thuần túy ($\gamma = 0, \alpha = W$).
     4. Weighted Focal Loss kết hợp ($\gamma > 0, \alpha = W$).

4. **Phân tích lỗi định tính (Error Analysis):**
   - Trích xuất danh sách các mẫu dự đoán sai giữa các cặp nhãn: CLEAN - OFFENSIVE, OFFENSIVE - HATE, CLEAN - HATE.
   - Đánh giá vai trò của việc kết hợp hàm mất mát trong việc phân biệt ranh giới mờ giữa phát ngôn thô tục thông thường và phát ngôn mang tính thù ghét.

#### 2. Tiêu chí nghiệm thu Checkpoint 5

- Lớp `WeightedFocalLoss` hoàn chỉnh, có khả năng cấu hình linh hoạt vector $\alpha$ và tham số $\gamma$.
- Báo cáo đối sánh tổng hợp hoàn thiện với đầy đủ các bảng chỉ số trên tập Test độc lập.
- Tổng kết toàn diện tác động của từng cải tiến lên chỉ số Macro-F1.

---

### Checkpoint 6: Khảo Sát Kiến Trúc Mô Hình Nền Tảng Mới (Alternative Backbone Model Architecture)

Mục tiêu của Checkpoint 6 là kiểm chứng giả thuyết về năng lực biểu diễn của mô hình nền tảng: liệu việc nâng cấp kiến trúc backbone lên quy mô lớn hơn (`vinai/phobert-large` với 370M tham số, 24 transformer layers) hoặc sử dụng kiến trúc Transformer đa ngôn ngữ (`xlm-roberta-base`) có tạo ra bước đột phá về khả năng nhận diện các dạng phát ngôn thù ghét tinh vi, ẩn ý và ngữ cảnh phức tạp so với `vinai/phobert-base` hay không.

#### 1. Các nội dung triển khai chi tiết

1. **Khảo sát Mô hình nền tảng PhoBERT-large:**
   - **Quy mô kiến trúc:** 24 layers, 16 attention heads, hidden dimension $d = 1024$ (so với 12 layers, 768 hidden dimension của PhoBERT-base).
   - **Không gian vốn từ (Vocabulary):** Đồng bộ không gian từ vựng và thuật toán tách từ BPE với `vinai/phobert-base`, tái sử dụng trọn vẹn đường ống tiền xử lý chuẩn hóa tiếng Việt từ Checkpoint 1.
   - **Năng lực biểu diễn:** Nâng cao độ nhạy đối với các bình luận châm biếm, mỉa mai, ngữ nghĩa ẩn dụ (sarcasm / veiled hate) nhờ số lượng tham số lớn gấp gần 3 lần ($370\text{M}$ vs $135\text{M}$).

2. **Tối ưu hóa tài nguyên phần cứng cho Mô hình quy mô lớn:**
   - Khi tăng kích thước mô hình lên gấp gần 3 lần và duy trì `max_length = 256`, tiêu thụ bộ nhớ VRAM của GPU tăng vọt.
   - Điều chỉnh cấu hình batch size và tích lũy gradient:
     - `per_device_train_batch_size = 8`
     - `gradient_accumulation_steps = 4`
     - Duy trì kích thước batch hiệu dụng: $8 \times 4 = 32$, đảm bảo tính tương đồng về động học tối ưu hóa so với các Checkpoint trước mà không gây tràn bộ nhớ (CUDA OOM) trên GPU Google Colab (T4 / V100 16GB).
   - Kích hoạt Mixed Precision Training (`fp16 = True`).

3. **Cân chỉnh Tốc độ học phân tầng cho Kiến trúc sâu (24 Layers):**
   - Áp dụng tốc độ học nhỏ hơn cho backbone: $\text{lr}_{\text{backbone}} = 1 \times 10^{-5}$ (thay vì $2 \times 10^{-5}$) nhằm ngăn hiện tượng vỡ biểu diễn (catastrophic forgetting) của 24 tầng transformer đã tiền huấn luyện.
   - Giữ nguyên tốc độ học cho Classification Head: $\text{lr}_{\text{head}} = 1 \times 10^{-4}$.
   - Áp dụng lịch trình Cosine Annealing Scheduler kèm Linear Warmup 10% tổng số bước.

4. **Tích hợp Hàm mất mát Tối ưu:**
   - Kết hợp mô hình mới với hàm mất mát giải quyết mất cân bằng lớp tối ưu đã kiểm chứng từ Checkpoint 3 / Checkpoint 5 (Focal Loss với $\gamma = 2.0$ hoặc Weighted Focal Loss).
   - Tự động nạp checkpoint tốt nhất theo `eval_macro_f1` với cơ chế Early Stopping (`patience = 2`).

#### 2. Tiêu chí nghiệm thu Checkpoint 6

- Module huấn luyện và notebook `Notebook/checkpoint06_alternative_model.ipynb` hoàn toàn tự chứa (100% self-contained).
- Hoàn thành so sánh trực tiếp hiệu năng giữa `vinai/phobert-base` và mô hình nền tảng mới trên cùng tập kiểm tra chuẩn `test.csv`.
- Phân tích chi tiết sự thay đổi chỉ số F1 trên từng lớp nhãn (đặc biệt là nhãn HATE và OFFENSIVE).
- Xuất đầy đủ 5 tệp kết quả đầu ra chuẩn hóa vào `outputs/checkpoint06/`.

---

## Ma Trận Thực Nghiệm Đối Sánh (Ablation Matrix)

Bảng đối sánh tổng hợp sau đây được thiết kế để theo dõi tiến trình thực nghiệm qua từng Checkpoint:

| Chỉ số / Cấu hình             | Checkpoint 0 (Baseline) |   Checkpoint 1    |     Checkpoint 2      |      Checkpoint 3       |      Checkpoint 4       |      Checkpoint 5       |      Checkpoint 6 (Model Khác)       |
| :---------------------------- | :---------------------: | :---------------: | :-------------------: | :---------------------: | :---------------------: | :---------------------: | :----------------------------------: |
| **Mô hình nền tảng**          |      PhoBERT-base       |   PhoBERT-base    |     PhoBERT-base      |      PhoBERT-base       |      PhoBERT-base       |      PhoBERT-base       |       **PhoBERT-large** (370M)       |
| **Tiền xử lý dữ liệu**        |   Gốc (lọc Stopwords)   |   Đã chuẩn hóa    |     Đã chuẩn hóa      |      Đã chuẩn hóa       |      Đã chuẩn hóa       |      Đã chuẩn hóa       |             Đã chuẩn hóa             |
| **Độ dài chuỗi (max_length)** |           100           |        100        |          256          |           256           |           256           |           256           |                 256                  |
| **Chiến lược huấn luyện**     |    3 Epochs, CE Loss    | 3 Epochs, CE Loss | 8 Epochs + Best Model |  8 Epochs + Best Model  |  8 Epochs + Best Model  |  8 Epochs + Best Model  | 8 Epochs + Diff LR (BS 8 x GA 4 = 32)|
| **Hàm mất mát**               |      Cross-Entropy      |   Cross-Entropy   |     Cross-Entropy     | Focal Loss ($\gamma=2$) | Weighted CE ($W_{ICF}$) |   Weighted Focal Loss   |       Focal / Weighted Loss          |
| **Class Weight ($\alpha$)**   |          Không          |       Không       |         Không         |          Không          |           Có            |           Có            |                 Tùy chọn             |
| **Focal Factor ($\gamma$)**   |      $\gamma = 0$       |   $\gamma = 0$    |     $\gamma = 0$      |     $\gamma = 2.0$      |      $\gamma = 0$       | $\gamma \in [1.5, 2.0]$ |            $\gamma = 2.0$            |
| **Validation Macro-F1**       |        Ghi nhận         |     Ghi nhận      |       Ghi nhận        |        Ghi nhận         |        Ghi nhận         |        Ghi nhận         |               Ghi nhận               |
| **Test Macro-F1**             |        Ghi nhận         |     Ghi nhận      |       Ghi nhận        |        Ghi nhận         |        Ghi nhận         |        Ghi nhận         |               Ghi nhận               |
| **F1 - Lớp CLEAN (0)**        |        Ghi nhận         |     Ghi nhận      |       Ghi nhận        |        Ghi nhận         |        Ghi nhận         |        Ghi nhận         |               Ghi nhận               |
| **F1 - Lớp OFFENSIVE (1)**    |        Ghi nhận         |     Ghi nhận      |       Ghi nhận        |        Ghi nhận         |        Ghi nhận         |        Ghi nhận         |               Ghi nhận               |
| **F1 - Lớp HATE (2)**         |        Ghi nhận         |     Ghi nhận      |       Ghi nhận        |        Ghi nhận         |        Ghi nhận         |        Ghi nhận         |               Ghi nhận               |

---

## Tổ Chức Thư Mục Mã Nguồn Triển Khai

Mỗi Checkpoint sẽ được ánh xạ vào cấu trúc mô-đun của kho mã nguồn theo quy chuẩn sau:

```
PhoBert-ViHSD/
|-- Docs/
|   |-- topics.md                                   # Phân tích học thuật bài báo gốc và ý tưởng
|   |-- roadmap.md                                  # Kế hoạch tổng thể 6 Checkpoints
|   |-- checkpoint_implementation_guide.md          # Hướng dẫn quy chuẩn triển khai kỹ thuật
|   |-- pipeline_checkpoint01_preprocessing.md      # Đặc tả kỹ thuật Pipeline Checkpoint 1
|   |-- pipeline_checkpoint02_training.md           # Đặc tả kỹ thuật Pipeline Checkpoint 2
|   |-- pipeline_checkpoint03_focal_loss.md         # Đặc tả kỹ thuật Pipeline Checkpoint 3
|   |-- pipeline_checkpoint04_class_weights.md      # Đặc tả kỹ thuật Pipeline Checkpoint 4
|   |-- pipeline_checkpoint05_combined.md           # Đặc tả kỹ thuật Pipeline Checkpoint 5
|   `-- pipeline_checkpoint06_alternative_backbone.md # Đặc tả kỹ thuật Pipeline Checkpoint 6
|-- Pipeline/
|   |-- preprocessing.py                  # Mã nguồn Checkpoint 1 (Làm sạch, Teencode, Tách từ)
|   |-- dataset.py                        # Quản lý nạp dữ liệu PyTorch Dataset
|   |-- losses.py                         # Mã nguồn Checkpoint 3, 4, 5 (Focal Loss, Weighted Loss)
|   |-- train.py                          # Mã nguồn Checkpoint 2 (Cấu hình Trainer, Callbacks)
|   `-- evaluate.py                       # Đánh giá chỉ số, trích xuất Confusion Matrix
|-- Scripts/
|   |-- preprocess_data.py                # Script chạy xử lý dữ liệu hàng loạt
|   |-- compute_class_weights.py          # Script tính toán vector trọng số lớp
|   |-- create_checkpoint01_notebook.py   # Script khởi tạo notebook Checkpoint 1
|   |-- create_checkpoint02_notebook.py   # Script khởi tạo notebook Checkpoint 2
|   |-- create_checkpoint03_notebook.py   # Script khởi tạo notebook Checkpoint 3
|   `-- create_checkpoint06_notebook.py   # Script khởi tạo notebook Checkpoint 6
`-- Notebook/
    |-- checkpoint01_data_preprocessing.ipynb       # Thực nghiệm Checkpoint 1
    |-- checkpoint02_training_strategies.ipynb      # Thực nghiệm Checkpoint 2
    |-- checkpoint03_focal_loss.ipynb               # Thực nghiệm Checkpoint 3
    |-- checkpoint04_class_weights.ipynb            # Thực nghiệm Checkpoint 4
    |-- checkpoint05_combined_focal_classweight.ipynb # Thực nghiệm Checkpoint 5 và Ablation Study
    `-- checkpoint06_alternative_model.ipynb        # Thực nghiệm Checkpoint 6 (PhoBERT-large / Alternative Model)
```
