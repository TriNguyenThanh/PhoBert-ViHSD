# Kế Hoạch Nghiên Cứu và Lộ Trình Thực Nghiệm (Research Roadmap)

Dự án: Nâng cao hiệu năng mô hình phát hiện phát ngôn thù ghét trên mạng xã hội tiếng Việt (ViHSD - Vietnamese Hate Speech Detection).

Tài liệu này xác lập lộ trình kỹ thuật và các mốc thực nghiệm (Checkpoints 01 đến 06) có tính kế thừa và đối chứng khoa học chặt chẽ, hướng đến giải quyết hai thách thức cốt lõi: đặc thù nhiễu ngôn ngữ mạng xã hội và hiện tượng mất cân bằng phân bố nhãn nghiêm trọng trong tập dữ liệu ViHSD.

---

## 1. Tổng Quan Bài Toán và Mục Tiêu Đánh Giá

- **Đặc tả tác vụ:** Phân loại văn bản 3 lớp trên bình luận mạng xã hội tiếng Việt:
  - Nhãn `0` - CLEAN: Bình luận sạch, không chứa nội dung thù ghét, độc hại (~82.7% tập dữ liệu).
  - Nhãn `1` - OFFENSIVE: Bình luận chứa từ ngữ thô tục, xúc phạm nhưng không nhắm vào đối tượng cụ thể (~6.7% tập dữ liệu).
  - Nhãn `2` - HATE: Phát ngôn thù ghét trực diện nhắm vào cá nhân, tổ chức, nhóm người hoặc cộng đồng (~10.6% tập dữ liệu).
- **Thước đo đánh giá chính:** **Macro-F1 Score** (trung bình không trọng số của F1 cả 3 lớp). Do tỷ lệ mất cân bằng nghiêm trọng giữa lớp đa số (CLEAN) và các lớp thiểu số (OFFENSIVE, HATE), chỉ số Accuracy thông thường không phản ánh chính xác năng lực phát hiện của mô hình.
- **Mô hình đường cơ sở (Baseline):** Kiến trúc `vinai/phobert-base` được huấn luyện theo cấu hình trong nghiên cứu gốc (Lưu et al., 2021) với 2 epochs, learning rate $2 \times 10^{-5}$, max_length = 100 và hàm mất mát Cross-Entropy tiêu chuẩn.

---

## 2. Sơ Đồ Kiến Trúc Lộ Trình 6 Checkpoint Tiến Bộ (Progressive Roadmap)

```
[Mô hình Baseline Gốc (Lưu et al., 2021)]
(PhoBERT-base, 2 epochs, lr=2e-5, max_length=100, Cross-Entropy Loss)
                      |
       +--------------+--------------+
       |                             |
       v                             v
[Checkpoint 01]               [Checkpoint 02]
Baseline + Cân Bằng Lớp Tĩnh   Baseline + Cân Bằng Lớp Động
(Smoothed Class Weights)      (Focal Loss, gamma=1.5)
       |                             |
       +--------------+--------------+
                      |
                      v
               [Checkpoint 03]
               Baseline + Tiền Xử Lý Dữ Liệu Chuyên Sâu
               (ViHSDPreprocessor: Teencode, Emojis, PyVi)
                      |
                      v
               [Checkpoint 04]
               Tối Ưu Hóa Chiến Lược Huấn Luyện Nâng Cao
               (Diff LR, Cosine Warmup, 8 Epochs, MaxLen 256, Early Stopping)
                      |
                      v
               [Checkpoint 05]
               Tích Hợp Toàn Diện 4 Giải Pháp Nâng Cấp
               (CP01 + CP02 + CP03 + CP04 trên vinai/phobert-base với PyVi)
               (Weighted Focal Loss: alpha Smoothed CW + gamma=1.5)
                      |
                      v
               [Checkpoint 06]
               Hiện Đại Hóa Kiến Trúc Nền Tảng
               (PhoBERT-base-v2 Modernization)
```

---

## 3. Đặc Tả Chi Tiết Từng Checkpoint Thực Nghiệm

### Checkpoint 01: Thiết Lập Baseline Kết Hợp Cân Bằng Lớp Tĩnh (Smoothed Class Weights)
- **Tên notebook:** `Notebook/checkpoint01_baseline_class_weights.ipynb`
- **Mục tiêu khoa học:** Đánh giá độc lập tác động của việc bổ sung vector trọng số tĩnh nhằm phạt sai số trên các lớp thiểu số (OFFENSIVE, HATE) trên cấu hình Baseline gốc của Lưu et al. (2021).
- **Kỹ thuật cốt lõi:**
  - Áp dụng công thức làm mịn căn bậc hai (Smoothed Inverse Class Frequency):
    $$w_c = \sqrt{\frac{N}{N_c}} \bigg/ \left( \frac{1}{K} \sum_{j=0}^{K-1} \sqrt{\frac{N}{N_j}} \right)$$
  - Kiểm soát tỷ lệ trọng số thiểu số / đa số ở mức lý tưởng (~3.52 lần), tránh hiện tượng phạt quá nặng gây sụp đổ Precision của lớp CLEAN khi dùng nghịch đảo tuyến tính thuần túy.
- **Cấu hình mô hình:** `vinai/phobert-base`, 2 epochs, lr = $2 \times 10^{-5}$, max_length = 100, batch size = 16, Weighted Cross-Entropy Loss.
- **Thư mục đầu ra:** `outputs/checkpoint01/`

### Checkpoint 02: Thiết Lập Baseline Kết Hợp Cân Bằng Lớp Động Theo Độ Khó (Focal Loss, gamma = 1.5)
- **Tên notebook:** `Notebook/checkpoint02_baseline_focal_loss.ipynb`
- **Mục tiêu khoa học:** Khảo sát cơ chế điều chế tổn thất động theo độ khó phân loại mà không sử dụng vector trọng số tĩnh ($\alpha = \text{None}$) trên cấu hình Baseline gốc.
- **Kỹ thuật cốt lõi:**
  - Hàm mất mát Focal Loss đa lớp:
    $$\mathcal{L}_{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t), \quad \gamma = 1.5$$
  - Khi mẫu dễ phân loại ($p_t \to 1$), hệ số điều chế $(1 - p_t)^\gamma \to 0$, triệt tiêu gradient lấn át từ hàng vạn mẫu CLEAN dễ học.
  - Tối ưu hóa số học PyTorch bằng cách tính $p_t = \exp(-\mathcal{L}_{CE})$ qua hàm `F.cross_entropy` với cơ chế log-sum-exp.
- **Cấu hình mô hình:** `vinai/phobert-base`, 2 epochs, lr = $2 \times 10^{-5}$, max_length = 100, batch size = 16, Focal Loss ($\gamma = 1.5$).
- **Thư mục đầu ra:** `outputs/checkpoint02/`

### Checkpoint 03: Tiền Xử Lý và Chuẩn Hóa Dữ Liệu Ngôn Ngữ Mạng Xã Hội (Data Preprocessing)
- **Tên notebook:** `Notebook/checkpoint03_data_preprocessing.ipynb`
- **Mục tiêu khoa học:** Khắc phục triệt để sai lầm của bài báo gốc khi loại bỏ stopwords và xóa emoji bừa bãi bằng cách xây dựng module tiền xử lý chuyên biệt `ViHSDPreprocessor`.
- **Kỹ thuật cốt lõi:**
  - Chuẩn hóa viết tắt và tiếng lóng mạng xã hội với từ điển 412 quy tắc (`teencode.csv`).
  - Ánh xạ biểu cảm emoji 2 pha (bảo vệ khoảng cách trước PyVi và chuyển đổi thành văn bản biểu đạt sắc thái cảm xúc từ `emojis.csv`).
  - Thu gọn ký tự lặp kéo dài cảm xúc về 1 ký tự (`ngonnnn` $\to$ `ngon`).
  - Phân đoạn từ tiếng Việt tự động bằng thư viện `pyvi` (`ViTokenizer.tokenize`).
  - **Bảo tồn tuyệt đối** đại từ nhân xưng (`mày`, `tao`, `nó`, `thằng`, `con`) và từ phủ định (`không`, `chẳng`, `chả`).
- **Cấu hình mô hình:** `vinai/phobert-base`, 2 epochs, lr = $2 \times 10^{-5}$, max_length = 100, Cross-Entropy Loss tiêu chuẩn để cô lập tác động của tiền xử lý.
- **Thư mục đầu ra:** `outputs/checkpoint03/`

### Checkpoint 04: Tối Ưu Hóa Chiến Lược Huấn Luyện Nâng Cao (Advanced Training Strategies)
- **Tên notebook:** `Notebook/checkpoint04_training_strategies.ipynb`
- **Mục tiêu khoa học:** Khắc phục hạn chế thời gian học quá ngắn (2 epochs) và độ dài chuỗi bị cắt cụt (100 tokens) của baseline.
- **Kỹ thuật cốt lõi:**
  - Mở rộng độ dài chuỗi tối đa lên `max_length = 256` (khai thác trọn vẹn vị trí nhúng tối đa của PhoBERT).
  - Phân tầng tốc độ học (Differential Learning Rates): $2 \times 10^{-5}$ cho backbone RoBERTa và $1 \times 10^{-4}$ cho Classification Head.
  - Lịch trình Cosine Annealing with Warmup (10% warmup steps) trong 8 epochs.
  - Tích lũy gradient (batch size 16, gradient accumulation steps 2 $\to$ effective batch size 32).
  - Tự động dừng sớm (Early Stopping với patience = 2) theo dõi `eval_macro_f1` và nạp lại mô hình tốt nhất (`load_best_model_at_end = True`).
- **Thư mục đầu ra:** `outputs/checkpoint04/`

### Checkpoint 05: Tích Hợp Toàn Diện Các Giải Pháp Nâng Cấp (Combined Enhancements)
- **Tên notebook:** `Notebook/checkpoint05_combined_enhancements.ipynb`
- **Mục tiêu khoa học:** Kết hợp đồng thời 4 cải tiến độc lập (CP01 Smoothed CW + CP02 Focal Loss + CP03 Preprocessing + CP04 Training strategies) trên nền tảng `vinai/phobert-base` để tạo ra mô hình tối ưu toàn diện.
- **Kỹ thuật cốt lõi:**
  - Tiền xử lý dữ liệu: `ViHSDPreprocessor` (CP03).
  - Chiến lược huấn luyện: 8 epochs, max_length = 256, differential lr, Cosine Warmup, Early Stopping (CP04).
  - Hàm mất mát tích hợp **Weighted Focal Loss**:
    $$\mathcal{L}_{WFL} = -\alpha_{y_i} (1 - p_{i, y_i})^\gamma \log(p_{i, y_i})$$
    kết hợp vector trọng số lớp làm mịn $\alpha$ từ CP01 và hệ số tập trung $\gamma = 1.5$ từ CP02.
- **Cấu hình mô hình:** `vinai/phobert-base`, max_length = 256, 8 epochs, Weighted Focal Loss.
- **Thư mục đầu ra:** `outputs/checkpoint05/`

### Checkpoint 06: Hiện Đại Hóa Kiến Trúc Mô Hình (PhoBERT-base-v2 Modernization)
- **Tên notebook:** `Notebook/checkpoint06_phobert_base_v2.ipynb`
- **Mục tiêu khoa học:** Đánh giá bước nhảy vọt về năng lực biểu diễn ngôn ngữ khi nâng cấp backbone từ `vinai/phobert-base` thế hệ đầu sang `vinai/phobert-base-v2` thế hệ mới.
- **Kỹ thuật cốt lõi:**
  - Backbone `vinai/phobert-base-v2` được tiền huấn luyện trên 20GB văn bản tiếng Việt hiện đại (gấp hơn 5 lần bản v1).
  - Kế thừa 100% pipeline tối ưu từ Checkpoint 05: `ViHSDPreprocessor`, `max_length = 256`, differential lr, Cosine Warmup, 8 epochs, Early Stopping, Weighted Focal Loss ($\alpha$ Smoothed CW, $\gamma = 1.5$).
  - Đối sánh trực tiếp với Checkpoint 05 để chứng minh độ nhạy và tính chuẩn xác của mô hình mới.
- **Cấu hình mô hình:** `vinai/phobert-base-v2`, max_length = 256, 8 epochs, Weighted Focal Loss.
- **Thư mục đầu ra:** `outputs/checkpoint06/`

---

## 4. Bảng Ma Trận Đối Sánh Thực Nghiệm (Cumulative Ablation Matrix)

| Chỉ số / Đặc điểm thực nghiệm | Baseline Gốc (2021) | Checkpoint 01 | Checkpoint 02 | Checkpoint 03 | Checkpoint 04 | Checkpoint 05 (Kết hợp PyVi) | Checkpoint 06 (PhoBERT-v2) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mô hình nền tảng (Backbone)** | `phobert-base` | `phobert-base` | `phobert-base` | `phobert-base` | `phobert-base` | `phobert-base` | **`phobert-base-v2`** |
| **Công cụ tách từ (Segmenter)** | Không (fastBPE thô) | Không | Không | `pyvi` | Không | `pyvi` | `pyvi` |
| **Tiền xử lý văn bản** | Gốc (lọc từ dừng) | Thô (giữ nguyên) | Thô (giữ nguyên) | **ViHSDPreprocessor** | Thô (giữ nguyên) | **ViHSDPreprocessor** | **ViHSDPreprocessor** |
| **Độ dài ngữ cảnh (max_len)** | 100 | 100 | 100 | 100 | **256** | **256** | **256** |
| **Số epochs tối đa** | 2 | 2 | 2 | 2 | **8 (Early Stopping)** | **8 (Early Stopping)** | **8 (Early Stopping)** |
| **Cơ chế tối ưu hóa tốc độ học** | AdamW cố định | AdamW cố định | AdamW cố định | AdamW cố định | **Diff LR + Cosine** | **Diff LR + Cosine** | **Diff LR + Cosine** |
| **Hàm mất mát áp dụng** | Cross-Entropy | **Weighted CE** | **Focal Loss** | Cross-Entropy | Cross-Entropy | **Weighted Focal Loss** | **Weighted Focal Loss** |
| **Vector trọng số tĩnh $\alpha$** | Không | **Smoothed CW** | Không | Không | Không | **Smoothed CW** | **Smoothed CW** |
| **Tham số điều chế $\gamma$** | $\gamma = 0$ | $\gamma = 0$ | **$\gamma = 1.5$** | $\gamma = 0$ | $\gamma = 0$ | **$\gamma = 1.5$** | **$\gamma = 1.5$** |
| **Macro-F1 (Thước đo cốt lõi)** | 0.6276 | 0.6493 | 0.6219 | 0.6627 | 0.6460 | 0.6686 | **0.6617** |
| **F1 - Lớp CLEAN (0)** | 0.9020 | 0.9319 | 0.9341 | 0.9410 | 0.9327 | 0.9357 | **0.9405** |
| **F1 - Lớp OFFENSIVE (1)** | 0.4610 | 0.3844 | 0.3688 | 0.4396 | 0.4336 | 0.4501 | **0.4584** |
| **F1 - Lớp HATE (2)** | 0.5200 | 0.5706 | 0.5627 | 0.6075 | 0.5717 | 0.6200 | **0.5860** |

---

## 5. Quy Chuẩn Xuất Kết Quả Bắt Buộc (Mandatory Artifacts Export)

Để đảm bảo tính minh bạch học thuật và khả năng tái lập thực nghiệm, toàn bộ 6 checkpoint notebook khi thực thi đều tự động xuất 4 tệp chuẩn hóa vào `outputs/checkpointXX/`:
1. `predictions.csv`: Chứa đầy đủ các cột `id`, `text`, `ground_truth`, `predicted_label`.
2. `confusion_matrix_checkpointXX.png` & `confusion_matrix.csv`: Biểu đồ nhiệt Seaborn và bảng dữ liệu ma trận nhầm lẫn.
3. `evaluation_scores.csv`: Bảng tổng hợp các chỉ số đánh giá `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `micro_f1` và F1 từng lớp.
4. `classification_report.csv`: Báo cáo chi tiết độ chuẩn xác, độ nhạy, F1 và số lượng mẫu của từng lớp với trường support dạng số nguyên.
