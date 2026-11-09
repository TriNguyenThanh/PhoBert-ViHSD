# Đặc Tả Pipeline Checkpoint 2: Thay Đổi Chiến Lược Huấn Luyện

Tài liệu này đặc tả chi tiết kiến trúc luồng huấn luyện, cấu hình tối ưu hóa siêu tham số, cơ chế chọn lọc mô hình và phương thức kiểm soát tài nguyên cho Checkpoint 2 trong bài toán phân loại phát ngôn thù ghét tiếng Việt (ViHSD).

---

## 1. Mục Tiêu Kỹ Thuật

- Khắc phục các hạn chế trong chiến lược huấn luyện cơ sở của notebook gốc (`Paper/ViHSD_monolingual_Transformers.ipynb`):
  - Số lượng epoch quá ngắn (2 epochs), chưa đạt trạng thái hội tụ tối ưu.
  - Không có cơ chế lưu checkpoint theo chỉ số Macro-F1; chỉ lưu mô hình ở epoch cuối cùng vốn dễ bị quá khớp (overfitting) vào lớp đa số (CLEAN).
  - Độ dài chuỗi bị giới hạn ở 100 token (`max_length = 100`), làm mất mát thông tin đối với các bình luận dài.
- Thiết lập quy trình huấn luyện tiêu chuẩn:
  - Nâng độ dài chuỗi tối đa lên `max_length = 256` (ngưỡng tối đa của PhoBERT).
  - Tăng số epoch huấn luyện lên 8 đến 10 epochs, tích hợp dừng sớm (Early Stopping).
  - Tự động nạp lại checkpoint đạt điểm `eval_macro_f1` cao nhất trên tập validation.
  - Áp dụng phân tầng tốc độ học (Differential Learning Rates) và lịch trình Cosine Annealing Scheduler.

---

## 2. Kiến Trúc Luồng Huấn Luyện (Training Pipeline Architecture)

```
[Dữ liệu đã tiền xử lý từ Checkpoint 1: train_X, dev_X, test_X]
                               |
                               v
  (Bước 1) Mã hóa chuỗi với AutoTokenizer (max_length = 256, padding, truncation)
                               |
                               v
  (Bước 2) Khởi tạo PyTorch BuildDataset (Tensor input_ids, attention_mask, labels)
                               |
                               v
  (Bước 3) Cấu hình Mô hình PhoBERT & Phân tầng Tốc độ học (AdamW Optimizer)
                               |
                               v
  (Bước 4) Huấn luyện theo Epoch với Mixed Precision (fp16 = True)
                               |
                               v
  (Bước 5) Đánh giá sau mỗi Epoch trên Dev Set -> Tính toán eval_macro_f1
                               |
               +---------------+---------------+
               |                               |
        (Nếu cải thiện)                 (Nếu không cải thiện)
               |                               |
               v                               v
    Lưu Best Checkpoint              Tăng bộ đếm Early Stopping (Patience = 2..3)
               |                               |
               +---------------+---------------+
                               |
                               v
  (Bước 6) Kết thúc huấn luyện -> Nạp lại Best Checkpoint
                               |
                               v
  (Bước 7) Đánh giá độc lập trên Test Set -> Xuất Báo cáo Macro-F1 & Confusion Matrix
```

---

## 3. Đặc Tả Dữ Liệu và Biểu Diễn Đầu Vào

### 3.1. Dữ liệu đầu vào
- Các mảng dữ liệu đã qua tiền xử lý từ Checkpoint 1: `train_X`, `dev_X`, `test_X` và mảng nhãn tương ứng `train_y`, `dev_y`, `test_y`.

### 3.2. Cấu trúc Lớp `BuildDataset`
Kế thừa hoàn toàn từ notebook gốc nhằm đảm bảo tính tương thích:

```python
import torch

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)
```

---

## 4. Cấu Hình Siêu Tham Số Chi Tiết (Hyperparameter Configuration)

| Tham số | Giá trị Checkpoint 2 | Giá trị Baseline Gốc | Giải thích kỹ thuật |
| :--- | :---: | :---: | :--- |
| **Backbone Model** | `vinai/phobert-base` | `vinai/phobert-base` | Giữ nguyên kiến trúc cơ sở để cô lập tác động của chiến lược huấn luyện |
| **Max Sequence Length** | **256** | 100 | Bao phủ trọn vẹn ngữ cảnh bình luận dài, tận dụng tối đa `max_position_embeddings` của PhoBERT |
| **Number of Epochs** | **8 - 10** | 2 | Cho phép mô hình hội tụ đầy đủ trên các mẫu khó |
| **Batch Size (Train)** | **16** | 16 | Phù hợp với bộ nhớ GPU khi nâng `max_length = 256` |
| **Gradient Accumulation**| **2** | 1 | Kích thước batch hiệu dụng bằng $16 \times 2 = 32$ |
| **Backbone Learning Rate**| **$2 \times 10^{-5}$** | $2 \times 10^{-5}$ | Bảo toàn biểu diễn ngôn ngữ tiền huấn luyện |
| **Head Learning Rate** | **$1 \times 10^{-4}$** | $2 \times 10^{-5}$ | Tăng tốc độ thích nghi của tầng phân loại tuyến tính mới khởi tạo |
| **Optimizer** | AdamW (`weight_decay=0.01`) | AdamW | Tối ưu hóa chuẩn cho Transformer |
| **Learning Rate Scheduler**| Cosine Annealing with Warmup | Linear Decay | Khởi động tuyến tính 10% số bước, suy giảm theo đường cong cosine |
| **Warmup Ratio** | **0.1 (10%)** | Cố định 500 steps | Thích ứng động theo tổng số bước huấn luyện thực tế |
| **Mixed Precision** | `fp16 = True` | `False` | Tăng tốc độ tính toán gấp ~2 lần và giảm 50% mức tiêu thụ VRAM |
| **Evaluation Strategy** | `epoch` | Không đánh giá định kỳ | Đánh giá toàn diện trên tập dev sau mỗi epoch |
| **Save Strategy** | `epoch` | Cuối lượt huấn luyện | Lưu checkpoint theo từng epoch để phục vụ việc chọn lọc |
| **Best Model Metric** | `eval_macro_f1` | Không cấu hình | Căn chỉnh mô hình trực tiếp theo mục tiêu Macro-F1 |
| **Early Stopping** | `patience = 2` hoặc `3` | Không có | Dừng huấn luyện tự động nếu 2-3 epoch liên tiếp không cải thiện Macro-F1 |

---

## 5. Cài Đặt Kỹ Thuật Chi Tiết

### 5.1. Hàm tính toán chỉ số đánh giá (Compute Metrics)
Hàm đánh giá được truyền trực tiếp vào Hugging Face `Trainer` để tính toán chỉ số sau mỗi epoch, tích hợp đầy đủ Accuracy, Precision, Recall và F1-Score:

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, preds)
    macro_p = precision_score(labels, preds, average='macro', zero_division=0)
    macro_r = recall_score(labels, preds, average='macro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro')
    micro_f1 = f1_score(labels, preds, average='micro')
    
    # Tính F1 riêng cho từng lớp (0: clean, 1: offensive, 2: hate)
    f1_per_class = f1_score(labels, preds, average=None)
    
    return {
        'eval_accuracy': acc,
        'eval_macro_precision': macro_p,
        'eval_macro_recall': macro_r,
        'eval_macro_f1': macro_f1,
        'eval_micro_f1': micro_f1,
        'eval_f1_clean': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
        'eval_f1_offensive': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        'eval_f1_hate': f1_per_class[2] if len(f1_per_class) > 2 else 0.0,
    }
```

### 5.2. Cấu hình Phân tầng Tốc độ học (Differential Learning Rates)
Tách biệt các nhóm tham số của mô hình thành 2 nhóm tối ưu:

```python
def get_optimizer_grouped_parameters(model, backbone_lr=2e-5, head_lr=1e-4, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight"]
    
    optimizer_grouped_parameters = [
        # Tham số backbone Transformer có áp dụng weight decay
        {
            "params": [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        },
        # Tham số backbone Transformer không áp dụng weight decay
        {
            "params": [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": backbone_lr,
            "weight_decay": 0.0,
        },
        # Tham số Classification Head
        {
            "params": [p for n, p in model.classifier.named_parameters()],
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
    ]
    return optimizer_grouped_parameters
```

---

## 6. Quy Trình Kiểm Thử và Tiêu Chuẩn Nghiệm Thu

1. **Kiểm tra đồ thị hội tụ (Learning Curves):**
   - Đồ thị `eval_macro_f1` qua các epoch phải phản ánh xu hướng tăng dần và ổn định.
   - Hiện tượng overfitting được khống chế bởi `EarlyStoppingCallback`.

2. **Tiêu chuẩn nghiệm thu trên tập Test:**
   - Thực thi notebook `Notebook/checkpoint02_training_strategies.ipynb`.
   - Nạp mô hình đạt điểm validation cao nhất và dự đoán trên `test_dataset`.
   - Báo cáo kết quả đầy đủ các chỉ số: **Accuracy**, **Macro-Precision**, **Macro-Recall**, **Macro-F1**, **Micro-F1**, cùng Precision, Recall và F1 theo từng nhãn (`clean`, `offensive`, `hate`).
   - Kết quả chỉ số **Macro-F1 trên tập Test** phải ghi nhận mức tăng trưởng vượt trội so với Checkpoint 1, chủ yếu nhờ vào việc nạp lại trọng số tại thời điểm tối ưu nhất thay vì trọng số bị quá khớp ở epoch cuối.
   - Ma trận nhầm lẫn phải thể hiện rõ sự gia tăng về độ nhạy (Recall) đối với 2 lớp thiểu số (OFFENSIVE và HATE).

---

## 7. Quy Chuẩn Xuất Kết Quả (Mandatory Artifacts Export)

Toàn bộ 4 tệp kết quả chuẩn hóa được tự động lưu vào `outputs/checkpoint02/`:
1. `predictions.csv`: Định danh, văn bản gốc, nhãn thực tế (`ground_truth`) và nhãn dự đoán (`predicted_label`).
2. `confusion_matrix_checkpoint02.png` & `confusion_matrix.csv`: Biểu đồ nhiệt ma trận nhầm lẫn theo chuẩn học thuật (không icon) và bảng CSV tương ứng.
3. `evaluation_scores.csv`: Điểm số tổng hợp cốt lõi:
   - Các chỉ số tổng thể: `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `micro_f1`.
   - Các chỉ số từng lớp: `precision_clean`, `recall_clean`, `f1_clean`, `precision_offensive`, `recall_offensive`, `f1_offensive`, `precision_hate`, `recall_hate`, `f1_hate`.
4. `classification_report.csv`: Báo cáo phân loại chi tiết với đầy đủ `class`, `precision`, `recall`, `f1_score`, `support` (giá trị support dạng số nguyên và hàng accuracy tổng thể).

