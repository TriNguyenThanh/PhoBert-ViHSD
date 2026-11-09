# Đặc Tả Pipeline Checkpoint 5: Xử Lý Mất Cân Bằng Lớp Bằng Cả Hai Phương Pháp

Tài liệu này đặc tả chi tiết cơ sở lý thuyết, kiến trúc mô hình hóa, quy trình điều phối siêu tham số tương tác, phương pháp nghiên cứu bóc tách thành phần (Ablation Study) và phân tích lỗi định tính cho Checkpoint 5 trong bài toán phân loại phát ngôn thù ghét tiếng Việt (ViHSD).

---

## 1. Mục Tiêu Kỹ Thuật

- Kết hợp đồng thời hai cơ chế tối ưu:
  1. **Cân bằng phân bố lớp tĩnh (Class Weight $\alpha_c$):** Bù đắp sự chênh lệch tần suất mẫu giữa lớp đa số và lớp thiểu số.
  2. **Điều chế mẫu khó động (Focal Modulating Factor $(1 - p_t)^\gamma$):** Giảm thiểu gradient từ các mẫu dễ phân loại và tập trung vào các ranh giới quyết định phức tạp.
- Điều phối tương tác giữa hai siêu tham số $\alpha_c$ và $\gamma$ nhằm ngăn chặn hiện tượng phạt quá mức (Over-penalization) gây tổn hại đến Precision của lớp CLEAN.
- Tiến hành thực nghiệm bóc tách thành phần toàn diện (Ablation Study) đối chiếu với các Checkpoint tiền đề và phân tích lỗi định tính (Error Analysis).

---

## 2. Kiến Trúc Luồng Xử Lý (Combined Strategy Pipeline Architecture)

```
[Dữ liệu tiền xử lý (CP1) & Chiến lược huấn luyện tối ưu (CP2)]
                                  |
               +------------------+------------------+
               |                                     |
               v                                     v
  Vector Trọng số Lớp alpha_c           Tham số Điều chế gamma tối ưu
    (Kế thừa từ Checkpoint 4)             (Kế thừa từ Checkpoint 3)
               |                                     |
               +------------------+------------------+
                                  |
                                  v
           Khởi tạo Module WeightedFocalLoss(alpha=W, gamma)
                                  |
                                  v
    Huấn luyện với CombinedLossTrainer (PhoBERT-base, max_len=256, 8 epochs)
                                  |
                                  v
    Đánh giá và Tinh chỉnh Lưới (alpha, gamma) trên Dev Set
                                  |
                                  v
    Đánh giá Tổng hợp trên Test Set -> Thực hiện Ablation Study (4 biến thể)
                                  |
                                  v
    Phân tích Lỗi Định tính (Error Analysis trên ma trận nhầm lẫn)
```

---

## 3. Cơ Sở Toán Học của Weighted Focal Loss

Hàm mất mát kết hợp được định nghĩa:
$$\mathcal{L}_{WFL}(p_t) = -\alpha_{y_i} (1 - p_{i, y_i})^\gamma \log(p_{i, y_i} + \epsilon)$$
Trong đó:
- $y_i \in \{0, 1, 2\}$ là nhãn thực tế của mẫu $i$.
- $p_{i, y_i}$ là xác suất dự đoán của mô hình đối với nhãn thực tế $y_i$ sau hàm Softmax.
- $\alpha_{y_i}$ là trọng số lớp tĩnh gán cho nhãn $y_i$ (được chuẩn hóa sao cho $\frac{1}{K}\sum_{c=0}^{K-1} \alpha_c = 1$).
- $(1 - p_{i, y_i})^\gamma$ là hệ số điều chế mẫu khó với $\gamma \ge 0$.

### Cơ chế tác động hiệp đồng
- Khi gặp mẫu CLEAN dễ phân loại ($y_i = 0, p_{i, 0} \approx 0.95$):
  - Trọng số $\alpha_0 \approx 0.4$ làm giảm tổn thất theo tỷ lệ tĩnh.
  - Hệ số $(1 - 0.95)^2 = 0.0025$ triệt tiêu gần như hoàn toàn gradient đóng góp.
- Khi gặp mẫu HATE khó phân loại ($y_i = 2, p_{i, 2} \approx 0.20$):
  - Trọng số $\alpha_2 \approx 3.1$ khuếch đại tổn thất theo phân bố lớp.
  - Hệ số $(1 - 0.20)^2 = 0.64$ duy trì mức phạt gradient cao, buộc mô hình phải điều chỉnh trọng số về hướng phân loại đúng mẫu thù ghét này.

---

## 4. Cài Đặt Kỹ Thuật Chi Tiết

### 4.1. Lớp `WeightedFocalLoss` trong PyTorch
Triển khai tại [Pipeline/losses.py](Pipeline/losses.py):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        logits: (batch_size, num_classes)
        targets: (batch_size,)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt = log_pt.exp()
        
        modulating_factor = (1.0 - pt) ** self.gamma
        loss = -modulating_factor * log_pt
        
        if self.alpha is not None:
            # Gán trọng số alpha tương ứng với nhãn thực tế
            alpha_factor = self.alpha.gather(dim=-1, index=targets)
            loss = alpha_factor * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
```

### 4.2. Lớp `CombinedLossTrainer`
Tương thích trực tiếp với luồng `Trainer` của Hugging Face:

```python
from transformers import Trainer

class CombinedLossTrainer(Trainer):
    def __init__(self, class_weights=None, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = WeightedFocalLoss(alpha=class_weights, gamma=gamma, reduction='mean')

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Đảm bảo buffer alpha cùng thiết bị tính toán với logits
        if self.loss_fn.alpha is not None and self.loss_fn.alpha.device != logits.device:
            self.loss_fn.alpha = self.loss_fn.alpha.to(logits.device)
            
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
```

---

## 5. Nghiên Cứu Bóc Tách Thành Phần (Ablation Study)

Tiến hành thực nghiệm so sánh độc lập 4 cấu hình hàm mất mát trên cùng một dữ liệu tiền xử lý (Checkpoint 1) và cùng một chiến lược huấn luyện (Checkpoint 2: `vinai/phobert-base`, `max_length = 256`, 8 epochs, Cosine Scheduler, Early Stopping):

| Biến thể Thử nghiệm | Hàm Mất Mát | Tham số $\alpha$ | Tham số $\gamma$ | Test Macro-F1 | F1 - CLEAN | F1 - OFFENSIVE | F1 - HATE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Baseline Chuẩn** | Standard Cross-Entropy | Không ($\alpha=1$) | $\gamma = 0$ | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |
| **M2: Focal Loss Đơn lẻ** | Focal Loss | Không ($\alpha=1$) | $\gamma = \gamma^*$ | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |
| **M3: Class Weight Đơn lẻ**| Weighted Cross-Entropy | Vector $W^*$ | $\gamma = 0$ | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |
| **M4: Phương pháp Kết hợp**| Weighted Focal Loss | Vector $W^*$ | $\gamma = \gamma^*$ | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |

*Ghi chú:* $\gamma^*$ là giá trị tối ưu tìm được từ Checkpoint 3; $W^*$ là phương pháp tính trọng số tối ưu từ Checkpoint 4.

---

## 6. Quy Trình Phân Tích Lỗi Định Tính (Qualitative Error Analysis)

Sau khi hoàn thành đánh giá trên tập kiểm tra độc lập (`test.csv`), tiến hành trích xuất và phân loại các trường hợp dự đoán sai nhằm đánh giá cơ chế nhận diện ngữ cảnh của mô hình:

1. **Lỗi nhầm lẫn giữa CLEAN và OFFENSIVE:**
   - Phân tích các câu chứa tiếng lóng hoặc từ ngữ nhạy cảm nhưng mang tính đùa vui, ngữ cảnh thân mật không có ác ý.
2. **Lỗi nhầm lẫn giữa OFFENSIVE và HATE:**
   - Đánh giá khả năng của mô hình trong việc nhận diện đối tượng công kích cụ thể (Target Identification). Kiểm tra xem mô hình có phân biệt được giữa chửi thề bộc phát (OFFENSIVE) và thù ghét nhắm vào danh tính, nhóm người hoặc tổ chức (HATE).
3. **Lỗi bỏ sót HATE thành CLEAN (False Negatives nghiêm trọng):**
   - Phân tích các câu thù ghét tinh vi sử dụng nghệ thuật châm biếm, mỉa mai (sarcasm) hoặc từ ngữ ẩn dụ không chứa từ ngữ thô tục hiển ngôn.

---

## 7. Tiêu Chuẩn Nghiệm Thu Checkpoint 5

- Notebook thực nghiệm hoàn chỉnh: `Notebook/checkpoint05_combined_focal_classweight.ipynb`.
- Báo cáo kết quả Ablation Study đầy đủ số liệu trên cả 3 tập (train, dev, test).
- Mô hình kết hợp M4 phải đạt điểm **Macro-F1 cao nhất** hoặc có sự cân bằng tốt nhất giữa F1 của cả 3 lớp so với các mô hình đơn lẻ M1, M2, M3.
