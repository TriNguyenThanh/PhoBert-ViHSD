# Đặc Tả Pipeline Checkpoint 3: Xử Lý Mất Cân Bằng Lớp Bằng Focal Loss

Tài liệu này đặc tả chi tiết cơ sở toán học, thiết kế mô-đun hàm mất mát Focal Loss, quy trình tìm kiếm siêu tham số và phương pháp tích hợp vào luồng huấn luyện cho Checkpoint 3 trong bài toán phân loại phát ngôn thù ghét tiếng Việt (ViHSD).

---

## 1. Mục Tiêu Kỹ Thuật

- Giải quyết triệt để hiện tượng lớp đa số (CLEAN, chiếm ~82.7%) lấn át gradient cập nhật của mô hình trong quá trình tối ưu hóa với hàm mất mát Cross-Entropy tiêu chuẩn.
- Ứng dụng cơ chế điều chế mẫu khó (Hard Example Mining) của Focal Loss: tự động giảm trọng số đóng góp của các mẫu dễ phân loại ($p_t \to 1$) và tập trung gradient vào các mẫu khó phân loại thuộc hai lớp thiểu số (OFFENSIVE và HATE).
- Khảo sát độc lập tác động của tham số tập trung $\gamma$ (Focusing Parameter) khi chưa áp dụng vector trọng số lớp tĩnh ($\alpha = 1$).

---

## 2. Kiến Trúc Luồng Xử Lý (Focal Loss Pipeline Architecture)

```
[Dữ liệu tiền xử lý (CP1) & Cấu hình huấn luyện (CP2: max_len=256, 8 epochs)]
                                  |
                                  v
            Forward Pass qua Mô hình vinai/phobert-base
                                  |
                                  v
                   Trích xuất Logits z (Kích thước: B x 3)
                                  |
                                  v
                 Hàm Tính Softmax -> Xác suất p_t của nhãn đích
                                  |
                                  v
             Tính Hệ số điều chế Modulating Factor: (1 - p_t)^gamma
                                  |
                                  v
        Tính Giá trị Tổn thất Focal Loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t)
                                  |
                                  v
         Backward Pass cập nhật Gradient tập trung vào mẫu khó
                                  |
                                  v
         Đánh giá Grid Search gamma in {1.0, 1.5, 2.0, 2.5} trên Dev Set
                                  |
                                  v
         Xác định gamma tối ưu -> Đánh giá độc lập trên Test Set
```

---

## 3. Cơ Sở Toán Học và Công Thức Tính Toán

### 3.1. Giới hạn của Cross-Entropy tiêu chuẩn
Hàm mất mát Cross-Entropy thông thường đối với mẫu $i$ có nhãn thực tế $y_i$ được biểu diễn:
$$\mathcal{L}_{CE}(p_t) = -\log(p_t)$$
Trong đó $p_t = \frac{\exp(z_{y_i})}{\sum_{c=0}^{2} \exp(z_c)}$ là xác suất dự đoán của nhãn đúng.
Khi tập dữ liệu mất cân bằng nặng, tổng tổn thất trên một batch chủ yếu đến từ hàng nghìn mẫu CLEAN dễ phân loại ($p_t \approx 0.9$). Dù mỗi mẫu đóng góp giá trị nhỏ, số lượng áp đảo của chúng sẽ lấn át gradient của các mẫu HATE và OFFENSIVE khó phân loại.

### 3.2. Công thức Focal Loss đa lớp (Multi-class Focal Loss)
Focal Loss bổ sung hệ số điều chế $(1 - p_t)^\gamma$ vào hàm Cross-Entropy:
$$\mathcal{L}_{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$
- Khi một mẫu bị phân loại sai nghiêm trọng ($p_t \to 0$): Hệ số $(1 - p_t)^\gamma \approx 1$, giá trị tổn thất được bảo toàn như Cross-Entropy thông thường.
- Khi một mẫu được phân loại đúng với độ tự tin cao ($p_t \to 1$): Hệ số $(1 - p_t)^\gamma \to 0$, giá trị tổn thất bị suy giảm đáng kể, triệt tiêu gradient của mẫu dễ.
- $\gamma \ge 0$ là siêu tham số điều khiển độ nhạy phạt:
  - $\gamma = 0$: Hàm mất mát trở về Cross-Entropy tiêu chuẩn.
  - $\gamma > 0$: Càng lớn thì mức độ triệt tiêu mẫu dễ càng mạnh.

Với một mini-batch gồm $B$ mẫu, tổn thất trung bình được tính:
$$\mathcal{L}_{batch} = -\frac{1}{B} \sum_{i=1}^{B} (1 - p_{i, y_i})^\gamma \log(p_{i, y_i} + \epsilon)$$
với $\epsilon = 10^{-8}$ là hằng số làm mịn tránh tràn số học khi tính logarit.

---

## 4. Cài Đặt Kỹ Thuật Chi Tiết

### 4.1. Lớp `FocalLoss` trong PyTorch
Được triển khai trong mô-đun [Pipeline/losses.py](Pipeline/losses.py) với khả năng tính toán ổn định số học (sử dụng `log_softmax`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: Tensor kích thước (batch_size, num_classes)
        targets: Tensor kích thước (batch_size,) chứa nhãn nguyên (0, 1, 2)
        """
        # Tính toán log-softmax để tránh tràn số học
        log_probs = F.log_softmax(logits, dim=-1)
        # Trích xuất xác suất log của nhãn thực tế
        log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        # Tính p_t
        pt = log_pt.exp()
        
        # Hệ số điều chế
        modulating_factor = (1.0 - pt) ** self.gamma
        loss = -modulating_factor * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
```

### 4.2. Tích hợp Custom Trainer trong Hugging Face
Ghi đè phương thức `compute_loss` để thay thế Cross-Entropy mặc định:

```python
from transformers import Trainer

class FocalLossTrainer(Trainer):
    def __init__(self, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = FocalLoss(gamma=focal_gamma, reduction='mean')

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
```

---

## 5. Quy Trình Khảo Sát Siêu Tham Số (Hyperparameter Grid Search)

Thực nghiệm quét lưới độc lập trên tập `dev.csv` với không gian giá trị:
$$\gamma \in \{1.0, 1.5, 2.0, 2.5\}$$

### Bảng theo dõi thực nghiệm Grid Search

| Tham số $\gamma$ | Train Loss | Dev Loss | Dev Macro-F1 | F1 - CLEAN (0) | F1 - OFFENSIVE (1) | F1 - HATE (2) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\gamma = 0.0$ (Baseline CP2) | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |
| $\gamma = 1.0$ | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |
| $\gamma = 1.5$ | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |
| $\gamma = 2.0$ | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |
| $\gamma = 2.5$ | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |

---

## 6. Quy Trình Kiểm Thử và Tiêu Chuẩn Nghiệm Thu

1. **Kiểm tra độ ổn định số học (Numerical Stability):**
   - Giá trị Loss không bao giờ xuất hiện `NaN` hoặc `Inf` trong suốt quá trình huấn luyện.
   - Khi thiết lập $\gamma = 0$, giá trị hàm mất mát phải tương đương tuyệt đối với `nn.CrossEntropyLoss()`.

2. **Tiêu chuẩn nghiệm thu trên tập Test:**
   - Thực thi notebook `Notebook/checkpoint03_focal_loss.ipynb`.
   - Chọn ra giá trị $\gamma$ tối ưu nhất từ tập dev để đánh giá độc lập trên tập `test.csv`.
   - **Chỉ số Macro-F1:** Ghi nhận sự cải thiện có ý nghĩa so với Checkpoint 2.
   - **Ma trận nhầm lẫn:** Tỉ lệ phân loại nhầm các mẫu thuộc nhãn OFFENSIVE và HATE thành nhãn CLEAN phải giảm rõ rệt, chứng minh cơ chế tập trung vào mẫu khó đã phát huy hiệu quả.
