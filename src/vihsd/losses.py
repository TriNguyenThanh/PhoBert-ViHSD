import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import Trainer
except ImportError:
    torch = None
    nn = None
    F = None
    Trainer = None


# Định nghĩa hàm mất mát Focal Loss đa lớp với cơ chế điều chế mẫu khó
class FocalLoss(nn.Module if nn is not None else object):
    """Hàm mất mát Focal Loss đa lớp triệt tiêu gradient từ mẫu dễ phân loại và tập trung vào mẫu khó.

    Công thức toán học:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Trong đó:
        - p_t là xác suất dự đoán của mô hình cho nhãn thực tế.
        - (1 - p_t)^gamma là hệ số điều chế (Modulating Factor).
        - gamma là tham số tập trung (Focusing Parameter): khi gamma = 0, hàm trở về Cross-Entropy tiêu chuẩn.
        - alpha_t là hệ số cân bằng nhãn (tùy chọn).
    """

    # Khởi tạo hàm mất mát với tham số tập trung gamma, hệ số trọng số alpha và cơ chế reduction
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        if nn is not None:
            super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    # Lan truyền xuôi tính toán giá trị tổn thất Focal Loss ổn định số học
    def forward(self, logits, targets):
        """Tính toán tổn thất Focal Loss trên mini-batch.

        Tham số:
            logits: Tensor kích thước (batch_size, num_classes) chứa logits chưa qua softmax.
            targets: Tensor kích thước (batch_size,) chứa nhãn nguyên thực tế.

        Trả về:
            Tensor tổn thất tương ứng theo cơ chế reduction đã cấu hình.
        """
        if F is None:
            raise ImportError(
                "Thư viện torch chưa được cài đặt. Vui lòng cài đặt qua lệnh: pip install torch"
            )

        # Tính toán Cross-Entropy từng mẫu để tận dụng tính ổn định số học của log-sum-exp
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Xác suất dự đoán của nhãn thực tế: p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # Hệ số điều chế triệt tiêu mẫu dễ: (1 - p_t)^gamma
        modulating_factor = (1.0 - pt) ** self.gamma
        focal_loss = modulating_factor * ce_loss

        # Áp dụng trọng số nhãn alpha nếu được cấu hình
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(logits.device)[targets]
                focal_loss = alpha_t * focal_loss
            else:
                focal_loss = self.alpha * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# Lớp Trainer tùy biến ghi đè phương thức compute_loss để áp dụng Focal Loss
class FocalLossTrainer(Trainer if Trainer is not None else object):
    """Lớp Trainer tùy biến tích hợp hàm mất mát Focal Loss cho quá trình huấn luyện Hugging Face."""

    # Khởi tạo Trainer với tham số gamma và alpha cho Focal Loss
    def __init__(self, *args, focal_gamma=2.0, focal_alpha=None, **kwargs):
        if Trainer is None:
            raise ImportError(
                "Thư viện transformers chưa được cài đặt. Vui lòng cài đặt qua lệnh: pip install transformers"
            )
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.loss_fn = FocalLoss(
            gamma=focal_gamma, alpha=focal_alpha, reduction="mean"
        )

    # Ghi đè phương thức tính toán tổn thất của Trainer bằng Focal Loss
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
    ):
        """Tính toán tổn thất trên batch sử dụng hàm mất mát Focal Loss."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Lớp Trainer tùy biến tích hợp trọng số lớp tĩnh cho Cross-Entropy Loss (hỗ trợ Checkpoint 4)
class WeightedCrossEntropyTrainer(Trainer if Trainer is not None else object):
    """Lớp Trainer áp dụng vector trọng số phạt mất cân bằng lớp vào Cross-Entropy Loss."""

    # Khởi tạo Trainer với vector trọng số lớp class_weights
    def __init__(self, *args, class_weights=None, **kwargs):
        if Trainer is None:
            raise ImportError(
                "Thư viện transformers chưa được cài đặt. Vui lòng cài đặt qua lệnh: pip install transformers"
            )
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    # Ghi đè phương thức tính toán tổn thất áp dụng trọng số lớp vào Cross-Entropy Loss
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
    ):
        """Tính toán tổn thất có trọng số lớp trên mini-batch."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
