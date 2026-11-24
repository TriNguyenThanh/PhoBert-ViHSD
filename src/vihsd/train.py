import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    EarlyStoppingCallback = None
    Trainer = None
    TrainingArguments = None


# Khởi tạo bộ tối ưu hóa AdamW với tốc độ học phân tầng cho backbone PhoBERT và classification head
def create_differential_optimizer(
    model, backbone_lr=2e-5, head_lr=1e-4, weight_decay=0.01
):
    """Khởi tạo bộ tối ưu hóa AdamW với tốc độ học phân tầng (Differential Learning Rates).

    Tham số backbone RoBERTa của PhoBERT được gán tốc độ học nhỏ (backbone_lr) nhằm bảo tồn
    tri thức ngôn ngữ tiền huấn luyện. Tầng phân loại (classification head) mới khởi tạo
    được gán tốc độ học lớn hơn (head_lr) để tăng tốc độ thích ứng.
    """
    if torch is None:
        raise ImportError(
            "Thư viện torch chưa được cài đặt. Vui lòng cài đặt qua lệnh: pip install torch"
        )

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        # Nhóm 1: Các tham số backbone áp dụng weight decay
        {
            "params": [
                p
                for n, p in model.roberta.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        },
        # Nhóm 2: Các tham số backbone không áp dụng weight decay (bias, LayerNorm)
        {
            "params": [
                p
                for n, p in model.roberta.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "lr": backbone_lr,
            "weight_decay": 0.0,
        },
        # Nhóm 3: Toàn bộ tham số tầng phân loại tuyến tính (Classification Head)
        {
            "params": [
                p for n, p in model.classifier.named_parameters() if p.requires_grad
            ],
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
    ]

    return torch.optim.AdamW(optimizer_grouped_parameters)


# Thiết lập cấu hình huấn luyện chuẩn hóa cho Checkpoint 2
def get_training_arguments(
    output_dir,
    num_train_epochs=8,
    train_batch_size=16,
    eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    metric_for_best_model="eval_macro_f1",
    fp16=None,
    seed=42,
    logging_steps=50,
    save_total_limit=2,
):
    """Tạo đối tượng TrainingArguments chuẩn hóa cho chiến lược huấn luyện tối ưu."""
    if TrainingArguments is None:
        raise ImportError(
            "Thư viện transformers chưa được cài đặt. Vui lòng cài đặt qua lệnh: pip install transformers"
        )

    if fp16 is None:
        fp16 = torch.cuda.is_available() if torch is not None else False

    training_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lr_scheduler_type": lr_scheduler_type,
        "do_eval": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": True,
        "fp16": fp16,
        "logging_steps": logging_steps,
        "seed": seed,
    }

    import inspect
    sig_params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in sig_params:
        training_kwargs["warmup_ratio"] = warmup_ratio
    else:
        # Với các phiên bản transformers không nhận warmup_ratio trực tiếp, quy đổi sang warmup_steps
        approx_total_steps = (
            24048 // (train_batch_size * gradient_accumulation_steps)
        ) * num_train_epochs
        training_kwargs["warmup_steps"] = max(1, int(approx_total_steps * warmup_ratio))

    training_args = TrainingArguments(**training_kwargs)
    return training_args


# Thiết lập đối tượng Trainer tích hợp Early Stopping và bộ tối ưu hóa phân tầng
def setup_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    compute_metrics=None,
    optimizer=None,
    early_stopping_patience=2,
):
    """Khởi tạo đối tượng Trainer với các cơ chế kiểm soát tối ưu hóa và dừng sớm."""
    if Trainer is None:
        raise ImportError(
            "Thư viện transformers chưa được cài đặt. Vui lòng cài đặt qua lệnh: pip install transformers"
        )

    callbacks = []
    if early_stopping_patience is not None and early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            )
        )

    optimizers = (optimizer, None) if optimizer is not None else (None, None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
    )
    return trainer


# Trực quan hóa đồ thị hội tụ huấn luyện (Loss và Validation Macro-F1 qua các Epoch)
def plot_training_curves(log_history, output_path=None):
    """Trích xuất lịch sử huấn luyện từ Trainer và vẽ đồ thị hội tụ theo chuẩn học thuật."""
    train_epochs, train_losses = [], []
    eval_epochs, eval_losses, eval_macro_f1s = [], [], []

    for log in log_history:
        if "eval_macro_f1" in log:
            eval_epochs.append(log.get("epoch", len(eval_epochs) + 1))
            eval_macro_f1s.append(log.get("eval_macro_f1", 0.0))
            if "eval_loss" in log:
                eval_losses.append(log.get("eval_loss", 0.0))
        elif "loss" in log and "epoch" in log:
            train_epochs.append(log.get("epoch", 0.0))
            train_losses.append(log.get("loss", 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Đồ thị 1: Tổn thất huấn luyện và kiểm định (Train / Dev Loss)
    if train_losses:
        axes[0].plot(train_epochs, train_losses, label="Train Loss", color="#1f77b4", alpha=0.7)
    if eval_losses:
        axes[0].plot(
            eval_epochs,
            eval_losses,
            label="Validation Loss",
            color="#d62728",
            marker="o",
            linewidth=2,
        )
    axes[0].set_title("Training and Validation Loss Across Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Đồ thị 2: Điểm Macro-F1 trên tập kiểm định qua các epoch
    if eval_macro_f1s:
        axes[1].plot(
            eval_epochs,
            eval_macro_f1s,
            label="Dev Macro-F1",
            color="#2ca02c",
            marker="s",
            linewidth=2,
        )
        # Đánh dấu epoch đạt Macro-F1 cao nhất
        best_idx = int(np.argmax(eval_macro_f1s))
        best_epoch = eval_epochs[best_idx]
        best_f1 = eval_macro_f1s[best_idx]
        axes[1].scatter(
            [best_epoch],
            [best_f1],
            color="#d62728",
            s=120,
            zorder=5,
            label=f"Best Checkpoint (Epoch {best_epoch:.1f}, F1: {best_f1:.4f})",
        )

    axes[1].set_title("Validation Macro-F1 Across Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1 Score")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)

    return fig
