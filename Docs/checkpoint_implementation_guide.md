# Implementation Guide for Subsequent Checkpoints (Checkpoints 2 - 6)

- **Project:** Vietnamese Hate Speech Detection (ViHSD) Performance Enhancement
- **Backbone Architectures:** `vinai/phobert-base` (Checkpoints 1 - 5) & `vinai/phobert-large` / Alternative Transformers (Checkpoint 6)
- **Primary Evaluation Metric:** Macro-F1 Score
- **Dataset Distribution:** `0: CLEAN` (~82.7%), `1: OFFENSIVE` (~6.7%), `2: HATE` (~10.6%)

This document establishes the mandatory engineering standards, modular code structures, required artifact export formats, and detailed implementation roadmaps for the upcoming research checkpoints: Checkpoint 2, Checkpoint 3, Checkpoint 4, Checkpoint 5, and Checkpoint 6.

---

## 1. Mandatory Engineering & Execution Principles

All code implementations and experimental evaluations must strictly adhere to the following standards:

### 1.1. Google Colab Execution and Google Drive Persistence
- **No Heavy Execution on Local Machines:** Transformer model training and fine-tuning must be executed exclusively on Google Colab GPU runtimes (T4, V100, or A100).
- **Persistent Storage Initialization in Cell 1:** The first code cell must mount Google Drive at `/content/drive/MyDrive/NLP_PhoBert_ViHSD`, automatically resolving and creating `checkpoints/checkpointXX/` and `outputs/checkpointXX/` directories.

### 1.2. 100% Self-Contained Notebook Requirement
- **No External Module Imports:** Every class definition (`ViHSDPreprocessor`, `BuildDataset`, `FocalLoss`, `WeightedFocalLoss`), custom training loop (`CustomTrainer`), evaluation callback, and plotting routine must be directly embedded (inlined) within the notebook cells.
- **Rationale:** Ensures notebooks run completely standalone on Google Colab or external compute environments without triggering `ModuleNotFoundError` or depending on cloned local file paths.

### 1.3. In-Place Notebook Modification
- **Do Not Overwrite Notebooks via Scripts:** When modifying existing `.ipynb` files, always edit the target cells in place using cell-level tools (`NotebookEdit`). Never regenerate the entire `.ipynb` file from a standalone generator script.
- **Rationale:** Overwriting the notebook file deletes all previous cell outputs, loss curves, and training logs that required substantial GPU compute time to generate.

### 1.4. Academic Tone and Code Standards
- **Strictly Zero Emojis or Icons:** No decorative icons or emojis anywhere in the repository, including code comments, Markdown headers, `print()` statements, plot titles, axis labels, and Git commit messages.
- **Standard Vietnamese Comments with Diacritics:** All critical algorithmic functions, custom PyTorch loss classes, and data transformation boundaries must include explanatory comments in grammatically standard Vietnamese with full diacritics.
- **Per-Cell Variable Scoping:** Declare all cell-specific hyperparameters, file paths, thresholds, and execution flags at the very top of each corresponding cell.

### 1.5. Modern Transformers Compatibility
- Always use `eval_strategy="epoch"` (replacing the deprecated `evaluation_strategy`).
- Do not pass `no_cuda` to `TrainingArguments` (removed in `transformers` v4.44+; CUDA is detected automatically).
- Load the PhoBERT tokenizer with `AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)`.

---

## 2. Standardized Artifact Export Specification

At the conclusion of model evaluation on the test set (`test_dataset`), every checkpoint notebook **must conclude with a dedicated final section** that exports exactly four standardized artifact files into `outputs/checkpointXX/`:

| File Name | Format | Schema & Description |
| :--- | :---: | :--- |
| **`predictions.csv`** | CSV | Detailed sample-level predictions containing four columns: `id`, `text`, `ground_truth`, `predicted_label`. |
| **`confusion_matrix.png`** | PNG | Academic heatmap visualization (size 6x5 inches, Blues colormap, font size 12, plain text labels). |
| **`confusion_matrix.csv`** | CSV | Raw 3x3 confusion matrix values with rows/columns corresponding to `clean`, `offensive`, `hate`. |
| **`evaluation_scores.csv`** | CSV | Core performance summary table with two columns: `metric` and `value` (`macro_f1`, `micro_f1`, `accuracy`, `f1_clean`, `f1_offensive`, `f1_hate`). |
| **`classification_report.csv`** | CSV | Structured class-wise classification report: `class`, `precision`, `recall`, `f1_score`, `support` (integer support counts; clean accuracy row without duplicated values). |

### Canonical Artifact Export Code Template (Shared Across All Checkpoints)
```python
# Cấu hình đường dẫn lưu trữ các tệp kết quả đầu ra
checkpoint_id = "02"  # Thay đổi theo checkpoint tương ứng: "02", "03", "04", "05"
predictions_csv_path = os.path.join(outputs_dir, "predictions.csv")
scores_csv_path = os.path.join(outputs_dir, "evaluation_scores.csv")
report_csv_path = os.path.join(outputs_dir, "classification_report.csv")
cm_png_path = os.path.join(outputs_dir, f"confusion_matrix_checkpoint{checkpoint_id}.png")
cm_csv_path = os.path.join(outputs_dir, "confusion_matrix.csv")

# Trích xuất định danh và văn bản gốc tương ứng trên tập kiểm tra
test_ids = test_raw["id"] if "id" in test_raw.columns else list(range(len(test_y)))
test_texts = test_raw["free_text"].tolist() if len(test_raw) == len(test_y) else test_X

# 1. Xuất tệp kết quả dự đoán chi tiết (predictions.csv)
df_predictions = pd.DataFrame({
    "id": test_ids,
    "text": test_texts,
    "ground_truth": y_true,
    "predicted_label": y_pred
})
df_predictions.to_csv(predictions_csv_path, index=False, encoding="utf-8")
print(f"1. Đã xuất tệp dự đoán: {predictions_csv_path} ({len(df_predictions)} mẫu)")

# 2. Xuất ma trận nhầm lẫn dạng biểu đồ nhiệt và bảng dữ liệu CSV
df_cm = pd.DataFrame(results["confusion_matrix"], index=class_names, columns=class_names)
df_cm.index.name = "true_label"
df_cm.to_csv(cm_csv_path)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="d", cbar=True, ax=ax, annot_kws={"size": 12})
ax.set_title(f"Confusion Matrix - ViHSD Checkpoint {checkpoint_id}")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig(cm_png_path, dpi=300)
plt.close(fig)
print(f"2. Đã xuất biểu đồ và dữ liệu ma trận nhầm lẫn: {cm_png_path}, {cm_csv_path}")

# 3. Xuất tệp điểm số tổng hợp cốt lõi (evaluation_scores.csv)
acc_val = round(float(results.get("accuracy", accuracy_score(y_true, y_pred))), 4)
macro_p_val = round(float(results.get("macro_precision", precision_score(y_true, y_pred, average="macro", zero_division=0))), 4)
macro_r_val = round(float(results.get("macro_recall", recall_score(y_true, y_pred, average="macro", zero_division=0))), 4)
per_class_p = results.get("per_class_precision", {name: round(precision_score(y_true, y_pred, average=None, zero_division=0)[i], 4) for i, name in enumerate(class_names)})
per_class_r = results.get("per_class_recall", {name: round(recall_score(y_true, y_pred, average=None, zero_division=0)[i], 4) for i, name in enumerate(class_names)})

scores_summary = {
    "metric": [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "micro_f1",
        "precision_clean",
        "recall_clean",
        "f1_clean",
        "precision_offensive",
        "recall_offensive",
        "f1_offensive",
        "precision_hate",
        "recall_hate",
        "f1_hate"
    ],
    "value": [
        acc_val,
        macro_p_val,
        macro_r_val,
        round(results["macro_f1"], 4),
        round(results["micro_f1"], 4),
        round(per_class_p["clean"], 4),
        round(per_class_r["clean"], 4),
        round(results["per_class_f1"]["clean"], 4),
        round(per_class_p["offensive"], 4),
        round(per_class_r["offensive"], 4),
        round(results["per_class_f1"]["offensive"], 4),
        round(per_class_p["hate"], 4),
        round(per_class_r["hate"], 4),
        round(results["per_class_f1"]["hate"], 4)
    ]
}
df_scores = pd.DataFrame(scores_summary)
df_scores.to_csv(scores_csv_path, index=False)
print(f"3. Đã xuất tệp điểm số tổng hợp: {scores_csv_path}")

# 4. Xuất tệp báo cáo phân loại chi tiết đã chuẩn hóa (classification_report.csv)
report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
accuracy_val = report_dict.pop("accuracy")

rows = []
for label in class_names:
    row = report_dict[label]
    rows.append({
        "class": label,
        "precision": round(row["precision"], 4),
        "recall": round(row["recall"], 4),
        "f1_score": round(row["f1-score"], 4),
        "support": int(row["support"])
    })

# Hàng accuracy tổng thể với support là tổng số mẫu kiểm tra
rows.append({
    "class": "accuracy",
    "precision": None,
    "recall": None,
    "f1_score": round(accuracy_val, 4),
    "support": len(y_true)
})

# Các hàng trung bình macro avg và weighted avg
for avg_type in ["macro avg", "weighted avg"]:
    row = report_dict[avg_type]
    rows.append({
        "class": avg_type,
        "precision": round(row["precision"], 4),
        "recall": round(row["recall"], 4),
        "f1_score": round(row["f1-score"], 4),
        "support": int(row["support"])
    })

df_report = pd.DataFrame(rows)
df_report.to_csv(report_csv_path, index=False)
print(f"4. Đã xuất tệp báo cáo phân loại chi tiết: {report_csv_path}")

print("\nBảng điểm số tổng kết:")
print(df_scores.to_string(index=False))

print("\nBảng phân loại chi tiết:")
print(df_report.to_string(index=False))
```

---

## 3. Checkpoint 2 Specification: Training Strategy Optimization

- **Notebook File:** `Notebook/checkpoint02_training_strategies.ipynb`
- **Output Directory:** `outputs/checkpoint02/`
- **Primary Objective:** Address the shortcomings of baseline training: increase sequence length to `max_length = 256`, extend training duration (8-10 epochs), apply differential learning rates, and save the best model based on validation Macro-F1 with Early Stopping.

### 3.1. Technical Rationale & Hypotheses
1. **Extended Sequence Length (`max_length = 256`):** The baseline notebook truncated texts at 100 tokens. Extending to 256 tokens fully utilizes PhoBERT's positional embedding capacity (`max_position_embeddings = 256`), preventing premature truncation of hate speech targets and offensive stances located at sentence ends.
2. **Differential Learning Rates:** A smaller learning rate ($2 \times 10^{-5}$) preserves general syntactic representations in the pre-trained RoBERTa backbone, while a larger learning rate ($1 \times 10^{-4}$) accelerates convergence in the randomly initialized classification head.
3. **Best Checkpoint Selection & Early Stopping:** Evaluates `eval_macro_f1` at each epoch. Restores the exact checkpoint achieving peak validation Macro-F1 (`load_best_model_at_end=True`), preventing late-epoch degradation caused by majority-class overfitting.

### 3.2. Hyperparameter Configuration

| Parameter | Checkpoint 2 Value | Baseline Value | Technical Rationale |
| :--- | :---: | :---: | :--- |
| `max_seq_length` | **256** | 100 | Covers full social media comment context without truncation. |
| `num_train_epochs` | **8 - 10** | 2 | Allows thorough convergence on minority and hard samples. |
| `per_device_train_batch_size` | **16** | 16 | Compatible with GPU VRAM limits at sequence length 256. |
| `gradient_accumulation_steps` | **2** | 1 | Yields effective batch size of $16 \times 2 = 32$. |
| `fp16` | **True** | False | Halves VRAM usage and doubles training throughput on Nvidia GPUs. |
| `backbone_lr` | **$2 \times 10^{-5}$** | $2 \times 10^{-5}$ | Preserves pre-trained language knowledge. |
| `head_lr` | **$1 \times 10^{-4}$** | $2 \times 10^{-5}$ | Accelerates classification head adaptation. |
| `lr_scheduler_type` | **cosine** | linear | Smooth convergence curve with linear warmup (10% of total steps). |
| `metric_for_best_model` | **eval_macro_f1** | loss / none | Directly optimizes the primary evaluation metric. |
| `early_stopping_patience` | **2 - 3** | None | Automatically halts training when validation Macro-F1 plateaus. |

### 3.3. Differential Optimizer Implementation
```python
def create_differential_optimizer(model, backbone_lr=2e-5, head_lr=1e-4, weight_decay=0.01):
    """Khởi tạo bộ tối ưu hóa AdamW với tốc độ học phân tầng cho backbone và classification head."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": backbone_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.classifier.named_parameters()],
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters)
```

### 3.4. Cell-by-Cell Notebook Blueprint
1. **Cell 0 (Markdown):** Checkpoint 2 objectives, theoretical background, and experimental design.
2. **Cell 1 (Code):** Mount Google Drive, set up directory paths for checkpoints and outputs.
3. **Cell 2 (Code):** Dependency installation (`pyvi`, `transformers`, `accelerate`, `scikit-learn`, `seaborn`, `matplotlib`).
4. **Cell 3 (Code):** Deterministic random seed initialization (`seed = 42`).
5. **Cell 4 - 5 (Markdown & Code):** Raw dataset loading and pre-pipeline verification.
6. **Cell 6 - 7 (Markdown & Code):** Inlined `ViHSDPreprocessor` class.
7. **Cell 8 - 9 (Markdown & Code):** Execution of data preprocessing and post-pipeline assertion checks.
8. **Cell 10 - 11 (Markdown & Code):** Inlined `BuildDataset` class and PhoBERT tokenization (`max_length = 256`).
9. **Cell 12 - 13 (Markdown & Code):** Inlined `compute_metrics_eval`, model instantiation, differential optimizer setup, and `TrainingArguments` (`eval_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="eval_macro_f1"`).
10. **Cell 14 - 15 (Markdown & Code):** Training execution with `EarlyStoppingCallback(early_stopping_patience=2)`.
11. **Cell 16 - 17 (Markdown & Code):** Convergence plotting (Loss and validation Macro-F1 curves across epochs).
12. **Cell 18 - 19 (Markdown & Code):** Prediction on `test_dataset` and detailed evaluation reporting.
13. **Cell 20 - 21 (Markdown & Code):** Cumulative benchmark comparison against Baseline and Checkpoint 1.
14. **Cell 22 - 23 (Markdown & Code):** Standardized Artifacts Export (5 required files).

---

## 4. Checkpoint 3 Specification: Focal Loss for Hard Example Mining

- **Notebook File:** `Notebook/checkpoint03_focal_loss.ipynb`
- **Output Directory:** `outputs/checkpoint03/`
- **Primary Objective:** Mitigate majority-class gradient dominance by introducing Focal Loss to down-weight easy examples ($p_t \to 1$) and focus optimization on difficult minority samples.

### 4.1. Mathematical Formulation
$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$
Where:
- $p_t$ is the model's estimated probability for the true class.
- $(1 - p_t)^\gamma$ is the modulating factor.
- $\gamma \ge 0$ is the tunable focusing parameter. When $\gamma = 0$, Focal Loss reduces to standard Cross-Entropy.
- In Checkpoint 3, $\alpha$ is held constant ($\alpha = 1$) to isolate the effect of dynamic sample hardness modulation without static class weighting.

### 4.2. Inlined PyTorch Focal Loss & Custom Trainer
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class FocalLoss(nn.Module):
    """Hàm mất mát Focal Loss triệt tiêu gradient từ mẫu dễ phân loại và tập trung vào mẫu khó."""
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # pt: xác suất dự đoán của nhãn thực tế
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class FocalLossTrainer(Trainer):
    """Lớp Trainer tùy biến ghi đè phương thức compute_loss để áp dụng Focal Loss."""
    def __init__(self, *args, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = FocalLoss(gamma=gamma)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
```

### 4.3. Experimental Protocol for Checkpoint 3
- Inherit the optimal training strategy from Checkpoint 2 (`max_length = 256`, differential learning rates, best model checkpointing).
- Train using `FocalLossTrainer` with $\gamma = 2.0$ (optionally conducting a grid search over $\gamma \in \{1.0, 1.5, 2.0, 2.5\}$).
- Analyze the shift in minority-class recall (reduction of false negatives where OFFENSIVE or HATE was misclassified as CLEAN).
- Export all five standardized artifacts into `outputs/checkpoint03/`.

---

## 5. Checkpoint 4 Specification: Class-Weighted Cross-Entropy Loss

- **Notebook File:** `Notebook/checkpoint04_class_weights.ipynb`
- **Output Directory:** `outputs/checkpoint04/`
- **Primary Objective:** Counterbalance class frequency disparity by applying static penalty weights inversely proportional to training class frequencies.

### 5.1. Mathematical Formulation
1. **Inverse Class Frequency (ICF):**
   $$w_c = \frac{N}{K \cdot N_c}$$
   Where $N$ is the total number of training samples ($24,046$), $K = 3$ is the number of classes, and $N_c$ is the frequency of class $c$.
   Empirical weights for ViHSD: $w_{\text{clean}} \approx 0.40$, $w_{\text{offensive}} \approx 4.95$, $w_{\text{hate}} \approx 3.18$.
2. **Smoothed Inverse Frequency:**
   $$w_c = \sqrt{\frac{N}{N_c}}$$
   Dampens extreme penalty disparities to prevent gradient instability on the rarest class.

### 5.2. Inlined Weighted Trainer
```python
import torch
import torch.nn as nn
from transformers import Trainer

class WeightedCrossEntropyTrainer(Trainer):
    """Lớp Trainer áp dụng vector trọng số phạt mất cân bằng lớp vào Cross-Entropy Loss."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
```

### 5.3. Experimental Protocol for Checkpoint 4
- Compute class weight tensor directly from `train_y` using ICF formulation.
- Train the model using `WeightedCrossEntropyTrainer` under Checkpoint 2 training configurations.
- Conduct precision-recall trade-off analysis: observe how much minority-class recall increases versus any potential precision degradation in the majority CLEAN class.
- Export all five standardized artifacts into `outputs/checkpoint04/`.

---

## 6. Checkpoint 5 Specification: Combined Weighted Focal Loss & Ablation Study

- **Notebook File:** `Notebook/checkpoint05_combined_focal_classweight.ipynb`
- **Output Directory:** `outputs/checkpoint05/`
- **Primary Objective:** Unify static class weighting ($\alpha_c$ from Checkpoint 4) and dynamic sample hardness modulation ($\gamma$ from Checkpoint 3) into a single composite objective, concluding with a comprehensive 6-way ablation benchmark and qualitative error analysis.

### 6.1. Mathematical Formulation of Weighted Focal Loss
$$\text{FL}_{\text{combined}}(p_t) = -\alpha_c (1 - p_t)^\gamma \log(p_t)$$
Where:
- $\alpha_c$ is the static class weight vector derived from the optimal weighting method in Checkpoint 4.
- $\gamma \in [1.5, 2.0]$ is the focusing parameter optimized in Checkpoint 3.

### 6.2. Inlined Weighted Focal Loss Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class WeightedFocalLoss(nn.Module):
    """Hàm mất mát tích hợp đồng thời trọng số lớp tĩnh và hệ số điều chế mẫu khó."""
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        modulating_term = (1.0 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            loss = alpha_t * modulating_term * ce_loss
        else:
            loss = modulating_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class CombinedLossTrainer(Trainer):
    """Lớp Trainer tích hợp hàm mất mát Weighted Focal Loss."""
    def __init__(self, *args, alpha=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = WeightedFocalLoss(alpha=alpha, gamma=gamma)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
```

### 6.3. Cumulative Ablation Benchmark Matrix
Checkpoint 5 concludes the research roadmap with a master ablation comparison table comparing all six experimental stages evaluated on the exact same test set:

| Model / Configuration | Preprocessing | Max Length | Training Strategy | Loss Function | Class Weight ($\alpha$) | Focal Factor ($\gamma$) | Macro-F1 (Test) | F1 Clean | F1 Offensive | F1 Hate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Checkpoint 0 (Baseline)** | Stopwords stripped, deEmojify | 100 | 2 Epochs, final checkpoint | Cross-Entropy | None | 0.0 | 0.6276 | 0.9020 | 0.4610 | 0.5200 |
| **Checkpoint 1 (Preprocessing)** | Normalized (Teencode, Emojis, PyVi) | 100 | 2 Epochs, final checkpoint | Cross-Entropy | None | 0.0 | 0.6574 | 0.9408 | 0.4200 | 0.6115 |
| **Checkpoint 2 (Training Strategy)**| Normalized | 256 | 8-10 Epochs, EarlyStop, Diff LR | Cross-Entropy | None | 0.0 | *Record* | *Record* | *Record* | *Record* |
| **Checkpoint 3 (Focal Loss)** | Normalized | 256 | 8-10 Epochs, EarlyStop, Diff LR | Focal Loss | None | 2.0 | *Record* | *Record* | *Record* | *Record* |
| **Checkpoint 4 (Class Weights)** | Normalized | 256 | 8-10 Epochs, EarlyStop, Diff LR | Weighted CE | ICF Vector | 0.0 | *Record* | *Record* | *Record* | *Record* |
| **Checkpoint 5 (Combined)** | Normalized | 256 | 8-10 Epochs, EarlyStop, Diff LR | Weighted Focal Loss| ICF Vector | 1.5 - 2.0 | *Record* | *Record* | *Record* | *Record* |

### 6.4. Qualitative Error Analysis
Extract and inspect misclassified sample cohorts:
1. `CLEAN` falsely predicted as `OFFENSIVE` or `HATE` (false alarms / over-penalization).
2. `OFFENSIVE` misclassified as `HATE` (boundary ambiguity between vulgar slang and targeted hate).
3. `HATE` misclassified as `CLEAN` (subtle sarcasm, veiled threats, or out-of-dictionary slang).

---

## 7. Checkpoint 6 Specification: Alternative Backbone Architecture (PhoBERT-large / Alternative Model)

- **Notebook File:** `Notebook/checkpoint06_alternative_model.ipynb`
- **Output Directory:** `outputs/checkpoint06/`
- **Primary Objective:** Investigate the empirical impact of scaling up the foundation model capacity (`vinai/phobert-large` with 370M parameters, 24 layers, 1024 hidden size) or alternative transformer backbones on detecting subtle, figurative, and sarcastic Vietnamese hate speech compared to `vinai/phobert-base` (135M parameters, 12 layers, 768 hidden size).

### 7.1. Technical Rationale & Hardware Optimization
1. **Model Capacity Scaling:** Moving from 12 layers to 24 layers increases the model's capacity to represent multi-hop contextual dependencies and nuanced linguistic patterns typical in social media hate speech (e.g. veiled threats, sarcasm, and sarcasm-laden insults).
2. **GPU Memory & Batch Configuration:**
   - PhoBERT-large triples the memory footprint of weights and activations.
   - At `max_length = 256`, a batch size of 16 risks GPU Out-Of-Memory (OOM) on standard 16GB Google Colab GPUs (Nvidia T4/V100).
   - Solution: Set `per_device_train_batch_size = 8` and `gradient_accumulation_steps = 4`, maintaining the exact same effective batch size of $8 \times 4 = 32$.
3. **Differential Learning Rate for 24-Layer Transformer:**
   - Large pre-trained transformers are more prone to representation destabilization (catastrophic forgetting) when fine-tuned with high learning rates.
   - Reduce backbone learning rate to `backbone_lr = 1e-5` (vs `2e-5` on base), while keeping `head_lr = 1e-4`.

### 7.2. Hyperparameter Comparison Table

| Parameter | Checkpoint 6 (Large Model) | Checkpoint 2 - 5 (Base Model) | Rationale |
| :--- | :---: | :---: | :--- |
| `pretrained_model_name` | **`vinai/phobert-large`** | `vinai/phobert-base` | Scales representation capacity to 370M parameters. |
| `max_seq_length` | **256** | 256 | Covers full context without truncation. |
| `per_device_train_batch_size` | **8** | 16 | Prevents CUDA OOM on 16GB GPU VRAM. |
| `gradient_accumulation_steps` | **4** | 2 | Yields identical effective batch size of 32. |
| `backbone_lr` | **$1 \times 10^{-5}$** | $2 \times 10^{-5}$ | Stabilizes 24 pre-trained transformer layers. |
| `head_lr` | **$1 \times 10^{-4}$** | $1 \times 10^{-4}$ | Accelerates classification head convergence. |
| `lr_scheduler_type` | **cosine** | cosine | Smooth annealing with 10% linear warmup. |
| `loss_function` | **Focal Loss ($\gamma = 2.0$)** | Variable per checkpoint | Integrates proven hard-example mining. |
| `metric_for_best_model` | **eval_macro_f1** | eval_macro_f1 | Optimizes target evaluation metric directly. |
| `early_stopping_patience` | **2** | 2 | Automatically halts on validation plateau. |

### 7.3. Cumulative Ablation Benchmark Matrix Across All Checkpoints

| Model / Configuration | Preprocessing | Max Length | Training Strategy | Loss Function | Class Weight ($\alpha$) | Focal Factor ($\gamma$) | Macro-F1 (Test) | F1 Clean | F1 Offensive | F1 Hate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Checkpoint 0 (Baseline)** | Stopwords stripped, deEmojify | 100 | 2 Epochs, final checkpoint | Cross-Entropy | None | 0.0 | 0.6276 | 0.9020 | 0.4610 | 0.5200 |
| **Checkpoint 1 (Preprocessing)** | Normalized (Teencode, Emojis, PyVi) | 100 | 2 Epochs, final checkpoint | Cross-Entropy | None | 0.0 | 0.6574 | 0.9408 | 0.4200 | 0.6115 |
| **Checkpoint 2 (Training Strategy)**| Normalized | 256 | 8-10 Epochs, EarlyStop, Diff LR | Cross-Entropy | None | 0.0 | *Record* | *Record* | *Record* | *Record* |
| **Checkpoint 3 (Focal Loss)** | Normalized | 256 | 8-10 Epochs, EarlyStop, Diff LR | Focal Loss | None | 2.0 | *Record* | *Record* | *Record* | *Record* |
| **Checkpoint 4 (Class Weights)** | Normalized | 256 | 8-10 Epochs, EarlyStop, Diff LR | Weighted CE | ICF Vector | 0.0 | *Record* | *Record* | *Record* | *Record* |
| **Checkpoint 5 (Combined)** | Normalized | 256 | 8-10 Epochs, EarlyStop, Diff LR | Weighted Focal Loss| ICF Vector | 1.5 - 2.0 | *Record* | *Record* | *Record* | *Record* |
| **Checkpoint 6 (PhoBERT-large)** | Normalized | 256 | 8 Epochs, BS 8x4=32, Diff LR | Focal Loss / Combined | Optional | 2.0 | *Record* | *Record* | *Record* | *Record* |

---

## 8. Pre-flight & Post-flight Verification Checklist

Prior to approving any checkpoint notebook as complete, confirm every item below:

- [ ] **Colab Environment:** Cell 1 cleanly mounts Google Drive and establishes persistent directories under `MyDrive/NLP_PhoBert_ViHSD`.
- [ ] **100% Self-Contained:** Zero imports from local `.py` scripts or the `Pipeline/` folder. All classes and helper functions are defined inline.
- [ ] **Preserved Outputs:** Modifying existing `.ipynb` files was conducted strictly via `NotebookEdit`, leaving cell execution outputs intact.
- [ ] **Zero Decorative Emojis/Icons:** Confirmed absence of emojis or symbols across code, comments, Markdown, and plots.
- [ ] **Vietnamese Comments with Diacritics:** All custom classes (`FocalLoss`, `WeightedTrainer`, etc.) and non-trivial logic blocks include standard Vietnamese explanatory comments.
- [ ] **Artifact Verification in `outputs/checkpointXX/`:**
  - `predictions.csv` (contains `id`, `text`, `ground_truth`, `predicted_label`).
  - `confusion_matrix_checkpointXX.png` (readable academic plot).
  - `confusion_matrix.csv` (3x3 CSV table).
  - `evaluation_scores.csv` (standardized `metric, value` format).
  - `classification_report.csv` (clean per-class metrics with integer support counts).
