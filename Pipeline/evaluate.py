import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# Hàm callback tính toán các chỉ số Accuracy, Precision, Recall, F1 trong quá trình huấn luyện
def compute_metrics_eval(eval_pred):
    """Hàm callback đánh giá tích hợp trực tiếp vào Hugging Face Trainer."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_p = precision_score(labels, preds, average="macro", zero_division=0)
    macro_r = recall_score(labels, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    per_class_f1 = f1_score(labels, preds, average=None)

    metrics = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "f1_clean": per_class_f1[0] if len(per_class_f1) > 0 else 0.0,
        "f1_offensive": per_class_f1[1] if len(per_class_f1) > 1 else 0.0,
        "f1_hate": per_class_f1[2] if len(per_class_f1) > 2 else 0.0,
    }
    return metrics


# Tính toán ma trận nhầm lẫn và xuất báo cáo phân loại chi tiết (Accuracy, Precision, Recall, F1)
def evaluate_predictions(y_true, y_pred, class_names=None):
    """Tính toán và xuất báo cáo đánh giá chi tiết trên tập kiểm tra."""
    if class_names is None:
        class_names = ["clean", "offensive", "hate"]

    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    results = {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class_precision": {name: per_class_precision[i] for i, name in enumerate(class_names)},
        "per_class_recall": {name: per_class_recall[i] for i, name in enumerate(class_names)},
        "per_class_f1": {name: per_class_f1[i] for i, name in enumerate(class_names)},
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return results


# Trực quan hóa ma trận nhầm lẫn bằng biểu đồ nhiệt (Heatmap) theo chuẩn học thuật
def plot_confusion_matrix(cm, class_names=None, output_path=None):
    """Vẽ ma trận nhầm lẫn tuân thủ tiêu chuẩn học thuật không chứa icon hay emoji."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if class_names is None:
        class_names = ["clean", "offensive", "hate"]

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        df_cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        cbar=True,
        ax=ax,
        annot_kws={"size": 12},
    )
    ax.set_title("Confusion Matrix - ViHSD Evaluation")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    return fig


# Xuất tệp dữ liệu dự đoán chi tiết gồm id, nội dung văn bản, nhãn thực tế và nhãn dự đoán
def save_predictions(y_true, y_pred, texts, output_path, ids=None):
    """Xuất tệp CSV lưu trữ kết quả dự đoán của mô hình trên tập kiểm tra."""
    if ids is None:
        ids = list(range(len(y_true)))

    df_predictions = pd.DataFrame(
        {
            "id": ids,
            "text": texts,
            "ground_truth": y_true,
            "predicted_label": y_pred,
        }
    )
    df_predictions.to_csv(output_path, index=False, encoding="utf-8")
    return df_predictions


# Xuất tệp dữ liệu ma trận nhầm lẫn dưới dạng bảng tính CSV
def save_confusion_matrix_csv(cm, class_names=None, output_path=None):
    """Lưu ma trận nhầm lẫn dưới dạng bảng DataFrame sang tệp CSV."""
    if class_names is None:
        class_names = ["clean", "offensive", "hate"]

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    if output_path:
        df_cm.to_csv(output_path)
    return df_cm


# Xuất tệp điểm số tổng hợp cốt lõi (Accuracy, Precision, Recall, F1)
def save_scores_summary(y_true, y_pred, class_names=None, output_path=None):
    """Xuất tệp CSV lưu trữ bảng điểm tổng kết các chỉ số quan trọng (Accuracy, Precision, Recall, F1)."""
    if class_names is None:
        class_names = ["clean", "offensive", "hate"]

    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None)

    metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "micro_f1",
    ]
    values = [
        round(acc, 4),
        round(macro_p, 4),
        round(macro_r, 4),
        round(macro_f1, 4),
        round(micro_f1, 4),
    ]

    for i, name in enumerate(class_names):
        metrics.extend([f"precision_{name}", f"recall_{name}", f"f1_{name}"])
        values.extend([round(per_class_p[i], 4), round(per_class_r[i], 4), round(per_class_f1[i], 4)])

    df_scores = pd.DataFrame({"metric": metrics, "value": values})
    if output_path:
        df_scores.to_csv(output_path, index=False)
    return df_scores


# Xuất tệp bảng điểm chi tiết các chỉ số đánh giá (Precision, Recall, F1-Score, Support)
def export_evaluation_report(y_true, y_pred, class_names=None, output_path=None):
    """Lưu trữ báo cáo phân loại chi tiết (Classification Report) đã được chuẩn hóa dưới dạng tệp CSV."""
    if class_names is None:
        class_names = ["clean", "offensive", "hate"]

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    accuracy_val = report_dict.pop("accuracy")

    rows = []
    for label in class_names:
        row = report_dict[label]
        rows.append(
            {
                "class": label,
                "precision": round(row["precision"], 4),
                "recall": round(row["recall"], 4),
                "f1_score": round(row["f1-score"], 4),
                "support": int(row["support"]),
            }
        )

    rows.append(
        {
            "class": "accuracy",
            "precision": None,
            "recall": None,
            "f1_score": round(accuracy_val, 4),
            "support": len(y_true),
        }
    )

    for avg_type in ["macro avg", "weighted avg"]:
        row = report_dict[avg_type]
        rows.append(
            {
                "class": avg_type,
                "precision": round(row["precision"], 4),
                "recall": round(row["recall"], 4),
                "f1_score": round(row["f1-score"], 4),
                "support": int(row["support"]),
            }
        )

    df_report = pd.DataFrame(rows)
    if output_path:
        df_report.to_csv(output_path, index=False)
    return df_report

