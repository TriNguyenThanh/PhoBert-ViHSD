"""Xuất so sánh baseline và run tốt nhất của checkpoint có Macro-F1 trung bình cao nhất."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["clean", "offensive", "hate"]
SUPPORT = np.array([5548, 444, 688])
SEEDS = [42, 456, 789, 1049]


def read_matrix(path):
    """Đọc CSV và xác minh thứ tự lớp, số đếm nguyên cùng hỗ trợ tập kiểm tra."""
    frame = pd.read_csv(path, index_col=0)
    if frame.index.name != "true_label" or list(frame.index) != LABELS or list(frame.columns) != LABELS:
        raise ValueError(f"Sai nhãn hoặc thứ tự lớp: {path}")
    matrix = frame.to_numpy(dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Ma trận phải hữu hạn, kích thước 3x3: {path}")
    if (matrix < 0).any() or not np.equal(matrix, np.floor(matrix)).all():
        raise ValueError(f"Ma trận phải chứa số nguyên không âm: {path}")
    if not np.array_equal(matrix.sum(axis=1), SUPPORT):
        raise ValueError(f"Hỗ trợ lớp không khớp tập kiểm tra: {path}")
    return matrix.astype(np.int64)


def matrix_metrics(matrix):
    """Tính các chỉ số trực tiếp từ ma trận; phép chia có mẫu bằng 0 trả về 0."""
    diagonal = np.diag(matrix).astype(float)
    precision = np.divide(diagonal, matrix.sum(axis=0), out=np.zeros(3), where=matrix.sum(axis=0) != 0)
    recall = np.divide(diagonal, matrix.sum(axis=1), out=np.zeros(3), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall != 0)
    accuracy = float(diagonal.sum() / matrix.sum())
    result = {"accuracy": accuracy, "macro_precision": float(precision.mean()),
              "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()), "micro_f1": accuracy}
    for index, label in enumerate(LABELS):
        result.update({f"precision_{label}": float(precision[index]),
                       f"recall_{label}": float(recall[index]), f"f1_{label}": float(f1[index])})
    return result


def read_scores(path):
    """Đọc chỉ số của một run, từ chối tên trùng hoặc giá trị ngoài khoảng [0, 1]."""
    frame = pd.read_csv(path)
    if not {"metric", "value"}.issubset(frame.columns) or frame.metric.duplicated().any():
        raise ValueError(f"Bảng chỉ số không hợp lệ: {path}")
    values = pd.to_numeric(frame.value, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"Giá trị chỉ số không hợp lệ: {path}")
    scores = dict(zip(frame.metric, values))
    if "macro_f1" not in scores:
        raise ValueError(f"Thiếu macro_f1: {path}")
    return scores


def select_best(outputs_dir):
    """Xếp hạng theo mean Macro-F1 rồi tên checkpoint; chọn run cao nhất rồi seed nhỏ nhất."""
    aggregate_path = outputs_dir / "evaluation_scores_mean_4seeds.csv"
    aggregate = pd.read_csv(aggregate_path)
    required = {"checkpoint", "metric", "mean", "std", *(f"seed_{seed}" for seed in SEEDS)}
    if not required.issubset(aggregate.columns):
        raise ValueError("Bảng tổng hợp thiếu cột bắt buộc")
    rows = aggregate.loc[aggregate.metric == "macro_f1"].copy()
    if rows.empty or rows.checkpoint.duplicated().any():
        raise ValueError("Danh sách checkpoint rỗng hoặc trùng")
    ranking, sources = [], [aggregate_path]
    for _, row in rows.iterrows():
        checkpoint = row.checkpoint
        if not isinstance(checkpoint, str) or not re.fullmatch(r"checkpoint\d+", checkpoint):
            raise ValueError("Tên checkpoint không hợp lệ")
        mean, std = float(row["mean"]), float(row["std"])
        if not np.isfinite([mean, std]).all() or not (0 <= mean <= 1) or std < 0:
            raise ValueError(f"Mean/std không hợp lệ: {checkpoint}")
        runs = []
        for seed in SEEDS:
            path = outputs_dir / f"seed_{seed}" / checkpoint / "evaluation_scores.csv"
            score = read_scores(path)["macro_f1"]
            if not np.isclose(score, float(row[f"seed_{seed}"]), atol=1e-8, rtol=0):
                raise ValueError(f"Macro-F1 tổng hợp không khớp run: {path}")
            runs.append((seed, score))
            sources.append(path)
        if not np.isclose(np.mean([value for _, value in runs]), mean, atol=0.0001, rtol=0):
            raise ValueError(f"Mean không khớp bốn run: {checkpoint}")
        best_seed, best_score = sorted(runs, key=lambda item: (-item[1], item[0]))[0]
        ranking.append({"cp": checkpoint, "macro_f1_mean": mean, "std": std,
                        "best_seed": best_seed, "best_macro_f1": best_score})
    ranking = pd.DataFrame(ranking).sort_values(["macro_f1_mean", "cp"], ascending=[False, True]).reset_index(drop=True)
    return ranking, sources


def validate_predictions(path, matrix):
    """Chỉ đọc hai cột nhãn và xác minh 6.680 dự đoán tái tạo đúng ma trận đã lưu."""
    predictions = pd.read_csv(path, usecols=["ground_truth", "predicted_label"])
    values = predictions.to_numpy(dtype=float)
    if len(predictions) != int(SUPPORT.sum()) or not np.isfinite(values).all() or not np.isin(values, [0, 1, 2]).all():
        raise ValueError("Nhãn dự đoán hoặc số mẫu không hợp lệ")
    actual = confusion_matrix(predictions.ground_truth, predictions.predicted_label, labels=[0, 1, 2])
    if not np.array_equal(actual, matrix):
        raise ValueError("Ma trận không khớp predictions.csv")


def draw_comparison(baseline, best, checkpoint, seed, path, normalized=False):
    """Vẽ hai ma trận với chung thang màu; tùy chọn chuẩn hóa theo từng nhãn thật."""
    matrices = [baseline, best]
    if normalized:
        matrices = [matrix / matrix.sum(axis=1, keepdims=True) * 100 for matrix in matrices]
    maximum = 100 if normalized else max(matrix.max() for matrix in matrices)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), layout="constrained")
    for ax, matrix, title in zip(axes, matrices, ["Baseline (seed unknown)", f"Best checkpoint: {checkpoint}, seed {seed}"]):
        artist = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
               xlabel="Predicted label", ylabel="True label", title=title)
        for row in range(3):
            for column in range(3):
                value = matrix[row, column]
                text = f"{value:.1f}%" if normalized else str(int(value))
                ax.text(column, row, text, ha="center", va="center", fontsize=12,
                        color="white" if value > maximum / 2 else "black")
    fig.colorbar(artist, ax=axes, shrink=0.8, label="Row percentage (%)" if normalized else "Number of samples")
    fig.suptitle(f"ViHSD test set (n = {int(SUPPORT.sum()):,}); class order: 0 clean, 1 offensive, 2 hate", fontsize=12)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def generate(outputs_dir, output_dir):
    """Kiểm tra toàn bộ đầu vào trước khi ghi bảng, hình và thông tin truy xuất nguồn."""
    outputs_dir, output_dir = outputs_dir.resolve(), output_dir.resolve()
    ranking, source_paths = select_best(outputs_dir)
    selected = ranking.iloc[0]
    checkpoint, seed = str(selected.cp), int(selected.best_seed)
    best_dir = outputs_dir / f"seed_{seed}" / checkpoint
    baseline_path = outputs_dir / "baseline/confusion_matrix.csv"
    baseline_image = outputs_dir / "baseline/confusion_matrix.png"
    best_path = best_dir / "confusion_matrix.csv"
    predictions_path = best_dir / "predictions.csv"
    baseline, best = read_matrix(baseline_path), read_matrix(best_path)
    validate_predictions(predictions_path, best)
    baseline_metrics, best_metrics = matrix_metrics(baseline), matrix_metrics(best)
    reported = read_scores(best_dir / "evaluation_scores.csv")
    for metric, value in best_metrics.items():
        if metric not in reported or not np.isclose(value, reported[metric], atol=0.000051, rtol=0):
            raise ValueError(f"Chỉ số run không khớp ma trận: {metric}")
    source_paths += [baseline_image, baseline_path, best_path, predictions_path]
    source_paths = sorted(set(path.resolve() for path in source_paths))
    sources = [{"path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in source_paths]
    names = ["baseline_confusion_matrix.csv", "best_confusion_matrix.csv", "evaluation_comparison.csv",
             "checkpoint_ranking.csv", "confusion_matrix_baseline_vs_best.png",
             "confusion_matrix_baseline_vs_best_normalized.png", "provenance.json"]
    if any((output_dir / name).resolve() in source_paths for name in names):
        raise ValueError("Đường dẫn xuất sẽ ghi đè dữ liệu nguồn")
    provenance = {
        "class_order": {str(index): label for index, label in enumerate(LABELS)},
        "sample_count": int(SUPPORT.sum()), "class_support": SUPPORT.tolist(),
        "baseline": {"origin": "Manually transcribed from baseline/confusion_matrix.png", "seed": None,
                     "predictions_available": False, "validation_limit": "Counts cannot be checked against baseline predictions."},
        "selection": {"rule": "Highest numeric mean Macro-F1 across four seeds; checkpoint name ascending breaks ties. Within that checkpoint, highest reported run Macro-F1; smallest seed breaks ties.",
                      "checkpoint": checkpoint, "seed": seed, "macro_f1_mean": float(selected.macro_f1_mean),
                      "macro_f1_std": float(selected["std"]), "reported_run_macro_f1": float(selected.best_macro_f1)},
        "metrics": "Calculated directly from confusion matrices; delta_pp = 100 * (best - baseline); zero division = 0.",
        "checks": {"best_predictions_match_matrix": True, "best_metrics_match_csv_to_rounding": True,
                   "matching_class_support": True}, "sources": sources,
    }
    comparison = pd.DataFrame([{"metric": metric, "baseline": value, "best": best_metrics[metric],
                                "delta_pp": 100 * (best_metrics[metric] - value)} for metric, value in baseline_metrics.items()])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in [(names[0], baseline), (names[1], best)]:
        pd.DataFrame(matrix, index=pd.Index(LABELS, name="true_label"), columns=LABELS).to_csv(output_dir / name)
    comparison.to_csv(output_dir / names[2], index=False)
    ranking.to_csv(output_dir / names[3], index=False)
    draw_comparison(baseline, best, checkpoint, seed, output_dir / names[4])
    draw_comparison(baseline, best, checkpoint, seed, output_dir / names[5], normalized=True)
    (output_dir / names[6]).write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported {checkpoint}, seed {seed}, n={int(SUPPORT.sum())} to {output_dir}")


def main():
    """Đọc tham số CLI; mặc định sử dụng đường dẫn của repo độc lập thư mục đang chạy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/comparison")
    args = parser.parse_args()
    generate(args.outputs_dir, args.output_dir)


if __name__ == "__main__":
    main()
