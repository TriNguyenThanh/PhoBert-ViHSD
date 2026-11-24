import json
import os


def create_checkpoint01_notebook():
    cells = []

    def add_markdown(source_lines):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_lines[:-1]] + [source_lines[-1]]
        })

    def add_code(source_lines):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_lines[:-1]] + [source_lines[-1]]
        })

    # Cell 0: Header Markdown
    add_markdown([
        "# Checkpoint 01: Thiết Lập Baseline Kết Hợp Cân Bằng Lớp Tĩnh (Smoothed Class Weights)",
        "",
        "Nghiên cứu: Nâng cao hiệu năng mô hình phát hiện phát ngôn thù ghét trên mạng xã hội tiếng Việt (ViHSD).",
        "",
        "Mục tiêu khoa học:",
        "1. Tái lập mô hình đường cơ sở (Baseline) theo bài báo của Lưu et al. (2021) trên mô hình `vinai/phobert-base`.",
        "2. Giữ nguyên cấu hình huấn luyện cơ sở: 2 epochs, max_length = 100, batch size = 16, learning rate = 2e-5, không tiền xử lý dữ liệu sâu.",
        "3. Khảo sát phương pháp cân bằng phân bố lớp tĩnh Smoothed Class Weights (làm mịn căn bậc hai):",
        "   $$w_c = \\sqrt{\\frac{N}{N_c}} \\bigg/ \\left( \\frac{1}{K} \\sum_{j=0}^{K-1} \\sqrt{\\frac{N}{N_j}} \\right)$$",
        "4. Kiểm soát tỷ lệ trọng số giữa lớp thiểu số và đa số ở mức lý tưởng (~3.52 lần), khắc phục hiện tượng phạt quá mức gây sụp đổ Precision của lớp CLEAN khi dùng nghịch đảo tuyến tính.",
        "5. Xuất đầy đủ 4 tệp Mandatory Artifacts theo quy chuẩn vào thư mục `outputs/checkpoint01/`.",
        "",
        "Ghi chú: Toàn bộ mã nguồn định nghĩa Dataset, tính toán trọng số, hàm mất mát và đo lường đánh giá được tích hợp trực tiếp 100% trong notebook nhằm đảm bảo khả năng thực thi độc lập trên Google Colab."
    ])

    # Cell 1: Environment Setup & Google Drive Mounting
    add_code([
        "import os",
        "import sys",
        "",
        "# Thiết lập môi trường làm việc và kết nối Google Drive trên Google Colab",
        "try:",
        "    from google.colab import drive",
        "    drive.mount('/content/drive')",
        "    project_dir = '/content/drive/MyDrive/NLP_PhoBert_ViHSD'",
        "except ImportError:",
        "    project_dir = os.path.abspath('..')",
        "",
        "checkpoints_dir = os.path.join(project_dir, 'checkpoints', 'checkpoint01')",
        "outputs_dir = os.path.join(project_dir, 'outputs', 'checkpoint01')",
        "",
        "os.makedirs(checkpoints_dir, exist_ok=True)",
        "os.makedirs(outputs_dir, exist_ok=True)",
        "",
        "print(f'Thư mục dự án: {project_dir}')",
        "print(f'Thư mục lưu checkpoint: {checkpoints_dir}')",
        "print(f'Thư mục lưu kết quả: {outputs_dir}')"
    ])

    # Cell 2: Dependencies Installation
    add_code([
        "# Cài đặt các thư viện cần thiết trên môi trường Google Colab",
        "!pip install -q transformers accelerate scikit-learn seaborn matplotlib pandas numpy torch"
    ])

    # Cell 3: Fixed Random Seed
    add_code([
        "# Thiết lập hạt giống ngẫu nhiên cố định để đảm bảo tính tái lập thực nghiệm",
        "random_seed = 42",
        "",
        "import os",
        "import random",
        "import numpy as np",
        "import torch",
        "",
        "os.environ['PYTHONHASHSEED'] = str(random_seed)",
        "random.seed(random_seed)",
        "np.random.seed(random_seed)",
        "torch.manual_seed(random_seed)",
        "if torch.cuda.is_available():",
        "    torch.cuda.manual_seed(random_seed)",
        "    torch.cuda.manual_seed_all(random_seed)",
        "    torch.backends.cudnn.deterministic = True",
        "    torch.backends.cudnn.benchmark = False",
        "",
        "print(f'Thiết lập random seed thành công: {random_seed}')",
        "print(f'Thiết bị tính toán: {\"CUDA (GPU)\" if torch.cuda.is_available() else \"CPU\"}')",
        "if torch.cuda.is_available():",
        "    print(f'Tên GPU: {torch.cuda.get_device_name(0)}')",
        "    print(f'Dung lượng VRAM khả dụng: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')"
    ])

    # Cell 4: Markdown Section 1
    add_markdown([
        "## 1. Nạp Tập Dữ Liệu Thô và Kiểm Tra Trước Xử Lý (Pre-pipeline Verification)",
        "",
        "Kiểm tra tính toàn vẹn của ba tập dữ liệu chuẩn: `train.csv` (24,048 mẫu), `dev.csv` (2,672 mẫu) và `test.csv` (6,680 mẫu)."
    ])

    # Cell 5: Load Raw Datasets
    add_code([
        "# Cấu hình đường dẫn tập dữ liệu",
        "data_dir = 'data' if os.path.exists('data/train.csv') else os.path.join(project_dir, 'data')",
        "train_path = os.path.join(data_dir, 'train.csv')",
        "dev_path = os.path.join(data_dir, 'dev.csv')",
        "test_path = os.path.join(data_dir, 'test.csv')",
        "",
        "import pandas as pd",
        "",
        "print(f'Đọc tập Train từ: {train_path}')",
        "print(f'Đọc tập Dev từ:   {dev_path}')",
        "print(f'Đọc tập Test từ:  {test_path}')",
        "",
        "train_raw = pd.read_csv(train_path)",
        "dev_raw = pd.read_csv(dev_path)",
        "test_raw = pd.read_csv(test_path)",
        "",
        "# Kiểm tra toàn vẹn số lượng mẫu và giá trị thiếu trước khi đưa vào pipeline",
        "assert len(train_raw) == 24048, f'Số lượng mẫu train không khớp: {len(train_raw)}'",
        "assert len(dev_raw) == 2672, f'Số lượng mẫu dev không khớp: {len(dev_raw)}'",
        "assert len(test_raw) == 6680, f'Số lượng mẫu test không khớp: {len(test_raw)}'",
        "",
        "print('Kiểm tra tập dữ liệu thô trước xử lý:')",
        "print(f'- Train: {len(train_raw)} mẫu, giá trị thiếu free_text: {train_raw[\"free_text\"].isna().sum()}')",
        "print(f'- Dev:   {len(dev_raw)} mẫu, giá trị thiếu free_text: {dev_raw[\"free_text\"].isna().sum()}')",
        "print(f'- Test:  {len(test_raw)} mẫu, giá trị thiếu free_text: {test_raw[\"free_text\"].isna().sum()}')",
        "",
        "# Thống kê phân bổ nhãn trên tập huấn luyện",
        "train_dist = train_raw['label_id'].value_counts().sort_index()",
        "print('\\nPhân bổ nhãn trên tập huấn luyện:')",
        "class_labels = {0: 'CLEAN', 1: 'OFFENSIVE', 2: 'HATE'}",
        "for label_id, count in train_dist.items():",
        "    pct = (count / len(train_raw)) * 100",
        "    print(f'  Lớp {label_id} ({class_labels.get(label_id, \"UNKNOWN\")}): {count:5d} mẫu ({pct:5.2f}%)')"
    ])

    # Cell 6: Markdown Section 2
    add_markdown([
        "## 2. Chuẩn Bị Dữ Liệu Đầu Vào Theo Chuẩn Baseline",
        "",
        "Giữ nguyên văn bản thô theo đúng thiết lập baseline (loại bỏ dòng null nếu có) và trích xuất cặp `(X, y)`."
    ])

    # Cell 7: Data Preparation
    add_code([
        "text_column = 'free_text'",
        "label_column = 'label_id'",
        "",
        "# Làm sạch dữ liệu rỗng và chuẩn bị cặp biến đặc trưng / nhãn theo chuẩn baseline",
        "train_clean = train_raw.dropna(subset=[text_column, label_column]).copy()",
        "dev_clean = dev_raw.dropna(subset=[text_column, label_column]).copy()",
        "test_clean = test_raw.dropna(subset=[text_column, label_column]).copy()",
        "",
        "train_clean[label_column] = train_clean[label_column].astype(int)",
        "dev_clean[label_column] = dev_clean[label_column].astype(int)",
        "test_clean[label_column] = test_clean[label_column].astype(int)",
        "",
        "train_X = train_clean[text_column].astype(str).tolist()",
        "train_y = train_clean[label_column].to_numpy()",
        "",
        "dev_X = dev_clean[text_column].astype(str).tolist()",
        "dev_y = dev_clean[label_column].to_numpy()",
        "",
        "test_X = test_clean[text_column].astype(str).tolist()",
        "test_y = test_clean[label_column].to_numpy()",
        "",
        "print('Kích thước tập dữ liệu sẵn sàng cho mã hóa:')",
        "print(f'- Train: {len(train_X)} mẫu')",
        "print(f'- Dev:   {len(dev_X)} mẫu')",
        "print(f'- Test:  {len(test_X)} mẫu')"
    ])

    # Cell 8: Markdown Section 3
    add_markdown([
        "## 3. Định Nghĩa Lớp BuildDataset và Mã Hóa Bằng PhoBERT FastBPE (max_length = 100)"
    ])

    # Cell 9: Tokenization & Dataset
    add_code([
        "pretrained_model_name = 'vinai/phobert-base'",
        "max_seq_length = 100",
        "",
        "import torch",
        "from transformers import AutoTokenizer",
        "",
        "",
        "class BuildDataset(torch.utils.data.Dataset):",
        "    def __init__(self, encodings, labels):",
        "        self.encodings = encodings",
        "        self.labels = labels",
        "",
        "    def __getitem__(self, idx):",
        "        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}",
        "        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)",
        "        return item",
        "",
        "    def __len__(self):",
        "        return len(self.labels)",
        "",
        "",
        "print(f'Khởi tạo Tokenizer: {pretrained_model_name} (use_fast=False, max_length={max_seq_length})')",
        "tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=False)",
        "",
        "train_encodings = tokenizer(train_X, truncation=True, padding=True, max_length=max_seq_length)",
        "dev_encodings = tokenizer(dev_X, truncation=True, padding=True, max_length=max_seq_length)",
        "test_encodings = tokenizer(test_X, truncation=True, padding=True, max_length=max_seq_length)",
        "",
        "train_dataset = BuildDataset(train_encodings, train_y)",
        "dev_dataset = BuildDataset(dev_encodings, dev_y)",
        "test_dataset = BuildDataset(test_encodings, test_y)",
        "",
        "print(f'Kích thước tập train sau mã hóa: {len(train_dataset)}')",
        "print(f'Kích thước tập dev sau mã hóa:   {len(dev_dataset)}')",
        "print(f'Kích thước tập test sau mã hóa:  {len(test_dataset)}')"
    ])

    # Cell 10: Markdown Section 4
    add_markdown([
        "## 4. Tính Toán Vector Trọng Số Smoothed Class Weights và Định Nghĩa Weighted Trainer",
        "",
        "Áp dụng công thức làm mịn căn bậc hai:",
        "$$w_c = \\sqrt{\\frac{N}{N_c}} \\bigg/ \\left( \\frac{1}{K} \\sum_{j=0}^{K-1} \\sqrt{\\frac{N}{N_j}} \\right)$$"
    ])

    # Cell 11: Compute Class Weights & Weighted Trainer
    add_code([
        "import numpy as np",
        "import torch",
        "import torch.nn as nn",
        "from transformers import Trainer",
        "",
        "",
        "def compute_smoothed_class_weights(labels):",
        "    \"\"\"Tính toán vector trọng số lớp làm mịn căn bậc hai (Smoothed Inverse Frequency).\"\"\"",
        "    labels = np.asarray(labels, dtype=int)",
        "    total_samples = len(labels)",
        "    classes, counts = np.unique(labels, return_counts=True)",
        "    counts_float = counts.astype(np.float64)",
        "",
        "    raw_smoothed = np.sqrt(total_samples / counts_float)",
        "    w_smoothed = raw_smoothed / np.mean(raw_smoothed)",
        "    return torch.tensor(w_smoothed, dtype=torch.float32), dict(zip(classes, counts))",
        "",
        "",
        "active_class_weights, class_counts = compute_smoothed_class_weights(train_y)",
        "class_names_str = ['CLEAN', 'OFFENSIVE', 'HATE']",
        "",
        "print('=' * 65)",
        "print('VECTOR TRỌNG SỐ LỚP LÀM MỊN CĂN BẬC HAI (CHECKPOINT 01)')",
        "print('=' * 65)",
        "for c_id, count in class_counts.items():",
        "    print(f'- Lớp {c_id} ({class_names_str[c_id]:9s}): {count:5d} mẫu -> Trọng số alpha: {active_class_weights[c_id]:.4f}')",
        "print('=' * 65)",
        "",
        "",
        "class WeightedLossTrainer(Trainer):",
        "    \"\"\"Lớp Trainer áp dụng Weighted Cross-Entropy Loss.\"\"\"",
        "",
        "    def __init__(self, *args, class_weights=None, **kwargs):",
        "        super().__init__(*args, **kwargs)",
        "        self.class_weights = class_weights",
        "",
        "    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):",
        "        labels = inputs.get('labels')",
        "        outputs = model(**inputs)",
        "        logits = outputs.get('logits')",
        "",
        "        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None",
        "        loss_fn = nn.CrossEntropyLoss(weight=weights)",
        "        loss = loss_fn(logits, labels)",
        "        return (loss, outputs) if return_outputs else loss",
        "",
        "print('Đã khởi tạo WeightedLossTrainer với trọng số làm mịn.')"
    ])

    # Cell 12: Markdown Section 5
    add_markdown([
        "## 5. Cấu Hình Siêu Tham Số Huấn Luyện (Baseline Alignment)"
    ])

    # Cell 13: Model Setup & Training Arguments
    add_code([
        "num_labels = 3",
        "num_epochs = 2",
        "train_batch_size = 16",
        "eval_batch_size = 16",
        "learning_rate = 2e-5",
        "warmup_steps = 500",
        "weight_decay = 0.01",
        "",
        "import numpy as np",
        "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score",
        "from transformers import AutoModelForSequenceClassification, TrainingArguments",
        "",
        "",
        "def compute_metrics_eval(eval_pred):",
        "    predictions, labels = eval_pred",
        "    preds = np.argmax(predictions, axis=-1)",
        "    acc = accuracy_score(labels, preds)",
        "    macro_p = precision_score(labels, preds, average='macro', zero_division=0)",
        "    macro_r = recall_score(labels, preds, average='macro', zero_division=0)",
        "    macro_f1 = f1_score(labels, preds, average='macro')",
        "    micro_f1 = f1_score(labels, preds, average='micro')",
        "    per_class_f1 = f1_score(labels, preds, average=None)",
        "    return {",
        "        'accuracy': acc,",
        "        'macro_precision': macro_p,",
        "        'macro_recall': macro_r,",
        "        'macro_f1': macro_f1,",
        "        'micro_f1': micro_f1,",
        "        'f1_clean': per_class_f1[0] if len(per_class_f1) > 0 else 0.0,",
        "        'f1_offensive': per_class_f1[1] if len(per_class_f1) > 1 else 0.0,",
        "        'f1_hate': per_class_f1[2] if len(per_class_f1) > 2 else 0.0,",
        "    }",
        "",
        "",
        "model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)",
        "",
        "training_args = TrainingArguments(",
        "    output_dir=checkpoints_dir,",
        "    num_train_epochs=num_epochs,",
        "    per_device_train_batch_size=train_batch_size,",
        "    per_device_eval_batch_size=eval_batch_size,",
        "    warmup_steps=warmup_steps,",
        "    weight_decay=weight_decay,",
        "    learning_rate=learning_rate,",
        "    do_eval=True,",
        "    eval_strategy='epoch',",
        "    save_strategy='epoch',",
        "    load_best_model_at_end=False,",
        "    fp16=torch.cuda.is_available(),",
        "    logging_steps=50,",
        "    seed=random_seed,",
        ")",
        "",
        "trainer = WeightedLossTrainer(",
        "    model=model,",
        "    args=training_args,",
        "    train_dataset=train_dataset,",
        "    eval_dataset=dev_dataset,",
        "    compute_metrics=compute_metrics_eval,",
        "    class_weights=active_class_weights,",
        ")",
        "",
        "print('Khởi tạo Trainer thành công cho Checkpoint 01.')"
    ])

    # Cell 14: Markdown Section 6
    add_markdown([
        "## 6. Huấn Luyện Mô Hình và Lưu Checkpoint"
    ])

    # Cell 15: Run Training
    add_code([
        "print('Bắt đầu huấn luyện Checkpoint 01 (Baseline + Smoothed CW)...')",
        "train_result = trainer.train()",
        "",
        "save_model_dir = os.path.join(checkpoints_dir, 'final_model')",
        "trainer.save_model(save_model_dir)",
        "tokenizer.save_pretrained(save_model_dir)",
        "print(f'Đã lưu mô hình tại: {save_model_dir}')"
    ])

    # Cell 16: Markdown Section 7
    add_markdown([
        "## 7. Đánh Giá Trên Tập Test Độc Lập"
    ])

    # Cell 17: Test Evaluation
    add_code([
        "class_names = ['clean', 'offensive', 'hate']",
        "",
        "import numpy as np",
        "import pandas as pd",
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score",
        "",
        "",
        "def evaluate_predictions(y_true, y_pred, class_names=None):",
        "    if class_names is None:",
        "        class_names = ['clean', 'offensive', 'hate']",
        "    acc = accuracy_score(y_true, y_pred)",
        "    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)",
        "    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)",
        "    macro_f1 = f1_score(y_true, y_pred, average='macro')",
        "    micro_f1 = f1_score(y_true, y_pred, average='micro')",
        "    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)",
        "    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)",
        "    per_class_f1 = f1_score(y_true, y_pred, average=None)",
        "    cm = confusion_matrix(y_true, y_pred)",
        "    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)",
        "    return {",
        "        'accuracy': acc,",
        "        'macro_precision': macro_precision,",
        "        'macro_recall': macro_recall,",
        "        'macro_f1': macro_f1,",
        "        'micro_f1': micro_f1,",
        "        'per_class_precision': {name: per_class_precision[i] for i, name in enumerate(class_names)},",
        "        'per_class_recall': {name: per_class_recall[i] for i, name in enumerate(class_names)},",
        "        'per_class_f1': {name: per_class_f1[i] for i, name in enumerate(class_names)},",
        "        'confusion_matrix': cm,",
        "        'classification_report': report,",
        "    }",
        "",
        "",
        "print('Thực hiện dự đoán với mô hình Checkpoint 01 trên tập kiểm tra...')",
        "predictions_output = trainer.predict(test_dataset)",
        "y_pred = np.argmax(predictions_output.predictions, axis=-1)",
        "y_true = test_y",
        "",
        "results = evaluate_predictions(y_true, y_pred, class_names=class_names)",
        "",
        "print('=' * 60)",
        "print('KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (CHECKPOINT 01)')",
        "print('=' * 60)",
        "print(f'Accuracy (Độ chính xác tổng thể):  {results[\"accuracy\"]:.4f}')",
        "print(f'Macro-Precision (Độ chuẩn xác):   {results[\"macro_precision\"]:.4f}')",
        "print(f'Macro-Recall (Độ nhạy):           {results[\"macro_recall\"]:.4f}')",
        "print(f'Macro-F1 Score (Chỉ số chính):    {results[\"macro_f1\"]:.4f}')",
        "print(f'Micro-F1 Score:                   {results[\"micro_f1\"]:.4f}')",
        "print('-' * 60)",
        "print('Chi tiết theo từng lớp (Per-class):')",
        "for cname in class_names:",
        "    print(f'- {cname:9s}: Precision = {results[\"per_class_precision\"][cname]:.4f}, Recall = {results[\"per_class_recall\"][cname]:.4f}, F1 = {results[\"per_class_f1\"][cname]:.4f}')",
        "print('=' * 60)",
        "print('Chi tiết Classification Report:')",
        "print(results['classification_report'])"
    ])

    # Cell 18: Markdown Section 8
    add_markdown([
        "## 8. Phân Tích Lỗi Định Tính và Trực Quan Hóa Ma Trận Nhầm Lẫn"
    ])

    # Cell 19: Plot Confusion Matrix
    add_code([
        "cm_output_path = os.path.join(outputs_dir, 'confusion_matrix_checkpoint01.png')",
        "",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
        "",
        "def plot_confusion_matrix(cm, class_names=None, output_path=None):",
        "    if class_names is None:",
        "        class_names = ['clean', 'offensive', 'hate']",
        "    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)",
        "    fig, ax = plt.subplots(figsize=(6, 5))",
        "    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d', cbar=True, ax=ax, annot_kws={'size': 12})",
        "    ax.set_title('Confusion Matrix - Checkpoint 01 (Smoothed CW)')",
        "    ax.set_xlabel('Predicted Label')",
        "    ax.set_ylabel('True Label')",
        "    plt.tight_layout()",
        "    if output_path:",
        "        plt.savefig(output_path, dpi=300)",
        "    return fig",
        "",
        "",
        "fig_cm = plot_confusion_matrix(results['confusion_matrix'], class_names=class_names, output_path=cm_output_path)",
        "plt.show()",
        "print(f'Đã lưu ma trận nhầm lẫn tại: {cm_output_path}')"
    ])

    # Cell 20: Markdown Section 9 (Mandatory Artifacts Export)
    add_markdown([
        "## 9. Xuất Các Tệp Kết Quả Chuẩn Hóa (Mandatory Artifacts Export)"
    ])

    # Cell 23: Mandatory Artifacts Export
    add_code([
        "predictions_csv_path = os.path.join(outputs_dir, 'predictions.csv')",
        "scores_csv_path = os.path.join(outputs_dir, 'evaluation_scores.csv')",
        "report_csv_path = os.path.join(outputs_dir, 'classification_report.csv')",
        "cm_csv_path = os.path.join(outputs_dir, 'confusion_matrix.csv')",
        "",
        "test_ids = test_raw['id'] if 'id' in test_raw.columns else list(range(len(test_y)))",
        "test_texts = test_raw['free_text'].tolist() if len(test_raw) == len(test_y) else test_X",
        "",
        "df_predictions = pd.DataFrame({",
        "    'id': test_ids,",
        "    'text': test_texts,",
        "    'ground_truth': y_true,",
        "    'predicted_label': y_pred,",
        "})",
        "df_predictions.to_csv(predictions_csv_path, index=False, encoding='utf-8')",
        "print(f'1. Đã xuất tệp dự đoán: {predictions_csv_path}')",
        "",
        "df_cm = pd.DataFrame(results['confusion_matrix'], index=class_names, columns=class_names)",
        "df_cm.index.name = 'true_label'",
        "df_cm.to_csv(cm_csv_path)",
        "print(f'2. Đã xuất tệp dữ liệu ma trận nhầm lẫn: {cm_csv_path}')",
        "",
        "acc_val = round(float(results.get('accuracy', accuracy_score(y_true, y_pred))), 4)",
        "macro_p_val = round(float(results.get('macro_precision', precision_score(y_true, y_pred, average='macro', zero_division=0))), 4)",
        "macro_r_val = round(float(results.get('macro_recall', recall_score(y_true, y_pred, average='macro', zero_division=0))), 4)",
        "per_class_p = results.get('per_class_precision', {name: round(precision_score(y_true, y_pred, average=None, zero_division=0)[i], 4) for i, name in enumerate(class_names)})",
        "per_class_r = results.get('per_class_recall', {name: round(recall_score(y_true, y_pred, average=None, zero_division=0)[i], 4) for i, name in enumerate(class_names)})",
        "",
        "scores_summary = {",
        "    'metric': [",
        "        'accuracy',",
        "        'macro_precision',",
        "        'macro_recall',",
        "        'macro_f1',",
        "        'micro_f1',",
        "        'precision_clean',",
        "        'recall_clean',",
        "        'f1_clean',",
        "        'precision_offensive',",
        "        'recall_offensive',",
        "        'f1_offensive',",
        "        'precision_hate',",
        "        'recall_hate',",
        "        'f1_hate',",
        "    ],",
        "    'value': [",
        "        acc_val,",
        "        macro_p_val,",
        "        macro_r_val,",
        "        round(results['macro_f1'], 4),",
        "        round(results['micro_f1'], 4),",
        "        round(per_class_p['clean'], 4),",
        "        round(per_class_r['clean'], 4),",
        "        round(results['per_class_f1']['clean'], 4),",
        "        round(per_class_p['offensive'], 4),",
        "        round(per_class_r['offensive'], 4),",
        "        round(results['per_class_f1']['offensive'], 4),",
        "        round(per_class_p['hate'], 4),",
        "        round(per_class_r['hate'], 4),",
        "        round(results['per_class_f1']['hate'], 4),",
        "    ],",
        "}",
        "df_scores = pd.DataFrame(scores_summary)",
        "df_scores.to_csv(scores_csv_path, index=False)",
        "print(f'3. Đã xuất tệp điểm số tổng hợp: {scores_csv_path}')",
        "",
        "report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)",
        "accuracy_val = report_dict.pop('accuracy')",
        "",
        "rows = []",
        "for label in class_names:",
        "    row = report_dict[label]",
        "    rows.append({",
        "        'class': label,",
        "        'precision': round(row['precision'], 4),",
        "        'recall': round(row['recall'], 4),",
        "        'f1_score': round(row['f1-score'], 4),",
        "        'support': int(row['support']),",
        "    })",
        "",
        "rows.append({",
        "    'class': 'accuracy',",
        "    'precision': None,",
        "    'recall': None,",
        "    'f1_score': round(accuracy_val, 4),",
        "    'support': len(y_true),",
        "})",
        "",
        "for avg_type in ['macro avg', 'weighted avg']:",
        "    row = report_dict[avg_type]",
        "    rows.append({",
        "        'class': avg_type,",
        "        'precision': round(row['precision'], 4),",
        "        'recall': round(row['recall'], 4),",
        "        'f1_score': round(row['f1-score'], 4),",
        "        'support': int(row['support']),",
        "    })",
        "",
        "df_report = pd.DataFrame(rows)",
        "df_report.to_csv(report_csv_path, index=False)",
        "print(f'4. Đã xuất tệp báo cáo phân loại chi tiết: {report_csv_path}')"
    ])

    notebook_content = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "collapsed_sections": [],
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    output_path = os.path.join("Notebook", "checkpoint01_baseline_class_weights.ipynb")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, ensure_ascii=False, indent=1)

    print(f"Khởi tạo notebook thành công tại: {output_path}")
    print(f"Tổng số cells: {len(cells)}")


if __name__ == "__main__":
    create_checkpoint01_notebook()
