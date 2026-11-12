import json
import os


def create_checkpoint02_notebook():
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
        "# Checkpoint 02: Thiết Lập Baseline Kết Hợp Cân Bằng Lớp Động (Focal Loss, gamma = 1.5)",
        "",
        "Nghiên cứu: Nâng cao hiệu năng mô hình phát hiện phát ngôn thù ghét trên mạng xã hội tiếng Việt (ViHSD).",
        "",
        "Mục tiêu khoa học:",
        "1. Khảo sát giải pháp cân bằng lớp động thông qua hàm mất mát Focal Loss trên cấu hình mô hình Baseline gốc (`vinai/phobert-base`, 2 epochs, max_length = 100, batch size = 16, lr = 2e-5).",
        "2. Cô lập hoàn toàn tác động của tham số tập trung $\\gamma = 1.5$ (không áp dụng trọng số tĩnh $\\alpha$) nhằm đánh giá năng lực tự động triệt tiêu gradient từ các mẫu đa số dễ phân loại (CLEAN):",
        "   $$\\mathcal{L}_{FL}(p_t) = -(1 - p_t)^\\gamma \\log(p_t), \\quad \\gamma = 1.5$$",
        "3. Tối ưu hóa tính ổn định số học trong PyTorch bằng cách tính $p_t = \\exp(-\\mathcal{L}_{CE})$ thông qua `F.cross_entropy` với cơ chế log-sum-exp.",
        "4. Đánh giá đa chiều và xuất đầy đủ 4 tệp Mandatory Artifacts vào thư mục `outputs/checkpoint02/`.",
        "",
        "Ghi chú: Toàn bộ mã nguồn định nghĩa Dataset, hàm mất mát Focal Loss và các thường trình đánh giá được tích hợp trực tiếp 100% trong notebook nhằm đảm bảo tính độc lập khi thực thi trên Google Colab."
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
        "checkpoints_dir = os.path.join(project_dir, 'checkpoints', 'checkpoint02')",
        "outputs_dir = os.path.join(project_dir, 'outputs', 'checkpoint02')",
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
        "## 1. Nạp Tập Dữ Liệu Thô và Kiểm Tra Trước Xử Lý (Pre-pipeline Verification)"
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
        "print(f'- Train: {len(train_raw)} mẫu, giá trị thiếu: {train_raw[\"free_text\"].isna().sum()}')",
        "print(f'- Dev:   {len(dev_raw)} mẫu, giá trị thiếu: {dev_raw[\"free_text\"].isna().sum()}')",
        "print(f'- Test:  {len(test_raw)} mẫu, giá trị thiếu: {test_raw[\"free_text\"].isna().sum()}')"
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
        "## 4. Định Nghĩa Hàm Mất Mát Focal Loss (gamma = 1.5) và Lớp FocalLossTrainer",
        "",
        "Công thức toán học của Focal Loss đa lớp:",
        "$$\\mathcal{L}_{FL}(p_t) = -(1 - p_t)^\\gamma \\log(p_t), \\quad \\gamma = 1.5$$",
        "Với $p_t = \\exp(-\\mathcal{L}_{CE})$ được tính toán ổn định qua log-sum-exp của PyTorch."
    ])

    # Cell 11: Focal Loss Definition & Trainer
    add_code([
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "from transformers import Trainer",
        "",
        "",
        "class FocalLoss(nn.Module):",
        "    \"\"\"Hàm mất mát Focal Loss đa lớp triệt tiêu gradient từ mẫu dễ phân loại.\"\"\"",
        "",
        "    def __init__(self, gamma=1.5, alpha=None, reduction='mean'):",
        "        super().__init__()",
        "        self.gamma = gamma",
        "        self.alpha = alpha",
        "        self.reduction = reduction",
        "",
        "    def forward(self, logits, targets):",
        "        ce_loss = F.cross_entropy(logits, targets, reduction='none')",
        "        pt = torch.exp(-ce_loss)",
        "        modulating_factor = (1.0 - pt) ** self.gamma",
        "        focal_loss = modulating_factor * ce_loss",
        "",
        "        if self.alpha is not None:",
        "            if isinstance(self.alpha, torch.Tensor):",
        "                alpha_t = self.alpha.to(logits.device)[targets]",
        "                focal_loss = alpha_t * focal_loss",
        "            else:",
        "                focal_loss = self.alpha * focal_loss",
        "",
        "        return focal_loss.mean() if self.reduction == 'mean' else focal_loss",
        "",
        "",
        "class FocalLossTrainer(Trainer):",
        "    \"\"\"Lớp Trainer áp dụng Focal Loss trong quá trình huấn luyện.\"\"\"",
        "",
        "    def __init__(self, *args, focal_loss_fn=None, **kwargs):",
        "        super().__init__(*args, **kwargs)",
        "        self.focal_loss_fn = focal_loss_fn if focal_loss_fn is not None else FocalLoss(gamma=1.5)",
        "",
        "    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):",
        "        labels = inputs.get('labels')",
        "        outputs = model(**inputs)",
        "        logits = outputs.get('logits')",
        "        loss = self.focal_loss_fn(logits, labels)",
        "        return (loss, outputs) if return_outputs else loss",
        "",
        "focal_loss_fn = FocalLoss(gamma=1.5)",
        "print('Khởi tạo FocalLoss thành công với gamma = 1.5 (alpha = None để cô lập tác động).')"
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
        "trainer = FocalLossTrainer(",
        "    model=model,",
        "    args=training_args,",
        "    train_dataset=train_dataset,",
        "    eval_dataset=dev_dataset,",
        "    compute_metrics=compute_metrics_eval,",
        "    focal_loss_fn=focal_loss_fn,",
        ")",
        "",
        "print('Khởi tạo FocalLossTrainer thành công cho Checkpoint 02.')"
    ])

    # Cell 14: Markdown Section 6
    add_markdown([
        "## 6. Huấn Luyện Mô Hình và Lưu Checkpoint"
    ])

    # Cell 15: Run Training
    add_code([
        "print('Bắt đầu huấn luyện Checkpoint 02 (Baseline + Focal Loss gamma = 1.5)...')",
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
        "print('Thực hiện dự đoán với mô hình Checkpoint 02 trên tập kiểm tra...')",
        "predictions_output = trainer.predict(test_dataset)",
        "y_pred = np.argmax(predictions_output.predictions, axis=-1)",
        "y_true = test_y",
        "",
        "results = evaluate_predictions(y_true, y_pred, class_names=class_names)",
        "",
        "print('=' * 60)",
        "print('KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (CHECKPOINT 02)')",
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
        "cm_output_path = os.path.join(outputs_dir, 'confusion_matrix_checkpoint02.png')",
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
        "    ax.set_title('Confusion Matrix - Checkpoint 02 (Focal Loss gamma = 1.5)')",
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

    output_path = os.path.join("Notebook", "checkpoint02_baseline_focal_loss.ipynb")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, ensure_ascii=False, indent=1)

    print(f"Khởi tạo notebook thành công tại: {output_path}")
    print(f"Tổng số cells: {len(cells)}")


if __name__ == "__main__":
    create_checkpoint02_notebook()
