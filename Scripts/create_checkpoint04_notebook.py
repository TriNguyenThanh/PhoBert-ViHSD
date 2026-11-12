import json
import os


def create_checkpoint04_notebook():
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
        "# Checkpoint 04: Tối Ưu Hóa Chiến Lược Huấn Luyện (Advanced Training Strategies)",
        "",
        "Nghiên cứu: Nâng cao hiệu năng mô hình phát hiện phát ngôn thù ghét trên mạng xã hội tiếng Việt (ViHSD).",
        "",
        "Mục tiêu khoa học:",
        "1. Khắc phục các hạn chế kỹ thuật trong cấu hình huấn luyện của Baseline gốc (2 epochs ngắn ngủi, độ dài chuỗi max_length = 100 gây cắt cụt thông tin).",
        "2. Mở rộng chiều dài chuỗi tối đa lên `max_length = 256` nhằm bao phủ hoàn chỉnh ngữ cảnh bình luận mạng xã hội, tận dụng tối đa giới hạn vị trí nhúng của PhoBERT.",
        "3. Tăng thời lượng huấn luyện lên 8 epochs kết hợp cơ chế dừng sớm (Early Stopping với patience = 2) theo dõi trực tiếp chỉ số `eval_macro_f1`.",
        "4. Áp dụng phân tầng tốc độ học (Differential Learning Rates): giữ tốc độ học nhỏ ($2 \\times 10^{-5}$) cho backbone RoBERTa và tốc độ học lớn hơn ($1 \\times 10^{-4}$) cho Classification Head.",
        "5. Tối ưu hóa cập nhật trọng số với lịch trình Cosine Annealing with Warmup và tích lũy gradient (gradient_accumulation_steps = 2, effective batch size = 32).",
        "6. Tự động phục hồi trọng số tối ưu nhất (`load_best_model_at_end = True`) tại điểm cực đại của Macro-F1 trên tập dev.",
        "7. Xuất đầy đủ 4 tệp Mandatory Artifacts cùng biểu đồ Learning Curves vào `outputs/checkpoint04/`.",
        "",
        "Ghi chú: Toàn bộ mã nguồn định nghĩa tối ưu hóa phân tầng, custom trainer, đồ thị học và đánh giá được tích hợp 100% trong notebook để chạy độc lập trên Google Colab."
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
        "checkpoints_dir = os.path.join(project_dir, 'checkpoints', 'checkpoint04')",
        "outputs_dir = os.path.join(project_dir, 'outputs', 'checkpoint04')",
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
        "print(f'Thiết lập random seed thành công: {random_seed}')"
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
        "train_raw = pd.read_csv(train_path)",
        "dev_raw = pd.read_csv(dev_path)",
        "test_raw = pd.read_csv(test_path)",
        "",
        "assert len(train_raw) == 24048, f'Số lượng mẫu train không khớp: {len(train_raw)}'",
        "assert len(dev_raw) == 2672, f'Số lượng mẫu dev không khớp: {len(dev_raw)}'",
        "assert len(test_raw) == 6680, f'Số lượng mẫu test không khớp: {len(test_raw)}'",
        "",
        "print('Kiểm tra tập dữ liệu thô trước xử lý:')",
        "print(f'- Train: {len(train_raw)} mẫu, null: {train_raw[\"free_text\"].isna().sum()}')",
        "print(f'- Dev:   {len(dev_raw)} mẫu, null: {dev_raw[\"free_text\"].isna().sum()}')",
        "print(f'- Test:  {len(test_raw)} mẫu, null: {test_raw[\"free_text\"].isna().sum()}')"
    ])

    # Cell 6: Markdown Section 2
    add_markdown([
        "## 2. Chuẩn Bị Dữ Liệu Đầu Vào và Làm Sạch Null"
    ])

    # Cell 7: Data Preparation
    add_code([
        "text_column = 'free_text'",
        "label_column = 'label_id'",
        "",
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
        "## 3. Định Nghĩa Lớp BuildDataset và Mã Hóa Với Độ Dài Mở Rộng (max_length = 256)"
    ])

    # Cell 9: Tokenization & Dataset
    add_code([
        "pretrained_model_name = 'vinai/phobert-base'",
        "max_seq_length = 256",
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
        "## 4. Cấu Hình Phân Tầng Tốc Độ Học (Differential Learning Rates) và Trainer Tùy Biến"
    ])

    # Cell 11: Differential Optimizer & Custom Trainer
    add_code([
        "from torch.optim import AdamW",
        "from transformers import Trainer, get_cosine_schedule_with_warmup",
        "",
        "",
        "class DifferentialTrainer(Trainer):",
        "    \"\"\"Lớp Trainer cấu hình phân tầng tốc độ học và lịch trình Cosine Annealing with Warmup.\"\"\"",
        "",
        "    def create_optimizer_and_scheduler(self, num_training_steps: int):",
        "        decay_parameters = [p for n, p in self.model.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])]",
        "        no_decay_parameters = [p for n, p in self.model.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight'])]",
        "",
        "        roberta_decay = [p for n, p in self.model.named_parameters() if 'roberta' in n and not any(nd in n for nd in ['bias', 'LayerNorm.weight'])]",
        "        roberta_no_decay = [p for n, p in self.model.named_parameters() if 'roberta' in n and any(nd in n for nd in ['bias', 'LayerNorm.weight'])]",
        "        classifier_decay = [p for n, p in self.model.named_parameters() if 'roberta' not in n and not any(nd in n for nd in ['bias', 'LayerNorm.weight'])]",
        "        classifier_no_decay = [p for n, p in self.model.named_parameters() if 'roberta' not in n and any(nd in n for nd in ['bias', 'LayerNorm.weight'])]",
        "",
        "        optimizer_grouped_parameters = [",
        "            {'params': roberta_decay, 'lr': 2e-5, 'weight_decay': 0.01},",
        "            {'params': roberta_no_decay, 'lr': 2e-5, 'weight_decay': 0.0},",
        "            {'params': classifier_decay, 'lr': 1e-4, 'weight_decay': 0.01},",
        "            {'params': classifier_no_decay, 'lr': 1e-4, 'weight_decay': 0.0},",
        "        ]",
        "",
        "        self.optimizer = AdamW(optimizer_grouped_parameters)",
        "        warmup_steps = int(num_training_steps * 0.10)",
        "        self.lr_scheduler = get_cosine_schedule_with_warmup(",
        "            self.optimizer,",
        "            num_warmup_steps=warmup_steps,",
        "            num_training_steps=num_training_steps,",
        "        )",
        "        print(f'Đã khởi tạo Optimizer phân tầng (lr_roberta=2e-5, lr_head=1e-4, warmup_steps={warmup_steps})')",
        "        return self.optimizer, self.lr_scheduler",
        "",
        "print('Định nghĩa DifferentialTrainer thành công.')"
    ])

    # Cell 12: Markdown Section 5
    add_markdown([
        "## 5. Cấu Hình Siêu Tham Số Huấn Luyện Nâng Cao và Dừng Sớm (Early Stopping)"
    ])

    # Cell 13: Model Setup & Training Arguments
    add_code([
        "num_labels = 3",
        "num_epochs = 8",
        "train_batch_size = 16",
        "eval_batch_size = 16",
        "gradient_accumulation_steps = 2",
        "weight_decay = 0.01",
        "",
        "import numpy as np",
        "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score",
        "from transformers import AutoModelForSequenceClassification, EarlyStoppingCallback, TrainingArguments",
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
        "    gradient_accumulation_steps=gradient_accumulation_steps,",
        "    weight_decay=weight_decay,",
        "    do_eval=True,",
        "    eval_strategy='epoch',",
        "    save_strategy='epoch',",
        "    save_total_limit=2,",
        "    load_best_model_at_end=True,",
        "    metric_for_best_model='macro_f1',",
        "    greater_is_better=True,",
        "    fp16=torch.cuda.is_available(),",
        "    logging_steps=50,",
        "    seed=random_seed,",
        ")",
        "",
        "trainer = DifferentialTrainer(",
        "    model=model,",
        "    args=training_args,",
        "    train_dataset=train_dataset,",
        "    eval_dataset=dev_dataset,",
        "    compute_metrics=compute_metrics_eval,",
        "    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],",
        ")",
        "",
        "print('Khởi tạo DifferentialTrainer thành công cho Checkpoint 04.')"
    ])

    # Cell 14: Markdown Section 6
    add_markdown([
        "## 6. Huấn Luyện Mô Hình và Lưu Checkpoint Tối Ưu"
    ])

    # Cell 15: Run Training
    add_code([
        "print('Bắt đầu huấn luyện Checkpoint 04 (Advanced Training Strategies)...')",
        "train_result = trainer.train()",
        "",
        "save_model_dir = os.path.join(checkpoints_dir, 'final_model')",
        "trainer.save_model(save_model_dir)",
        "tokenizer.save_pretrained(save_model_dir)",
        "print(f'Đã lưu mô hình tốt nhất tại: {save_model_dir}')"
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
        "print('Thực hiện dự đoán với mô hình Checkpoint 04 trên tập kiểm tra...')",
        "predictions_output = trainer.predict(test_dataset)",
        "y_pred = np.argmax(predictions_output.predictions, axis=-1)",
        "y_true = test_y",
        "",
        "results = evaluate_predictions(y_true, y_pred, class_names=class_names)",
        "",
        "print('=' * 60)",
        "print('KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (CHECKPOINT 04)')",
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
        "## 8. Trực Quan Hóa Đồ Thị Học (Learning Curves) và Ma Trận Nhầm Lẫn"
    ])

    # Cell 19: Plot Curves & Confusion Matrix
    add_code([
        "curves_output_path = os.path.join(outputs_dir, 'learning_curves_checkpoint04.png')",
        "cm_output_path = os.path.join(outputs_dir, 'confusion_matrix_checkpoint04.png')",
        "",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
        "# Vẽ đồ thị tiến trình học",
        "log_history = trainer.state.log_history",
        "train_loss = [x['loss'] for x in log_history if 'loss' in x]",
        "eval_f1 = [x['eval_macro_f1'] for x in log_history if 'eval_macro_f1' in x]",
        "",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))",
        "if train_loss:",
        "    ax1.plot(train_loss, label='Training Loss', color='blue')",
        "    ax1.set_title('Training Loss Curve')",
        "    ax1.set_xlabel('Logging Steps')",
        "    ax1.set_ylabel('Loss')",
        "    ax1.legend()",
        "if eval_f1:",
        "    ax2.plot(range(1, len(eval_f1) + 1), eval_f1, label='Validation Macro-F1', color='green', marker='o')",
        "    ax2.set_title('Validation Macro-F1 per Epoch')",
        "    ax2.set_xlabel('Epoch')",
        "    ax2.set_ylabel('Macro-F1')",
        "    ax2.legend()",
        "plt.tight_layout()",
        "plt.savefig(curves_output_path, dpi=300)",
        "plt.show()",
        "print(f'Đã lưu đồ thị học tại: {curves_output_path}')",
        "",
        "# Vẽ ma trận nhầm lẫn",
        "df_cm = pd.DataFrame(results['confusion_matrix'], index=class_names, columns=class_names)",
        "fig_cm, ax_cm = plt.subplots(figsize=(6, 5))",
        "sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d', cbar=True, ax=ax_cm, annot_kws={'size': 12})",
        "ax_cm.set_title('Confusion Matrix - Checkpoint 04 (Advanced Training)')",
        "ax_cm.set_xlabel('Predicted Label')",
        "ax_cm.set_ylabel('True Label')",
        "plt.tight_layout()",
        "plt.savefig(cm_output_path, dpi=300)",
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

    output_path = os.path.join("Notebook", "checkpoint04_training_strategies.ipynb")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, ensure_ascii=False, indent=1)

    print(f"Khởi tạo notebook thành công tại: {output_path}")
    print(f"Tổng số cells: {len(cells)}")


if __name__ == "__main__":
    create_checkpoint04_notebook()
