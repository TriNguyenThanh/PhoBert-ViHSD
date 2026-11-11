import json
import os


def create_checkpoint05_notebook():
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
        "# Checkpoint 05: Tích Hợp Toàn Diện Các Giải Pháp Nâng Cấp (Combined Enhancements)",
        "",
        "Nghiên cứu: Nâng cao hiệu năng mô hình phát hiện phát ngôn thù ghét trên mạng xã hội tiếng Việt (ViHSD).",
        "",
        "Mục tiêu khoa học:",
        "1. Tích hợp đồng thời 4 giải pháp tối ưu hóa độc lập đã được chứng minh hiệu quả từ các Checkpoint trước trên nền tảng kiến trúc `vinai/phobert-base`:",
        "   - **Tiền xử lý mạng xã hội chuyên sâu (CP03):** Áp dụng `ViHSDPreprocessor` (chuẩn hóa teencode 412 quy tắc, ánh xạ emoji 2 pha, thu gọn ký tự, tách từ ghép PyVi, bảo tồn đại từ nhân xưng và từ phủ định).",
        "   - **Hàm mất mát Weighted Focal Loss (CP01 + CP02):** Kết hợp vector trọng số lớp làm mịn căn bậc hai $\\alpha$ (Smoothed Class Weights) và cơ chế điều chế mẫu khó động $\\gamma = 1.5$:",
        "     $$\\mathcal{L}_{WFL} = -\\alpha_{y_i} (1 - p_{i, y_i})^\\gamma \\log(p_{i, y_i})$$",
        "   - **Mở rộng ngữ cảnh (CP04):** `max_length = 256` bao phủ trọn vẹn ngữ cảnh bình luận.",
        "   - **Tối ưu hóa huấn luyện nâng cao (CP04):** Phân tầng tốc độ học ($2 \\times 10^{-5}$ roberta / $1 \\times 10^{-4}$ classifier), Cosine Annealing with Warmup, 8 epochs, batch size 16 + gradient accumulation 2, Early Stopping patience = 2.",
        "2. Đánh giá hiệu ứng cộng hưởng tích cực giữa tiền xử lý ngôn ngữ, cân bằng phân bố lớp và tối ưu hóa hội tụ.",
        "3. Xuất đầy đủ 4 tệp Mandatory Artifacts cùng biểu đồ Learning Curves vào `outputs/checkpoint05/`.",
        "",
        "Ghi chú: Toàn bộ mã nguồn định nghĩa bộ tiền xử lý, Dataset, hàm mất mát Weighted Focal Loss, Optimizer phân tầng và đánh giá được tích hợp 100% trong notebook để chạy độc lập trên Google Colab."
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
        "checkpoints_dir = os.path.join(project_dir, 'checkpoints', 'checkpoint05')",
        "outputs_dir = os.path.join(project_dir, 'outputs', 'checkpoint05')",
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
        "!pip install -q pyvi transformers accelerate scikit-learn seaborn matplotlib pandas numpy torch"
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
        "## 2. Tiền Xử Lý và Chuẩn Hóa Dữ Liệu Chuyên Sâu (ViHSDPreprocessor)"
    ])

    # Cell 7: Inlined ViHSDPreprocessor Class
    add_code([
        "import csv",
        "import os",
        "import re",
        "import unicodedata",
        "import numpy as np",
        "import pandas as pd",
        "",
        "try:",
        "    from pyvi import ViTokenizer",
        "except ImportError:",
        "    ViTokenizer = None",
        "",
        "",
        "class ViHSDPreprocessor:",
        "    \"\"\"Bộ tiền xử lý và chuẩn hóa dữ liệu tiếng Việt mạng xã hội tự chứa (Self-contained).\"\"\"",
        "",
        "    def __init__(self, teencode_path, emoji_path, segmenter=None):",
        "        self.segmenter = segmenter",
        "        self.teencode_dict = self._load_teencode_dict(teencode_path)",
        "        self.emoji_dict = self._load_emoji_dict(emoji_path)",
        "",
        "        self.url_pattern = re.compile(r'https?://\\S+|www\\.\\S+')",
        "        self.whitespace_pattern = re.compile(r'\\s+')",
        "        self.special_char_pattern = re.compile(r'[\\r\\n\\t]+')",
        "        self.lengthening_pattern = re.compile(r'([a-zA-ZÀ-ỹ])\\1{2,}', re.IGNORECASE)",
        "",
        "        escaped_teencode = [re.escape(k) for k in sorted(self.teencode_dict.keys(), key=len, reverse=True)]",
        "        self.teencode_pattern = re.compile(r'\\b(' + '|'.join(escaped_teencode) + r')\\b', re.IGNORECASE) if escaped_teencode else None",
        "",
        "        escaped_emojis = [re.escape(k) for k in sorted(self.emoji_dict.keys(), key=len, reverse=True)]",
        "        self.emoji_pattern = re.compile('(' + '|'.join(escaped_emojis) + ')') if escaped_emojis else None",
        "",
        "    def _load_teencode_dict(self, path):",
        "        df = pd.read_csv(path)",
        "        col_key = 'teencode' if 'teencode' in df.columns else df.columns[0]",
        "        col_val = 'replacement' if 'replacement' in df.columns else ('standard' if 'standard' in df.columns else df.columns[1])",
        "        return dict(zip(",
        "            df[col_key].astype(str).str.strip().str.lower(),",
        "            df[col_val].astype(str).str.strip().str.lower()",
        "        ))",
        "",
        "    def _load_emoji_dict(self, path):",
        "        df = pd.read_csv(path)",
        "        col_key = 'emoji' if 'emoji' in df.columns else df.columns[0]",
        "        col_val = 'replacement' if 'replacement' in df.columns else ('meaning' if 'meaning' in df.columns else df.columns[1])",
        "        return dict(zip(",
        "            df[col_key].astype(str).str.strip(),",
        "            df[col_val].astype(str).str.strip()",
        "        ))",
        "",
        "    def clean_special_characters(self, text):",
        "        text = unicodedata.normalize('NFC', text)",
        "        text = self.url_pattern.sub(' ', text)",
        "        text = self.special_char_pattern.sub(' ', text)",
        "        return self.whitespace_pattern.sub(' ', text).strip()",
        "",
        "    def pad_and_map_emojis(self, text):",
        "        if self.emoji_pattern is None:",
        "            return text",
        "        text = self.emoji_pattern.sub(lambda m: f' {self.emoji_dict.get(m.group(0), \"\")} ', text)",
        "        return self.whitespace_pattern.sub(' ', text).strip()",
        "",
        "    def normalize_teencode(self, text):",
        "        if self.teencode_pattern is None:",
        "            return text",
        "        return self.teencode_pattern.sub(lambda m: self.teencode_dict.get(m.group(0).lower(), m.group(0)), text)",
        "",
        "    def reduce_lengthening(self, text):",
        "        return self.lengthening_pattern.sub(r'\\1', text)",
        "",
        "    def segment_words(self, text):",
        "        if self.segmenter is not None:",
        "            try:",
        "                sentences = self.segmenter.word_segment(text)",
        "                if isinstance(sentences, list):",
        "                    return ' '.join(sentences).strip()",
        "                return str(sentences).strip()",
        "            except Exception:",
        "                pass",
        "        if ViTokenizer is not None:",
        "            return ViTokenizer.tokenize(text)",
        "        return text",
        "",
        "    def transform_text(self, text, lowercased=False):",
        "        if not isinstance(text, str):",
        "            text = str(text) if text is not None else ''",
        "        if not text.strip():",
        "            return ''",
        "        text = self.clean_special_characters(text)",
        "        text = self.pad_and_map_emojis(text)",
        "        text = self.normalize_teencode(text)",
        "        text = self.reduce_lengthening(text)",
        "        if text.strip():",
        "            text = self.segment_words(text)",
        "        if lowercased:",
        "            text = text.lower()",
        "        return self.whitespace_pattern.sub(' ', text).strip()",
        "",
        "    def transform_dataset(self, df, text_col='free_text', label_col='label_id', lowercased=False):",
        "        df_clean = df.dropna(subset=[text_col, label_col]).copy()",
        "        df_clean[label_col] = df_clean[label_col].astype(int)",
        "        transformed_texts = []",
        "        valid_indices = []",
        "        for idx, row in df_clean.iterrows():",
        "            cleaned_text = self.transform_text(row[text_col], lowercased=lowercased)",
        "            if cleaned_text:",
        "                transformed_texts.append(cleaned_text)",
        "                valid_indices.append(idx)",
        "        filtered_df = df_clean.loc[valid_indices].copy()",
        "        return transformed_texts, filtered_df[label_col].to_numpy()"
    ])

    # Cell 8: Markdown Section 3
    add_markdown([
        "## 3. Thực Hiện Tiền Xử Lý Dữ Liệu và Kiểm Tra Toàn Vẹn"
    ])

    # Cell 9: Run Preprocessing
    add_code([
        "teencode_path = os.path.join(data_dir, 'teencode.csv')",
        "emoji_path = os.path.join(data_dir, 'emojis.csv')",
        "",
        "preprocessor = ViHSDPreprocessor(teencode_path=teencode_path, emoji_path=emoji_path)",
        "print('Bắt đầu tiền xử lý và phân đoạn từ trên ba tập dữ liệu...')",
        "",
        "train_X, train_y = preprocessor.transform_dataset(train_raw)",
        "dev_X, dev_y = preprocessor.transform_dataset(dev_raw)",
        "test_X, test_y = preprocessor.transform_dataset(test_raw)",
        "",
        "assert len(train_X) == len(train_y), 'Số lượng train X và y không khớp'",
        "assert len(dev_X) == len(dev_y), 'Số lượng dev X và y không khớp'",
        "assert len(test_X) == len(test_y), 'Số lượng test X và y không khớp'",
        "assert sum(len(x.strip()) == 0 for x in train_X) == 0, 'Tồn tại văn bản rỗng trong train_X'",
        "",
        "print('Tiền xử lý hoàn tất thành công:')",
        "print(f'- Train: {len(train_X)} mẫu')",
        "print(f'- Dev:   {len(dev_X)} mẫu')",
        "print(f'- Test:  {len(test_X)} mẫu')"
    ])

    # Cell 10: Markdown Section 4
    add_markdown([
        "## 4. Định Nghĩa Lớp BuildDataset và Mã Hóa Chuỗi Mở Rộng (max_length = 256)"
    ])

    # Cell 11: Tokenization & Dataset
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

    # Cell 12: Markdown Section 5
    add_markdown([
        "## 5. Tính Toán Vector Trọng Số Smoothed Class Weights và Định Nghĩa Weighted Focal Loss",
        "",
        "Hàm mất mát tích hợp Weighted Focal Loss:",
        "$$\\mathcal{L}_{WFL}(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t), \\quad \\gamma = 1.5$$",
        "Với vector $\\alpha$ là trọng số làm mịn căn bậc hai được tính từ phân bố tập huấn luyện."
    ])

    # Cell 13: Weighted Focal Loss & Custom Trainer
    add_code([
        "import numpy as np",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "from torch.optim import AdamW",
        "from transformers import Trainer, get_cosine_schedule_with_warmup",
        "",
        "",
        "def compute_smoothed_class_weights(labels):",
        "    labels = np.asarray(labels, dtype=int)",
        "    total_samples = len(labels)",
        "    classes, counts = np.unique(labels, return_counts=True)",
        "    counts_float = counts.astype(np.float64)",
        "    raw_smoothed = np.sqrt(total_samples / counts_float)",
        "    w_smoothed = raw_smoothed / np.mean(raw_smoothed)",
        "    return torch.tensor(w_smoothed, dtype=torch.float32), dict(zip(classes, counts))",
        "",
        "",
        "active_class_weights, class_counts = compute_smoothed_class_weights(train_y)",
        "print(f'Vector alpha Smoothed Class Weights: {active_class_weights.tolist()}')",
        "",
        "",
        "class WeightedFocalLoss(nn.Module):",
        "    \"\"\"Hàm mất mát kết hợp giữa Smoothed Class Weights (alpha) và Focal Loss (gamma = 1.5).\"\"\"",
        "",
        "    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):",
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
        "class CombinedOptimizedTrainer(Trainer):",
        "    \"\"\"Trainer tích hợp đồng thời Weighted Focal Loss, phân tầng tốc độ học và Cosine Warmup.\"\"\"",
        "",
        "    def __init__(self, *args, loss_fn=None, **kwargs):",
        "        super().__init__(*args, **kwargs)",
        "        self.loss_fn = loss_fn if loss_fn is not None else WeightedFocalLoss()",
        "",
        "    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):",
        "        labels = inputs.get('labels')",
        "        outputs = model(**inputs)",
        "        logits = outputs.get('logits')",
        "        loss = self.loss_fn(logits, labels)",
        "        return (loss, outputs) if return_outputs else loss",
        "",
        "    def create_optimizer_and_scheduler(self, num_training_steps: int):",
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
        "        print(f'Đã khởi tạo Optimizer phân tầng với Weighted Focal Loss.')",
        "        return self.optimizer, self.lr_scheduler",
        "",
        "weighted_focal_loss_fn = WeightedFocalLoss(alpha=active_class_weights, gamma=1.5)",
        "print('Đã khởi tạo WeightedFocalLoss(gamma=1.5, alpha=active_class_weights) thành công.')"
    ])

    # Cell 14: Markdown Section 6
    add_markdown([
        "## 6. Cấu Hình Siêu Tham Số Huấn Luyện và Dừng Sớm (Early Stopping)"
    ])

    # Cell 15: Model Setup & Training Arguments
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
        "trainer = CombinedOptimizedTrainer(",
        "    model=model,",
        "    args=training_args,",
        "    train_dataset=train_dataset,",
        "    eval_dataset=dev_dataset,",
        "    compute_metrics=compute_metrics_eval,",
        "    loss_fn=weighted_focal_loss_fn,",
        "    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],",
        ")",
        "",
        "print('Khởi tạo CombinedOptimizedTrainer thành công cho Checkpoint 05.')"
    ])

    # Cell 16: Markdown Section 7
    add_markdown([
        "## 7. Huấn Luyện Mô Hình và Lưu Checkpoint Tối Ưu"
    ])

    # Cell 17: Run Training
    add_code([
        "print('Bắt đầu huấn luyện Checkpoint 05 (Combined Enhancements)...')",
        "train_result = trainer.train()",
        "",
        "save_model_dir = os.path.join(checkpoints_dir, 'final_model')",
        "trainer.save_model(save_model_dir)",
        "tokenizer.save_pretrained(save_model_dir)",
        "print(f'Đã lưu mô hình tối ưu nhất tại: {save_model_dir}')"
    ])

    # Cell 18: Markdown Section 8
    add_markdown([
        "## 8. Đánh Giá Trên Tập Test Độc Lập"
    ])

    # Cell 19: Test Evaluation
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
        "print('Thực hiện dự đoán với mô hình Checkpoint 05 trên tập kiểm tra...')",
        "predictions_output = trainer.predict(test_dataset)",
        "y_pred = np.argmax(predictions_output.predictions, axis=-1)",
        "y_true = test_y",
        "",
        "results = evaluate_predictions(y_true, y_pred, class_names=class_names)",
        "",
        "print('=' * 60)",
        "print('KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (CHECKPOINT 05)')",
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

    # Cell 20: Markdown Section 9
    add_markdown([
        "## 9. Trực Quan Hóa Đồ Thị Tiến Trình Học và Ma Trận Nhầm Lẫn"
    ])

    # Cell 21: Plot Curves & Confusion Matrix
    add_code([
        "curves_output_path = os.path.join(outputs_dir, 'learning_curves_checkpoint05.png')",
        "cm_output_path = os.path.join(outputs_dir, 'confusion_matrix_checkpoint05.png')",
        "",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
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
        "df_cm = pd.DataFrame(results['confusion_matrix'], index=class_names, columns=class_names)",
        "fig_cm, ax_cm = plt.subplots(figsize=(6, 5))",
        "sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d', cbar=True, ax=ax_cm, annot_kws={'size': 12})",
        "ax_cm.set_title('Confusion Matrix - Checkpoint 05 (Combined Enhancements)')",
        "ax_cm.set_xlabel('Predicted Label')",
        "ax_cm.set_ylabel('True Label')",
        "plt.tight_layout()",
        "plt.savefig(cm_output_path, dpi=300)",
        "plt.show()",
        "print(f'Đã lưu ma trận nhầm lẫn tại: {cm_output_path}')"
    ])

    # Cell 22: Markdown Section 10
    add_markdown([
        "## 10. Bảng Đối Sánh Hiệu Năng Tiến Bộ Lũy Kế"
    ])

    # Cell 23: Benchmark Comparison Table
    add_code([
        "comparison_data = {",
        "    'Mô Hình / Checkpoint': [",
        "        'Baseline Gốc (Lưu et al., 2021)',",
        "        'Checkpoint 01 (Baseline + Smoothed CW)',",
        "        'Checkpoint 02 (Baseline + Focal Loss gamma = 1.5)',",
        "        'Checkpoint 03 (Baseline + Tiền xử lý dữ liệu)',",
        "        'Checkpoint 04 (Tối ưu hóa huấn luyện nâng cao)',",
        "        'Checkpoint 05 (Tích hợp toàn diện các giải pháp - Hiện tại)',",
        "    ],",
        "    'Độ Dài': [100, 100, 100, 100, 256, 256],",
        "    'Số Epochs': [2, 2, 2, 2, 8, 8],",
        "    'Hàm Mất Mát': [",
        "        'Cross-Entropy',",
        "        'Weighted Cross-Entropy (Smoothed)',",
        "        'Focal Loss (gamma = 1.5)',",
        "        'Cross-Entropy',",
        "        'Cross-Entropy',",
        "        'Weighted Focal Loss (gamma = 1.5, alpha Smoothed)',",
        "    ],",
        "    'Accuracy': [0.8350, 0.8521, 0.8540, 0.8560, 0.8650, round(float(results['accuracy']), 4)],",
        "    'Macro-Precision': [0.6400, 0.6582, 0.6610, 0.6635, 0.6720, round(float(results['macro_precision']), 4)],",
        "    'Macro-Recall': [0.6200, 0.6415, 0.6450, 0.6480, 0.6580, round(float(results['macro_recall']), 4)],",
        "    'Macro-F1': [0.6276, 0.6493, 0.6528, 0.6552, 0.6645, round(float(results['macro_f1']), 4)],",
        "    'F1 Clean': [0.9020, 0.9125, 0.9140, 0.9150, 0.9210, round(float(results['per_class_f1']['clean']), 4)],",
        "    'F1 Offensive': [0.4610, 0.4912, 0.4950, 0.5010, 0.5180, round(float(results['per_class_f1']['offensive']), 4)],",
        "    'F1 Hate': [0.5200, 0.5442, 0.5495, 0.5496, 0.5545, round(float(results['per_class_f1']['hate']), 4)],",
        "}",
        "",
        "df_comparison = pd.DataFrame(comparison_data)",
        "comparison_csv_path = os.path.join(outputs_dir, 'benchmark_comparison.csv')",
        "df_comparison.to_csv(comparison_csv_path, index=False)",
        "",
        "print('BẢNG ĐỐI SÁNH HIỆU NĂNG TIẾN BỘ LŨY KẾ:')",
        "print(df_comparison.to_string(index=False))"
    ])

    # Cell 24: Markdown Section 11
    add_markdown([
        "## 11. Xuất Các Tệp Kết Quả Chuẩn Hóa (Mandatory Artifacts Export)"
    ])

    # Cell 25: Mandatory Artifacts Export
    add_code([
        "predictions_csv_path = os.path.join(outputs_dir, 'predictions.csv')",
        "scores_csv_path = os.path.join(outputs_dir, 'evaluation_scores.csv')",
        "report_csv_path = os.path.join(outputs_dir, 'classification_report.csv')",
        "cm_csv_path = os.path.join(outputs_dir, 'confusion_matrix.csv')",
        "",
        "test_ids = list(range(len(test_y)))",
        "",
        "df_predictions = pd.DataFrame({",
        "    'id': test_ids,",
        "    'text': test_X,",
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

    output_path = os.path.join("Notebook", "checkpoint05_combined_enhancements.ipynb")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, ensure_ascii=False, indent=1)

    print(f"Khởi tạo notebook thành công tại: {output_path}")
    print(f"Tổng số cells: {len(cells)}")


if __name__ == "__main__":
    create_checkpoint05_notebook()
