import json
import os


def create_checkpoint03_notebook():
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
        "# Checkpoint 03: Tiền Xử Lý và Chuẩn Hóa Dữ Liệu Mạng Xã Hội (Data Preprocessing)",
        "",
        "Nghiên cứu: Nâng cao hiệu năng mô hình phát hiện phát ngôn thù ghét trên mạng xã hội tiếng Việt (ViHSD).",
        "",
        "Mục tiêu khoa học:",
        "1. Khắc phục hạn chế của pipeline tiền xử lý gốc (vốn loại bỏ stopwords bừa bãi và dùng deEmojify tiếng Anh gây nhiễu).",
        "2. Xây dựng lớp xử lý chuyên biệt `ViHSDPreprocessor` tích hợp trực tiếp trong notebook:",
        "   - Làm sạch nhiễu URL, ký tự điều khiển, chuẩn hóa khoảng trắng.",
        "   - Ánh xạ biểu cảm emoji 2 pha (giữ trọn sắc thái thù ghét/xúc phạm) qua từ điển chuyên dụng `emojis.csv`.",
        "   - Chuẩn hóa tiếng lóng, viết tắt mạng xã hội qua từ điển `teencode.csv` (412 quy tắc chuẩn).",
        "   - Thu gọn ký tự kéo dài về 1 ký tự (`ngonnnn` -> `ngon`).",
        "   - Phân đoạn từ tiếng Việt tự động bằng thư viện `pyvi` theo cấu trúc từ ghép của PhoBERT (`ViTokenizer.tokenize`).",
        "   - **Bảo tồn tuyệt đối** đại từ nhân xưng (`mày`, `tao`, `nó`, `thằng`, `con`) và từ phủ định (`không`, `chẳng`, `chả`).",
        "3. Đánh giá tác động độc lập của bước tiền xử lý trên cấu hình Baseline gốc (`vinai/phobert-base`, 2 epochs, Cross-Entropy Loss tiêu chuẩn).",
        "4. Xuất đầy đủ 4 tệp Mandatory Artifacts theo chuẩn vào thư mục `outputs/checkpoint03/`.",
        "",
        "Ghi chú: Toàn bộ mã nguồn `ViHSDPreprocessor`, `BuildDataset` và hàm đánh giá được tích hợp 100% trong notebook để bảo đảm khả năng chạy độc lập trên Google Colab."
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
        "checkpoints_dir = os.path.join(project_dir, 'checkpoints', 'checkpoint03')",
        "outputs_dir = os.path.join(project_dir, 'outputs', 'checkpoint03')",
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
        "print(f'- Train: {len(train_raw)} mẫu, null free_text: {train_raw[\"free_text\"].isna().sum()}')",
        "print(f'- Dev:   {len(dev_raw)} mẫu, null free_text: {dev_raw[\"free_text\"].isna().sum()}')",
        "print(f'- Test:  {len(test_raw)} mẫu, null free_text: {test_raw[\"free_text\"].isna().sum()}')"
    ])

    # Cell 6: Markdown Section 2
    add_markdown([
        "## 2. Định Nghĩa Bộ Tiền Xử Lý Dữ Liệu Chuyên Sâu (ViHSDPreprocessor)",
        "",
        "Lớp xử lý văn bản tiếng Việt mạng xã hội tự chứa, giải quyết các đặc thù tiếng lóng, emoji và phân đoạn từ."
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
        "    DEFAULT_TEENCODE_DICT = {",
        "        'đm': 'địt mẹ', 'dm': 'địt mẹ', 'dcm': 'địt con mẹ', 'dmm': 'địt mẹ mày',",
        "        'vcl': 'vãi cả lồn', 'vkl': 'vãi cả lồn', 'vl': 'vãi lồn', 'cl': 'cặc lồn',",
        "        'cc': 'cục cặc', 'đéo': 'không', 'deo': 'không', 'đbh': 'đéo bao giờ',",
        "        'ko': 'không', 'k': 'không', 'khong': 'không', 'hok': 'không', 'hổng': 'không',",
        "        'dc': 'được', 'đc': 'được', 'duoc': 'được', 'ntn': 'như thế nào',",
        "        'ng': 'người', 'ngta': 'người ta', 'nyc': 'người yêu cũ',",
        "        'b': 'bạn', 'm': 'mày', 't': 'tao', 'e': 'em', 'a': 'anh',",
        "        'trc': 'trước', 'ncl': 'nói chung là', 'ib': 'nhắn tin', 'inbox': 'nhắn tin',",
        "        'fb': 'facebook', 'yt': 'youtube', 'acc': 'tài khoản',",
        "    }",
        "",
        "    DEFAULT_EMOJI_DICT = {",
        "        '😡': 'tức giận', '😠': 'bực bội', '🤬': 'chửi bới', '👿': 'ác độc', '😈': 'ác quỷ',",
        "        '💩': 'chất thải', '🤮': 'nôn mửa', '🤢': 'kinh tởm', '🖕': 'ngón giữa', '👎': 'tệ hại',",
        "        '💀': 'chết tiệt', '☠️': 'nguy hiểm', '🤡': 'hề hước lố bịch', '🤥': 'dối trá',",
        "        '😭': 'khóc lóc', '😢': 'buồn bã', '😂': 'cười cợt', '🤣': 'cười lăn lộn',",
        "        '❤️': 'yêu thích', '👍': 'tuyệt vời', '🙏': 'cầu xin',",
        "    }",
        "",
        "    def __init__(self, teencode_path=None, emoji_path=None):",
        "        self.teencode_dict = dict(self.DEFAULT_TEENCODE_DICT)",
        "        self.emoji_dict = dict(self.DEFAULT_EMOJI_DICT)",
        "",
        "        if teencode_path and os.path.exists(teencode_path):",
        "            self._load_teencode_dict(teencode_path)",
        "        if emoji_path and os.path.exists(emoji_path):",
        "            self._load_emoji_dict(emoji_path)",
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
        "        try:",
        "            df = pd.read_csv(path)",
        "            if 'teencode' in df.columns and 'standard' in df.columns:",
        "                for _, row in df.iterrows():",
        "                    if pd.notna(row['teencode']) and pd.notna(row['standard']):",
        "                        self.teencode_dict[str(row['teencode']).strip().lower()] = str(row['standard']).strip().lower()",
        "        except Exception as e:",
        "            print(f'Cảnh báo khi đọc teencode: {e}')",
        "",
        "    def _load_emoji_dict(self, path):",
        "        try:",
        "            df = pd.read_csv(path)",
        "            if 'emoji' in df.columns and 'meaning' in df.columns:",
        "                for _, row in df.iterrows():",
        "                    if pd.notna(row['emoji']) and pd.notna(row['meaning']):",
        "                        self.emoji_dict[str(row['emoji']).strip()] = str(row['meaning']).strip()",
        "        except Exception as e:",
        "            print(f'Cảnh báo khi đọc emoji: {e}')",
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
        "## 3. Thực Hiện Tiền Xử Lý Dữ Liệu và Kiểm Tra Tính Toàn Vẹn (Post-pipeline Verification)"
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
        "## 4. Định Nghĩa Lớp BuildDataset và Mã Hóa Bằng PhoBERT FastBPE (max_length = 100)"
    ])

    # Cell 11: Tokenization & Dataset
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

    # Cell 12: Markdown Section 5
    add_markdown([
        "## 5. Cấu Hình Huấn Luyện Cơ Sở và Hàm Mất Mát Tiêu Chuẩn",
        "",
        "Giữ nguyên hàm mất mát Cross-Entropy tiêu chuẩn để cô lập chính xác mức độ đóng góp của bước tiền xử lý chuyên sâu `ViHSDPreprocessor`."
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
        "from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments",
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
        "trainer = Trainer(",
        "    model=model,",
        "    args=training_args,",
        "    train_dataset=train_dataset,",
        "    eval_dataset=dev_dataset,",
        "    compute_metrics=compute_metrics_eval,",
        ")",
        "",
        "print('Khởi tạo Trainer thành công cho Checkpoint 03.')"
    ])

    # Cell 14: Markdown Section 6
    add_markdown([
        "## 6. Huấn Luyện Mô Hình và Lưu Checkpoint"
    ])

    # Cell 15: Run Training
    add_code([
        "print('Bắt đầu huấn luyện Checkpoint 03 (Baseline + Preprocessing)...')",
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
        "print('Thực hiện dự đoán với mô hình Checkpoint 03 trên tập kiểm tra...')",
        "predictions_output = trainer.predict(test_dataset)",
        "y_pred = np.argmax(predictions_output.predictions, axis=-1)",
        "y_true = test_y",
        "",
        "results = evaluate_predictions(y_true, y_pred, class_names=class_names)",
        "",
        "print('=' * 60)",
        "print('KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (CHECKPOINT 03)')",
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
        "cm_output_path = os.path.join(outputs_dir, 'confusion_matrix_checkpoint03.png')",
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
        "    ax.set_title('Confusion Matrix - Checkpoint 03 (Data Preprocessing)')",
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

    # Cell 20: Markdown Section 9
    add_markdown([
        "## 9. Bảng Đối Sánh Hiệu Năng So Với Baseline và Các Checkpoint Trước"
    ])

    # Cell 21: Benchmark Comparison Table
    add_code([
        "comparison_data = {",
        "    'Mô Hình / Checkpoint': [",
        "        'Baseline Gốc (Lưu et al., 2021)',",
        "        'Checkpoint 01 (Baseline + Smoothed CW)',",
        "        'Checkpoint 02 (Baseline + Focal Loss gamma = 1.5)',",
        "        'Checkpoint 03 (Baseline + Tiền xử lý dữ liệu - Hiện tại)',",
        "    ],",
        "    'Độ Dài': [100, 100, 100, 100],",
        "    'Số Epochs': [2, 2, 2, 2],",
        "    'Hàm Mất Mát': [",
        "        'Cross-Entropy',",
        "        'Weighted Cross-Entropy (Smoothed)',",
        "        'Focal Loss (gamma = 1.5)',",
        "        'Cross-Entropy',",
        "    ],",
        "    'Accuracy': [0.8350, 0.8521, 0.8540, round(float(results['accuracy']), 4)],",
        "    'Macro-Precision': [0.6400, 0.6582, 0.6610, round(float(results['macro_precision']), 4)],",
        "    'Macro-Recall': [0.6200, 0.6415, 0.6450, round(float(results['macro_recall']), 4)],",
        "    'Macro-F1': [0.6276, 0.6493, 0.6528, round(float(results['macro_f1']), 4)],",
        "    'F1 Clean': [0.9020, 0.9125, 0.9140, round(float(results['per_class_f1']['clean']), 4)],",
        "    'F1 Offensive': [0.4610, 0.4912, 0.4950, round(float(results['per_class_f1']['offensive']), 4)],",
        "    'F1 Hate': [0.5200, 0.5442, 0.5495, round(float(results['per_class_f1']['hate']), 4)],",
        "}",
        "",
        "df_comparison = pd.DataFrame(comparison_data)",
        "comparison_csv_path = os.path.join(outputs_dir, 'benchmark_comparison.csv')",
        "df_comparison.to_csv(comparison_csv_path, index=False)",
        "",
        "print('BẢNG ĐỐI SÁNH HIỆU NĂNG:')",
        "print(df_comparison.to_string(index=False))"
    ])

    # Cell 22: Markdown Section 10
    add_markdown([
        "## 10. Xuất Các Tệp Kết Quả Chuẩn Hóa (Mandatory Artifacts Export)"
    ])

    # Cell 23: Mandatory Artifacts Export
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

    output_path = os.path.join("Notebook", "checkpoint03_data_preprocessing.ipynb")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, ensure_ascii=False, indent=1)

    print(f"Khởi tạo notebook thành công tại: {output_path}")
    print(f"Tổng số cells: {len(cells)}")


if __name__ == "__main__":
    create_checkpoint03_notebook()
