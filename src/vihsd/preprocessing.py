import os
import re
import unicodedata
import pandas as pd
import numpy as np

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None


class ViHSDPreprocessor:
    """Bộ tiền xử lý và chuẩn hóa văn bản cho bài toán phát hiện phát ngôn thù ghét (ViHSD)."""

    # Khởi tạo đối tượng tiền xử lý với các đường dẫn từ điển teencode và emoji
    def __init__(self, teencode_path="data/teencode.csv", emoji_path="data/emojis.csv"):
        self.teencode_dict = self.load_teencode(teencode_path)
        self.emoji_dict = self.load_emoji_mapping(emoji_path)
        self._init_regex_patterns()

    # Biên dịch trước toàn bộ biểu thức chính quy nhằm tối ưu tốc độ xử lý trên tập dữ liệu lớn
    def _init_regex_patterns(self):
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
        self.email_pattern = re.compile(r"\S+@\S+", re.IGNORECASE)
        self.html_pattern = re.compile(r"<.*?>")
        self.whitespace_pattern = re.compile(r"\s+")

        # Nhận diện ký tự chữ cái lặp lại từ 3 lần trở lên bao gồm cả bảng chữ cái tiếng Việt
        self.lengthening_pattern = re.compile(
            r"([a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ])\1{2,}",
            re.IGNORECASE,
        )

        # Đệm khoảng trắng bao quanh emoji để tránh dính từ khi tách từ bằng PyVi
        self.emoji_padding_pattern = re.compile(
            r"(\s*)([\U00010000-\U0010ffff☀-⟿])(\s*)"
        )

        # Mẫu khớp từ điển emoji theo thứ tự độ dài giảm dần
        sorted_emojis = sorted(self.emoji_dict.keys(), key=len, reverse=True)
        self.emoji_replace_pattern = re.compile(
            "|".join(re.escape(k) for k in sorted_emojis)
        )

        # Mẫu khớp teencode kết hợp biên giới từ để tránh thay thế nhầm từ tố hợp lệ
        sorted_teencode = sorted(self.teencode_dict.keys(), key=len, reverse=True)
        teencode_patterns = []
        for k in sorted_teencode:
            escaped = re.escape(k)
            left = r"(?<!\w)" if re.match(r"^\w", k) else ""
            right = r"(?!\w)" if re.search(r"\w$", k) else ""
            teencode_patterns.append(f"{left}{escaped}{right}")
        self.teencode_pattern = re.compile("|".join(teencode_patterns), re.IGNORECASE)

    # Đọc từ điển teencode và từ lóng từ tệp CSV thành từ điển ánh xạ chuẩn hóa
    def load_teencode(self, path):
        df = pd.read_csv(path)
        return dict(
            zip(
                df["teencode"].astype(str).str.strip().str.lower(),
                df["replacement"].astype(str).str.strip().str.lower(),
            )
        )

    # Đọc bảng ánh xạ biểu tượng cảm xúc (emoji) từ tệp CSV thành từ điển thay thế
    def load_emoji_mapping(self, path):
        df = pd.read_csv(path)
        return dict(
            zip(
                df["emoji"].astype(str).str.strip(),
                df["replacement"].astype(str).str.strip(),
            )
        )

    # Loại bỏ đường dẫn URL, địa chỉ email, thẻ HTML và chuẩn hóa các khoảng trắng thừa
    def clean_special_characters(self, text):
        text = self.url_pattern.sub(" ", text)
        text = self.email_pattern.sub(" ", text)
        text = self.html_pattern.sub(" ", text)
        return self.whitespace_pattern.sub(" ", text).strip()

    # Xử lý emoji 2 pha: đệm khoảng trắng bảo vệ từ tố rồi ánh xạ ngữ nghĩa sang từ tố PhoBERT
    def pad_and_map_emojis(self, text):
        text = self.emoji_padding_pattern.sub(r" \2 ", text)
        text = self.emoji_replace_pattern.sub(
            lambda m: f" {self.emoji_dict[m.group(0)]} ", text
        )
        return self.whitespace_pattern.sub(" ", text).strip()

    # Chuẩn hóa từ viết tắt và tiếng lóng dựa trên biên giới từ để tránh thay thế sai từ vựng gốc
    def normalize_teencode(self, text):
        return self.teencode_pattern.sub(
            lambda m: self.teencode_dict.get(m.group(0).lower(), m.group(0)), text
        )

    # Thu gọn các ký tự chữ cái lặp lại liên tiếp từ 3 lần trở lên về 1 ký tự duy nhất
    def reduce_lengthening(self, text):
        return self.lengthening_pattern.sub(r"\1", text)

    # Phân đoạn từ tiếng Việt tự động bằng thư viện PyVi theo quy chuẩn từ vựng của PhoBERT
    def segment_words(self, text):
        if ViTokenizer is None:
            raise ImportError(
                "Thư viện pyvi chưa được cài đặt. Vui lòng cài đặt qua lệnh: pip install pyvi"
            )
        return ViTokenizer.tokenize(text)

    # Thực hiện quy trình tiền xử lý hoàn chỉnh cho một đoạn văn bản theo đúng thứ tự logic
    def transform_text(self, text, lowercased=False):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if not text.strip():
            return ""

        # Quy trình tuần tự: Làm sạch -> Emoji -> Teencode -> Thu gọn lặp -> Tách từ PyVi
        text = self.clean_special_characters(text)
        text = self.pad_and_map_emojis(text)
        text = self.normalize_teencode(text)
        text = self.reduce_lengthening(text)

        # Bảo tồn đại từ nhân xưng và từ phủ định, không lọc từ dừng
        if text.strip():
            text = self.segment_words(text)

        if lowercased:
            text = text.lower()

        return self.whitespace_pattern.sub(" ", text).strip()

    # Áp dụng chuẩn hóa trên toàn bộ DataFrame, loại bỏ dòng rỗng/thiếu và trả về mảng đặc trưng X và nhãn y
    def transform_dataset(self, df, text_col="free_text", label_col="label_id", lowercased=False):
        df_clean = df.dropna(subset=[text_col, label_col]).copy()
        df_clean[label_col] = df_clean[label_col].astype(int)

        transformed_texts = []
        valid_indices = []

        for idx, row in df_clean.iterrows():
            cleaned_text = self.transform_text(row[text_col], lowercased=lowercased)
            if cleaned_text:
                transformed_texts.append(cleaned_text)
                valid_indices.append(idx)

        filtered_df = df_clean.loc[valid_indices].copy()
        X = transformed_texts
        y = filtered_df[label_col].to_numpy()

        return X, y

    # Chuẩn hóa chuỗi văn bản phục vụ so khớp và phát hiện trùng lặp/rò rỉ dữ liệu
    @staticmethod
    def normalize_matching_key(text):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = unicodedata.normalize("NFC", text).strip().lower()
        return re.sub(r"\s+", " ", text)

    # Kiểm kê toàn diện hiện tượng trùng lặp nội bộ và rò rỉ dữ liệu (Data Leakage) sang Dev/Test
    @classmethod
    def audit_duplicates_and_leakage(
        cls,
        train_df,
        dev_df=None,
        test_df=None,
        text_col="free_text",
        label_col="label_id",
    ):
        train_valid = train_df.dropna(subset=[text_col, label_col]).copy()
        train_valid[label_col] = train_valid[label_col].astype(int)
        train_valid["__norm_key__"] = train_valid[text_col].apply(cls.normalize_matching_key)

        dev_keys = (
            set(dev_df[text_col].dropna().apply(cls.normalize_matching_key))
            if dev_df is not None
            else set()
        )
        test_keys = (
            set(test_df[text_col].dropna().apply(cls.normalize_matching_key))
            if test_df is not None
            else set()
        )
        eval_keys = dev_keys | test_keys

        leak_dev = train_valid["__norm_key__"].isin(dev_keys)
        leak_test = train_valid["__norm_key__"].isin(test_keys)
        leak_eval = train_valid["__norm_key__"].isin(eval_keys)

        label_counts = train_valid.groupby("__norm_key__")[label_col].nunique()
        conflict_keys = set(label_counts[label_counts > 1].index)
        conflict_rows = train_valid["__norm_key__"].isin(conflict_keys)

        exact_dups = train_valid.duplicated(subset=["__norm_key__", label_col], keep="first")

        return {
            "total_train_rows": len(train_valid),
            "unique_train_texts": int(train_valid["__norm_key__"].nunique()),
            "leak_dev_rows": int(leak_dev.sum()),
            "leak_dev_unique": int(train_valid.loc[leak_dev, "__norm_key__"].nunique()),
            "leak_test_rows": int(leak_test.sum()),
            "leak_test_unique": int(train_valid.loc[leak_test, "__norm_key__"].nunique()),
            "leak_total_rows": int(leak_eval.sum()),
            "leak_total_unique": int(train_valid.loc[leak_eval, "__norm_key__"].nunique()),
            "conflict_unique_texts": len(conflict_keys),
            "conflict_total_rows": int(conflict_rows.sum()),
            "exact_dup_rows": int(exact_dups.sum()),
        }

    # Xử lý khử trùng lặp và triệt tiêu hoàn toàn rò rỉ dữ liệu (Data Leakage) khỏi tập huấn luyện
    @classmethod
    def clean_and_deduplicate(
        cls,
        train_df,
        dev_df=None,
        test_df=None,
        text_col="free_text",
        label_col="label_id",
        drop_conflicting=True,
        verbose=True,
    ):
        """
        Làm sạch tập dữ liệu huấn luyện:
        1. Triệt tiêu rò rỉ dữ liệu: Lọc bỏ các mẫu train có mặt trong tập dev hoặc test.
        2. Loại bỏ nhiễu nhãn mâu thuẫn: Loại bỏ các câu văn bản cùng nội dung nhưng khác nhãn.
        3. Khử trùng lặp nội bộ: Giữ lại bản ghi duy nhất cho các câu giống hệt nhau.
        """
        stats = cls.audit_duplicates_and_leakage(
            train_df, dev_df, test_df, text_col=text_col, label_col=label_col
        )

        train_valid = train_df.dropna(subset=[text_col, label_col]).copy()
        train_valid[label_col] = train_valid[label_col].astype(int)
        train_valid["__norm_key__"] = train_valid[text_col].apply(cls.normalize_matching_key)

        # 1. Loại bỏ rò rỉ dữ liệu sang tập Dev và Test
        dev_keys = (
            set(dev_df[text_col].dropna().apply(cls.normalize_matching_key))
            if dev_df is not None
            else set()
        )
        test_keys = (
            set(test_df[text_col].dropna().apply(cls.normalize_matching_key))
            if test_df is not None
            else set()
        )
        eval_keys = dev_keys | test_keys

        df_no_leak = train_valid[~train_valid["__norm_key__"].isin(eval_keys)].copy()

        # 2. Xử lý mẫu mâu thuẫn nhãn
        label_counts = df_no_leak.groupby("__norm_key__")[label_col].nunique()
        rem_conflict_keys = set(label_counts[label_counts > 1].index)

        if drop_conflicting:
            df_no_conflict = df_no_leak[~df_no_leak["__norm_key__"].isin(rem_conflict_keys)].copy()
        else:
            df_no_conflict = df_no_leak.drop_duplicates(subset=["__norm_key__"], keep="first").copy()

        # 3. Khử trùng lặp nội bộ
        df_clean = df_no_conflict.drop_duplicates(subset=["__norm_key__"], keep="first").copy()
        removed_count = len(train_valid) - len(df_clean)

        if verbose:
            print("=" * 75)
            print("BÁO CÁO KIỂM KÊ TRÙNG LẶP VÀ RÒ RỈ DỮ LIỆU (DATA LEAKAGE & DUPLICATES AUDIT)")
            print("=" * 75)
            print(f"- Tổng số mẫu Train ban đầu hợp lệ:        {stats['total_train_rows']:>6} mẫu ({stats['unique_train_texts']} văn bản duy nhất)")
            print(f"- Rò rỉ dữ liệu sang tập Dev (Validation): {stats['leak_dev_rows']:>6} dòng ({stats['leak_dev_unique']} văn bản)")
            print(f"- Rò rỉ dữ liệu sang tập Test (Độc lập):   {stats['leak_test_rows']:>6} dòng ({stats['leak_test_unique']} văn bản)")
            print(f"- Tổng mẫu Train trùng lặp với Dev/Test:   {stats['leak_total_rows']:>6} dòng ({stats['leak_total_unique']} văn bản rò rỉ)")
            print(f"- Mẫu mâu thuẫn nhãn nội bộ (Label Noise): {stats['conflict_total_rows']:>6} dòng ({stats['conflict_unique_texts']} văn bản khác nhãn)")
            print(f"- Mẫu trùng lặp cùng nhãn (Exact Dups):    {stats['exact_dup_rows']:>6} dòng")
            print("-" * 75)
            print(f"- Đã loại bỏ triệt để khỏi tập Train:      {removed_count:>6} mẫu không hợp lệ/rò rỉ")
            print(f"- Số lượng mẫu Train sạch sau xử lý:       {len(df_clean):>6} mẫu chuẩn hóa")
            print("=" * 75)

        return df_clean.drop(columns=["__norm_key__"])
