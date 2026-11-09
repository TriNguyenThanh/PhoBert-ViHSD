import os
import re
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
