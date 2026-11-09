"""
Kich ban hop nhat va chuan hoa tu dien teencode tu hai nguon:
- Scripts/teencode.json (nguon uu tien cao nhat)
- Scripts/teencode.json (nguon bo sung)

Ket qua xuat ra:
- data/teencode.csv (dong bo dinh dang voi cac tap tin dataset khac)
"""

import json
import re
import pandas as pd


def clean_normalized_text(text: str) -> str:
    """Lam sach chuoi chuan hoa tu cot Dang chuan hoa cua file Excel."""
    if not isinstance(text, str):
        return ""

    # Loai bo phan chu thich trong ngoac don, vi du: 'em (nhan manh/nung)' -> 'em'
    text = re.sub(r"\(.*?\)", "", text).strip()

    # Neu chuoi chua cac lua chon cach nhau boi dau gach cheo, lay tu tieu bieu dau tien
    if "/" in text:
        parts = [p.strip() for p in text.split("/") if p.strip()]
        if parts:
            text = parts[0]

    # Loai bo khoang trang thua va chuyen ve chu thuong
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def clean_teencode_key(key: str) -> list[str]:
    """Tach cac bien the teencode duoc viet chung trong mot o."""
    if not isinstance(key, str):
        return []

    # Tach theo dau gach cheo neu co nhieu bien the
    keys = [k.strip().lower() for k in key.split("/") if k.strip()]
    return keys


def main():
    json_path = "Scripts/teencode.json"
    xlsx_path = "Scripts/teencode.xlsx"
    output_csv_path = "data/teencode.csv"

    # 1. Doc nguon bo sung tu file Excel
    df_xlsx = pd.read_excel(xlsx_path)
    excel_dict = {}

    for _, row in df_xlsx.iterrows():
        raw_key = str(row.get("Teencode", ""))
        raw_norm = str(row.get("Dạng chuẩn hóa", ""))
        cleaned_norm = clean_normalized_text(raw_norm)

        if not cleaned_norm:
            continue

        keys = clean_teencode_key(raw_key)
        for k in keys:
            # Bo qua cac tu co khoa va gia tri giong het nhau
            if k == cleaned_norm:
                continue
            # Bo qua cac ky hieu bieu tuong cam xuc da co bo xu ly emoji rieng
            if k in [":)", ":(", ":d", ":p", ":v", "^^", "uwu", "owo"]:
                continue
            excel_dict[k] = cleaned_norm

    # 2. Doc nguon goc teencode.json (nguon co do tin cay cao nhat)
    with open(json_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)

    # 3. Hop nhat hai nguon: uu tien gia tri tu teencode.json khi xay ra xung dot
    merged_dict = {}
    merged_dict.update(excel_dict)

    for k, v in json_dict.items():
        k_clean = str(k).strip().lower()
        v_clean = str(v).strip().lower()
        if k_clean and v_clean:
            merged_dict[k_clean] = v_clean

    # 4. Loai bo cac cap gia tri khong hop le hoac bi trung khoa - gia tri
    final_dict = {}
    for k, v in merged_dict.items():
        k = k.strip()
        v = v.strip()
        if k and v and k != v:
            final_dict[k] = v

    # 5. Sap xep va xuat ra dinh dang CSV thong nhat
    records = [{"teencode": k, "replacement": v} for k, v in sorted(final_dict.items())]
    df_merged = pd.DataFrame(records)
    df_merged.to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"Hop nhat thanh cong tu dien teencode:")
    print(f"- So luong tu file JSON: {len(json_dict)}")
    print(f"- So luong dong tu file Excel: {len(df_xlsx)}")
    print(f"- Tong so cap teencode sau khi lam sach va hop nhat: {len(df_merged)}")
    print(f"- Tap tin ket qua: {output_csv_path}")


if __name__ == "__main__":
    main()
