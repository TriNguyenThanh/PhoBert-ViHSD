import torch


class BuildDataset(torch.utils.data.Dataset):
    """Lớp quản lý và đóng gói tập dữ liệu tương thích với Hugging Face Trainer và mô hình PhoBERT."""

    # Khởi tạo đối tượng tập dữ liệu với đặc trưng đã được token hóa và mảng nhãn tương ứng
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # Trích xuất mẫu dữ liệu tại chỉ mục xác định và đóng gói thành định dạng PyTorch Tensor
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    # Trả về tổng số lượng mẫu quan sát có trong tập dữ liệu
    def __len__(self):
        return len(self.labels)
