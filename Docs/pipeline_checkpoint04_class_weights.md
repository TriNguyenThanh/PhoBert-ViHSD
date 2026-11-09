# Đặc Tả Pipeline Checkpoint 4: Xử Lý Mất Cân Bằng Lớp Bằng Class Weight

Tài liệu này đặc tả chi tiết các công thức toán học tính toán vector trọng số lớp, phương thức tích hợp vào hàm mất mát Weighted Cross-Entropy, phân tích sự đánh đổi Precision-Recall và quy trình thực nghiệm cho Checkpoint 4 trong bài toán phân loại phát ngôn thù ghét tiếng Việt (ViHSD).

---

## 1. Mục Tiêu Kỹ Thuật

- Cân bằng độ dốc gradient giữa các lớp thông qua cơ chế phạt tĩnh: gán trọng số phạt lớn hơn cho các mẫu thuộc lớp thiểu số (OFFENSIVE và HATE) khi mô hình dự đoán sai.
- Khảo sát và so sánh 3 phương pháp tính trọng số lớp:
  1. Nghịch đảo tần suất lớp (Inverse Class Frequency - ICF).
  2. Làm mịn căn bậc hai (Smoothed Inverse Frequency).
  3. Số lượng mẫu hiệu dụng (Effective Number of Samples - Cui et al., 2019).
- Đánh giá sự cân bằng giữa độ nhạy (Recall) của lớp thiểu số và độ chính xác (Precision) của lớp đa số.

---

## 2. Kiến Trúc Luồng Xử Lý (Class Weight Pipeline Architecture)

```
[Tập dữ liệu huấn luyện: train.csv (24.048 mẫu)]
                     |
                     v
  (Bước 1) Đếm phân bố tần suất lớp N_c (c = 0: CLEAN, 1: OFFENSIVE, 2: HATE)
                     |
                     v
  (Bước 2) Tính toán Vector trọng số W qua 3 phương pháp (ICF, Smoothed, Effective)
                     |
                     v
  (Bước 3) Chuẩn hóa Vector trọng số và chuyển thành Tensor trên GPU
                     |
                     v
  (Bước 4) Cấu hình nn.CrossEntropyLoss(weight=class_weights) trong Trainer
                     |
                     v
  (Bước 5) Huấn luyện mô hình theo cấu hình CP2 (max_len=256, 8 epochs)
                     |
                     v
  (Bước 6) Đánh giá và chọn phương pháp tính trọng số tối ưu trên Dev Set
                     |
                     v
  (Bước 7) Đánh giá độc lập trên Test Set -> So sánh với Baseline và Checkpoint 3
```

---

## 3. Cơ Sở Toán Học và Các Công Thức Tính Trọng Số Lớp

### 3.1. Phương pháp 1: Nghịch đảo tần suất lớp (Inverse Class Frequency - ICF)
Công thức tiêu chuẩn của thư viện Scikit-Learn:
$$w_c = \frac{N}{K \cdot N_c}$$
Trong đó:
- $N$ là tổng số mẫu trong tập huấn luyện ($N = 24.048$).
- $K = 3$ là số lớp phân loại.
- $N_c$ là số lượng mẫu của lớp $c$.

*Số liệu ước tính trên tập ViHSD:*
- Lớp 0 (CLEAN): $N_0 \approx 19.888$ mẫu $\to w_0 = \frac{24048}{3 \times 19888} \approx 0.403$
- Lớp 1 (OFFENSIVE): $N_1 \approx 1.610$ mẫu $\to w_1 = \frac{24048}{3 \times 1610} \approx 4.979$
- Lớp 2 (HATE): $N_2 \approx 2.550$ mẫu $\to w_2 = \frac{24048}{3 \times 2550} \approx 3.144$

### 3.2. Phương pháp 2: Làm mịn căn bậc hai (Smoothed Inverse Frequency)
Nhằm tránh tình trạng trọng số của lớp 1 (OFFENSIVE) quá lớn gây bất ổn định gradient và làm sụp đổ Precision của lớp CLEAN:
$$w_c = \sqrt{\frac{N}{N_c}}$$
Sau đó chuẩn hóa vector để giá trị trung bình bằng 1:
$$\hat{w}_c = \frac{w_c}{\frac{1}{K}\sum_{j=0}^{K-1} w_j}$$

### 3.3. Phương pháp 3: Số lượng mẫu hiệu dụng (Effective Number of Samples - Cui et al., 2019)
Phương pháp này dựa trên lý thuyết thông tin biên: khi số lượng mẫu của một lớp tăng lên, mỗi mẫu mới thêm vào đóng góp ít thông tin độc lập hơn do hiện tượng chồng lấn không gian đặc trưng.
Thể tích bao phủ hiệu dụng được tính:
$$E_{N_c} = \frac{1 - \beta^{N_c}}{1 - \beta}$$
Trong đó $\beta \in [0.99, 0.9999]$ là siêu tham số đại diện cho tỷ lệ không gian đặc trưng mới.
Trọng số tương ứng:
$$w_c \propto \frac{1}{E_{N_c}} = \frac{1 - \beta}{1 - \beta^{N_c}}$$

---

## 4. Cài Đặt Kỹ Thuật Chi Tiết

### 4.1. Script tính toán trọng số lớp
Triển khai tại [Scripts/compute_class_weights.py](Scripts/compute_class_weights.py):

```python
import numpy as np
import pandas as pd
import torch

def compute_weights(train_csv_path, method='icf', beta=0.999):
    df = pd.read_csv(train_csv_path)
    labels = df['label_id'].values
    num_samples = len(labels)
    classes, counts = np.unique(labels, return_counts=True)
    num_classes = len(classes)
    
    if method == 'icf':
        weights = num_samples / (num_classes * counts.astype(np.float32))
    elif method == 'smoothed':
        weights = np.sqrt(num_samples / counts.astype(np.float32))
        weights = weights / weights.mean()
    elif method == 'effective':
        effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        weights = 1.0 / effective_num
        weights = weights / weights.sum() * num_classes
    else:
        raise ValueError("Phương pháp không hợp lệ.")
        
    return torch.tensor(weights, dtype=torch.float32)
```

### 4.2. Cấu hình Hàm mất mát Weighted Cross-Entropy trong PyTorch
Truyền vector trọng số trực tiếp vào hàm mất mát tiêu chuẩn của PyTorch:

```python
import torch
import torch.nn as nn
from transformers import Trainer

class WeightedCETrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            # Gán vector trọng số vào thiết bị tính toán (GPU/CPU)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Đảm bảo vector trọng số cùng thiết bị với logits
        if self.loss_fn.weight is not None and self.loss_fn.weight.device != logits.device:
            self.loss_fn.weight = self.loss_fn.weight.to(logits.device)
            
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
```

---

## 5. Quy Trình Khảo Sát và Bảng Thực Nghiệm Đối Sánh

Thực nghiệm so sánh 3 phương pháp tính trọng số trên cùng một cấu hình mô hình (PhoBERT-base, `max_length = 256`, 8 epochs, optimizer cấu hình như Checkpoint 2):

| Cấu hình Trọng số | Vector $W = [w_0, w_1, w_2]$ | Dev Macro-F1 | Test Macro-F1 | F1 - CLEAN | F1 - OFFENSIVE | F1 - HATE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Không trọng số (CP2)** | $[1.00, 1.00, 1.00]$ | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |
| **Nghịch đảo tần suất (ICF)**| $[0.40, 4.98, 3.14]$ | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |
| **Làm mịn căn bậc hai** | Ghi nhận giá trị | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |
| **Mẫu hiệu dụng ($\beta=0.999$)**| Ghi nhận giá trị | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận | Ghi nhận |

---

## 6. Quy Trình Kiểm Thử và Tiêu Chuẩn Nghiệm Thu

1. **Phân tích sự đánh đổi Precision - Recall:**
   - Khi áp dụng vector trọng số lớn cho lớp 1 và lớp 2, Recall của OFFENSIVE và HATE sẽ tăng mạnh.
   - Cần kiểm tra chỉ số Precision của lớp 0 (CLEAN): nếu Precision giảm dưới ngưỡng chấp nhận được (gây ra tình trạng báo động giả - false positive quá nhiều), cần ưu tiên phương pháp làm mịn căn bậc hai hoặc số lượng mẫu hiệu dụng.

2. **Tiêu chuẩn nghiệm thu:**
   - Thực thi notebook `Notebook/checkpoint04_class_weights.ipynb`.
   - Lựa chọn phương pháp tính trọng số đạt cân bằng Macro-F1 tốt nhất trên tập dev để đưa vào đánh giá cuối cùng trên `test.csv`.
   - Kết quả chỉ số **Macro-F1 trên tập Test** phải ghi nhận sự cải thiện rõ rệt so với mô hình cơ sở không có trọng số.
