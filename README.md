# AI for Security: A Multimodal Framework for Financial Fraud Detection Using Graph Analytics and Behavioral Velocity

## 1. Giới thiệu

Đề tài xây dựng hệ thống phát hiện gian lận giao dịch tài chính theo kiến trúc **đa tầng, đa phương thức (multimodal)**, kết hợp:

- **Rule Engine** — lọc nhanh các trường hợp gian lận "dễ", suy ra trực tiếp từ luật rõ ràng (thẻ hết hạn, thẻ bị khóa/mất, vượt hạn mức).
- **Isolation Forest** — phát hiện bất thường không giám sát, làm giàu đặc trưng đầu vào cho tầng học có giám sát.
- **Graph Analytics** — trích xuất đặc trưng từ mạng lưới quan hệ khách hàng–merchant (degree, PageRank, clustering coefficient, community detection).
- **Velocity Feature** — đặc trưng hành vi theo thời gian (tần suất giao dịch, impossible travel, z-score bất thường theo cửa sổ thời gian).
- **XGBoost** — mô hình phân loại chính, học trên toàn bộ tập đặc trưng tổng hợp.
- **SHAP** — giải thích mô hình theo từng nhóm phương thức (tabular / velocity / graph), phục vụ hiển thị trực quan cho người dùng cuối.

Mục tiêu là xây dựng một ứng dụng cho phép xem trạng thái tài khoản, tra cứu từng giao dịch (gian lận hay không), và biết được phương thức nào đã phát hiện ra gian lận đó.

## 2. Bộ dữ liệu

Link: https://www.kaggle.com/datasets/jatinkhandelwal112/indian-financial-fraud-dataset
Dữ liệu quan hệ gồm 4 bảng, tổng cộng 250,000 giao dịch:

| Bảng | Mô tả |
|---|---|
| `Transaction_Data_250k.csv` | Giao dịch — bảng trung tâm |
| `Cards_Data.csv` | Thông tin thẻ |
| `Customer_data.csv` | Thông tin khách hàng |
| `merchant_table.csv` | Thông tin merchant |

Tỷ lệ gian lận: **5.39%** (13,473 / 250,000 giao dịch).

## 3. Kiến trúc pipeline

```
Dữ liệu thô
   │
   ▼
Tầng 1: Rule Engine (bắt gian lận "dễ")
   │
   ▼
Tầng 2: ML Pipeline
   ├─ Feature Engineering (Tabular + Velocity + Graph)
   ├─ Isolation Forest (anomaly score)
   ├─ XGBoost (phân loại)
   └─ SHAP (giải thích)
   │
   ▼
Ứng dụng hiển thị (dashboard / tra cứu giao dịch)
```

## 4. Nhóm thực hiện

| Họ và tên | Vai trò | Liên hệ |
|---|---|---|
| Bui Tuan Hai  | AI Engineer | bhai.130720@gmail.com |
| Dang Hoang Hai | Data Engineer | danghoanghai1404@gmail.com |
| Hoang Quang Minh | Research + Backend/Frontend | hoangquangminh241105@gmail.com |

## 5. Kết quả

### 5.1. Hiệu năng mô hình

| Chỉ số | Rule Engine (Easy Fraud) | ML Pipeline (Hard Fraud) |
|---|---|---|
| AUROC |  |  |
| AUPRC |  |  |
| Precision |  |  |
| Recall |  |  |
| F1-score |  |  |

### 5.2. Ablation study (đóng góp từng nhánh)

| Cấu hình | AUPRC |
|---|---|
| Chỉ Tabular |  |
| Tabular + Velocity |  |
| Tabular + Velocity + Graph |  |

### 5.3. Ghi chú / nhận xét

_(để trống)_

## 6. Cấu trúc thư mục

```
├── indian_fraud_data/
├── notebooks/
├── src/
├── app/
└── README.md
```

## 7. Tài liệu tham khảo

[1] D. Cheng, Y. Zou, S. Xiang, and C. Jiang, "Graph neural networks for financial 
    fraud detection: A review," arXiv:2411.05815, Nov. 2024. [Online]. Available: 
    https://arxiv.org/abs/2411.05815

[2] F. Moradi, M. Tarif, and M. Homaei, "Semi-supervised supply chain fraud detection 
    with unsupervised pre-filtering," arXiv:2508.06574, Aug. 2025. [Online]. Available: 
    https://arxiv.org/abs/2508.06574