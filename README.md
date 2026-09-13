# Hybrid Financial Fraud Detection System
### A Two-Stage Fraud Detection Framework Combining Rule-Based Screening, Behavioral Modeling, Graph Analytics, Anomaly Detection, and Explainable Machine Learning

## 1. Giới thiệu

Đề tài xây dựng một hệ thống phát hiện gian lận giao dịch tài chính theo kiến trúc **đa tầng và đa phương thức (multimodal)**. Hệ thống kết hợp luật nghiệp vụ, đặc trưng hành vi theo thời gian, thông tin quan hệ dưới dạng đồ thị và mô hình học máy nhằm phân biệt các trường hợp gian lận dễ xác định với các trường hợp gian lận có biểu hiện phức tạp hơn.

Kiến trúc gồm hai tầng chính:

- **Tầng 1 — Rule Engine:** Xử lý nhanh các trường hợp gian lận có thể xác định trực tiếp bằng các luật nghiệp vụ rõ ràng, chẳng hạn thẻ hết hạn, thẻ bị khóa/mất hoặc giao dịch vượt hạn mức.

- **Tầng 2 — ML Pipeline:** Tập trung phát hiện các trường hợp gian lận còn lại dựa trên các mẫu hành vi và mối quan hệ phức tạp. Tầng này gồm:

  - **Isolation Forest:** Mô hình phát hiện bất thường không giám sát, được sử dụng để tạo điểm bất thường (`if_anomaly_score`) và bổ sung thông tin cho quá trình phân loại.

  - **Graph Analytics & GraphSAGE:** Khai thác cấu trúc quan hệ giữa khách hàng, thẻ và merchant, sử dụng các đặc trưng như node degree và GraphSAGE embeddings.

  - **Velocity Features:** Mô tả hành vi giao dịch theo thời gian, bao gồm tần suất giao dịch, các đặc trưng theo cửa sổ thời gian, impossible travel và độ lệch số tiền giao dịch.

  - **XGBoost:** Mô hình phân loại chính, được huấn luyện trên tập đặc trưng tổng hợp.

  - **SHAP:** Giải thích đóng góp của các đặc trưng vào quyết định của mô hình, hỗ trợ phân tích các nhóm đặc trưng như tabular, velocity và graph.

---

## 2. Bộ dữ liệu

- **Nguồn dữ liệu:** [Indian Financial Fraud Dataset (Kaggle)](https://www.kaggle.com/datasets/jatinkhandelwal112/indian-financial-fraud-dataset)

- **Quy mô:** Bộ dữ liệu gồm 4 bảng quan hệ với tổng cộng **250.000 giao dịch**.

| Bảng | Số dòng | Mô tả | Khóa nối |
|---|---:|---|---|
| `Transaction_Data_250k.csv` | 250.000 | Bảng trung tâm, mỗi dòng tương ứng với một giao dịch | `Transaction_ID` |
| `Cards_Data.csv` | 32.458 | Thông tin thẻ, bao gồm loại thẻ, hạn mức và trạng thái | `Card_ID` |
| `Customer_data.csv` | 25.001 | Thông tin khách hàng, bao gồm nhân khẩu học và phân khúc | `Customer_ID` |
| `merchant_table.csv` | 501 | Thông tin merchant, bao gồm ngành hàng và mức rủi ro | `Merchant_ID` |

### Phân bố nhãn gian lận

- **Tổng số giao dịch gian lận:** 13.473 / 250.000 giao dịch (**5,39%**).

- **Tầng Rule Engine (Easy Fraud):** Phát hiện **6.885 giao dịch gian lận rõ ràng**, gồm:
  - 2.470 giao dịch liên quan đến thẻ hết hạn (`Expired Card`);
  - 1.976 giao dịch vượt hạn mức (`Over Credit Limit`);
  - 1.591 giao dịch liên quan đến thẻ bị khóa (`Blocked Card`);
  - 848 giao dịch liên quan đến thẻ bị mất (`Lost Card`).

- **Tầng ML Pipeline (Hard Fraud):** Xử lý **243.115 giao dịch còn lại**.

---

## 3. Kiến trúc Pipeline

```text
Dữ liệu thô
250.000 giao dịch
   │
   ▼
Tầng 1: Rule Engine
   │
   ├─ Phát hiện 6.885 trường hợp Easy Fraud
   │  (Expired Card, Blocked Card, Lost Card, Over Credit Limit)
   │
   ▼
Tầng 2: ML Pipeline
   │
   ├─ Feature Engineering
   │  ├─ Tabular Features
   │  ├─ Velocity Features
   │  └─ Graph Analytics / GraphSAGE
   │
   ├─ Isolation Forest
   │  └─ Sinh if_anomaly_score
   │
   ├─ Resampling
   │  └─ RandomUnderSampler (tỷ lệ 1:2)
   │
   ├─ XGBoost Classifier
   │  └─ Đánh giá trên tập Test theo Time-Based Split
   │
   └─ SHAP
      └─ Phân tích đóng góp của các nhóm đặc trưng
```

---

## 4. Nhóm thực hiện

| Họ và tên | Vai trò | Liên hệ |
|---|---|---|
| Bui Tuan Hai | AI Engineer | bhai.13072020@gmail.com |
| Dang Hoang Hai | Data Engineer | danghoanghai1404@gmail.com |
| Hoang Quang Minh | Research | hoangquangminh241105@gmail.com |

---

## 5. Kết quả thực nghiệm

### 5.1. Hiệu năng mô hình

- **Tầng 1 (Rule Engine):** Đạt độ chính xác **100%** trên tập luật xác định (Easy Fraud).

- **Tầng 2 (ML Pipeline):** Được đánh giá độc lập trên tập **Test gồm 36.468 giao dịch**, sử dụng phương pháp chia dữ liệu theo thời gian (**Time-Based Split**).

| Chỉ số | Isolation Forest (Unsupervised Baseline) | XGBoost Final Pipeline (Hard Fraud) |
|---|---:|---:|
| **AUROC** | 0,8673 | **0,9499** |
| **AUPRC** | 0,3521 | **0,5514** |
| **Precision** | 0,2700 | **0,4287** |
| **Recall** | 0,5200 | **0,5948** |
| **F1-score** | 0,3600 | **0,4982** |

#### Confusion Matrix của XGBoost

Kết quả trên tập Test được tính với **ngưỡng tối ưu theo F1 là 0,8150**:

- **True Negative (TN):** 34.548
- **False Positive (FP):** 849
- **False Negative (FN):** 434
- **True Positive (TP):** 637

> Lưu ý: các giá trị trên được giữ nguyên theo kết quả thực nghiệm hiện tại của dự án.

---

### 5.2. Ablation Study

Ablation Study được sử dụng để đánh giá mức đóng góp của từng nhóm đặc trưng đối với hiệu năng dự đoán, sử dụng **AUPRC** và **ROC-AUC** trên tập kiểm thử.

| Cấu hình đặc trưng | Số lượng đặc trưng | Test AUPRC | Test ROC-AUC |
|---|---:|---:|---:|
| **(a) Chỉ Tabular** | 47 | 0,2214 | 0,7451 |
| **(b) Tabular + Velocity** | 106 | 0,5398 | 0,9471 |
| **(c) Tabular + Velocity + Graph** | 106 | 0,5398 | 0,9471 |
| **(d) Tabular + Velocity + Graph + `if_anomaly_score`** | 107 | 0,5282 | 0,9457 |
| **(e) Full Features + RandomUnderSampler (1:2)** | 107 | **0,5514** | **0,9499** |

**Nhận xét:** Trong kết quả hiện tại, việc bổ sung Graph vào cấu hình Tabular + Velocity chưa tạo ra sự thay đổi về số lượng đặc trưng hoặc các chỉ số đánh giá. Vì vậy, chưa nên kết luận rằng GraphSAGE đã tạo ra mức cải thiện độc lập cho mô hình. Đây là điểm cần tiếp tục kiểm tra trong quá trình phát triển.

---

### 5.3. So sánh các chiến lược cân bằng dữ liệu

Các chiến lược resampling được đánh giá trên mô hình XGBoost với tập đặc trưng đầy đủ.

| Chiến lược | Tỷ lệ Train (c0 : c1) | Threshold (Val F1) | AUPRC | ROC-AUC | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **(a) Không resample (Baseline)** | 164.942 : 5.238 | 0,2727 | 0,5282 | 0,9457 | 0,4199 | 0,5528 | 0,4772 |
| **(b) RandomUnderSampler (1:2)** | 10.476 : 5.238 | **0,8150** | **0,5514** | **0,9499** | 0,4287 | **0,5948** | **0,4982** |
| **(c) SMOTE (1:4)** | 164.942 : 41.235 | 0,6681 | 0,4953 | 0,9437 | 0,4319 | 0,5416 | 0,4805 |
| **(d) SMOTE (1:2)** | 164.942 : 82.471 | 0,8071 | 0,4919 | 0,9429 | 0,4322 | 0,5182 | 0,4713 |
| **(e) SMOTE + Tomek-links** | 164.915 : 41.208 | 0,6870 | 0,5004 | 0,9423 | **0,4549** | 0,5089 | 0,4804 |

---

## 5.4. Nhận xét và đánh giá

### 1. Vai trò nổi bật của đặc trưng Velocity

Khi bổ sung nhóm Velocity vào tập Tabular:

- **AUPRC tăng từ 0,2214 lên 0,5398**, tương đương mức tăng khoảng **143,8%**.
- **ROC-AUC tăng từ 0,7451 lên 0,9471**.

Kết quả cho thấy các đặc trưng mô tả hành vi giao dịch theo thời gian có đóng góp lớn trong bài toán này. Các tín hiệu quan trọng bao gồm tần suất giao dịch trong các cửa sổ thời gian, độ lệch số tiền và các đặc trưng liên quan đến hành vi địa lý.

### 2. Hiệu quả của RandomUnderSampler

Trong các chiến lược được thử nghiệm, **RandomUnderSampler với tỷ lệ 1:2** đạt:

- AUPRC cao nhất: **0,5514**;
- ROC-AUC cao nhất: **0,9499**;
- Recall cao nhất trong các cấu hình được liệt kê: **0,5948**;
- F1-score cao nhất: **0,4982**.

Trong khi đó, các cấu hình sử dụng SMOTE có AUPRC thấp hơn trong thí nghiệm hiện tại.

Do đó, với bộ dữ liệu và thiết lập thực nghiệm hiện tại, RandomUnderSampler 1:2 là chiến lược cân bằng dữ liệu tốt nhất trong số các phương pháp đã thử nghiệm.

### 3. Khả năng phân tách của XGBoost

Theo phân tích score trên tập Test, tỷ lệ giao dịch gian lận thực bị dự đoán với score `< 0,01` là **0,0%**, trong khi tỷ lệ có score `< 0,05` là **0,45%**.

Kết quả này cho thấy mô hình có khả năng phân biệt tốt trong vùng score thấp. Tuy nhiên, các chỉ số Precision, Recall và F1 vẫn cho thấy mô hình còn dư địa để cải thiện ở ngưỡng ra quyết định cuối cùng.

### 4. Giải thích mô hình bằng SHAP

Theo phân tích SHAP, các đặc trưng có ảnh hưởng đáng kể đến quyết định của mô hình gồm:

- `amount_deviation_category`: Độ lệch số tiền giao dịch so với mức thông thường theo danh mục.
- `amount_deviation_merchant`: Độ lệch số tiền giao dịch so với lịch sử tại merchant.
- `declined_unique_cards_1h`: Số lượng thẻ duy nhất bị từ chối trong một giờ gần nhất.
- `if_anomaly_score`: Điểm bất thường được sinh bởi Isolation Forest.
- `Is_International`: Chỉ báo giao dịch quốc tế.
- `merchant_graphsage_emb_*`: Các vector embedding của merchant được trích xuất bằng GraphSAGE.

SHAP giúp chuyển kết quả dự đoán của mô hình từ một score khó diễn giải thành các nhóm tín hiệu có thể được phân tích và trình bày trực quan.

---

## 6. Cấu trúc thư mục

```text
indian_fraud_data/
├── *.csv                         # 4 bảng dữ liệu thô

notebooks/
├── 01_eda_and_rule_engine.ipynb
├── 02_feature_engineering_velocity_graph.ipynb
└── 03_model_training_and_shap_evaluation.ipynb

src/
├── rules/                        # Tầng 1: Rule Engine
├── features/                     # Tabular, Velocity và Graph Features
├── models/                       # Isolation Forest và XGBoost Pipeline
└── explainability/               # Phân tích SHAP

app/                              # Web Dashboard
README.md
```

---

## 7. Tài liệu tham khảo

[1] D. Cheng, Y. Zou, S. Xiang, and C. Jiang, "Graph neural networks for financial fraud detection: A review," *arXiv preprint arXiv:2411.05815*, Nov. 2024. [Online]. Available: https://arxiv.org/abs/2411.05815

[2] F. Moradi, M. Tarif, and M. Homaei, "Semi-supervised supply chain fraud detection with unsupervised pre-filtering," *arXiv preprint arXiv:2508.06574*, Aug. 2025. [Online]. Available: https://arxiv.org/abs/2508.06574

[3] C. Chen, C. Lee, Y. Huang, and C. Peng, "Credit Card Fraud Detection via Intelligent Sampling and Self-supervised Learning," *ACM Transactions on Intelligent Systems and Technology*, vol. 15, no. 4, pp. 1–24, Aug. 2024. [Online]. Available: https://doi.org/10.1145/3653707

[4] M. Anwar, A. DeCunha, and S. Raza, "Zombie Cards Back Online: Reviving Expired Credit Cards for Contactless Payments," in *Proc. 35th USENIX Security Symposium (USENIX Security 26)*, Baltimore, MD, USA, Aug. 2026. [Online]. Available: https://www.usenix.org/conference/usenixsecurity26

[5] A. C. Bahnsen, D. Aouada, A. Stojanovic, and B. Ottersten, "Feature engineering strategies for credit card fraud detection," *Expert Systems with Applications*, vol. 51, pp. 134–142, May 2016. [Online]. Available: https://doi.org/10.1016/j.eswa.2015.12.030

[6] Y. Lucas, P. E. Portier, L. Laporte, L. He-Guelton, O. Caelen, M. Granitzer, and S. Calabretto, "Towards automated feature engineering for credit card fraud detection using sequential behavioral patterns," *Future Generation Computer Systems*, vol. 102, pp. 381–392, Jan. 2020. [Online]. Available: https://doi.org/10.1016/j.future.2019.09.005

[7] V. Van Vlasselaer, T. Eliassi-Rad, L. Akoglu, M. Snoeck, and B. Baesens, "APATE: A novel approach for automated credit card fraud detection using network-based extensions," *Decision Support Systems*, vol. 95, pp. 97–107, Mar. 2017. [Online]. Available: https://doi.org/10.1016/j.dss.2017.01.002

[8] M. Jullum, A. Løland, R. B. Huseby, G. Ånsjøen, and M. Lorentzen, "Data engineering for fraud detection," *Decision Support Systems*, vol. 143, p. 113492, Apr. 2021. [Online]. Available: https://doi.org/10.1016/j.dss.2021.113492

[9] X. Zhang, Y. Liu, X. Zhou, and H. Wang, "HOBA: History-Observed Behavioral Aggregation for financial fraud detection," *Information Sciences*, vol. 565, pp. 294–308, Jul. 2021. [Online]. Available: https://doi.org/10.1016/j.ins.2021.03.003

[10] M. Tayebi and S. El Kafhali, "A novel approach based on XGBoost classifier and Bayesian optimization for credit card fraud detection," *Cyber Security and Applications*, vol. 3, p. 100093, 2025. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2772918425000104


