# Ensemble Learning

Project học máy về **Ensemble Learning**, thực nghiệm phân loại khối u lành
tính/ác tính trên bộ dữ liệu **Breast Cancer Wisconsin (Diagnostic)** từ UCI
Machine Learning Repository. Mã nguồn chạy được dưới dạng chương trình Python
hoặc notebook.

> Project phục vụ mục đích học tập và nghiên cứu. Kết quả mô hình không phải
> chẩn đoán y khoa, không thay thế chuyên gia và không được dùng để quyết định
> điều trị.

## Cấu trúc thư mục

Cây dưới đây ghi lại đầy đủ các thư mục và tệp của project; các tệp cache sinh
tự động như `__pycache__` và `.pyc` được bỏ qua.

```text
Ensemble-learning/                         # Thư mục gốc của project
├── data/                                  # Dữ liệu đầu vào và dữ liệu đã xử lý
│   ├── processed/                         # Nơi lưu các tập train/test sau khi chia và mã hóa nhãn
│   │   ├── .gitkeep                       # Giữ thư mục khi chưa chạy pipeline
│   │   ├── wdbc_test.csv                  # Tập test; ID giữ để truy vết, không dùng làm feature
│   │   └── wdbc_train.csv                 # Tập train; Diagnosis được mã hóa B=0, M=1
│   └── raw/                               # Dữ liệu nguồn chưa qua xử lý
│       ├── .gitkeep                       # Giữ thư mục nguồn kể cả khi chưa có CSV
│       └── wdbc.csv                       # Dataset Breast Cancer Wisconsin tải từ UCI
├── docs/                                  # Tài liệu trình bày project
│   ├── GOI_B_QUY_TRINH_VA_VI_DU_TINH_TOAN.md # Chuyên đề quy trình thuật toán & tính toán số học AdaBoost
│   ├── KE_HOACH_VA_PHAN_CONG.md           # Kế hoạch thực hiện 7 ngày & phân công vai trò nhóm
│   ├── reports/                           # Báo cáo dạng có thể chỉnh sửa và bản xuất PDF
│   │   ├── MSSV1_MSSV2_MSSV3.docx         # Báo cáo project dạng Word
│   │   └── MSSV1_MSSV2_MSSV3.pdf          # Bản PDF của báo cáo project
│   └── slides/                            # Slide thuyết trình
│       ├── presentation.pdf               # Slide ở định dạng PDF để xem/in
│       └── presentation.pptx              # Slide PowerPoint có thể chỉnh sửa
├── notebooks/                             # Notebook trình bày và chạy thí nghiệm
│   ├── 01_step_by_step_example.ipynb      # Ví dụ từng bước: dữ liệu, baseline, hard voting
│   ├── 02_application_experiments.ipynb   # Chạy pipeline và xem bảng/biểu đồ kết quả
│   └── 03_adaboost_toy_calculation.ipynb  # Tính toán số học AdaBoost từng bước và trực quan hóa trọng số (Gói B)
├── results/                               # Số liệu và sản phẩm sinh ra từ thực nghiệm
│   ├── figures/                           # Các hình trực quan hóa kết quả
│   │   ├── .gitkeep                       # Giữ thư mục hình khi chưa có ảnh kết quả
│   │   ├── confusion_matrices.png         # Ma trận nhầm lẫn của các mô hình
│   │   ├── feature_importance.png         # Biểu đồ độ quan trọng đặc trưng
│   │   ├── model_comparison.png           # Biểu đồ so sánh các chỉ số mô hình
│   │   └── roc_curves.png                 # Đường ROC của các mô hình
│   ├── experiment_notes.md                # Nhận xét được tạo từ kết quả lần chạy gần nhất
│   ├── feature_importance.csv             # Số liệu độ quan trọng từng đặc trưng
│   ├── metrics.csv                         # Bảng metric và thời gian fit dạng CSV
│   ├── metrics.json                        # Metric cùng thông tin dataset/cấu hình chạy
│   └── misclassified_random_forest.csv     # ID, nhãn thật và nhãn đoán sai của Random Forest
├── src/                                   # Các module Python của pipeline
│   ├── __init__.py                        # Đánh dấu src là package Python
│   ├── evaluate.py                        # Huấn luyện, tính metric, vẽ hình và ghi kết quả
│   ├── model.py                           # Khai báo base models, ensemble và hàm hard voting
│   ├── preprocess.py                      # Tải, kiểm tra và chuẩn bị dataset
│   └── utils.py                           # Hàm tiện ích in thông tin dữ liệu và metric
├── tests/                                 # Kiểm thử tự động cho các thành phần chính
│   └── test_pipeline.py                   # Kiểm tra tiền xử lý và quy tắc hard voting
├── .gitignore                             # Quy định tệp/thư mục không theo dõi trong Git
├── main.py                                # Điểm vào chạy toàn bộ pipeline thực nghiệm
├── README.md                              # Tài liệu giới thiệu, cài đặt, chạy và cấu trúc project
└── requirements.txt                       # Danh sách thư viện Python cần cài
```

### Vai trò các thành phần

- `main.py`: điểm vào của chương trình; điều phối đọc dữ liệu, chia tập, chạy
  mô hình và xuất kết quả.
- `src/preprocess.py`: tải/đọc dữ liệu UCI, kiểm tra các cột và chuẩn bị đặc
  trưng, nhãn, mã mẫu.
- `src/model.py`: khai báo các mô hình và hàm hard voting minh họa.
- `src/evaluate.py`: huấn luyện, tính metric, tạo biểu đồ và ghi nhận xét/lỗi
  dự đoán.
- `src/utils.py`: in thông tin tổng quan dữ liệu và bảng metric.
- `notebooks/`: notebook giải thích từng bước và notebook chạy/hiển thị thực
  nghiệm.
- `data/raw/`: dataset đầu vào; `data/processed/` chứa các CSV train/test sau
  khi chia stratified và mã hóa nhãn. Cột `ID` được giữ để truy vết mẫu, nhưng
  luôn bị loại khỏi feature đầu vào.
- `results/`: bảng số liệu, nhận xét, các mẫu Random Forest dự đoán sai và biểu
  đồ do chương trình tạo.
- `docs/reports/`, `docs/slides/`: tài liệu báo cáo và slide của project.
- `tests/`: kiểm thử tiền xử lý và hàm hard voting.
- `.gitignore`: bỏ qua cache, môi trường ảo và các tệp dữ liệu không cần theo
  dõi; giữ `data/raw/wdbc.csv` trong project.

## Dataset và bài toán

- Nguồn: [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- Dataset có **569 mẫu**, **30 đặc trưng số** và cột mục tiêu `Diagnosis`.
  Các đặc trưng mô tả nhân tế bào được trích xuất từ ảnh chọc hút kim nhỏ (FNA).
- Theo metadata UCI, dataset không có giá trị thiếu. Chương trình vẫn đặt
  `SimpleImputer` trong pipeline để xử lý các giá trị thiếu nếu file CSV được
  thay thế hoặc cập nhật.
- `ID` là mã định danh mẫu, không được dùng làm đặc trưng đầu vào.
- Nhãn được mã hóa `B` (benign/lành tính) thành `0` và `M` (malignant/ác tính)
  thành `1`. `M` là lớp dương tính khi tính Precision, Recall, F1 và ROC-AUC.

## Tiền xử lý và mô hình

Mặc định, dữ liệu được chia thành train/test theo tỷ lệ 80/20, dùng stratify
để giữ phân bố nhãn và `random_state=42` để tái lập lần chạy. Imputation và
chuẩn hóa được fit bên trong pipeline từ tập train, tránh học thông tin từ tập
test. Logistic Regression, k-NN và SVM dùng `StandardScaler`.

Sau bước chia tập, chương trình lưu `data/processed/wdbc_train.csv` và
`data/processed/wdbc_test.csv`. Hai tệp này chứa ID để đối chiếu, 30 cột đặc
trưng và `Diagnosis` đã mã hóa thành 0/1. Chúng là các tập dữ liệu đã tách và
mã hóa nhãn, chưa chứa giá trị được impute hoặc được chuẩn hóa: các bước đó
được fit riêng bên trong từng mô hình bằng train data để tránh rò rỉ từ test
và cho phép mỗi loại mô hình dùng cách tiền xử lý phù hợp.

Các mô hình được so sánh:

- **Baseline:** Decision Tree, Logistic Regression, k-NN và SVM với kernel RBF.
- **Ensemble:** Random Forest, AdaBoost, Gradient Boosting, Soft Voting và
  Stacking (5-fold cross-validation nội bộ).
- **Minh họa thuật toán:** hard voting tự cài đặt bằng hàm `majority_vote` trong
  `src/model.py`. Khi số phiếu hòa, hàm chọn nhãn nhỏ hơn; với cách mã hóa hiện
  tại, điều này tương ứng chọn `B=0`.

Chương trình tính Accuracy, Precision, Recall, F1, ROC-AUC và thời gian fit.
ROC-AUC của hard voting sử dụng số phiếu làm điểm xếp hạng, nên chỉ có một số
mức điểm rời rạc.

## Cài đặt

Yêu cầu Python **3.10 trở lên**. Từ thư mục `Ensemble-learning`, cài các gói
được khai báo trong `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Chạy thực nghiệm

```bash
python main.py
```

Khi `data/raw/wdbc.csv` chưa tồn tại, chương trình tải CSV từ UCI vào vị trí
đó. Các lần chạy tiếp theo dùng lại file đã tải. Nếu không có kết nối mạng, hãy
tải CSV từ trang UCI ở trên và lưu thành `data/raw/wdbc.csv` trước khi chạy.

Có thể tùy chỉnh đường dẫn dữ liệu, thư mục kết quả, tỷ lệ test và seed:

```bash
python main.py --data-path data/raw/wdbc.csv --output-dir results --test-size 0.2 --random-state 42
```

`--test-size` phải nằm trong khoảng `(0, 1)`. Lỗi tải hoặc dữ liệu không đúng
định dạng sẽ được thông báo; chương trình không tạo dữ liệu thay thế.

## Chạy notebook

Mở notebook bằng Jupyter sau khi cài dependencies:

```bash
python -m jupyter lab
```

- `notebooks/01_step_by_step_example.ipynb`: xem thông tin dữ liệu, chia tập,
  huấn luyện baseline và thử hard voting trên ví dụ nhỏ.
- `notebooks/02_application_experiments.ipynb`: chạy pipeline đầy đủ, hiển thị
  metric, ROC curve, confusion matrix, feature importance và các trường hợp
  Random Forest dự đoán sai.

Notebook thứ hai gọi `main.py` và đọc các kết quả trong `results/`.

## Kết quả đầu ra

Sau khi chạy chương trình, các tệp được tạo hoặc cập nhật như sau:

| Đường dẫn | Nội dung |
|---|---|
| `results/metrics.csv` | Metric và thời gian fit của từng mô hình |
| `results/metrics.json` | Metric kèm thông tin dataset, nhãn dương tính và cấu hình chia tập |
| `results/figures/model_comparison.png` | Biểu đồ so sánh Accuracy, Precision, Recall, F1 và ROC-AUC |
| `results/figures/roc_curves.png` | ROC curve của các mô hình |
| `results/figures/confusion_matrices.png` | Confusion matrix; hàng là nhãn thật, cột là nhãn dự đoán |
| `results/figures/feature_importance.png` | Feature importance của Random Forest và AdaBoost |
| `results/feature_importance.csv` | Giá trị feature importance để xem hoặc xử lý tiếp |
| `results/misclassified_random_forest.csv` | ID, nhãn thật và nhãn dự đoán của các mẫu Random Forest dự đoán sai |
| `results/experiment_notes.md` | Nhận xét được tạo từ kết quả của lần chạy gần nhất |
| `data/processed/wdbc_train.csv` | Tập train đã chia; nhãn mã hóa thành 0/1, ID chỉ để đối chiếu |
| `data/processed/wdbc_test.csv` | Tập test đã chia; nhãn mã hóa thành 0/1, ID chỉ để đối chiếu |

Confusion matrix dùng `B` cho benign và `M` cho malignant. Với bài toán phân
loại này, cần xem xét Recall của lớp malignant cùng Precision và các mẫu âm
tính giả, thay vì chỉ dựa vào Accuracy. Feature importance của cây phản ánh
cách mô hình sử dụng đặc trưng để chia nhánh; không chứng minh quan hệ nhân quả.

Các tệp kết quả có sẵn trong thư mục `results/` là kết quả của một lần chạy cụ
thể. Metric và thời gian có thể thay đổi nếu thay seed, dữ liệu, phiên bản thư
viện hoặc môi trường chạy. Một phép chia train/test không đủ để khẳng định mô
hình tổng quát hóa tốt hơn; đánh giá nghiên cứu nên dùng cross-validation độc
lập. Đây là thực nghiệm học tập, không phải công cụ chẩn đoán.

## Kiểm thử

Chạy các unit test từ thư mục project:

```bash
python -m unittest discover -s tests -v
```
