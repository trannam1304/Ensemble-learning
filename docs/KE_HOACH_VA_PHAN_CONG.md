# KẾ HOẠCH THỰC HIỆN VÀ PHÂN CÔNG TIỂU LUẬN GIỮA KỲ
**Môn học:** Nhập môn Học máy (Introduction to Machine Learning)  
**Đề tài:** **Ensemble Learning** *(Xác định theo quy tắc Modulo 8: (7 + 0 + 6) mod 8 = 5)*  
**Thời gian thực hiện:** 7 ngày (Từ ngày 20/10/2026 đến hết ngày 26/10/2026)  
**Hạn nộp chính thức của Khoa:** 23:59 ngày 30/10/2026 *(Nhóm hoàn thành sớm 4 ngày để có thời gian dự phòng)*  
**Quy cách đặt tên file nộp:** `52400217_52400230_52400186.zip`

---

## 👥 THÀNH VIÊN NHÓM & BẢNG ĐĂNG KÝ VAI TRÒ

| STT | Họ và tên | Mã số sinh viên | Chữ số cuối | Vai trò đăng ký (Gói A / B / C) | Trạng thái xác nhận |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **Trần Hữu Nam** | 52400217 | 7 | *[Chờ chọn]* | ⏳ Đang chờ |
| 2 | **Nguyễn Trọng Quý** | 52400230 | 0 | *[Chờ chọn]* | ⏳ Đang chờ |
| 3 | **Nguyễn Anh Hào** | 52400186 | 6 | *[Chờ chọn]* | ⏳ Đang chờ |

---

## 📊 MÔ TẢ & ĐÁNH GIÁ MỨC ĐỘ CÁC GÓI CÔNG VIỆC (WORK PACKAGES)

Nhóm chia khối lượng công việc thành **3 Gói nhiệm vụ độc lập** tương đương nhau về độ quan trọng. Mỗi thành viên tự đánh giá thế mạnh để đăng ký **1 gói chính**.

### 🔹 GÓI A: Lý Thuyết Nền Tảng & Tổng Hợp Định Dạng Báo Cáo
> **Phù hợp với:** Thành viên có khả năng đọc hiểu tài liệu tốt, viết văn học thuật mạch lạc, cẩn thận và tỉ mỉ trong việc định dạng văn bản (Word/PDF).

* **Điểm Rubric phụ trách trực tiếp:** **2.0 điểm** (Tiêu chí 1: Nguyên lý hoạt động) + Đảm bảo không bị trừ 10%–50% lỗi định dạng.
* **Đánh giá mức độ:**
  * **Độ khó chuyên môn:** ⭐⭐⭐ (3/5 - Trung bình)
  * **Khối lượng công việc:** ⭐⭐⭐⭐ (4/5 - Khá nhiều về cuối giai đoạn)
  * **Mức độ gấp gáp:** 🟡 Vừa phải ở 3 ngày đầu, 🔴 Rất gấp ở ngày 4 - 5 khi tổng hợp toàn bộ bài.
  * **Thời gian ước tính:** ~14 – 16 giờ làm việc.
* **Nhiệm vụ cụ thể:**
  1. Soạn thảo chi tiết cơ sở lý thuyết Ensemble Learning: Khái niệm, vì sao Ensemble hiệu quả (Condorcet’s Jury Theorem, Bias-Variance Tradeoff).
  2. Phân tích nguyên lý 3 nhánh chính: **Bagging** (Bootstrap Aggregating), **Boosting** (Tuần tự sửa lỗi), **Stacking** (Meta-learning).
  3. Quản lý file Báo cáo Word chuẩn: Thiết lập trang bìa, mục lục tự động, canh lề, font chữ, bảng biểu, danh mục hình ảnh, trích dẫn tài liệu tham khảo (IEEE/APA).
  4. Rà soát trùng lặp nội dung, kiểm tra đạo văn và xuất file PDF cuối cùng.

---

### 🔹 GÓI B: Quy Trình Thuật Toán, Ví Dụ Tính Toán Số Học & Thiết Kế Slide
> **Phù hợp với:** Thành viên có tư duy toán học/logic tốt, cẩn thận tính toán số học từng bước, có thẩm mỹ tốt về thiết kế bài thuyết trình trực quan.

* **Điểm Rubric phụ trách trực tiếp:** **2.0 điểm** (Tiêu chí 2: Quy trình và ví dụ minh họa) + Hỗ trợ điểm Trình bày (Tiêu chí 5).
* **Đánh giá mức độ:**
  * **Độ khó chuyên môn:** ⭐⭐⭐⭐ (4/5 - Khá khó: Cần hiểu sâu bản chất để giải tay không sai số)
  * **Khối lượng công việc:** ⭐⭐⭐ (3/5 - Vừa phải)
  * **Mức độ gấp gáp:** 🔴 Cần hoàn thành sớm (trong ngày 2 - 3) để đưa vào báo cáo và slide.
  * **Thời gian ước tính:** ~13 – 15 giờ làm việc.
* **Nhiệm vụ cụ thể:**
  1. Trình bày quy trình chi tiết từng bước (Algorithm steps / Flowchart) của thuật toán tiêu biểu (khuyến nghị: **AdaBoost** hoặc **Random Forest**).
  2. **Tự tạo 1 Toy Dataset nhỏ (5–8 dòng dữ liệu số cụ thể)**:
     - Tính toán thủ công từng bước: Tính Gini/Entropy (với Decision Tree) hoặc cập nhật trọng số mẫu $w_i$, tính sai số $\epsilon_t$, tính trọng số mô hình $\alpha_t$ (với AdaBoost).
     - Trình bày dạng bảng so sánh qua 2–3 vòng lặp (iteration) để minh họa rõ bản chất hoạt động.
  3. Thiết kế toàn bộ Slide báo cáo (15–20 slide) trên PowerPoint / Canva: Bố cục khoa học, nhiều sơ đồ quy trình, trực quan, hạn chế text dài dòng.

---

### 🔹 GÓI C: Dữ Liệu, Lập Trình Python & Thực Nghiệm Mô Hình
> **Phù hợp với:** Thành viên thích code, thành thạo Python, thư viện khoa học dữ liệu (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).

* **Điểm Rubric phụ trách trực tiếp:** **4.0 điểm** (Tiêu chí 3: Bài toán ứng dụng [2đ] + Tiêu chí 4: Triển khai mã nguồn và kết quả [2đ]).
* **Đánh giá mức độ:**
  * **Độ khó chuyên môn:** ⭐⭐⭐⭐ (4/5 - Khá khó: Cần xử lý dữ liệu và so sánh nhiều mô hình)
  * **Khối lượng công việc:** ⭐⭐⭐⭐ (4/5 - Nặng ở 3 ngày đầu)
  * **Mức độ gấp gáp:** 🔴 Rất gấp ở ngày 1 - 3 (phải ra kết quả thực nghiệm thì nhóm mới có số liệu viết báo cáo).
  * **Thời gian ước tính:** ~15 – 18 giờ làm việc.
* **Nhiệm vụ cụ thể:**
  1. Chọn bài toán thực tế & dataset dạng bảng sạch (VD: *Heart Disease* hoặc *Customer Churn* từ Kaggle/UCI). Làm sạch và phân chia Train/Test.
  2. Triển khai code:
     - Viết mã nguồn minh họa bản chất (Simple Voting / Bagging hoặc AdaBoost cơ bản).
     - Áp dụng các mô hình thư viện hoàn chỉnh: Decision Tree (Base model), Random Forest, AdaBoost, Gradient Boosting / Stacking.
  3. Đo lường & Trực quan hóa kết quả: Vẽ biểu đồ so sánh Accuracy, Precision, Recall, F1-Score, ROC-AUC Curve, Confusion Matrix.
  4. Đóng gói code sạch đẹp (file `.ipynb` hoặc `.py`), có comment giải thích rõ ràng kèm file hướng dẫn chạy code (`requirements.txt`).

---

### 🔹 PHẦN CHUNG: Kỹ Năng Thuyết Trình & Phản Biện Nhóm (Bắt buộc cả 3 người)
* **Điểm Rubric:** **2.0 điểm** (Tiêu chí 5).
* **Quy định đề bài:** Tất cả thành viên đều phải tham gia thuyết trình.
* **Phân bổ lượt nói dự kiến:**
  - **Người làm Gói A:** Trình bày Phần Mở đầu & Cơ sở lý thuyết nền tảng (~3–4 phút).
  - **Người làm Gói B:** Trình bày Quy trình thuật toán & Giải thích ví dụ tính toán số học (~4–5 phút).
  - **Người làm Gói C:** Trình bày Bài toán ứng dụng, Demo chạy mã nguồn & Phân tích kết quả thực nghiệm (~4–5 phút).
  - **Cả 3 thành viên:** Cùng tham gia trả lời câu hỏi vấn đáp phản biện của Giảng viên (~5 phút).

---

## 🗓️ LỊCH TRÌNH CHI TIẾT 7 NGÀY (TIMELINE & DAILY CHECKLIST)

```
[Ngày 1: 20/10] ──> Khởi động, chọn gói công việc, chốt Dataset & Outline
[Ngày 2: 21/10] ──> Soạn Lý thuyết, giải ví dụ số học, code Baseline
[Ngày 3: 22/10] ──> Hoàn thiện Code Ensemble, chạy kết quả & xuất biểu đồ
[Ngày 4: 23/10] ──> Ghép toàn bộ nội dung vào Báo cáo Word bản nháp
[Ngày 5: 24/10] ──> Hoàn thiện Slide thuyết trình, chuẩn hóa format Báo cáo
[Ngày 6: 25/10] ──> Họp duyệt slide, luyện tập thuyết trình (Dry-run) & chuẩn bị Q&A
[Ngày 7: 26/10] ──> Kiểm tra chéo theo Rubric, đóng gói ZIP hoàn tất
```

### 📍 NGÀY 1 (Thứ Ba, 20/10/2026): Khởi động & Thống nhất đề cương
- [ ] **Cả nhóm**: Họp nhanh 30 phút, chính thức đăng ký Gói A, B, C vào bảng phân công.
- [ ] **Gói A**: Tạo file Google Docs/Word chung, thiết lập khung bài báo cáo (mục lục các chương mục theo rubric).
- [x] **Gói B**: Nghiên cứu thuật toán AdaBoost/Random Forest, chọn hướng xây dựng ví dụ số học *(Đã hoàn thành: Chọn AdaBoost với Decision Stumps, xem [GOI_B_QUY_TRINH_VA_VI_DU_TINH_TOAN.md](file:///d:/Jo/Hk1_2026-2027/Introduction%20to%20Machine%20Learning/Midterm/GOI_B_QUY_TRINH_VA_VI_DU_TINH_TOAN.md))*.
- [ ] **Gói C**: Tìm và tải dataset (Heart Disease/Customer Churn), xác nhận dữ liệu khả thi và chia sẻ link dữ liệu cho nhóm.

### 📍 NGÀY 2 (Thứ Tư, 21/10/2026): Lý thuyết, Tính toán số & Viết code nền tảng
- [ ] **Gói A**: Hoàn thành mục Lý thuyết Ensemble: Định nghĩa, Voting, Bagging, Boosting, Stacking; phân tích Bias-Variance Tradeoff.
- [x] **Gói B**: Xây dựng toy dataset (5–8 dòng) và tính toán thủ công từng bước qua 2–3 vòng lặp; lập bảng số liệu chi tiết *(Đã hoàn thành: Toy dataset 8 mẫu qua 3 vòng lặp, kiểm chứng 100% bằng script [verify_adaboost_calculation.py](file:///d:/Jo/Hk1_2026-2027/Introduction%20to%20Machine%20Learning/Midterm/verify_adaboost_calculation.py))*.
- [ ] **Gói C**: Tiền xử lý dữ liệu (xử lý null, encode, scale), huấn luyện các mô hình Baseline (Decision Tree, Logistic Regression).

### 📍 NGÀY 3 (Thứ Năm, 22/10/2026): Thực nghiệm chuyên sâu & Triển khai thuật toán
- [ ] **Gói C**: Triển khai các thuật toán Ensemble (Random Forest, AdaBoost, Stacking); đo đạc các chỉ số (Accuracy, F1, AUC), xuất biểu đồ và ma trận nhầm lẫn.
- [ ] **Gói B**: Hoàn thiện bài viết phần "Quy trình thực hiện và ví dụ minh họa" đưa vào file nháp.
- [ ] **Gói A**: Đưa phần lý thuyết đã hoàn chỉnh vào file báo cáo chính thức, viết lời mở đầu và phần giới thiệu tổng quan.

### 📍 NGÀY 4 (Thứ Sáu, 23/10/2026): Tổng hợp Báo cáo toàn diện
- [ ] **Gói C**: Cung cấp đoạn mã nguồn giải thích quan trọng và toàn bộ hình ảnh biểu đồ thực nghiệm cho Gói A; viết nhận xét thực nghiệm.
- [ ] **Gói B**: Bắt đầu dựng khung Slide thuyết trình (dàn ý các slide).
- [ ] **Gói A**: Ghép toàn bộ nội dung từ Gói B và Gói C vào file Báo cáo Word; kiểm tra tính liền mạch giữa các phần.

### 📍 NGÀY 5 (Thứ Bảy, 24/10/2026): Hoàn thiện Slide & Chuẩn hóa định dạng Báo cáo
- [ ] **Gói B**: Thiết kế hoàn chỉnh bộ Slide báo cáo (15–20 slide) đẹp mắt, trực quan hóa quy trình và kết quả.
- [ ] **Gói A**: Rà soát định dạng bài báo cáo theo quy định khoa (font chữ, lề, số trang, chú thích bảng/hình, tài liệu tham khảo), xuất file PDF.
- [ ] **Gói C**: Rà soát, dọn dẹp sạch mã nguồn Python, ghi chú chú thích chi tiết, kiểm tra file chạy độc lập.

### 📍 NGÀY 6 (Chủ Nhật, 25/10/2026): Tập dượt thuyết trình & Chuẩn bị phản biện
- [ ] **Cả nhóm**: Họp online/offline chạy thử slide (Dry-run):
  - Bấm giờ chính xác (12–15 phút tổng bài nói).
  - Khớp nối mượt mà giữa các phần trình bày của từng thành viên.
- [ ] **Cả nhóm**: Liệt kê 5–10 câu hỏi giảng viên có thể hỏi và phân chia người trả lời (VD: *Tại sao Random Forest giảm variance tốt hơn Decision Tree?*, *AdaBoost xử lý nhiễu thế nào?*).

### 📍 NGÀY 7 (Thứ Hai, 26/10/2026): Kiểm tra theo Rubric & Đóng gói sản phẩm
- [ ] **Cả nhóm**: Thực hiện tự đánh giá chéo theo **Bảng đối soát Rubric (ở phần bên dưới)** để đảm bảo đạt 10/10 điểm.
- [ ] **Đóng gói file nén**: Kiểm tra đầy đủ 4 thành phần file trước khi nộp.
- [ ] **Nộp bài chính thức**: Đại diện nhóm nộp lên hệ thống eLearning trước hạn 30/10/2026 và chụp ảnh màn hình xác nhận.

---

## ✅ BẢNG CHECKLIST KIỂM TRA NHIỆM VỤ TỪNG GÓI (TASK COMPLETION AUDIT)

Mỗi thành viên sau khi nhận gói công việc sẽ dùng checklist này để tự kiểm tra tiến độ của mình trước khi bàn giao cho nhóm.

### 📋 Checklist Cho Gói A: Lý Thuyết & Tổng Hợp Báo Cáo
*Người phụ trách: [...........................................]*
- [ ] **Khái niệm & Bản chất:** Đã nêu rõ định nghĩa Ensemble Learning và giải thích tại sao kết hợp nhiều mô hình yếu (Weak Learners) lại tạo ra mô hình mạnh (Strong Learner).
- [ ] **Định lý & Cơ sở lý thuyết:** Đã trình bày định lý Condorcet’s Jury Theorem (hoặc nguyên lý Diversity & Accuracy) để chứng minh tính hiệu quả bằng lý thuyết.
- [ ] **Phân tích Bias - Variance Tradeoff:** Đã giải thích rõ ràng tại sao Bagging giúp giảm Variance, Boosting giúp giảm Bias, và Stacking tối ưu hóa dự đoán.
- [ ] **Phân loại các phương pháp Ensemble chính:**
  - [ ] Phương pháp Voting / Averaging (Hard voting vs Soft voting).
  - [ ] Phương pháp Bagging (Bootstrap Aggregating & Random Forest).
  - [ ] Phương pháp Boosting (AdaBoost, Gradient Boosting, XGBoost).
  - [ ] Phương pháp Stacking (Base models & Meta-model).
- [ ] **Chuẩn hóa Báo cáo Word & PDF:**
  - [ ] Trang bìa đầy đủ thông tin: Trường, Khoa, Môn học, Tên đề tài, Giảng viên, Danh sách 3 thành viên kèm MSSV.
  - [ ] Mục lục tự động, Danh mục bảng biểu, Danh mục hình vẽ.
  - [ ] Font chữ chuẩn (Times New Roman 13 hoặc Arial 12), giãn dòng 1.3 - 1.5 lines, căn lề chuẩn (Left 3cm, Right 2cm, Top 2cm, Bottom 2cm).
  - [ ] Đánh số trang đầy đủ (không tính trang bìa, mục lục).
  - [ ] Đã có phần **Tài liệu tham khảo** trích dẫn theo chuẩn (IEEE/APA) ở cuối bài.
  - [ ] Đã rà soát văn phong, kiểm tra không bị lỗi chính tả và đảm bảo nội dung tự viết không đạo văn.
  - [ ] Đã xuất ra file PDF chất lượng cao, không bị nhảy trang hay lỗi font.

---

### 📋 Checklist Cho Gói B: Thuật Toán, Tính Toán Số Học & Slide
*Người phụ trách: [Gói B]*
- [x] **Sơ đồ quy trình (Flowchart):** Đã vẽ/trình bày sơ đồ khối trực quan thể hiện luồng hoạt động của thuật toán Ensemble được chọn (AdaBoost).
- [x] **Trình bày thuật toán dạng mã giả (Pseudocode) hoặc từng bước:** Nêu rõ Input, Output, điều kiện dừng, công thức cập nhật ở từng bước.
- [x] **Ví dụ tính toán số học bằng tay (Bắt buộc theo Rubric):**
  - [x] Đã tạo tập dữ liệu mẫu (Toy Dataset) cụ thể gồm 5–8 dòng dữ liệu với số liệu đơn giản, dễ theo dõi (8 mẫu).
  - [x] **Bước 1:** Khởi tạo trọng số mẫu ban đầu $w_i = \frac{1}{N} = 0.1250$.
  - [x] **Bước 2 (Vòng lặp 1):** Chọn mô hình yếu (Decision Stump), tính sai số $\epsilon_1 = 0.2500$, tính trọng số mô hình $\alpha_1 = 0.5493$, cập nhật trọng số mẫu $w_i$ cho vòng sau (mẫu sai tăng trọng số lên $0.2500$, mẫu đúng giảm xuống $0.0833$).
  - [x] **Bước 3 (Vòng lặp 2):** Lặp lại tính toán trên tập trọng số mới để chứng minh thuật toán tập trung vào mẫu khó (sửa đúng mẫu 6, 7; $\epsilon_2 = 0.2500$, $\alpha_2 = 0.5493$).
  - [x] **Bước 4 (Kết hợp):** Tính toán kết quả dự đoán cuối cùng qua cơ chế bỏ phiếu có trọng số (Weighted Voting), đạt độ chính xác 100%.
  - [x] Các bảng tính số liệu rõ ràng, không sai sót công thức hay nhầm lẫn số học (đã kiểm chứng bằng mã nguồn python).
- [ ] **Thiết kế Slide Báo Cáo:**
  - [ ] Số lượng slide phù hợp (15–20 slide), thời lượng trình bày 12–15 phút.
  - [ ] Slide mở đầu: Tên đề tài, danh sách thành viên, phân công nội dung trình bày.
  - [ ] Slide nội dung: Sử dụng icon, hình vẽ, biểu đồ; hạn chế tối đa các đoạn văn bản dài (bullet point súc tích).
  - [ ] Đưa ví dụ số học lên slide dưới dạng bảng so sánh trực quan, dễ nhìn khi thuyết trình.
  - [ ] Slide kết quả thực nghiệm: Đầy đủ biểu đồ so sánh mô hình, ma trận nhầm lẫn (Confusion Matrix).
  - [ ] Slide kết luận & Tài liệu tham khảo.

---

### 📋 Checklist Cho Gói C: Dữ Liệu, Lập Trình Python & Thực Nghiệm
*Người phụ trách: [...........................................]*
- [ ] **Dữ liệu thực tế:**
  - [ ] Đã chọn bài toán cụ thể (VD: Phân loại nguy cơ bệnh tim mạch / Phân loại khách hàng rời mạng).
  - [ ] Đã nêu rõ nguồn gốc dataset (Kaggle/UCI), số lượng mẫu (instances), số lượng đặc trưng (features), biến mục tiêu (target label).
  - [ ] Đã thực hiện tiền xử lý: Xử lý giá trị thiếu (Missing values), mã hóa dữ liệu phân loại (Encoding), chuẩn hóa dữ liệu (Scaling) và chia Train/Test hợp lý (ví dụ: 80/20 hoặc 70/30 có Stratified).
- [ ] **Mã nguồn (Source Code):**
  - [ ] Có đoạn code minh họa bản chất thuật toán (viết hàm tính toán đơn giản từ đầu - From scratch hoặc giải thích chi tiết luồng xử lý).
  - [ ] Sử dụng thư viện `scikit-learn` huấn luyện các Base Models (Decision Tree, Logistic Regression, k-NN/SVM).
  - [ ] Huấn luyện các mô hình Ensemble tương ứng (Random Forest, AdaBoost, Stacking / Voting).
  - [ ] Mã nguồn sạch, phân chia cell/function logic, có chú thích (comment) tiếng Việt giải thích từng bước.
- [ ] **Thực nghiệm & Đánh giá so sánh:**
  - [ ] Bảng tổng hợp các độ đo đánh giá: **Accuracy, Precision, Recall, F1-Score, ROC-AUC**.
  - [ ] Biểu đồ trực quan: So sánh các mô hình bằng Barplot, vẽ ROC Curve, vẽ Heatmap Confusion Matrix.
  - [ ] Biểu đồ Feature Importance (Độ quan trọng của các đặc trưng) trích xuất từ Random Forest / AdaBoost.
  - [ ] **Nhận xét thực nghiệm:** Giải thích tại sao mô hình Ensemble lại có kết quả vượt trội hơn các Base Model, phân tích trường hợp dự đoán sai và đánh giá thời gian huấn luyện (trade-off giữa độ chính xác và chi phí tính toán).
- [ ] **Đóng gói mã nguồn:**
  - [ ] File notebook (`.ipynb`) chạy mượt từ đầu đến cuối không báo lỗi (Restart Kernel & Run All Cells thành công).
  - [ ] Có file dữ liệu `.csv` kèm theo và file `requirements.txt` (nếu cần).

---

### 📋 Checklist Phần Chung: Kỹ Năng Thuyết Trình & Phản Biện (Cả 3 thành viên)
- [ ] **Phân chia thời gian:** Đảm bảo cả 3 thành viên đều có phần nói trực tiếp trên slide (khoảng 3–4 phút/thành viên).
- [ ] **Thuyết trình thử (Dry-run):** Đã tổ chức ít nhất 01 buổi họp luyện tập bấm giờ thực tế.
- [ ] **Chuyển tiếp (Handover):** Các thành viên phối hợp chuyển giao lời nói nhịp nhàng, liền mạch.
- [ ] **Chuẩn bị trả lời phản biện (Q&A):** Đã chuẩn bị sẵn câu trả lời cho các câu hỏi thường gặp:
  - [ ] *Vì sao Ensemble Learning có thể giảm Overfitting trong khi dùng nhiều cây quyết định?*
  - [ ] *Sự khác nhau căn bản giữa Bagging và Boosting là gì?*
  - [ ] *Khi nào nên dùng Hard Voting, khi nào nên dùng Soft Voting?*
  - [ ] *Dựa vào đâu để chọn mô hình Meta-learner trong Stacking?*
  - [ ] *Nếu một Base model bị lỗi/yếu quá mức thì ảnh hưởng thế nào đến cả hệ thống Ensemble?*

---

## 🎯 BẢNG TỰ ĐỐI SOÁT RUBRIC CHẤM ĐIỂM (RUBRIC SELF-AUDIT - MỤC TIÊU 10/10)

Bảng đối chiếu trực tiếp với **Rubric chấm điểm của Giảng viên (trang 3–4 trong file PDF)** để tự chấm điểm trước khi nộp:

| Tiêu chí | Trọng số | Yêu cầu đạt 100% điểm (2.0 điểm) | Bộ câu hỏi tự kiểm tra (Self-Audit Questions) | Đạt yêu cầu? |
| :--- | :---: | :--- | :--- | :---: |
| **1. Nguyên lý hoạt động** | **2.0 đ** | Giải thích chính xác, đầy đủ cơ sở lý thuyết, ý tưởng và nguyên lý hoạt động của thuật toán. | • Bài viết đã giải thích bản chất vì sao kết hợp mô hình yếu lại thành mô hình mạnh chưa?<br>• Đã phân tích đủ các nhóm Ensemble chính (Bagging, Boosting, Stacking) chưa?<br>• Có nêu và giải thích công thức toán học/lý thuyết cốt lõi (Bias-Variance tradeoff) chưa? | [ ] ĐẠT |
| **2. Quy trình & Ví dụ minh họa** | **2.0 đ** | Trình bày chính xác từng bước của thuật toán; minh họa đầy đủ quá trình thực hiện và tính toán thông qua ví dụ cụ thể. | • Có sơ đồ khối hoặc các bước thực hiện rõ ràng từ input đến output không?<br>• **Đã có ví dụ số học tính tay từng bước chưa?** (Có bảng số liệu mẫu, tính trọng số, tính sai số, chứng minh cách tính toán cụ thể qua từng vòng lặp). | [ ] ĐẠT |
| **3. Bài toán ứng dụng** | **2.0 đ** | Giới thiệu rõ bài toán, mục tiêu, dữ liệu đầu vào/đầu ra và cách áp dụng thuật toán để giải quyết bài toán. | • Bài toán được chọn có tính thực tế không?<br>• Đã mô tả rõ ràng mục tiêu kinh doanh/bài toán, từng thuộc tính đầu vào (Input features) và nhãn dự đoán (Output label) chưa?<br>• Đã giải thích lý do vì sao bài toán này phù hợp để áp dụng Ensemble Learning chưa? | [ ] ĐẠT |
| **4. Triển khai mã nguồn & Kết quả** | **2.0 đ** | Mã nguồn hoạt động đúng, giải thích rõ các thành phần chính, trình bày kết quả thực nghiệm và nhận xét phù hợp. | • Mã nguồn có chạy mượt mà, không phát sinh lỗi không?<br>• Code có comment giải thích rõ ràng từng khối lệnh không?<br>• Có bảng đối chiếu và biểu đồ so sánh giữa Base Model và Ensemble Model không?<br>• Đã có nhận xét, phân tích chuyên sâu về kết quả thực nghiệm chưa? | [ ] ĐẠT |
| **5. Kỹ năng thuyết trình & Phối hợp nhóm** | **2.0 đ** | Trình bày rõ ràng, mạch lạc, đúng thời gian; các thành viên phối hợp tốt, sử dụng phương tiện minh họa hiệu quả và trả lời tốt câu hỏi. | • Tất cả 3 thành viên đều tham gia thuyết trình chưa?<br>• Slide có đẹp mắt, trực quan, hỗ trợ tốt cho bài nói không?<br>• Cả nhóm đã diễn tập trước, đảm bảo nói đúng khung thời gian quy định chưa?<br>• Đã chuẩn bị tinh thần và kiến thức để trả lời các câu hỏi vấn đáp phản biện chưa? | [ ] ĐẠT |

---

## 📦 CHECKLIST KIỂM TRA ĐÓNG GÓI HỒ SƠ NỘP BÀI (SUBMISSION AUDIT)

Trước khi nhấn nút nộp bài lên hệ thống eLearning, kiểm tra kỹ các tiêu chí sau:

- [ ] **Định dạng file nộp:** Đã gom tất cả tài liệu vào 01 file nén định dạng `.zip` (hoặc `.rar`).
- [ ] **Tên file nén:** Đặt chính xác theo cú pháp MSSV của cả 3 thành viên:
  $$\mathbf{52400217\_52400230\_52400186.zip}$$
- [ ] **Các thành phần bắt buộc bên trong file nén:**
  - [ ] 01 file Báo cáo dạng Word: `BaoCao_52400217_52400230_52400186.docx`
  - [ ] 01 file Báo cáo dạng PDF: `BaoCao_52400217_52400230_52400186.pdf`
  - [ ] 01 file Slide thuyết trình: `Slide_52400217_52400230_52400186.pptx` (hoặc `.pdf`)
  - [ ] Thư mục `SourceCode/`: Chứa notebook Jupyter (`.ipynb`) hoặc file code `.py`, dataset `.csv`, và hướng dẫn chạy code.
- [ ] **Kiểm tra dung lượng file:** Đảm bảo file nén không vượt quá giới hạn dung lượng upload của hệ thống eLearning khoa.
- [ ] **Thời gian nộp bài:** Nộp trước 23:59 ngày 30/10/2026 (Tuyệt đối không để sát giờ tránh nghẽn mạng).

---

## ⚠️ CÁC NGUYÊN TẮC "SỐNG CÒN" CẦN NHỚ
1. **Tuyệt đối không đạo văn:** Tất cả tài liệu tham khảo phải được diễn đạt lại (paraphrase) theo cách hiểu của nhóm và có trích dẫn nguồn đầy đủ (đạo văn bị 0 điểm theo quy định khoa).
2. **Không nộp file cẩu thả:** Bài thi có thể bị trừ 10% đến 50% tổng điểm nếu vi phạm định dạng.
3. **Không dùng thư viện "đen xì" một cách thụ động:** Rubric nêu rõ: *"không chỉ giới thiệu lý thuyết hoặc sử dụng thư viện có sẵn"*. Vì vậy phần tính tay (Gói B) và giải thích luồng hoạt động thuật toán (Gói C) là chìa khóa quyết định điểm 9–10.
