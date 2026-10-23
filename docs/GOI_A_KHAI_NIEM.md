# Cơ sở lý thuyết Ensemble Learning



## 1. Cơ sở lý thuyết Ensemble Learning

### 1.1. Khái niệm

**Ensemble Learning (học kết hợp)** là phương pháp xây dựng nhiều mô hình học máy (gọi là *base learner* hay *weak learner*) rồi **kết hợp dự đoán của chúng** để tạo ra một mô hình cuối cùng mạnh hơn bất kỳ mô hình thành phần nào.

Ý tưởng cốt lõi giống "trí tuệ đám đông": một hội đồng nhiều chuyên gia thường quyết định tốt hơn một chuyên gia đơn lẻ. Trong bài toán chẩn đoán ung thư vú, thay vì tin vào một cây quyết định duy nhất, ta để hàng trăm cây (hoặc nhiều loại mô hình khác nhau) cùng "bỏ phiếu" cho kết luận *lành tính* hay *ác tính*.

### 1.2. Nguyên lý hoạt động

Một hệ ensemble gồm ba thành phần:

1. **Tạo ra các mô hình thành phần** từ cùng thuật toán hoặc nhiều thuật toán khác nhau, trên dữ liệu hoặc đặc trưng khác nhau.
2. **Đảm bảo tính đa dạng (diversity)** giữa các mô hình, để chúng mắc lỗi ở những chỗ khác nhau.
3. **Tổng hợp kết quả** bằng một quy tắc:
   - *Phân loại*: bỏ phiếu đa số (hard voting) hoặc trung bình xác suất (soft voting).
   - *Hồi quy*: trung bình (có thể có trọng số).
   - *Học có mô hình tổng hợp*: dùng một meta-model (Stacking).

**Hai điều kiện để ensemble hiệu quả:**

- Mỗi mô hình phải **tốt hơn đoán ngẫu nhiên** (accuracy > 50% với bài toán nhị phân).
- Các mô hình phải **đa dạng**, tức sai số không tương quan hoàn toàn với nhau.

**Vì sao ensemble hiệu quả? (Dietterich, 2000)** có ba lý do:

- *Thống kê*: dữ liệu hữu hạn nên nhiều giả thuyết cùng khớp dữ liệu. Lấy trung bình giảm rủi ro chọn nhầm một giả thuyết kém.
- *Tính toán*: thuật toán tối ưu có thể kẹt ở cực trị địa phương. Nhiều lần khởi tạo khác nhau rồi kết hợp sẽ xấp xỉ tốt hơn.
- *Biểu diễn*: hàm đích thực có thể nằm ngoài không gian giả thuyết của một mô hình đơn, nhưng tổ hợp nhiều mô hình mở rộng được không gian đó.

### 1.3. Định lý bồi thẩm đoàn Condorcet (Condorcet's Jury Theorem, 1785)

**Phát biểu.** Giả sử có *n* người bỏ phiếu (n lẻ) cho một quyết định đúng/sai, thỏa mãn:

- Mỗi người **độc lập** với nhau.
- Mỗi người có xác suất quyết định đúng là **p**, như nhau.

Khi đó xác suất quyết định theo đa số là đúng:

```
P_đa số = Σ (k từ (n+1)/2 đến n)  C(n,k) · p^k · (1−p)^(n−k)
```

**Kết luận:**

- Nếu **p > 0.5** thì P_đa số tăng theo n và **→ 1 khi n → ∞**.
- Nếu **p < 0.5** thì P_đa số **→ 0** (đông người hơn lại tệ hơn).
- Nếu p = 0.5 thì P_đa số = 0.5.

**Ví dụ số (p = 0.6):**

| Số mô hình n | Xác suất đa số đúng |
|---|---|
| 1 | 0.600 |
| 3 | 3·(0.6²·0.4) + 0.6³ = 0.432 + 0.216 = **0.648** |
| 5 | 0.3456 + 0.2592 + 0.0778 ≈ **0.683** |
| 101 | ≈ **0.98** (xấp xỉ chuẩn) |

Dù mỗi mô hình chỉ đúng 60%, kết hợp 101 mô hình độc lập đạt khoảng 98%.

**Ý nghĩa với Machine Learning và hạn chế:**

- Định lý ánh xạ trực tiếp sang ensemble: mỗi base learner là một "cử tri", và điều kiện *p > 0.5* chính là yêu cầu weak learner tốt hơn ngẫu nhiên.
- Trên thực tế, các mô hình huấn luyện trên **cùng một tập dữ liệu** nên **không độc lập hoàn toàn**. Vì vậy lợi ích thực tế thấp hơn lý thuyết, và các kỹ thuật như bootstrap, random subspace hay khác thuật toán sinh ra là để **tăng tính đa dạng** (giảm tương quan).

---

## 2. Phân tích Bias–Variance Tradeoff

### 2.1. Phân rã sai số

Với mô hình học f̂(x) huấn luyện trên tập dữ liệu ngẫu nhiên, sai số bình phương kỳ vọng tại điểm x phân rã thành:

```
E[(y − f̂(x))²] = Bias²[f̂(x)] + Var[f̂(x)] + σ²
```

| Thành phần | Ý nghĩa |
|---|---|
| **Bias** = E[f̂(x)] − f(x) | Độ lệch trung bình giữa dự đoán và giá trị thật. Do mô hình **quá đơn giản**, giả định sai, gây *underfitting*. |
| **Variance** = E[(f̂ − E[f̂])²] | Mức dao động của dự đoán khi đổi tập huấn luyện. Do mô hình **quá phức tạp**, gây *overfitting*. |
| **σ²** (nhiễu) | Nhiễu vốn có của dữ liệu, không thể giảm. |

### 2.2. Tradeoff

- Tăng độ phức tạp mô hình thì **bias giảm** nhưng **variance tăng**.
- Giảm độ phức tạp thì **bias tăng** nhưng **variance giảm**.
- Tổng sai số có dạng chữ U, và điểm tối ưu nằm ở mức phức tạp trung gian.

**Minh họa trên dữ liệu ung thư vú:**

| Mô hình | Bias | Variance | Hiện tượng |
|---|---|---|---|
| Decision Stump (cây sâu 1) | Cao | Thấp | Underfit |
| Cây quyết định sâu không giới hạn | Thấp | **Cao** | Overfit (nhớ nhiễu training set) |
| Logistic Regression | Trung bình | Thấp | Cân bằng khá tốt |

### 2.3. Ensemble giải quyết tradeoff như thế nào?

**(a) Giảm variance bằng trung bình hóa (Bagging).**

Với M mô hình, mỗi mô hình có phương sai σ² và tương quan đôi một là ρ, phương sai của trung bình:

```
Var(trung bình) = ρ·σ² + (1 − ρ)·σ² / M
```

- Khi M → ∞, số hạng thứ hai → 0, và phương sai còn lại là **ρ·σ²**.
- Nếu các mô hình độc lập (ρ = 0) thì variance giảm còn σ²/M.
- Nếu các mô hình giống hệt nhau (ρ = 1) thì không giảm gì.

Vì vậy cần **giảm ρ**, đây chính là lý do Random Forest còn chọn ngẫu nhiên tập đặc trưng ở mỗi nút. Bias của trung bình **gần như giữ nguyên** so với mô hình đơn, nên Bagging hợp với **mô hình bias thấp, variance cao** (cây sâu).

**(b) Giảm bias bằng học tuần tự (Boosting).**

Mỗi mô hình mới sửa lỗi của tổ hợp trước, nên mô hình cộng dồn ngày càng khớp dữ liệu hơn. Boosting hợp với **mô hình bias cao, variance thấp** (cây nông, stump). Nếu chạy quá nhiều vòng mà không có regularization thì variance có thể tăng và gây overfit.

**(c) Stacking** kết hợp nhiều loại mô hình có thiên lệch khác nhau, và meta-learner học cách cân bằng giữa chúng, nên có thể giảm cả hai thành phần.

| Phương pháp | Chủ yếu giảm | Base learner lý tưởng |
|---|---|---|
| Bagging / Random Forest | **Variance** | Cây sâu (low bias, high variance) |
| Boosting | **Bias** (và một phần variance) | Cây nông (high bias, low variance) |
| Stacking | Cả hai (tùy cấu hình) | Đa dạng nhiều loại |

---

## 3. Phân tích ba nhánh Ensemble

### 3.1. Bagging (Bootstrap Aggregating, Breiman 1996)

**Ý tưởng:** huấn luyện **song song, độc lập** nhiều mô hình trên các mẫu bootstrap khác nhau, rồi tổng hợp bằng bỏ phiếu hoặc trung bình.

**Quy trình:**

1. Từ tập huấn luyện N mẫu, lấy **M mẫu bootstrap**, mỗi mẫu gồm N điểm rút **có hoàn lại**.
2. Huấn luyện một base learner (thường là cây quyết định) trên mỗi mẫu bootstrap.
3. Dự đoán: **majority vote** (phân loại) hoặc **trung bình** (hồi quy).

**Tính chất toán học:**

- Xác suất một mẫu **không** nằm trong bootstrap: (1 − 1/N)^N → 1/e ≈ **0.368**. Do đó mỗi bootstrap chứa khoảng **63.2%** mẫu duy nhất.
- 36.8% mẫu còn lại là **Out-of-Bag (OOB)**, dùng để ước lượng sai số tổng quát hóa mà **không cần tập validation** riêng.

**Random Forest** là Bagging cộng thêm: tại mỗi nút chia, chỉ xét một tập con ngẫu nhiên gồm **√d** đặc trưng (với d = 30 thì xét khoảng 5 đặc trưng). Việc này làm các cây ít tương quan hơn (giảm ρ), và nhờ đó giảm variance tốt hơn.

| Ưu điểm | Nhược điểm |
|---|---|
| Giảm overfitting, ổn định | Ít giảm được bias |
| Huấn luyện song song được | Mô hình lớn, khó diễn giải hơn một cây |
| Có OOB error và feature importance | |
| Chịu nhiễu và outlier tốt | |

### 3.2. Boosting

**Ý tưởng:** huấn luyện các weak learner **tuần tự**. Mỗi learner mới **tập trung vào những mẫu mà các learner trước dự đoán sai**, và kết quả cuối là tổng có trọng số.

#### a) AdaBoost (Freund & Schapire, 1997), nhãn y ∈ {−1, +1}

1. Khởi tạo trọng số mẫu: `w_i = 1/N`.
2. Với mỗi vòng t = 1..T:
   - Huấn luyện weak learner `h_t` trên dữ liệu có trọng số `w`.
   - Tính sai số có trọng số: `ε_t = Σ w_i · 1[h_t(x_i) ≠ y_i]`.
   - Tính trọng số mô hình: `α_t = ½ · ln((1 − ε_t) / ε_t)`.
   - Cập nhật: `w_i ← w_i · exp(−α_t · y_i · h_t(x_i))`, rồi chuẩn hóa để Σ w_i = 1.
3. Dự đoán cuối: `H(x) = sign( Σ α_t · h_t(x) )`.

*Giải thích:* mẫu bị phân loại sai được **tăng trọng số**, mẫu đúng bị giảm. Mô hình có ε_t nhỏ thì α_t lớn, tức có tiếng nói mạnh hơn.

**Ví dụ tính tay một vòng** (N = 10, mỗi mẫu `w = 0.1`, stump sai 2 mẫu):

- ε₁ = 2 × 0.1 = 0.2
- α₁ = ½ · ln(0.8/0.2) = ½ · ln 4 ≈ **0.693**
- Mẫu sai: w = 0.1 · e^{0.693} = 0.2 (mỗi mẫu). Mẫu đúng: w = 0.1 · e^{−0.693} = 0.05 (mỗi mẫu).
- Tổng = 2·0.2 + 8·0.05 = 0.8, chuẩn hóa ra: mẫu sai **0.25**, mẫu đúng **0.0625**.

Sau một vòng, hai mẫu sai chiếm **50%** tổng trọng số, nên vòng 2 buộc phải tập trung vào chúng.

#### b) Gradient Boosting (Friedman, 2001)

Xây mô hình cộng tính `F_m(x) = F_{m−1}(x) + ν · h_m(x)`, trong đó `h_m` được huấn luyện để khớp **gradient âm của hàm mất mát** (với MSE chính là phần dư). `ν` là learning rate. Các biến thể hiện đại gồm **XGBoost, LightGBM, CatBoost**, thêm regularization, subsampling và tối ưu bậc hai.

| Ưu điểm | Nhược điểm |
|---|---|
| Độ chính xác rất cao trên dữ liệu bảng | Dễ overfit nếu quá nhiều vòng hoặc nhiễu nhiều |
| Giảm bias hiệu quả | Tuần tự nên khó song song hóa |
| Nhiều biến thể mạnh | Nhạy với siêu tham số (learning rate, số cây, độ sâu) |

### 3.3. Stacking (Stacked Generalization, Wolpert 1992)

**Ý tưởng:** dùng **một meta-model** học cách kết hợp tối ưu đầu ra của các base model **khác loại**, thay vì dùng quy tắc cố định như vote hay trung bình.

**Kiến trúc hai tầng:**

- **Level-0 (base learners):** ví dụ SVM, Random Forest, KNN, Logistic Regression.
- **Level-1 (meta-learner):** thường là mô hình đơn giản như Logistic Regression, nhận đầu vào là dự đoán (xác suất) của các base model.

**Quy trình:**

1. Chia training set thành K fold.
2. Với mỗi base model, huấn luyện trên K−1 fold và dự đoán fold còn lại. Lặp đủ K lần để có **dự đoán out-of-fold** cho toàn bộ training set.
3. Ghép các dự đoán out-of-fold của các base model thành **ma trận đặc trưng mới** (N × số mô hình).
4. Huấn luyện meta-learner trên ma trận này với nhãn thật.
5. Khi dự đoán dữ liệu mới: base model (huấn luyện lại trên toàn bộ train) dự đoán, rồi meta-learner tổng hợp.

**Lưu ý quan trọng:** phải dùng dự đoán **out-of-fold**. Nếu meta-learner học từ dự đoán của base model trên chính dữ liệu nó đã huấn luyện, sẽ bị **rò rỉ dữ liệu (data leakage)** và overfit nặng.

| Ưu điểm | Nhược điểm |
|---|---|
| Tận dụng điểm mạnh của nhiều họ mô hình | Phức tạp, tốn tính toán |
| Thường đạt hiệu năng cao nhất | Khó diễn giải |
| Linh hoạt trong lựa chọn mô hình | Dễ leakage nếu cài đặt sai |

### 3.4. Bảng so sánh tổng hợp

| Tiêu chí | Bagging | Boosting | Stacking |
|---|---|---|---|
| Cách huấn luyện | Song song, độc lập | Tuần tự, phụ thuộc nhau | Hai tầng |
| Dữ liệu cho mỗi model | Bootstrap sample | Toàn bộ, đã đổi trọng số | Toàn bộ (có K-fold) |
| Loại base learner | Thường đồng nhất | Thường đồng nhất (weak) | **Đa dạng (heterogeneous)** |
| Cách tổng hợp | Vote / trung bình | Tổng có trọng số | Meta-learner |
| Giảm chủ yếu | Variance | Bias | Cả hai |
| Rủi ro | Ít giảm bias | Overfit, nhạy nhiễu | Leakage, phức tạp |
| Đại diện | Random Forest | AdaBoost, XGBoost | StackingClassifier |

**Cài đặt tương ứng trong scikit-learn:** `BaggingClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`, `StackingClassifier` (và `XGBClassifier` nếu dùng thư viện xgboost).

---

## Tài liệu tham khảo

- Condorcet, M. (1785). *Essai sur l'application de l'analyse à la probabilité des décisions rendues à la pluralité des voix.*
- Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2).
- Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2).
- Freund, Y., & Schapire, R. E. (1997). A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. *Journal of Computer and System Sciences*, 55(1).
- Dietterich, T. G. (2000). Ensemble Methods in Machine Learning. *MCS 2000*.
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5).
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1).
- Wolberg, W. H., Street, W. N., & Mangasarian, O. L. Breast Cancer Wisconsin (Diagnostic) Data Set. UCI Machine Learning Repository.