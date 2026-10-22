# Nhận xét thực nghiệm

Các chỉ số dưới đây được tính trên một tập test phân tầng 20%;
đây là kết quả của một lần chia dữ liệu, không phải ước lượng tổng quát hóa.

- Mô hình có F1 cao nhất trong lần chạy này: **Stacking** (F1=0.988, Recall=0.976, ROC-AUC=0.996).
- Base model tốt nhất theo F1: **Logistic Regression** (F1=0.964); chênh lệch F1 giữa hai mô hình là +0.024. Kết quả này không khẳng định ensemble luôn tốt hơn.
- Với bài toán sàng lọc, Recall của lớp malignant cần được xem xét cùng Precision vì bỏ sót ca malignant có chi phí cao; ROC-AUC đánh giá khả năng xếp hạng ở nhiều ngưỡng.
- Random Forest dự đoán sai 3 mẫu trên test, gồm 3 false negative (M bị đoán thành B) và 0 false positive (B bị đoán thành M). Xem `results/misclassified_random_forest.csv` để kiểm tra từng trường hợp.
- Feature importance của cây là mức đóng góp theo tiêu chí chia trong các cây, không chứng minh quan hệ nhân quả và có thể thiên lệch với các đặc trưng tương quan.
- Ensemble thường giảm phương sai (Random Forest), kết hợp xác suất (Soft Voting/Stacking) hoặc tuần tự tập trung vào mẫu khó (AdaBoost), nhưng có thể tốn thời gian hơn và không đảm bảo thắng trên mọi lần chia.
- Các mô hình chỉ phục vụ học tập trên bộ dữ liệu công khai; không dùng để chẩn đoán hay thay thế ý kiến chuyên gia y tế.
