Bước 1 – Làm cho số liệu trung thực (nhanh, bắt buộc):

Thêm precision/recall/F1 theo từng lớp cảm xúc, macro-F1 và F1 theo từng miền vào evaluate_model.py.
Đánh giá trên toàn bộ dev.
Sửa tab 3 của app.py để đọc kết quả thật từ file.
Sửa lại các câu chưa chính xác ở trên trong slide.

Bước 2 – Xử lý mất cân bằng, nhưng chỉ sau khi có số đo từng lớp. Có thể neutral recall chỉ khoảng 20% và đó mới là vấn đề thật. Lúc đó chọn một trong các cách sau:

Oversample các câu có neutral hoặc negative.
Tăng trọng số loss cho token cảm xúc (khả thi với seq2seq, cần viết loss tùy chỉnh).
Báo cáo macro-F1 thay vì chỉ micro-F1.

Mỗi cách cần chạy một ablation để có số liệu so sánh.

Bước 3 – Làm đúng yêu cầu multi-task (quan trọng nhất, tốn công nhất):

Xây mô hình XLM-R hoặc mBERT làm encoder dùng chung, rẽ hai head:
Head 1: gán nhãn BIO để trích xuất khía cạnh.
Head 2: phân loại cảm xúc, với loss tổng L = L_aspect + λ·L_sent.
Giữ mT5 làm baseline để so sánh.
Cách này khớp trực tiếp với "Hard Parameter Sharing" và vẫn mở rộng được sang đa ngôn ngữ vì XLM-R là mô hình đa ngôn ngữ.
Chi phí: dữ liệu chỉ có chuỗi khía cạnh chứ không có vị trí, nên phải tự dò vị trí trong câu. Các khía cạnh NULL không gán BIO được, nên cần một head cấp câu hoặc loại bỏ chúng.
Phần cứng không phải vấn đề: XLM-R base chạy được trên 4GB VRAM.

Bước 4 – Ablation tối thiểu để chứng minh đa miền có ích: đơn miền so với đa miền, có và không có pretrained. Mỗi cấu hình là một lần chạy ngắn.