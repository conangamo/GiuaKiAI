# BÁO CÁO THỐNG KÊ TẬP DỮ LIỆU ĐA MIỀN (MULTI-DOMAIN ABSA) - GIAI ĐOÀN 2

Tài liệu này tổng hợp cấu trúc, phân bố và thống kê của tập dữ liệu Đa miền đã được trộn để phục vụ nội dung bắt buộc cho **Slide 2** của báo cáo giữa kỳ.

---

## 1. CÁC MIỀN THAM GIA TRỘN (DOMAINS)

Theo đúng yêu cầu của đề tài ("kết hợp dữ liệu từ 2 miền trở lên, ví dụ Laptop, Hotel, Restaurant"), tập dữ liệu đa miền được tổng hợp từ 3 miền có sự đa dạng phong phú về từ vựng và ngữ cảnh:
1. **Hotel (Khách sạn):** Đánh giá về cơ sở vật chất, vị trí, dịch vụ phòng, sự thoải mái.
2. **Laptop (Máy tính xách tay):** Đánh giá về linh kiện phần cứng, pin, màn hình, hiệu năng, bàn phím.
3. **Restaurant (Nhà hàng):** Đánh giá về món ăn, giá cả, thái độ phục vụ, không gian ẩm thực.

---

## 2. BẢNG THỐNG KÊ TỔNG QUAN TẬP DỮ LIỆU ĐA MIỀN

| Phân chia (Split) | Miền Hotel | Miền Laptop | Miền Restaurant | **Tổng số câu (Sentences)** | **Tổng số khía cạnh (Triplets)** | Phân bố Cảm xúc (Pos / Neg / Neu) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 1,255 | 1,264 | 1,264 | **3,783** | **5,377** | 3,812 / 1,350 / 215 |
| **DEV** | 308 | 326 | 316 | **950** | **1,301** | 891 / 347 / 63 |
| **TEST** | 584 | 532 | 544 | **1,660** | **2,531** | 1,877 / 553 / 101 |
| **TỔNG CỘNG** | **2,147** | **2,122** | **2,124** | **6,393** | **9,209** | **6,580 / 2,250 / 379** |

---

## 3. ĐẶC ĐIỂM VÀ PHÂN BỐ CỦA DỮ LIỆU

### 1. Tỷ lệ đóng góp giữa các miền:
* Dữ liệu từ 3 miền đóng góp đồng đều theo tỷ lệ xấp xỉ **1 : 1 : 1** (~33.3% mỗi miền).
* Điều này giúp mô hình không bị thiên lệch (bias) về bất kỳ miền dữ liệu nào trong quá trình huấn luyện.

### 2. Phân bố nhãn quan điểm (Sentiment Polarity):
* **Tích cực (Positive):** 71.4% (6,580 cặp)
* **Tiêu cực (Negative):** 24.4% (2,250 cặp)
* **Trung tính (Neutral):** 4.1% (379 cặp)
*(Đây là phân bố tự nhiên đặc trưng của các bài đánh giá sản phẩm thực tế, nơi người dùng thường khen ngợi hoặc chê bai rõ ràng hơn là trung tính).*

### 3. Đa dạng khía cạnh (Cross-domain Aspects):
* Tập dữ liệu chứa hơn **1,850 khía cạnh duy nhất** từ vựng khác biệt:
  * Khách sạn: *room, staff, bed, location, breakfast, shower, view...*
  * Laptop: *battery, keyboard, screen, processor, fan, memory, weight...*
  * Nhà hàng: *food, pizza, waiter, price, wine, atmosphere, sushi...*

---

## 4. TỆP DỮ LIỆU ĐÃ TẠO

Dữ liệu đã được lưu trữ sẵn sàng tại:
* Thư mục dự án: `data/multi_domain/`
  * `train.txt` (3,783 dòng)
  * `dev.txt` (950 dòng)
  * `test.txt` (1,660 dòng)
  * `dataset_statistics.json` (Chi tiết JSON)
* Bản sao đồng bộ tại: `M-ABSA/data/multi_domain/`

---

## 5. CÁCH XEM TRỰC QUAN TRÊN GIAO DIỆN WEB UI

1. Chạy lệnh:
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
2. Vào **Tab 2: 📊 Khám Phá Dữ Liệu Đa Miền**.
3. Chọn miền: `multi_domain (Trộn Hotel + Laptop + Restaurant)`.
4. Giao diện sẽ hiển thị ngay lập tức biểu đồ tròn phân bố cảm xúc và các câu mẫu đã trộn ngẫu nhiên.

