# HƯỚNG DẪN TỰ CHẠY VÀ ĐÁNH GIÁ - GIAI ĐOÀN 3 (HUẤN LUYỆN ĐA MIỀN)

Tài liệu này hướng dẫn bạn tự chạy quá trình huấn luyện mô hình trên tập dữ liệu **Đa miền (Multi-Domain)**, kiểm thử đánh giá và xem kết quả trực quan trên Web UI.

---

## 1. TỔNG QUAN HUẤN LUYỆN GIAI ĐOÀN 3

* **Tập dữ liệu:** `data/multi_domain` (gồm **3,783 câu** kết hợp từ Hotel, Laptop và Restaurant).
* **Mô hình:** `google/mt5-small` (~300 triệu tham số).
* **Tác vụ:** `uabsa` (Trích xuất đồng thời cặp Khía cạnh - Cảm xúc).
* **Thời gian chạy dự kiến:** Khoảng **15 – 25 phút** cho 3 epoch trên GPU GTX 1650 (4GB VRAM).
* **Checkpoints:** Tự động theo dõi điểm F1 trên tập Dev và lưu checkpoint tốt nhất vào thư mục `checkpoints/best_model`.

---

## 2. BƯỚC 1: LỆNH CHẠY HUẤN LUYỆN (TRAINING)

Mở terminal PowerShell tại thư mục gốc của dự án (`F:\laptrinhPython\chuyenDe\github\ABSA`), dán lệnh sau và nhấn **Enter**:

```powershell
.\.venv\Scripts\python.exe src/train.py --model_name_or_path google/mt5-small --data_dir data/multi_domain --task uabsa --epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --lr 3e-4 --max_seq_len 64 --output_dir checkpoints
```

### 🔍 Giải thích các tham số:
* `--data_dir data/multi_domain`: Nạp tập dữ liệu đa miền đã trộn.
* `--epochs 3`: Huấn luyện 3 vòng lặp (đủ để mô hình hội tụ tốt ngữ nghĩa cảm xúc mà không mất quá nhiều thời gian).
* `--batch_size 4`: Kích thước batch tối ưu tốc độ và an toàn cho 4GB VRAM.
* `--gradient_accumulation_steps 4`: Tích lũy 4 bước -> Effective batch size = 16.
* `--max_seq_len 64`: Độ dài chuỗi tối ưu cho câu review, tiết kiệm bộ nhớ đồ họa.

### 🖥️ Màn hình hiển thị trong lúc chạy:
* Thanh tiến trình `Epoch 1/3: [===>...]` sẽ hiển thị tốc độ (~1.5 - 2.0 batch/giây).
* Chỉ số `loss` sẽ giảm dần từ ~`16.0` xuống ~`2.0 - 4.0`.
* Cuối mỗi epoch, chương trình sẽ tự động đánh giá trên tập Validation:
  ```text
  --- Epoch 1 Validation ---
  Loss: 4.1205 | Precision: 72.30% | Recall: 68.50% | F1: 70.34%
  --> New best model saved to checkpoints/best_model (F1: 70.34%)
  ```

---

## 3. BƯỚC 2: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ (TEST SET)

Sau khi huấn luyện xong, chạy lệnh đánh giá trên tập kiểm thử đa miền (`1,660 câu test`):

```powershell
.\.venv\Scripts\python.exe src/evaluate_model.py --model_path checkpoints/best_model --data_dir data/multi_domain --split test --task uabsa --num_examples_to_show 5
```

### 📊 Màn hình kết quả:
Chương trình sẽ in ra bảng số liệu chuẩn mực:
```text
========================================
EVALUATION RESULTS (TEST SET):
Precision       : xx.xx%
Recall          : xx.xx%
F1-Score        : xx.xx%
Total Gold Pairs: 2531
Total Pred Pairs: xxxx
True Positives  : xxxx
========================================
```
*(Bạn ghi lại 3 chỉ số **Precision, Recall, F1** này để thông báo cho tôi và đưa vào Slide báo cáo).*

---

## 4. BƯỚC 3: THỬ NGHIỆM TRỰC QUAN TRÊN GIAO DIỆN WEB UI

Sau khi có `checkpoints/best_model`, bạn mở Web UI:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 🎯 Thử nghiệm lại câu lúc nãy:
Ở thanh bên trái (Sidebar), chọn: **`checkpoints/best_model (Mô hình đã Fine-tune)`**.
Sau đó nhập câu bạn vừa thử:
> *"The food was absolutely amazing, but the service was terrible and slow."*

Nhấn **🚀 Phân Tích Quan Điểm**:
* **Kết quả mong đợi sau khi Fine-tune:**
  * **FOOD:** Tích cực 👍 (Xanh lá)
  * **SERVICE:** Tiêu cực 👎 (Đỏ)
* Thử tiếp các câu của các miền khác:
  * Miền Laptop: *"The screen is gorgeous and battery life is great, but the keyboard is too stiff."*
  * Miền Hotel: *"The hotel room was very clean with a stunning sea view, but the price was outrageous."*

---

## 5. THÔNG TIN BẠN CẦN BÁO LẠI SAU KHI CHẠY XONG:

Sau khi chạy xong, bạn chỉ cần báo lại cho tôi:
1. **Điểm F1-Score cuối cùng** trên tập test (khoảng bao nhiêu %).
2. Kết quả khi thử nghiệm câu trên Web UI (mô hình đã tách đúng cảm xúc tích cực/tiêu cực chưa).

Sau đó chúng ta sẽ bước thẳng sang **Giai đoạn 4: Soạn nội dung bộ Slide 5–8 trang hoàn chỉnh** cho bạn nộp giữa kỳ!

