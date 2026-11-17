# HƯỚNG DẪN CHẠY VÀ SỬ DỤNG DỰ ÁN - GIAI ĐOÀN 1

Tài liệu này hướng dẫn chi tiết cách chạy, kiểm thử và theo dõi kết quả của dự án Phân tích Quan điểm theo Khía cạnh (ABSA) ở **Giai đoạn 1 (Kiểm thử Baseline trên 1 miền)**.

---

## 1. TỔNG QUAN GIAI ĐOÀN 1

* **Bài toán:** Unified ABSA (`uabsa`) – Trích xuất đồng thời cặp `(Khía cạnh, Cảm xúc)`.
* **Mô hình sử dụng:** `google/mt5-small` (Seq2Seq / Text-to-Text).
* **Cơ chế:** Encoder-Decoder chia sẻ toàn phần biểu diễn tiềm ẩn để sinh trực tiếp chuỗi kết quả.
* **Tối ưu phần cứng:** Đã tinh chỉnh để chạy ổn định trên GPU **NVIDIA GeForce GTX 1650 (4GB VRAM)**, bộ nhớ chiếm dụng ~2.5 - 3.0 GB, không bị lỗi tràn RAM đồ họa (OOM).

---

## 2. KÍCH HOẠT MÔI TRƯỜNG ẢO (`.venv`)

Mở terminal PowerShell tại thư mục gốc của dự án (`F:\laptrinhPython\chuyenDe\github\ABSA`):

```powershell
# Kích hoạt virtual environment
.\.venv\Scripts\Activate.ps1
```

*(Nếu PowerShell báo lỗi hạn chế script, bạn có thể chạy trực tiếp bằng đường dẫn: `.\.venv\Scripts\python.exe ...`)*

---

## 3. CÁC LỆNH CHẠY TRONG GIAI ĐOÀN 1

### 🔹 Cách 1: Chạy kiểm tra nhanh đường ống (Dry Run - mất ~10 giây)
Lệnh này kiểm tra quá trình nạp dữ liệu, khởi tạo mô hình, forward pass, tính loss và cập nhật gradient qua 2 batch mà không cần đợi train hết:

```powershell
.\.venv\Scripts\python.exe src/train.py --model_name_or_path google/mt5-small --data_dir restaurant --task uabsa --batch_size 2 --gradient_accumulation_steps 2 --dry_run
```

**Kết quả mong đợi:**
```text
=== ABSA Multi-Task Training with google/mt5-small ===
Device: cuda (NVIDIA GeForce GTX 1650)
Task: uabsa | Domain: restaurant | Language: en
...
Train samples: 1264 | Dev samples: 316
Sample Input : We have gone for dinner only a few times but the same great quality and service is given .
Sample Target: (service, positive); (dinner, positive)
Epoch 1/5: [Dry Run] Loss: 19.6897
[Dry Run] Successfully verified forward, backward and optimizer steps! Pipeline is fully functional.
```

---

### 🔹 Cách 2: Chạy huấn luyện thực tế trên 1 miền (Full Training)
Huấn luyện mô hình trên miền Nhà hàng (`restaurant`) với 3 đến 5 epoch:

```powershell
.\.venv\Scripts\python.exe src/train.py --model_name_or_path google/mt5-small --data_dir restaurant --task uabsa --epochs 3 --batch_size 2 --gradient_accumulation_steps 4 --lr 3e-4 --output_dir checkpoints
```

**Các tham số quan trọng:**
* `--data_dir`: Tên miền dữ liệu (`restaurant`, `hotel`, `laptop`).
* `--epochs`: Số vòng lặp huấn luyện (khuyên dùng `3` - `5` cho máy cá nhân).
* `--batch_size 2`: Kích thước batch phù hợp nhất với VRAM 4GB.
* `--gradient_accumulation_steps 4`: Tích lũy gradient qua 4 bước (tương đương effective batch size = 8).
* `--output_dir checkpoints`: Nơi lưu checkpoint mô hình tốt nhất (`checkpoints/best_model`).

---

### 🔹 Cách 3: Đánh giá mô hình đã huấn luyện (Evaluation & Inference)
Sau khi huấn luyện xong, chạy lệnh đánh giá trên tập kiểm thử (`test.txt`):

```powershell
.\.venv\Scripts\python.exe src/evaluate_model.py --model_path checkpoints/best_model --data_dir restaurant --split test --task uabsa --num_examples_to_show 5
```

---

### 🔹 Cách 4: Khởi chạy Giao diện Trực quan Web UI (Streamlit Dashboard)
Để quan sát trực quan, demo cho giảng viên hoặc thử nghiệm các câu review tùy ý trên giao diện web:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Trình duyệt sẽ tự động mở trang dashboard tại `http://localhost:8501` với 3 tab:
* **🔍 Trực quan hóa Dự đoán (Live Demo):** Nhập câu review bất kỳ (tiếng Anh/Việt), hệ thống tự highlight khía cạnh và gắn badge cảm xúc.
* **📊 Khám phá Dữ liệu Đa miền:** Xem phân bố nhãn cảm xúc theo biểu đồ tròn (Pie chart), số lượng câu và duyệt mẫu dữ liệu.
* **📈 Bảng Đánh Giá Hiệu Năng:** Bảng chỉ số Precision, Recall, F1 và đối sánh kết quả dự đoán.

---

## 4. CÁCH ĐỌC VÀ HIỂU ĐẦU RA / KẾT QUẢ

### 1. Định dạng kết quả sinh chuỗi (Text-to-Text Format)
* **Đầu vào (Input Sentence):**
  > *"The food was delicious, but the waiter was rude."*
* **Đầu ra mục tiêu (Gold Target):**
  > `(food, positive); (waiter, negative)`
* **Đầu ra mô hình dự đoán (Prediction):**
  > `(food, positive); (waiter, negative)`

Mỗi cặp ngoặc đơn biểu diễn một khía cạnh được trích xuất cùng với trạng thái cảm xúc tương ứng: `(khía cạnh, nhãn cảm xúc)`. Nếu câu không chứa khía cạnh nào mang quan điểm, mô hình sẽ xuất ra `None`.

### 2. Bảng chỉ số đánh giá (Evaluation Metrics)
Khi chạy `evaluate_model.py`, chương trình sẽ xuất ra bảng đo lường chuẩn:
```text
========================================
EVALUATION RESULTS (TEST SET):
Precision       : 74.25%   --> Trong số các cặp mô hình dự đoán, bao nhiêu % là đúng
Recall          : 71.80%   --> Mô hình tìm ra được bao nhiêu % khía cạnh thực tế trong văn bản
F1-Score        : 73.00%   --> Điểm số trung hòa giữa Precision và Recall (chỉ số quan trọng nhất)
Total Gold Pairs: 450      --> Tổng số cặp thực tế cần trích xuất
Total Pred Pairs: 435      --> Tổng số cặp mô hình đã sinh ra
True Positives  : 323      --> Số cặp mô hình trích xuất đúng cả khía cạnh lẫn cảm xúc
========================================
```

### 3. Xem các ví dụ trực quan (Qualitative Inspection)
Chương trình sẽ tự động in ra các ví dụ cụ thể để bạn kiểm tra lỗi:
```text
[1] Sentence: Excellent atmosphere , delicious dishes good and friendly service .
    Gold Target: (atmosphere, positive); (dishes, positive); (service, positive)
    Prediction : (atmosphere, positive); (dishes, positive); (service, positive)

[2] Sentence: Not what I would expect for the price and prestige of this location .
    Gold Target: (price, neutral); (location, neutral)
    Prediction : (price, negative); (location, neutral)
```
*(Qua các ví dụ này, bạn có thể giải thích trực tiếp trên slide lý do mô hình nhầm lẫn giữa `neutral` và `negative`)*.

---

## 5. THƯ MỤC LƯU TRỮ TRONG GIAI ĐOÀN 1

```text
ABSA/
├── .venv/                      # Môi trường ảo Python 3.11 + PyTorch CUDA
├── checkpoints/
│   └── best_model/             # Checkpoint mô hình có F1 cao nhất
│       ├── config.json
│       ├── model.safetensors
│       ├── spiece.model
│       └── tokenizer_config.json
├── docs/
│   ├── requirement.md          # Yêu cầu đề tài
│   ├── progress.md             # Tiến độ các giai đoạn
│   └── huong_dan_giai_doan_1.md # File hướng dẫn này
├── M-ABSA/data/                # Kho dữ liệu gốc (7 miền, 21 ngôn ngữ)
└── src/
    ├── dataset.py              # Xử lý nạp dữ liệu ABSA
    ├── metrics.py              # Đánh giá Precision, Recall, F1
    ├── train.py                # Huấn luyện mô hình mT5
    ├── evaluate_model.py       # Đánh giá và in mẫu dự đoán
    └── build_multi_domain.py   # Script chuẩn bị cho Giai đoạn 2
```

