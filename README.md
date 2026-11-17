# 🎯 Báo Cáo Giữa Kỳ AI: Khai Phá Ý Kiến Dựa Trên Khía Cạnh Đa Miền (Multi-Domain ABSA)

Dự án nghiên cứu và triển khai bài toán **Aspect-Based Sentiment Analysis (ABSA)** trên dữ liệu đa miền (Restaurant, Laptop, Hotel) sử dụng kiến trúc Seq2Seq sinh chuỗi văn bản **mT5-small (Multilingual T5)**, tối ưu hóa hàm mất mát có trọng số (Weighted Loss) để giải quyết hiện tượng mất cân bằng nhãn trầm trọng.

---

## 🌟 Điểm Nổi Bật Của Dự Án

1. **Khung tiếp cận Text-to-Text đa tác vụ (Unified Seq2Seq ABSA):**
   - Hỗ trợ tác vụ **uabsa** (Aspect-Sentiment extraction) và **tasd** (Triplet: Aspect-Category-Sentiment).
   - Mô hình học biểu diễn khía cạnh và phân loại cảm xúc đồng thời theo cơ chế chia sẻ tham số toàn phần (Hard Parameter Sharing).
2. **Xây dựng tập dữ liệu đa miền cân bằng:**
   - Trích xuất và tiền xử lý từ bộ dữ liệu M-ABSA với 3 miền: `restaurant`, `laptop`, `hotel`.
   - Tổng cộng **6,393 câu** (Train: 3,783, Dev: 950, Test: 1,660) và **9,209 khía cạnh**.
3. **Giải quyết mất cân bằng nhãn (Class Imbalance Mitigation):**
   - Thiết kế kỹ thuật **Token-level Weighted Loss** tăng trọng số phạt đối với các nhãn thiểu số (Neutral và Negative).
   - Đưa F1 của lớp Neutral từ **0%** lên **16.83%** và cải thiện rõ rệt Recall nhãn Negative (từ **18.79%** lên **33.07%**).
4. **Giao diện Web tương tác Streamlit (`app.py`):**
   - **Tab 1: Live Demo Suy Luận** - Nhập câu đánh giá bất kỳ và trực quan hóa các cặp khía cạnh - cảm xúc dạng tag màu.
   - **Tab 2: Khám Phá Dữ Liệu** - Biểu đồ phân phối mẫu, tỷ lệ cảm xúc theo từng miền thực tế.
   - **Tab 3: Bảng Đánh Giá Hiệu Năng** - Phân tích chi tiết Precision / Recall / F1 từng nhãn, Confusion Matrix, và phân tích lỗi.
5. **Tài liệu báo cáo & Slide thuyết trình đầy đủ:**
   - Slide thuyết trình chuẩn mực kèm hình minh họa kiến trúc và kết quả tại `docs/slide.md` và `docs/picture/`.

---

## 📂 Cấu Trúc Thư Mục

```text
├── app.py                     # Giao diện Web UI Streamlit tương tác
├── requirements.txt           # Danh sách các thư viện phụ thuộc
├── data/
│   └── multi_domain/          # Dữ liệu 3 miền đã trộn (train.txt, dev.txt, test.txt)
├── docs/                      # Slide, thuyết trình, biểu đồ minh họa và báo cáo
│   ├── noi_dung_slide_bao_cao_giua_ky.md
│   ├── slide.md
│   ├── progress.md
│   └── picture/               # Các biểu đồ kiến trúc và kết quả huấn luyện
├── M-ABSA/                    # Bộ dữ liệu gốc và mã nguồn tham chiếu M-ABSA
├── results/                   # Báo cáo đánh giá chi tiết (JSON & TSV predictions)
└── src/                       # Mã nguồn xử lý chính
    ├── analyze_data.py        # Thống kê phân bố nhãn và từ vựng
    ├── build_multi_domain.py  # Trộn tập dữ liệu đa miền
    ├── dataset.py             # Xử lý nạp dữ liệu và tokenizer Seq2Seq
    ├── engine.py              # Huấn luyện và đánh giá mô hình
    ├── evaluate_model.py      # Đánh giá chi tiết từng lớp & confusion matrix
    ├── metrics.py             # Tính toán F1, Precision, Recall với Fuzzy Matching
    ├── remove_conflict_labels.py # Làm sạch nhãn conflict
    ├── run_ablation.py        # Kịch bản thực nghiệm so sánh
    └── train.py               # Script huấn luyện với Weighted Loss & AdamW
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Chuẩn Bị Môi Trường

Khuyến nghị Python 3.10 hoặc 3.11:

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Kích hoạt môi trường (Linux / macOS)
# source .venv/bin/activate

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

### 2. Chạy Ứng Dụng Web UI (Streamlit)

```bash
streamlit run app.py
```
Truy cập ứng dụng tại địa chỉ: `http://localhost:8501`.

### 3. Huấn Luyện Mô Hình (Training)

Để huấn luyện mô hình với cấu hình tối ưu trên GPU 4GB VRAM (GTX 1650):

```bash
python src/train.py \
    --data_dir data/multi_domain \
    --output_dir checkpoints/weighted_loss_s42 \
    --task uabsa \
    --model_name google/mt5-small \
    --num_train_epochs 3 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --fp16 \
    --loss_mode weighted \
    --neutral_weight 5.0 \
    --negative_weight 2.0
```

### 4. Đánh Giá Mô Hình (Evaluation)

Đánh giá chi tiết mô hình trên tập kiểm thử:

```bash
python src/evaluate_model.py \
    --checkpoint checkpoints/weighted_loss_s42/best_model \
    --task uabsa \
    --data_file data/multi_domain/test.txt \
    --output_json results/weighted_loss_s42_test.json \
    --output_tsv results/weighted_loss_s42_test_predictions.tsv
```

---

## 📊 Kết Quả Thực Nghiệm

| Phương Pháp / Cấu Hình | Precision (%) | Recall (%) | Micro F1 (%) | Macro F1 (%) | Neutral F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Chuẩn (Standard Loss)** | 66.27 | 56.38 | 60.93 | 36.31 | 0.00 |
| **Đề Xuất (Weighted Loss, Seed 42)** | 61.54 | 50.39 | 55.41 | **40.02** | **16.83** |

*Ghi chú:* Nhờ cơ chế phạt có trọng số theo token, mô hình đã khắc phục triệt để điểm mù đối với lớp cảm xúc thiểu số (Neutral tăng từ 0% lên 16.83% F1, Negative Recall tăng từ 18.79% lên 33.07%).
