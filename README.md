# 🎯 Phân Tích Quan Điểm Theo Khía Cạnh Đa Miền (Multi-Domain ABSA)

Dự án nghiên cứu và triển khai bài toán **Aspect-Based Sentiment Analysis (ABSA)** trên dữ liệu tiếng Anh đa miền (`hotel`, `laptop`, `restaurant`) sử dụng kiến trúc Seq2Seq sinh chuỗi văn bản **mT5-small (Multilingual T5)**, tích hợp kỹ thuật xử lý mất cân bằng nhãn (**Token-level Weighted Cross-Entropy** & **Oversampling**) và hệ thống đánh giá đa chiều (Micro-F1, Macro-F1, Per-class, Per-domain, Explicit vs `NULL`).

---

## 🌟 Điểm Nổi Bật Của Dự Án

1. **Khung tiếp cận Text-to-Text đa tác vụ (Unified Seq2Seq UABSA):**
   - Thực hiện đồng thời **Trích xuất khía cạnh (Aspect Extraction)** và **Phân loại cảm xúc (Sentiment Classification)** trong một forward pass duy nhất (chia sẻ 100% tham số Encoder–Decoder).
   - Xử lý tự nhiên cả khía cạnh hiện và khía cạnh ẩn (`NULL`).
2. **Tập dữ liệu tiếng Anh đa miền cân bằng 1 : 1 : 1:**
   - Trộn từ 3 miền `hotel`, `laptop`, `restaurant` của bộ M-ABSA / SemEval (`src/build_multi_domain.py`).
   - Tổng cộng **6,393 câu** (Train: 3,783 | Dev: 950 | Test: 1,660), **9,209 nhãn bộ ba** sau khi làm sạch nhãn `conflict`, tương ứng **7,882 cặp `(aspect, polarity)` duy nhất** (Test: 2,185 cặp chuẩn).
3. **Xử lý mất cân bằng nhãn (Class Imbalance Mitigation):**
   - Hỗ trợ `oversample`, `weighted_loss` (tăng trọng số trên đúng token cảm xúc hiếm trong từ điển SentencePiece) và `both`, với hệ số `--power` điều chỉnh mức cân bằng $w_c = (N_{\max} / N_c)^{\text{power}}$.
4. **Giao diện Web tương tác Streamlit ([app.py](file:///f:/laptrinhPython/chuyenDe/github/ABSA/app.py)):**
   - **Tab 1 – Live Demo:** Tự động phát hiện các checkpoint trong `checkpoints/`, suy luận trực tiếp trên tiếng Anh tự nhiên và tô màu khía cạnh an toàn.
   - **Tab 2 – Khám Phá Dữ Liệu:** Biểu đồ phân bố cảm xúc và xem mẫu theo từng miền/ngôn ngữ.
   - **Tab 3 – Bảng Đánh Giá Hiệu Năng:** Nạp động dữ liệu thật từ `results/*.json` và `results/*_predictions.tsv` (Per-class, Domain Breakdown, Confusion Matrix, bộ lọc soi lỗi).

---

## 📂 Cấu Trúc Thư Mục

```text
├── app.py                        # Giao diện Web UI Streamlit tương tác
├── requirements.txt              # Danh sách các thư viện phụ thuộc
├── data/
│   └── multi_domain/             # Dữ liệu 3 miền đã trộn (train/dev/test.txt & *_domains.txt)
├── docs/                         # Slide, lời thoại thuyết trình, biểu đồ và kế hoạch
│   ├── noi_dung_slide_bao_cao_giua_ky.md
│   ├── slide.md
│   ├── progress.md
│   └── picture/                  # Script generate_slide_charts.py & 10 ảnh biểu đồ 300 DPI
├── M-ABSA/                       # Bộ dữ liệu gốc và mã nguồn tham chiếu M-ABSA
├── results/                      # Báo cáo đánh giá thật (JSON & TSV predictions)
└── src/                          # Mã nguồn xử lý chính
    ├── analyze_data.py           # Thống kê dữ liệu thật -> results/data_analysis.json
    ├── build_multi_domain.py     # Trộn tập dữ liệu 3 miền cân bằng 1:1:1
    ├── dataset.py                # Dataset PyTorch, format_target & chuẩn hóa đầu vào tiếng Anh
    ├── engine.py                 # trim_batch, compute_val_loss, generate_predictions
    ├── evaluate_model.py         # Đánh giá đầy đủ dev/test -> JSON & TSV
    ├── metrics.py                # Micro/Macro F1, Per-class, Per-domain, Confusion Matrix
    ├── remove_conflict_labels.py # Làm sạch nhãn conflict
    ├── run_ablation.py           # Chạy tự động các cấu hình ablation & đánh giá
    └── train.py                  # Huấn luyện mT5 (AdamW, Warmup, Weighted Loss, Oversample)
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Chuẩn Bị Môi Trường (Python 3.10 / 3.11)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Chạy Giao Diện Web UI (Streamlit)

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

### 3. Huấn Luyện Mô Hình (Tối ưu cho GPU 4GB VRAM - GTX 1650)

> **Lưu ý:** Không bật `--fp16` với `google/mt5-small` vì kiến trúc mT5 dễ bị tràn số `NaN` ở FP16; dùng FP32 kết hợp `trim_batch()` và `gradient_accumulation_steps=4`.

```powershell
# Huấn luyện Baseline hội tụ (8 epoch)
.\.venv\Scripts\python.exe src/train.py `
    --model_name_or_path google/mt5-small `
    --data_dir data/multi_domain `
    --task uabsa `
    --epochs 8 `
    --batch_size 4 `
    --gradient_accumulation_steps 4 `
    --lr 3e-4 `
    --max_seq_len 128 `
    --max_target_len 64 `
    --imbalance none `
    --select_metric f1 `
    --early_stopping_patience 3 `
    --output_dir checkpoints/baseline_e8_s42

# Hoặc chạy tự động toàn bộ Ablation (baseline, wl_p03, oversample)
.\.venv\Scripts\python.exe src/run_ablation.py --configs baseline wl_p03 oversample --epochs 8 --seed 42
```

### 4. Đánh Giá Mô Hình Trên Toàn Bộ Tập Test / Dev

```powershell
.\.venv\Scripts\python.exe src/evaluate_model.py `
    --model_path checkpoints/baseline_e8_s42/best_model `
    --data_dir data/multi_domain `
    --split test `
    --run_name baseline_e8_s42 `
    --num_examples_to_show 5
```

---

## 📊 Kết Quả Thực Nghiệm (Tập Test Tiếng Anh Đa Miền — 1,660 câu, 2,185 cặp chuẩn)

| Cấu Hình Mô Hình | Precision (%) | Recall (%) | Micro-F1 (%) | Macro-F1 (%) | Positive F1 | Negative F1 | Neutral F1 | Aspect F1 | Polarity Acc* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (CE chuẩn, 3 epoch)** | 66.33 | 56.34 | 60.93 | 39.09 | 65.91 | 51.36 | 0.00 | 70.77 | 84.96% |
| **Weighted Loss (`power=0.5`, 3 epoch)** | 61.54 | 50.39 | 55.41 | 40.02 | 61.97 | 41.27 | 16.83 | 67.89 | 80.54% |
| **Baseline Hội tụ (CE chuẩn, 8 epoch)** | 66.46 | 62.75 | 64.55 | 46.43 | 68.48 | **57.96** | 12.84 | **73.07** | 87.55% |
| **Weighted Loss (`power=0.3`, 8 epoch — Tốt nhất)** | **66.54** | **62.79** | **64.61** | **52.23** | **68.84** | 57.35 | **30.51** | 73.05 | **87.72%** |

* **Phân rã theo 3 miền trên tập Test (`wl_p03_e8_s42`):**
  * **Hotel (584 câu):** Micro-F1 = **65.14%** | Macro-F1 = **47.24%** | Neutral F1 = **23.53%**
  * **Laptop (532 câu):** Micro-F1 = **65.12%** | Macro-F1 = **55.01%** | Neutral F1 = **31.46%**
  * **Restaurant (544 câu):** Micro-F1 = **63.57%** | Macro-F1 = **51.84%** | Neutral F1 = **33.33%**
* **Kiểm thử nhanh 20 câu tiếng Anh tự nhiên (`src/smoke_test_en.py`):** Đạt chất lượng chính xác cao trên cả câu khen, chê, trung tính, câu trái chiều và khía cạnh ẩn `NULL`.
