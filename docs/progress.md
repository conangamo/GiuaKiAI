📌 GIAI ĐOÀN 1: Chuẩn bị Môi trường & Sửa lỗi Mã nguồn (Ưu tiên số 1)
Thời gian dự kiến: 1 – 2 ngày.
Các công việc cần làm:
- [x] Tạo Virtual Environment (.venv với Python 3.11).
- [x] Cài đặt PyTorch có hỗ trợ CUDA tương thích với GPU GTX 1650 (torch-2.6.0+cu124).
- [x] Cài đặt: transformers, sentencepiece, tqdm, editdistance, scikit-learn, pandas, protobuf, tiktoken.
- [x] Refactor nhẹ mã nguồn để tương thích & không phụ thuộc phiên bản cũ:
  - [x] Tạo `src/dataset.py` sửa đường dẫn nạp file để trỏ linh hoạt vào data/<domain>/<lang>/... và chuẩn hóa định dạng uabsa/tasd.
  - [x] Tạo `src/metrics.py` tính Precision, Recall, F1 với fuzzy matching.
  - [x] Tạo `src/train.py` huấn luyện mượt mà trên PyTorch thuần, hỗ trợ fp16, gradient accumulation.
  - [x] Tạo `src/evaluate_model.py` đánh giá và trích xuất dự đoán.
  - [x] Tạo `src/build_multi_domain.py` trộn dữ liệu đa miền và thống kê.
- [x] Tối ưu tham số cho GPU GTX 1650 (4GB):
  - [x] Thiết lập train_batch_size = 2 hoặc 4.
  - [x] Thiết lập gradient_accumulation_steps = 4 - 8.
  - [x] Kích hoạt fp16 = True để tiết kiệm 50% VRAM.
- [x] Chạy thử nghiệm một batch nhỏ (Dry run) để đảm bảo loss giảm và mã nguồn chạy thông suốt (Đã xác nhận loss ổn định, gradient update chuẩn xác trên GPU NVIDIA GeForce GTX 1650).
📌 GIAI ĐOÀN 2: Xây dựng Tập dữ liệu Đa miền (Trọng tâm Giữa kỳ)
Thời gian dự kiến: 1 – 2 ngày.
Các công việc cần làm:
- [x] Viết script build_multi_domain.py (`src/build_multi_domain.py`).
- [x] Lấy dữ liệu từ 3 miền: restaurant, hotel, laptop (tiếng Anh en).
- [x] Trộn theo tỷ lệ cân bằng 1:1:1 và phân chia lại thành: train.txt, dev.txt, test.txt.
- [x] Lưu vào thư mục `data/multi_domain/` và `M-ABSA/data/multi_domain/`.
- [x] Xuất bảng số liệu & vẽ biểu đồ phân tích (dành cho Slide):
  - [x] Tổng số câu: 6,393 câu (Train: 3,783, Dev: 950, Test: 1,660).
  - [x] Tổng số khía cạnh: 9,209 triplets (Positive: 6,580, Negative: 2,250, Neutral: 379).
  - [x] Tạo tài liệu thống kê chi tiết sẵn sàng cho slide tại `docs/thong_ke_da_mien_giai_doan_2.md`.
  - [x] Tích hợp tùy chọn `multi_domain` vào giao diện Web UI Streamlit để xem trực quan biểu đồ.
📌 GIAI ĐOÀN 3: Huấn luyện & Đánh giá Thực nghiệm (Fine-tuning & Evaluation)
Thời gian dự kiến: 2 – 3 ngày.
Các công việc cần làm:
- [x] Huấn luyện mô hình mT5 trên tập dữ liệu đa miền (3,783 câu train, batch_size=4, grad_accum=4, 3 epochs).
- [x] Lưu checkpoint mô hình tốt nhất tại `checkpoints/best_model`.
- [x] Đánh giá kết quả trên tập test.txt đa miền (1,660 câu):
  - [x] Thu thập bảng kết quả: Precision 66.27% | Recall 56.38% | F1-Score 60.93%.
  - [x] Phân tích lỗi thực tế (Qualitative Error Analysis: hiện tượng lệch ranh giới cụm từ như `backlit keyboard` vs `keyboard`).

📌 GIAI ĐOÀN 4: Hoàn thành Sản phẩm Báo cáo Giữa kỳ (Midterm Deliverables)
Thời gian dự kiến: 1 – 2 ngày.
Các sản phẩm cần nộp:
- [x] Bộ Slide báo cáo (Chuẩn 7 slide, nằm trong khoảng 5 đến 8 slide theo quy định):
  - [x] Slide 1: Giới thiệu đề tài & Bản chất bài toán ABSA.
  - [x] Slide 2: Cấu trúc tập dữ liệu đa miền đã trộn (Thống kê số lượng mẫu, biểu đồ nhãn).
  - [x] Slide 3: Kiến trúc mô hình mT5 (Trình bày cơ chế Text-to-Text sinh chuỗi).
  - [x] Slide 4: Cơ chế Multi-task Learning (Full Parameter Sharing: Encoder-Decoder).
  - [x] Slide 5: Kỹ thuật Transfer Learning & Thích ứng miền (AdamW, Linear Warmup, Gradient Accumulation).
  - [x] Slide 6: Kết quả thực nghiệm và nhận xét hiệu năng (F1 ~61%, Loss curve, Phân tích lỗi).
  - [x] Slide 7: Giao diện trực quan hóa (Web UI Streamlit) & Định hướng cuối kỳ (Đa ngôn ngữ).
  - [x] Xuất tài liệu slide hoàn chỉnh kèm Lời thoại thuyết trình (Speaker Notes) tại `docs/noi_dung_slide_bao_cao_giua_ky.md`.
- [x] Chuẩn bị Source Code sạch:
  - [x] Đóng gói các file mã nguồn rõ ràng trong `src/` (`train.py`, `evaluate_model.py`, `dataset.py`, `metrics.py`, `build_multi_domain.py`) và ứng dụng Web UI `app.py`.

📌 CẢI TIẾN HỌC THUẬT & TRUNG THỰC HÓA SỐ LIỆU (BƯỚC 1 & BƯỚC 2):
Thời gian thực hiện: Hoàn tất trước báo cáo giữa kỳ.
Các công việc đã thực hiện:
- [x] BƯỚC 1: ĐO LƯỜNG TRUNG THỰC & CHUẨN XÁC KHOA HỌC:
  - [x] Phân tích toàn diện bộ dữ liệu đa miền: 6,393 câu, 1,767 khía cạnh duy nhất, MD5 hash kiểm định toàn vẹn không trôi dạt dữ liệu.
  - [x] Đánh giá toàn bộ tập Dev (950 câu) & Test (1,660 câu) của mô hình baseline:
    - Micro F1: 60.93% (P=66.33%, R=56.34%).
    - Polarity Accuracy khi bắt đúng aspect: 84.96%.
    - Phân rã theo miền: Laptop 62.99%, Hotel 63.02%, Restaurant 56.76%.
  - [x] Phát hiện hiện tượng cốt lõi "Neutral Collapse":
    - Positive F1: 65.91% | Negative F1: 51.36% | Neutral F1: 0.00% (Gold=95, Pred=0).
    - Macro-F1 thực tế là 39.09% (do mô hình baseline bỏ quên lớp trung tính).
  - [x] Cập nhật Tab 3 trong Web UI `app.py`: Xóa bỏ hoàn toàn số liệu hardcode; nạp động kết quả thực từ `results/*.json`, bổ sung ma trận nhầm lẫn và công cụ soi lỗi chi tiết từ `tsv`.
  - [x] Đồng bộ số liệu thực vào bộ slide (`docs/slide.md`, `docs/noi_dung_slide_bao_cao_giua_ky.md`) và biểu đồ trực quan (`docs/picture/`).
- [x] BƯỚC 2: XỬ LÝ MẤT CÂN BẰNG NHÃN (IMBALANCE MITIGATION):
  - [x] Cài đặt kỹ thuật Weighted Cross-Entropy trên token cảm xúc và Oversampling theo phân bố nghịch đảo trong `src/train.py`.
  - [x] Huấn luyện thành công mô hình cân bằng trọng số `weighted_loss_s42` (3 epoch, 32.9 phút, lưu tại `checkpoints/weighted_loss_s42/best_model`).
  - [x] Đánh giá toàn diện trên Dev (950 câu) & Test (1,660 câu):
    - **Neutral F1 trên Dev:** Tăng từ **0.00% ➔ 21.05%** (Macro-F1 dev tăng từ 37.46% ➔ **41.56%**).
    - **Neutral F1 trên Test:** Tăng từ **0.00% ➔ 16.83%** (Neutral Recall tăng từ **0.00% ➔ 17.89%** với 107 cặp neutral được dự đoán).
    - **Neutral F1 theo từng miền:** Hotel đạt 17.65%, Laptop đạt 15.24%, Restaurant đạt 19.05% (toàn bộ trước đó đều 0.00%).
    - **Test Macro-F1:** Tăng từ **39.09% ➔ 40.02%**.
  - [x] Xuất báo cáo so sánh chi tiết và tích hợp trực tiếp vào Tab 3 Web UI Streamlit.
- [x] HOÀN THIỆN HỆ THỐNG TIẾNG ANH ĐA MIỀN (HỘI TỤ 8 EPOCH & KIỂM THỬ):
  - [x] Sửa toàn bộ lỗi logic trong `src/dataset.py`, `src/train.py`, `src/run_ablation.py`, `app.py` và đồng bộ `README.md`.
  - [x] Huấn luyện hội tụ 8 epoch cho `baseline_e8_s42` (Test Micro-F1 = 64.55%, Macro-F1 = 46.43%) và `wl_p03_e8_s42` (Test Micro-F1 = 64.61%, Macro-F1 = 52.23%, Neutral F1 = 30.51%, Polarity Acc = 87.72%).
  - [x] Kiểm thử thành công trên 20 câu đánh giá tiếng Anh tự nhiên (`src/smoke_test_en.py`).
  - [x] Cập nhật tự động biểu đồ từ `results/*.json` (`docs/picture/generate_slide_charts.py`) và đồng bộ Slide/README.
📌 GIAI ĐOÀN 5: Mở rộng Cuối kỳ (Final Term Scope - Đa ngôn ngữ)
Thời gian dự kiến: Thực hiện sau khi kết thúc báo cáo giữa kỳ.
Thuận lợi lớn: Codebase M-ABSA và mô hình mT5 vốn được xây dựng cho 21 ngôn ngữ.
Nội dung thực hiện:
Đổi tập dữ liệu huấn luyện hoặc kiểm thử sang dạng Đa ngôn ngữ (Ví dụ: Train trên tiếng Anh, Test trên tiếng Việt vi, hoặc Train hỗn hợp đa ngôn ngữ).
Đánh giá tính khái quát hóa và khả năng chuyển giao xuyên ngôn ngữ (Cross-lingual Transfer).
So sánh mô hình baseline mT5 với các LLM (LLaMA, Gemma) thông qua các file có sẵn trong eval_llm/.
Hoàn thiện Technical Report và Báo cáo tổng kết đồ án.
