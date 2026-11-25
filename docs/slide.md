# BỘ SLIDE BÁO CÁO GIỮA KỲ (CHUẨN 7 SLIDE)
**ĐỒ ÁN:** Phân tích Quan điểm theo Khía cạnh (Aspect-Based Sentiment Analysis - ABSA)  
**YÊU CẦU BẮT BUỘC TRÊN SLIDE (Theo quy định Giữa kỳ):**
1. **Giới thiệu tập dữ liệu (cách trộn dữ liệu đa miền).**
2. **Giới thiệu mô hình sử dụng.**
3. **Kỹ thuật Transfer Learning đã áp dụng để thích ứng miền.**
4. **Cơ chế Multi-task Learning đã áp dụng (cách xác định và thực hiện các tác vụ đồng thời).**

---

## 📌 BẢNG ÁNH XẠ 7 SLIDE VỚI YÊU CẦU & HÌNH ẢNH MINH HỌA

| Slide | Nội dung chuẩn mực | Thuộc yêu cầu bắt buộc | Hình ảnh đính kèm (`docs/picture/`) |
| :---: | :--- | :--- | :--- |
| **Slide 1** | Giới thiệu Đề tài & Bản chất Bài toán ABSA Đa tác vụ | *Mục tiêu lõi đề tài* | `slide1_absa_concept.png` |
| **Slide 2** | Giới thiệu Tập Dữ liệu & Quy trình Trộn Dữ liệu Đa miền | **Yêu cầu 1 (Tập dữ liệu & Cách trộn)** | `slide2_data_mixing_pipeline.png`<br>`slide2_domain_distribution.png`<br>`slide2_sentiment_donut.png` |
| **Slide 3** | Giới thiệu Kiến trúc Mô hình Sử dụng (`google/mt5-small`) | **Yêu cầu 2 (Mô hình sử dụng)** | `slide3_model_architecture.png` |
| **Slide 4** | Cơ chế Multi-Task Learning (Xác định & Thực hiện đồng thời) | **Yêu cầu 4 (Cơ chế Multi-task)** | `slide4_multitask_uabsa.png` |
| **Slide 5** | Kỹ thuật Transfer Learning & Thích ứng Đa miền | **Yêu cầu 3 (Transfer Learning)** | `slide5_transfer_learning.png` |
| **Slide 6** | Kết quả Thực nghiệm, Đối sánh Cân bằng nhãn & Phân tích Lỗi | *Thực nghiệm & Cải tiến học thuật* | `slide6_training_metrics.png`<br>`slide6_imbalance_comparison.png`<br>`slide6_error_analysis.png` |
| **Slide 7** | Giao diện Trực quan hóa (Streamlit UI) & Định hướng Cuối kỳ | *Sản phẩm & Định hướng mở rộng* | *(Giao diện demo trực tiếp)* |

---

# SLIDE 1: GIỚI THIỆU ĐỀ TÀI & BẢN CHẤT BÀI TOÁN ABSA

### 🎯 Tiêu đề Slide:
**PHÂN TÍCH QUAN ĐIỂM THEO KHÍA CẠNH (ASPECT-BASED SENTIMENT ANALYSIS - ABSA)**  
*Phương pháp Học chuyển giao Đa tác vụ trên Dữ liệu Đa miền*

### 📋 Nội dung hiển thị trên Slide:
* **Hình ảnh đính kèm:** ![So sánh Cấp câu vs ABSA](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide1_absa_concept.png)
* **Hạn chế của phân loại cảm xúc truyền thống (Sentence/Document-level):**
  * Chỉ gán một nhãn duy nhất cho cả câu, dễ bỏ sót hoặc sai lệch khi câu chứa nhiều quan điểm trái chiều.
  * *Ví dụ:* *"The food was absolutely amazing, but the service was terrible and slow."* $\rightarrow$ Gán nhãn `Negative` hoặc `Mixed` sẽ làm mất hoàn toàn lời khen về món ăn.
* **Mục tiêu cốt lõi của bài toán ABSA (Fine-grained NLP):**
  * Đi sâu vào cấp độ khía cạnh chi tiết, bóc tách chính xác từng thực thể và sắc thái cảm xúc tương ứng.
* **Giải quyết đồng thời 2 tác vụ trong một mô hình duy nhất (Unified Multi-Task UABSA):**
  1. **Aspect Extraction (AE - Trích xuất khía cạnh):** Xác định ranh giới từ của khía cạnh trong câu (hoặc nhận diện khía cạnh ẩn `NULL`).
  2. **Aspect Sentiment Classification (ASC - Phân loại cảm xúc):** Xác định cực tính cảm xúc (`positive`, `negative`, `neutral`) gắn liền với đúng khía cạnh đó.
* **Ba trụ cột kỹ thuật trọng tâm báo cáo Giữa kỳ:**
  * **Đa miền (Multi-domain):** Trộn dữ liệu từ 3 lĩnh vực khác nhau (Hotel, Laptop, Restaurant).
  * **Mô hình tiền huấn luyện (Pre-trained LM):** Sử dụng `google/mt5-small` đa ngôn ngữ.
  * **Học đa tác vụ (Multi-task Learning):** Chia sẻ tham số toàn phần và tối ưu hóa hàm mất mát chung.

---

# SLIDE 2: GIỚI THIỆU TẬP DỮ LIỆU & CÁCH TRỘN DỮ LIỆU ĐA MIỀN
*(Đáp ứng Yêu cầu bắt buộc 1: Giới thiệu tập dữ liệu & cách trộn dữ liệu đa miền)*

### 🎯 Tiêu đề Slide:
**CẤU TRÚC TẬP DỮ LIỆU ĐA MIỀN & QUY TRÌNH TRỘN DỮ LIỆU (MULTI-DOMAIN DATASET)**

### 📋 Nội dung hiển thị trên Slide:
* **Hình ảnh đính kèm:**
  * Sơ đồ quy trình trộn: ![Quy trình trộn dữ liệu](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide2_data_mixing_pipeline.png)
  * Biểu đồ phân bố miền: ![Phân bố 3 miền](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide2_domain_distribution.png)
  * Biểu đồ phân bố cảm xúc: ![Phân bố cảm xúc](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide2_sentiment_donut.png)
* **Ba miền dữ liệu kết hợp (Nguồn: Benchmark M-ABSA / SemEval tiếng Anh):**
  * 🏨 **Hotel (2,147 câu):** Từ vựng về tiện nghi, vị trí, phòng ốc (`room`, `location`, `staff`, `bathroom`).
  * 💻 **Laptop (2,122 câu):** Từ vựng phần cứng, kỹ thuật (`battery life`, `screen`, `keyboard`, `performance`).
  * 🍽️ **Restaurant (2,124 câu):** Từ vựng ẩm thực, phục vụ (`food`, `service`, `portions`, `atmosphere`).
* **Cách thức trộn dữ liệu đa miền (Triển khai tại `src/build_multi_domain.py`):**
  1. **Trộn riêng biệt theo từng split (Train / Dev / Test):** Tránh hiện tượng rò rỉ dữ liệu (Data Leakage) giữa các tập.
  2. **Xáo trộn ngẫu nhiên có kiểm soát:** Sử dụng bộ sinh số ngẫu nhiên với `seed = 42` cố định để đảm bảo kết quả tái lập 100%.
  3. **Sinh siêu dữ liệu theo dõi miền (`{split}_domains.txt`):** Ghi nhớ nguồn gốc miền của từng câu, cho phép đánh giá phân rã hiệu năng theo từng miền độc lập.
  4. **Tỷ lệ cân bằng hoàn hảo:** Tỷ lệ số câu đóng góp đạt **$1 : 1 : 1$** (mỗi miền $\approx 33.3\%$), triệt tiêu thiên vị miền (Domain Bias).
* **Bảng thống kê số liệu sau khi trộn (`data/multi_domain/`):**

| Tập phân chia (Split) | Miền Hotel | Miền Laptop | Miền Restaurant | **Tổng số câu** | **Số cặp `(aspect, polarity)` chuẩn UABSA** | Tỷ lệ Cảm xúc (Pos / Neg / Neu) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 1,255 | 1,264 | 1,264 | **3,783** | **4,583** *(+1 conflict)* | 3,145 / 1,229 / 208 *(68.6% / 26.8% / 4.5%)* |
| **DEV** | 308 | 326 | 316 | **950** | **1,115** | 743 / 310 / 62 *(66.6% / 27.8% / 5.6%)* |
| **TEST** | 584 | 532 | 544 | **1,660** | **2,185** | 1,579 / 511 / 95 *(72.3% / 23.4% / 4.3%)* |
| **TỔNG CỘNG** | **2,147** | **2,122** | **2,124** | **6,393** | **7,883** | **5,467 / 2,050 / 365** *(69.4% / 26.0% / 4.6%)* |

* **Đặc tính kỹ thuật & Thách thức học máy:**
  * Dữ liệu bao phủ **1,767 khía cạnh duy nhất** xuyên suốt 3 miền.
  * *Lưu ý về quy đổi dữ liệu:* Tập Test có 2,531 nhãn bộ ba thô; sau khi chuẩn hóa sang tác vụ UABSA và khử trùng lặp cặp trong cùng một câu, tập Test có đúng **2,185 cặp chuẩn**.
  * **Thách thức mất cân bằng nhãn nặng:** Lớp `Neutral` chỉ chiếm **~4.0% - 4.5%** dữ liệu (tỷ lệ lệch 15:1 so với Positive), đòi hỏi giải pháp kỹ thuật can thiệp trọng số loss.

---

# SLIDE 3: GIỚI THIỆU MÔ HÌNH SỬ DỤNG (mT5 SEQ2SEQ)
*(Đáp ứng Yêu cầu bắt buộc 2: Giới thiệu mô hình sử dụng)*

### 🎯 Tiêu đề Slide:
**KIẾN TRÚC MÔ HÌNH: mT5 (MULTILINGUAL TEXT-TO-TEXT TRANSFER TRANSFORMER)**

### 📋 Nội dung hiển thị trên Slide:
* **Hình ảnh đính kèm:** ![Kiến trúc mô hình mT5](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide3_model_architecture.png)
* **Mô hình lựa chọn:** `google/mt5-small` (~300 triệu tham số, cấu trúc Transformer Encoder-Decoder tiêu chuẩn).
  * **Từ điển đa ngôn ngữ:** SentencePiece với 250,100 subword tokens, tiền huấn luyện trên 101 ngôn ngữ từ kho ngữ liệu mC4.
* **Định dạng bài toán dạng Sinh chuỗi (Generative Text-to-Text Framework):**
  * Toàn bộ bài toán ABSA được quy về một quy trình Sequence-to-Sequence đồng nhất:
    * **Đầu vào (Source Text $X$):** Câu đánh giá thô của khách hàng ($L_{\max} = 128$ tokens).
    * **Đầu ra mục tiêu (Target Text $Y$):** Chuỗi chuẩn hóa các cặp khía cạnh – cảm xúc phân tách bằng dấu chấm phẩy ($L_{\text{target}} = 64$ tokens):
      $$\text{Target } Y = \text{``(aspect}_1, \text{polarity}_1\text{); (aspect}_2, \text{polarity}_2\text{)''}$$
  * *Ví dụ minh họa:*
    * Đầu vào: *"The food was amazing, but the service was slow."*
    * Đầu ra: `"(food, positive); (service, negative)"`
* **3 Lý do cốt lõi lựa chọn kiến trúc mT5 Seq2Seq:**
  1. **Xử lý tự nhiên khía cạnh ẩn (`NULL` aspect):** Trong dữ liệu có tới **2,072 / 9,210 (~22.5%)** nhãn khía cạnh ẩn (ví dụ câu không nêu tên đối tượng cụ thể nhưng mang sắc thái chung: *"I'd definitely stay there again."* $\rightarrow$ `(NULL, positive)`). Mô hình gán nhãn token tuần tự (như BIO của BERT) hoàn toàn thất bại vì không có từ trong câu để gắn nhãn, trong khi mT5 tự động sinh token `NULL`.
  2. **Trích xuất cụm khía cạnh có độ dài linh hoạt:** Hỗ trợ từ đơn (`screen`), cụm từ dài (`thermaltake auto laptop cooling pad`), và câu có nhiều khía cạnh cùng lúc.
  3. **Tương thích trực tiếp cho Giai đoạn Cuối kỳ (Đa ngôn ngữ):** Giữ nguyên 100% luồng xử lý kỹ thuật khi chuyển sang tiếng Việt (`vi`) hoặc đa ngôn ngữ.

---

# SLIDE 4: CƠ CHẾ MULTI-TASK LEARNING ĐÃ ÁP DỤNG
*(Đáp ứng Yêu cầu bắt buộc 4: Cơ chế Multi-task Learning - cách xác định và thực hiện đồng thời)*

### 🎯 Tiêu đề Slide:
**CƠ CHẾ HỌC ĐA TÁC VỤ (MULTI-TASK PARAMETER SHARING) & HÀM MẤT MÁT CÂN BẰNG**

### 📋 Nội dung hiển thị trên Slide:
* **Hình ảnh đính kèm:** ![Cơ chế Multi-task UABSA](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide4_multitask_uabsa.png)
* **1. Cách xác định 2 tác vụ thành phần:**
  * **Tác vụ 1 (Aspect Extraction - AE):** Định vị và nhận diện ranh giới từ ngữ của khía cạnh trong câu (hoặc sinh token `NULL`).
  * **Tác vụ 2 (Aspect Sentiment Classification - ASC):** Phân loại cực tính cảm xúc (`positive`, `negative`, `neutral`) tương ứng với khía cạnh đó.
* **2. Cách thực hiện đồng thời trong một mô hình duy nhất:**
  * **Chia sẻ tham số toàn phần (Hard / Full Parameter Sharing):**
    * Toàn bộ các tầng **Encoder** và **Decoder** của mT5 được chia sẻ 100% tham số cho cả 2 tác vụ (không tách 2 head độc lập, không dùng pipeline 2 giai đoạn).
    * Loại bỏ hoàn toàn hiện tượng **lan truyền lỗi tích lũy (Error Propagation)** vốn là nhược điểm của mô hình tách rời (AE sai kéo theo ASC sai).
  * **Cơ chế Tự hồi quy có điều kiện (Conditional Autoregressive Decoding):**
    * Decoder sinh lần lượt: `token khía cạnh` $\rightarrow$ dấu phẩy $\rightarrow$ `token cảm xúc`.
    * Nhờ cơ chế **Self-Attention** và **Cross-Attention**, xác suất sinh từ cảm xúc được đặt điều kiện trực tiếp lên khía cạnh vừa sinh trước đó ($P(\text{polarity} \mid \text{aspect}, X)$).
* **3. Cải tiến kỹ thuật: Hàm mất mát Token-level Weighted Cross-Entropy:**
  * **Vấn đề:** Do lớp `Neutral` chỉ chiếm 4.5%, hàm Cross-Entropy thông thường dẫn tới **Neutral Collapse (mô hình đoán 0 mẫu trung tính)**.
  * **Giải pháp (`src/train.py`):** Gán trọng số nghịch đảo tần suất lớp $w_c = (N_{\max} / N_c)^{0.5}$ vào đúng các token cảm xúc đặc trưng trong từ điển mT5, giữ nguyên trọng số 1.0 cho các token khía cạnh:
    $$\mathcal{L}_{\text{weighted}} = - \sum_{t=1}^{T} w(y_t) \cdot \log P(y_t \mid y_{<t}, X; \theta), \quad \begin{cases} w(\text{positive}) = 1.00 \\ w(\text{negative}) = 1.60 \\ w(\text{neutral}) = \mathbf{3.89} \end{cases}$$

---

# SLIDE 5: KỸ THUẬT TRANSFER LEARNING & THÍCH ỨNG MIỀN
*(Đáp ứng Yêu cầu bắt buộc 3: Kỹ thuật Transfer Learning đã áp dụng để thích ứng miền)*

### 🎯 Tiêu đề Slide:
**KỸ THUẬT TRANSFER LEARNING & THÍCH ỨNG ĐA MIỀN (DOMAIN ADAPTATION)**

### 📋 Nội dung hiển thị trên Slide:
* **Hình ảnh đính kèm:** ![Quy trình Transfer Learning](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide5_transfer_learning.png)
* **Quy trình Học chuyển giao 2 Giai đoạn (Transfer Learning Pipeline):**
  1. **Pre-trained Foundation (Tri thức nền tảng):** Khởi tạo từ `google/mt5-small` đã học sẵn tri thức ngôn ngữ, ngữ pháp và biểu diễn ngữ nghĩa tổng quát từ kho ngữ liệu khổng lồ mC4 (101 ngôn ngữ).
  2. **Multi-Domain Fine-tuning (Thích ứng đa miền đồng thời):** Tinh chỉnh toàn bộ mô hình (End-to-End Fine-tuning) trên tập dữ liệu đa miền trộn lẫn (3,783 câu từ `Hotel + Laptop + Restaurant`).
     * Mô hình tận dụng sự giao thoa ngữ nghĩa: Các khía cạnh chung xuyên miền (*price, service, staff, quality*) củng cố biểu diễn tổng quát; đồng thời Encoder học phân biệt các khía cạnh đặc thù từng miền dựa trên ngữ cảnh xung quanh (*battery, ram* của Laptop vs *room, bed* của Hotel vs *food, drink* của Restaurant).
* **Chiến lược Huấn luyện & Tối ưu hóa Phần cứng (Triển khai tại `src/train.py`, `src/engine.py`):**

| Thành phần kỹ thuật | Thiết lập cấu hình | Ý nghĩa & Giải thích chi tiết |
| :--- | :--- | :--- |
| **Optimizer** | `AdamW` ($\text{lr} = 3 \times 10^{-4}$, $\text{weight\_decay} = 0.01$) | Tách nhóm tham số: không áp dụng suy giảm trọng số lên `bias` và `LayerNorm`. |
| **Learning Rate Schedule** | `Linear Warmup` (10% tổng bước) + `Linear Decay` | Tránh gây sốc gradient ở đầu quá trình học và giúp mô hình hội tụ mượt mà. |
| **Gradient Clipping** | `clip_grad_norm = 1.0` | Ổn định gradient, chống bùng nổ gradient trên kiến trúc mT5. |
| **Gradient Accumulation** | `batch_size = 4` $\times$ `grad_accum = 4` $\rightarrow$ **Effective Batch = 16** | Đảm bảo kích thước batch hiệu dụng tối ưu trên **GPU cá nhân GTX 1650 (4GB VRAM)**. |
| **Dynamic Batch Trimming** | Hàm `trim_batch()` + chế độ **FP32 Stable** | Cắt bỏ padding dư thừa theo mẫu dài nhất mỗi batch (tăng tốc ~30%); tắt FP16 để tránh lỗi tràn số `NaN`. |
| **Đánh giá & Chọn Checkpoint** | Đánh giá trên **toàn bộ 950 câu Dev** mỗi epoch | Tính Validation Loss thực tế và lưu `best_model` theo Macro-F1 để tối ưu toàn diện. |

---

# SLIDE 6: KẾT QUẢ THỰC NGHIỆM, CÂN BẰNG NHÃN & PHÂN TÍCH LỖI

### 🎯 Tiêu đề Slide:
**KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH, KHẮC PHỤC NEUTRAL COLLAPSE & PHÂN TÍCH LỖI**

### 📋 Nội dung hiển thị trên Slide:
* **Hình ảnh đính kèm:**
  * Tiến trình hội tụ & kết quả Test: ![Tiến trình hội tụ](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide6_training_metrics.png)
  * Biểu đồ đối sánh khắc phục mất cân bằng: ![Khắc phục mất cân bằng](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide6_imbalance_comparison.png)
  * Minh họa phân tích lỗi: ![Phân tích lỗi](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide6_error_analysis.png)
* **1. Bảng Kết quả Thực nghiệm Đối sánh trên Tập Test Tiếng Anh Đa Miền (1,660 câu - 2,185 cặp chuẩn):**

| Cấu hình mô hình | Micro-P (%) | Micro-R (%) | **Micro-F1 (%)** | **Macro-F1 (%)** | F1 `Positive` | F1 `Negative` | **F1 `Neutral`** | Aspect F1 | Polarity Acc* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 3ep (CE chuẩn)** | 66.33 | 56.34 | 60.93 | 39.09 | 65.91 | 51.36 | **0.00** *(0/95)* | 70.77 | 84.96% |
| **Baseline 8ep (Hội tụ)** | 66.46 | 62.75 | 64.55 | 46.43 | 68.48 | **57.96** | 12.84 *(7/95)* | **73.07** | 87.55% |
| **Weighted Loss (`p=0.3`, 8ep — Tốt nhất)** | **66.54** | **62.79** | **64.61** | **52.23** | **68.84** | 57.35 | **30.51** *(27/95)* | 73.05 | **87.72%** |

*(Chú thích *: Độ chính xác phân loại cảm xúc tính trên các khía cạnh đã trích xuất đúng ranh giới từ. Trên Dev 950 câu, `wl_p03_e8_s42` đạt Micro-F1 = **64.41%**, Macro-F1 = **54.56%**, Neutral F1 = **36.04%**).*

* **2. Phân rã Hiệu năng theo từng Miền trên Tập Test (1,660 câu — `wl_p03_e8_s42`):**
  * **Hotel (584 câu):** Micro-F1 = **65.14%** | Macro-F1 = **47.24%** | Neutral F1 = **23.53%** *(Baseline 3ep: 0.00%)*.
  * **Laptop (532 câu):** Micro-F1 = **65.12%** | Macro-F1 = **55.01%** | Neutral F1 = **31.46%** *(Baseline 3ep: 0.00%)*.
  * **Restaurant (544 câu):** Micro-F1 = **63.57%** | Macro-F1 = **51.84%** | Neutral F1 = **33.33%** *(Baseline 3ep: 0.00%)*.
  * 👉 **Đột phá không đánh đổi:** Khi huấn luyện hội tụ 8 epoch kết hợp Weighted Loss nhẹ (`power=0.3`, $w_{\text{pos}}=1.00, w_{\text{neg}}=1.33, w_{\text{neu}}=2.26$), mô hình **tăng đồng thời cả Micro-F1 (64.61%) lẫn Macro-F1 (52.23%, +13.14% so với baseline ban đầu)**!
* **3. Phân tích Lỗi Định tính Thực tế (Qualitative Error Analysis):**
  * **Ca 1 (Khớp tuyệt đối 100%):** *"it works well as a tablet ."* $\rightarrow$ Gold: `(tablet, positive)` | Pred: `(tablet, positive)`.
  * **Ca 2 (Khôi phục Neutral thành công - Hotel #388):** *"Rooms are on the basic side , but they have everything that you ' d need ."* $\rightarrow$ Gold: `(Rooms, neutral)` | Baseline đoán sai thành `(Rooms, positive)` | **Weighted Loss đoán đúng `(Rooms, neutral)`**.
  * **Ca 3 (Lệch ranh giới cụm từ - Laptop #4):** *"no backlit keyboard ."* $\rightarrow$ Gold: `(keyboard, negative)` vs Pred: `(backlit keyboard, negative)` *(Bị trừ điểm do cơ chế Exact Match khắt khe, nhưng dự đoán của mô hình thực tế còn chi tiết và cụ thể hơn nhãn gốc)*.

---

# SLIDE 7: GIAO DIỆN TRỰC QUAN HÓA & ĐỊNH HƯỚNG CUỐI KỲ

### 🎯 Tiêu đề Slide:
**GIAO DIỆN TRỰC QUAN (STREAMLIT WEB UI) & ĐỊNH HƯỚNG CUỐI KỲ**

### 📋 Nội dung hiển thị trên Slide:
* **Hệ thống Giao diện Web UI Trực quan (`app.py` - Chạy lệnh `streamlit run app.py`):**
  * **Tab 1 – Live Demo (Suy luận thời gian thực):**
    * Cho phép chuyển đổi linh hoạt giữa checkpoint **Weighted Loss** và **Baseline**.
    * Nhập câu đánh giá bất kỳ $\rightarrow$ Bóc tách và tô màu trực quan (🟢 Xanh: `positive`, 🔴 Đỏ: `negative`, 🟡 Vàng: `neutral`).
  * **Tab 2 – Dataset Explorer (Khám phá dữ liệu đa miền):**
    * Trực quan hóa số lượng câu, phân bố khía cạnh, biểu đồ cảm xúc xuyên suốt các miền và các ngôn ngữ.
  * **Tab 3 – Evaluation Dashboard (Đo lường trung thực & Phân tích lỗi):**
    * Nạp động dữ liệu đo lường thực tế từ `results/*.json` (Micro/Macro F1, bảng Per-class, Domain breakdown).
    * Hiển thị ma trận nhầm lẫn cảm xúc (Confusion Matrix) và bộ lọc soi lỗi từng câu từ file dự đoán `.tsv`.
* **Kế hoạch Mở rộng Giai đoạn Cuối kỳ (Final Term Scope - Đa ngôn ngữ):**
  * **Kế thừa 100% luồng kỹ thuật:** Mô hình `mT5` và pipeline trong `src/` vốn được thiết kế đa ngôn ngữ, hỗ trợ sẵn tham số `--lang` cho 21 ngôn ngữ có sẵn trong kho dữ liệu M-ABSA.
  * **Các mục tiêu thực nghiệm cuối kỳ:**
    1. Huấn luyện và đánh giá trên tiếng Việt (`vi`) và các ngôn ngữ khác (`fr`, `zh`, `ja`...).
    2. Đánh giá năng lực chuyển giao xuyên ngôn ngữ không cần huấn luyện (Zero-shot Cross-lingual Transfer: Train tiếng Anh $\rightarrow$ Test tiếng Việt).
    3. Thực hiện các thí nghiệm Ablation đối sánh Đơn miền (Single-domain) vs Đa miền (Multi-domain) và so chuẩn với các LLMs (LLaMA, Gemma).
