# TÀI LIỆU BÁO CÁO GIỮA KỲ: NỘI DUNG 7 SLIDE, LỜI THOẠI & KỊCH BẢN GIẢI THÍCH CODE
**ĐỒ ÁN:** Phân tích Quan điểm theo Khía cạnh (Aspect-Based Sentiment Analysis - ABSA)  
**YÊU CẦU BẮT BUỘC TRÊN SLIDE (Theo quy định Giữa kỳ trong [requirement.md](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/requirement.md)):**
1. **Giới thiệu tập dữ liệu (cách trộn dữ liệu đa miền).**
2. **Giới thiệu mô hình sử dụng.**
3. **Kỹ thuật Transfer Learning đã áp dụng để thích ứng miền.**
4. **Cơ chế Multi-task Learning đã áp dụng (cách xác định và thực hiện các tác vụ đồng thời).**
5. **Kèm theo: Chiếu và giải thích source code trực tiếp ngay sau khi trình bày slide.**

---

## 📌 BẢNG ÁNH XẠ 7 SLIDE VỚI YÊU CẦU, HÌNH ẢNH & FILE CODE ĐỐI ỨNG

| Slide | Tên Slide & Chủ đề | Yêu cầu bắt buộc tương ứng | Hình ảnh đính kèm (`docs/picture/`) | File Source Code đối ứng khi Demo |
| :---: | :--- | :--- | :--- | :--- |
| **Slide 1** | Giới thiệu Đề tài & Bản chất Bài toán ABSA | *Mục tiêu lõi đề tài* | `slide1_absa_concept.png` | `src/dataset.py` |
| **Slide 2** | Giới thiệu Tập Dữ liệu & Cách Trộn Dữ liệu Đa miền | **Yêu cầu 1: Dữ liệu & Cách trộn** | `slide2_data_mixing_pipeline.png`<br>`slide2_domain_distribution.png`<br>`slide2_sentiment_donut.png` | `src/build_multi_domain.py`<br>`src/analyze_data.py` |
| **Slide 3** | Giới thiệu Kiến trúc Mô hình Sử dụng (`mT5`) | **Yêu cầu 2: Mô hình sử dụng** | `slide3_model_architecture.png` | `src/dataset.py`<br>`src/engine.py` |
| **Slide 4** | Cơ chế Multi-Task Learning (AE + ASC) & Weighted Loss | **Yêu cầu 4: Cơ chế Multi-task** | `slide4_multitask_uabsa.png` | `src/train.py` (dòng 72–145, 185–215) |
| **Slide 5** | Kỹ thuật Transfer Learning & Thích ứng Đa miền | **Yêu cầu 3: Transfer Learning** | `slide5_transfer_learning.png` | `src/train.py` (dòng 150–250)<br>`src/engine.py` |
| **Slide 6** | Kết quả Thực nghiệm, Đối sánh Cân bằng nhãn & Soi lỗi | *Thực nghiệm & Cải tiến khoa học* | `slide6_training_metrics.png`<br>`slide6_imbalance_comparison.png`<br>`slide6_error_analysis.png` | `src/evaluate_model.py`<br>`src/metrics.py` |
| **Slide 7** | Giao diện Trực quan hóa (Streamlit) & Định hướng Cuối kỳ | *Chuẩn đầu ra & Hướng mở rộng* | *(Giao diện demo trực tiếp)* | `app.py` |

---

# SLIDE 1: GIỚI THIỆU ĐỀ TÀI & BẢN CHẤT BÀI TOÁN ABSA

### 🎯 Tiêu đề Slide:
**PHÂN TÍCH QUAN ĐIỂM THEO KHÍA CẠNH (ASPECT-BASED SENTIMENT ANALYSIS - ABSA)**  
*Phương pháp Học chuyển giao Đa tác vụ trên Dữ liệu Đa miền*

### 📋 Nội dung hiển thị trên Slide:
* **Hình ảnh minh họa chèn vào Slide:** ![So sánh Cấp câu vs ABSA](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide1_absa_concept.png)
* **Hạn chế của phân loại cảm xúc truyền thống (Sentence/Document-level):**
  * Chỉ gán một nhãn duy nhất cho cả câu, dễ bỏ sót hoặc sai lệch khi câu chứa nhiều quan điểm trái chiều.
  * *Ví dụ:* *"The food was absolutely amazing, but the service was terrible and slow."* $\rightarrow$ Gán nhãn `Negative` hoặc `Mixed` sẽ làm mất hoàn toàn lời khen về món ăn.
* **Mục tiêu cốt lõi của ABSA (Fine-grained Sentiment Analysis):**
  * Bóc tách thông tin ở cấp độ thực thể/khía cạnh cụ thể, giải quyết đồng thời 2 tác vụ trong một mô hình duy nhất (**Unified Multi-Task UABSA**):
    1. **Aspect Extraction (AE - Trích xuất khía cạnh):** Nhận diện ranh giới từ ngữ của khía cạnh trong câu (hoặc nhận diện khía cạnh ẩn `NULL`).
    2. **Aspect Sentiment Classification (ASC - Phân loại cảm xúc):** Xác định cực tính cảm xúc (`positive`, `negative`, `neutral`) gắn liền với đúng khía cạnh đó.
* **Ba trụ cột kỹ thuật trọng tâm báo cáo Giữa kỳ:**
  1. **Dữ liệu Đa miền (Multi-domain):** Kết hợp cân bằng 3 miền Hotel + Laptop + Restaurant.
  2. **Học chuyển giao (Transfer Learning):** Tinh chỉnh (Fine-tuning) mô hình tiền huấn luyện `google/mt5-small`.
  3. **Học đa tác vụ (Multi-task Learning):** Chia sẻ tham số toàn phần và tối ưu hóa hàm mất mát cân bằng trọng số lớp (**Token-level Weighted Cross-Entropy**).

### 🗣️ Lời thoại gợi ý (Speaker Notes):
> *"Kính thưa cô và các bạn, hôm nay nhóm xin báo cáo tiến độ giữa kỳ đề tài Phân tích quan điểm theo khía cạnh (ABSA). Khác với phân loại cảm xúc cấp câu truyền thống vốn chỉ gán một nhãn chung chung cho cả câu, bài toán ABSA yêu cầu một mô hình duy nhất phải thực hiện đồng thời hai nhiệm vụ: vừa trích xuất khía cạnh được nhắc tới, vừa xác định cảm xúc khen, chê hay trung tính trên chính khía cạnh đó. Ở giai đoạn giữa kỳ, nhóm tập trung giải quyết trọn vẹn các yêu cầu kỹ thuật: xây dựng tập dữ liệu đa miền từ 3 lĩnh vực, áp dụng Transfer Learning với mô hình tiền huấn luyện mT5, và thiết lập cơ chế Multi-task chia sẻ tham số kèm kỹ thuật xử lý mất cân bằng nhãn."*

---

# SLIDE 2: GIỚI THIỆU TẬP DỮ LIỆU & CÁCH TRỘN DỮ LIỆU ĐA MIỀN
*(Đáp ứng Yêu cầu bắt buộc 1: Giới thiệu tập dữ liệu & cách trộn dữ liệu đa miền)*

### 🎯 Tiêu đề Slide:
**CẤU TRÚC TẬP DỮ LIỆU ĐA MIỀN & QUY TRÌNH TRỘN DỮ LIỆU (MULTI-DOMAIN DATASET)**

### 📋 Nội dung hiển thị trên Slide:
* **Hình ảnh biểu đồ chèn vào Slide:**
  * Sơ đồ quy trình trộn: ![Quy trình trộn dữ liệu](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide2_data_mixing_pipeline.png)
  * Biểu đồ cột phân bố 3 miền: ![Phân bố 3 miền](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide2_domain_distribution.png)
  * Biểu đồ tròn phân bố cảm xúc: ![Phân bố cảm xúc](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide2_sentiment_donut.png)
* **Ba miền dữ liệu kết hợp (Nguồn: Benchmark M-ABSA / SemEval tiếng Anh):**
  * 🏨 **Hotel (2,147 câu):** Từ vựng về tiện nghi phòng ốc, vị trí, thái độ nhân viên (`room`, `location`, `staff`, `bathroom`).
  * 💻 **Laptop (2,122 câu):** Từ vựng kỹ thuật phần cứng (`battery life`, `screen`, `keyboard`, `performance`).
  * 🍽️ **Restaurant (2,124 câu):** Từ vựng ẩm thực, khẩu phần, không gian (`food`, `service`, `portions`, `atmosphere`).
* **Quy trình và cách trộn dữ liệu đa miền (Triển khai tại `src/build_multi_domain.py`):**
  1. **Trộn riêng biệt theo từng split (Train / Dev / Test):** Đảm bảo tuyệt đối không có hiện tượng rò rỉ dữ liệu (Data Leakage) giữa các tập.
  2. **Xáo trộn ngẫu nhiên có kiểm soát:** Sử dụng thuật toán shuffle với hạt giống ngẫu nhiên `seed = 42` cố định để đảm bảo tính tái lập 100%.
  3. **Sinh siêu dữ liệu theo dõi nguồn gốc miền (`{split}_domains.txt`):** Ghi nhận nhãn miền của từng câu, cho phép đánh giá phân rã hiệu năng theo từng miền độc lập.
  4. **Tỷ lệ cân bằng hoàn hảo:** Tỷ lệ số câu đóng góp giữa 3 miền đạt **$1 : 1 : 1$** (mỗi miền $\approx 33.3\%$), triệt tiêu hoàn toàn hiện tượng thiên vị miền (Domain Bias).
* **Bảng thống kê số liệu sau khi trộn (`data/multi_domain/`):**

| Tập phân chia (Split) | Miền Hotel | Miền Laptop | Miền Restaurant | **Tổng số câu** | **Số cặp `(aspect, polarity)` chuẩn UABSA** | Phân bố Cảm xúc (Pos / Neg / Neu) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 1,255 | 1,264 | 1,264 | **3,783** | **4,583** *(+1 conflict)* | 3,145 / 1,229 / 208 *(68.6% / 26.8% / 4.5%)* |
| **DEV** | 308 | 326 | 316 | **950** | **1,115** | 743 / 310 / 62 *(66.6% / 27.8% / 5.6%)* |
| **TEST** | 584 | 532 | 544 | **1,660** | **2,185** | 1,579 / 511 / 95 *(72.3% / 23.4% / 4.3%)* |
| **TỔNG CỘNG** | **2,147** | **2,122** | **2,124** | **6,393** | **7,883** | **5,467 / 2,050 / 365** *(69.4% / 26.0% / 4.6%)* |

* **Đặc tính kỹ thuật & Thách thức học máy:**
  * Dữ liệu bao phủ **1,767 khía cạnh duy nhất** xuyên suốt 3 miền.
  * *Lưu ý về quy đổi dữ liệu:* Tập dữ liệu gốc chứa 9,210 nhãn bộ ba `(aspect, category, polarity)`; khi chuẩn hóa sang tác vụ UABSA và khử trùng lặp cặp trong cùng một câu, tập Test có đúng **2,185 cặp chuẩn**.
  * **Thách thức mất cân bằng nhãn sâu sắc (Label Imbalance):** Lớp `Neutral` chỉ chiếm **~4.0% - 4.5%** dữ liệu (tỷ lệ lệch 15:1 so với Positive), đòi hỏi giải pháp kỹ thuật can thiệp trọng số loss.

### 🗣️ Lời thoại gợi ý (Speaker Notes):
> *"Để đáp ứng yêu cầu đa miền, nhóm đã viết script build_multi_domain.py để trộn tự động dữ liệu từ 3 lĩnh vực có đặc thù từ vựng rất khác nhau: Khách sạn, Laptop và Nhà hàng. Tổng cộng dữ liệu sau khi trộn gồm 6,393 câu, phân chia rành mạch thành Train (3,783 câu), Dev (950 câu) và Test (1,660 câu). Điểm mấu chốt trong kỹ thuật trộn của nhóm là: thứ nhất, trộn độc lập trên từng split để tránh rò rỉ dữ liệu; thứ hai, xáo trộn có kiểm soát với seed cố định; thứ ba, cân đối chính xác tỷ lệ 1:1:1 giữa 3 miền để triệt tiêu thiên vị miền; và thứ tư, xuất kèm file domains.txt để theo dõi nguồn gốc miền phục vụ đánh giá phân rã sau này. Dữ liệu gốc lưu dưới dạng bộ ba aspect-category-polarity, khi chuẩn hóa sang cặp aspect-polarity của tác vụ UABSA và khử trùng lặp trong cùng một câu thì tập Test có đúng 2,185 cặp chuẩn. Một thách thức lớn mà nhóm nhận diện ngay từ đầu là lớp Neutral chỉ chiếm hơn 4%, đặt ra bài toán phải xử lý mất cân bằng nhãn."*

---

# SLIDE 3: GIỚI THIỆU MÔ HÌNH SỬ DỤNG (mT5 SEQ2SEQ)
*(Đáp ứng Yêu cầu bắt buộc 2: Giới thiệu mô hình sử dụng)*

### 🎯 Tiêu đề Slide:
**KIẾN TRÚC MÔ HÌNH: mT5 (MULTILINGUAL TEXT-TO-TEXT TRANSFER TRANSFORMER)**

### 📋 Nội dung hiển thị trên Slide:
* **Hình ảnh sơ đồ chèn vào Slide:** ![Kiến trúc mô hình mT5](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide3_model_architecture.png)
* **Mô hình nền tảng:** `google/mt5-small` (~300 triệu tham số, cấu trúc Transformer Encoder-Decoder tiêu chuẩn).
  * **Từ điển đa ngôn ngữ:** SentencePiece với 250,100 subword tokens, tiền huấn luyện trên 101 ngôn ngữ từ kho ngữ liệu khổng lồ mC4.
* **Định dạng bài toán dạng Sinh chuỗi (Generative Text-to-Text Framework):**
  * Toàn bộ bài toán ABSA phức tạp được chuẩn hóa thành một quy trình Sequence-to-Sequence đồng nhất:
    * **Đầu vào (Source Text $X$):** Câu đánh giá thô của khách hàng ($L_{\max} = 128$ tokens).
    * **Đầu ra mục tiêu (Target Text $Y$):** Chuỗi chuẩn hóa các cặp khía cạnh – cảm xúc phân tách bằng dấu chấm phẩy ($L_{\text{target}} = 64$ tokens):
      $$\text{Target } Y = \text{``(aspect}_1, \text{polarity}_1\text{); (aspect}_2, \text{polarity}_2\text{)''}$$
  * *Ví dụ minh họa luồng xử lý:*
    * Đầu vào: *"The food was amazing, but the service was slow."*
    * Đầu ra: `"(food, positive); (service, negative)"`
* **3 Lý do cốt lõi lựa chọn kiến trúc mT5 Seq2Seq:**
  1. **Xử lý tự nhiên khía cạnh ẩn (`NULL` aspect):** Trong dữ liệu có tới **2,072 / 9,210 (~22.5%)** nhãn khía cạnh ẩn (ví dụ câu không nêu tên đối tượng cụ thể nhưng mang sắc thái chung: *"I'd definitely stay there again."* $\rightarrow$ `(NULL, positive)`). Mô hình gán nhãn token tuần tự (như BIO của BERT) hoàn toàn thất bại vì không có từ trong câu để gắn nhãn, trong khi mT5 tự động sinh token `NULL`.
  2. **Trích xuất cụm khía cạnh có độ dài linh hoạt:** Hỗ trợ từ đơn (`screen`), cụm từ dài (`thermaltake auto laptop cooling pad`), và câu có nhiều khía cạnh cùng lúc.
  3. **Tương thích trực tiếp cho Giai đoạn Cuối kỳ (Đa ngôn ngữ):** Giữ nguyên 100% luồng xử lý kỹ thuật khi chuyển sang tiếng Việt (`vi`) hoặc đa ngôn ngữ.

### 🗣️ Lời thoại gợi ý (Speaker Notes):
> *"Về mô hình sử dụng, nhóm lựa chọn google/mt5-small theo hướng tiếp cận Generative Text-to-Text. Thay vì sử dụng các mô hình gán nhãn chuỗi BIO tuần tự truyền thống trên từng token, mT5 quy toàn bộ bài toán về dạng Sequence-to-Sequence: đầu vào là câu đánh giá thô, và Decoder tự động sinh ra chuỗi có cấu trúc chứa các cặp aspect và polarity. Nhóm chọn mT5 vì 3 lý do kỹ thuật quyết định: thứ nhất, trong dữ liệu có tới 22.5% mẫu chứa khía cạnh ẩn NULL, các mô hình gán nhãn theo vị trí từ không thể gán nhãn cho từ không xuất hiện, trong khi mT5 sinh trực tiếp token NULL một cách tự nhiên; thứ hai, mô hình sinh được cụm từ khía cạnh có độ dài bất kỳ; và thứ ba, kiến trúc này vốn được xây dựng cho 101 ngôn ngữ, giúp nhóm sẵn sàng kế thừa 100% cho bài toán đa ngôn ngữ ở giai đoạn cuối kỳ."*

---

# SLIDE 4: CƠ CHẾ MULTI-TASK LEARNING ĐÃ ÁP DỤNG
*(Đáp ứng Yêu cầu bắt buộc 4: Cơ chế Multi-task Learning - cách xác định và thực hiện các tác vụ đồng thời)*

### 🎯 Tiêu đề Slide:
**CƠ CHẾ HỌC ĐA TÁC VỤ (MULTI-TASK PARAMETER SHARING) & HÀM MẤT MÁT CÂN BẰNG**

### 📋 Nội dung hiển thị trên Slide:
* **Hình ảnh sơ đồ chèn vào Slide:** ![Cơ chế Multi-task UABSA](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide4_multitask_uabsa.png)
* **1. Cách xác định 2 tác vụ thành phần:**
  * **Tác vụ 1 (Aspect Extraction - AE):** Định vị và nhận diện ranh giới từ ngữ của khía cạnh trong câu (hoặc sinh token `NULL`).
  * **Tác vụ 2 (Aspect Sentiment Classification - ASC):** Phân loại cực tính cảm xúc (`positive`, `negative`, `neutral`) tương ứng với khía cạnh đó.
* **2. Cách thực hiện đồng thời trong một mô hình duy nhất:**
  * **Chia sẻ tham số toàn phần (Hard / Full Parameter Sharing):**
    * Toàn bộ các tầng **Encoder** và **Decoder** của mT5 được chia sẻ 100% tham số cho cả 2 tác vụ (không tách 2 head độc lập, không dùng pipeline 2 giai đoạn).
    * Loại bỏ hoàn toàn hiện tượng **lan truyền lỗi tích lũy (Error Propagation)** vốn là nhược điểm của mô hình tách rời (AE sai kéo theo ASC sai).
  * **Cơ chế Tự hồi quy có điều kiện (Conditional Autoregressive Decoding):**
    * Trong một lần forward duy nhất (Single Forward Pass), Decoder sinh lần lượt: `token khía cạnh` $\rightarrow$ dấu phẩy $\rightarrow$ `token cảm xúc`.
    * Nhờ cơ chế **Self-Attention** và **Cross-Attention**, xác suất sinh từ cảm xúc được đặt điều kiện trực tiếp lên khía cạnh vừa sinh trước đó ($P(\text{polarity} \mid \text{aspect}, X)$).
* **3. Cải tiến kỹ thuật: Hàm mất mát Token-level Weighted Cross-Entropy:**
  * **Vấn đề:** Do lớp `Neutral` chỉ chiếm 4.5%, hàm Cross-Entropy thông thường dẫn tới **Neutral Collapse (mô hình đoán 0 mẫu trung tính)**.
  * **Giải pháp (`src/train.py`):** Gán trọng số nghịch đảo tần suất lớp $w_c = (N_{\max} / N_c)^{0.5}$ vào đúng các token cảm xúc đặc trưng trong từ điển mT5, giữ nguyên trọng số 1.0 cho các token khía cạnh:
    $$\mathcal{L}_{\text{weighted}} = - \sum_{t=1}^{T} w(y_t) \cdot \log P(y_t \mid y_{<t}, X; \theta), \quad \begin{cases} w(\text{positive}) = 1.00 \\ w(\text{negative}) = 1.60 \\ w(\text{neutral}) = \mathbf{3.89} \end{cases}$$

### 🗣️ Lời thoại gợi ý (Speaker Notes):
> *"Tại Slide 4, nhóm giải thích rõ cơ chế Multi-task Learning theo đúng yêu cầu: Thứ nhất, về cách xác định: bài toán gồm 2 tác vụ là Trích xuất khía cạnh (AE) và Phân loại cảm xúc (ASC). Thứ hai, về cách thực hiện đồng thời: nhóm áp dụng cơ chế chia sẻ tham số toàn phần (Hard Parameter Sharing). Toàn bộ 300 triệu tham số Encoder-Decoder được dùng chung, Decoder sinh lần lượt token khía cạnh rồi đến token cảm xúc trong cùng một chuỗi. Cơ chế Self-Attention giúp xác suất sinh cực tính cảm xúc được đặt điều kiện trực tiếp trên chính khía cạnh vừa sinh trước đó, giải quyết triệt để lỗi tích lũy của mô hình 2 giai đoạn. Đặc biệt, khi chạy thực nghiệm baseline với hàm loss mặc định, nhóm phát hiện hiện tượng sụp đổ lớp trung tính vì lớp này chỉ chiếm 4.5%. Nhóm đã chủ động thiết kế hàm Token-level Weighted Cross-Entropy trong file train.py: nhân trọng số phạt 3.89 lần cho token Neutral, 1.60 lần cho Negative, nhưng giữ nguyên 1.0 cho các từ khía cạnh để không làm nhiễu việc rút trích aspect."*

---

# SLIDE 5: KỸ THUẬT TRANSFER LEARNING & THÍCH ỨNG MIỀN
*(Đáp ứng Yêu cầu bắt buộc 3: Kỹ thuật Transfer Learning đã áp dụng để thích ứng miền)*

### 🎯 Tiêu đề Slide:
**KỸ THUẬT TRANSFER LEARNING & THÍCH ỨNG ĐA MIỀN (DOMAIN ADAPTATION)**

### 📋 Nội dung hiển thị trên Slide:
* **Hình ảnh sơ đồ chèn vào Slide:** ![Quy trình Transfer Learning](file:///f:/laptrinhPython/chuyenDe/github/ABSA/docs/picture/slide5_transfer_learning.png)
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
| **Dynamic Batch Trimming** | Hàm `trim_batch()` + chế độ **FP32 Stable** | Cắt bỏ padding dư thừa theo mẫu dài nhất mỗi batch (tăng tốc ~30%); tắt FP16 để tránh lỗi tràn số `NaN`. |
| **Đánh giá & Chọn Checkpoint** | Đánh giá trên **toàn bộ 950 câu Dev** mỗi epoch | Tính Validation Loss thực tế và lưu `best_model` theo Macro-F1 để tối ưu toàn diện. |

### 🗣️ Lời thoại gợi ý (Speaker Notes):
> *"Về kỹ thuật Transfer Learning, nhóm thực hiện tinh chỉnh toàn bộ tham số từ checkpoint tiền huấn luyện google/mt5-small sang tập dữ liệu đa miền 3,783 câu. Việc tinh chỉnh đồng thời trên cả 3 miền giúp mô hình vừa học được các khía cạnh chung như giá cả, chất lượng phục vụ, vừa phân hóa được từ vựng đặc thù của từng lĩnh vực. Để huấn luyện hiệu quả mô hình 300 triệu tham số trên GPU cá nhân GTX 1650 chỉ có 4GB VRAM, nhóm áp dụng 3 kỹ thuật tối ưu trong code: thứ nhất là Gradient Accumulation 4 bước để đạt batch size hiệu dụng là 16; thứ hai là hàm trim_batch cắt bỏ phần đệm dư thừa trong từng batch giúp tăng tốc độ huấn luyện khoảng 30%; và thứ ba là đánh giá nghiêm ngặt trên toàn bộ 950 câu của tập Dev kèm tính Validation Loss thực tế sau mỗi epoch để lưu lại mô hình tốt nhất."*

---

# SLIDE 6: KẾT QUẢ THỰC NGHIỆM, CÂN BẰNG NHÃN & PHÂN TÍCH LỖI

### 🎯 Tiêu đề Slide:
**KẾT QUẢ THỰC NGHIỆM ĐỐI SÁNH, KHẮC PHỤC NEUTRAL COLLAPSE & PHÂN TÍCH LỖI**

### 📋 Nội dung hiển thị trên Slide:
* **Hình ảnh biểu đồ chèn vào Slide:**
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

### 🗣️ Lời thoại gợi ý (Speaker Notes):
> *"Trên tập kiểm thử gồm 1,660 câu với 2,185 cặp chuẩn, mô hình Baseline đạt Micro-F1 tổng thể 60.93%, trong đó riêng khả năng trích xuất khía cạnh đạt 70.77% và khi đã bắt đúng khía cạnh thì độ chính xác phân loại cảm xúc đạt tới 84.96%. Tuy nhiên, nhóm không chỉ dừng lại ở chỉ số Micro-F1 mà đo lường sâu từng lớp và phát hiện hạn chế lớn của Cross-Entropy truyền thống: lớp Neutral có F1 bằng 0% ở cả 3 miền, kéo Macro-F1 xuống 39.09%. Sau khi áp dụng kỹ thuật Weighted Loss ở Bước 2, mô hình đã phá vỡ hoàn toàn điểm mù này: Neutral F1 tăng từ 0% lên 21.05% trên tập Dev và 16.83% trên tập Test, nâng Macro-F1 lên 41.56% trên Dev và 40.02% trên Test. Đồng thời, sự đánh đổi nhẹ ở Micro-F1 (từ 60.93% xuống 55.41%) phản ánh đúng quy luật học máy khi mô hình dịch chuyển biên quyết định để ưu tiên nhận diện lớp hiếm."*

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

### 🗣️ Lời thoại gợi ý (Speaker Notes):
> *"Để trực quan hóa toàn bộ kết quả nghiên cứu, nhóm đã xây dựng ứng dụng Web UI bằng Streamlit với 3 phân hệ. Trên giao diện này, cô và các bạn có thể chuyển đổi trực tiếp giữa checkpoint Baseline và checkpoint Weighted Loss để kiểm chứng khả năng bắt nhãn Neutral trong thời gian thực, cũng như tra cứu ma trận nhầm lẫn và lọc xem từng câu dự đoán sai trên tập Test. Về định hướng cuối kỳ, nhờ lựa chọn kiến trúc đa ngôn ngữ mT5 ngay từ đầu và thiết kế sẵn tham số ngôn ngữ trong pipeline, nhóm có thể giữ nguyên 100% luồng xử lý kỹ thuật để mở rộng sang tiếng Việt và các ngôn ngữ khác theo đúng yêu cầu của môn học. Sau đây, nhóm xin phép chuyển sang phần chiếu và giải thích source code trực tiếp."*

---

# 💻 PHẦN 2: KỊCH BẢN CHIẾU & GIẢI THÍCH SOURCE CODE TRỰC TIẾP
*(Dành riêng cho phần yêu cầu: "Chiếu và giải thích code trực tiếp ngay sau khi trình bày slide")*

Khi kết thúc Slide 7, hãy mở VS Code và trình bày theo đúng 4 bước tương ứng với 4 nội dung bắt buộc:

### Bước 1: Chiếu Code Trộn Dữ Liệu Đa Miền
* **File:** [src/build_multi_domain.py](file:///f:/laptrinhPython/chuyenDe/github/ABSA/src/build_multi_domain.py)
* **Giải thích:** 
  * Chỉ vào hàm `build_split(split, seed=42)`: Đọc từ 3 thư mục `restaurant`, `hotel`, `laptop`.
  * Chỉ vào đoạn ghi `{split}_domains.txt`: Ghi nhớ câu nào thuộc miền nào để khi đánh giá có thể phân rã hiệu năng theo từng miền.
* **File:** [src/dataset.py](file:///f:/laptrinhPython/chuyenDe/github/ABSA/src/dataset.py)
  * Chỉ vào hàm `format_target(raw_labels)` (dòng 50–90): Biến danh sách bộ ba thô thành chuỗi `(aspect, polarity); ...` và loại bỏ trùng lặp cặp trong cùng một câu cho tác vụ UABSA.

### Bước 2: Chiếu Code Kiến trúc Seq2Seq & Transfer Learning
* **File:** [src/train.py](file:///f:/laptrinhPython/chuyenDe/github/ABSA/src/train.py) (dòng 125–170)
* **Giải thích:**
  * Khởi tạo `AutoTokenizer.from_pretrained('google/mt5-small')` và `AutoModelForSeq2SeqLM.from_pretrained('google/mt5-small')`.
  * Chỉ vào hàm `trim_batch` trong [src/engine.py](file:///f:/laptrinhPython/chuyenDe/github/ABSA/src/engine.py): Cắt padding động theo câu dài nhất mỗi batch, giúp tiết kiệm bộ nhớ trên GPU GTX 1650 4GB.

### Bước 3: Chiếu Code Cơ chế Multi-task & Weighted Loss
* **File:** [src/train.py](file:///f:/laptrinhPython/chuyenDe/github/ABSA/src/train.py) (dòng 72–120 và 185–215)
* **Giải thích:**
  * **Hàm `class_weights`:** Tính trọng số nghịch đảo $w_c = (N_{\max}/N_c)^{0.5}$ $\rightarrow$ `positive=1.0, negative=1.6, neutral=3.89`.
  * **Hàm `polarity_token_weight_vector`:** Ánh xạ trọng số vào đúng token ID của từ cảm xúc trong từ điển mT5.
  * **Vòng lặp huấn luyện:** Dùng `F.cross_entropy(logits, labels, weight=tok_weight_vec)` kết hợp `gradient_accumulation_steps=4`.

### Bước 4: Chiếu Code Đánh giá & Demo Web UI
* **File:** [src/metrics.py](file:///f:/laptrinhPython/chuyenDe/github/ABSA/src/metrics.py)
  * Chỉ vào hàm `compute_metrics`: Phân tích chuỗi dự đoán bằng Regex, tính Micro-F1, Macro-F1, Per-class, Domain breakdown và Ma trận nhầm lẫn.
* **Chạy Web UI:** Mở terminal chạy `.\.venv\Scripts\streamlit.exe run app.py` $\rightarrow$ Trình diễn Tab 1 (nhập câu có nhãn Neutral) và Tab 3 (soi kết quả thực nghiệm từ JSON/TSV).

---

# ❓ PHẦN 3: BỘ CÂU HỎI PHẢN BIỆN DỰ KIẾN & CÂU TRẢ LỜI CHUẨN XÁC (Q&A DEFENSE)

### Câu hỏi 1: *"Tại sao nhóm khẳng định mô hình mT5 của mình là Multi-task Learning? Nó khác gì so với mô hình BERT gắn 2 head?"*
* **Trả lời:**
  > *"Thưa cô, trong bài toán ABSA, hai nhiệm vụ cốt lõi là Trích xuất khía cạnh (AE) và Phân loại cảm xúc (ASC). Nếu dùng BERT truyền thống, ta phải gắn 1 head gán nhãn chuỗi BIO (cho AE) và 1 head phân loại (cho ASC). Còn với kiến trúc Sequence-to-Sequence mT5, nhóm áp dụng cơ chế **Unified Generative Multi-Task qua Hard / Full Parameter Sharing**: toàn bộ Encoder và Decoder được chia sẻ 100% tham số cho cả 2 tác vụ, tối ưu chung trên một hàm mất mát thống nhất. Decoder sinh lần lượt token khía cạnh rồi đến token cảm xúc, giúp xác suất của cảm xúc được đặt điều kiện trực tiếp lên khía cạnh vừa sinh phía trước ($P(\text{polarity} \mid \text{aspect}, X)$). Hơn thế nữa, cách tiếp cận này giải quyết được **22.5% mẫu khía cạnh ẩn (`NULL`)** trong tập dữ liệu mà nhánh BIO trên BERT không thể gán nhãn vị trí từ được."*

### Câu hỏi 2: *"Tại sao ở Slide 2 tập Test có 2,531 cặp, nhưng sang Slide 6 đánh giá lại ghi 2,185 cặp chuẩn?"*
* **Trả lời:**
  > *"Thưa cô, tập dữ liệu gốc M-ABSA được gán nhãn theo bộ ba `(aspect, category, polarity)` gồm 2,531 bộ ba trên tập Test. Trong một số câu, cùng một khía cạnh (ví dụ `room`) được gán với 2 danh mục khác nhau nhưng mang cùng cảm xúc (ví dụ `(room, rooms_general, positive)` và `(room, rooms_cleanliness, positive)`). Khi chuyển sang tác vụ UABSA chỉ xét cặp `(aspect, polarity)`, hai bộ ba này gộp thành một cặp `(room, positive)` duy nhất. Sau khi khử trùng lặp trong cùng một câu, tập Test có đúng **2,185 cặp chuẩn**, đảm bảo việc đánh giá không bị trùng lặp giả tạo."*

### Câu hỏi 3: *"Tại sao khi dùng Weighted Loss thì Macro-F1 và Neutral F1 tăng mạnh nhưng Micro-F1 lại giảm từ 60.93% xuống 55.41%?"*
* **Trả lời:**
  > *"Thưa cô, đây là sự đánh đổi (trade-off) hoàn toàn tự nhiên trong học máy khi xử lý bài toán mất cân bằng nhãn nặng (Positive chiếm 72.3%, Neutral chỉ 4.3%). Ở mô hình Baseline dùng Cross-Entropy tiêu chuẩn, mô hình học mẹo 'an toàn' là không bao giờ đoán nhãn Neutral (dự đoán đúng 0 mẫu) để dồn xác suất vào lớp đa số, nhờ đó đạt Micro-F1 60.93% nhưng bị 'liệt' hoàn toàn lớp Neutral (F1 = 0%, Macro-F1 chỉ 39.09%). Khi nhóm áp dụng Weighted Loss phạt nặng lỗi lớp Neutral gấp 3.89 lần, mô hình chủ động mở rộng biên quyết định sang lớp trung tính (sinh ra 107 dự đoán Neutral, bắt đúng 17/95 cặp chuẩn, kéo Neutral F1 từ 0% lên 16.83% trên Test và 21.05% trên Dev). Sự đánh đổi nhẹ ở Micro-F1 chứng minh mô hình đã thực sự học được biểu diễn của lớp thiểu số thay vì học vẹt theo lớp đa số."*
