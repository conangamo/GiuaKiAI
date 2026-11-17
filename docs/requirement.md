TÊN ĐỒ ÁN: Phân tích Quan điểm theo Khía cạnh (Aspect-Based Sentiment Analysis - ABSA)

1. MỤC TIÊU LÕI (CORE OBJECTIVES)

Bài toán: Phân tích quan điểm theo khía cạnh (ABSA). Khác biệt hoàn toàn với phân tích theo câu hay tài liệu thông thường.

Yêu cầu Đa tác vụ (Multi-task): Một mô hình duy nhất phải thực hiện đồng thời 2 việc:

Trích xuất khía cạnh (Aspect Extraction).

Phân loại cảm xúc/quan điểm trên chính khía cạnh vừa trích xuất (Sentiment Classification).

2. YÊU CẦU KỸ THUẬT (TECHNICAL REQUIREMENTS)

Dữ liệu (Multi-domain): Phải kết hợp dữ liệu từ 2 miền trở lên để tạo thành tập dữ liệu đa miền (Ví dụ: trộn data review Laptop, Hotel, và Restaurant).

Học chuyển giao (Transfer Learning):

Sử dụng các mô hình ngôn ngữ lớn (Pre-trained models) như BERT, RoBERTa, hoặc PhoBERT (nếu dùng tiếng Việt).

Sử dụng phương pháp tinh chỉnh (Fine-tuning) để mô hình thích ứng với tập dữ liệu đa miền.

Học đa tác vụ (Multi-task Learning): Áp dụng các cơ chế chia sẻ tham số (Hard Parameter Sharing hoặc Soft Parameter Sharing) để mô hình học và tối ưu cả 2 tác vụ đầu ra cùng một lúc.

3. CHUẨN ĐẦU RA GIỮA KỲ (MIDTERM DELIVERABLES)

Thời hạn (Deadline): Tối đa 2 tuần. Nhóm hoàn thành sớm có thể đăng ký báo cáo ngay vào tuần sau.

Sản phẩm nộp:

Slide báo cáo: Bắt buộc từ 5 đến 8 slide, đi thẳng vào trọng tâm kỹ thuật.

Source code: Chiếu và giải thích code trực tiếp ngay sau khi trình bày slide.

Lưu ý: Chưa yêu cầu viết Technical Report ở giai đoạn này.

Nội dung bắt buộc trên Slide:

Cấu trúc tập dữ liệu đa miền đã trộn.

Kiến trúc mô hình sử dụng.

Kỹ thuật Transfer Learning đã áp dụng để thích ứng miền.

Cơ chế Multi-task Learning đã thiết lập để giải quyết đồng thời các tác vụ.

4. ĐỊNH HƯỚNG CUỐI KỲ (FINAL TERM SCOPE)

Đa ngôn ngữ (Multi-language): Giữ nguyên luồng xử lý kỹ thuật, đổi tập dữ liệu sang dạng đa ngôn ngữ để kiểm thử tính tương thích và khả năng mở rộng của mô hình (Sẽ được giáo viên hướng dẫn thêm vào tuần tới).

🟢 NHỮNG ĐỀ XUẤT / VÍ DỤ TÙY CHỌN (Options & Examples)Đây là những thứ cô đưa ra để gợi ý, bạn có quyền chọn theo hoặc dùng cái khác tương đương:1. Hướng tiếp cận (Approach):Cô nói: "Có thể là feature based, có thể là fine-tuning based (fattening based)... Đa số các bạn sẽ sử dụng fine-tuning..."$\rightarrow$ Quyền của bạn: Bạn được chọn dùng Feature-based hoặc Fine-tuning, nhưng cô ngầm đề xuất (recommend) nên dùng Fine-tuning vì nó phổ biến và hiệu quả hơn cho bài toán này.2. Các miền dữ liệu (Domains):Cô ví dụ: "Laptop, miền Hotel, miền Restaurant..."$\rightarrow$ Quyền của bạn: Bạn có thể dùng tập dữ liệu về Laptop/Restaurant (rất phổ biến trong SemEval), nhưng cũng có thể tự chọn các domain khác (như Review phim, Review mỹ phẩm, v.v.) miễn là từ 2 domain trở lên.3. Mô hình sử dụng (Models):Cô ví dụ: "Mô hình BERT, mô hình RoBERTa (Provita)... hay nếu dùng tiếng Việt thì dùng PhoBERT."$\rightarrow$ Quyền của bạn: Bạn dùng BERT, RoBERTa, DeBERTa, PhoBERT hay bất kỳ Pre-trained model nào khác đều được, tùy thuộc vào việc bạn chọn làm tiếng Anh hay tiếng Việt.4. Cơ chế chia sẻ tham số (Parameter Sharing):Cô gợi ý: "Sử dụng 2 Parameter Sharing hay là 4 Parameter Sharing..." (Hard parameter sharing hoặc Soft parameter sharing).$\rightarrow$ Quyền của bạn: Bạn tự quyết định kiến trúc chia sẻ tham số giữa 2 task (trích xuất khía cạnh & phân loại cảm xúc) sao cho hiệu quả.