# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Kiều Đình Đoàn (MSSV: HE170794)  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6500 | 0.8800 | +0.2300 |
| Answer Relevancy | 0.7100 | 0.8900 | +0.1800 |
| Context Precision | 0.5800 | 0.8600 | +0.2800 |
| Context Recall | 0.6200 | 0.9100 | +0.2900 |

*Nhận xét nhanh: Sau khi lên bản Production với đầy đủ các kỹ thuật (chia đoạn theo cấu trúc & phân cấp, kết hợp tìm kiếm BM25 + Vector Search và thêm lớp Reranker lọc lại), điểm số tăng vọt từ 0.18 đến gần 0.3 điểm trên tất cả các tiêu chí. Thấy rõ nhất là khả năng bốc đúng tài liệu (Context Recall & Precision tăng mạnh) và câu trả lời của bot bớt hẳn tình trạng nói mò (Faithfulness).*

---

## Latency Breakdown Report

Bảng phân rã thời gian xử lý thực tế qua từng giai đoạn của pipeline:

| Giai đoạn trong Pipeline | Cơ chế thực hiện | Latency trung bình (p50) | Latency p95 | Tỷ lệ thời gian |
|--------------------------|------------------|--------------------------|-------------|-----------------|
| 1. Tiền xử lý & Tách từ tiếng Việt | Underthesea segmentation | 8 ms | 15 ms | ~2% |
| 2. Hybrid Search (Dense + BM25) | Qdrant Vector + BM25Okapi + RRF | 35 ms | 52 ms | ~9% |
| 3. Cross-Encoder Reranking | BAAI/bge-reranker-v2-m3 (top 20 → top 3) | 138 ms | 185 ms | ~34% |
| 4. LLM Generation / Synthesis | Context assembly + Generation | 225 ms | 390 ms | ~55% |
| **Tổng thời gian End-to-End** | **Hoàn chỉnh 1 lượt hỏi - đáp** | **~406 ms** | **~642 ms** | **100%** |

*Ghi chú quan sát:*
- Hai khâu chiếm nhiều thời gian nhất là bước sinh câu trả lời của LLM (~55%) và tầng Cross-Encoder reranking (~34%).
- Tầng tìm kiếm Hybrid Search (kết hợp Dense + BM25 qua thuật toán RRF) diễn ra rất nhanh (chỉ khoảng 35ms) do vector search được tối ưu và BM25 chạy in-memory.

---

## Bottom-5 Failures

Dưới đây là 5 trường hợp hệ thống vẫn còn lúng túng hoặc trả lời chưa chuẩn, cùng lý do thực tế và cách xử lý:

### #1
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** 12 ngày phép năm (hoặc lẫn lộn giữa bản 2023 và 2024).
- **Worst metric:** Faithfulness / Context Precision
- **Error Tree:** Output sai → Context bị lẫn cả văn bản cũ và mới → Câu hỏi người dùng ngắn gọn, không ghi rõ năm áp dụng.
- **Root cause:** Trong kho dữ liệu có cả 2 bản cũ (2023 - 12 ngày) và mới (2024 - 15 ngày). Nếu chỉ tìm kiếm câu chữ đơn thuần mà không lọc theo hiệu lực thì hệ thống rất dễ bốc nhầm bản cũ đã hết hạn.
- **Suggested fix:** Đánh dấu metadata cho tài liệu đang áp dụng (`status="active"`), hoặc ở Module 5 gắn thẳng ghi chú `[Chính sách 2024 - Đang áp dụng]` vào đầu đoạn văn bản để bot không bị nhầm.

### #2
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự. Chính sách cũ (v1.0) yêu cầu 8 ký tự.
- **Got:** 8 ký tự (hoặc trả lời cả 8 và 12 ký tự làm người dùng hoang mang).
- **Worst metric:** Context Precision
- **Error Tree:** Output chưa chuẩn → Context trả về lẫn lộn giữa v1.0 và v2.0 → Query đúng ý nhưng không nói rõ phiên bản.
- **Root cause:** Cả hai văn bản v1 và v2 đều nói về mật khẩu nên từ khóa giống hệt nhau. Bộ tìm kiếm bốc cả hai lên, và nếu không có thêm ngữ cảnh thì reranker cũng khó phân biệt cái nào mới hơn.
- **Suggested fix:** Ở bước Enrichment trích xuất rõ phiên bản tài liệu (v2.0) và gắn cờ đã thay thế cho bản cũ để hệ thống ưu tiên văn bản mới nhất.

### #3
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách v2.0, mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày.
- **Got:** 90 ngày.
- **Worst metric:** Faithfulness
- **Error Tree:** Output sai mốc thời gian → Context bốc nhầm tài liệu v1.0.
- **Root cause:** Tài liệu cũ có cụm từ "90 ngày" rất rõ, khi người dùng hỏi chung chung thì thuật toán dễ bắt dính câu từ cũ này.
- **Suggested fix:** Bổ sung cơ chế xếp hạng ưu tiên theo thời gian (Temporal Reranking), tài liệu nào ban hành gần đây hơn thì được cộng thêm điểm ưu tiên.

### #4
- **Question:** Nhân viên thử việc có được nghỉ phép năm không?
- **Expected:** Không được nghỉ phép năm trong thời gian thử việc, chỉ được nghỉ việc riêng không lương hoặc nghỉ ốm có xác nhận y tế.
- **Got:** Trả lời chung chung về quy định nghỉ phép 15 ngày mà quên mất ngoại lệ của diện thử việc.
- **Worst metric:** Context Recall
- **Error Tree:** Output thiếu ý quan trọng → Context bị phân mảnh do thông tin nằm ở 2 văn bản khác nhau.
- **Root cause:** Quy định cho nhân viên thử việc lại nằm ở file `thu_viec.md` chứ không nằm trong file `nghi_phep_nam_v2024.md`. Khi người dùng hỏi về nghỉ phép, hệ thống chỉ chăm chăm đi tìm trong file nghỉ phép.
- **Suggested fix:** Tận dụng kỹ thuật sinh câu hỏi giả định (HyQA ở Module 5) để tạo trước các câu hỏi kiểu "Nhân viên thử việc có được nghỉ phép không?", giúp việc tìm kiếm chéo giữa các văn bản nhạy hơn nhiều.

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 18 ngày phép (15 ngày cơ bản + 3 ngày thâm niên: cứ 3 năm được cộng 1 ngày).
- **Got:** 16 ngày hoặc 15 ngày (LLM nhớ nhầm quy tắc cũ là 5 năm mới được cộng 1 ngày).
- **Worst metric:** Faithfulness (bị lỗi ở khâu tính toán số học)
- **Error Tree:** Context tìm về chuẩn (ghi rõ 3 năm cộng 1 ngày) → Nhưng phần tính toán của mô hình ngôn ngữ bị lệch.
- **Root cause:** Mô hình ngôn ngữ tự suy luận số học bị ảo giác (hallucination) hoặc vô tình áp dụng công thức cũ trong văn bản 2023.
- **Suggested fix:** Căn dặn mô hình trong prompt là phải suy nghĩ từng bước (Chain-of-Thought) và viết rõ phép tính cộng ra trước khi đưa ra con số cuối cùng.

---

## Case Study (cho presentation)

**Câu hỏi chọn để mổ xẻ:** *"Nhân viên được nghỉ bao nhiêu ngày phép năm?"*

**Lần theo luồng kiểm tra (Error Tree walkthrough):**
1. **Câu trả lời đúng chưa?** → Chưa chuẩn nếu hệ thống chỉ bảo "12 ngày", Chuẩn khi trả lời "15 ngày theo chính sách 2024 mới nhất".
2. **Context đưa vào đúng chưa?** → Kiểm tra top 3 đoạn văn bản trả về. Nếu thấy đoạn trích từ bản 2023 nằm đè lên bản 2024 thì biết ngay khâu Retrieval đang bị lỗi bốc nhầm tài liệu cũ.
3. **Câu hỏi người dùng thế nào?** → Người dùng thường hỏi rất ngắn gọn, không ai ghi thêm chữ "theo quy định năm 2024", làm cả BM25 lẫn Dense Search đều chấm điểm cao cho cả 2 văn bản.
4. **Cách khắc phục qua từng khâu:** 
   - **Khâu Chunking (M1):** Cắt theo cấu trúc Markdown để luôn giữ được dòng tiêu đề `Phiên bản 2024`.
   - **Khâu Enrichment (M5):** Gắn thẳng thẻ `[Chính sách nghỉ phép 2024 - Đang áp dụng]` vào đầu đoạn văn.
   - **Khâu Rerank (M3):** Bộ lọc Reranker sẽ nhìn thấy ngữ cảnh phiên bản đang áp dụng và đẩy bản mới lên trước.

**Hướng tối ưu mở rộng (nếu có thêm thời gian):**
- **Thêm bước viết lại câu hỏi (Query Rewrite):** Khi người dùng hỏi ngắn gọn, con bot sẽ tự động chèn thêm ngữ cảnh thời gian (ví dụ: "chính sách mới nhất hiện nay") trước khi đem đi tìm kiếm.
- **Lọc theo trạng thái tài liệu:** Thêm bộ lọc metadata đơn giản, tài liệu nào cũ thì đánh dấu lưu trữ (archive), chỉ tìm kiếm trong các tài liệu đang có hiệu lực.
- **Hoàn thiện cơ chế Parent-Child Retrieval:** Khi tìm kiếm thì dùng các mẩu nhỏ (child chunk) cho chính xác và nhanh, nhưng khi gửi cho LLM trả lời thì lấy nguyên cả đoạn văn lớn (parent chunk) xung quanh để đảm bảo không bị thiếu thông tin.
