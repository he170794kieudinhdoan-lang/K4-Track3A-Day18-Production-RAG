# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Kiều Đình Đoàn  
**MSSV:** HE170794  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 05/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Tổng hợp những điểm đúc kết được khi đối chiếu lý thuyết bài học với quá trình code thực tế:

| Khái niệm bài giảng | Module | Hàm cụ thể | Trải nghiệm & Nhận xét thực tế |
|---------------------|--------|------------|--------------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng khoảng cách ngữ nghĩa để gom câu thấy tự nhiên hơn hẳn. Các câu cùng mạch ý đi chung với nhau, không bị tình trạng đang nói dở câu thì bị cắt ngang như cách đếm ký tự thông thường. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Cơ chế hoạt động rất thông minh: lúc tìm kiếm thì soi mẩu nhỏ (child 256 ký tự) cho chuẩn và nhanh, nhưng khi nạp cho LLM đọc thì lấy cả mảng to (parent 2048 ký tự) nên không lo mất ngữ cảnh bao quát. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Rất phù hợp với tài liệu chính sách hay hướng dẫn có nhiều tiêu đề H1-H3 và bảng biểu. Tách theo heading vừa giữ nguyên vẹn nội dung bảng, vừa xác định chính xác chunk thuộc mục nào. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 bắt từ khóa và con số cụ thể rất nhạy, còn Dense thì hiểu ngữ nghĩa câu hỏi. Kết hợp cả hai qua thuật toán RRF giúp bù trừ điểm yếu cho nhau, kết quả tìm kiếm đầy đủ hơn nhiều. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Mô hình này soi trực tiếp cả cặp (câu hỏi, tài liệu) nên chấm điểm chuẩn xác rõ rệt. Tài liệu liên quan nhất được đẩy thẳng lên top đầu, dù tốn thêm chút thời gian xử lý nhưng mang lại hiệu quả lọc rõ rệt. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Thay vì ngồi đọc từng câu rồi đoán xem chatbot trả lời tốt hay dở, RAGAS chỉ ra luôn hệ thống đang hổng ở khâu tìm kiếm (precision/recall) hay khâu sinh câu trả lời (faithfulness/relevancy). |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Trước khi biến văn bản thành vector, nhét thêm đoạn tóm tắt ngắn và vài câu hỏi thường gặp vào đầu chunk. Làm vậy câu hỏi của người dùng dễ "bắt sóng" đúng chunk cần tìm hơn hẳn. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

Những vấn đề kỹ thuật phát sinh trong quá trình chạy thực tế và cách xử lý:

- **Lỗi bảng mã tiếng Việt trên môi trường Windows:**  
  Khi in kết quả ra terminal PowerShell xuất hiện lỗi `UnicodeEncodeError: 'charmap'`. Nguyên nhân do terminal Windows mặc định dùng bảng mã cũ thay vì UTF-8. Cách xử lý là thêm dòng `sys.stdout.reconfigure(encoding='utf-8')` ngay đầu file để hiển thị tiếng Việt trơn tru.

- **Mô hình Cross-Encoder dung lượng lớn gây chậm hệ thống:**  
  Model `bge-reranker-v2-m3` nặng hơn 2GB, ban đầu mỗi hàm test lại tải và nạp lại từ đầu, làm máy giật lag và tốn nhiều thời gian. Giải pháp là tạo một dictionary cache model ở cấp module, nạp một lần rồi tái sử dụng xuyên suốt quá trình chạy test, tốc độ tăng lên thấy rõ.

- **Vấn đề tách từ tiếng Việt khi so khớp BM25:**  
  Khi dùng thư viện tách từ tiếng Việt, các từ ghép thường nối bằng dấu gạch dưới (ví dụ: `nghỉ_phép`). Trong khi đó người dùng gõ câu hỏi thì viết cách bình thường (`nghỉ phép`). Để tránh việc BM25 coi đây là hai từ khác nhau dẫn đến trượt kết quả, giải pháp là thay thế dấu gạch dưới thành dấu cách để việc so khớp từ khóa đồng nhất và chính xác hơn.

---

## Phần 3: Kế hoạch áp dụng vào dự án thực tế (Action Plan)

Kế hoạch nâng cấp hệ thống hỏi đáp dựa trên các kỹ thuật đã hoàn thiện:

### Dự án: Trợ lý Hỏi - Đáp Nội bộ Doanh nghiệp

#### 1. Tình trạng hiện tại của dự án
- Hệ thống hiện tại mới chỉ triển khai RAG dạng cơ bản: cắt văn bản theo đoạn cố định 1000 ký tự bằng LangChain rồi lưu vào FAISS vector store.
- Vấn đề gặp phải: Khi người dùng hỏi về các con số cụ thể, ngày tháng hay chính sách mới thì bot trả lời lẫn lộn giữa bản cũ và bản mới; các bảng biểu trong sổ tay nhân viên bị cắt vụn nên thông tin trả về bị thiếu hụt.

#### 2. Định hướng áp dụng kỹ thuật
1. **Chia nhỏ văn bản linh hoạt hơn:** Chuyển sang dùng Hierarchical Chunking (chia đoạn con 256 ký tự để tìm kiếm, lấy đoạn cha 2048 ký tự để trả lời) kết hợp bóc tách theo cấu trúc đề mục Markdown để giữ nguyên các bảng quy định.
2. **Nâng cấp công cụ tìm kiếm:** Không chỉ dựa vào mỗi Vector search, mà triển khai kết hợp thêm BM25 (đã xử lý tách từ tiếng Việt) bằng công thức RRF để bắt dính các từ khóa mã số, điều khoản.
3. **Thêm bước lọc Reranker:** Bổ sung một tầng Cross-Encoder hoặc FlashRank để lọc lại top 3 tài liệu sát nhất trước khi gửi cho LLM, tránh việc nhồi nhét quá nhiều tài liệu rác làm câu trả lời bị loãng.
4. **Đo đạc chất lượng bài bản:** Dùng bộ công cụ RAGAS chạy thử nghiệm định kỳ với bộ câu hỏi thực tế để theo dõi xem mỗi lần điều chỉnh thuật toán thì điểm số tăng giảm ra sao, loại bỏ cách đánh giá cảm tính.
5. **Làm giàu ngữ cảnh (Enrichment):** Trước khi nạp tài liệu vào kho, cho LLM tóm tắt nhanh 1 câu về ngữ cảnh và phiên bản của tài liệu gắn vào đầu chunk để tránh tình trạng nhầm lẫn văn bản cũ - mới.

#### 3. Kế hoạch thời gian dự kiến
- **Tuần 1:** Tái cấu trúc lại dữ liệu tài liệu công ty theo dạng cây đề mục và chạy thử nghiệm kết hợp BM25 + Vector Search.
- **Tuần 2:** Tích hợp Reranker và bộ đo RAGAS để nghiệm thu xem độ chính xác có cải thiện rõ rệt như kết quả thực nghiệm trong bài lab hay không.
