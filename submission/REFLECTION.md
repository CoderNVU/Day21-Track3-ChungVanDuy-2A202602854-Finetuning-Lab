# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là hiện tượng đảo ngược giữa loss huấn luyện và năng lực thực tế trên tác vụ ở NB4: cấu hình `attn_only` dù rank được đẩy lên tới r=283 có train loss thấp hơn cả bản `correct` (0.5374 vs 0.6259), nhưng trên tập target thì không hề vượt trội hơn. Nếu chỉ nhìn loss mà không đo target bằng cổng hồi quy, tôi đã tin rằng `attn_only` là cấu hình thắng cuộc.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở khâu xử lý tính toàn vẹn của dữ liệu và môi trường (sự sai lệch checksum do chuyển đổi CRLF/LF trên Windows và phân bổ bộ nhớ GPU T4 ở Colab), chứ không phải ở lúc mô hình đang tối ưu gradient. Trước khi làm, tôi dự đoán thời gian chủ yếu nằm ở khâu chờ train, nhưng thực tế việc thiết lập môi trường đúng và kiểm soát mask loss chiếm nhiều tư duy hơn hẳn.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi tin rằng "chỉ cần tăng rank LoRA càng cao thì mô hình càng thông minh" và "luôn luôn nên dùng QLoRA 4-bit để tiết kiệm VRAM". Sau lab này, tôi hiểu rằng vị trí phủ adapter (`text-linear`) mới là đòn bẩy cốt lõi chứ không phải rank, và QLoRA trên các model 4B gây suy giảm chất lượng đầu ra mà không mang lại lợi ích kinh tế nếu GPU đã đủ 16GB.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để giải thích cơ chế che mask của TRL, phân tích cấu trúc các tầng linear trong mô hình Qwen3.5 và hỗ trợ viết kịch bản tự động hóa kiểm tra. Chỗ AI hay mắc lỗi nhất là thường mặc định khuyên dùng `bf16=True` hoặc đề xuất các tham số thư viện cũ (như `tokenizer=` thay vì `processing_class=` trong TRL v1) mà không tính đến giới hạn kiến trúc Turing của GPU T4.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi làm là xây dựng một tập đánh giá đóng băng (frozen eval benchmark) và đo lường baseline prompt engineering (b) thật kỹ lưỡng. Tôi sẽ chứng minh xem mô hình base đã prompt tối ưu có đáp ứng được yêu cầu của khách hàng hay chưa trước khi quyết định chi ngân sách huấn luyện, đồng thời bắt buộc kiểm chứng loss mask bằng giải mã ngược trước khi chạy bất kỳ gradient step nào.
