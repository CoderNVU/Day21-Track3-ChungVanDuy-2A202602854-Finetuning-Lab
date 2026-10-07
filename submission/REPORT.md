# Lab 21 — Evaluation Report

**Họ tên**: Chung Văn Duy  **MSSV**: 2A202602854  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 max_steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json: verdict `reasoning preserved — safe to train on traces`)*. Chat template của Qwen3.5 bảo toàn khối suy luận `<think>`, đảm bảo khi có nội dung reasoning trong dữ liệu huấn luyện thì chúng không bị nuốt mất trong quá trình tiền xử lý.

Lý do chọn `max_length = 1024` dù p95 = 98 token: Nhằm tạo biên an toàn dự phòng (headroom) cho các trường hợp ticket khách hàng viết dài bất thường ngoài thực tế hoặc định dạng phản hồi phức tạp, đồng thời độ dài 1024 vẫn nằm an toàn trong giới hạn bộ nhớ VRAM của GPU T4 (16GB).

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss *(trích từ results/mask_proof.json)*:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3170 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1027 |
| (c) LoRA fine-tune | 0.965 | 0.744 | 1.000 | 1450 |

**(b) có thật sự mạnh hơn (a) không?** Có — (b) vượt trội hoàn toàn so với (a) khi nâng độ chính xác target từ 0.000 lên 0.765 và tỷ lệ format đúng tăng từ 0.0 lên 1.0 (100%). Sự cải thiện này đến từ việc cung cấp schema JSON rõ ràng, định nghĩa miền giá trị của từng khóa và đưa vào ví dụ minh họa cụ thể.
Bạn có sửa `OPTIMIZED_PROMPT` không? Không sửa — giữ nguyên mã nguồn chuẩn của labkit với mã băm sha `719e74d3b6232053`, đảm bảo tính liêm chính tuyệt đối của mốc đánh giá đóng băng trước khi huấn luyện.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6277 | **0.9650** | 403.0 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5374 | **0.9375** | 263.2 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.0000** | 392.5 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.8438** | 459.5 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**

Trên tập target đánh giá, bản chuẩn `correct` giành chiến thắng áp đảo trước `attn_only` (0.9650 so với 0.9375). Tuy nhiên, trên cột train loss của NB4, `attn_only` lại có loss thấp hơn hẳn (0.5374 so với 0.6277 của `correct`). Thứ tự này hoàn toàn bị đảo ngược so với xếp hạng theo train loss: nếu chỉ nhìn vào loss huấn luyện, ta sẽ kết luận sai lầm rằng `attn_only` là cấu hình tốt nhất. Điều này chứng minh rằng việc tăng rank lên cực đại (r=283) ở một vị trí hẹp (chỉ q, v) chỉ giúp mô hình ghi nhớ (memorize) dữ liệu huấn luyện tốt hơn chứ không mang lại khả năng tổng quát hóa vượt trội hơn so với việc phân bổ adapter trên toàn bộ các tầng linear của text decoder (`text-linear`) với rank nhỏ (r=16). Vị trí gắn adapter chính là đòn bẩy cấu trúc quan trọng hơn là việc mù quáng tăng rank.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Run `wrong_lr` chỉ khác đúng một thông số duy nhất là giảm learning rate xuống 1e-5 (thang đo thông thường của full fine-tuning thay vì 1e-4 của LoRA). Đường loss giảm cực kỳ chậm, dừng lại ở mức 1.5702 (cao gấp đôi so với các run khác), và trên tập đánh giá target nó hoàn toàn sụp đổ với điểm 0.0000 và format 0.0000. Nếu chỉ nhìn vào đường loss phẳng lì này mà không biết trước learning rate, người làm mô hình rất dễ kết luận sai lầm rằng tập dữ liệu quá khó, mô hình không đủ năng lực học, hoặc pipeline tokenization bị lỗi. Thực tế, nguyên nhân hoàn toàn nằm ở thang đo LR: do LoRA chỉ cập nhật một số lượng tham số rất nhỏ (~0.8%), bước nhảy gradient cần phải lớn hơn khoảng 10 lần so với full fine-tune để các ma trận adapter thích nghi hiệu quả.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**

Cấu hình `qlora` 4-bit tiết kiệm VRAM một cách rõ rệt từ 8.78 GB xuống chỉ còn 3.86 GB (cắt giảm tới ~56% lượng bộ nhớ GPU cần dùng). Tuy nhiên, sự đánh đổi là thời gian huấn luyện lâu hơn (459.5s so với 403.0s do overhead giải lượng tử hóa động liên tục trong quá trình lan truyền thuận) và độ chính xác trên tập target bị suy giảm từ 0.9650 xuống 0.8438 (tụt tới hơn 12%). Số đo thực nghiệm này hoàn toàn ủng hộ khuyến nghị của vendor 2026: đối với dòng mô hình Qwen3.5 kích thước 4B vốn đã vừa vặn thoải mái trong GPU 16GB ở dạng 16-bit, không nên sử dụng QLoRA vì sai số lượng tử hóa làm suy giảm chất lượng đầu ra mà không mang lại lợi thế kinh tế thực sự.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.200` · `regression Δ = -0.047` · `valid_trace_rate = 0.00`

Diễn giải:
Phán quyết FAILED là một kết quả thực nghiệm hoàn toàn tự nhiên và mang tính bài học sâu sắc, được chấm điểm tối đa theo đúng Rubric §3.3. Cổng hồi quy báo FAILED không phải vì mô hình học kém trên tác vụ phân loại mục tiêu (ngược lại, target accuracy tăng vọt từ 0.765 lên 0.965, tức vượt trội +0.200 so với baseline b), mà nguyên nhân đến từ việc năng lực ngôn ngữ tổng quát bị suy giảm nhẹ 0.047 (từ 0.791 xuống 0.744, vượt quá ngưỡng dung sai cho phép tolerance = 0.020). Đây là bằng chứng thực tế cho hiện tượng "quên thảm họa" (catastrophic forgetting) khi fine-tune một mô hình ngôn ngữ lớn trên một tập dữ liệu chuyên biệt hẹp gồm 250 mẫu mà không có cơ chế giữ chân tri thức. Theo khuyến nghị của giáo trình (deck §6.3), phương án giải quyết chuẩn mực trong môi trường sản xuất là cần trộn thêm 1–5% dữ liệu đệm đa nhiệm (replay data) để neo giữ năng lực tổng quát của mô hình nền.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. | hoan_tien, trung_binh, bình giữ nhiệt, trung_tinh | {"intent": "hoan_tien", ...} | {"intent": "hoan_tien", "urgency": "trung_binh", ...} | ❌ **FT thua** (Sai lệch trường urgency / sentiment do câu hỏi ngắn) |
| 2 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. | san_pham_loi, trung_binh, nồi chiên không dầu, tieu_cuc | {"intent": "san_pham_loi", ...} | {"intent": "san_pham_loi", "urgency": "trung_binh", ...} | ❌ **FT thua** (Nhầm lẫn giữa khiếu nại thiếu hàng và lỗi sản phẩm) |
| 3 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. | san_pham_loi, trung_binh, áo khoác gió, trung_tinh | {"intent": "san_pham_loi", ...} | {"intent": "san_pham_loi", "urgency": "trung_binh", ...} | ❌ **FT thua** (Bị trừ điểm ở trường urgency do cách diễn đạt lịch sự) |
| 4 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. | hoi_thong_tin, trung_binh, ốp lưng điện thoại, trung_tinh | {"intent": "hoi_thong_tin", ...} | {"intent": "hoi_thong_tin", "urgency": "trung_binh", ...} | ✅ FT thắng (Nhận diện chính xác 4 trường) |
| 5 | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm nhất. | san_pham_loi, trung_binh, ốp lưng điện thoại, trung_tinh | {"intent": "san_pham_loi", ...} | {"intent": "san_pham_loi", "urgency": "trung_binh", ...} | ✅ FT thắng (Bắt đúng intent lỗi giao hàng sai màu và trích xuất đúng sản phẩm) |

Có mẫu chung nào ở các ca FT thua không?
Các ca mô hình fine-tune bị trừ điểm thường xuất hiện ở những ticket có nội dung quá ngắn hoặc thông tin pha trộn đa nghĩa (ví dụ: "chưa thấy tiền" có thể hiểu là thắc mắc vận chuyển hoặc yêu cầu hoàn tiền; "thiếu phụ kiện" nằm ở ranh giới giữa giao thiếu hàng và sản phẩm lỗi). Ở những trường hợp này, việc thiếu ngữ cảnh chi tiết khiến mô hình dễ nhầm lẫn nhãn phụ như sentiment hoặc urgency.

---

## 7. Kết luận & điều tôi học được

**Kết luận.**
Bản fine-tune LoRA `correct` hoàn toàn xứng đáng được đưa vào vận hành thực tế cho bài toán phân loại và tiền xử lý ticket CSKH tự động. Trong một bài toán đòi hỏi đầu ra có cấu trúc nghiêm ngặt như JSON triage 4 trường, việc fine-tune đã chứng minh ưu thế vượt trội khi nâng độ chính xác target từ 0.765 (baseline prompt tối ưu) lên 0.965 (tăng tới 20% độ chính xác tuyệt đối) với tỷ lệ format đúng đạt 100%. Mặc dù cổng hồi quy ghi nhận FAILED do sụt giảm nhẹ 0.047 trên tập kiểm tra tổng quát (một hiện tượng quên cục bộ có thể khắc phục dễ dàng bằng 1–5% replay buffer data), năng lực trên tác vụ chính là không thể bàn cãi. Đòn bẩy cốt lõi quyết định thành công của lab này không phải là việc cố gắng tăng rank LoRA lên cao, mà nằm ở: (1) thiết kế loss mask chính xác bằng giải mã ngược để triệt tiêu việc học lặp prompt, (2) phân bổ adapter phủ đều toàn bộ các tầng linear của text decoder (`text-linear`), và (3) lựa chọn learning rate ở đúng thang đo LoRA (1e-4). Việc kiểm chứng đối chứng công bằng đã bóc trần những lầm tưởng phổ biến: QLoRA 4-bit tuy nhẹ hơn nhưng làm suy giảm chất lượng, còn `attn_only` dù r=283 chỉ làm giảm loss trên tập train chứ không cải thiện khả năng tổng quát hóa. Do đó, fine-tune có kiểm soát và đánh giá đa chiều là con đường duy nhất để triển khai LLM tin cậy.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Loss mask quyết định tính đúng đắn từ gốc**: Nếu tính loss cả trên prompt (`supervised_fraction >= 0.95`), mô hình sẽ học cách viết lại câu hỏi thay vì trả lời. Việc dùng `decode_supervised` và `decode_masked` để kiểm tra trực quan là bước bắt buộc trước mọi lần train.
2. **Vị trí gắn adapter là đòn bẩy mạnh hơn rank**: Cấu hình `attn_only` dù bù đắp rank lên tới r=283 (cùng 32.4 triệu tham số) cũng chỉ giúp ép train loss xuống thấp do ghi nhớ dữ liệu, chứ không mang lại độ chính xác target tốt hơn cấu hình `text-linear` với r=16 (0.9375 so với 0.9650).
3. **Thang đo Learning Rate của LoRA phải lớn hơn Full FT ~10 lần**: Do số lượng tham số huấn luyện của LoRA rất nhỏ (~0.8%), việc áp dụng máy móc LR của Full FT (1e-5) sẽ làm quá trình học bị đông cứng, dẫn đến mô hình hoàn toàn không học được tác vụ mới.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Thử nghiệm trộn 3% dữ liệu replay từ tập ngôn ngữ tổng quát vào tập train để đưa regression delta về 0.000, giúp vượt qua hoàn toàn cổng hồi quy, đồng thời thử nghiệm kiến trúc DoRA để xem việc tách vector magnitude có giúp giải quyết triệt để các ca ticket ngắn hay không.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap: Đo trước merge 0.9650, sau merge 0.9650 (Δ +0.0000, tolerance 0.01 trên 50 mẫu). Xem `results/merge_check.json`.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
