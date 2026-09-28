# Bài 38: Pretraining và fine-tuning - bức tranh tổng thể

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 8:** Từ pretraining đến trợ lý & dự án cuối

---

### Giới thiệu

Ở _Bài 37_ ta đã thấy các _scaling laws_: mô hình càng lớn, dữ liệu càng nhiều, tính toán càng khủng thì loss càng giảm theo một quy luật đẹp. Nhưng một mô hình khổng lồ được huấn luyện trên hàng nghìn tỷ token chỉ mới biết _“đoán chữ tiếp theo cho văn bản Internet nói chung”_. Làm sao để nó trở thành trợ lý y khoa, trợ lý pháp lý, hay bot chăm sóc khách hàng của _riêng công ty bạn_? Câu trả lời là hai giai đoạn: _pretraining_ (huấn luyện trước) rồi _fine-tuning_ (tinh chỉnh).

Hãy tưởng tượng một sinh viên đại học. Suốt 16 năm đi học, bạn ấy đọc đủ thứ: văn học, toán, lịch sử, khoa học - đó là _pretraining_: học _kiến thức chung về thế giới và ngôn ngữ_, tốn rất nhiều thời gian và tài nguyên. Khi ra trường vào làm ở một công ty bảo hiểm, bạn ấy chỉ cần _vài tuần_ đọc tài liệu nội bộ là làm được việc - đó là _fine-tuning_: tận dụng nền tảng khổng lồ đã có, chỉ _điều chỉnh_ cho nhiệm vụ cụ thể. Không ai bắt sinh viên đi học lại từ mẫu giáo chỉ để làm bảo hiểm!

> 🔑 **Một câu để nhớ:** Pretraining dạy mô hình _ngôn ngữ và thế giới_ (đắt, một lần, dùng chung); fine-tuning dạy mô hình _nhiệm vụ/miền của bạn_ (rẻ, ít dữ liệu, nhờ nền tảng đã có).

---

### Bạn sẽ học được gì?

- [ ] Phân biệt rõ _pretraining_ và _fine-tuning_: mục tiêu, dữ liệu, chi phí khác nhau ra sao
- [ ] Hiểu vì sao _ít dữ liệu vẫn fine-tune được_ (transfer learning + đặc trưng đã học sẵn)
- [ ] Biết các _kiểu_ fine-tuning: full fine-tune, fine-tune một phần, và ý tưởng PEFT/LoRA
- [ ] Nắm cách _tái dùng GPT mini_ đã build: nạp trọng số pretrained rồi huấn luyện tiếp trên dữ liệu mới
- [ ] Tránh các bẫy: _catastrophic forgetting_, overfitting khi dữ liệu ít, learning rate quá lớn

📌 **Yêu cầu trước:** Bạn đã hiểu training loop, cross-entropy cho next-token, AdamW (Bài 27-30) và scaling laws (Bài 37). Đã có một GPT mini chạy được từ các bài trước.

---

### 1. Pretraining là gì? Học kiến thức chung

_Pretraining_ = huấn luyện mô hình từ _trọng số ngẫu nhiên_ trên _một lượng văn bản khổng lồ_ với mục tiêu duy nhất: _đoán token tiếp theo_ (next-token prediction). Không có nhãn do con người gán - nhãn chính là _token kế tiếp trong văn bản_. Đây gọi là _học tự giám sát_ (self-supervised).

**Analogy:** giống như bạn học tiếng Anh bằng cách đọc cả nghìn cuốn sách, mỗi lần che một từ rồi tự đoán. Đoán sai thì sửa. Đoán hàng tỷ lần thì bạn _ngấm_ ngữ pháp, từ vựng, và cả kiến thức trong sách - dù chẳng ai dạy trực tiếp.

Hàm mất mát vẫn là _cross-entropy_ trên phân phối token tiếp theo:

$$\text{loss} = -\frac{1}{N} \sum_i \log P(\text{token}_i \text{ thật} \mid \text{các token trước đó})$$

```python
import torch
import torch.nn.functional as F

# Pretraining objective: next-token prediction (self-supervised)
# logits: model output, shape (B, T, vocab_size)
# targets: input dịch trái 1 vị trí, shape (B, T)
B, T, vocab = 2, 5, 100
logits = torch.randn(B, T, vocab) # (B, T, vocab) = (2, 5, 100)
targets = torch.randint(0, vocab, (B, T)) # (B, T) = (2, 5)

# Cross-entropy cần (N, C) và (N,): gộp batch và time lại
loss = F.cross_entropy(
    logits.view(B * T, vocab), # (B*T, vocab) = (10, 100)
    targets.view(B * T),       # (B*T,)       = (10,)
)
print(loss.item()) # ví dụ ~4.6 lúc chưa học (~ln(vocab) = ln(100))
# Trước khi học, loss ~ ln(vocab_size): mô hình đoán ngẫu nhiên đều
```

💡 Loss khởi điểm $\sim \ln(\text{vocab\_size})$ là _lành mạnh_: nghĩa là mô hình đang đoán đều mọi token. Nếu ban đầu loss nhỏ bất thường là có bug (rò rỉ nhãn, hoặc quên shift target).

Pretraining là giai đoạn _đắt nhất_: hàng nghìn GPU, hàng nghìn tỷ token, hàng tuần/tháng. Đó là lý do rất ít tổ chức tự pretraining một mô hình lớn từ đầu.

---

### 2. Fine-tuning là gì? Điều chỉnh cho nhiệm vụ/miền

_Fine-tuning_ = _lấy mô hình đã pretrained_ rồi _huấn luyện tiếp_ trên một tập dữ liệu _nhỏ hơn nhiều_, chuyên biệt cho nhiệm vụ hoặc miền của bạn (văn bản y khoa, code Python, hội thoại hỗ trợ khách hàng…).

Điểm mấu chốt: ta _không_ khởi tạo lại trọng số ngẫu nhiên. Ta _bắt đầu từ trọng số đã học_ và chỉ _nhích_ chúng đi một chút. Learning rate thường nhỏ hơn nhiều so với pretraining.

```python
import torch

# 1) Nạp GPT mini đã pretrained (trọng số đã học kiến thức chung)
model = GPTMini(config) # cấu trúc như lúc pretraining
state = torch.load("gpt_mini_pretrained.pt") # dict các tensor trọng số
model.load_state_dict(state) # nạp trọng số pretrained

# 2) Fine-tune trên dữ liệu MIỀN: learning rate NHỎ (nhích nhẹ, đừng phá nền tảng)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5) # << nhỏ hơn pretraining (vd 3e-4)
model.train()

for xb, yb in domain_dataloader: # xb, yb: (B, T) token của dữ liệu miền
    logits = model(xb) # (B, T, vocab)
    B, T, V = logits.shape
    loss = torch.nn.functional.cross_entropy(logits.view(B*T, V), yb.view(B*T))
    optimizer.zero_grad()
    loss.backward() # gradient chảy qua CÙNG kiến trúc
    optimizer.step() # nhích trọng số pretrained
```

⚠️ Learning rate fine-tuning _nhỏ hơn_ pretraining. Nếu để lớn, mô hình bị “giật” khỏi vùng đã học tốt và _quên_ kiến thức chung (catastrophic forgetting).

---

### 3. Bảng so sánh: pretraining vs fine-tuning

| Tiêu chí             | Pretraining                            | Fine-tuning                            |
| :------------------- | :------------------------------------- | :------------------------------------- |
| **Điểm khởi đầu**    | Trọng số _ngẫu nhiên_                  | Trọng số _pretrained_                  |
| **Dữ liệu**          | Khổng lồ, đa dạng (nghìn tỷ token)     | Nhỏ, chuyên biệt (nghìn - triệu token) |
| **Mục tiêu**         | Kiến thức _chung_ về ngôn ngữ/thế giới | Nhiệm vụ/miền _cụ thể_                 |
| **Chi phí**          | Rất cao (nghìn GPU, tuần/tháng)        | Thấp (1 GPU, giờ/ngày)                 |
| **Learning rate**    | Lớn hơn (vd $3\times 10^{-4}$)         | Nhỏ hơn (vd $3\times 10^{-5}$)         |
| **Số lần thực hiện** | Một lần, dùng chung                    | Nhiều lần, mỗi nhiệm vụ một bản        |
| **Giám sát**         | Tự giám sát (next-token)               | Tự giám sát _hoặc_ có nhãn             |

---

### 4. Vì sao ít dữ liệu vẫn fine-tune được?

Nếu mô hình lớn cần hàng nghìn tỷ token để pretraining, sao fine-tune chỉ vài nghìn ví dụ lại đủ?

Chìa khóa là _transfer learning_ (học chuyển giao): trong quá trình pretraining, các lớp đầu-giữa của Transformer đã học được _những đặc trưng dùng chung_ - cách biểu diễn từ, ngữ pháp, cấu trúc câu, quan hệ ngữ nghĩa. Fine-tune chỉ cần _tái sử dụng_ những đặc trưng này và _điều chỉnh nhẹ_ phần trên cho nhiệm vụ mới. Bạn không dạy lại “câu là gì”, chỉ dạy “trong lĩnh vực này, câu thường trông thế nào”.

**Analogy:** một đầu bếp đã thạo dao, lửa, gia vị (đặc trưng chung). Muốn học món Ý mới, họ chỉ cần công thức và vài lần luyện - không phải học cầm dao lại từ đầu.

```text
Pretraining (đắt, 1 lần)              Fine-tuning (rẻ, nhiều lần)
  ┌────────────────────────────┐        ┌────────────────────────────┐
  |  văn bản Internet khổng lồ |        |  dữ liệu miền nhỏ (y khoa)  |
  |            │                |        |            │                |
  |            v                |        |            v                |
  |  [ Transformer blocks ]  ───┼──────► |  [ CÙNG blocks, nhích nhẹ ] |
  |   học đặc trưng CHUNG        | copy   |   giữ nền tảng + chuyên hóa |
  |            │                | weights|            │                |
  |            v                |        |            v                |
  |   next-token trên web       |        |   next-token trên y khoa    |
  └────────────────────────────┘        └────────────────────────────┘
```

🎯 Đặc trưng đã học = “vốn liếng” miễn phí. Fine-tune = trả một khoản nhỏ để chuyển vốn đó sang việc mới. Đó là lý do vài nghìn ví dụ chất lượng cao thường đủ.

---

### 5. Các kiểu fine-tuning (từ nặng đến nhẹ)

Không phải lúc nào cũng phải cập nhật _toàn bộ_ trọng số. Có nhiều mức:

```python
# (A) FULL fine-tune: cập nhật MỌI tham số. Mạnh nhất, tốn bộ nhớ/tính toán nhất.
for p in model.parameters():
    p.requires_grad = True # tất cả tham số đều được cập nhật

# (B) PARTIAL fine-tune: đóng băng phần dưới, chỉ mở vài lớp trên (đỡ tốn, đỡ quên).
for p in model.parameters():
    p.requires_grad = False # đóng băng toàn bộ trước
for p in model.lm_head.parameters(): # chỉ mở lớp đầu ra (project sang vocab)
    p.requires_grad = True # chỉ lớp này có gradient

# (C) PEFT/LoRA (ý tưởng): giữ NGUYÊN trọng số gốc W, thêm 2 ma trận nhỏ A, B.
# W_hiệu_dụng = W + B @ A ; chỉ HỌC A, B (rất ít tham số, ~<1% mô hình).
#   W: (d, d) đóng băng ; A: (r, d), B: (d, r) với r nhỏ (vd r=8) -> học rất nhẹ.
```

| Kiểu                    | Số tham số cập nhật | Bộ nhớ/chi phí | Rủi ro quên | Khi nào dùng                        |
| :---------------------- | :------------------ | :------------- | :---------- | :---------------------------------- |
| **Full fine-tune**      | 100%                | Cao            | Cao hơn     | Có nhiều dữ liệu + GPU khỏe         |
| **Partial (đóng băng)** | Vài %               | Vừa            | Thấp hơn    | Dữ liệu ít, muốn giữ nền tảng       |
| **LoRA/PEFT**           | < 1%                | Rất thấp       | Thấp        | Ít dữ liệu, nhiều nhiệm vụ, GPU nhỏ |

💡 _LoRA_ (Low-Rank Adaptation) rất phổ biến: chỉ học 2 ma trận hạng thấp $A, B$ chèn vào; trọng số gốc $W$ được đóng băng. Bạn có thể lưu nhiều “bộ LoRA” cho nhiều nhiệm vụ mà vẫn dùng chung một mô hình nền.

---

### Nhìn sâu hơn

- **Fine-tuning KHÔNG phải RLHF:** Fine-tuning cổ điển vẫn là _next-token_ trên dữ liệu miền. Việc biến mô hình thành _trợ lý biết nghe lời_ (instruction tuning) rồi _căn chỉnh theo sở thích con người_ (RLHF/DPO) là bước _sau nữa_ - đó chính là nội dung _Bài 39_.
- **Catastrophic forgetting (quên thảm khốc):** nếu fine-tune quá mạnh trên miền hẹp, mô hình mất khả năng chung. Cách giảm: learning rate nhỏ, ít epoch, trộn thêm dữ liệu chung, hoặc dùng LoRA (không đụng trọng số gốc).
- **Overfitting khi dữ liệu ít:** vài trăm ví dụ mà train nhiều epoch, mô hình sẽ _học thuộc lòng_ thay vì tổng quát hóa. Theo dõi loss trên tập validation; dừng khi val loss tăng lại (early stopping).
- **Chi phí bộ nhớ:** full fine-tune cần lưu trọng số + gradient + trạng thái optimizer (AdamW ~2x tham số). Với mô hình lớn, LoRA tiết kiệm bộ nhớ khổng lồ vì chỉ tối ưu vài ma trận nhỏ.

---

### Thực hành

- **[Dễ]** Với `vocab_size = 5000`, giá trị loss _hợp lý_ ngay khi bắt đầu pretraining (chưa học gì) xấp xỉ bao nhiêu? (Gợi ý: $\ln(\text{vocab})$ - tính `math.log(5000)`).
- **[Trung bình]** Sửa đoạn code fine-tune ở Mục 5: đóng băng tất cả trừ _2 block cuối_ và `lm_head`. In ra tổng số tham số `requires_grad=True` bằng `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
- **[Thử thách]** Viết vòng lặp fine-tune có _early stopping_: sau mỗi epoch tính `val_loss`; nếu `val_loss` không giảm sau 3 lần liên tiếp thì dừng và nạp lại checkpoint tốt nhất. So sánh với việc train cứng 20 epoch xem cái nào val loss thấp hơn.

---

### Hiểu lầm thường gặp

- ❌ _“Fine-tune tức là huấn luyện lại mô hình từ đầu trên dữ liệu của tôi.”_ ➔ ✅ Fine-tune _bắt đầu từ trọng số pretrained_, chỉ nhích nhẹ; nếu train lại từ ngẫu nhiên thì đó là pretraining và cần cực nhiều dữ liệu.
- ❌ _“Ít dữ liệu thì chắc chắn không fine-tune được.”_ ➔ ✅ Nhờ transfer learning, vài nghìn (thậm chí vài trăm) ví dụ chất lượng cao thường đủ để chuyên hóa.
- ❌ _“Fine-tune xong là mô hình biết nghe lệnh như ChatGPT.”_ ➔ ✅ Biết nghe lệnh cần _instruction tuning + RLHF_ (Bài 39), khác với fine-tune next-token thông thường.
- ❌ _“Learning rate fine-tune nên bằng hoặc lớn hơn pretraining cho nhanh.”_ ➔ ✅ Learning rate fine-tune thường _nhỏ hơn_ để tránh phá nền tảng và catastrophic forgetting.
- ❌ _“Full fine-tune luôn tốt hơn LoRA.”_ ➔ ✅ Khi dữ liệu ít hoặc GPU nhỏ, LoRA thường _bằng hoặc tốt hơn_ mà rẻ hơn nhiều và ít quên hơn.

---

### Mẹo ghi nhớ

- _Pre = học rộng (đắt, một lần)_; _Fine = học hẹp (rẻ, nhiều lần)_.
- Fine-tune: _learning rate nhỏ, ít epoch, theo dõi val loss_.
- Ít dữ liệu → nghĩ _LoRA/partial_ trước, đừng vội full fine-tune.
- Thứ tự đúng: _pretraining → fine-tuning → instruction tuning/RLHF_.
- 🧠 Mẹo: “Sinh viên học 16 năm (pre), đi làm học 2 tuần (fine)”. Bạn không học lại mẫu giáo để đi làm.

---

### Tóm tắt trực quan

| Khái niệm                    | Một dòng                                                                       |
| :--------------------------- | :----------------------------------------------------------------------------- |
| **Pretraining**              | Học next-token trên dữ liệu khổng lồ, từ trọng số ngẫu nhiên → kiến thức chung |
| **Fine-tuning**              | Huấn luyện tiếp từ trọng số pretrained trên dữ liệu miền nhỏ, LR nhỏ           |
| **Transfer learning**        | Tái dùng đặc trưng đã học → ít dữ liệu vẫn đủ                                  |
| **Catastrophic forgetting**  | Fine-tune quá mạnh → quên kiến thức chung                                      |
| **Full / Partial / LoRA**    | Cập nhật 100% / vài % / <1% tham số                                            |
| **Cross-entropy next-token** | Cùng một hàm loss cho cả pretraining lẫn fine-tune next-token                  |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [[1hr Talk] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)
- PyTorch docs: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT (Karpathy): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Attention Is All You Need: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Hugging Face docs: [https://huggingface.co/docs](https://huggingface.co/docs)

---

### Tổng kết

- _Pretraining_ dạy mô hình _kiến thức chung_ qua next-token trên dữ liệu khổng lồ - đắt, làm một lần, dùng chung.
- _Fine-tuning_ _nhích_ trọng số pretrained trên dữ liệu miền nhỏ - rẻ, nhanh, làm nhiều lần cho nhiều nhiệm vụ.
- _Ít dữ liệu vẫn fine-tune được_ nhờ transfer learning: đặc trưng đã học sẵn được tái dùng.
- Nhớ hạ _learning rate_, theo dõi _val loss_, và cân nhắc _LoRA/partial_ để tránh quên và overfitting.

Vậy fine-tune next-token giúp mô hình _giỏi một miền_, nhưng vẫn chưa biết _nghe lệnh và trả lời như một trợ lý_. Ở _Bài 39: Instruction tuning và RLHF - cách ChatGPT/Claude được căn chỉnh_, ta sẽ xem cách dạy mô hình làm theo chỉ dẫn của con người và căn chỉnh theo sở thích, biến một mô hình “đoán chữ” thành một trợ lý thực sự hữu ích. Hẹn gặp ở bài kế! 🤖
