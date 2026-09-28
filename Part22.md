# Bài 22: Token embedding + positional embedding + lm_head

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 5:** Xây dựng GPT mini

---

### Giới thiệu

Ở _Bài 21_, mình đã xếp chồng nhiều Transformer block lại với nhau và thấy phần “thân” của GPT dần thành hình. Nhưng một chồng block, dù đẹp đến mấy, vẫn chỉ là một cái máy xử lý _vector_. Nó không biết đọc chữ, và nó cũng không tự phun ra được từ tiếp theo. Chúng ta còn thiếu _hai đầu_: một cái đầu vào biến chữ thành số (embedding), và một cái đầu ra biến số thành “điểm số cho từng từ” (`lm_head`).

Hãy tưởng tượng khối Transformer như một _phòng họp toàn chuyên gia_ cực giỏi, nhưng họ chỉ nói được bằng “ngôn ngữ vector”. Muốn nhờ họ, bạn phải: (1) _phiên dịch_ câu chữ của bạn thành vector để đưa vào phòng (đó là token embedding), (2) kèm theo _số thứ tự chỗ ngồi_ để họ biết ai nói trước ai nói sau (positional embedding), và (3) khi họ họp xong, _phiên dịch ngược_ kết quả ra thành “gợi ý từ tiếp theo” (`lm_head`). Bài này chính là hai người phiên dịch ở hai đầu đó.

> 🔑 **Một câu để nhớ:** Đầu vào = token embedding + positional embedding (chữ hóa số + đánh dấu vị trí); đầu ra = `lm_head` (một Linear chiếu về `vocab_size` để ra logits).

---

### Bạn sẽ học được gì?

- [ ] Vì sao cần _token embedding_ thay vì one-hot, và nó khác one-hot ra sao.
- [ ] Vì sao Transformer BẮT BUỘC cần _positional embedding_ (không như RNN).
- [ ] Cách _cộng_ hai embedding lại thành đầu vào cho các block (và vì sao cộng được).
- [ ] `lm_head` là gì: một `nn.Linear` chiếu ra _logits_ trên toàn bộ vocabulary.
- [ ] Phân biệt rõ _logits_ và _xác suất_; khi nào dùng cái nào.
- [ ] _Weight tying_: dùng chung trọng số embedding và `lm_head` để tiết kiệm tham số.

📌 **Yêu cầu trước:** Bài 21 (xếp chồng block), hiểu `nn.Embedding`, `nn.Linear`, softmax và cross-entropy next-token.

---

### 1. Token embedding: từ điển tra vector

Máy tính không hiểu chữ “mèo”. Sau bước tokenization, mỗi token đã thành một _số nguyên_ (ID), ví dụ `"mèo" -> 42`. Nhưng số 42 tự nó vô nghĩa: 42 không “lớn hơn” hay “gần” 41 về mặt ngữ nghĩa.

Cách ngây thơ là _one-hot_: token 42 trong vocab 50000 từ thành một vector dài 50000 chiều toàn số 0, chỉ vị trí thứ 42 là 1. Vấn đề: (a) tốn khủng khiếp (50000 chiều cho mỗi token), (b) mọi token “cách đều” nhau, không mã hóa được “mèo gần chó hơn gần bàn”.

_Embedding_ giải quyết cả hai: nó là một _bảng tra cứu_ (lookup table) học được. Mỗi token ID trỏ tới một hàng là vector dày (dense) `n_embd` chiều (vd 384). Analogy: như _tra từ điển_ - đưa vào một từ, nhận về “trang định nghĩa” của nó (một vector giàu ý nghĩa mà mô hình tự học).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

vocab_size = 50257 # e.g. GPT-2 BPE vocab
n_embd = 384     # embedding dimension
tok_emb_table = nn.Embedding(vocab_size, n_embd) # weight shape: (vocab_size, n_embd)

# idx: a batch of token IDs. shape (B, T) = (batch, sequence length)
idx = torch.tensor([[42, 7, 999], [1, 2, 3]]) # (B=2, T=3)
tok_emb = tok_emb_table(idx) # (B, T, n_embd) = (2, 3, 384)
print(tok_emb.shape) # torch.Size([2, 3, 384])
```

💡 `nn.Embedding` KHÔNG phải phép nhân ma trận one-hot thật sự - nó chỉ _lấy hàng_ (index select) nên rất nhanh. Về mặt toán học nó tương đương `one_hot(idx) @ W`, nhưng cài đặt là tra hàng trực tiếp.

---

### 2. Positional embedding: đánh số chỗ ngồi

Đây là chỗ Transformer khác hẳn RNN. RNN đọc token _tuần tự_ nên tự biết thứ tự. Còn self-attention nhìn _tất cả token cùng lúc_ và _hoán vị bất biến_ (permutation-invariant): nếu bạn xáo trộn thứ tự token, attention thuần túy cho ra kết quả… như nhau! Với nó, “mèo đuổi chuột” và “chuột đuổi mèo” trông y hệt.

⚠️ Không có thông tin vị trí, mô hình sẽ mất hoàn toàn khái niệm thứ tự - thảm họa cho ngôn ngữ.

Cách sửa (kiểu GPT gốc, cũng là kiểu nanoGPT làm): dùng một _positional embedding học được_ - một bảng thứ hai, tra theo _vị trí_ 0, 1, 2, ... chứ không theo token ID. Analogy: mỗi ghế trong phòng họp có một _tấm biển số ghế_; ai ngồi ghế đó cũng được gắn thêm “tôi đang ở vị trí thứ k”.

```python
block_size = 256 # max context length (T <= block_size)
pos_emb_table = nn.Embedding(block_size, n_embd) # weight: (block_size, n_embd)

B, T = idx.shape # (2, 3)
pos = torch.arange(T) # [0, 1, 2], shape (T,)
pos_emb = pos_emb_table(pos) # (T, n_embd) = (3, 384)

# Add token + position. Broadcasting: (B,T,C) + (T,C) -> (B,T,C)
x = tok_emb + pos_emb # (B, T, n_embd) = (2, 3, 384)
print(x.shape) # torch.Size([2, 3, 384])
```

Vì sao _cộng_ được (chứ không phải nối)? Vì cả hai cùng sống trong không gian `n_embd` chiều; cộng lại cho mỗi token một vector “vừa mang ý nghĩa từ, vừa mang dấu vết vị trí”. Mô hình tự học tách bạch hai loại thông tin này trong lúc huấn luyện. (Lưu ý: đây là _learned absolute positional embedding_, khác với _sinusoidal_ trong paper gốc và khác _RoPE_ của các mô hình mới - sẽ nhắc ở mục sau.)

---

### 3. Đầu vào chảy vào các block

Sơ đồ luồng dữ liệu qua toàn bộ GPT mini (đây chính là bức tranh Bài 23 sẽ ghép hoàn chỉnh):

```text
idx (B, T)  token IDs, integers
     |
     |  tok_emb_table (lookup)        pos_emb_table (lookup on arange(T))
     v                                       |
  tok_emb (B,T,C) ------ (+) ------ pos_emb (T,C)   # broadcast add
                          |
                          v
                    x (B, T, C)              # C = n_embd
                          |
                 [ Transformer block 1 ]     # attention + FFN + residual + LayerNorm
                          |
                 [ Transformer block 2 ]
                          |
                         ...
                 [ Transformer block N ]
                          |
                    LayerNorm (final ln_f)   # (B, T, C)
                          |
                 lm_head: Linear(C -> vocab) # (B, T, vocab_size)
                          |
                    logits (B, T, vocab_size)
```

Shape đi từ `(B, T)` integer -> `(B, T, C)` vector đi xuyên các block giữ nguyên -> cuối cùng bung ra `(B, T, vocab_size)`.

---

### 4. lm_head: chiếu ra logits trên vocabulary

Sau khi qua hết các block và một LayerNorm cuối (`ln_f`), mỗi vị trí có một vector `n_embd` chiều đã “chín”. Giờ ta cần: _với mỗi vị trí, chấm điểm cho từng từ trong vocab_ xem từ nào có khả năng là token tiếp theo. Đó là việc của `lm_head` - một `nn.Linear` không hơn không kém, chiếu từ `n_embd` về `vocab_size`.

Analogy: như một _hội đồng chấm thi_ - đưa vào bài làm (vector trạng thái), hội đồng cho ra _điểm thô_ (logits) cho từng “thí sinh” (từ trong vocab). Chưa phải xác suất - chỉ là điểm.

```python
n_layer = 6
ln_f = nn.LayerNorm(n_embd) # final layer norm
lm_head = nn.Linear(n_embd, vocab_size, bias=False) # (C -> vocab_size)

# Pretend x already went through N transformer blocks; here we just normalize.
x = ln_f(x) # (B, T, n_embd) = (2, 3, 384)
logits = lm_head(x) # (B, T, vocab_size) = (2, 3, 50257)
print(logits.shape) # torch.Size([2, 3, 50257])

# logits are raw scores, NOT probabilities. Turn into probs only when needed:
probs = F.softmax(logits, dim=-1) # (B, T, vocab_size), each row sums to 1
print(probs[0, -1].sum()) # tensor(1.0000..) approximately 1.0
```

🎯 **logits vs xác suất:** logits là số thực bất kỳ (có thể âm, có thể lớn). Chỉ khi qua softmax chúng mới thành _xác suất_ (không âm, tổng = 1). Khi _huấn luyện_ ta thường KHÔNG tự softmax - `F.cross_entropy` nhận thẳng logits (nó tự làm log-softmax bên trong, ổn định số học hơn).

#### Tính loss next-token

```python
# targets: the next token for each position, shape (B, T)
targets = torch.tensor([[7, 999, 5], [2, 3, 4]]) # (B=2, T=3)

# cross_entropy expects logits (N, C) and targets (N,), so flatten B*T:
B, T, C = logits.shape # (2, 3, 50257)
loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
print(loss.item()) # a single scalar, e.g. ~10.8 for random init (~ln(vocab_size))
```

💡 Với init ngẫu nhiên, loss ban đầu xấp xỉ $\ln(\text{vocab\_size})$ (vì mô hình đoán đều nhau). Với vocab ~50257 thì $\ln(50257) \approx 10.82$. Thấy loss khởi đầu quanh con số đó là dấu hiệu bảng logits/embedding được nối đúng.

---

### 5. Weight tying: dùng chung trọng số

Để ý: bảng token embedding có shape `(vocab_size, n_embd)`, còn `lm_head` (bỏ bias) có ma trận trọng số shape `(vocab_size, n_embd)` - _giống hệt_! Ý tưởng “weight tying”: _buộc hai cái là MỘT_ ma trận chung. Trực giác: ánh xạ “token -> vector” (embedding) và “vector -> điểm cho token” (`lm_head`) là hai chiều của cùng một mối quan hệ token ↔ vector, chia sẻ được.

```python
# Weight tying: share the same weight matrix for input embedding and output head.
lm_head.weight = tok_emb_table.weight # both now point to one (vocab_size, n_embd) tensor
```

Lợi ích: _giảm số tham số_ đáng kể (một bảng `vocab × n_embd` không nhỏ), và thường _cải thiện chất lượng_ một chút. Đây là kỹ thuật chuẩn trong GPT-2 và nanoGPT.

⚠️ Weight tying chỉ hợp lệ khi `lm_head` KHÔNG có bias và kích thước khớp `(vocab_size, n_embd)`. Nếu bạn tự nhân matmul thủ công thì nhớ dùng đúng transpose.

---

### Nhìn sâu hơn

- **Chi phí tham số của hai đầu:** riêng embedding + `lm_head` đã là $\text{vocab\_size} \times \text{n\_embd}$ tham số MỖI cái. Với vocab 50257 và `n_embd` 384 là ~19.3 triệu tham số/bảng - rất lớn so với thân mô hình nhỏ. Đây là lý do weight tying tiết kiệm rất đáng kể (gộp hai thành một).
- **Vì sao cần LayerNorm cuối (`ln_f`) trước `lm_head`?** Kiến trúc pre-norm (GPT-2/nanoGPT) đặt LayerNorm trước mỗi sub-layer; đường residual đi thẳng đến cuối nên cần một LayerNorm chốt trước khi chiếu ra logits để chuẩn hóa phân phối đầu ra.
- **Positional embedding học được có giới hạn:** bảng chỉ có `block_size` hàng, nên độ dài ngữ cảnh bị chặn cứng ở `block_size`. Muốn ngữ cảnh dài hơn phải mở rộng bảng hoặc dùng phương án khác (sinusoidal, RoPE, ALiBi) - các mô hình hiện đại chuộng RoPE vì tổng quát hóa độ dài tốt hơn.
- **Sinusoidal vs learned:** paper “Attention Is All You Need” dùng vị trí bằng hàm sin/cos cố định (không học). GPT chọn learned absolute - đơn giản, hiệu quả trong giới hạn `block_size`.

---

### Thực hành

- **[Dễ]** Tạo `tok_emb = nn.Embedding(1000, 64)` và `pos_emb = nn.Embedding(128, 64)`. Với `idx` shape `(4, 10)`, in shape của `tok_emb(idx)`, `pos_emb(torch.arange(10))`, và tổng của chúng. Xác nhận broadcasting cho ra `(4, 10, 64)`.
- **[Trung bình]** Cho `logits` shape `(2, 5, 100)`. Viết code lấy phân phối xác suất tại _vị trí cuối cùng_ của mỗi câu (`logits[:, -1, :]`), softmax nó, và kiểm tra tổng mỗi hàng $\approx 1$. Đây chính là bước dùng khi _generate_ token kế tiếp.
- **[Thử thách]** Cài `lm_head = nn.Linear(n_embd, vocab_size, bias=False)` rồi thực hiện weight tying với bảng token embedding. Đếm tổng tham số của model TRƯỚC và SAU khi tie (dùng `sum(p.numel() for p in model.parameters())`) và giải thích con số chênh lệch đúng bằng $\text{vocab\_size} \times \text{n\_embd}$.

---

### Hiểu lầm thường gặp

- ❌ “Embedding chỉ là one-hot đổi tên.” → ✅ One-hot là vector thưa, dài `vocab_size`, mọi token cách đều nhau. Embedding là vector dày `n_embd` chiều _học được_, mã hóa được sự tương đồng ngữ nghĩa.
- ❌ “Transformer tự biết thứ tự token như RNN.” → ✅ Self-attention _hoán vị bất biến_; phải cộng positional embedding thì mô hình mới biết thứ tự.
- ❌ “logits chính là xác suất.” → ✅ logits là điểm thô (số thực bất kỳ); phải qua softmax mới thành xác suất tổng bằng 1.
- ❌ “Phải softmax rồi mới tính loss.” → ✅ `F.cross_entropy` nhận _logits_ trực tiếp (tự làm log-softmax bên trong, ổn định số học hơn). Đừng softmax hai lần.
- ❌ “Nối (concat) token và positional embedding.” → ✅ Kiểu GPT _cộng_ chúng (cùng chiều `n_embd`), không nối.

---

### Mẹo ghi nhớ

- Đầu vào = _tok + pos_ (nội dung + vị trí). Đầu ra = _lm_head_ (Linear ra vocab).
- Nhớ shape: `(B,T) → embedding → (B,T,C) → blocks → (B,T,C) → lm_head → (B,T,vocab)`.
- _Mẹo:_ “Số vào - vector giữa - logits ra.” Ba chặng: integer → dense vector → điểm cho từng từ.
- Loss init $\approx \ln(\text{vocab\_size})$ là bài kiểm tra nhanh xem hai đầu nối đúng chưa.

---

### Tóm tắt trực quan

| Khái niệm                | Một dòng                                                         |
| :----------------------- | :--------------------------------------------------------------- |
| **Token embedding**      | Bảng tra `(vocab_size, n_embd)`: token ID → vector dày           |
| **Positional embedding** | Bảng tra `(block_size, n_embd)`: vị trí → vector, cộng vào token |
| **Đầu vào block**        | `x = tok_emb + pos_emb`, shape `(B, T, n_embd)`                  |
| **ln_f**                 | LayerNorm cuối trước `lm_head` (kiến trúc pre-norm)              |
| **lm_head**              | `nn.Linear(n_embd, vocab_size)` → logits `(B, T, vocab_size)`    |
| **logits vs xác suất**   | logits = điểm thô; softmax mới ra xác suất                       |
| **Weight tying**         | Dùng chung trọng số embedding và `lm_head`, tiết kiệm tham số    |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Let’s build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- nanoGPT (Andrej Karpathy) - xem `GPT.__init__` với `wte`, `wpe`, `lm_head`, weight tying: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- `nn.Embedding` và `nn.Linear`: [pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- The Illustrated Transformer (positional encoding): [jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Attention Is All You Need (positional encoding gốc): [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Tài liệu Hugging Face (kiến trúc GPT-2): [huggingface.co/docs](https://huggingface.co/docs)

---

### Tổng kết

- Token embedding biến ID số thành vector dày học được (khác one-hot); positional embedding thêm thông tin vị trí mà self-attention không tự có.
- Đầu vào các block = `tok_emb + pos_emb`, shape `(B, T, n_embd)`; sau `ln_f`, `lm_head` (một Linear) chiếu ra _logits_ `(B, T, vocab_size)`.
- _logits ≠ xác suất_: softmax mới ra xác suất; huấn luyện dùng `cross_entropy` trên logits.
- _Weight tying_ gộp trọng số embedding và `lm_head` để tiết kiệm tham số và thường tốt hơn.

Giờ ta đã có đủ hai đầu (embedding vào, `lm_head` ra) và cả thân (chồng block ở Bài 21). Ở _Bài 23: Ghép toàn bộ thành một mô hình GPT hoàn chỉnh_, mình sẽ đóng tất cả vào một class `GPT` duy nhất - forward nhận `idx`, `targets` và trả về `logits` cùng `loss` - sẵn sàng để huấn luyện. Hẹn gặp ở bài kế! 🤖
