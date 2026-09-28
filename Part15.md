# Bài 15: Cài đặt một attention head hoàn chỉnh (PyTorch)

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 3:** Cơ chế Attention

---

### Giới thiệu

Hãy tưởng tượng bạn vừa mua đủ linh kiện để lắp một chiếc máy: mô-tơ, bánh răng, dây curoa, bộ phanh. Từng món bạn đã cầm nắm và hiểu công dụng - nhưng chỉ khi vặn chúng lại thành _một cụm hoàn chỉnh_, cỗ máy mới thật sự chạy. Bốn bài vừa qua chính là bốn linh kiện của cơ chế attention: bạn đã học _Query/Key/Value_ (Bài 12), phép _scaled dot-product_ chia cho $\sqrt{d_k}$ (Bài 13), và _causal mask_ tam giác để mô hình sinh văn bản không “nhìn trộm tương lai” (Bài 14).

Ở Bài 14, chúng ta đã che các vị trí tương lai bằng một ma trận tam giác. Nhưng đoạn code còn rời rạc: Q, K, V nằm mỗi nơi một dòng, mask áp thủ công. Bài này gộp _tất cả_ thành một `nn.Module` gọn gàng - một _Head_ hoàn chỉnh - sẵn sàng để bài sau xếp nhiều Head song song thành multi-head.

> 🔑 **Một câu để nhớ:** Một attention head = một khối `nn.Module` biến chuỗi vector vào thành chuỗi vector ra, nơi mỗi token “trộn” thông tin từ các token quá khứ theo trọng số $\text{softmax}(QK^T/\sqrt{d_k})$ đã che mask.

---

### Bạn sẽ học được gì?

- [ ] Gộp Linear cho Q/K/V + scaling + causal mask + dropout thành _một_ module `Head`.
- [ ] Hiểu vì sao dùng `register_buffer` cho mask (không phải parameter học được).
- [ ] Theo dõi _shape tensor_ đi qua từng bước, từ $(B, T, C)$ tới $(B, T, \text{head\_size})$.
- [ ] Biết chỗ đặt dropout và vì sao che mask _trước_ softmax.
- [ ] Chạy thử head trên dữ liệu giả và đọc ma trận attention.

📌 **Yêu cầu trước:** Bài 12 (Q/K/V), Bài 13 (scaled dot-product), Bài 14 (causal mask). Biết `nn.Linear`, `F.softmax`, phép nhân ma trận `@` và broadcasting cơ bản trong PyTorch.

---

### 1. Bức tranh lớn: một head làm gì?

**Analogy:** hãy coi một head như _một buổi họp lớp_. Mỗi học sinh (token) giơ tay hỏi “ai có thông tin liên quan tới mình?” - đó là _Query_. Mỗi bạn khác dán lên trán một tấm bảng tên mô tả “tôi biết về chủ đề này” - đó là _Key_. Nội dung thật sự bạn ấy muốn chia sẻ là _Value_. Học sinh đang hỏi so khớp Query của mình với Key của mọi người, ai khớp nhất thì “nghe” Value của người đó nhiều hơn.

Về mặt toán học, với đầu vào $x$ có shape $(B, T, C)$ (batch $B$ câu, mỗi câu $T$ token, mỗi token vector $C$ chiều), một head làm đúng bốn việc:

```text
1) Chiếu tuyến tính:  q = x Wq,  k = x Wk,  v = x Wv     (mỗi cái -> head_size chiều)
2) Điểm tương đồng:   scores = q @ kᵀ / sqrt(head_size)  (shape (B,T,T))
3) Che tương lai:     scores[i,j] = -inf  nếu  j > i     (causal mask tam giác)
4) Trọng số + trộn:   out = softmax(scores) @ v          (shape (B,T,head_size))
```

Công thức gọn của scaled dot-product attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

trong đó $d_k = \text{head\_size}$ là số chiều của Key/Query. Chia cho $\sqrt{d_k}$ giữ cho tích vô hướng không phình quá lớn khiến softmax “bão hòa” (Bài 13 đã giải thích).

💡 Đừng nhầm _attention_ với _feed-forward_: attention cho token _trao đổi thông tin với nhau_ (giao tiếp giữa các vị trí), còn feed-forward xử lý _mỗi token độc lập_ (tính toán riêng từng vị trí). Bài này chỉ lo phần giao tiếp.

---

### 2. Sơ đồ luồng dữ liệu và shape

Hãy theo dõi shape đi qua head với $B=2, T=4, C=8, \text{head\_size}=5$:

```text
x                (B, T, C)   = (2, 4, 8)
      | Linear Wk (C -> hs)
k                (B, T, hs)  = (2, 4, 5)
      | Linear Wq (C -> hs)
q                (B, T, hs)  = (2, 4, 5)
      | q @ k.transpose(-2,-1) / sqrt(hs)
scores           (B, T, T)   = (2, 4, 4)
      | masked_fill (tam giac duoi) + softmax
weights          (B, T, T)   = (2, 4, 4)   # moi hang cong lai = 1
      | Linear Wv (C -> hs)
v                (B, T, hs)  = (2, 4, 5)
      | weights @ v
out              (B, T, hs)  = (2, 4, 5)
```

Ma trận `scores` ($T \times T$) sau khi che mask có dạng tam giác dưới - token ở hàng $i$ chỉ được “chấm điểm” các cột $j \le i$:

```text
       k0    k1    k2    k3
 q0 [  0.9  -inf  -inf  -inf ]   token 0 chi thay chinh no
 q1 [  0.3   0.7  -inf  -inf ]   token 1 thay 0,1
 q2 [  0.1   0.2   0.7  -inf ]   token 2 thay 0,1,2
 q3 [  0.2   0.1   0.3   0.4 ]   token 3 thay tat ca qua khu
```

Sau softmax theo từng _hàng_, các `-inf` biến thành xác suất 0, phần còn lại của mỗi hàng cộng lại đúng bằng 1. Đó chính là “trọng số chú ý” để trộn Value.

---

### 3. Cài đặt module Head hoàn chỉnh

Giờ ghép tất cả vào một `nn.Module`. Chú ý mấy điểm thiết kế quan trọng trong comment (tiếng Anh):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """ One head of self-attention (causal / decoder style). """

    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        # Linear projections: no bias, following the Transformer convention.
        # Each maps C (n_embd) -> head_size.
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Lower-triangular causal mask. register_buffer => saved with the model
        # and moved by .to(device), but NOT a learnable parameter (no grad).
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # Dropout applied to the attention weights (regularization).
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Scaled dot-product scores: (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # Divide by sqrt(head_size) to keep variance stable (Bai 13).
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)

        # Causal mask: slice tril to current T, forbid attending to the future.
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Softmax over the LAST dim (keys) -> each row sums to 1.
        weights = F.softmax(scores, dim=-1) # (B, T, T)
        weights = self.dropout(weights)      # drop AFTER softmax

        v = self.value(x) # (B, T, head_size)
        out = weights @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
```

Vài quyết định thiết kế đáng nhớ:

- **`bias=False` cho Q/K/V:** theo quy ước Transformer, các phép chiếu attention thường bỏ bias vì LayerNorm phía trước đã căn giữa dữ liệu; bỏ bias giảm nhẹ số tham số mà gần như không mất chất lượng.
- **`register_buffer("tril", ...)` thay vì `self.tril = ...`:** mask là _hằng số_, không cần học. Dùng buffer để nó (a) tự đi theo `model.to(device)`, (b) được lưu trong `state_dict`, (c) _không_ bị optimizer cập nhật. Nếu để làm `nn.Parameter` thì mô hình sẽ vô tình “học” cả cái mask - sai hoàn toàn.
- **Che mask TRƯỚC softmax:** đặt `-inf` vào ô bị cấm, sau softmax nó thành 0. Nếu bạn softmax trước rồi mới zero-out thì tổng mỗi hàng không còn bằng 1 $\to$ sai xác suất.
- **Dropout SAU softmax:** ta drop trên _trọng số chú ý_ (giống bản gốc “Attention Is All You Need”), để mô hình không phụ thuộc quá mức vào một vài cặp token.

---

### 4. Chạy thử trên dữ liệu giả

```python
torch.manual_seed(1337)
B, T, C = 2, 4, 8 # batch=2, block(time)=4, embedding dim=8
head_size, block_size = 5, 8
head = Head(n_embd=C, head_size=head_size, block_size=block_size, dropout=0.0)

x = torch.randn(B, T, C) # fake batch of token embeddings: (2, 4, 8)
out = head(x)
print(out.shape) # -> torch.Size([2, 4, 5])  == (B, T, head_size)
```

Kiểm tra trọng số chú ý có đúng “nhân quả” không (đặt `dropout=0` để đọc cho rõ). Ta gọi lại các bước bên trong với một mẫu:

```python
# Recompute weights for the first sample to inspect them.
k = head.key(x); q = head.query(x)
scores = q @ k.transpose(-2, -1) * (head_size ** -0.5)
scores = scores.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)
print(weights[0])
# Example output (values vary by seed):
# tensor([[1.0000, 0.0000, 0.0000, 0.0000],
#         [0.4712, 0.5288, 0.0000, 0.0000],
#         [0.3011, 0.3684, 0.3305, 0.0000],
#         [0.2489, 0.2503, 0.2607, 0.2401]])
print(weights[0].sum(dim=-1)) # -> tensor([1., 1., 1., 1.])  each row sums to 1
```

Đọc kết quả: token đầu tiên chỉ chú ý tới chính nó (1.0), các token sau chia đều dần cho những token quá khứ, và _mọi_ ô phía trên đường chéo đều bằng 0 - đúng tính nhân quả mà Bài 14 yêu cầu. ✅

---

### Nhìn sâu hơn

_Chi phí bộ nhớ và tính toán._ Ma trận `scores` có shape $(B, T, T)$. Nghĩa là bộ nhớ và số phép nhân _tăng theo $T^2$_ (bình phương độ dài chuỗi). Với `block_size` nhỏ (vài trăm) thì ổn, nhưng khi muốn ngữ cảnh dài hàng chục nghìn token, chính phần $q @ k^T$ này trở thành nút thắt - lý do sinh ra hàng loạt nghiên cứu attention hiệu quả (FlashAttention, attention thưa…). Ta sẽ chạm tới các tối ưu này ở cuối khóa.

_`head_size` là gì so với `n_embd`?_ `n_embd` ($C$) là chiều embedding của mỗi token đi vào; `head_size` là chiều _trong_ của một head. Ở bài sau (multi-head), thường đặt `head_size = n_embd / n_head` để khi ghép nhiều head lại, tổng chiều đầu ra vẫn bằng `n_embd`. Ví dụ $n\_embd=512, n\_head=8 \to head\_size=64$.

_`transpose(-2, -1)` chứ không phải `.T`._ Với tensor 3 chiều $(B, T, hs)$, ta chỉ muốn hoán hai chiều _cuối_ để thành $(B, hs, T)$, giữ nguyên chiều batch. `.T` sẽ đảo toàn bộ và làm sai batch.

⚠️ Buffer `tril` được cắt `[:T, :T]` mỗi lần forward. Điều này cho phép head hoạt động với chuỗi ngắn hơn `block_size` (ví dụ lúc sinh văn bản token đầu tiên), nhưng nếu $T > block\_size$ sẽ lỗi index - đó là lý do ta luôn cắt ngữ cảnh về tối đa `block_size` khi generate.

---

### Thực hành

- **[Dễ]** Đổi `head_size` từ 5 sang 16 rồi in `out.shape`. Nó phải thành `(2, 4, 16)`. Giải thích vì sao $C$ (đầu vào) không cần đổi.
- **[Trung bình]** Bỏ dòng `masked_fill(...)` đi rồi in lại `weights[0]`. Bạn sẽ thấy hàng đầu tiên _không còn_ là `[1, 0, 0, 0]` nữa mà chú ý cả tương lai. Hãy giải thích tại sao điều này làm hỏng mô hình sinh văn bản (gợi ý: “nhìn trộm” đáp án token kế tiếp).
- **[Thử thách]** Thêm một tham số `causal=True/False` vào `__init__`. Khi `causal=False` thì _không_ áp mask (kiểu attention của encoder / phân loại câu). Kiểm chứng: với `causal=False`, mỗi hàng của `weights` phải cộng lại bằng 1 nhưng không còn dạng tam giác.

---

### Hiểu lầm thường gặp

- ❌ “Mask nên là một `nn.Parameter` để mô hình tự học chỗ nào cần che.” ➔ ✅ Mask là _quy tắc cố định_ (không nhìn tương lai), phải dùng `register_buffer` - không có gradient, không được học.
- ❌ “Cứ softmax trước rồi mới gán 0 cho các ô tương lai cũng như nhau.” ➔ ✅ Sai. Phải gán `-inf` _trước_ softmax để tổng mỗi hàng vẫn bằng 1. Zero-out sau softmax làm tổng $< 1 \to$ phân phối lệch.
- ❌ “Attention xử lý token tuần tự từng cái như RNN.” ➔ ✅ Attention tính _song song_ cho mọi cặp token cùng lúc qua phép nhân ma trận; mask chỉ giới hạn thông tin, không làm nó tuần tự.
- ❌ “`head_size` phải bằng `n_embd`.” ➔ ✅ Không. `head_size` là siêu tham số riêng; multi-head thường đặt nó nhỏ hơn (`n_embd / n_head`).

---

### Mẹo ghi nhớ

- Nhớ 4 bước: _chiếu → chấm điểm chia $\sqrt{d_k}$ → che mask → softmax rồi trộn Value_.
- Nhớ shape: vào $(B, T, C)$, scores $(B, T, T)$, ra $(B, T, \text{head\_size})$.
- Mẹo: đọc `masked_fill(tril==0, -inf)` là “chỗ nào tril bằng 0 (ô tương lai) thì nhét âm vô cực vào”.
- `buffer` = “đồ đạc theo model nhưng không học”; `parameter` = “đồ được học”.

---

### Tóm tắt trực quan

| Khái niệm           | Một dòng                                                                     |
| :------------------ | :--------------------------------------------------------------------------- |
| **Head module**     | Gộp Q/K/V + scale + mask + dropout thành 1 khối `nn.Module`                  |
| **Q / K / V**       | 3 `nn.Linear(n_embd, head_size, bias=False)`                                 |
| **Scaling**         | Nhân scores với `head_size ** -0.5` (chia $\sqrt{d_k}$) giữ variance ổn định |
| **Causal mask**     | `masked_fill(tril==0, -inf)` trước softmax, tam giác dưới                    |
| **register_buffer** | Mask cố định, đi theo device, KHÔNG học                                      |
| **Dropout**         | Áp lên trọng số attention, SAU softmax                                       |
| **Shape ra**        | $(B, T, \text{head\_size})$; scores tốn bộ nhớ theo $T^2$                    |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Tutorial 07: Coding Self-Attention from Scratch | Trainable Key, Value, and Query in PyTorch](https://www.youtube.com/watch?v=xYb-FTOkkGE)
- PyTorch `nn.Module`, `nn.Linear`, `nn.Dropout`: [pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT (Andrej Karpathy) - tham chiếu triển khai Head/Block: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- “Attention Is All You Need” (bản gốc scaled dot-product attention): [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

---

### Tổng kết

- Một _attention head_ hoàn chỉnh chỉ gồm 3 `nn.Linear` (Q/K/V) + scaling $\sqrt{d_k}$ + causal mask (qua `register_buffer`) + dropout, gói trong một `nn.Module` với forward sạch sẽ.
- Shape đi từ $(B, T, C) \to$ scores $(B, T, T) \to$ ra $(B, T, \text{head\_size})$; luôn che mask _trước_ softmax và drop _sau_ softmax.
- Ta đã chạy thử trên dữ liệu giả và xác nhận trọng số chú ý đúng tính nhân quả, mỗi hàng cộng lại bằng 1.

Nhưng một góc nhìn duy nhất thì hạn hẹp: token có thể vừa muốn để ý tới _cú pháp_ (chủ ngữ - động từ) vừa muốn để ý tới _ngữ nghĩa_ (chủ đề). Ở _Bài 16: Multi-head attention - nhiều góc nhìn song song_, chúng ta sẽ xếp nhiều Head chạy song song rồi nối kết quả lại, để mô hình “nhìn” chuỗi từ nhiều lăng kính cùng lúc. Hẹn gặp ở bài kế tiếp! 🤖
