# Bài 14: Causal mask cho mô hình sinh văn bản

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 3:** Cơ chế Attention

---

### Giới thiệu

Ở _Bài 13_, bạn đã lắp ráp trái tim của Transformer: _scaled dot-product attention_. Ta tính điểm tương đồng giữa các token bằng $Q @ K^T$, chia cho $\sqrt{d_k}$ để giữ độ lớn ổn định, đưa qua softmax thành trọng số, rồi trộn các vector $V$ lại. Kết quả là mỗi token được phép “nhìn” và tổng hợp thông tin từ các token khác. Rất mạnh! Nhưng… có một lỗ hổng chết người nếu ta muốn _sinh văn bản_. 🤖

Hãy tưởng tượng bạn đang thi viết một câu chuyện, mỗi lượt viết đúng _một chữ_ rồi đoán chữ tiếp theo. Nếu ai đó lén cho bạn xem trước những chữ ở _tương lai_ của bài mẫu, thì việc “đoán” trở thành “chép” - dễ như ăn kẹo lúc luyện tập, nhưng lúc thi thật (không có tương lai để nhìn) bạn sẽ đơ hoàn toàn. Đó chính xác là điều xảy ra nếu attention của ta cho token nhìn thấy các token phía sau nó.

Hôm nay chúng ta học cách bịt mắt tương lai lại: _causal mask_ (mặt nạ nhân quả). Đây là mảnh ghép biến attention thông thường thành attention _tự hồi quy_ (autoregressive) - nền tảng của mọi mô hình kiểu GPT.

> 🔑 **Một câu để nhớ:** Causal mask đặt điểm attention của mọi vị trí “tương lai” thành `-inf` _trước_ khi softmax, khiến softmax gán cho chúng trọng số 0 - nhờ đó token tại vị trí $t$ chỉ được nhìn các vị trí $0..t$.

---

### Bạn sẽ học được gì?

- [ ] Hiểu vì sao mô hình sinh văn bản (decoder) BẮT BUỘC không được nhìn tương lai
- [ ] Dựng ma trận mask tam giác dưới bằng `torch.tril`
- [ ] Áp mask vào điểm attention bằng `masked_fill(..., float('-inf'))` _trước_ softmax
- [ ] Hiểu vì sao `-inf` cho ra trọng số đúng bằng 0 sau softmax
- [ ] Phân biệt attention _hai chiều_ (encoder, kiểu BERT) và attention _nhân quả_ (decoder, kiểu GPT)

📌 **Yêu cầu trước:** Bài 13 (scaled dot-product attention), quen `softmax`, và biết cách nhân ma trận trong PyTorch.

---

### 1. Vì sao không được nhìn tương lai?

Mục tiêu huấn luyện của GPT là _dự đoán token kế tiếp_ (next-token prediction). Với một chuỗi `[c1, c2, c3, c4]`, mô hình học 4 bài toán con _song song_ trong một lần forward:

- Nhìn `c1` -> đoán `c2`
- Nhìn `c1, c2` -> đoán `c3`
- Nhìn `c1, c2, c3` -> đoán `c4`
- Nhìn `c1, c2, c3, c4` -> đoán `c5`

Nếu tại bước “nhìn `c1` đoán `c2`” mà attention lỡ cho `c1` thấy luôn `c2`, thì đáp án bị lộ. Loss sẽ tụt về gần 0 trên tập train một cách gian lận, nhưng lúc sinh văn bản thật (chưa hề có `c2`) mô hình vô dụng. Ta gọi lỗi này là _information leakage_ (rò rỉ thông tin từ tương lai).

⚠️ Đây KHÔNG phải chuyện “cho đẹp”. Quên causal mask là một trong những bug âm thầm nguy hiểm nhất: train loss đẹp bất thường nhưng generate ra rác.

---

### 2. Ma trận tam giác dưới - hình dạng của "quá khứ"

Attention cho ta một ma trận điểm số `scores` kích thước $(T, T)$ với $T$ là độ dài chuỗi (block size). Phần tử `scores[i][j]` là “token thứ $i$ chú ý tới token thứ $j$ bao nhiêu”. Ta muốn: token $i$ _chỉ được_ nhìn $j \le i$.

Những ô hợp lệ ($j \le i$) tạo thành một _tam giác dưới_ (lower-triangular). Với $T = 4$:

```text
key        j=0  j=1  j=2  j=3
query i=0 [  .    x    x    x  ]   -> chi thay j=0
query i=1 [  .    .    x    x  ]   -> thay j=0,1
query i=2 [  .    .    .    x  ]   -> thay j=0,1,2
query i=3 [  .    .    .    .  ]   -> thay j=0,1,2,3

  '.' = duoc phep (giu nguyen diem)
  'x' = tuong lai (bit lai, dat -inf)
```

Ma trận mask là tam giác dưới toàn số 1 (phần được phép), phần trên đường chéo là 0 (phần cấm):

```text
tril(ones(4,4)) =
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

Trong PyTorch, `torch.tril` (triangular-lower) làm đúng việc này:

```python
import torch

T = 4
# lower-triangular ones: 1 where allowed (j <= i), 0 where future
mask = torch.tril(torch.ones(T, T))
print(mask)
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])   shape: (T, T) = (4, 4)
```

---

### 3. Áp mask: đặt tương lai thành -inf TRƯỚC softmax

Đây là chi tiết dễ sai nhất, nên đọc chậm. Ta KHÔNG nhân mask vào sau softmax, mà _thay_ các điểm số ở vùng cấm bằng `-inf` _trước_ khi softmax.

Vì sao là `-inf`? Nhắc lại công thức softmax cho một hàng điểm số $s$:

$$\text{softmax}(s)_j = \frac{\exp(s_j)}{\sum_k \exp(s_k)}$$

Khi $s_j = -\infty$ thì $\exp(-\infty) = 0$. Số hạng đó biến mất khỏi cả tử và mẫu, nên trọng số của vị trí tương lai đúng bằng 0, còn các vị trí hợp lệ vẫn tự chuẩn hóa lại thành tổng bằng 1. Gọn gàng và _chính xác về toán học_.

⚠️ Nếu bạn đặt 0 thay vì `-inf` trước softmax, $\exp(0) = 1 \to$ tương lai vẫn được một phần trọng số dương $\to$ rò rỉ. Nếu bạn nhân mask (0/1) SAU softmax rồi không chuẩn hóa lại, tổng trọng số của các hàng đầu sẽ $< 1 \to$ sai lệch. Vì vậy luôn dùng `masked_fill(mask == 0, -inf)` TRƯỚC softmax.

Ghép vào scaled dot-product attention của Bài 13:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, d_k = 1, 4, 8 # batch=1, seq_len=4, key dim=8
q = torch.randn(B, T, d_k) # queries, shape (B, T, d_k)
k = torch.randn(B, T, d_k) # keys,    shape (B, T, d_k)
v = torch.randn(B, T, d_k) # values,  shape (B, T, d_k)

# 1) scaled dot-product scores: (B, T, d_k) @ (B, d_k, T) -> (B, T, T)
scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)

# 2) build causal mask once (lower-triangular ones), shape (T, T)
mask = torch.tril(torch.ones(T, T))

# 3) set future positions to -inf BEFORE softmax
scores = scores.masked_fill(mask == 0, float('-inf')) # (B, T, T)

# 4) softmax over the KEY dimension (last dim) -> attention weights
attn = F.softmax(scores, dim=-1) # (B, T, T), each row sums to 1
print(attn[0])
# tensor([[1.00, 0.00, 0.00, 0.00],   <- token 0 only sees itself
#         [0.53, 0.47, 0.00, 0.00],   <- token 1 sees {0,1}
#         [0.31, 0.42, 0.27, 0.00],   <- token 2 sees {0,1,2}
#         [0.22, 0.19, 0.34, 0.25]])  <- token 3 sees {0,1,2,3}

# 5) weighted sum of values: (B, T, T) @ (B, T, d_k) -> (B, T, d_k)
out = attn @ v
print(out.shape) # torch.Size([1, 4, 8])
```

Hãy để ý _hình tam giác_ trong `attn`: mọi ô phía trên đường chéo đều bằng 0.00, và mỗi hàng vẫn cộng lại đúng 1. Token 0 chỉ có một lựa chọn (chính nó) nên trọng số là 1.00.

---

### 4. Sơ đồ toàn cảnh: mask nằm ở đâu trong luồng?

```text
tokens -> embedding -> [ Q, K, V ]
                             |
                    scores = Q K^T / sqrt(d_k)     shape (T, T)
                             |
                +----------- causal mask ----------+
                | scores[i][j] = -inf  neu j > i    |
                +-----------------------------------+
                             |
                    softmax (theo chieu key)        shape (T, T)
                             |
                    attn @ V                        shape (T, d_k)
                             |
                          output
```

Causal mask được cắm vào _giữa_ bước tính scores và bước softmax. Đó là điểm chèn duy nhất bạn cần nhớ.

---

### 5. Cài trong nn.Module: register_buffer để mask đi theo model

Trong lớp attention thật, ta không tạo lại mask mỗi lần forward. Ta đăng ký nó làm _buffer_ - một tensor gắn với module nhưng KHÔNG phải tham số học được (không có gradient, không bị optimizer cập nhật), và tự động theo model khi `.to(device)`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # not a parameter: a fixed lower-triangular mask that follows the model
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x: (B, T, n_embd)
        B, T, C = x.shape
        q = self.query(x) # (B, T, head_size)
        k = self.key(x)   # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        head_size = k.shape[-1]
        scores = q @ k.transpose(-2, -1) / (head_size ** 0.5) # (B, T, T)

        # slice mask to current seq len T (T may be < block_size)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1) # (B, T, T)
        out = attn @ v                   # (B, T, head_size)
        return out

# quick smoke test
x = torch.randn(2, 5, 32) # (B=2, T=5, n_embd=32)
head = CausalSelfAttentionHead(n_embd=32, head_size=16, block_size=8)
print(head(x).shape) # torch.Size([2, 5, 16])
```

💡 _Vì sao `self.tril[:T, :T]`?_ Ta khai báo mask cho độ dài tối đa `block_size`, nhưng lúc chạy chuỗi có thể ngắn hơn ($T \le \text{block\_size}$), nên chỉ lấy góc trên-trái tương ứng.

---

### Nhìn sâu hơn

_Encoder vs Decoder - hai kiểu attention khác nhau về mask._ Không phải mô hình nào cũng cần causal mask. So sánh:

| Đặc điểm                  | Attention hai chiều (encoder, kiểu BERT)  | Attention nhân quả (decoder, kiểu GPT) |
| :------------------------ | :---------------------------------------- | :------------------------------------- |
| **Token nhìn thấy**       | Cả trái lẫn phải (toàn chuỗi)             | Chỉ trái + chính nó                    |
| **Có causal mask?**       | Không                                     | Có (tam giác dưới)                     |
| **Mục tiêu điển hình**    | Điền chỗ trống (masked LM), hiểu ngữ cảnh | Sinh token kế tiếp (autoregressive)    |
| **Dùng để sinh văn bản?** | Không tự nhiên                            | Có - đúng bài toán của nó              |

Vì ta đang xây một _mô hình sinh văn bản_ kiểu GPT, ta luôn dùng causal mask.

_Chi phí bộ nhớ._ Ma trận scores có kích thước $(B, nh, T, T)$ (thêm chiều batch $B$ và số head $nh$). Bộ nhớ tăng theo $T^2$ - đây chính là lý do độ dài ngữ cảnh (context length) tốn kém và là động lực cho các kỹ thuật như FlashAttention, KV-cache (sẽ gặp ở các bài sau).

_Mẹo cài đặt thực tế._ Trong PyTorch hiện đại có `F.scaled_dot_product_attention(q, k, v, is_causal=True)` tự lo mask + tối ưu tốc độ/bộ nhớ. Nhưng lúc HỌC, ta viết tay `masked_fill` để thấy rõ cơ chế - hiểu rồi mới nên dùng hàm dựng sẵn.

🎯 Một chuỗi hằng số `-inf` có thể tạo ra `NaN` nếu CẢ một hàng bị mask (không còn ô hợp lệ nào). Với causal mask tam giác dưới, mỗi hàng luôn có ít nhất một ô hợp lệ (chính nó), nên an toàn.

---

### Thực hành

- **[Dễ]** Chạy đoạn code ở mục 3, in ma trận `attn[0]`. Kiểm tra: (a) mọi ô trên đường chéo có bằng 0 không? (b) tổng mỗi hàng có bằng 1 không (`attn[0].sum(dim=-1)`)?
- **[Trung bình]** Cố tình “phá”: bỏ dòng `masked_fill`, chạy lại và so sánh `attn[0]`. Quan sát cách các ô tương lai giờ nhận trọng số dương - hình dung nếu đây là lúc train thì đáp án đã bị lộ ra sao.
- **[Thử thách]** Sửa `CausalSelfAttentionHead` để đổi giữa hai chế độ bằng cờ `causal: bool`. Khi `causal=False` -> attention hai chiều (không mask, kiểu encoder); khi `causal=True` -> có mask. Với cùng input, in `attn` của hai chế độ và giải thích khác nhau ở đâu.

---

### Hiểu lầm thường gặp

- ❌ “Causal mask được nhân vào SAU softmax.” ➔ ✅ Sai. Phải đặt `-inf` _TRƯỚC_ softmax; nhân 0 sau softmax làm tổng trọng số các hàng đầu nhỏ hơn 1 -> sai chuẩn hóa.
- ❌ “Đặt điểm tương lai bằng 0 là đủ để chặn.” ➔ ✅ Không. $\exp(0) = 1$, tương lai vẫn có trọng số dương. Phải là `-inf` để $\exp(-\infty) = 0$.
- ❌ “Mọi Transformer đều cần causal mask.” ➔ ✅ Chỉ _decoder_ (GPT) cần. Encoder (BERT) dùng attention hai chiều, không mask.
- ❌ “Causal mask làm attention chạy tuần tự như RNN.” ➔ ✅ Không. Toàn bộ chuỗi vẫn được xử lý _song song_ trong một phép nhân ma trận; mask chỉ đặt một số ô về `-inf`, không hề bắt tính từng bước.

---

### Mẹo ghi nhớ

- _Tam giác DƯỚI = quá khứ được phép._ `tril` giữ lại phần dưới đường chéo ($j \le i$).
- _-inf -> 0:_ vì $\exp(-\infty) = 0$, softmax tự loại tương lai và tự chuẩn hóa lại.
- _Thứ tự vàng:_ scores -> mask (-inf) -> softmax -> @ V. Mask luôn nằm GIỮA scores và softmax.
- Mẹo một câu: _“Bịt mắt tương lai bằng âm vô cực, để đoán chữ tiếp theo cho công bằng.”_

---

### Tóm tắt trực quan

| Khái niệm                       | Một dòng                                                   |
| :------------------------------ | :--------------------------------------------------------- |
| **Causal mask**                 | Cấm token nhìn các vị trí tương lai ($j > i$)              |
| **torch.tril**                  | Tạo ma trận tam giác dưới toàn số 1 (vùng được phép)       |
| **masked_fill(mask==0, -inf)**  | Đặt điểm số vùng cấm thành `-inf` trước softmax            |
| **Vì sao -inf**                 | $\exp(-\infty) = 0 \to$ trọng số tương lai đúng bằng 0     |
| **Tự hồi quy (autoregressive)** | Token $t$ chỉ dựa vào $0..t \to$ sinh văn bản đúng đắn     |
| **Encoder vs Decoder**          | Encoder không mask (hai chiều); Decoder có mask (nhân quả) |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Let’s build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- PyTorch docs - `torch.tril`, `masked_fill`, `softmax`: [pytorch.org/docs/stable](https://pytorch.org/docs/stable/)
- nanoGPT (Andrej Karpathy) - xem cách cài tril buffer trong self-attention: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- “Attention Is All You Need” - phần masked self-attention của decoder: [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)

---

### Tổng kết

- Mô hình sinh văn bản (decoder/GPT) _không được nhìn tương lai_, nếu không đáp án bị rò rỉ khi train.
- _Causal mask_ = ma trận tam giác dưới; ô tương lai ($j > i$) bị đặt `-inf` _trước_ softmax.
- $\exp(-\infty) = 0$ nên softmax gán trọng số 0 cho tương lai và tự chuẩn hóa lại phần còn lại.
- Trong module, đăng ký mask bằng `register_buffer` và cắt `self.tril[:T, :T]` theo độ dài chuỗi.
- Attention hai chiều (encoder) khác attention nhân quả (decoder) đúng ở chỗ có hay không có mask này.

Giờ ta đã có đủ ba mảnh - scaled dot-product attention (Bài 13), causal mask (bài này) - để ghép thành một khối hoàn chỉnh. Ở _Bài 15: Cài đặt một attention head hoàn chỉnh (PyTorch)_, ta sẽ gói key/query/value + scaling + causal mask + dropout vào một `nn.Module` gọn gàng, chạy thử end-to-end và soi từng shape tensor. Hẹn gặp ở khối attention head đầu tiên của bạn! 🚀
