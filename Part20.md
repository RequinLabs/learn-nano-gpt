# Bài 20: Positional embedding - đưa thông tin vị trí vào mô hình

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 4:** Transformer block

---

### Giới thiệu

Hãy tưởng tượng bạn nhận được một tin nhắn nhưng các chữ bị _xáo trộn_ hết thứ tự: “ăn đói tôi cơm quá”. Bạn vẫn đoán ra đủ các từ, nhưng ý nghĩa thì mờ mịt - vì tiếng Việt (và mọi ngôn ngữ) phụ thuộc rất nhiều vào _thứ tự_. “Chó cắn người” khác hẳn “người cắn chó”, dù dùng đúng ba từ như nhau.

Ở Bài 19, chúng ta vừa ghép attention + feed-forward + residual + LayerNorm thành một _Transformer block_ hoàn chỉnh. Block đó rất mạnh, nhưng nó có một “khiếm khuyết bẩm sinh”: cơ chế attention _không hề biết token nào đứng trước, token nào đứng sau_. Với attention, một câu chỉ là một _tập hợp_ các vector, không phải một _chuỗi_ có thứ tự. Bài này sẽ vá đúng lỗ hổng đó bằng _positional embedding_ - cách bơm thông tin vị trí vào mô hình.

> 🔑 **Một câu để nhớ:** Attention là “mù thứ tự” (permutation-invariant); ta cộng thêm một vector vị trí vào embedding của mỗi token để mô hình biết token nào ở đâu.

---

### Bạn sẽ học được gì?

- [ ] Hiểu vì sao self-attention _không biết thứ tự_ và điều đó gây hại thế nào.
- [ ] Phân biệt _token embedding_ (token này là chữ gì) với _positional embedding_ (token này ở vị trí nào).
- [ ] Cài đặt _learned positional embedding_ bằng `nn.Embedding(block_size, n_embd)` và _cộng_ vào token embedding.
- [ ] Theo dõi _shape tensor_ khi cộng hai embedding với broadcasting.
- [ ] Biết sơ lược _sinusoidal positional encoding_ (sin/cos) trong paper gốc và khác gì bản học được.

📌 **Yêu cầu trước:** Bài 11 (token embedding qua `nn.Embedding`), Bài 12-15 (self-attention), Bài 19 (Transformer block). Biết `nn.Embedding`, broadcasting, và shape `(B, T, C)`.

---

### 1. Vì sao attention "mù thứ tự"?

**Analogy:** hãy coi self-attention như một _buổi họp mà mọi người phát biểu cùng lúc_. Mỗi người (token) chỉ so khớp nội dung Query của mình với Key của người khác - không ai để ý “ai ngồi ghế số mấy”. Nếu bạn tráo chỗ ngồi của hai người, kết quả cuộc họp không đổi, vì attention chỉ nhìn _nội dung_, không nhìn _vị trí_.

Về mặt toán học, điều này gọi là _permutation-equivariant_: nếu ta hoán vị thứ tự các token đầu vào, đầu ra của attention cũng chỉ bị hoán vị y hệt, chứ nội dung mỗi vector _không đổi_. Nói cách khác, với attention thuần túy:

```text
Attention("chó", "cắn", "người")  cho ra cùng bộ vector với
Attention("người", "cắn", "chó")   (chi bi doi cho, noi dung khong doi)
```

Nhưng hai câu này có nghĩa _hoàn toàn khác nhau_! Đây là vấn đề nghiêm trọng: một mô hình ngôn ngữ mà không phân biệt được thứ tự thì không thể học ngữ pháp. (Lưu ý: _causal mask_ ở Bài 14 có giới hạn được token nhìn về đâu, nhưng nó không cho mô hình biết _khoảng cách_ giữa các vị trí - vị trí 1 và vị trí 5 vẫn không có “nhãn” số nào.)

💡 So sánh với _RNN/LSTM_: chúng xử lý token _lần lượt_ nên thứ tự được nhúng sẵn trong quá trình tính. Transformer đánh đổi tính tuần tự đó lấy khả năng _tính song song_, nên phải bơm thông tin vị trí vào bằng tay.

---

### 2. Ý tưởng: cộng thêm một "vector vị trí"

Giải pháp cực kỳ đơn giản và đẹp: _mỗi vị trí (0, 1, 2, …) cũng có một vector embedding riêng_, và ta _cộng_ nó vào token embedding tại vị trí đó.

- _Token embedding_: tra bảng theo _chỉ số token_ (“từ này là chữ gì”) → vector $C$ chiều.
- _Positional embedding_: tra bảng theo _vị trí_ (“đây là chỗ thứ mấy trong câu”) → vector $C$ chiều.
- Đầu vào của block đầu tiên = _token_emb + pos_emb_.

Sơ đồ luồng dữ liệu tổng thể của GPT mini (nơi positional embedding nằm ở đầu):

```text
idx (B, T)  token ids
   |
   |  token_embedding_table  nn.Embedding(vocab_size, C)
   v
tok_emb (B, T, C) ----+
                       |  (+)  cong element-wise (broadcasting theo B)
pos_emb (T, C) --------+
   ^                    |
   |  position_embedding_table nn.Embedding(block_size, C)
   |  tra bang cho arange(0,1,...,T-1)
   v
   x = tok_emb + pos_emb   (B, T, C)
   |
   v  -> [ Transformer block ] -> ... -> [ Transformer block ]
   |
   v  -> LayerNorm -> Linear(C, vocab_size)
logits (B, T, vocab_size)
```

Mỗi token bây giờ mang _hai loại thông tin trộn chung trong một vector_: nội dung (nó là chữ gì) + vị trí (nó ở đâu). Mạng sẽ tự học cách tách hai tín hiệu này khi cần.

🎯 Đừng nhầm _positional embedding_ với _token embedding_. Cả hai đều là `nn.Embedding` trả về vector $C$ chiều, nhưng một cái tra theo _id token_, cái kia tra theo _chỉ số vị trí_. Chúng có kích thước bảng khác nhau: `(vocab_size, C)` vs `(block_size, C)`.

---

### 3. Cài đặt learned positional embedding

Đây là kiểu positional embedding mà nanoGPT (và GPT-2) dùng: một bảng `nn.Embedding` _học được_ cho các vị trí. Code minh họa:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTEmbeddings(nn.Module):
    # Combine token + learned positional embeddings (GPT-2 / nanoGPT style).
    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        # Token embedding table: maps a token id -> a C-dim vector.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (vocab_size, C)
        # Positional embedding table: maps a position 0..block_size-1 -> a C-dim vector.
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # (block_size, C)
        self.block_size = block_size

    def forward(self, idx):
        # idx: (B, T) integer token ids
        B, T = idx.shape
        assert T <= self.block_size, "sequence longer than block_size"

        tok_emb = self.token_embedding_table(idx) # (B, T, C)

        # Positions 0,1,...,T-1 on the same device as idx.
        pos = torch.arange(T, device=idx.device) # (T,)
        pos_emb = self.position_embedding_table(pos) # (T, C)

        # Broadcasting: (B, T, C) + (T, C) -> (B, T, C)
        # pos_emb is added to EVERY sequence in the batch.
        x = tok_emb + pos_emb # (B, T, C)
        return x
```

Chạy thử để thấy shape:

```python
torch.manual_seed(1337)
vocab_size, block_size, n_embd = 65, 8, 32
emb = GPTEmbeddings(vocab_size, block_size, n_embd)

idx = torch.randint(0, vocab_size, (4, 8)) # batch=4, T=8 token ids
x = emb(idx)
print(x.shape) # -> torch.Size([4, 8, 32])  == (B, T, C)

# Inspect broadcasting: the SAME pos_emb row is added at position t
# for every sample in the batch.
pos = torch.arange(8)
print(emb.position_embedding_table(pos).shape) # -> torch.Size([8, 32]) == (T, C)
```

Điểm mấu chốt về _broadcasting_: `tok_emb` có shape `(B, T, C) = (4, 8, 32)`, còn `pos_emb` là `(T, C) = (8, 32)`. Khi cộng, PyTorch tự “nhân bản” `pos_emb` ra đủ $B$ bản (chiều batch được phát tán), nên vị trí thứ $t$ được cộng _cùng một_ vector vị trí cho mọi câu trong batch. Điều đó hợp lý: “vị trí số 3” nghĩa như nhau ở mọi câu.

⚠️ Nhớ dùng `torch.arange(T)` (đúng độ dài chuỗi hiện tại), _không_ phải `arange(block_size)`. Lúc sinh văn bản, $T$ thường nhỏ hơn `block_size`; nếu tra nhầm cả `block_size` vị trí sẽ lệch shape ngay.

---

### 4. Sinusoidal positional encoding (bản gốc sin/cos)

Trước khi GPT chọn cách “học” vị trí, paper gốc “Attention Is All You Need” (2017) dùng một công thức _cố định_ bằng hàm sin và cos - gọi là _positional encoding_ (không có tham số học). Với vị trí $pos$ và chiều $i$ trong vector $C$ chiều:

$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/C}}\right)$$
$$\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/C}}\right)$$

Ý tưởng: mỗi vị trí được mã hóa thành một “chữ ký” gồm nhiều sóng sin/cos ở các _tần số_ khác nhau - giống như biểu diễn nhị phân của một số, nhưng bằng sóng liên tục. Nhờ vậy mô hình có thể suy ra _khoảng cách tương đối_ giữa hai vị trí.

```python
import math
import torch

def sinusoidal_positional_encoding(block_size, n_embd):
    # Returns a fixed (non-learnable) table of shape (block_size, n_embd).
    pe = torch.zeros(block_size, n_embd) # (T, C)
    position = torch.arange(0, block_size).unsqueeze(1) # (T, 1)

    # div_term: 10000^(2i/C) for even dims i.
    div_term = torch.exp(
        torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd)
    ) # (C/2,)

    pe[:, 0::2] = torch.sin(position * div_term) # even dims
    pe[:, 1::2] = torch.cos(position * div_term) # odd dims
    return pe

pe = sinusoidal_positional_encoding(block_size=8, n_embd=32)
print(pe.shape) # -> torch.Size([8, 32])  == (block_size, C), no gradient
```

Bảng so sánh hai cách:

| Tiêu chí                               | Learned (GPT-2/nanoGPT)        | Sinusoidal (paper gốc)                    |
| :------------------------------------- | :----------------------------- | :---------------------------------------- |
| **Cài đặt**                            | `nn.Embedding(block_size, C)`  | Công thức sin/cos cố định                 |
| **Có tham số học?**                    | Có (`block_size × C` tham số)  | Không (0 tham số)                         |
| **Ngoại suy chuỗi dài hơn khi train?** | Kém (chỉ học tới `block_size`) | Tốt hơn về lý thuyết (công thức liên tục) |
| **Cách đưa vào**                       | Cộng vào token embedding       | Cộng vào token embedding                  |
| **Dùng ở đâu**                         | GPT-2, nanoGPT                 | Transformer gốc                           |

💡 Cả hai đều _cộng_ vào token embedding và đều cho ra `(T, C)`. Điểm khác chỉ là: một cái _học_ các vector vị trí, cái kia _tính sẵn_ bằng công thức. Trong khóa này ta theo nanoGPT nên dùng bản _học được_ cho gọn.

---

### Nhìn sâu hơn

- **Vì sao “cộng” mà không “nối” (concat)?** Nối `tok_emb` và `pos_emb` sẽ làm chiều vector phình lên $2C$, tốn thêm tham số ở mọi lớp sau. Phép cộng giữ nguyên $C$ và thực nghiệm cho thấy mô hình thừa sức tách hai tín hiệu ra khi cần (không gian $C$ chiều rất rộng). Đây là một đánh đổi đơn giản mà hiệu quả.
- **Giới hạn `block_size`.** Bảng positional embedding học được chỉ có `block_size` hàng, nên mô hình _không thể_ xử lý chuỗi dài hơn `block_size` (sẽ vượt chỉ số bảng). Đây là lý do GPT-2 có “context window” cố định (1024 token). Muốn ngữ cảnh dài hơn phải huấn luyện lại với bảng lớn hơn hoặc dùng kỹ thuật vị trí khác (RoPE, ALiBi… - ngoài phạm vi khóa cơ bản này).
- **Số tham số thêm vào.** Learned positional embedding thêm đúng `block_size × n_embd` tham số. Ví dụ `block_size=256, n_embd=384` $\to$ thêm $256 \times 384 = 98.304$ tham số - rất nhỏ so với tổng số tham số của mô hình, nhưng đủ để dạy mô hình biết thứ tự.

⚠️ **Gotcha thường gặp:** quên đưa `pos = torch.arange(T)` lên đúng device (`device=idx.device`). Nếu model ở GPU mà `pos` ở CPU, PyTorch sẽ báo lỗi “tensors on different devices” ngay khi tra bảng.

---

### Thực hành

- **[Dễ]** Tạo `GPTEmbeddings(vocab_size=50, block_size=16, n_embd=64)`, cho `idx = torch.randint(0, 50, (2, 10))` và in `x.shape`. Nó phải là `(2, 10, 64)`. Giải thích vì sao $T=10$ chứ không phải $16$.
- **[Trung bình]** Sửa `forward` để _bỏ_ phần `pos_emb` (chỉ trả `tok_emb`). Đưa hai câu `idx1 = [1, 2, 3]` và `idx2 = [3, 2, 1]` qua một attention head (không mask), rồi so sánh: khi không có positional embedding, tập vector đầu ra chỉ khác nhau ở thứ tự chứ nội dung như nhau. Đây chính là minh chứng “attention mù thứ tự”.
- **[Thử thách]** Cài `sinusoidal_positional_encoding` rồi vẽ (bằng `matplotlib`) heatmap của bảng (`block_size=64, n_embd=128`). Quan sát các cột dao động theo tần số giảm dần từ trái sang phải. So sánh với việc in một bảng `nn.Embedding` khởi tạo ngẫu nhiên (không có cấu trúc sóng).

---

### Hiểu lầm thường gặp

- ❌ “Causal mask đã cho mô hình biết thứ tự rồi, không cần positional embedding.” ➔ ✅ Sai. Causal mask chỉ _giới hạn_ token nhìn về đâu (không nhìn tương lai), nhưng không gán “nhãn vị trí” nên mô hình vẫn không biết khoảng cách giữa các token. Vẫn cần positional embedding.
- ❌ “Positional embedding và token embedding là cùng một bảng.” ➔ ✅ Hai bảng _khác nhau_: `(vocab_size, C)` tra theo id token, và `(block_size, C)` tra theo vị trí.
- ❌ “Phải _nối_ (concat) positional embedding vào token embedding.” ➔ ✅ Ta _cộng_ (element-wise). Cộng giữ nguyên chiều $C$; mô hình vẫn tách được hai tín hiệu.
- ❌ “Sinusoidal encoding tốt hơn learned nên GPT phải dùng sin/cos.” ➔ ✅ Không tuyệt đối. GPT-2/nanoGPT dùng _learned_ positional embedding và vẫn rất tốt; mỗi cách có đánh đổi riêng.

---

### Mẹo ghi nhớ

- Câu thần chú: _“attention mù thứ tự $\to$ cộng vector vị trí vào”_.
- Nhớ hai bảng: token `(vocab_size, C)`, position `(block_size, C)`; đầu vào block = `tok_emb + pos_emb`.
- Nhớ broadcasting: `(B, T, C) + (T, C) -> (B, T, C)` (phát tán theo batch).
- Learned = có tham số, giới hạn `block_size`; sinusoidal = công thức sin/cos, không tham số.

---

### Tóm tắt trực quan

| Khái niệm                | Một dòng                                                                               |
| :----------------------- | :------------------------------------------------------------------------------------- |
| **Vấn đề**               | Self-attention permutation-invariant $\to$ mù thứ tự token                             |
| **Positional embedding** | Vector $C$ chiều gắn với mỗi vị trí, cộng vào token embedding                          |
| **Token vs Position**    | `(vocab_size, C)` theo id token vs `(block_size, C)` theo vị trí                       |
| **Learned pos-emb**      | `nn.Embedding(block_size, C)`, có tham số, dùng ở GPT-2/nanoGPT                        |
| **Sinusoidal**           | $\text{PE}(pos,2i)=\sin(\dots)$, $\text{PE}(pos,2i+1)=\cos(\dots)$, cố định, 0 tham số |
| **Cách kết hợp**         | _Cộng_ (broadcasting) chứ không nối: `(B,T,C) + (T,C)`                                 |
| **Giới hạn**             | Learned pos-emb chỉ tới `block_size` (context window cố định)                          |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Positional Embedding in Transformer Neural Networks | Positional Encoding Explained with Code](https://www.youtube.com/watch?v=1biZfFLPRSY)
- PyTorch `nn.Embedding`: [pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT (Andrej Karpathy) - tham chiếu token + positional embedding: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- “Attention Is All You Need” (mục 3.5 - positional encoding sin/cos): [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer (minh họa positional encoding): [jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

---

### Tổng kết

- Self-attention _không biết thứ tự_ (permutation-invariant): nếu tráo token, nội dung vector ra không đổi - điều này khiến mô hình không phân biệt được “chó cắn người” với “người cắn chó”.
- Ta vá bằng _positional embedding_: mỗi vị trí có một vector $C$ chiều riêng, được _cộng_ vào token embedding qua broadcasting `(B,T,C) + (T,C)`.
- _Learned_ positional embedding (`nn.Embedding(block_size, C)`) là cách nanoGPT/GPT-2 dùng; paper gốc dùng _sinusoidal_ sin/cos cố định. Cả hai đều cộng vào embedding, khác nhau ở chỗ _học_ hay _tính sẵn_.
- Nhớ đặt `pos` đúng device và giới hạn $T \le \text{block\_size}$.

Giờ ta đã có đủ mọi mảnh ghép: token embedding, positional embedding, và Transformer block hoàn chỉnh (Bài 19). Ở _Bài 21: Kiến trúc GPT - xếp chồng nhiều Transformer block_, chúng ta sẽ _lắp ráp_ tất cả lại thành một mô hình GPT mini thực thụ - xếp chồng nhiều block, thêm LayerNorm cuối và lớp Linear chiếu ra logits trên toàn bộ từ vựng. Đây là bài mà cả kiến trúc cuối cùng lộ diện. Hẹn gặp ở bài kế tiếp! 🤖
