# Bài 21: Kiến trúc GPT - xếp chồng nhiều Transformer block

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 5:** Xây dựng GPT mini

---

### Giới thiệu

Ở _Bài 20_, chúng ta đã học cách tiêm thông tin vị trí vào mô hình bằng positional embedding - vì bản thân attention không biết token nào đứng trước, token nào đứng sau. Giờ chúng ta có đủ nguyên liệu: token embedding, positional embedding, và một Transformer block (attention + feed-forward, bọc bởi residual và LayerNorm). Câu hỏi tự nhiên là: _ghép tất cả lại thành một mô hình GPT hoàn chỉnh như thế nào?_

Hãy tưởng tượng một _dây chuyền lắp ráp trong nhà máy_. Mỗi công nhân (một Transformer block) nhận sản phẩm dở dang từ người trước, tinh chỉnh thêm một chút, rồi chuyền cho người sau. Không ai làm xong mọi thứ một mình - nhưng sau khi đi qua đủ nhiều trạm, sản phẩm thô ban đầu (một chuỗi token) trở thành một biểu diễn giàu ngữ nghĩa, đủ để đoán chữ tiếp theo. GPT chính là _một chồng nhiều Transformer block giống hệt nhau xếp lên nhau_, cộng với vài chi tiết ở hai đầu vào/ra.

> 🔑 **Một câu để nhớ:** GPT = token embedding + positional embedding -> N khối Transformer giống nhau xếp chồng -> LayerNorm cuối -> lm_head chiếu ra logits cho từng token trong từ vựng.

---

### Bạn sẽ học được gì?

- [ ] Toàn cảnh kiến trúc GPT _decoder-only_ và vì sao gọi là “decoder-only”.
- [ ] Luồng dữ liệu đầy đủ: từ chỉ số token tới logits, kèm shape tensor qua từng bước.
- [ ] Ý nghĩa 3 siêu tham số cốt lõi: `n_layer`, `n_head`, `n_embd` và cách chúng ảnh hưởng số tham số.
- [ ] Vì sao _xếp chồng_ nhiều block lại mạnh hơn một block, và weight tying là gì.

📌 **Yêu cầu trước:** đã hiểu self-attention (Bài 14-17), Transformer block với residual + LayerNorm (Bài 18-19), và positional embedding (Bài 20).

---

### 1. GPT là một Transformer "decoder-only"

Transformer gốc trong paper _Attention Is All You Need_ có _2 nửa_: encoder (đọc câu nguồn) và decoder (sinh câu đích) - hợp cho dịch máy. GPT bỏ hẳn nửa encoder, chỉ giữ lại _decoder_, nên gọi là _decoder-only_.

Điều làm GPT trở thành mô hình ngôn ngữ _tự hồi quy_ (autoregressive) là _causal mask_: mỗi token chỉ được “nhìn” các token đứng TRƯỚC nó (và chính nó), không được nhìn tương lai. Nhờ vậy, khi dự đoán chữ tiếp theo, mô hình không “gian lận” bằng cách liếc đáp án phía sau.

```text
Ma trận attention QK^T sau khi áp causal mask (T=4 token, đánh dấu 0..3):

         key0   key1   key2   key3
 query0 [ s00 | -inf | -inf | -inf ]   <- token 0 chỉ thấy chính nó
 query1 [ s10 | s11  | -inf | -inf ]   <- token 1 thấy 0,1
 query2 [ s20 | s21  | s22  | -inf ]   <- token 2 thấy 0,1,2
 query3 [ s30 | s31  | s32  | s33  ]   <- token 3 thấy tất cả trước nó

(-inf -> sau softmax thành xác suất 0. Đây là tam giác dưới.)
```

💡 So sánh nhanh với RNN: RNN xử lý token _tuần tự_ qua thời gian. Attention thì tính _song song_ toàn bộ chuỗi cùng lúc; “tính nhân quả” (không nhìn tương lai) được đảm bảo bằng mask chứ không phải bằng thứ tự tính toán.

---

### 2. Luồng dữ liệu từ token tới logits

Đây là sơ đồ tổng thể - hãy đọc theo mũi tên `->`, kèm shape (với batch size `B`, độ dài chuỗi `T`, kích thước embedding `n_embd = C`):

```text
idx (B, T)                       # token ids, int64
   -> token_emb = wte(idx)       # (B, T, C)  tra bảng embedding
   -> pos_emb   = wpe(0..T-1)    # (T, C)     embedding vị trí
   -> x = token_emb + pos_emb    # (B, T, C)  cộng (broadcast theo B)
   -> Block 1 -> Block 2 -> ... -> Block N   # mỗi block giữ nguyên (B, T, C)
   -> x = ln_f(x)                # (B, T, C)  LayerNorm cuối
   -> logits = lm_head(x)        # (B, T, vocab_size)
```

Mỗi Block bên trong có cấu trúc _pre-norm_ (LayerNorm đặt TRƯỚC sub-layer, kiểu nanoGPT/GPT-2):

```text
x (B,T,C)
         |
   +-----+------------------+
   |     v                  |
   |   LayerNorm            |     residual đi thẳng (skip)
   |     v                  |
   |   Multi-Head Attention |
   |     v                  |
   +---> (+) <--------------+     x = x + attn(ln1(x))
         |
   +-----+------------------+
   |     v                  |
   |   LayerNorm            |
   |     v                  |
   |   Feed-Forward (MLP)   |
   |     v                  |
   +---> (+) <--------------+     x = x + mlp(ln2(x))
         |
         v  (B,T,C)  -> sang block kế tiếp
```

🎯 Điểm mấu chốt: _shape (B, T, C) được giữ nguyên_ khi đi qua mỗi block. Nhờ vậy ta xếp chồng bao nhiêu block tùy thích - đầu ra block này khớp đầu vào block sau.

---

### 3. Cài đặt lớp GPT bằng PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65  # e.g. character-level vocab
    block_size: int = 128 # max context length T
    n_layer: int = 6      # number of Transformer blocks stacked
    n_head: int = 6       # attention heads per block
    n_embd: int = 384     # embedding dim C (must be divisible by n_head)

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # token embedding table
        self.wpe = nn.Embedding(config.block_size, config.n_embd) # positional embedding table
        # Stack N identical Transformer blocks (Block defined in earlier lessons)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd) # final LayerNorm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # project to logits

        # Weight tying: share weights between input embedding and output projection
        self.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape # idx: (B, T) int64
        pos = torch.arange(T, device=idx.device) # (T,)

        tok_emb = self.wte(idx) # (B, T, C)
        pos_emb = self.wpe(pos) # (T, C) -> broadcast over B
        x = tok_emb + pos_emb   # (B, T, C)

        for block in self.blocks:
            x = block(x)        # (B, T, C) unchanged
        x = self.ln_f(x)        # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten to (B*T, vocab) vs (B*T,) for next-token cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

# --- Quick shape check ---
cfg = GPTConfig()
model = GPT(cfg)
idx = torch.randint(0, cfg.vocab_size, (2, 16)) # (B=2, T=16)
logits, _ = model(idx)
print(logits.shape) # torch.Size([2, 16, 65]) == (B, T, vocab_size)
```

💡 _Weight tying_ (`self.wte.weight = self.lm_head.weight`): dùng chung ma trận trọng số cho lớp embedding đầu vào và lớp chiếu ra logits. Việc này tiết kiệm `vocab_size * n_embd` tham số và thường cải thiện chất lượng - trực giác là “cùng một không gian nghĩa dùng cho cả đọc vào lẫn ghi ra”.

---

### 4. Ba siêu tham số cốt lõi và số tham số

| Siêu tham số   | Ý nghĩa                                         | Tăng nó lên thì…                                                                       |
| :------------- | :---------------------------------------------- | :------------------------------------------------------------------------------------- |
| **n_embd (C)** | Bề rộng của mỗi vector biểu diễn token          | Mỗi token “giàu” thông tin hơn; tham số và bộ nhớ tăng ~bậc hai                        |
| **n_head**     | Số “góc nhìn” attention song song trong 1 block | Nhiều kiểu quan hệ được học; mỗi head có chiều $C / \text{n\_head}$                    |
| **n_layer**    | Số Transformer block xếp chồng                  | Mô hình sâu hơn, tổng hợp quan hệ phức tạp hơn; tham số tăng tuyến tính theo `n_layer` |

Ràng buộc BẮT BUỘC: _`n_embd` phải chia hết cho `n_head`_ (vì mỗi head lấy một lát chiều $\text{head\_dim} = \text{n\_embd} / \text{n\_head}$). Ví dụ `n_embd=384`, `n_head=6` -> mỗi head 64 chiều.

Ước lượng thô số tham số của phần block (bỏ qua embedding và LayerNorm nhỏ): mỗi block có attention ($\sim 4 \cdot C^2$) + MLP ($\sim 8 \cdot C^2$) $\approx 12 \cdot C^2$. Với $N = \text{n\_layer}$ block:

```text
params_blocks ≈ n_layer * 12 * n_embd^2

Ví dụ: n_layer=6, n_embd=384
  ≈ 6 * 12 * 384^2 = 6 * 12 * 147456 ≈ 10.6 triệu tham số (chỉ riêng các block)
```

Cộng thêm bảng embedding `wte` (`vocab_size * n_embd`) - nhưng nhờ _weight tying_, `lm_head` không tính thêm.

⚠️ Đừng nhầm _parameter_ (trọng số học được, cố định sau khi train) với _token_ (đơn vị dữ liệu mô hình xử lý). Một mô hình “10M tham số” có thể được huấn luyện trên hàng tỉ token.

---

### Nhìn sâu hơn

_Vì sao xếp chồng nhiều block lại mạnh?_ Một block đơn lẻ chỉ trộn thông tin một lần. Xếp chồng cho phép _tổng hợp phân cấp_: block đầu học quan hệ cục bộ (từ liền kề, cú pháp), block giữa học cụm ý, block cuối tổng hợp ngữ cảnh dài để dự đoán. Đây là lý do “sâu” (deep) trong deep learning.

_Residual (skip connection) là chất keo giữ cho chồng sâu train được._ Vì $x = x + f(x)$, gradient có một “đường cao tốc” chạy thẳng từ cuối về đầu mà không bị nhân nhỏ dần qua từng lớp - tránh vanishing gradient. Không có residual, GPT 12-48 lớp gần như không huấn luyện nổi.

_Pre-norm vs post-norm:_ paper gốc dùng post-norm (LayerNorm SAU sub-layer). GPT-2 và nanoGPT chuyển sang _pre-norm_ (LayerNorm TRƯỚC sub-layer) vì ổn định hơn khi mô hình rất sâu, ít cần warmup cầu kỳ. Đó là lý do code ở trên gọi `ln1(x)` rồi mới `attn`.

💡 `ln_f` (LayerNorm cuối, sau block cuối cùng) dễ bị quên nhưng quan trọng: nó chuẩn hóa biểu diễn trước khi chiếu ra `lm_head`, giúp logits có thang đo ổn định.

---

### Thực hành

- **[Dễ]** Chạy đoạn “Quick shape check” ở trên. Đổi $B$ và $T$, in `logits.shape`. Xác nhận chiều cuối luôn bằng `vocab_size` bất kể $B$, $T$.
- **[Trung bình]** Viết hàm đếm tham số và so sánh 2 cấu hình:

  ```python
  def num_params(m):
      return sum(p.numel() for p in m.parameters())

  small = GPT(GPTConfig(n_layer=4, n_head=4, n_embd=128))
  big   = GPT(GPTConfig(n_layer=8, n_head=8, n_embd=512))
  print(num_params(small), num_params(big)) # big should be far larger
  # Observe: doubling n_embd roughly quadruples the per-block params (~C^2)
  ```

- **[Thử thách]** Cố ý đặt `n_embd=100, n_head=6` rồi khởi tạo mô hình. Bạn sẽ gặp lỗi (vì 100 không chia hết cho 6). Đọc thông báo lỗi, giải thích vì sao ràng buộc `n_embd % n_head == 0` là cần thiết, rồi sửa lại `n_embd` cho hợp lệ.

---

### Hiểu lầm thường gặp

- ❌ “GPT dùng cả encoder và decoder như Transformer gốc.” ➔ ✅ GPT là _decoder-only_; nó bỏ nửa encoder và dùng causal mask để chỉ nhìn về quá khứ.
- ❌ “Mỗi block có nhiệm vụ khác nhau, phải thiết kế riêng.” ➔ ✅ Các block _giống hệt nhau về cấu trúc_; chúng chỉ khác ở trọng số học được. Ta xếp chồng bản sao của cùng một lớp `Block`.
- ❌ “Tăng `n_head` thì mỗi head vẫn có `n_embd` chiều.” ➔ ✅ Tổng chiều chia đều: mỗi head có `n_embd / n_head` chiều, nên tăng head không tự động tăng tổng độ rộng.
- ❌ “Đầu ra của GPT đã là xác suất.” ➔ ✅ `lm_head` trả về _logits_ (số thực chưa chuẩn hóa); phải qua softmax mới thành xác suất.

---

### Mẹo ghi nhớ

- Nhớ 3 chữ _L-H-E_: `n_Layer` (sâu bao nhiêu), `n_Head` (bao nhiêu góc nhìn/block), `n_Embd` (rộng bao nhiêu).
- Hình dung GPT như _bánh sandwich_: hai đầu là embedding và `lm_head`, phần nhân giữa là $N$ lát block giống hệt nhau, xen bởi `ln_f` trước khi ra.
- Mẹo shape: mọi thứ ở giữa đều là `(B, T, C)`; chỉ có đầu (`idx` là `(B, T)`) và cuối (`logits` là `(B, T, vocab)`) khác đi.

---

### Tóm tắt trực quan

| Khái niệm           | Một dòng                                                                  |
| :------------------ | :------------------------------------------------------------------------ |
| **decoder-only**    | GPT bỏ encoder, chỉ giữ decoder + causal mask                             |
| **causal mask**     | Tam giác dưới; token chỉ nhìn quá khứ, tương lai bị `-inf` -> softmax = 0 |
| **xếp chồng block** | $N$ block giống nhau, mỗi block giữ shape `(B, T, C)`                     |
| **residual**        | $x = x + f(x)$, đường cao tốc cho gradient, cho phép chồng sâu            |
| **pre-norm**        | LayerNorm đặt TRƯỚC attention/MLP (GPT-2/nanoGPT), ổn định hơn            |
| **ln_f**            | LayerNorm cuối trước `lm_head`                                            |
| **weight tying**    | Chia sẻ trọng số `wte` và `lm_head`, tiết kiệm tham số                    |
| **logits**          | Đầu ra `(B, T, vocab)`, chưa phải xác suất (cần softmax)                  |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M)
- PyTorch docs: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- nanoGPT (Andrej Karpathy): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Attention Is All You Need: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Hugging Face docs: [https://huggingface.co/docs](https://huggingface.co/docs)

---

### Tổng kết

- GPT là Transformer _decoder-only_: token embedding + positional embedding -> _N block giống nhau xếp chồng_ -> `ln_f` -> `lm_head` ra logits `(B, T, vocab)`.
- Mỗi block giữ nguyên shape `(B, T, C)` nhờ residual, nên xếp chồng tùy ý; residual + pre-norm giúp chồng sâu vẫn train được.
- Ba siêu tham số `n_layer`, `n_head`, `n_embd` quyết định độ sâu, số góc nhìn và độ rộng; nhớ ràng buộc `n_embd % n_head == 0`.
- Weight tying chia sẻ trọng số giữa embedding và `lm_head`.

Chúng ta đã có bộ khung GPT hoàn chỉnh về mặt kiến trúc. Trong _Bài 22: Token embedding + positional embedding + lm_head_, mình sẽ mổ xẻ kỹ hơn ba thành phần ở hai đầu vào/ra của khung này - cách bảng embedding hoạt động, vì sao cộng token và positional embedding lại hợp lý, và `lm_head` biến biểu diễn thành logits ra sao. Hẹn gặp lại! 🤖
