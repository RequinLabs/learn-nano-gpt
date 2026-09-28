# Bài 23: Ghép toàn bộ thành một mô hình GPT hoàn chỉnh

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 5:** Xây dựng GPT mini

---

### Giới thiệu

Ở _Bài 22_, mình đã lắp xong hai “cánh cửa” quan trọng của mô hình: token embedding biến mỗi token thành một vector giàu ý nghĩa, positional embedding thêm thông tin “vị trí thứ mấy”, và `lm_head` ở cuối biến vector trở lại thành logits trên toàn bộ từ vựng. Nhưng ở giữa hai đầu đó, mình mới chỉ có… một chồng Transformer block rời rạc, chưa xâu chuỗi lại.

Hãy tưởng tượng bạn đang lắp một chiếc xe đạp: đã có bánh xe (embedding), ghi-đông (`lm_head`), khung sườn và bộ truyền động (các block). Nhưng nếu chưa _vặn ốc nối chúng theo đúng thứ tự_, bạn vẫn chưa đạp đi đâu được. Bài này là lúc vặn ốc: gói tất cả vào _một class `GPT` duy nhất_, viết `forward` cho dữ liệu chảy suốt từ token đến `logits`, và viết `generate` để mô hình biết cách… tự nói.

> 🔑 **Một câu để nhớ:** Một mô hình GPT hoàn chỉnh = embedding (ý nghĩa + vị trí) → chồng Transformer block (trộn thông tin) → LayerNorm cuối → `lm_head` (ra `logits`); `forward` để học, `generate` để nói.

---

### Bạn sẽ học được gì?

- [ ] Viết class `GPT` hoàn chỉnh kế thừa `nn.Module`, gom embedding + các block + `ln_f` + `lm_head`.
- [ ] Hiểu luồng dữ liệu và _shape tensor_ qua từng bước trong `forward`.
- [ ] Tính loss bằng `cross_entropy` cho bài toán đoán token kế tiếp (next-token).
- [ ] Viết hàm `generate` sinh văn bản tự hồi quy (autoregressive) với `temperature` và `top-k`.
- [ ] Chạy thử `forward` trên một batch giả và _kiểm tra shape đầu ra_ khớp kỳ vọng.

📌 **Yêu cầu trước:** đã nắm token/positional embedding + `lm_head` (Bài 22), Transformer block (residual + LayerNorm + attention + feed-forward), và causal mask.

---

### 1. Bức tranh lớn: một GPT nhìn từ trên xuống

Trước khi lao vào code, hãy nhìn toàn cảnh. Một câu đầu vào (chuỗi các số nguyên chỉ token) sẽ đi qua đúng đường ống này:

```text
idx (B, T)   chuỗi token id
     |
     v
 [token_emb]  tra bảng -> (B, T, n_embd)
     +          cộng thêm
 [pos_emb]    vị trí 0..T-1 -> (T, n_embd) broadcast
     |
     v
  x (B, T, n_embd)
     |
     v
 [Block 1] -> [Block 2] -> ... -> [Block L]   (mỗi block giữ nguyên shape)
     |
     v
 [ln_f]  LayerNorm cuoi -> (B, T, n_embd)
     |
     v
 [lm_head]  Linear -> (B, T, vocab_size)  = logits
```

Điểm mấu chốt: _shape `(B, T, n_embd)` được giữ nguyên xuyên suốt các block_. Block không đổi kích thước, nó chỉ “trộn” thông tin giữa các token và làm giàu vector. Chỉ có `lm_head` ở cuối mới đổi chiều cuối từ `n_embd` sang `vocab_size`.

💡 **Nhớ 3 ký hiệu:** $B$ = batch size (số câu xử lý cùng lúc), $T$ = số token trong mỗi câu (độ dài chuỗi), `n_embd` = số chiều embedding. Nắm 3 chữ này là đọc được mọi shape.

---

### 2. Định nghĩa class `GPT`

Giống như một bản thiết kế nhà: `__init__` khai báo có những “phòng” nào (các lớp con), còn `forward` mô tả “khách đi qua các phòng theo thứ tự nào”. Mình gom tất cả module con vào một `nn.ModuleDict` cho gọn gàng, giống nanoGPT của Andrej Karpathy.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65    # so ky tu / token trong tu vung
    block_size: int = 128   # do dai chuoi toi da (context length)
    n_layer: int = 4        # so Transformer block xep chong
    n_head: int = 4         # so attention head trong moi block
    n_embd: int = 128       # so chieu embedding
    dropout: float = 0.1

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Sub-modules assembled in one dict, like nanoGPT
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),      # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),      # positional embedding
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # stack of blocks
            ln_f = nn.LayerNorm(config.n_embd),                        # final LayerNorm
        ))

        # Output head: n_embd -> vocab_size (produces logits)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output head
        self.transformer.wte.weight = self.lm_head.weight
```

🎯 **Weight tying (buộc trọng số):** `wte` (bảng embedding) và `lm_head` dùng CHUNG một ma trận trọng số. Trực giác: “vector đại diện cho token X khi ĐỌC vào” và “vector dùng để CHẤM điểm token X khi VIẾT ra” nên giống nhau. Kỹ thuật này giảm kha khá tham số và thường cải thiện chất lượng (Press & Wolf, 2017).

---

### 3. Viết hàm `forward`: cho dữ liệu chảy

`forward` là trái tim của mô hình. Nó nhận `idx` (các token id) và tùy chọn `targets` (đáp án đúng để tính `loss`). Hãy theo dõi shape từng dòng như dõi theo một gói hàng qua băng chuyền:

```python
    def forward(self, idx, targets=None):
        B, T = idx.shape # idx: (B, T) integer token ids
        assert T <= self.config.block_size, "chuoi dai hon block_size"

        # Positions 0, 1, ..., T-1
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T,)

        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd) -> broadcast add
        x = self.transformer.drop(tok_emb + pos_emb) # (B, T, n_embd)

        for block in self.transformer.h: # pass through each block
            x = block(x) # (B, T, n_embd) -> (B, T, n_embd)

        x = self.transformer.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x)     # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Reshape to compute next-token cross-entropy over all positions
            loss = F.cross_entropy(
                logits.view(B * T, -1), # (B*T, vocab_size)
                targets.view(B * T),    # (B*T,)
                ignore_index=-1,
            )

        return logits, loss
```

#### Vì sao lại cross_entropy như vậy?

Bài toán của GPT là _đoán token kế tiếp_. Với mỗi vị trí $t$, mô hình cho ra một phân phối xác suất trên toàn từ vựng, đáp án đúng là token thật ở vị trí $t+1$. `cross_entropy` đo “độ ngạc nhiên”: đặt xác suất cao cho token đúng thì loss nhỏ.

Công thức cho một vị trí, với đáp án đúng là chỉ số $y$:

$$\text{loss} = -\ln(\text{softmax}(\text{logits})_y) = -\ln\left(\frac{\exp(\text{logits}_y)}{\sum_j \exp(\text{logits}_j)}\right)$$

⚠️ **Phân biệt logits vs xác suất:** `logits` là số thực bất kỳ (có thể âm, dương, chưa chuẩn hóa). Chỉ SAU khi qua softmax chúng mới thành xác suất (không âm, cộng lại bằng 1). `cross_entropy` của PyTorch nhận thẳng logits (nó tự làm `log_softmax` bên trong cho ổn định số học) - nên _đừng tự `softmax` trước rồi mới đưa vào `cross_entropy`_, sẽ tính sai.

Mình `view(B*T, -1)` để gộp mọi vị trí của mọi câu thành một danh sách dài $B \times T$ mẫu, mỗi mẫu là một bài phân loại `vocab_size` lớp. Đơn giản mà đúng.

---

### 4. Hàm `generate`: dạy mô hình tự nói

Sinh văn bản là _tự hồi quy_ (autoregressive): giống nhắn tin đoán chữ tiếp theo - dự đoán 1 token, ghép vào đuôi câu, rồi lại lấy câu mới đó làm đầu vào để đoán token kế. Cứ thế lặp lại.

```python
    @torch.no_grad() # no gradients during generation
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx: (B, T) current context
        for _ in range(max_new_tokens):
            # Crop context to at most block_size (model can't see further)
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond) # (B, T, vocab_size)

            # Take logits at the LAST position only
            logits = logits[:, -1, :] # (B, vocab_size)
            logits = logits / temperature # scale by temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') # keep only top-k

            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) sample
            idx = torch.cat((idx, idx_next), dim=1) # append -> (B, T+1)

        return idx
```

#### temperature và top-k điều khiển gì?

Hãy hình dung một cái núm vặn độ “liều lĩnh”:

```text
temp THAP (0.5): logits/0.5 -> softmax NHON -> hay chon token top -> an toan, lap
temp CAO  (1.5): logits/1.5 -> softmax PHANG-> chon dan trai      -> sang tao, de loan
```

`top-k` là bộ lọc “chỉ chọn trong $K$ ứng viên xác suất cao nhất”, cắt bỏ đuôi token rác trước khi lấy mẫu - tránh mô hình bốc trúng một token cực vô lý.

💡 `torch.multinomial` _lấy mẫu ngẫu nhiên_ theo xác suất, khác với `torch.argmax` (luôn lấy token cao nhất, gọi là greedy). Lấy mẫu giúp câu văn đa dạng, không lặp máy móc.

---

### 5. Chạy thử `forward` trên một batch và kiểm tra shape

Đây là khoảnh khắc “đề-pa”: tạo dữ liệu giả và xem đường ống có chảy đúng không. Kiểm tra shape là cách rẻ nhất để bắt bug - sai shape gần như luôn là sai logic.

```python
torch.manual_seed(1337)
config = GPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=128)
model = GPT(config)

B, T = 8, 16 # 8 cau, moi cau 16 token
idx = torch.randint(0, config.vocab_size, (B, T)) # (8, 16) fake tokens
targets = torch.randint(0, config.vocab_size, (B, T)) # (8, 16) fake labels
logits, loss = model(idx, targets)

print(logits.shape) # torch.Size([8, 16, 65])  -> (B, T, vocab_size)
print(loss.item())   # ~4.2   -> gan -log(1/65) = 4.174 khi chua hoc gi

# Test generation from a single start token
ctx = torch.zeros((1, 1), dtype=torch.long) # (1, 1) start token
out = model.generate(ctx, max_new_tokens=20, temperature=0.8, top_k=10)
print(out.shape) # torch.Size([1, 21])  -> 1 start + 20 new
```

🎯 **Kiểm tra tỉnh táo (sanity check):** khi mô hình vừa khởi tạo (chưa học), nó phải “mù tịt”, đoán ngẫu nhiên đều trên `vocab_size` lớp. Vậy loss ban đầu phải xấp xỉ $-\ln(1/\text{vocab\_size}) = \ln(\text{vocab\_size})$. Với `vocab_size=65`, đó là $\ln(65) \approx 4.174$. Nếu loss ban đầu ra ~4.2 - tuyệt, đúng như kỳ vọng! Nếu ra 50 hay NaN, chắc chắn có gì đó sai (khởi tạo, shape, hoặc quên `ln_f`).

---

### Nhìn sâu hơn

- **Vì sao có `ln_f` (LayerNorm) ngay trước `lm_head`?** GPT hiện đại dùng kiến trúc _pre-norm_: LayerNorm đặt TRƯỚC attention/feed-forward trong mỗi block, cộng thêm `ln_f` ở cuối chồng block. Thiếu `ln_f`, phân phối giá trị đưa vào `lm_head` dễ trôi dạt qua nhiều tầng residual, làm huấn luyện kém ổn định. LayerNorm chuẩn hóa vector về trung bình 0, phương sai 1 trên chiều `n_embd`.
- **Về `block_size` và cắt ngữ cảnh trong `generate`:** `wpe` chỉ có `block_size` vị trí. Nếu câu sinh ra dài hơn, mình phải cắt `idx[:, -block_size:]` để vị trí không tràn bảng. Đây là giới hạn cửa sổ ngữ cảnh (context window) - mô hình chỉ “nhìn thấy” `block_size` token gần nhất.
- **Chi phí bộ nhớ:** attention có độ phức tạp $\mathcal{O}(T^2)$ theo độ dài chuỗi. Tăng `block_size` gấp đôi thì bộ nhớ attention tăng gấp bốn - lý do các mô hình lớn phải dùng KV-cache hay flash-attention.

---

### Thực hành

- **[Dễ]** Đổi `n_layer` từ 4 xuống 2 rồi chạy lại đoạn kiểm tra. Shape của `logits` có đổi không? (Gợi ý: số block chỉ ảnh hưởng độ sâu, KHÔNG đổi shape đầu ra.)
- **[Trung bình]** Trong `generate`, đặt `temperature=0.1` rồi `temperature=1.5`, mỗi cái sinh 30 token từ cùng một start token. So sánh: bản nào lặp lại nhiều hơn, bản nào “loạn” hơn? Giải thích bằng công thức `logits / temperature`.
- **[Thử thách]** Thêm chế độ `top_p` (nucleus sampling): thay vì giữ K token cao nhất, giữ tập token nhỏ nhất có tổng xác suất $\ge p$ (vd 0.9). Gợi ý: sort xác suất giảm dần, cộng dồn (`cumsum`), cắt tại ngưỡng $p$, đưa phần còn lại về `-inf`.

---

### Hiểu lầm thường gặp

- ❌ “`logits` chính là xác suất.” ➔ ✅ `logits` là số thực chưa chuẩn hóa; phải qua softmax mới thành xác suất (không âm, tổng = 1).
- ❌ “Nên softmax trước rồi mới đưa vào cross_entropy cho chắc.” ➔ ✅ `F.cross_entropy` nhận thẳng logits và tự làm `log_softmax` bên trong; tự softmax trước sẽ tính sai và kém ổn định số học.
- ❌ “Càng nhiều block thì shape đầu ra càng to.” ➔ ✅ Mỗi block giữ nguyên `(B, T, n_embd)`; số block chỉ tăng độ sâu, không đổi shape. Chỉ `lm_head` mới đổi chiều cuối.
- ❌ “`generate` có thể sinh câu dài vô hạn thoải mái.” ➔ ✅ Phải cắt ngữ cảnh về `block_size` mỗi bước, vì bảng vị trí `wpe` chỉ có `block_size` chỗ.
- ❌ “`temperature` càng cao thì văn càng hay.” ➔ ✅ Cao quá làm phân phối phẳng → chọn cả token vô lý → văn lộn xộn; thấp quá thì lặp lại nhàm. Cần cân bằng (thường 0.7-1.0) + `top-k` / `top-p`.

---

### Mẹo ghi nhớ

- _“Vào ý nghĩa, giữa trộn thông tin, ra logits”_: embedding → blocks → `ln_f` → `lm_head`.
- Sanity check số 1: loss ban đầu $\approx \ln(\text{vocab\_size})$. Nhớ con số này là bắt bug được ngay.
- `forward(idx, targets)` → dùng để HỌC (có loss); `generate(idx, ...)` → dùng để NÓI (không gradient).
- Mẹo vàng: _luôn `print(tensor.shape)`_ sau mỗi bước. Debug mô hình = debug shape.

---

### Tóm tắt trực quan

| Thành phần          | Một dòng để nhớ                                    |
| :------------------ | :------------------------------------------------- |
| **wte (token emb)** | Tra bảng: token id → vector ý nghĩa (B, T, n_embd) |
| **wpe (pos emb)**   | Thêm thông tin vị trí 0..T-1                       |
| **Block × L**       | Trộn thông tin + làm giàu, GIỮ NGUYÊN shape        |
| **ln_f**            | LayerNorm cuối, ổn định trước khi ra logits        |
| **lm_head**         | Linear: n_embd → vocab_size = logits               |
| **forward**         | Chảy dữ liệu + tính cross_entropy (để học)         |
| **generate**        | Tự hồi quy: đoán 1 token, ghép, lặp (để nói)       |
| **Sanity loss**     | $\approx \ln(\text{vocab\_size})$ khi chưa học     |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Coding a ChatGPT Like Transformer From Scratch in PyTorch](https://www.youtube.com/watch?v=C9QSpl5nmrY)
- nanoGPT của Andrej Karpathy (bản `model.py` là nguồn cảm hứng trực tiếp): [nanoGPT on GitHub](https://github.com/karpathy/nanoGPT)
- Paper gốc “Attention Is All You Need”: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer (trực quan hóa từng bước): [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- PyTorch docs (`nn.Module`, `F.cross_entropy`, `torch.multinomial`): [PyTorch Documentation](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

### Tổng kết

- Một mô hình GPT hoàn chỉnh gói gọn trong một class: embedding (token + vị trí) → chồng Transformer block → `ln_f` → `lm_head`.
- `forward(idx, targets)` cho ra logits shape `(B, T, vocab_size)` và tính `cross_entropy` cho bài toán next-token.
- `generate` sinh văn bản tự hồi quy, điều khiển sự đa dạng bằng `temperature` + `top-k`; nhớ cắt ngữ cảnh về `block_size`.
- Kiểm tra shape và loss ban đầu ($\approx \ln(\text{vocab\_size})$) là hai bài kiểm tra tỉnh táo cực rẻ mà đắt giá.

Mô hình của mình đã “sống” và biết chảy dữ liệu! Nhưng nó “to” cỡ nào - bao nhiêu tham số, và con số đó nói lên điều gì? Ở _Bài 24: Đếm tham số và hiểu kích thước mô hình_, mình sẽ đếm chính xác số tham số và ước lượng độ lớn mô hình từ các siêu tham số. Hẹn gặp ở bài sau nhé! 🤖
