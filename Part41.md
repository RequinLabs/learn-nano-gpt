# Bài 41: Dự án cuối - hoàn thiện và tự host GPT mini của bạn

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 8:** Từ pretraining đến trợ lý & dự án cuối

---

### Giới thiệu

Ở _Bài 40_ chúng ta đã bàn kỹ về hạn chế, chi phí, an toàn và đạo đức khi xây/tự host một LLM: mô hình nhỏ dễ “nói nhảm” (hallucination), tự host tốn VRAM và điện, và bạn chịu trách nhiệm về nội dung sinh ra. Với tất cả hiểu biết đó, đã đến lúc gói ghém mọi thứ thành _một sản phẩm hoàn chỉnh của riêng bạn_.

Hãy tưởng tượng bạn học nấu ăn suốt 8 tuần. Bài cuối này giống _bữa tiệc tự tay bạn dọn ra_: nhận nguyên liệu (dữ liệu), huấn luyện GPT mini, lưu “công thức” (checkpoint), rồi mở một quán nhỏ (API) để người khác gọi món (sinh văn bản). Không một dòng nào còn là hộp đen với bạn nữa.

> 🔑 **Một câu để nhớ:** Một GPT mini hoàn chỉnh = _dữ liệu -> tokenizer -> mô hình -> vòng huấn luyện -> checkpoint đã lưu -> hàm generate -> API tự host_, và bạn đã tự tay viết mọi mắt xích đó.

---

### Bạn sẽ học được gì?

- [ ] Ghép _toàn bộ pipeline_ từ đầu đến cuối thành một dự án chạy được
- [ ] _Lưu và tải_ checkpoint đúng cách (`state_dict` + cấu hình mô hình)
- [ ] Viết một _API tự host_ tối giản để sinh văn bản theo yêu cầu
- [ ] Biết cách _đánh giá nhanh_ chất lượng GPT mini của mình
- [ ] Nhìn lại hành trình 8 tuần và _biết đi đâu tiếp theo_

📌 **Yêu cầu trước:** đã nắm attention, Transformer block, GPT mini, vòng huấn luyện (AdamW, cross-entropy), tokenization và sinh văn bản với temperature/top-k/top-p (Bài 1-40).

---

### 1. Bức tranh toàn cảnh của dự án

Trước khi viết code, hãy nhìn cả hệ thống. Một GPT mini end-to-end đi qua các trạm:

```text
Văn bản thô            Tokenizer          Mô hình GPT                Sinh văn bản
 "xin chao ..."  ---->  [char/BPE]  ---->  Embedding + Blocks  ---->  logits -> softmax
     (raw)                encode()            (nn.Module)              -> token tiep theo
                             |                    ^                          |
                             v                    |                          v
                        ids (B, T)           train loop (AdamW,        API tra ve chuoi
                        LongTensor           cross-entropy)            (tu host, FastAPI)
                                                  |  -> checkpoint .pt (state_dict)
```

**Phân biệt quan trọng (đừng lẫn):**

| Khái niệm       | Là gì                                          | Không phải là |
| :-------------- | :--------------------------------------------- | :------------ |
| **logits**      | vector số thực chưa chuẩn hóa từ lớp cuối      | xác suất      |
| **xác suất**    | `softmax(logits)`, cộng lại bằng 1             | logits        |
| **pretraining** | học đoán token tiếp theo trên văn bản lớn      | fine-tuning   |
| **fine-tuning** | tinh chỉnh mô hình đã pretrain cho việc cụ thể | RLHF          |

---

### 2. Ghép mô hình GPT mini (nhắc lại kiến trúc)

Một GPT mini là chồng nhiều _Transformer block_ (mỗi block = self-attention nhiều đầu + feed-forward, kiểu pre-norm với _residual + LayerNorm_). Công thức lõi là _scaled dot-product attention_, chia $\sqrt{d_k}$ để giữ phương sai ổn định trước softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{mask}\right) V$$

`mask` là _causal mask_ (tam giác dưới): mỗi token chỉ nhìn về quá khứ và chính nó, KHÔNG nhìn tương lai.

```text
QKᵀ (T=4)          causal mask (tam giac duoi)      sau mask
 [ . . . . ]        [ 0  -inf -inf -inf ]         token0 -> {0}
 [ . . . . ]   +    [ 0   0   -inf -inf ]    =    token1 -> {0,1}
 [ . . . . ]        [ 0   0    0   -inf ]         token2 -> {0,1,2}
 [ . . . . ]        [ 0   0    0    0   ]         token3 -> {het}
```

Mô hình gọn (đủ chạy trên Colab/GPU tiêu dùng):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """One head of causal self-attention."""
    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular matrix for the causal mask; not a learnable parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # (batch, time, channels=n_embd)
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Scaled dot-product: divide by sqrt(d_k) to stabilise variance
        att = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5) # (B, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # causal mask
        att = F.softmax(att, dim=-1) # (B, T, T) -> rows sum to 1
        att = self.dropout(att)
        return att @ self.value(x) # (B, T, head_size)
```

💡 `register_buffer` giữ ma trận tam giác đi theo mô hình (kể cả khi `.to('cuda')`) nhưng KHÔNG coi là tham số học được - nên nó không xuất hiện trong `optimizer`.

Ghép nhiều Head thành _multi-head_, thêm feed-forward, residual và LayerNorm để có một block, rồi chồng nhiều block:

```python
class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.sa = nn.ModuleList([Head(cfg.n_embd, head_size, cfg.block_size, cfg.dropout) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ffwd = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        # Multi-head attention with residual connection (pre-norm)
        sa_out = torch.cat([h(self.ln1(x)) for h in self.sa], dim=-1)
        x = x + self.dropout(self.proj(sa_out))
        # Feed-forward with residual connection (pre-norm)
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTConfig:
    vocab_size = 65
    block_size = 128 # context length (T max)
    n_embd = 128     # embedding dim (C)
    n_head = 4
    n_layer = 4
    dropout = 0.1

class GPTMini(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd) # token embedding
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) # positional embedding
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size) # -> logits

    def forward(self, idx, targets=None):
        B, T = idx.shape # idx: (B, T) token ids
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos) # (B, T, C) broadcast pos over batch
        for blk in self.blocks:
            x = blk(x) # (B, T, C)
        logits = self.head(self.ln_f(x)) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy for next-token prediction; flatten time into batch
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss
```

⚠️ _Embedding KHÁC one-hot._ `nn.Embedding` là bảng tra cứu học được: mỗi id -> một vector đặc, không phải vector 0/1 thưa như one-hot - nhờ vậy nó mang được ngữ nghĩa.

---

### 3. Vòng huấn luyện end-to-end

Có dữ liệu -> encode thành ids -> lấy batch -> forward -> `loss.backward()` -> `optimizer.step()`. Dùng _AdamW_ (Adam + weight decay tách rời) - chuẩn cho Transformer.

```python
model = GPTMini(GPTConfig()).to('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(max_steps):
    xb, yb = get_batch('train') # xb, yb: (B, T) LongTensor
    logits, loss = model(xb, yb) # forward
    optimizer.zero_grad(set_to_none=True)
    loss.backward() # backprop through autograd
    optimizer.step() # update weights

    if step % 500 == 0:
        print(f"step {step}: loss {loss.item():.4f}")
        # Example output: step 0: 4.17 -> step 500: 2.43 -> step 1000: 1.90 (loss down -> learning)
```

🎯 Loss giảm dần là dấu hiệu mô hình đang học. Với character-level, loss ~1.5-1.8 đã cho văn bản đọc được.

---

### 4. Lưu và tải checkpoint đúng cách

Đây là bước hay bị làm sai. _Nên lưu `state_dict` (từ điển trọng số) kèm cấu hình_, KHÔNG lưu cả object bằng pickle (dễ hỏng khi đổi phiên bản).

```python
# --- Save ---
torch.save({
    'model_state': model.state_dict(), # only weights (tensors)
    'config': vars(GPTConfig()),        # dims to rebuild architecture
    'stoi': stoi, 'itos': itos,         # tokenizer maps (char-level)
}, 'gpt_mini.pt')

# --- Load ---
ckpt = torch.load('gpt_mini.pt', map_location='cpu')
cfg = GPTConfig()
for k, v in ckpt['config'].items():
    setattr(cfg, k, v)

model = GPTMini(cfg)
model.load_state_dict(ckpt['model_state']) # restore weights
model.eval() # tat dropout khi suy luan
```

⚠️ Nhớ gọi `model.eval()` khi suy luận để _tắt dropout_, và bọc sinh văn bản trong `torch.no_grad()` để tiết kiệm bộ nhớ.

---

### 5. Hàm generate (sinh văn bản)

Sinh _tự hồi quy_ (autoregressive): dự đoán token tiếp, nối vào, lặp lại. `temperature` (chia logits) và `top-k` (giữ k logits cao nhất) điều chỉnh độ ngẫu nhiên.

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:] # crop to context window (B, <=block_size)
        logits, _ = model(idx_cond)      # (B, T, vocab)
        logits = logits[:, -1, :] / temperature # last step only -> (B, vocab)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf') # keep top-k, mask rest
        probs = F.softmax(logits, dim=-1) # logits -> probabilities
        idx_next = torch.multinomial(probs, num_samples=1) # sample (B, 1)
        idx = torch.cat((idx, idx_next), dim=1) # append -> (B, T+1)
    return idx
```

💡 `temperature < 1` làm phân phối “nhọn” hơn (chắc chắn, lặp lại); `> 1` làm “phẳng” hơn (đa dạng, dễ vô nghĩa). Cao KHÔNG có nghĩa là tốt hơn.

---

### 6. Tự host bằng một API tối giản

Đóng gói thành dịch vụ web nhỏ (FastAPI): nạp model _một lần_ khi khởi động, mỗi request chỉ chạy `generate`.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
ckpt = torch.load('gpt_mini.pt', map_location='cpu') # load ONCE at startup
# ... rebuild model, stoi, itos as in section 4 ...
model.eval()

class Req(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 200
    temperature: float = 0.8

@app.post('/generate')
def gen(req: Req):
    ids = torch.tensor([[stoi[c] for c in req.prompt]], dtype=torch.long) # (1, T)
    out = generate(model, ids, req.max_new_tokens, cfg.block_size, req.temperature, top_k=40)
    return {"text": ''.join(itos[i] for i in out[0].tolist())}

# Run:  uvicorn app:app --port 8000
# Call: curl -X POST localhost:8000/generate -d '{"prompt":"To be"}'
```

🎯 Nạp model _ngoài_ hàm route để tránh tải lại mỗi request. Với GPU: đặt `model.to('cuda')` và đưa tensor input lên cùng thiết bị.

---

### Nhìn sâu hơn

- **Bộ nhớ khi tự host.** GPT mini fp32 với $P$ tham số cần $\sim 4 \cdot P$ byte riêng trọng số (fp16 thì $\sim 2 \cdot P$). Khi _huấn luyện_ còn cộng gradient + optimizer state (AdamW giữ 2 moment) + activation - nên train tốn gấp nhiều lần suy luận.
- **Tốc độ suy luận.** Mỗi token mới lại chạy lại cả chuỗi -> chậm khi context dài. Kỹ thuật _KV-cache_ lưu Key/Value đã tính để không tính lại; đây là điểm nâng cấp đáng giá đầu tiên cho dự án cuối.
- **Đánh giá nhanh.** Đừng chỉ nhìn văn bản sinh ra: đo _validation loss_ trên tập giữ riêng; train loss giảm mà val loss tăng -> _overfitting_. Có thể so cấu hình qua $\text{perplexity} = \exp(\text{loss})$.

---

### Thực hành

- **[Dễ]** Chạy đủ pipeline: train GPT mini trên một file `.txt` bất kỳ, lưu checkpoint, rồi tải lại và sinh 500 ký tự.
- **[Trung bình]** Thêm `top_p` (nucleus sampling) vào `generate`: giữ tập token nhỏ nhất có tổng xác suất $\ge p$ rồi lấy mẫu. So sánh với top-k và suy luận shape của probs sau khi lọc.
- **[Thử thách]** Dựng API FastAPI, đưa model lên GPU nếu có, đo thời gian sinh 200 token; thử thêm KV-cache và so tốc độ trước/sau.

---

### Hiểu lầm thường gặp

- ❌ “Lưu model là `torch.save(model)` cho gọn.” ➔ ✅ Nên lưu `state_dict` + config; pickle cả object dễ hỏng khi đổi phiên bản.
- ❌ “Nhiệt độ càng cao thì sinh càng hay.” ➔ ✅ Cao làm đa dạng nhưng dễ vô nghĩa; thường 0.7-1.0 là hợp lý.
- ❌ “Logits là xác suất.” ➔ ✅ Logits chưa chuẩn hóa; phải qua softmax mới thành xác suất cộng bằng 1.
- ❌ “Nạp model trong hàm route API cho tiện.” ➔ ✅ Nạp một lần lúc khởi động; nạp mỗi request cực chậm.
- ❌ “Đã pretrain là mô hình biết trả lời như trợ lý.” ➔ ✅ Pretraining chỉ đoán token tiếp; cần fine-tuning/RLHF.

---

### Mẹo ghi nhớ

- _Pipeline 7 trạm:_ dữ liệu -> tokenizer -> model -> train -> checkpoint -> generate -> API.
- Suy luận: luôn `model.eval()` + `torch.no_grad()`; lưu _trọng số + config_, đừng lưu cả object.
- Mẹo: đọc pipeline như một dây chuyền - hỏng ở trạm nào thì shape/kết quả trạm đó lệch trước.

---

### Tóm tắt trực quan

| Khái niệm                    | Một dòng cốt lõi                           |
| :--------------------------- | :----------------------------------------- |
| **state_dict**               | từ điển tensor trọng số để lưu/tải mô hình |
| **model.eval() + no_grad()** | suy luận đúng, tiết kiệm bộ nhớ            |
| **generate**                 | sinh tự hồi quy: đoán -> nối -> lặp        |
| **temperature**              | chia logits: <1 chắc chắn, >1 đa dạng      |
| **top-k / top-p**            | giới hạn tập token trước khi lấy mẫu       |
| **tự host**                  | nạp model một lần, phục vụ qua API         |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Let’s build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- PyTorch docs: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT (Andrej Karpathy): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- “Attention Is All You Need”: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Hugging Face docs: [https://huggingface.co/docs](https://huggingface.co/docs)

---

### 🎓 Chúc mừng bạn đã hoàn thành khóa học!

Bạn vừa hoàn thành một hành trình mà nhiều người chỉ dám mơ: _tự tay xây dựng một LLM kiểu GPT từ số 0_. Nhìn lại 8 tuần qua:

- _Tuần 1-2 - Môi trường & phần cứng:_ cài Python/PyTorch, hiểu GPU/VRAM, làm chủ tensor và autograd.
- _Tuần 3 - Mô hình ngôn ngữ nơ-ron đầu tiên:_ embedding, dự đoán token tiếp theo, cross-entropy.
- _Tuần 4 - Attention:_ scaled dot-product $\text{softmax}(QK^T/\sqrt{d_k})V$, causal mask.
- _Tuần 5 - Transformer block:_ multi-head, feed-forward, residual + LayerNorm.
- _Tuần 6 - GPT mini & huấn luyện:_ chồng block, vòng train với AdamW, sinh văn bản.
- _Tuần 7 - Tokenization, tối ưu & tự host:_ BPE, KV-cache, đưa mô hình lên GPU cục bộ.
- _Tuần 8 - Pretraining đến trợ lý & dự án cuối:_ fine-tuning, an toàn/đạo đức, dự án end-to-end.

Bạn không còn coi LLM là hộp đen: bạn hiểu từng con số, từng phép nhân ma trận, từng thiết kế. Nền tảng cực kỳ vững. 🤖

**Đi đâu tiếp theo?**

- 🎥 Xem lại “Let’s build GPT” của Andrej Karpathy và đào sâu _nanoGPT_ để nâng quy mô.
- 🔧 Học _fine-tuning_ và _LoRA_ để tinh chỉnh mô hình pretrain một cách rẻ.
- 📚 Đọc paper gốc _“Attention Is All You Need”_ ([https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)) - giờ bạn đủ trình để đọc hiểu.
- 🔎 Tìm hiểu _RAG_ và _agents_ để ghép LLM vào ứng dụng thật.
- 📈 Thử _scaling_ lên mô hình lớn hơn, nhiều dữ liệu hơn.

Cảm ơn bạn đã đi tới tận đây. Giờ hãy đi xây thứ gì đó tuyệt vời với GPT mini của chính bạn! 🚀

---

### Tổng kết

- Một GPT mini hoàn chỉnh = _dữ liệu -> tokenizer -> model -> train -> checkpoint -> generate -> API_.
- _Lưu `state_dict` + config_, tải lại rồi `model.eval()`; _tự host_ = nạp model một lần, phục vụ qua API tối giản (chú ý VRAM và KV-cache).
- Bạn đã đi trọn 8 tuần và tự xây được một LLM kiểu GPT từ đầu đến cuối - _chúc mừng và chúc bạn đi thật xa!_
