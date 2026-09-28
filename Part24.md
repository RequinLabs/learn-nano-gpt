# Bài 24: Đếm tham số và hiểu kích thước mô hình

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 5:** Xây dựng GPT mini

---

### Giới thiệu

Ở _Bài 23_, bạn đã ghép mọi mảnh ghép lại thành một GPT mini hoàn chỉnh: token embedding + positional embedding -> nhiều Transformer block (mỗi block có multi-head attention + feed-forward, gói trong residual và LayerNorm) -> LayerNorm cuối -> lớp `lm_head` bắn ra logits. Mô hình chạy được, forward ra đúng shape. Nhưng khi khoe với bạn bè, câu hỏi đầu tiên họ hỏi là: _“Mô hình của cậu bao nhiêu tham số? Có nhét vừa card đồ họa không?”_

Hãy tưởng tượng bạn vừa lắp xong một chiếc xe: hai câu hỏi tự nhiên là _xe nặng bao nhiêu ký?_ và _bình xăng chứa được bao nhiêu?_. Với mô hình ngôn ngữ cũng vậy: _số tham số_ giống “trọng lượng” - quyết định mô hình chiếm bao nhiêu bộ nhớ; còn _VRAM_ giống “khoang chứa” của card đồ họa mà chiếc xe phải nằm vừa trong đó.

Bài này bạn sẽ học cách _đếm chính xác số tham số_, hiểu _số đó đến từ đâu_ (embedding hay block), và _liên hệ với VRAM_ - nối lại bài phần cứng (Bài 2).

> 🔑 **Một câu để nhớ:** Số tham số = tổng số con số học được (weight + bias) trong mô hình; nhân với số byte mỗi số là ra dung lượng bộ nhớ tối thiểu để chứa nó.

---

### Bạn sẽ học được gì?

- [ ] Định nghĩa _tham số (parameter)_ và phân biệt với _token_, _activation_.
- [ ] Đếm tham số bằng PyTorch qua `p.numel()` và `model.parameters()`.
- [ ] Suy ra công thức số tham số của _embedding_ và của _một Transformer block_.
- [ ] Hiểu vì sao GPT mini thì _embedding_ chiếm tỷ trọng lớn, còn mô hình to thì các _block_ thống trị.
- [ ] Ước lượng thô _VRAM_ cho inference và training theo số tham số + kiểu dữ liệu (fp32/fp16).

📌 **Yêu cầu trước:** đã hiểu kiến trúc GPT mini ở Bài 23 (embedding, block, `lm_head`) và khái niệm VRAM ở Bài 2.

---

### 1. Tham số là gì? Phân biệt với token và activation

**Analogy:** hãy coi mô hình như một công thức nấu ăn khổng lồ. _Tham số_ là các “tỷ lệ gia vị” đã được ghi cố định trong công thức (2 thìa muối, 100g đường…). _Token_ là nguyên liệu bạn đổ vào mỗi lần nấu (khác nhau mỗi bữa). _Activation_ là các món dở dang trên bếp trong lúc chế biến. Chỉ có tham số là thứ mô hình “nhớ” và được lưu vào file.

Định nghĩa chính xác: _tham số_ là mọi con số _học được bằng gradient descent_ - tức mọi weight và bias trong các lớp `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`. Đây chính là những giá trị bị optimizer (AdamW) cập nhật ở mỗi bước huấn luyện.

| Khái niệm               | Là gì                         | Có được lưu vào file mô hình?       |
| :---------------------- | :---------------------------- | :---------------------------------- |
| **Parameter (tham số)** | weight/bias học được          | ✅ Có (chính là kích thước mô hình) |
| **Token**               | đơn vị đầu vào (số nguyên id) | ❌ Không (chỉ là dữ liệu)           |
| **Activation**          | tensor trung gian khi forward | ❌ Không (sinh ra rồi bỏ)           |
| **Hyperparameter**      | vd learning rate, `n_layer`   | ❌ Không phải tham số học được      |

⚠️ Đừng nhầm _số tham số_ với _số token trong tập huấn luyện_. GPT-3 có 175 tỷ _tham số_ nhưng được huấn luyện trên hàng trăm tỷ _token_ - hai con số hoàn toàn khác nhau, khác cả đơn vị.

```python
import torch
import torch.nn as nn

# A tiny linear layer: out = x @ W^T + b
linear = nn.Linear(in_features=4, out_features=3)

# Weight shape: (out, in) = (3, 4) -> 12 numbers
# Bias shape:   (out,)   = (3,)   -> 3 numbers
for name, p in linear.named_parameters():
    print(name, tuple(p.shape), p.numel())
# weight (3, 4) 12
# bias   (3,)   3
# Total parameters of this layer = 12 + 3 = 15
```

`p.numel()` (number of elements) trả về _tổng số phần tử_ trong tensor - đúng bằng số tham số của tensor đó.

---

### 2. Đếm tham số trong PyTorch

Cách chuẩn: duyệt qua `model.parameters()`, cộng `numel()` của từng tensor. Ta thường tách riêng tham số _trainable_ (có `requires_grad=True`).

```python
def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# Example on a stack of two linear layers
model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
total, trainable = count_parameters(model)
print(total, trainable)
# Layer1: 4*8 + 8 = 40 ; Layer2: 8*2 + 2 = 18 ; Total = 58
# -> prints: 58 58
```

💡 Mẹo báo cáo cho gọn: chia cho `1e6` để ra “triệu tham số” (M) hoặc `1e9` để ra “tỷ tham số” (B). Một mô hình “124M” nghĩa là ~124 triệu tham số (đúng bằng kích cỡ GPT-2 nhỏ).

---

### 3. Số tham số đến từ đâu? Công thức embedding vs block

Đây là phần cốt lõi. Ta ký hiệu các siêu tham số của GPT mini:

| Ký hiệu | Ý nghĩa                               |
| :------ | :------------------------------------ |
| **V**   | `vocab_size` (số token trong từ điển) |
| **T**   | `block_size` (độ dài ngữ cảnh tối đa) |
| **d**   | `n_embd` (số chiều embedding)         |
| **L**   | `n_layer` (số Transformer block)      |
| **H**   | `n_head` (số đầu attention)           |

#### 3a. Phần embedding

```text
token embedding : nn.Embedding(V, d)  -> V * d params
positional embed: nn.Embedding(T, d)  -> T * d params
lm_head         : nn.Linear(d, V)     -> V * d (+ V bias, thường bỏ bias)
```

Lưu ý: token embedding và `lm_head` _có thể chia sẻ trọng số_ (weight tying) - kỹ thuật của GPT-2 - khi đó chỉ tính $V \times d$ một lần. Nếu KHÔNG tie, ta tốn $2 \times V \times d$.

#### 3b. Phần một Transformer block

Một block gồm: attention (Q, K, V, và projection ra) + feed-forward (thường phình lên $4d$ rồi thu lại) + 2 LayerNorm.

```text
Attention (Q,K,V gộp + output proj), bias bỏ qua cho gọn:
  qkv proj : d -> 3d   ->  3 * d * d = 3 d^2
  out proj : d -> d    ->      d * d =  d^2
  => attention ~ 4 d^2

Feed-forward (expansion factor 4):
  fc1 : d -> 4d  -> 4 d^2
  fc2 : 4d -> d  -> 4 d^2
  => ffn ~ 8 d^2

Một block ~ 4 d^2 + 8 d^2 = 12 d^2  (bỏ qua bias và LayerNorm, rất nhỏ)
```

🎯 Công thức thô đáng nhớ: _tham số block $\approx 12 \times L \times d^2$_ (cộng phần embedding). LayerNorm chỉ có $2d$ mỗi lớp - cực nhỏ so với $d^2$, thường bỏ qua khi ước lượng.

#### Ví dụ tính tay từng bước

Giả sử $V=1000, T=128, d=128, L=4, H=4$ (weight tying):

```text
embedding tok+pos = V*d + T*d = 1000*128 + 128*128
                  = 128000  + 16384  = 144384
blocks            = 12 * d^2 * L = 12 * 128^2 * 4
                  = 12 * 16384 * 4 = 786432
total ~ 144384 + 786432 = 930816  (~0.93M, chưa kể bias/LayerNorm nhỏ)
```

```python
# Verify the estimate against a real tiny model roughly
V, T, d, L = 1000, 128, 128, 4
emb = V * d + T * d      # token + positional embedding
blocks = 12 * d * d * L  # ~12 d^2 per block
print(emb, blocks, emb + blocks) # 144384 786432 930816
print(f"~{(emb + blocks)/1e6:.2f}M params") # ~0.93M params
```

💡 Với GPT _mini_ ($d$ nhỏ, $V$ có thể lớn nếu dùng BPE), phần _embedding_ chiếm tỷ trọng đáng kể. Nhưng khi phóng to $d$ và $L$, phần block ($12 L d^2$) lớn theo bình phương $d$ nên _thống trị_ - đó là lý do các mô hình lớn gần như toàn bộ tham số nằm ở các block chứ không phải embedding.

---

### 4. Sơ đồ luồng và nơi tham số nằm

```text
tokens (B, T) ------------------+   [P] = nơi chứa tham số học được
                                  v
     [P] token embedding  nn.Embedding(V, d)  -> (B, T, d)
        +                                            |
     [P] pos embedding    nn.Embedding(T, d)  -------+
                                                     v (B, T, d)
        +--------------------------------------------------+
        |  L x Transformer block  (mỗi block ~12 d^2)      |
        |   x --> [P]LayerNorm --> [P]MHA --> (+x)          | residual
        |     --> [P]LayerNorm --> [P]FFN --> (+x)          |
        +--------------------------------------------------+
                                                     v
                                   [P] final LayerNorm (2d)
                                                     v
                                [P] lm_head nn.Linear(d, V)
                                                     v
                                        logits (B, T, V)
```

Mọi ô `[P]` là nơi chứa tham số. Các phép cộng residual, softmax, cộng embedding KHÔNG có tham số.

```python
# Breakdown params by module group on a GPT-mini-like model
def param_report(model):
    groups = {"embedding": 0, "blocks": 0, "head+ln": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        if any(k in name for k in ("emb", "wte", "wpe")):
            groups["embedding"] += n
        elif "block" in name or "h." in name:
            groups["blocks"] += n
        else:
            groups["head+ln"] += n
    total = sum(groups.values())
    for k, v in groups.items():
        print(f"{k:10s}: {v:>9,d} ({100*v/total:5.1f}%)")
    return total

# embedding :   144,384  ( 15.5%)
# blocks    :   786,432  ( 84.5%)
# head+ln   :         0  (  0.0%)   # tied head, LayerNorm tiny
```

---

### 5. Từ tham số sang VRAM: ước lượng thô

Mỗi tham số là một số thực. Kiểu dữ liệu quyết định số _byte_ mỗi số:

| dtype                   | bytes/số | Ghi chú                                        |
| :---------------------- | :------- | :--------------------------------------------- |
| **fp32 (float32)**      | 4        | mặc định PyTorch, chính xác nhất               |
| **fp16 / bf16**         | 2        | tiết kiệm nửa bộ nhớ, phổ biến khi train/infer |
| **int8 (lượng tử hóa)** | 1        | dùng cho inference tối ưu                      |

_Chỉ để lưu trọng số (inference thô):_ bộ nhớ $\approx \text{số\_tham\_số} \times \text{bytes\_mỗi\_số}$.

```text
Ví dụ mô hình 0.93M tham số ở fp32:
  0.93e6 * 4 bytes ≈ 3.72 MB  (rất nhỏ, chạy được cả trên CPU)

Mô hình 124M (GPT-2 small) ở fp32:
  124e6 * 4 ≈ 496 MB  (chỉ riêng weight)
Ở fp16: 124e6 * 2 ≈ 248 MB
```

_Khi TRAINING, VRAM còn cần thêm_ (nối bài phần cứng, Bài 2):

```text
Training memory ~ weights + gradients + optimizer state + activations
  - gradients   : ~1x số tham số (cùng dtype)
  - AdamW state : ~2x (lưu momentum m và variance v)
  - activations : phụ thuộc batch_size, T, d (có thể rất lớn)
Ngón tay cái: training tốn ~3-4x bộ nhớ so với chỉ lưu weight.
```

⚠️ Đừng chỉ nhân tham số $\times$ bytes rồi tưởng đó là toàn bộ VRAM khi train. Activations và optimizer state (AdamW giữ 2 tensor phụ mỗi tham số) thường mới là thủ phạm gây lỗi CUDA out of memory.

```python
def estimate_memory_mb(num_params, bytes_per_param=4, training=False):
    weights = num_params * bytes_per_param
    # training: weights + grad(1x) + AdamW m,v(2x) ~ 4x (no activations)
    return (weights * (4 if training else 1)) / 1e6

print(f"{estimate_memory_mb(124_000_000):.0f} MB (infer fp32)")       # 496 MB
print(f"{estimate_memory_mb(124_000_000, 4, True):.0f} MB (train fp32)") # 1984 MB
```

---

### Nhìn sâu hơn

- **Vì sao feed-forward chiếm ~2/3 tham số của một block?** Vì factor mở rộng 4: fc1 ($d \to 4d$) và fc2 ($4d \to d$) cho $8 d^2$, trong khi attention chỉ $4 d^2$. Đây là chỗ nhiều mô hình lớn “đổ” phần lớn dung lượng.
- **Weight tying tiết kiệm bao nhiêu?** Chia sẻ token embedding với `lm_head` cắt đi $V \times d$ tham số. Với $V$ lớn (BPE ~50k) và $d$ vừa, đây là khoản tiết kiệm đáng kể và còn giúp mô hình học tốt hơn ở lớp đầu ra.
- **LayerNorm cực rẻ:** mỗi lớp chỉ $2d$ (scale + shift). Với $d=768$ đó là 1536 số - không đáng kể so với $d^2 = 589824$. Vì thế công thức thô bỏ qua LayerNorm mà vẫn khá đúng.
- **Activations ngốn VRAM lúc train:** tensor trung gian $(B, T, d)$ phải giữ cho backward - lý do có gradient checkpointing (tính lại thay vì lưu).

---

### Thực hành

- **[Dễ]** Viết hàm `count_parameters` như ở mục 2 và chạy trên một `nn.Linear(768, 768)`. Xác nhận nó ra $768 \times 768 + 768 = 590592$. Đổi sang không bias (`bias=False`) và kiểm tra ra 589824.
- **[Trung bình]** Dùng công thức `emb = V*d + T*d` và `blocks = 12*d*d*L` để ước lượng số tham số cho cấu hình GPT-2 small: $V=50257, T=1024, d=768, L=12$. So sánh kết quả (~124M) với con số công bố. (Gợi ý: nhớ weight tying nên chỉ tính embedding một lần.)
- **[Thử thách]** Viết `estimate_memory_mb` cho cả inference lẫn training, rồi vẽ (in bảng) VRAM cần cho các mô hình 10M / 100M / 1B tham số ở fp32 và fp16. Tự hỏi: với card 8GB VRAM, mô hình 1B tham số có TRAIN được ở fp16 không nếu tính cả optimizer state? (Gợi ý: $10^9 \times 2 \times 4 \approx 8\text{ GB}$ - vừa khít, chưa kể activations $\to$ gần như không).

---

### Hiểu lầm thường gặp

- ❌ _“Số tham số bằng số token dùng để huấn luyện.”_ ➔ ✅ Hai đại lượng khác hẳn: tham số là weight/bias học được; token là dữ liệu đầu vào. GPT-3 có 175B tham số nhưng train trên số token còn lớn hơn.
- ❌ _“Nhân số tham số với 4 byte là ra toàn bộ VRAM cần khi train.”_ ➔ ✅ Đó mới chỉ là weight. Training còn cần gradient, optimizer state (AdamW ~2x), và activations - thường tổng ~3-4x hoặc hơn.
- ❌ _“Embedding luôn chiếm phần nhỏ không đáng kể.”_ ➔ ✅ Với GPT _mini_ ($d$ nhỏ, $V$ có thể lớn), embedding chiếm tỷ trọng đáng kể; chỉ khi mô hình phóng to ($d$, $L$ lớn) thì các block mới thống trị.
- ❌ _“LayerNorm và bias làm tăng đáng kể số tham số.”_ ➔ ✅ Chúng chỉ cỡ $\mathcal{O}(d)$ mỗi lớp, cực nhỏ so với ma trận $\mathcal{O}(d^2)$; bỏ qua vẫn ước lượng khá đúng.

---

### Mẹo ghi nhớ

- _“12 d bình phương mỗi block”_: attention $4d^2$ + feed-forward $8d^2 = 12d^2$, nhân với $L$.
- Embedding = $(V + T) \times d$; nếu tie weight thì `lm_head` dùng chung, khỏi cộng thêm.
- VRAM lưu weight = params $\times$ bytes (fp32=4, fp16=2). Train thì nhân thêm ~3-4x.
- 🧠 Mẹo vàng: nhìn sơ đồ - mỗi ô `[P]` là một túi tham số, cộng lại là ra tổng.

---

### Tóm tắt trực quan

| Khái niệm            | Một dòng                                                 |
| :------------------- | :------------------------------------------------------- |
| **Parameter**        | weight/bias học được, được lưu vào file mô hình          |
| **p.numel()**        | số phần tử của tensor = số tham số của nó                |
| **Embedding params** | $(V + T) \times d$ (tie weight thì không cộng `lm_head`) |
| **Block params**     | $\sim 12 d^2$ mỗi block, tổng $\sim 12 L d^2$            |
| **Feed-forward**     | $\sim 8 d^2$ mỗi block (2/3 số tham số của block)        |
| **Attention**        | $\sim 4 d^2$ mỗi block                                   |
| **LayerNorm**        | $\sim 2d$ mỗi lớp, bỏ qua khi ước lượng                  |
| **VRAM inference**   | params $\times$ bytes (fp32=4, fp16=2)                   |
| **VRAM training**    | ~3-4x weight                                             |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Let’s reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- PyTorch docs (`nn.Module.parameters`, `Tensor.numel`): [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT của Andrej Karpathy (xem hàm `get_num_params` trong `model.py`): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Attention Is All You Need (kiến trúc Transformer gốc): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Hugging Face docs: [https://huggingface.co/docs](https://huggingface.co/docs)

---

### Tổng kết

- _Tham số_ là mọi weight/bias học được; đếm bằng `sum(p.numel() for p in model.parameters())`.
- Số tham số đến từ _embedding_ $(V+T) \times d$ và _các block_ $\sim 12 L d^2$; feed-forward chiếm ~2/3 mỗi block.
- Với GPT mini, embedding chiếm tỷ trọng lớn; mô hình càng to thì block càng thống trị.
- VRAM lưu weight = params $\times$ bytes; _training tốn thêm ~3-4x_ vì gradient + optimizer state + activations - đúng như bài phần cứng đã cảnh báo.

Bây giờ bạn đã biết mô hình “nặng” bao nhiêu và vì sao. Nhưng một mô hình mới khởi tạo với trọng số ngẫu nhiên bừa bãi thì rất khó huấn luyện - loss có thể bùng nổ hoặc kẹt cứng. Ở _Bài 25: Khởi tạo trọng số và ổn định huấn luyện_, ta sẽ học cách khởi tạo các tham số này thông minh (scale theo $d$, xử lý residual) để huấn luyện diễn ra êm ả từ bước đầu. Hẹn gặp ở bài kế tiếp! 🤖
