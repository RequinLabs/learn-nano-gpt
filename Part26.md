# Bài 26: Vòng lặp huấn luyện đầy đủ

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 6:** Huấn luyện GPT

---

### Giới thiệu

Hãy tưởng tượng bạn đang tập gym theo một lịch cố định: (1) lấy tạ ra, (2) nâng đúng động tác, (3) đo xem hôm nay yếu chỗ nào, (4) điều chỉnh tư thế, (5) _quan trọng nhất_ - đặt tạ về chỗ cũ trước khi lấy hiệp mới. Nếu quên bước “đặt tạ về”, hiệp sau bạn sẽ vác cả tạ cũ lẫn tạ mới - càng tập càng sai. Một _vòng lặp huấn luyện_ (training loop) cho GPT mini cũng y hệt: lấy batch dữ liệu, forward tính loss, backward tính gradient, `optimizer.step()` cập nhật trọng số, và `optimizer.zero_grad()` để “đặt tạ về” - xóa gradient cũ trước hiệp mới.

Ở _Bài 25_ chúng ta đã lo phần “sức khỏe”: khởi tạo trọng số hợp lý, LayerNorm, dropout, grad clipping, AdamW/warmup để mô hình không “chuột rút” (loss bùng nổ hay chết cứng). Bây giờ mọi trợ thủ ổn định đã sẵn sàng, đây là lúc lắp tất cả vào một _guồng máy chạy thật_: ghép data loader, forward, cross-entropy, backward, cập nhật tham số, rồi theo dõi tiến trình bằng val-loss và checkpoint.

> 🔑 **Một câu để nhớ:** Training loop = lặp lại năm bước “batch -> forward -> loss -> backward -> step” cho tới khi loss giảm ổn định, và ĐỪNG BAO GIỜ quên `zero_grad()`.

---

### Bạn sẽ học được gì?

- [ ] Cấu trúc chuẩn của một training loop trong PyTorch và thứ tự các bước.
- [ ] Vai trò của `zero_grad()`, `loss.backward()`, `optimizer.step()` - và vì sao thứ tự quan trọng.
- [ ] Cách lấy batch (x, y) cho bài toán next-token và tính cross-entropy đúng shape.
- [ ] Theo dõi tiến trình: train-loss vs val-loss, `torch.no_grad()`, `model.train()` / `model.eval()`.
- [ ] Lưu và nạp checkpoint để không mất công huấn luyện.

📌 **Yêu cầu trước:** bạn đã có mô hình GPT mini (Bài 21-24), hiểu cross-entropy cho next-token (Bài về loss), và các kỹ thuật ổn định (Bài 25).

---

### 1. Bức tranh lớn: một "hiệp" huấn luyện gồm những gì?

Trước khi viết code, hãy nhìn toàn cảnh. Dữ liệu văn bản đã được token hóa thành một mảng số nguyên `ids` rất dài. Mỗi hiệp (một _iteration_), ta cắt ra một _batch_ các đoạn ngắn, cho mô hình đoán token kế tiếp, đo sai số, rồi đẩy sai số ngược về để chỉnh trọng số.

```text
+-------------------- vòng lặp (mỗi iteration) --------------------+
|                                                                  |
ids (dữ liệu) --> get_batch --> (x, y)  --> model(x) --> logits --> cross_entropy --> loss
                                [B, T]      forward       [B,T,V]     so với y         (số)
                                                                                        |
                  optimizer.step() <-- gradient <-- loss.backward() <------------------+
                        |                              (autograd điền .grad)
                        v
                  zero_grad()  --> quay lại lấy batch mới
```

- $B$ = batch size (số đoạn trong một batch), $T$ = block size / context length (số token mỗi đoạn), $V$ = vocab size (số token khác nhau).
- _forward_: `logits = model(x)` có shape `[B, T, V]` - với mỗi vị trí, mô hình cho điểm số (logits) cho từng token có thể là token kế tiếp. Nhớ: _logits chưa phải xác suất_; softmax mới biến chúng thành phân phối xác suất.
- _loss_: cross-entropy so logits với $y$ (đáp án - token thật ở vị trí kế tiếp).
- _backward_: `loss.backward()` chạy autograd, điền `.grad` cho mọi tham số.
- _step_: `optimizer.step()` dùng `.grad` để cập nhật trọng số theo AdamW.
- _zero_grad_: xóa `.grad` để hiệp sau không cộng dồn nhầm.

---

### 2. Lấy batch cho bài toán next-token

**Analogy:** như đang che tay đoán chữ tiếp theo khi bạn của bạn nhắn tin - bạn nhìn phần đã hiện ($x$) và đoán chữ ngay sau ($y$). Với ngôn ngữ, $y$ chỉ đơn giản là $x$ _dịch sang phải một token_.

```python
import torch

# ids: 1-D LongTensor of token ids for the whole dataset, shape [N]
def get_batch(ids, block_size, batch_size, device):
    # pick batch_size random start positions; each sample has block_size tokens
    ix = torch.randint(len(ids) - block_size, (batch_size,)) # shape [B]
    x = torch.stack([ids[i     : i + block_size] for i in ix])     # shape [B, T]
    y = torch.stack([ids[i + 1 : i + block_size + 1] for i in ix]) # shape [B, T] (x shifted by 1)
    return x.to(device), y.to(device)

# Example: N=1000, block_size=8, batch_size=4 -> x, y are [4, 8]
# x[0] = [12, 5, 88, ...]   y[0] = [5, 88, ...]  (y is next-token target of x)
```

💡 Vì sao $y$ là $x$ dịch phải 1? Vì GPT huấn luyện kiểu _tự hồi quy (autoregressive)_: tại vị trí $t$, mô hình chỉ nhìn token $\le t$ (nhờ _causal mask_ tam giác dưới) và phải đoán token $t+1$. Một batch `[B, T]` thực ra cho $B \times T$ cặp “đầu vào -> đáp án” cùng lúc.

---

### 3. Tính loss: cross-entropy đúng shape

Cross-entropy đo “mô hình bất ngờ đến đâu” trước đáp án đúng. Công thức cho một vị trí: $\text{loss} = -\ln(p[\text{token\_đúng}])$, với $p = \text{softmax}(\text{logits})$. Nhưng `F.cross_entropy` nhận thẳng _logits_ (nó tự làm log-softmax bên trong, ổn định số học hơn), nên ta KHÔNG softmax trước.

Điểm dễ sai: logits có shape `[B, T, V]` còn $y$ là `[B, T]`, trong khi `F.cross_entropy` muốn `[N, V]` và `[N]`. Ta _trải phẳng_ hai chiều đầu:

```python
import torch.nn.functional as F

logits = model(x) # shape [B, T, V]
B, T, V = logits.shape

loss = F.cross_entropy(
    logits.view(B * T, V), # [B*T, V]
    y.view(B * T)          # [B*T]
)
# loss is a scalar tensor, e.g. tensor(4.17) (~ -ln(1/V) khi mô hình chưa học gì)
```

🎯 **Kiểm tra trực giác:** nếu $V=65$ (character-level) và mô hình đoán ngẫu nhiên đều, loss ban đầu xấp xỉ $-\ln(1/65) = \ln(65) \approx 4.17$. Thấy loss khởi đầu quanh con số này nghĩa là khởi tạo lành mạnh; nếu loss ban đầu là 30 thì có gì đó sai (thường do khởi tạo/scale).

---

### 4. Năm bước lõi và thứ tự BẤT KHẢ XÂM PHẠM

Đây là trái tim của bài. Thứ tự phải đúng, nếu không mô hình học sai hoặc không học gì:

```python
optimizer.zero_grad(set_to_none=True) # 1) clear old gradients (.grad -> None)
logits = model(x)                     # 2) forward pass, shape [B, T, V]
B, T, V = logits.shape
loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T)) # 3) compute loss (scalar)
loss.backward()                       # 4) backprop: autograd fills p.grad for every param
optimizer.step()                      # 5) update weights using .grad (AdamW rule)
```

Vì sao `zero_grad` cần thiết? Vì trong PyTorch `loss.backward()` _cộng dồn_ gradient vào `.grad` (nó `+=`, không ghi đè). Nếu không xóa, hiệp sau gradient sẽ là tổng của nhiều hiệp - sai hoàn toàn. `set_to_none=True` vừa nhanh vừa sạch hơn (đặt `.grad` về `None` thay vì tensor 0).

⚠️ **Cạm bẫy kinh điển:** gọi `optimizer.step()` _trước_ `loss.backward()` -> cập nhật bằng gradient cũ (hoặc `None`). Hoặc quên `zero_grad()` -> gradient tích lũy chồng chất -> loss nhảy loạn xạ. Nhớ thứ tự: _zero -> forward -> loss -> backward -> step_.

---

### 5. Theo dõi tiến trình: train-loss vs val-loss

Chỉ nhìn train-loss dễ tự lừa mình (mô hình có thể “học vẹt” dữ liệu train). Ta chia dữ liệu thành train / val và định kỳ đo cả hai bằng `@torch.no_grad()` để không tốn bộ nhớ tính gradient khi chỉ đánh giá:

```python
@torch.no_grad() # no autograd graph -> saves memory & time
def estimate_loss(model, data_splits, eval_iters, block_size, batch_size, device):
    out = {}
    model.eval() # eval mode: dropout OFF, use running stats
    for split, ids in data_splits.items(): # e.g. {'train': ..., 'val': ...}
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(ids, block_size, batch_size, device)
            logits = model(x) # [B, T, V]
            B, T, V = logits.shape
            losses[k] = F.cross_entropy(logits.view(B*T, V), y.view(B*T)).item()
        out[split] = losses.mean().item()
    model.train() # back to train mode: dropout ON
    return out
# example print: step 500: train loss 1.83, val loss 2.05
```

💡 `model.train()` vs `model.eval()`: chúng bật/tắt dropout và đổi cách LayerNorm/BatchNorm hoạt động. Quên `model.eval()` khi đánh giá -> dropout vẫn bật -> val-loss nhiễu và cao giả tạo. Nhớ trả về `model.train()` sau khi đo!

**Bảng so sánh chế độ:**

| Chế độ                              | Dropout | Gradient (autograd)       | Dùng khi                |
| :---------------------------------- | :------ | :------------------------ | :---------------------- |
| `model.train()`                     | Bật     | Có (trong forward thường) | Huấn luyện              |
| `model.eval()` + `@torch.no_grad()` | Tắt     | Không tính, không lưu     | Đánh giá / sinh văn bản |

---

### 6. Ghép thành training loop hoàn chỉnh + checkpoint

```python
import torch

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # AdamW: decoupled weight decay
data_splits = {'train': train_ids, 'val': val_ids}

max_iters, eval_interval, eval_iters = 5000, 500, 200
best_val = float('inf')

for it in range(max_iters):
    # --- periodic evaluation & checkpoint ---
    if it % eval_interval == 0 or it == max_iters - 1:
        stats = estimate_loss(model, data_splits, eval_iters, block_size, batch_size, device)
        print(f"step {it}: train loss {stats['train']:.4f}, val loss {stats['val']:.4f}")
        if stats['val'] < best_val: # keep the best model so far
            best_val = stats['val']
            torch.save({
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'iter': it
            }, 'ckpt.pt')

    # --- one training step (the 5 core steps) ---
    x, y = get_batch(train_ids, block_size, batch_size, device) # [B, T], [B, T]

    optimizer.zero_grad(set_to_none=True) # 1
    logits = model(x)                     # 2 [B, T, V]
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T)) # 3 scalar
    loss.backward()                       # 4
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # (Bài 25) grad clipping
    optimizer.step()                      # 5

# --- resume later ---
# ckpt = torch.load('ckpt.pt')
# model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optim'])
```

🎯 Lưu _cả_ `optimizer.state_dict()`: AdamW giữ moment ước lượng ($m, v$) cho từng tham số. Nạp lại chỉ trọng số mà bỏ optimizer state -> huấn luyện tiếp bị “khựng” vài trăm bước vì phải hâm nóng lại moment.

---

### Nhìn sâu hơn

- **Iteration vs epoch:** một _iteration_ = một batch/một lần cập nhật; một _epoch_ = duyệt hết dữ liệu một lượt. Với LLM, người ta thường đếm theo _số iteration/số token đã thấy_ hơn là epoch, vì dữ liệu khổng lồ và lấy batch ngẫu nhiên có lặp.
- **Vì sao đo loss trung bình nhiều batch?** Một batch quá nhiễu để phán xét. Trung bình `eval_iters` batch cho ước lượng ổn định hơn về “mô hình đang ở đâu”.
- **Bộ nhớ:** khi huấn luyện, autograd lưu _activation_ để tính gradient -> ngốn VRAM. Dùng `@torch.no_grad()` lúc đánh giá/sinh giúp cắt phần này. Đây cũng là lý do batch huấn luyện phải nhỏ hơn batch inference.
- **Non-determinism:** `torch.randint` khiến mỗi lần chạy khác nhau. Đặt `torch.manual_seed(1337)` đầu script để tái lập kết quả khi debug.

---

### Thực hành

- **[Dễ]** Thêm dòng in `loss.item()` mỗi 100 iteration và vẽ nhanh đường loss (`list` + `print`). Quan sát nó có giảm không.
- **[Trung bình]** Cố tình _xóa_ dòng `optimizer.zero_grad()` rồi chạy vài trăm bước. Ghi lại hiện tượng loss (thường bùng nổ hoặc dao động mạnh) và giải thích vì sao.
- **[Thử thách]** Thêm _gradient accumulation_: gọi `loss = loss / accum_steps`, `backward()` mỗi bước, nhưng chỉ `optimizer.step()` + `zero_grad()` sau mỗi `accum_steps` batch. Giải thích vì sao điều này mô phỏng batch lớn hơn mà không tốn thêm VRAM.

---

### Hiểu lầm thường gặp

- ❌ “Gọi `loss.backward()` là trọng số tự cập nhật luôn.” ➔ ✅ `backward()` chỉ _tính_ và điền `.grad`. Phải `optimizer.step()` mới _cập nhật_ trọng số.
- ❌ “Không cần `zero_grad()` nếu loss đang giảm.” ➔ ✅ PyTorch _cộng dồn_ gradient; thiếu `zero_grad()` là gradient của nhiều hiệp chồng lên nhau -> sai và bất ổn.
- ❌ “Val-loss cao hơn train-loss là mô hình hỏng.” ➔ ✅ Val hơi cao hơn train là _bình thường_. Chỉ khi val-loss _tăng dần_ trong khi train-loss vẫn giảm mới là dấu hiệu overfitting.
- ❌ “Đánh giá và sinh văn bản cứ để `model.train()` cũng được.” ➔ ✅ Phải `model.eval()` để tắt dropout; quên là kết quả nhiễu và không tái lập.

---

### Mẹo ghi nhớ

- Thần chú 5 bước: _“Zero, Forward, Loss, Backward, Step”_ (Z-F-L-B-S).
- “`backward` tính, `step` cập nhật, `zero_grad` dọn dẹp.”
- `no_grad` + `eval()` đi cùng nhau khi _đánh giá_; `train()` + gradient đi cùng nhau khi _học_.
- Mẹo: dán một comment `# 1..5` ngay cạnh 5 dòng lõi để không bao giờ đảo thứ tự.

---

### Tóm tắt trực quan

| Khái niệm            | Một dòng                                                       |
| :------------------- | :------------------------------------------------------------- |
| **get_batch**        | Cắt ngẫu nhiên $x$ `[B,T]` và $y = x$ dịch phải 1 (next-token) |
| **forward**          | `logits = model(x)` shape `[B,T,V]` (logits, chưa softmax)     |
| **loss**             | `F.cross_entropy(logits.view(B*T,V), y.view(B*T))` -> scalar   |
| **backward()**       | Autograd điền `.grad` cho mọi tham số (cộng dồn)               |
| **step()**           | AdamW cập nhật trọng số từ `.grad`                             |
| **zero_grad()**      | Xóa `.grad` trước hiệp mới (BẮT BUỘC)                          |
| **eval() + no_grad** | Đo val-loss / sinh: tắt dropout, không lưu graph               |
| **checkpoint**       | `torch.save` cả model + optimizer state để resume              |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Understanding the PyTorch Training Loop | Forward, Loss, Backprop Explained (GenAISA Module 1)](https://www.youtube.com/watch?v=w_L0o3bRD1k)
- PyTorch docs: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials (training loop, optimization): [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT (training loop thực tế của Karpathy): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Attention Is All You Need (paper gốc Transformer): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

---

### Tổng kết

- Training loop = lặp năm bước _zero_grad -> forward -> loss -> backward -> step_; thứ tự này là bất khả xâm phạm.
- `loss.backward()` chỉ _tính_ gradient (và _cộng dồn_), `optimizer.step()` mới _cập nhật_ trọng số, `zero_grad` dọn dẹp trước hiệp mới.
- Lấy batch cho next-token: $y$ là $x$ dịch phải 1 token; cross-entropy nhận thẳng logits, nhớ trải phẳng `[B*T, V]` và `[B*T]`.
- Theo dõi tiến trình bằng train-loss vs val-loss với `model.eval()` + `@torch.no_grad()`, và lưu checkpoint (kèm optimizer state) để resume.

Ta đã có một guồng máy huấn luyện chạy được, nhưng “bộ não điều khiển tốc độ học” - chính là _optimizer_ - vẫn đang để mặc định. Ở _Bài 27: Optimizer AdamW, learning rate và scheduler_, chúng ta sẽ mổ xẻ vì sao AdamW thắng SGD cho Transformer, cách chọn learning rate, và dùng warmup + cosine scheduler để loss giảm mượt hơn. Hẹn gặp ở bài kế tiếp! 🤖
