# Bài 30: Lưu/tải checkpoint và sinh văn bản từ mô hình đã huấn luyện

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 6:** Huấn luyện GPT

---

### Giới thiệu

Ở _Bài 29_ bạn đã ngồi hàng chục phút (hoặc hàng giờ) nhìn loss tụt dần khi huấn luyện GPT mini trên dữ liệu thật. Nhưng có một câu hỏi đáng sợ: nếu bây giờ bạn tắt Colab, hay máy sập nguồn, thì _toàn bộ công sức đó bay hết_? Mô hình chỉ tồn tại trong RAM/VRAM - đóng notebook là mất trắng.

Hãy nghĩ tới việc _chơi game_: bạn đánh boss cả buổi, và điều đầu tiên bạn làm là bấm _Save_. Lần sau mở game lên, bạn _Load_ lại đúng chỗ đã dừng - không phải chơi lại từ đầu. Huấn luyện mô hình cũng vậy: ta cần biết cách _lưu (save)_ các trọng số đã học ra file, và _tải (load)_ chúng lại để dùng hoặc học tiếp.

Bài này khép lại Tuần 6: bạn sẽ học cách _lưu/tải_ `state_dict`, hiểu _checkpoint_ gồm những gì, rồi dùng chính mô hình đã học để _sinh văn bản_ - và so sánh xem trước với sau khi huấn luyện khác nhau ra sao.

> 🔑 **Một câu để nhớ:** `state_dict` là cuốn từ điển chứa toàn bộ trọng số mô hình; lưu nó ra file là bạn đã “Save game”, và `load_state_dict` là “Load game”.

---

### Bạn sẽ học được gì?

- [ ] Hiểu `state_dict` là gì và vì sao ta lưu nó thay vì lưu cả object mô hình
- [ ] Lưu và tải trọng số bằng `torch.save` / `torch.load` / `load_state_dict`
- [ ] Đóng gói một _checkpoint đầy đủ_ (model + optimizer + step + loss + config) để học tiếp
- [ ] Nạp checkpoint đúng cách, kể cả khi chuyển giữa GPU và CPU (`map_location`)
- [ ] Sinh văn bản từ mô hình đã học và _so sánh trước/sau huấn luyện_

📌 **Yêu cầu trước:** Bài 29 (đã có vòng lặp huấn luyện chạy được), biết `model.eval()` / `torch.no_grad()`, và hàm `generate` autoregressive từ các bài trước.

---

### 1. `state_dict` - cuốn sổ trọng số của mô hình

Hãy tưởng tượng mô hình là một _tủ hồ sơ_: mỗi ngăn kéo có tên (vd `token_embedding.weight`, `blocks.0.attn.query.weight`) và bên trong là một tensor số thực. `state_dict` chính là _danh mục ánh xạ tên -> tensor_ đó. Nó _không_ chứa code kiến trúc, chỉ chứa _giá trị các trọng số_.

```python
import torch
import torch.nn as nn

# A tiny model just to inspect its state_dict
model = nn.Sequential(
    nn.Embedding(65, 32), # vocab=65, d_model=32
    nn.Linear(32, 65),    # projection back to vocab
)

for name, tensor in model.state_dict().items():
    print(name, tuple(tensor.shape))
# Output:
# 0.weight (65, 32)   -> embedding table
# 1.weight (65, 32)   -> Linear weight
# 1.bias   (65,)      -> Linear bias
```

Vì `state_dict` chỉ là dữ liệu số, nó _nhẹ, ổn định giữa các phiên bản code_ và dễ chia sẻ. Nếu bạn lưu cả object bằng `torch.save(model, ...)` (pickle nguyên class), file sẽ _phụ thuộc vào đúng class Python_ lúc load - đổi tên file/thư mục/refactor là hỏng. Vì thế PyTorch khuyến nghị mạnh: _lưu `state_dict`, không lưu cả object_.

💡 **Nguyên tắc vàng:** _Lưu `state_dict`, load `state_dict`._ Class mô hình bạn tự định nghĩa lại trong code; file checkpoint chỉ mang trọng số.

---

### 2. Lưu và tải trọng số cơ bản

Giống việc _Save game_: xuất trạng thái ra một file `.pt` (hoặc `.pth`).

```python
# ---- SAVE ----
torch.save(model.state_dict(), 'model_weights.pt') # write a dict of tensors to disk

# ---- LOAD ----
# 1) Re-create the SAME architecture in code first
model2 = nn.Sequential(
    nn.Embedding(65, 32),
    nn.Linear(32, 65),
)

# 2) Read the saved weights back and copy them in
state = torch.load('model_weights.pt', map_location='cpu') # dict name -> tensor
model2.load_state_dict(state) # copy values into model2
model2.eval() # switch to inference mode (disable dropout, etc.)
```

Chú ý luồng _2 bước_ khi load: (1) dựng lại kiến trúc y hệt trong code, (2) đổ trọng số vào. Nếu kiến trúc không khớp (sai `d_model`, thiếu layer…), `load_state_dict` sẽ báo lỗi _missing/unexpected keys_ - đây là tính năng bảo vệ, không phải bug.

```text
SAVE:  model ---.state_dict()---> {name: tensor}  --torch.save--> file.pt
LOAD:  file.pt --torch.load--> {name: tensor}  --load_state_dict--> model (đã dựng lại)
```

⚠️ `.eval()` khác `torch.no_grad()`. `.eval()` đổi _hành vi layer_ (tắt dropout, dùng thống kê chạy của BatchNorm…). `torch.no_grad()` chỉ _tắt tính gradient_ để tiết kiệm bộ nhớ. Khi sinh văn bản, bạn cần _cả hai_.

---

### 3. Checkpoint đầy đủ - để "học tiếp", không chỉ để chạy

Chỉ lưu trọng số là đủ để _suy luận (inference)_. Nhưng nếu bạn muốn _tạm dừng rồi huấn luyện tiếp_ (resume) thì phải lưu thêm: _trạng thái optimizer_ (AdamW giữ moment $m$, $v$ cho từng tham số!), _step/epoch hiện tại_, loss, và _config_ để biết dựng lại đúng kiến trúc.

Hãy hình dung một checkpoint như một _cặp hồ sơ bàn giao ca_: không chỉ có “kết quả” mà còn “đang làm tới đâu, dụng cụ ở trạng thái nào”.

```python
# ---- SAVE a full checkpoint (a plain Python dict) ----
checkpoint = {
    'step': step,                 # e.g. 5000
    'model_state': model.state_dict(),       # weights
    'optim_state': optimizer.state_dict(),   # AdamW momentum/variance buffers
    'val_loss': best_val_loss,    # float, for bookkeeping
    'config': {
        'vocab_size': 65,
        'd_model': 128,
        'n_layer': 4,
        'n_head': 4,
        'block_size': 256
    },                            # to rebuild architecture
}
torch.save(checkpoint, 'ckpt.pt')

# ---- LOAD to resume training ----
ckpt = torch.load('ckpt.pt', map_location=device)

model = GPTMini(**ckpt['config']).to(device) # rebuild EXACT architecture
model.load_state_dict(ckpt['model_state'])

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
optimizer.load_state_dict(ckpt['optim_state']) # restore AdamW state -> smooth resume

start_step = ckpt['step'] + 1 # continue from next step
model.train() # back to training mode
```

| Lưu gì                   | Chỉ để chạy (inference) | Để học tiếp (resume)                     |
| :----------------------- | :---------------------- | :--------------------------------------- |
| `model.state_dict()`     | ✅                      | ✅                                       |
| `optimizer.state_dict()` | ❌                      | ✅ (thiếu = AdamW mất moment, khựng lại) |
| `step` / `epoch`         | ❌                      | ✅                                       |
| `config` (để dựng lại)   | 👍 nên có               | ✅                                       |
| `val_loss`, `rng_state`  | tùy                     | 👍 nên có                                |

🎯 Bỏ quên `optimizer.state_dict()` là lỗi kinh điển: resume xong AdamW mất hết moment tích lũy, loss bật lên rồi mới xuống lại - lãng phí bước huấn luyện.

---

### 4. `map_location` - cầu nối GPU ↔ CPU

Bạn huấn luyện trên GPU Colab, tải checkpoint về máy chỉ có CPU - nếu `torch.load` cứ đòi CUDA sẽ báo lỗi. `map_location` bảo PyTorch _đưa tensor về thiết bị nào khi nạp_.

```python
# Robust device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Always load to CPU/desired device explicitly; never assume the save device
ckpt = torch.load('ckpt.pt', map_location=device)

# Then .to(device) the rebuilt model as usual
```

💡 Thói quen an toàn: _luôn_ truyền `map_location`. Máy lưu và máy nạp thường khác nhau (Colab GPU -> laptop CPU).

---

### 5. Sinh văn bản từ mô hình đã học

Đây là phần “trả công”: dùng mô hình đã lưu để _đoán chữ tiếp theo, lặp lại_ (autoregressive), giống khi bạn gõ tin nhắn và bàn phím gợi ý từ kế tiếp. Ở character-level, mỗi bước mô hình cho ra _logits_ trên toàn bộ vocab; ta biến logits thành xác suất bằng `softmax`, rồi _lấy mẫu_ một ký tự, nối vào chuỗi, và lặp lại.

```text
[B, T] tokens --embed--> [B, T, d]  --GPT blocks-->  [B, T, d]
                                          |
                                     lm_head (Linear d->V)
                                          v
                                    logits [B, T, V]

  lấy bước cuối logits[:, -1, :]  ->  [B, V]  --softmax-->  probs [B, V]
  multinomial sample -> next token [B, 1] -> nối vào chuỗi -> lặp
```

```python
@torch.no_grad() # no gradients while generating
def generate(model, idx, max_new_tokens, block_size,
             temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:] # crop context to block_size  [B, T<=block]
        logits = model(idx_cond)        # [B, T, V]
        logits = logits[:, -1, :] / temperature # last step only -> [B, V]

        if top_k is not None:
            v, _ = torch.topk(logits, top_k) # keep k highest logits
            logits[logits < v[:, [-1]]] = float('-inf') # mask the rest

        probs = torch.softmax(logits, dim=-1) # [B, V] probabilities, sum=1
        next_id = torch.multinomial(probs, num_samples=1) # sample [B, 1]
        idx = torch.cat((idx, next_id), dim=1) # append -> [B, T+1]

    return idx

# Usage: start from a single newline/BOS token
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = generate(model, context, max_new_tokens=200, block_size=256, temperature=0.8, top_k=40)
print(decode(out[0].tolist())) # decode = map token ids back to characters
```

Hai núm điều khiển quan trọng:

| Tham số       | Ý nghĩa                     | Giá trị nhỏ                          | Giá trị lớn                     |
| :------------ | :-------------------------- | :----------------------------------- | :------------------------------ |
| `temperature` | chia logits trước softmax   | phân phối “nhọn” -> an toàn, lặp lại | “phẳng” -> đa dạng, dễ vô nghĩa |
| `top_k`       | chỉ giữ k token khả dĩ nhất | k nhỏ -> bảo thủ                     | k lớn -> sáng tạo hơn           |

⚠️ `logits` _không_ phải xác suất. Chúng là điểm số thực bất kỳ (có thể âm); chỉ sau softmax mới thành xác suất cộng lại bằng 1.

---

### 6. So sánh trước và sau huấn luyện

Mẹo trực quan hóa “mô hình có học được gì không”: sinh văn bản _ngay khi vừa khởi tạo_ (trọng số ngẫu nhiên) rồi so với sau khi huấn luyện.

```python
# BEFORE training: random weights -> near-uniform logits -> gibberish
# e.g. "qZ,k!3\nx;PvJ .Wm" (rác, không giống ngôn ngữ nào)

# AFTER training on Shakespeare-like text:
# e.g. "To be the king, and see the world of men..."
# words are real, spacing/newlines look like the corpus
```

Trước huấn luyện: logits gần như đều -> softmax cho phân phối gần đều -> chữ ra hoàn toàn ngẫu nhiên. Sau huấn luyện: mô hình đã _nén được thống kê ngôn ngữ_ vào trọng số, nên đoán chữ tiếp theo hợp lý hơn nhiều.

🤖 Đây chính là bằng chứng “biết đọc” của mô hình: cùng một kiến trúc, chỉ khác _giá trị_ `state_dict` - thứ bạn vừa học cách lưu/tải.

---

### Nhìn sâu hơn

- **Kích thước file:** file checkpoint $\approx$ số tham số $\times$ số byte mỗi số. Trọng số float32 = 4 byte; mô hình 10 triệu tham số $\approx$ 40 MB chỉ riêng trọng số. Lưu thêm optimizer state của AdamW $\approx$ _gấp đôi/ba_ (vì có $m$ và $v$ cho mỗi tham số). Đừng ngạc nhiên khi `ckpt.pt` nặng gấp 2-3 lần file chỉ-trọng-số.
- **`strict=False`:** khi fine-tune và kiến trúc lệch chút ít, `load_state_dict(state, strict=False)` cho phép bỏ qua key thiếu/thừa - nhưng phải hiểu rõ mình đang bỏ qua gì.
- **`weights_only=True`:** `torch.load(..., weights_only=True)` (PyTorch mới) an toàn hơn khi nạp file lạ vì không cho phép thực thi code pickle tùy ý. Nên bật khi có thể.
- **Lưu định kỳ + lưu “best”:** nên lưu checkpoint mỗi $N$ bước (chống crash) và lưu riêng bản có `val_loss` thấp nhất (để dùng khi suy luận).

---

### Thực hành

1. **[Dễ]** In ra tất cả key trong `state_dict` của GPT mini của bạn ở Bài 29 và tổng số tham số bằng `sum(p.numel() for p in model.parameters())`. Ước tính kích thước file `.pt` theo float32.
2. **[Trung bình]** Lưu một checkpoint đầy đủ (model + optimizer + step + config), khởi động lại notebook, nạp lại và _huấn luyện tiếp_ vài trăm bước. Kiểm tra loss nối tiếp mượt (không bật vọt) - nếu bật vọt, khả năng bạn quên nạp `optimizer.state_dict()`.
3. **[Thử thách]** Viết vòng lặp sinh 3 mẫu văn bản với `temperature = 0.5, 1.0, 1.5` (cùng `top_k=40`) và so sánh độ mạch lạc/đa dạng. Ghi lại: nhiệt độ càng cao thì văn bản thay đổi thế nào, và vì sao (liên hệ tới độ “nhọn/phẳng” của phân phối sau softmax).

---

### Hiểu lầm thường gặp

- ❌ “Nên `torch.save(model, path)` cho gọn.” ➔ ✅ Cách đó pickle cả class -> dễ vỡ khi refactor/đổi phiên bản. _Lưu `state_dict`_ rồi dựng lại kiến trúc trong code.
- ❌ “Load xong là chạy sinh văn bản được ngay.” ➔ ✅ Phải gọi `model.eval()` (tắt dropout) và bọc `torch.no_grad()`; quên `.eval()` khiến kết quả nhiễu và tốn bộ nhớ.
- ❌ “Chỉ cần lưu trọng số là resume huấn luyện được.” ➔ ✅ Thiếu `optimizer.state_dict()` thì AdamW mất moment tích lũy -> loss bật lên khi học tiếp.
- ❌ “`logits` chính là xác suất chọn token.” ➔ ✅ `logits` là điểm số thô; phải qua softmax mới thành xác suất.
- ❌ “Huấn luyện trên GPU thì load cũng phải có GPU.” ➔ ✅ Dùng `map_location` để đưa tensor về CPU hoặc thiết bị bất kỳ khi nạp.

---

### Mẹo ghi nhớ

- _Save game / Load game_: `state_dict` là ô nhớ; `torch.save` là Save, `load_state_dict` là Load.
- _Bàn giao ca_: inference chỉ cần “kết quả” (trọng số); resume cần “đang làm tới đâu” (optimizer + step).
- `eval` + `no_grad` luôn đi cặp khi sinh văn bản.
- Mẹo nhớ thứ tự load: _dựng kiến trúc trước, đổ trọng số sau_ (không thể đổ nước vào cái ly chưa tồn tại).

---

### Tóm tắt trực quan

| Khái niệm               | Một dòng                                                  |
| :---------------------- | :-------------------------------------------------------- |
| **state_dict**          | Từ điển tên -> tensor trọng số, không chứa code kiến trúc |
| **torch.save/load**     | Ghi/đọc dict tensor ra/từ file .pt                        |
| **load_state_dict**     | Đổ trọng số vào mô hình đã dựng lại                       |
| **Checkpoint đầy đủ**   | model + optimizer + step + config + loss (để resume)      |
| **map_location**        | Chọn thiết bị (CPU/GPU) khi nạp                           |
| **model.eval()**        | Đổi hành vi layer (tắt dropout) cho inference             |
| **generate**            | Lặp: logits -> softmax -> sample -> nối token             |
| **temperature / top_k** | Điều khiển độ đa dạng khi sinh                            |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Let’s reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- Lưu/tải mô hình (chính thức): [https://pytorch.org/tutorials/beginner/saving_loading_models.html](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- Tài liệu API PyTorch: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- nanoGPT của Andrej Karpathy (có logic save/load & generate): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Attention Is All You Need: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

---

### Tổng kết

- `state_dict` là cuốn sổ trọng số; _lưu `state_dict`, không lưu cả object_ để tránh vỡ khi refactor.
- Load theo 2 bước: _dựng lại kiến trúc -> đổ trọng số_; luôn dùng `map_location` và `model.eval()` khi suy luận.
- _Checkpoint đầy đủ_ (thêm optimizer + step + config) mới cho phép _học tiếp mượt mà_.
- Sinh văn bản = lặp logits -> softmax -> sample; `temperature` / `top_k` điều khiển độ đa dạng.
- So sánh trước/sau huấn luyện là bằng chứng trực quan mô hình đã học được thống kê ngôn ngữ.

Đến đây, GPT mini của bạn vẫn đang làm việc ở mức _ký tự (character-level)_ - đơn giản nhưng tốn token và khó nắm bắt từ dài. Ở _Bài 31: Subword tokenization - Byte-Pair Encoding (BPE)_, chúng ta sẽ nâng cấp cách “chặt” văn bản: thay vì từng ký tự, mô hình sẽ làm việc với các “mảnh từ” (subword) do BPE học ra - đúng cách các LLM thật (GPT-2/GPT-3) tokenize. Hẹn gặp ở bài kế tiếp! 🚀
