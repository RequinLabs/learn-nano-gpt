# Bài 29: Huấn luyện GPT mini trên dữ liệu thật

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 6:** Huấn luyện GPT

---

### Giới thiệu

Ở _Bài 28_, bạn đã học cách _nhìn vào loss như một bác sĩ đọc phim X-quang_: phân biệt training loss với validation loss, nhận ra dấu hiệu overfitting (train giảm nhưng val tăng), và dùng regularization (dropout, weight decay). Nhưng suốt các bài trước, ta chủ yếu chạy trên dữ liệu đồ chơi (toy data) vài dòng chữ. Hôm nay là ngày _lên sàn thật_ 🤖.

Hãy tưởng tượng bạn vừa lắp xong một chiếc xe đua trong garage (mô hình GPT mini với attention, các Transformer block, positional embedding…). Giờ là lúc _đổ xăng thật_ (một dataset đủ lớn) và _chạy đường trường_ (một vòng huấn luyện đầy đủ nhiều nghìn bước), rồi theo dõi đồng hồ đo để biết xe có tăng tốc đúng cách hay không.

Trong bài này, chúng ta sẽ chọn một dataset thật (Tiny Shakespeare - tuyển tập kịch của Shakespeare, hoặc một tập thơ/văn tiếng Việt), viết một _training loop hoàn chỉnh_ có đánh giá định kỳ, và học cách _đọc tiến bộ của mô hình_: loss giảm dần, văn bản sinh ra từ _hỗn loạn ngẫu nhiên_ dần thành _những câu gần giống tiếng người_.

> 🔑 **Một câu để nhớ:** Huấn luyện GPT = lặp lại hàng nghìn lần chu trình _lấy một batch dữ liệu -> forward tính cross-entropy loss -> backward tính gradient -> AdamW cập nhật tham số_, và đo tiến bộ bằng validation loss chứ không chỉ training loss.

---

### Bạn sẽ học được gì?

- [ ] Chọn và chuẩn bị một dataset thật (Tiny Shakespeare / thơ tiếng Việt) cho character-level tokenization
- [ ] Chia dữ liệu train/val và viết hàm `get_batch` lấy mẫu ngẫu nhiên các đoạn liên tiếp
- [ ] Viết một _training loop đầy đủ_: forward -> cross-entropy loss -> backward -> AdamW step -> zero_grad
- [ ] Ước lượng loss ổn định bằng cách trung bình nhiều batch trong `torch.no_grad()`
- [ ] Theo dõi tiến bộ: đọc đường loss, sinh mẫu văn bản định kỳ để “nghe” mô hình học
- [ ] Nhận ra khi nào huấn luyện hội tụ, khi nào cần dừng

📌 **Yêu cầu trước:** đã hiểu kiến trúc GPT mini (Bài 20-27), cross-entropy cho next-token, và training/validation loss (Bài 28).

---

### 1. Chọn dataset: character-level Tiny Shakespeare

**Analogy:** dạy một đứa trẻ đoán chữ tiếp theo khi nhắn tin. Cho nó đọc cả một cuốn tiểu thuyết, nó sẽ dần _bắt được nhịp_ của ngôn ngữ - chữ nào hay đi sau chữ nào. Dataset chính là _cuốn sách_ mà GPT mini được đọc đi đọc lại.

Tiny Shakespeare là ~1MB văn bản (khoảng 1 triệu ký tự) - đủ nhỏ để chạy trên Colab, đủ lớn để học cấu trúc thật của ngôn ngữ. Với tiếng Việt, bạn có thể dùng một tập thơ/truyện `.txt`; hãy đảm bảo file lưu _UTF-8_ để giữ nguyên dấu tiếng Việt.

Ở mức _character-level_ (khác BPE - Byte Pair Encoding của LLM lớn), mỗi ký tự là một token. Cách này đơn giản, không cần thư viện tokenizer, rất hợp để học nguyên lý.

```python
import torch

# Read the raw text file (UTF-8 to keep Vietnamese diacritics)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(len(text)) # e.g. 1115394 characters

# Build the character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars) # e.g. 65 for Tiny Shakespeare

# Two dicts: char -> id and id -> char
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # str -> list[int]
decode = lambda l: ''.join(itos[i] for i in l) # list[int] -> str

# Encode the whole dataset into one long 1-D tensor of token ids
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype) # torch.Size([1115394]) torch.int64
```

💡 _Character-level vs BPE:_ character-level cho `vocab_size` nhỏ (vài chục) nhưng chuỗi rất dài (mỗi ký tự 1 token). BPE gộp cụm ký tự hay gặp thành 1 token -> `vocab_size` lớn hơn (vài chục nghìn) nhưng chuỗi ngắn hơn. LLM thật dùng BPE; ta dùng character-level để học nguyên lý.

---

### 2. Chia train/val và lấy batch

Giống Bài 28: ta giữ lại ~10% dữ liệu làm _validation set_ mà mô hình _không bao giờ được học_, để đo khả năng khái quát (generalization) chứ không phải học thuộc.

```python
# 90/10 train/validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 256 # context length: how many tokens the model sees at once
batch_size = 64  # how many independent sequences per batch

def get_batch(split):
    d = train_data if split == 'train' else val_data
    # Random start positions, one per sequence in the batch
    ix = torch.randint(len(d) - block_size, (batch_size,)) # shape: (batch_size,)
    x = torch.stack([d[i : i + block_size] for i in ix])       # (batch_size, block_size)
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix]) # (batch_size, block_size)
    return x, y

xb, yb = get_batch('train')
print(xb.shape, yb.shape) # torch.Size([64, 256]) torch.Size([64, 256])
```

Mỗi cặp $(x, y)$ là _đầu vào_ và _đáp án dịch phải 1 vị trí_: tại vị trí $t$, mô hình nhìn `x[:, :t+1]` và phải đoán `y[:, t]` (token kế tiếp). Đây là bản chất của _next-token prediction_.

```text
x = [ H  e  l  l  o ]      (block_size tokens vào)
y = [ e  l  l  o  ,]      (mỗi vị trí = token KẾ tiếp của x)
      ^  ^  ^  ^  ^
   dự đoán y[t] chỉ dựa vào x[0..t] (causal mask đảm bảo điều này)
```

---

### 3. Chu trình huấn luyện: forward -> loss -> backward -> step

Analogy dây chuyền lắp ráp: mỗi bước huấn luyện là một vòng. Nguyên liệu (batch) đi vào, sản phẩm (dự đoán) đi ra, so với bản mẫu (đáp án) để tính _sai lệch_ (loss), rồi _thợ_ (AdamW) chỉnh máy móc (tham số) một chút. Bốn bước KHÔNG được đảo thứ tự:

```text
batch (x, y)
      |
      v
  logits = model(x)              # forward: (B, T, vocab_size)
      |
      v
  loss = cross_entropy(logits, y)  # so dự đoán với đáp án
      |
      v
  loss.backward()                # backward: autograd tính d(loss)/d(param)
      |
      v
  optimizer.step()               # AdamW cập nhật param theo gradient
      |
      v
  optimizer.zero_grad()          # xóa gradient cũ cho vòng sau
```

```python
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPTMini(vocab_size).to(device) # model built in earlier lessons

# AdamW = Adam + decoupled weight decay (the standard choice for Transformers)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

max_iters = 5000
for step in range(max_iters):
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb) # forward; loss = cross-entropy over next-token
    # logits shape: (batch_size, block_size, vocab_size) = (64, 256, 65)

    optimizer.zero_grad(set_to_none=True) # clear old gradients
    loss.backward()                       # backprop: fill .grad of every param
    optimizer.step()                      # AdamW updates parameters
```

Bên trong `model(xb, yb)`, cross-entropy được tính như sau (nhắc lại từ các bài trước):

```python
# Inside the model forward, when targets are provided:
# logits: (B, T, C) with C = vocab_size ; targets: (B, T)
B, T, C = logits.shape
loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
# cross_entropy applies softmax internally: it turns LOGITS into probabilities
# then takes -log(prob of the correct next token). LOWER loss = better.
```

⚠️ _logits KHÁC xác suất._ logits là số thực bất kỳ (âm/dương). `F.cross_entropy` áp softmax bên trong để biến logits thành phân phối xác suất rồi tính $-\log$. ĐỪNG tự softmax trước rồi đưa vào cross_entropy - sẽ tính softmax hai lần và sai.

_Loss khởi điểm nên bằng bao nhiêu?_ Trước khi học gì, mô hình đoán đều trên `vocab_size` token, nên loss kỳ vọng $\approx -\ln(1/\text{vocab\_size}) = \ln(\text{vocab\_size})$. Với `vocab_size = 65`: $\ln(65) \approx 4.17$. Loss ban đầu quanh 4.2 -> khởi tạo đúng; nếu ~10 -> sai (khởi tạo weight quá lớn).

---

### 4. Ước lượng loss ổn định (train + val)

Loss một batch dao động mạnh. Để đọc tiến bộ _đáng tin_, ta trung bình loss trên nhiều batch trong `torch.no_grad()` + `model.eval()` để tiết kiệm bộ nhớ và tắt dropout.

```python
@torch.no_grad() # no gradient tracking -> faster, less memory
def estimate_loss(eval_iters=200):
    out = {}
    model.eval() # turn OFF dropout during evaluation
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean() # average over eval_iters batches
    model.train() # turn dropout back ON for training
    return out
```

Gắn nó vào loop, in định kỳ:

```python
eval_interval = 500
for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Example output on Tiny Shakespeare (numbers illustrative):
# step 0:    train 4.2100, val 4.2098
# step 500:  train 2.4300, val 2.4500
# step 2000: train 1.7200, val 1.7600
# step 5000: train 1.4800, val 1.5100
```

🎯 Khoảng cách train / val nhỏ (vd 1.48 vs 1.51) = khái quát tốt. Nếu train tụt sâu (vd 0.9) mà val _tăng lại_ -> overfitting, nên dừng sớm (early stopping) hoặc tăng dropout (Bài 28).

---

### 5. "Nghe" mô hình tiến bộ: sinh mẫu định kỳ

Con số loss là một chuyện, cách trực quan nhất là _nghe_ mô hình nói - sinh vài dòng văn bản sau mỗi mốc để cảm nhận nó đang học.

```python
@torch.no_grad()
def sample(n_tokens=300):
    model.eval()
    # Start from a single newline token as context: shape (1, 1)
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=n_tokens) # autoregressive
    model.train()
    return decode(out[0].tolist())

print(sample())
```

Tiến bộ thường như sau:

```text
step 0    : "qZ;kJ  vX!\nb.q W?xm"     <- hỗn loạn ngẫu nhiên
step 500  : "Ther the mand and to t"   <- đã ra chữ, khoảng trắng đúng chỗ
step 2000 : "The king shall be gone,"  <- câu gần đúng ngữ pháp
step 5000 : "KING RICHARD III: ..."    <- bắt được văn phong kịch
```

Mô hình _chưa hiểu nghĩa_, nhưng nó học được _thống kê của ngôn ngữ_: chữ nào hay theo sau chữ nào, tên nhân vật viết hoa, xuống dòng ở đâu.

---

### Nhìn sâu hơn

- **Chi phí bộ nhớ.** Mỗi bước huấn luyện phải lưu _activation_ của toàn bộ forward để backward dùng lại (khác inference chỉ cần forward), nên huấn luyện tốn VRAM hơn suy luận nhiều. Hết VRAM (`CUDA out of memory`) -> giảm `batch_size` hoặc `block_size`.
- **Vì sao dùng AdamW?** AdamW thích nghi learning rate theo từng tham số và tách weight decay khỏi bước gradient (decoupled) - rất ổn định cho Transformer. Learning rate 3e-4 là điểm khởi đầu kinh điển. Thực tế còn dùng _warmup_ + _cosine decay_ để mượt hơn.
- **Gradient clipping.** Với chuỗi dài, gradient có thể _nổ_ (exploding). Thêm `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` ngay trước `optimizer.step()` để giới hạn độ lớn gradient, tránh loss nhảy loạn thành `NaN`.

💡 Trên Colab, đặt runtime GPU và kiểm `torch.cuda.is_available()` trả `True`. Với mô hình mini vài triệu tham số + Tiny Shakespeare, 5000 bước thường chỉ mất vài phút trên GPU (rất lâu trên CPU).

---

### Thực hành

- **[Dễ]** In `vocab_size` và $\ln(\text{vocab\_size})$, chạy 1 forward và kiểm tra loss ban đầu có gần $\ln(\text{vocab\_size})$ không. Vì sao đây là “loss của kẻ đoán mò”?
- **[Trung bình]** Thay `eval_interval` / `eval_iters`, vẽ đồ thị `train_loss` và `val_loss` theo step (dùng `matplotlib`). Xác định mốc bắt đầu overfitting (nếu có).
- **[Thử thách]** Thay Tiny Shakespeare bằng một tập thơ/văn tiếng Việt UTF-8. Kiểm tra `vocab_size` thay đổi thế nào (nhiều ký tự có dấu hơn), thêm `clip_grad_norm_`, so sánh chất lượng văn bản sinh ra sau 3000 bước.

---

### Hiểu lầm thường gặp

- ❌ “Training loss giảm là mô hình chắc chắn tốt lên.” ➔ ✅ Chỉ khi _validation loss_ cũng giảm. Train giảm mà val tăng = overfitting.
- ❌ “Quên `optimizer.zero_grad()` cũng không sao.” ➔ ✅ Gradient sẽ _cộng dồn_ qua các bước -> cập nhật sai hoàn toàn. Luôn `zero_grad` mỗi vòng.
- ❌ “Cần một GPU siêu mạnh mới huấn luyện được LLM.” ➔ ✅ GPT _mini_ character-level chạy tốt trên GPU miễn phí của Colab.
- ❌ “Loss thấp nghĩa là mô hình đã hiểu nghĩa.” ➔ ✅ Nó mới học _thống kê_ của ký tự tiếp theo, chưa “hiểu” như con người.
- ❌ “Đánh giá cứ để `model.train()` cũng được.” ➔ ✅ Phải `model.eval()` để tắt dropout, nếu không val loss bị nhiễu.

---

### Mẹo ghi nhớ

- Thứ tự 4 bước là _cố định_: _forward -> loss -> backward -> step_ (rồi `zero_grad` cho vòng sau).
- “Batch loss ồn ào, val loss là sự thật”: luôn đo bằng `estimate_loss` trung bình nhiều batch.
- Loss khởi điểm $\approx \ln(\text{vocab\_size})$ là _bài kiểm tra sức khỏe_ đầu tiên.
- 🧠 Mẹo một câu: _Đọc chữ (batch) -> đoán -> chấm điểm (loss) -> rút kinh nghiệm (backward+step) -> lặp lại nghìn lần._

---

### Tóm tắt trực quan

| Khái niệm           | Một dòng                                                         |
| :------------------ | :--------------------------------------------------------------- |
| **Dataset thật**    | Tiny Shakespeare (~1MB) hoặc thơ/văn tiếng Việt UTF-8            |
| **Character-level** | Mỗi ký tự = 1 token; vocab_size nhỏ, chuỗi dài                   |
| **get_batch**       | Lấy ngẫu nhiên batch_size đoạn dài block_size, y = x dịch phải 1 |
| **Cross-entropy**   | Áp softmax bên trong; loss thấp = đoán token kế tiếp tốt         |
| **Loss khởi điểm**  | $\approx \ln(\text{vocab\_size})$ (kẻ đoán mò)                   |
| **AdamW**           | Optimizer chuẩn cho Transformer; lr 3e-4 là điểm khởi đầu        |
| **estimate_loss**   | Trung bình nhiều batch trong no_grad + eval()                    |
| **Sinh mẫu**        | “Nghe” mô hình tiến bộ từ hỗn loạn -> câu gần đúng               |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Let’s build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- nanoGPT của Andrej Karpathy (training loop đầy đủ, chuẩn để đối chiếu): [nanoGPT on GitHub](https://github.com/karpathy/nanoGPT)
- PyTorch docs (AdamW, cross_entropy, autograd): [PyTorch Documentation](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [PyTorch Tutorials](https://pytorch.org/tutorials/)
- The Illustrated Transformer (trực quan hóa attention): [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Attention Is All You Need (paper gốc): [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

### Tổng kết

- Huấn luyện GPT mini = lặp chu trình _forward -> cross-entropy loss -> backward -> AdamW step -> zero_grad_ hàng nghìn bước trên dataset thật.
- Chia _train/val_, đo tiến bộ bằng `estimate_loss` (trung bình nhiều batch trong no_grad + eval), cảnh giác overfitting khi val loss tăng.
- Loss khởi điểm nên $\approx \ln(\text{vocab\_size})$; dùng _gradient clipping_ và (nâng cao) warmup + cosine decay để ổn định.
- Sinh mẫu định kỳ để “nghe” mô hình đi từ hỗn loạn sang những câu gần giống tiếng người.

Bạn vừa huấn luyện xong một GPT mini thật sự! Nhưng nếu tắt máy bây giờ, mọi tham số vừa học sẽ _bay hơi_. Ở _Bài 30: Lưu/tải checkpoint và sinh văn bản từ mô hình đã huấn luyện_, chúng ta sẽ học cách _đóng băng_ mô hình (`torch.save` / `torch.load`), khôi phục đúng trạng thái đã huấn luyện, và sinh văn bản với `temperature`, `top-k`, `top-p` để điều khiển độ sáng tạo. Hẹn gặp ở bài kế tiếp! 🚀
