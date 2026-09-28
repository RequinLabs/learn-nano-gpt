# Bài 28: Theo dõi loss, overfitting và regularization

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 6:** Huấn luyện GPT

---

### Giới thiệu

Ở [Bài 27](https://learningvn.com/learn/xay-dung-llm-tu-dau-tu-attention-den-gpt-mini-pytorch/lesson/6dff66ab-31c7-435a-ac67-dcdbc22cffc9#), bạn đã lắp AdamW, chỉnh learning rate và gắn scheduler (warmup + cosine) để quá trình huấn luyện đi xuống mượt mà. Giờ mô hình đang chạy, mỗi bước in ra một con số loss. Nhưng con số đó có ý nghĩa gì? Khi nào nên vui, khi nào nên lo, và khi nào nên _dừng lại_?

Hãy tưởng tượng bạn ôn thi. Bạn có một tập _đề đã làm nhiều lần_ (train) và một _đề thi thử chưa từng thấy_ (validation). Nếu bạn học thuộc lòng đáp án của đề cũ, điểm trên đề cũ sẽ rất cao, nhưng gặp đề thi thử lạ thì tạch. Đó chính là _overfitting_: mô hình “học thuộc” dữ liệu huấn luyện thay vì học quy luật tổng quát. Loss train và loss val giống như điểm hai loại đề đó - so hai đường này với nhau mới biết mô hình đang thực sự giỏi hay chỉ đang học vẹt.

> 🔑 **Một câu để nhớ:** Loss train cho biết mô hình khớp dữ liệu cũ tốt cỡ nào; loss val cho biết nó _tổng quát hóa_ ra dữ liệu mới tốt cỡ nào - và khoảng cách giữa hai đường mới là thứ nói lên overfitting.

---

### Bạn sẽ học được gì?

- [ ] Đọc và hiểu ý nghĩa của _đường loss_ train/val theo bước huấn luyện
- [ ] Nhận biết _overfitting_ qua dấu hiệu val loss tăng trong khi train loss vẫn giảm
- [ ] Dùng _dropout_ và _weight decay_ để giảm overfitting, hiểu vì sao chúng hiệu quả
- [ ] Biết _khi nào dừng_ (early stopping) và lưu checkpoint tốt nhất

📌 **Yêu cầu trước:** đã có training loop chạy được với AdamW + scheduler (Bài 26-27), hiểu cross-entropy cho next-token.

---

### 1. Loss là gì và vì sao nó là log?

Mô hình GPT dự đoán token kế tiếp: nó xuất ra logits (điểm thô cho mỗi token trong từ điển), rồi softmax biến thành _xác suất_. Loss là _cross-entropy_ giữa xác suất dự đoán và token đúng thực tế:

$$\text{loss} = -\ln(p[\text{token\_đúng}])$$

**Analogy:** Loss giống “mức ngạc nhiên”. Nếu mô hình gán xác suất 1.0 cho token đúng thì không ngạc nhiên chút nào $\to \text{loss} = -\ln(1) = 0$. Nếu gán 0.5 $\to -\ln(0.5) \approx 0.69$. Nếu đoán bừa trên từ điển $V$ token, xác suất trung bình là $1/V \to \text{loss} \approx \ln(V)$.

💡 Với từ điển ký tự nhỏ $V = 65$ (kiểu Shakespeare của nanoGPT), mô hình _chưa học gì_ sẽ có $\text{loss} \approx \ln(65) \approx 4.17$. Đây là mốc “sanity check” đầu tiên: bước 0 mà loss không xấp xỉ $\ln(V)$ thì code khởi tạo có vấn đề.

```python
import torch
import torch.nn.functional as F

vocab_size = 65
# logits from an untrained model: near-uniform over vocab
# shape: (batch=4, seq_len=8, vocab_size=65)
logits = torch.zeros(4, 8, vocab_size)
targets = torch.randint(0, vocab_size, (4, 8)) # shape: (4, 8)

# cross_entropy expects (N, C) and (N,), so we flatten batch*seq
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
print(loss.item())                                  # ~4.17 (== log(65))
print(torch.log(torch.tensor(65.0)).item())         # 4.174...
```

---

### 2. Đọc đường loss: train vs validation

Mỗi vài trăm bước, ta ước lượng loss trên cả tập train và tập val (dữ liệu mô hình _không_ được cập nhật gradient trên đó). Vẽ hai đường theo step:

```text
loss
 |
5 |\
 |  \  train         val
4 |   \___          o__
 |       \__       o  o__
3 |          \___   o     o__      <- val chạm đáy rồi... đi lên
 |              \__ o        o___
2 |                 \__ o          o___   train vẫn tụt
 |                    \_____________
 +--------------------------------------> step
         (điểm val thấp nhất: DỪNG TỐT NHẤT)
```

Ba kịch bản kinh điển:

| Dấu hiệu                           | Chẩn đoán                             | Hành động                            |
| :--------------------------------- | :------------------------------------ | :----------------------------------- |
| **Cả train & val còn giảm**        | Đang học tốt (underfit nếu còn cao)   | Cứ huấn luyện tiếp                   |
| **Train giảm, val chững rồi tăng** | _Overfitting_                         | Thêm regularization / dừng ở đáy val |
| **Cả hai cao và phẳng**            | Underfitting / LR sai / model quá nhỏ | Tăng model, chỉnh LR (Bài 27)        |

⚠️ _Chỉ nhìn train loss là bẫy chết người._ Train loss gần như luôn giảm được (mô hình đủ lớn có thể học thuộc cả tập train $\to$ loss về $\approx 0$), nhưng điều đó không có nghĩa mô hình tốt. _Val loss mới là thước đo mục tiêu thực sự._

```python
@torch.no_grad() # no gradients during evaluation -> save memory
def estimate_loss(model, get_batch, eval_iters=200):
    out = {}
    model.eval() # disable dropout for stable estimate
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # shape: (eval_iters,)
        for k in range(eval_iters):
            xb, yb = get_batch(split) # xb, yb shape: (B, T)
            _, loss = model(xb, yb)   # forward returns (logits, loss)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train() # re-enable dropout for training
    return out

# Example output every 500 steps:
# step 0:    train 4.17, val 4.17
# step 2000: train 2.05, val 2.10   <- healthy, gap small
# step 6000: train 1.20, val 1.55   <- gap widening = overfitting starts
```

🎯 **Lưu ý `model.eval()` và `@torch.no_grad()`:** khi đánh giá phải _tắt dropout_ và không tính gradient, nếu không con số val loss sẽ nhiễu và tốn bộ nhớ vô ích. Sau đó nhớ `model.train()` để bật lại.

---

### 3. Overfitting: mô hình học thuộc thay vì học hiểu

Overfitting xảy ra khi mô hình có _quá nhiều tham số_ so với lượng dữ liệu (hoặc train quá lâu): nó bắt đầu ghi nhớ nhiễu và chi tiết vụn vặt của tập train thay vì quy luật chung. Dấu hiệu số học: _khoảng cách (gap) `val_loss - train_loss` ngày càng lớn_, và val loss chạm đáy rồi bò lên.

```text
gap = val_loss - train_loss

gap nhỏ, ổn định  ->  mô hình tổng quát tốt      ✅
gap phình to      ->  học thuộc train, tạch val  ❌ (overfitting)
```

**Analogy:** Học sinh giải đúng 100% đề cũ nhưng 40% đề mới $\to$ gap lớn $\to$ học vẹt. Ta cần các “phanh” để mô hình _khó học thuộc hơn_: đó là regularization.

---

### 4. Dropout: bắt mô hình đừng dựa dẫm

_Dropout_ trong lúc huấn luyện _tắt ngẫu nhiên_ một tỉ lệ $p$ các neuron (đặt về 0) ở mỗi bước. **Analogy:** trong một nhóm làm bài, mỗi buổi ngẫu nhiên vắng vài người $\to$ cả nhóm buộc phải học cách không phụ thuộc vào một cá nhân duy nhất, kiến thức trải đều hơn $\to$ khó “học thuộc” theo lối mòn.

Về mặt kỹ thuật: dropout tạo hiệu ứng như huấn luyện đồng thời rất nhiều mạng con khác nhau, giúp mô hình bền vững hơn. Khi `eval()` dropout tự tắt (dùng toàn bộ neuron), và các activation được scale lại để kỳ vọng khớp.

```python
import torch
import torch.nn as nn

dropout = nn.Dropout(p=0.2) # drop 20% of units during training
x = torch.ones(1, 8)        # shape: (1, 8)

dropout.train() # training mode
print(dropout(x))
# e.g. tensor([[1.25, 0., 1.25, 1.25, 0., 1.25, 1.25, 1.25]])
# kept units scaled by 1/(1-p)=1.25 so E[output] stays same

dropout.eval()  # eval mode
print(dropout(x))
# tensor([[1., 1., 1., 1., 1., 1., 1., 1.]]) -> identity
```

Trong một GPT, dropout thường đặt ở: sau softmax của attention weights, sau lớp projection của attention, sau feed-forward, và sau khi cộng positional embedding.

```python
class Block(nn.Module):
    """One Transformer block: attention + MLP, both with residual + LayerNorm."""
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_head, n_embd, dropout) # dropout inside
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), # <- regularize the MLP output
        )

    def forward(self, x):
        # x shape: (B, T, n_embd)
        x = x + self.attn(self.ln1(x)) # pre-norm residual
        x = x + self.ffwd(self.ln2(x)) # pre-norm residual
        return x # shape: (B, T, n_embd)
```

💡 $p$ điển hình là $0.1 - 0.2$ cho GPT nhỏ. Dữ liệu càng ít / model càng to thì tăng dropout; dữ liệu khổng lồ (pretraining thật) thường để rất thấp hoặc 0.

---

### 5. Weight decay: phạt trọng số quá lớn

_Weight decay_ thêm một khoản phạt tỉ lệ với độ lớn của trọng số, đẩy các weight về gần 0 trừ khi chúng thực sự cần lớn. **Analogy:** đánh thuế “nhà quá to” - bạn chỉ xây to khi thật cần, tránh xây hoành tráng vô ích. Trọng số nhỏ hơn $\to$ hàm mượt hơn $\to$ khó bám nhiễu $\to$ ít overfitting.

Trong AdamW, weight decay được _tách rời_ khỏi gradient (đó là chữ _W_ = decoupled Weight decay), khác với L2 cũ dính vào gradient của Adam. Công thức cập nhật (đơn giản hóa):

$$w \leftarrow w - \text{lr} \cdot (\text{adam\_update}(\text{grad}) + \text{weight\_decay} \cdot w)$$

```python
from torch.optim import AdamW

# Common practice: decay weights of Linear/Embedding matrices,
# but NOT biases and LayerNorm params (they are 1D)
decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.dim() >= 2:
        decay.append(p)    # weight matrices
    else:
        no_decay.append(p) # biases, LayerNorm gains/shifts

optimizer = AdamW([
    {'params': decay, 'weight_decay': 0.1},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=3e-4, betas=(0.9, 0.95))
```

⚠️ Đừng áp weight decay lên bias và tham số LayerNorm - phạt chúng thường làm hại nhiều hơn lợi. Đây là mẹo chuẩn của nanoGPT.

| Kỹ thuật           | Tác động lên              | Chỉ hoạt động khi train? | Giá trị điển hình         |
| :----------------- | :------------------------ | :----------------------- | :------------------------ |
| **Dropout**        | Tắt ngẫu nhiên activation | ✅ (eval thì tắt)        | $p = 0.1 - 0.2$           |
| **Weight decay**   | Phạt độ lớn weight        | ✅ (mọi bước update)     | $\text{wd} = 0.1$         |
| **Early stopping** | Dừng khi val chạm đáy     | Là chiến lược dừng       | patience $3 - 5$ lần eval |

---

### 6. Khi nào dừng? Early stopping + checkpoint tốt nhất

Đừng train mù tới hết số bước. Hãy theo dõi val loss, _lưu checkpoint mỗi khi val loss đạt kỷ lục thấp mới_, và _dừng_ nếu val loss không cải thiện sau `patience` lần đánh giá liên tiếp (nó đang bò lên = overfitting).

```python
best_val = float('inf')
patience, wait = 5, 0 # stop after 5 evals with no improvement

for step in range(max_iters):
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        m = estimate_loss(model, get_batch) # {'train': .., 'val': ..}
        print(f"step {step}: train {m['train']:.3f}, val {m['val']:.3f}")

        if m['val'] < best_val:
            best_val = m['val']
            torch.save(model.state_dict(), 'best.pt') # keep best checkpoint
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping: val loss no longer improving")
                break
```

🎯 **Điểm mấu chốt:** mô hình “tốt nhất” không phải mô hình ở bước cuối cùng, mà là mô hình tại _đáy val loss_. Vì thế ta lưu `best.pt` và nạp lại nó khi kết thúc.

---

### Nhìn sâu hơn

- **Gap bao nhiêu là chấp nhận được?** Không có ngưỡng tuyệt đối. Gap nhỏ và ổn định là tốt; điều quan trọng là _xu hướng_: val loss còn giảm thì cứ train, khi nó bắt đầu tăng đều mới lo.
- **Regularization không phải liều thuốc chữa mọi bệnh.** Nếu val loss cao vì _thiếu dữ liệu_, cách tốt nhất là thêm dữ liệu. Nếu underfitting (cả hai đều cao), tăng regularization sẽ làm tệ hơn - lúc đó cần model to hơn / train lâu hơn / LR đúng.
- **Nhiễu trong val loss:** dùng `eval_iters` đủ lớn (vd 200 batch) để trung bình ổn định, tránh bị một batch xui làm hiểu nhầm là overfitting.
- **Đơn vị dễ hiểu hơn:** $\text{perplexity} = \exp(\text{loss})$ = “số lựa chọn trung bình mô hình phân vân”. Loss 4.17 $\to$ perplexity 65 (đoán bừa trên 65 ký tự); loss 1.2 $\to$ perplexity $\approx 3.3$ (đã thu hẹp còn ~3 lựa chọn). Theo dõi perplexity đôi khi trực quan hơn.

---

### Thực hành

- **[Dễ]** Chạy `estimate_loss` ở bước 0 và xác nhận $\text{loss} \approx \ln(\text{vocab\_size})$. Thử đổi `vocab_size` và kiểm tra công thức.
- **[Trung bình]** Vẽ (bằng `matplotlib`) hai đường train/val loss của một lần train GPT mini. Xác định bước val loss chạm đáy - đó là nơi nên dừng.
- **[Thử thách]** Thí nghiệm siêu tham số: chạy 3 cấu hình dropout $p \in \{0.0, 0.1, 0.3\}$ với cùng số bước, so gap val-train cuối cùng. Cấu hình nào tổng quát tốt nhất? Lặp lại với $\text{weight\_decay} \in \{0.0, 0.1\}$.

---

### Hiểu lầm thường gặp

- ❌ “Train loss giảm là mô hình đang tốt lên.” ➔ ✅ Chỉ đúng nếu _val loss cũng giảm_. Train loss về 0 mà val loss tăng = học thuộc, mô hình vô dụng với dữ liệu mới.
- ❌ “Dropout luôn bật, kể cả khi sinh văn bản.” ➔ ✅ Dropout _chỉ bật lúc train_. Khi `eval()`/inference nó tự tắt; quên gọi `model.eval()` sẽ khiến kết quả nhiễu ngẫu nhiên.
- ❌ “Thêm nhiều regularization luôn tốt hơn.” ➔ ✅ Quá nhiều dropout/weight decay gây _underfitting_ (cả train lẫn val đều cao). Regularization là để cân bằng, không phải càng nhiều càng ngon.
- ❌ “Weight decay và L2 trong Adam là một.” ➔ ✅ AdamW tách rời (decoupled) weight decay khỏi bước Adam - đó chính là lý do nó tên là AdamW và hoạt động tốt hơn L2 gộp vào gradient.

---

### Mẹo ghi nhớ

- _Train loss = học thuộc; val loss = thi thật._ Luôn nhìn cả hai.
- _Gap phình = overfit; cả hai cao = underfit._
- _Dropout “vắng người” lúc học, đủ mặt lúc thi; weight decay đánh thuế nhà to._
- Mẹo vàng: _Lưu checkpoint tại đáy val loss, dừng khi val hết cải thiện._

---

### Tóm tắt trực quan

| Khái niệm              | Một dòng                                                                      |
| :--------------------- | :---------------------------------------------------------------------------- |
| **Cross-entropy loss** | $-\ln(p[\text{token\_đúng}])$, mốc khởi đầu $\approx \ln(\text{vocab\_size})$ |
| **Train vs val loss**  | Train = khớp dữ liệu cũ; val = tổng quát ra dữ liệu mới                       |
| **Overfitting**        | Val loss tăng trong khi train loss còn giảm (gap phình)                       |
| **Dropout**            | Tắt ngẫu nhiên neuron lúc train; tắt hẳn lúc eval                             |
| **Weight decay**       | Phạt trọng số lớn (AdamW tách rời), làm hàm mượt hơn                          |
| **Early stopping**     | Dừng khi val loss hết cải thiện; giữ checkpoint tốt nhất                      |
| **Perplexity**         | $\exp(\text{loss})$, số lựa chọn trung bình mô hình phân vân                  |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Coding a Neural Network Training Loop from Scratch in PyTorch | LLMs Step by Step](https://www.youtube.com/watch?v=--FaMzvwIDk)
- PyTorch - `nn.Dropout`, `AdamW`, `cross_entropy`: [PyTorch Documentation](https://pytorch.org/docs/stable/)
- PyTorch Tutorials: [PyTorch Tutorials](https://pytorch.org/tutorials/)
- nanoGPT (train loop, weight decay grouping, eval): [nanoGPT on GitHub](https://github.com/karpathy/nanoGPT)
- The Illustrated Transformer: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Attention Is All You Need: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

### Tổng kết

- Loss train/val là _la bàn_ của quá trình huấn luyện; đọc hai đường cùng nhau mới thấy overfitting.
- _Overfitting_ = val loss tăng trong khi train loss còn giảm (gap phình to).
- _Dropout_ (tắt neuron ngẫu nhiên) và _weight decay_ (phạt trọng số lớn, tách rời trong AdamW) là hai phanh giảm overfitting hiệu quả và rẻ.
- _Early stopping_: lưu checkpoint tại đáy val loss, dừng khi val hết cải thiện - đừng train mù.

Giờ bạn đã có đủ đồ nghề để “đọc vị” một quá trình huấn luyện và giữ cho mô hình tổng quát tốt. Ở _Bài 29: Huấn luyện GPT mini trên dữ liệu thật_, ta sẽ ráp tất cả lại - nạp một bộ dữ liệu văn bản thật, chạy full training loop với AdamW + scheduler + dropout + weight decay, theo dõi train/val loss và early stopping - để lần đầu tiên tự tay huấn luyện một GPT mini biết sinh văn bản. Hẹn gặp ở bài tới! 🤖
