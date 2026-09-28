# Bài 10: Theo dõi loss, train/val và vì sao mô hình bigram còn yếu

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 2:** Mô hình ngôn ngữ nơ-ron đầu tiên

---

### Giới thiệu

Ở _Bài 9_, bạn đã cho mô hình bigram “nói chuyện”: bắt đầu từ một token, lấy logits, biến thành xác suất bằng softmax, rồi lấy mẫu (sampling) từng ký tự để sinh văn bản. Vui thì vui, nhưng chắc bạn cũng nhận ra kết quả nghe như người vừa tỉnh dậy nói mớ - vài cụm chữ trông “hao hao” tiếng thật, nhưng ghép lại thì vô nghĩa. 🤖

Hôm nay chúng ta làm hai việc rất quan trọng của một kỹ sư ML thực thụ. Thứ nhất: _đo lường_ mô hình tốt hay dở bằng con số loss - và đo trên _cả tập train lẫn tập validation_ để biết mô hình có đang học thật hay chỉ học vẹt. Thứ hai: hiểu _vì sao_ bigram, dù huấn luyện xong, vẫn cứ “đần” - dẫn thẳng tới lý do ta cần một cơ chế tên là _attention_.

Hãy tưởng tượng bạn ôn thi bằng cách học thuộc lòng đáp án đúng 100 câu trong sách bài tập. Điểm trên chính 100 câu đó (train) sẽ rất cao. Nhưng khi ra đề thi thật với câu mới (validation), bạn ngã ngửa. Đo loss trên train/val chính là cách phát hiện mô hình đang _thực sự hiểu quy luật_ hay chỉ _học vẹt_.

> 🔑 **Một câu để nhớ:** train loss cho biết mô hình khớp dữ liệu đã thấy đến đâu; val loss cho biết nó _tổng quát hóa_ ra dữ liệu chưa thấy đến đâu - và bigram “yếu” vì nó chỉ nhìn được _đúng 1 token ngay trước_.

---

### Bạn sẽ học được gì?

- [ ] Hiểu cross-entropy loss cho bài toán dự đoán token kế tiếp thật sự đo cái gì
- [ ] Ước lượng loss ổn định trên train và val bằng cách trung bình nhiều batch với `@torch.no_grad()`
- [ ] Đọc được đường cong train/val: thế nào là học tốt, thế nào là overfitting, thế nào là underfitting
- [ ] Chỉ ra chính xác _hạn chế cốt lõi_ của bigram: chỉ phụ thuộc 1 token trước, mất sạch ngữ cảnh xa
- [ ] Đặt được vấn đề: cần một cơ chế cho token “nhìn lại” _cả ngữ cảnh_ - đó là attention

📌 **Yêu cầu trước:** đã hoàn thành Bài 8-9 (dựng mô hình bigram, training loop với AdamW, sinh văn bản). Quen `torch`, `torch.nn`, `F.cross_entropy`, khái niệm batch / logits / softmax.

---

### 1. Loss là gì? Từ "đoán chữ" tới một con số

**Analogy trước đã:** hãy nghĩ loss như _mức độ ngạc nhiên_ của mô hình. Bạn đưa cho nó một câu và hỏi “ký tự tiếp theo là gì?”. Nếu mô hình dự đoán ký tự đúng với xác suất cao (vd 0.9), nó “không ngạc nhiên” - loss thấp. Nếu ký tự đúng lại bị mô hình gán xác suất bèo bọt (vd 0.01), mô hình “sốc” - loss cao. Huấn luyện = làm mô hình bớt ngạc nhiên trên dữ liệu thật.

Con số hóa sự ngạc nhiên đó là _cross-entropy_ (entropy chéo). Với một vị trí, gọi $p$ là xác suất mô hình gán cho _token đúng_ (token thực sự đứng sau). Loss tại vị trí đó là:

```python
loss_tai_vi_tri = -log(p_dung)

# p_dung = 1.0  -> loss = -log(1)   = 0.0     (đoán chắc chắn đúng, không ngạc nhiên)
# p_dung = 0.5  -> loss = -log(0.5) = 0.693
# p_dung = 0.1  -> loss = -log(0.1) = 2.302
# p_dung -> 0   -> loss -> +vô cực           (rất tự tin nhưng SAI = thảm họa)
```

💡 Chú ý phân biệt: logits là các số thực thô ở đầu ra mô hình; đưa qua softmax mới ra _xác suất_ (dương, cộng lại bằng 1). `F.cross_entropy` trong PyTorch nhận thẳng _logits_ (nó tự làm `log_softmax` bên trong cho ổn định số học) - ĐỪNG softmax trước rồi mới đưa vào, sẽ sai.

Loss toàn bộ = trung bình `-log(p_dung)` trên mọi vị trí. Với bộ ký tự (character-level) có `vocab_size = 65`, nếu mô hình đoán bừa hoàn toàn (mỗi ký tự xác suất $1/65$) thì:

```python
import torch
import torch.nn.functional as F

vocab_size = 65
# Loss của mô hình đoán bừa (uniform) = -log(1/vocab_size) = log(vocab_size)
baseline = torch.log(torch.tensor(vocab_size, dtype=torch.float))
print(baseline.item()) # ~4.174
```

Mốc ~4.17 là “đường cơ sở” (baseline): bigram _phải_ thấp hơn con số này mới thực sự học được gì. Train xong loss vẫn quanh 4.17 nghĩa là mô hình gần như đoán bừa.

---

### 2. Vì sao cần đo trên CẢ train lẫn val?

Ta chia dữ liệu làm hai phần ngay từ Bài 8: train (mô hình được nhìn + cập nhật trọng số từ nó) và val (validation - mô hình _không bao giờ_ học từ đây, chỉ dùng để chấm điểm). Đây là quy tắc bất khả xâm phạm.

```python
# From lesson 8: split the whole text into 90% train, 10% val
data = torch.tensor(encode(text), dtype=torch.long) # shape: (N,)
n = int(0.9 * len(data))
train_data = data[:n] # model learns from here
val_data   = data[n:] # HELD OUT - only for evaluation, never for gradient updates
```

So sánh hai con số:

| Tình huống                     | train loss | val loss                | Ý nghĩa                                 |
| :----------------------------- | :--------- | :---------------------- | :-------------------------------------- |
| **Học tốt**                    | Thấp       | Thấp, ~gần train        | Mô hình nắm được quy luật tổng quát ✅  |
| **Overfitting** (học vẹt)      | Rất thấp   | Cao, ngày càng xa train | Học thuộc train, không tổng quát ⚠️     |
| **Underfitting** (chưa học đủ) | Cao        | Cao                     | Mô hình quá đơn giản / train chưa đủ ⚠️ |

⚠️ Chỉ nhìn train loss là _bẫy chết người_: nó gần như luôn giảm, dễ khiến bạn tưởng mọi thứ ổn trong khi mô hình đang học vẹt. Val loss mới là “đề thi thật”.

---

### 3. Ước lượng loss ổn định: hàm `estimate_loss()`

Có một vấn đề thực tế: loss của _một_ batch dao động rất mạnh (batch này toàn ký tự dễ, batch kia toàn ký tự khó). Đo một batch rồi kết luận thì như chấm học lực qua đúng 1 câu hỏi - hên xui. Cách chuẩn (đúng như nanoGPT làm): lấy trung bình loss trên _nhiều_ batch.

Hai điều tối quan trọng khi đánh giá:

1. Bọc trong `@torch.no_grad()` - ta chỉ đo, KHÔNG cần autograd xây đồ thị đạo hàm $\to$ tiết kiệm bộ nhớ + nhanh hơn.
2. Chuyển sang `model.eval()` khi đo, rồi `model.train()` lại khi huấn luyện tiếp (quan trọng khi sau này có Dropout / LayerNorm hành xử khác giữa train và eval).

```python
eval_iters = 200 # number of batches to average over

@torch.no_grad() # no gradient tracking during evaluation
def estimate_loss(model):
    out = {}
    model.eval() # switch to eval mode (affects Dropout / BatchNorm etc.)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # shape: (eval_iters,)
        for k in range(eval_iters):
            X, Y = get_batch(split) # X, Y shape: (batch_size, block_size)
            _, loss = model(X, Y)   # bigram model returns (logits, loss)
            losses[k] = loss.item()
        out[split] = losses.mean().item() # average loss over eval_iters batches
    model.train() # switch back to train mode
    return out

# Example usage inside the training loop:
# for step in range(max_iters):
#     if step % eval_interval == 0:
#         losses = estimate_loss(model)
#         print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
#     xb, yb = get_batch('train')
#     _, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
```

Output điển hình khi train bigram trên tập ký tự nhỏ (kiểu Tiny Shakespeare) sẽ dần dần trông như:

```text
step 0:    train 4.7305, val 4.7241   # gần baseline ~4.17, mô hình còn ngu ngơ
step 1000: train 2.5211, val 2.5389   # đã giảm rõ, học được thống kê cặp ký tự
step 3000: train 2.4842, val 2.4930   # ... và rồi CHỮNG LẠI quanh ~2.48
```

🎯 Để ý điều lạ: train và val gần như bằng nhau (2.48 vs 2.49) và cả hai đều _không giảm thêm_ dù train lâu hơn. Đây KHÔNG phải overfitting - đây là mô hình đã _chạm trần năng lực_ của chính nó.

---

### 4. Vì sao bigram chạm trần ~2.48? Hạn chế cốt lõi

Phần cốt lõi của bài. Nhớ lại bigram: để đoán ký tự tiếp theo, nó _chỉ nhìn đúng 1 ký tự ngay trước_ rồi tra một bảng. Dự đoán tại vị trí $t$ chỉ phụ thuộc token $x_{t-1}$:

$$P(x_t \mid x_1, x_2, \dots, x_{t-1}) = P(x_t \mid x_{t-1}) \quad \text{<-- bigram giả định như VẬY}$$

**Analogy:** bạn phải đoán từ tiếp theo của câu _nhưng chỉ được nhìn đúng MỘT từ liền trước, mọi từ trước nữa bị che kín_. Ai đó đưa từ “con” và bảo “đoán từ sau”. Là “mèo”? “chó”? “số”? “đường”? Bạn chịu, vì không được thấy “Nhà tôi nuôi một…” phía trước.

Bảng tra của bigram chính là ma trận embedding kích thước `(vocab_size, vocab_size)` - mỗi hàng là phân phối token kế tiếp cho đúng một token đầu vào:

```python
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each row = logits over next token, given ONE current token
        # shape: (vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: (B, T)  -> logits shape: (B, T, vocab_size)
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            B, T, C = logits.shape # cross_entropy expects (N, C) logits and (N,) targets
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss
```

Dù ta đưa vào cả một chuỗi dài `block_size` token (`idx` shape `(B, T)`), khi tính logits tại vị trí $t$, mô hình chỉ dùng `idx[:, t]` - _các cột $0..t-1$ không hề tham gia_ vào dự đoán ở cột $t$. Ngữ cảnh xa bị vứt bỏ hoàn toàn.

Sơ đồ so sánh luồng thông tin:

```text
BIGRAM (mỗi cột chỉ tự lo)                ĐIỀU TA CẦN (cột t nhìn về mọi cột <= t)

  x0   x1   x2   x3                          x0   x1   x2   x3
  |    |    |    |                           |  \  |  \  |  \  |
  v    v    v    v                           v   \ v   \ v   \ v
 [E]  [E]  [E]  [E]   <- cùng 1 bảng tra      x0 x0,x1 x0..x2 x0..x3
  |    |    |    |                             \   |    /    /
  v    v    v    v                              tổng hợp ngữ cảnh (attention!)
 y1'  y2'  y3'  y4'                                     |
                                                        v
 (dự đoán y_t chỉ dựa x_{t-1})                 (dự đoán y_t dựa CẢ ngữ cảnh)
```

Vì thế ~2.48 là mức tốt nhất mà thống kê cặp-ký-tự có thể đạt. Muốn thấp hơn, mô hình _bắt buộc_ phải biết dùng thông tin từ các token xa hơn.

💡 So sánh: bigram nhìn 1 token trước; trigram nhìn 2; n-gram nhìn $n-1$. Nhưng tăng $n$ thì bảng phình theo cấp số nhân ($V^n$) và vẫn cố định độ dài ngữ cảnh. Ta cần thứ vừa _nhìn xa tùy ý_ vừa _không phình bảng_ - đó là attention.

---

### Nhìn sâu hơn

- **Vì sao val loss của bigram không tách khỏi train loss?** Vì bigram chỉ có `vocab_size * vocab_size` tham số (vocab 65 $\to$ ~4.225 tham số) - quá ít để “học thuộc” một corpus hàng trăm nghìn ký tự. Nó _không đủ sức_ overfit. Overfitting là bệnh của mô hình _thừa_ năng lực; bigram thì ngược lại - _thiếu_ năng lực. Khi lắp attention + nhiều tham số hơn, khoảng cách train/val mới quan trọng và ta sẽ cần kỹ thuật chống overfit (dropout, weight decay).
- **Ước lượng loss là ngẫu nhiên:** Vì `get_batch` lấy vị trí ngẫu nhiên, hai lần gọi `estimate_loss` cho số hơi khác nhau. Tăng `eval_iters` $\to$ ổn định hơn nhưng chậm hơn; 200 là điểm cân bằng nanoGPT hay dùng.
- **Perplexity - cách đọc loss thân thiện hơn:** $\text{perplexity} = \exp(\text{loss})$. Loss 2.48 $\to$ perplexity ~11.9: mô hình “phân vân giữa khoảng 12 lựa chọn” mỗi bước. Baseline uniform 4.17 $\to$ perplexity 65 (đúng số ký tự - phân vân cả bảng chữ). Càng nhỏ càng tự tin.

---

### Thực hành

- **[Dễ]** Cho `p_dung = 0.25`, tính `loss = -log(0.25)` bằng tay rồi kiểm bằng `torch.log`. So sánh với baseline `log(65)`: mô hình này đã tốt hơn đoán bừa chưa?
- **[Trung bình]** Chỉnh `eval_iters` từ 200 xuống 10 rồi lên 500. Gọi `estimate_loss` nhiều lần với mỗi giá trị và quan sát: giá trị nào cho train / val _ổn định_ (ít dao động giữa các lần gọi) hơn? Giải thích vì sao.
- **[Thử thách]** Viết hàm tính perplexity từ dict loss (`exp(loss)`), train bigram tới khi chững, in perplexity và trả lời “phân vân giữa khoảng bao nhiêu ký tự?”. Rồi tăng `block_size` từ 8 lên 32 - val loss có giảm không? (Gợi ý: KHÔNG đáng kể; hãy giải thích dựa vào mục 4.)

---

### Hiểu lầm thường gặp

- ❌ “Train loss giảm đều là mô hình đang tốt.” ➔ ✅ Train loss gần như luôn giảm; chỉ val loss mới nói mô hình có _tổng quát hóa_ hay không.
- ❌ “Val loss cao hơn train loss một chút nghĩa là hỏng.” ➔ ✅ Val hơi cao hơn train là _bình thường_. Đáng lo là khi val _tăng dần_ trong khi train tiếp tục giảm (dấu hiệu overfitting).
- ❌ “Bigram sinh chữ dở vì train chưa đủ lâu.” ➔ ✅ Train lâu hơn cũng vô ích - nó _chạm trần kiến trúc_: chỉ nhìn 1 token trước thì không thể giỏi hơn.
- ❌ “Cứ đưa `block_size` lớn vào là bigram thấy được ngữ cảnh dài.” ➔ ✅ Bigram bỏ qua mọi cột trừ cột ngay trước; `block_size` lớn chỉ giúp tính song song nhiều dự đoán, KHÔNG cho nó thấy xa hơn.
- ❌ “Nên softmax trước rồi mới đưa vào `F.cross_entropy`.” ➔ ✅ `F.cross_entropy` nhận _logits_ thô và tự làm `log_softmax` ổn định số học bên trong.

---

### Mẹo ghi nhớ

- Loss = _mức ngạc nhiên_; `-log(p_dung)`; càng chắc + đúng $\to$ càng gần 0.
- Luôn đo _train VÀ val_, luôn `@torch.no_grad()` + `model.eval()/train()`.
- Trung bình _nhiều batch_ (`eval_iters`) để ước lượng khỏi nhiễu.
- Baseline uniform = `log(vocab_size)`; bigram trần ~2.48; muốn thấp hơn $\to$ cần _ngữ cảnh_.
- 🧠 Mẹo: _“Bi = 2, gram = chữ”_ $\to$ bigram nhìn 1 chữ trước để đoán chữ sau. Một chữ thì thấy được bao nhiêu? $\to$ Đó là lý do ta cần attention.

---

### Tóm tắt trực quan

| Khái niệm              | Một dòng                                                           |
| :--------------------- | :----------------------------------------------------------------- |
| **cross-entropy loss** | `-log(xác suất mô hình gán cho token đúng)`, trung bình mọi vị trí |
| **baseline uniform**   | `log(vocab_size)` - mốc “đoán bừa”; loss phải thấp hơn             |
| **train loss**         | Khớp dữ liệu đã thấy; gần như luôn giảm                            |
| **val loss**           | Tổng quát hóa ra dữ liệu chưa thấy; “đề thi thật”                  |
| **estimate_loss()**    | Trung bình loss nhiều batch, `@torch.no_grad()` + `eval()`         |
| **overfitting**        | Train thấp, val tăng dần - học vẹt                                 |
| **Hạn chế bigram**     | Chỉ dùng $x_{t-1}$; mất ngữ cảnh xa; trần loss ~2.48               |
| **perplexity**         | `exp(loss)` - “phân vân giữa bao nhiêu lựa chọn”                   |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Coding a Neural Network Training Loop from Scratch in PyTorch | LLMs Step by Step](https://www.youtube.com/watch?v=--FaMzvwIDk)
- PyTorch docs: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- `F.cross_entropy`: [https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- nanoGPT (hàm `estimate_loss`, chia train/val): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Attention Is All You Need (paper gốc): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

---

### Tổng kết

- loss = mức ngạc nhiên của mô hình; cross-entropy = `-log(p_dung)` trung bình trên mọi vị trí.
- Luôn đo _cả train lẫn val_, dùng `@torch.no_grad()` + `model.eval()` và trung bình nhiều batch để ước lượng ổn định.
- Đọc đường cong: train↓ val↓ = tốt; train↓ val↑ = overfitting; cả hai cao = underfitting.
- Bigram chạm trần ~2.48 không vì thiếu train mà vì _kiến trúc chỉ nhìn 1 token trước_ - mọi ngữ cảnh xa bị vứt bỏ.
- Ta cần một cơ chế cho mỗi token “nhìn lại” toàn bộ ngữ cảnh phía trước một cách có trọng số - vừa nhìn xa tùy ý, vừa không phình bảng.

Ở _Bài 11: Vì sao cần Attention - hạn chế của trung bình ngữ cảnh_, ta sẽ thử ý tưởng đơn giản nhất để “nhìn ngữ cảnh”: lấy _trung bình_ các token phía trước. Bạn sẽ thấy nó đã tốt hơn bigram nhưng vẫn còn ngây thơ (mọi token quá khứ được coi trọng như nhau) - và chính chỗ hụt đó sẽ mở đường cho self-attention thực thụ. Hẹn gặp ở bài sau! 🚀
