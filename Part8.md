# Bài 8: Hàm mất mát cross-entropy và vòng lặp huấn luyện đầu tiên

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 2:** Mô hình ngôn ngữ nơ-ron đầu tiên

---

### Giới thiệu

Ở _Bài 7_, bạn đã dựng xong mô hình _bigram_ bằng PyTorch: một bảng embedding biến mỗi token id thành một hàng logits - điểm số thô cho mọi token có thể là _token tiếp theo_. Bạn cho một câu vào, mô hình nhả ra logits, rồi softmax biến chúng thành xác suất. Nhưng có một câu hỏi treo lơ lửng: mô hình _đoán_ như vậy - _đúng hay sai đến mức nào?_ Và quan trọng hơn: làm sao để nó _học cách đoán tốt hơn_? 🤖

Hãy tưởng tượng bạn đang luyện thi trắc nghiệm. Sau mỗi câu, thầy giáo không chỉ nói “đúng/sai” mà chấm điểm theo _mức độ tự tin_: bạn khoanh đáp án đúng với 90% chắc chắn thì bị trừ ít; nhưng nếu bạn _rất tự tin_ vào một đáp án SAI thì bị trừ nặng. _Cross-entropy_ chính là ông thầy chấm điểm đó - nó đo mức “ngạc nhiên” của mô hình khi thấy đáp án thật. Đoán trúng và tự tin → mất mát (loss) thấp. Đoán trật mà vẫn hùng hồn → loss cao ngất.

> 🔑 **Một câu để nhớ:** Cross-entropy đo mức “bất ngờ” của mô hình trước token đúng - nó thấp khi mô hình gán xác suất CAO cho token thật, và cao khi mô hình gán xác suất THẤP cho token thật; huấn luyện = liên tục vặn tham số để loss này nhỏ dần.

Trong bài này ta ghép loss vào mô hình bigram của Bài 7 và viết _vòng lặp huấn luyện tối thiểu_: forward → tính loss → backward → cập nhật tham số. Đây là trái tim đập của MỌI mô hình học sâu, từ bigram tí hon tới GPT khổng lồ - chỉ khác quy mô.

---

### Bạn sẽ học được gì?

- [ ] Hiểu vì sao dùng _cross-entropy_ cho bài toán dự đoán token (phân loại nhiều lớp)
- [ ] Nắm công thức cross-entropy và tính được bằng tay một ví dụ nhỏ
- [ ] Dùng đúng `F.cross_entropy(logits, targets)` trong PyTorch (kể cả cách reshape shape)
- [ ] Viết một _vòng lặp huấn luyện tối thiểu_: forward → loss → backward → step
- [ ] Hiểu vai trò của `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` - và vì sao thứ tự quan trọng
- [ ] Biết loss khởi đầu “hợp lý” nên bằng bao nhiêu (`-ln(1/vocab_size)`) để phát hiện lỗi sớm

📌 **Yêu cầu trước:** đã học Bài 7 (mô hình bigram, embedding → logits), hiểu softmax biến logits thành xác suất (Bài 4), quen tensor PyTorch và autograd cơ bản.

---

### 1. Vì sao lại là cross-entropy?

Dự đoán token tiếp theo thực chất là một bài toán _phân loại nhiều lớp (multi-class classification)_: trong `vocab_size` token, đâu là token đúng kế tiếp? Với phân loại, ta muốn một hàm loss thỏa hai điều:

1. _Phạt theo xác suất gán cho đáp án đúng_: gán xác suất cao cho token thật → loss thấp; gán thấp → loss cao.
2. _Có đạo hàm đẹp_ để backward chạy trơn tru (cross-entropy ghép với softmax cho gradient rất gọn).

**Analogy:** cross-entropy giống _tiền phạt cá cược_. Bạn đặt cược (phân bổ xác suất) lên các lựa chọn. Nếu bạn dồn cược đúng vào kết quả thật → phạt nhẹ. Nếu bạn dồn cược vào chỗ sai → phạt nặng tỉ lệ với độ “chắc như đinh” của cú đặt sai đó.

_Công thức_ cho MỘT ví dụ, khi token đúng có chỉ số $y$:

```text
loss = -ln( p[y] )

trong đó p = softmax(logits)  ->  p[y] là xác suất mô hình gán cho token đúng y
```

Vì $p[y]$ nằm trong $(0, 1]$, nên $-\ln(p[y])$ luôn $\ge 0$:

- $p[y] = 1.0 \to \text{loss} = -\ln(1) = 0$ (đoán hoàn hảo, không bất ngờ)
- $p[y] = 0.5 \to \text{loss} = -\ln(0.5) \approx 0.693$
- $p[y] = 0.01 \to \text{loss} = -\ln(0.01) \approx 4.605$ (rất bất ngờ vì gán xác suất bé cho đáp án đúng)

💡 Đừng nhầm _logits_ với _xác suất_. logits là điểm số thô (số thực bất kỳ, âm dương đều được). softmax mới biến chúng thành _xác suất_ (dương, tổng = 1). Cross-entropy nhận _logits_ ở đầu vào và tự làm softmax bên trong - ta KHÔNG softmax thủ công trước khi đưa vào `F.cross_entropy`.

---

### 2. Tính cross-entropy bằng tay

Giả sử `vocab_size = 3` (3 token: A, B, C) và với một vị trí, mô hình cho `logits = [2.0, 1.0, 0.1]`. Token đúng là _A_ (chỉ số $y = 0$).

**Bước 1 - softmax (biến logits → xác suất):**

```text
exp(2.0)=7.389   exp(1.0)=2.718   exp(0.1)=1.105
tong = 7.389 + 2.718 + 1.105 = 11.212

p = [7.389/11.212, 2.718/11.212, 1.105/11.212]
  = [0.659,        0.242,        0.099]        (tong = 1.0)
```

**Bước 2 - cross-entropy chỉ nhìn xác suất của token đúng $y = 0$:**

```text
loss = -ln(p[0]) = -ln(0.659) = 0.417
```

Mô hình khá tự tin vào A (0.659) nên loss thấp. Nếu token đúng là _C_ ($y=2$, xác suất chỉ 0.099) thì $\text{loss} = -\ln(0.099) = 2.31$ - cao hơn nhiều, vì mô hình “bất ngờ”.

🎯 **Mẹo kiểm tra:** khi mô hình _chưa học gì_ (khởi tạo ngẫu nhiên), nó nên đoán “đều tay” mọi token với xác suất $1/\text{vocab\_size}$. Khi đó $\text{loss} \approx -\ln(1/\text{vocab\_size}) = \ln(\text{vocab\_size})$. Ví dụ `vocab_size=65` (kiểu bộ ký tự tiếng Anh) → loss khởi đầu nên xấp xỉ $\ln(65) \approx 4.17$. Nếu loss ban đầu lệch xa con số này, khả năng cao bạn có lỗi ở khởi tạo hoặc shape.

---

### 3. Cross-entropy trong PyTorch - và bẫy shape

PyTorch có sẵn `torch.nn.functional.cross_entropy`. Nó nhận _logits thô_ (chưa softmax) và _chỉ số lớp đúng_ (số nguyên), rồi tự làm log-softmax + negative-log-likelihood bên trong - vừa gọn vừa ổn định số học.

```python
import torch
import torch.nn.functional as F

# One position: logits over a vocab of size 3, target class = 0 (token A)
logits = torch.tensor([[2.0, 1.0, 0.1]]) # shape (1, 3) = (batch, vocab_size)
targets = torch.tensor([0])              # shape (1,)   = correct class ids
loss = F.cross_entropy(logits, targets)
print(loss.item()) # ~0.417  -> matches our hand calculation above
```

**Bẫy quan trọng** với mô hình ngôn ngữ: đầu ra của bigram có shape `(B, T, vocab_size)` - B câu, mỗi câu T vị trí, mỗi vị trí một hàng logits. Nhưng `F.cross_entropy` mong đợi logits dạng `(N, vocab_size)` và targets dạng `(N,)`. Vì thế ta phải _làm phẳng (reshape)_ hai chiều đầu:

```python
B, T, V = 4, 8, 65                       # batch=4, block/time=8, vocab_size=65
logits = torch.randn(B, T, V)            # (4, 8, 65) raw scores from the model
targets = torch.randint(0, V, (B, T))    # (4, 8) correct next-token ids

# Flatten (B, T) into one long dimension so every position becomes one sample
logits_flat = logits.view(B * T, V)      # (32, 65)  -> N = B*T = 32 samples
targets_flat = targets.view(B * T)       # (32,)      -> 32 correct class ids
loss = F.cross_entropy(logits_flat, targets_flat)
print(loss.item()) # ~4.17 for random logits (close to ln(65)=4.174)
```

Sơ đồ tư duy shape:

```text
logits  (B, T, V)   --- view --->   (B*T, V)   -.
                                                  |--> F.cross_entropy --> loss (scalar)
targets (B, T)      --- view --->   (B*T,)     -'

   B=batch cau,  T=so vi tri/cau,  V=vocab_size
   Moi vi tri (B*T cai) tro thanh 1 mau phan loai
```

⚠️ Nếu quên reshape, `F.cross_entropy` sẽ báo lỗi hoặc hiểu sai chiều. Đây là lỗi phổ biến nhất khi ghép loss vào mô hình ngôn ngữ. Nhớ: _gộp $(B, T)$ thành $N = B \times T$ mẫu_, giữ nguyên chiều V cuối.

---

### 4. Tích hợp loss vào `forward` của mô hình

Cách gọn gàng (chuẩn nanoGPT) là để `forward` nhận thêm `targets` tùy chọn và tự trả về cả `logits` lẫn `loss`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token id directly indexes a row of logits over the vocab
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # (V, V)

    def forward(self, idx, targets=None):
        # idx, targets: (B, T) integer tensors of token ids
        logits = self.token_embedding_table(idx) # (B, T, V)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)   # (B*T, V)
            targets = targets.view(B * T)    # (B*T,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

vocab_size = 65
model = BigramLanguageModel(vocab_size)
xb = torch.randint(0, vocab_size, (4, 8)) # (B, T) inputs
yb = torch.randint(0, vocab_size, (4, 8)) # (B, T) targets (next tokens)
logits, loss = model(xb, yb)
print(loss.item()) # ~4.17 before any training (close to ln(65))
```

---

### 5. Vòng lặp huấn luyện tối thiểu

Đây là phần “phép màu”: lặp đi lặp lại _forward → loss → backward → step_, mỗi vòng vặn tham số một chút để loss giảm.

**Analogy:** giống _học chơi phi tiêu bịt mắt_. Bạn ném (forward), người bên cạnh nói lệch bao xa và về hướng nào (loss + gradient), bạn chỉnh tay một nhịp nhỏ (step), rồi ném lại. Lặp nhiều lần, mũi tên dần trúng đích.

```python
# AdamW is a robust default optimizer for language models
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(1000):
    xb, yb = get_batch()                  # (B, T) input and (B, T) target ids
    logits, loss = model(xb, yb)          # 1) FORWARD: predict + measure error
    optimizer.zero_grad(set_to_none=True) # 2) clear old gradients first!
    loss.backward()                       # 3) BACKWARD: autograd fills .grad for every param
    optimizer.step()                      # 4) STEP: nudge params in the direction that lowers loss

    if step % 100 == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")

# Example output (numbers illustrative, will vary):
# step    0 | loss 4.7312
# step  100 | loss 3.1204
# step  900 | loss 2.4876   <- loss decreasing -> the model is learning!
```

**Bốn bước, đúng thứ tự:**

| Bước             | Lệnh                    | Làm gì                                                                                                  |
| :--------------- | :---------------------- | :------------------------------------------------------------------------------------------------------ |
| **1. Forward**   | `model(xb, yb)`         | Chạy dữ liệu qua mô hình, tính logits và loss                                                           |
| **2. Zero grad** | `optimizer.zero_grad()` | Xóa gradient CŨ (PyTorch cộng dồn grad, không tự xóa)                                                   |
| **3. Backward**  | `loss.backward()`       | Autograd tính $\partial\text{loss}/\partial\text{param}$ cho mọi tham số, lưu vào `.grad`               |
| **4. Step**      | `optimizer.step()`      | Cập nhật $\text{param} \leftarrow \text{param} - \text{lr} \times \text{grad}$ (AdamW tinh vi hơn chút) |

⚠️ Thứ tự _rất quan trọng_. Nếu quên `zero_grad()`, gradient của các vòng trước bị _cộng dồn_ làm bước cập nhật sai bét. Nếu gọi `step()` trước `backward()`, `.grad` còn rỗng → không cập nhật gì cả. Ghi nhớ khẩu quyết: _zero → backward → step_.

💡 Vì sao _AdamW_ chứ không phải SGD thuần? AdamW tự điều chỉnh tốc độ học cho từng tham số dựa trên lịch sử gradient, và tách phần _weight decay_ ra đúng cách - nên hội tụ nhanh và ổn định hơn với mô hình ngôn ngữ. Đây là optimizer mặc định trong nanoGPT và hầu hết LLM hiện đại.

---

### Nhìn sâu hơn

_Vì sao ghép softmax + cross-entropy lại đẹp?_ Khi tính đạo hàm của cross-entropy theo logits, kết quả rút gọn thành $p - y_{\text{onehot}}$ (xác suất dự đoán trừ đi vector one-hot của đáp án đúng) - một biểu thức cực gọn, số học ổn định. Đó là lý do PyTorch gộp hai bước vào một hàm `cross_entropy` thay vì bắt bạn tự làm softmax rồi log (dễ tràn số/underflow).

_Loss là một tensor có `grad_fn`._ Khi gọi `loss.backward()`, PyTorch đi ngược đồ thị tính toán (autograd graph) từ loss về từng tham số, áp dụng quy tắc chuỗi (chain rule) để điền `.grad`. Bạn không phải tự viết đạo hàm - đó chính là sức mạnh của autograd đã học ở đầu khóa.

_Đánh giá không chỉ nhìn loss huấn luyện._ Loss train giảm chưa chắc mô hình tốt trên dữ liệu mới - có thể _overfit_. Ở các bài sau ta sẽ tách tập _validation_ để đo loss riêng, nhưng vòng lặp cốt lõi vẫn đúng như trên.

_Gotcha thường gặp:_ loss ra `nan` thường do learning rate quá lớn (bước cập nhật văng đi), hoặc quên reshape đúng chiều. Loss đứng im hoàn toàn thường vì quên `zero_grad` / `step`, hoặc lr quá nhỏ.

---

### Thực hành

- **[Dễ]** Với `vocab_size = 27`, loss khởi đầu “hợp lý” của mô hình chưa huấn luyện nên xấp xỉ bao nhiêu? (Gợi ý: $\ln(\text{vocab\_size})$.) Tính $\ln(27)$.
- **[Trung bình]** Cho `logits = torch.tensor([[1.0, 3.0, 0.0]])` và `targets = torch.tensor([1])`. Tự tính `softmax`, rồi $-\ln(p[1])$ bằng tay; sau đó kiểm tra bằng `F.cross_entropy(logits, targets).item()`. Hai con số có khớp không?
- **[Thử thách]** Cố tình _bỏ dòng_ `optimizer.zero_grad()` trong vòng lặp huấn luyện rồi chạy vài chục bước. Quan sát loss thay đổi bất thường thế nào (thường dao động hỗn loạn hoặc phân kỳ). Giải thích tại sao gradient bị cộng dồn lại phá hỏng bước cập nhật. Sau đó thêm lại dòng đó và xác nhận loss giảm mượt.

---

### Hiểu lầm thường gặp

- ❌ “Phải tự chạy softmax rồi mới đưa vào `F.cross_entropy`.” → ✅ `F.cross_entropy` nhận _logits thô_ và tự làm log-softmax bên trong. Softmax thủ công trước đó là SAI (làm hai lần).
- ❌ “Loss thấp trên tập train nghĩa là mô hình chắc chắn tốt.” → ✅ Có thể _overfit_; phải đo trên tập validation riêng để biết mô hình tổng quát hóa.
- ❌ “Thứ tự backward / step / zero*grad không quan trọng.” → ✅ Bắt buộc \_zero_grad → backward → step*; sai thứ tự làm gradient cộng dồn hoặc không cập nhật.
- ❌ “Cross-entropy chỉ dùng cho bài phân loại ảnh, không hợp cho ngôn ngữ.” → ✅ Dự đoán token tiếp theo CHÍNH là phân loại nhiều lớp (chọn 1 trong `vocab_size` token), nên cross-entropy là lựa chọn tự nhiên.

---

### Mẹo ghi nhớ

- Cross-entropy = _“độ ngạc nhiên”_: $-\ln(p[\text{đúng}])$. Xác suất cho token đúng càng cao → càng ít ngạc nhiên → loss càng thấp.
- Loss khởi đầu chuẩn $\approx \ln(\text{vocab\_size})$ - dùng nó làm “đèn báo” phát hiện lỗi ngay từ bước 0.
- Vòng lặp huấn luyện = khẩu quyết _“forward - zero - backward - step”_.
- 🧠 **Mẹo vàng:** đưa _logits (chưa softmax)_ vào `F.cross_entropy`, và nhớ reshape `(B,T,V) -> (B*T, V)`.

---

### Tóm tắt trực quan

| Khái niệm              | Một dòng để nhớ                                                            |
| :--------------------- | :------------------------------------------------------------------------- |
| **Cross-entropy**      | Loss phân loại: $-\ln(p[\text{token đúng}])$, thấp khi đoán trúng & tự tin |
| **Vì sao dùng**        | Dự đoán token = phân loại nhiều lớp; gradient ghép softmax rất gọn         |
| **Logits vs xác suất** | logits: điểm thô; softmax mới ra xác suất (tổng=1)                         |
| **F.cross_entropy**    | Nhận logits thô + class id; tự làm log-softmax bên trong                   |
| **Reshape**            | `(B,T,V) -> (B*T,V)` và `(B,T) -> (B*T,)` trước khi tính loss              |
| **Loss khởi đầu**      | $\approx \ln(\text{vocab\_size})$ khi mô hình chưa học gì                  |
| **Vòng lặp**           | forward → zero_grad → backward → step (đúng thứ tự)                        |
| **AdamW**              | Optimizer mặc định, hội tụ nhanh & ổn định cho LLM                         |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Understanding the PyTorch Training Loop | Forward, Loss, Backprop Explained (GenAISA Module 1)](https://www.youtube.com/watch?v=w_L0o3bRD1k)
- PyTorch - `torch.nn.functional.cross_entropy` (tài liệu chính thức): [pytorch.org/docs/stable](https://pytorch.org/docs/stable/)
- PyTorch - hướng dẫn training loop cơ bản: [pytorch.org/tutorials](https://pytorch.org/tutorials/)
- nanoGPT - Andrej Karpathy (bigram + training loop tối giản): [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- The Illustrated Transformer - Jay Alammar: [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- Attention Is All You Need (paper gốc Transformer): [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

---

### Tổng kết

- _Cross-entropy_ đo mức “ngạc nhiên” của mô hình trước token đúng: $\text{loss} = -\ln(p[y])$, thấp khi gán xác suất cao cho đáp án thật.
- Dự đoán token tiếp theo là _phân loại nhiều lớp_, nên cross-entropy là lựa chọn tự nhiên; ghép với softmax cho gradient gọn và ổn định.
- `F.cross_entropy` nhận _logits thô_ (KHÔNG softmax trước) + class id; với mô hình ngôn ngữ phải reshape `(B,T,V) -> (B*T,V)`.
- _Vòng lặp huấn luyện tối thiểu_ = forward → zero_grad → backward → step; sai thứ tự làm gradient hỏng.
- Dùng _AdamW_ làm optimizer mặc định; loss khởi đầu nên xấp xỉ $\ln(\text{vocab\_size})$ để phát hiện lỗi sớm.

👉 _Tiếp theo - Bài 9: “Sinh văn bản (sampling) từ mô hình”_: giờ mô hình đã biết _chấm điểm_ và _học_, ta sẽ cho nó _tự viết_! Bạn sẽ dùng logits ở bước cuối, chạy softmax để lấy phân phối token kế tiếp, rồi _lấy mẫu (sample)_ để sinh ra văn bản mới từng ký tự một - và làm quen với temperature điều khiển độ “táo bạo” của mô hình. Hẹn gặp ở bài sau! 🚀
