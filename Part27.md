# Bài 27: Optimizer AdamW, learning rate và scheduler

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 6:** Huấn luyện GPT

---

### Giới thiệu

Ở _Bài 26_ bạn đã ráp xong một _vòng lặp huấn luyện đầy đủ_: lấy batch dữ liệu, chạy forward, tính loss bằng cross-entropy, gọi `loss.backward()` để có gradient, rồi `optimizer.step()` để cập nhật trọng số. Bạn đã dùng optimizer như một chiếc hộp đen. Hôm nay mình mở chiếc hộp đó ra.

Hãy tưởng tượng bạn đang xuống núi trong sương mù dày đặc để về thung lũng (điểm loss thấp nhất). Bạn chỉ cảm nhận được _độ dốc dưới chân_ (chính là gradient). Câu hỏi sống còn: mỗi bước nên bước _bao xa_ (learning rate), có nên _nhớ hướng đã đi_ để không đổi hướng lung tung (momentum), và làm gì khi gặp một vách đá dốc đứng khiến bạn suýt lao xuống vực (gradient bùng nổ)? Optimizer, learning rate, scheduler và gradient clipping chính là bốn câu trả lời cho việc “đi xuống núi” một cách an toàn và nhanh chóng.

> 🔑 **Một câu để nhớ:** AdamW quyết định _hướng và độ lớn tương đối_ của mỗi bước, learning rate + scheduler quyết định _độ dài bước theo thời gian_, còn gradient clipping là _dây an toàn_ chống lao xuống vực.

---

### Bạn sẽ học được gì?

- [ ] Hiểu vì sao Transformer thường dùng _AdamW_ thay vì SGD thuần, và AdamW khác Adam ở điểm nào (weight decay tách rời).
- [ ] Hiểu _learning rate_ là gì và vì sao nó là siêu tham số quan trọng nhất.
- [ ] Cài đặt lịch _warmup + cosine decay_ cho learning rate bằng PyTorch.
- [ ] Dùng _gradient clipping_ để chống bùng nổ gradient trong lúc huấn luyện.
- [ ] Ghép tất cả vào vòng lặp huấn luyện của Bài 26 đúng thứ tự.

📌 **Yêu cầu trước:** đã hiểu vòng lặp huấn luyện (Bài 26), gradient/backprop, và biết cross-entropy dùng cho next-token.

---

### 1. Optimizer làm gì? Từ SGD tới Adam

**Analogy:** SGD (Stochastic Gradient Descent) giống người đi bộ chỉ nhìn độ dốc _ngay dưới chân_ rồi bước một đoạn cố định theo hướng dốc nhất. Nếu địa hình gồ ghề (gradient dao động mạnh), người đó cứ giật qua giật lại, đi rất chậm.

Công thức SGD cơ bản cho trọng số $w$:

```text
w <- w - lr * g        (g = gradient của loss theo w)
```

_Adam_ thông minh hơn: nó giữ hai “cuốn sổ ghi nhớ” (moving average) cho mỗi tham số:

- $m$ = trung bình động của gradient (momentum - hướng trung bình gần đây).
- $v$ = trung bình động của _bình phương_ gradient (độ lớn trung bình - để chuẩn hóa).

```text
m <- b1*m + (1-b1)*g            # 1st moment (momentum)
v <- b2*v + (1-b2)*g^2          # 2nd moment (variance)
m_hat = m / (1 - b1^t)          # bias correction
v_hat = v / (1 - b2^t)
w <- w - lr * m_hat / (sqrt(v_hat) + eps)
```

Ý nghĩa trực giác: tham số nào gradient hay lớn thì $\sqrt{\hat{v}}$ lớn $\to$ bước bị _chia nhỏ lại_ (đi thận trọng); tham số nào gradient nhỏ đều thì được bước dài hơn. Đây là _learning rate thích nghi cho từng tham số_, cực hợp với Transformer nơi các lớp có độ lớn gradient rất khác nhau.

---

### 2. Vì sao AdamW chứ không phải Adam?

_Weight decay_ (giảm trọng số) là kỹ thuật regularization: kéo trọng số về gần 0 để mô hình đơn giản hơn, chống overfitting (chi tiết ở Bài 28). Vấn đề: trong _Adam gốc_, weight decay bị nhét vào gradient rồi mới đưa qua bộ chia $\sqrt{\hat{v}}$, khiến lượng decay bị _méo_ theo từng tham số - không còn là decay “thuần” nữa.

_AdamW_ (Loshchilov & Hutter, 2017) sửa điều này bằng cách _tách rời_ (decoupled) weight decay ra khỏi bước Adam: decay được áp trực tiếp lên trọng số, độc lập với $\sqrt{\hat{v}}$.

```text
# AdamW: weight decay được áp TRỰC TIẾP, tách khỏi bước adaptive
w <- w - lr * ( m_hat / (sqrt(v_hat) + eps) )   # bước Adam thuần
w <- w - lr * wd * w                            # decay tách rời (decoupled)
```

| Đặc điểm                             | SGD + momentum | Adam                        | AdamW             |
| :----------------------------------- | :------------- | :-------------------------- | :---------------- |
| **Learning rate thích nghi/tham số** | Không          | Có                          | Có                |
| **Weight decay**                     | Đúng (L2)      | Bị méo qua $v_{\text{hat}}$ | Tách rời, đúng    |
| **Dùng cho GPT/Transformer**         | Ít             | Được                        | ✅ Chuẩn de-facto |

💡 Gần như mọi mô hình GPT hiện đại (GPT-2, GPT-3, nanoGPT) đều dùng _AdamW_ với `betas=(0.9, 0.95)` và `weight_decay ≈ 0.1`. Lưu ý $\beta_2=0.95$ (thấp hơn mặc định 0.999) giúp thích nghi nhanh hơn với loss ồn của mô hình ngôn ngữ.

```python
import torch

# model: GPT mini của bạn từ các bài trước
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,            # learning rate cao nhất (peak)
    betas=(0.9, 0.95),  # 2nd moment giảm nhanh hơn mặc định cho LLM
    eps=1e-8,
    weight_decay=0.1,   # decoupled weight decay
)
# optimizer giữ state m, v cho từng tham số -> tốn ~2x bộ nhớ so với chỉ params
```

---

### 3. Learning rate: siêu tham số quan trọng nhất

**Analogy:** learning rate ($lr$) là _độ dài mỗi bước chân_. Bước quá ngắn $\to$ đi cả ngày không tới (huấn luyện chậm, có khi kẹt). Bước quá dài $\to$ nhảy qua cả thung lũng, thậm chí văng lên đỉnh đối diện $\to$ loss _phân kỳ_ (tăng vọt, thành `NaN`).

```text
lr quá nhỏ:   .  .  .  .  .        (loss giảm rất chậm)
lr vừa đủ:    . _ . _ .            (loss giảm mượt, nhanh)
lr quá lớn:   . / \ / \ /  NaN     (loss bật lên, phân kỳ)
```

⚠️ Nếu loss bỗng tăng vọt hoặc thành `NaN` ngay khi bắt đầu huấn luyện, thủ phạm số một là learning rate quá lớn. Hãy giảm nó xuống (vd chia 3 hoặc 10) trước khi nghi ngờ chỗ khác.

---

### 4. Scheduler: warmup + cosine decay

Giữ $lr$ cố định không tối ưu. Chiến lược chuẩn cho Transformer gồm hai pha:

1. _Warmup:_ vài trăm bước đầu, tăng $lr$ _tuyến tính_ từ ~0 lên đỉnh $lr$. Lý do: lúc mới khởi tạo, các sổ ghi nhớ $m, v$ của AdamW còn rỗng và thiếu chính xác; bước lớn ngay lập tức dễ làm mô hình bất ổn.
2. _Cosine decay:_ sau warmup, giảm $lr$ theo hình _cos_ mượt về gần 0 tới cuối huấn luyện, để những bước cuối tinh chỉnh nhẹ nhàng.

```text
lr
^          _______
|        /         \___
|      /               \____
|    /                      \____
|  / warmup   cosine decay        \___
+--------------------------------------> step
  0   warmup_iters              max_iters
```

Công thức (tự cài, dễ hiểu, khớp nanoGPT):

```python
import math

def get_lr(it, warmup_iters=200, max_iters=5000, max_lr=3e-4, min_lr=3e-5):
    # 1) linear warmup for the first warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    # 2) after training ends, hold at min_lr
    if it > max_iters:
        return min_lr
    # 3) cosine decay from max_lr down to min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters) # in [0, 1]
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))          # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)

# Ví dụ giá trị lr theo bước:
# get_lr(0)     -> ~1.5e-6   (mới bắt đầu warmup)
# get_lr(200)   -> 3e-4      (đỉnh, hết warmup)
# get_lr(2600)  -> ~1.65e-4  (giữa chặng cosine)
# get_lr(5000)  -> 3e-5      (min_lr, cuối)
```

Cách nhét $lr$ này vào optimizer trước mỗi bước:

```python
lr = get_lr(it)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr # ghi đè lr thủ công cho tất cả nhóm tham số
```

💡 PyTorch cũng có sẵn `torch.optim.lr_scheduler.CosineAnnealingLR` và `LinearLR` (ghép qua `SequentialLR`), nhưng tự viết `get_lr` như trên minh bạch và dễ debug hơn cho khóa này.

---

### 5. Gradient clipping: dây an toàn chống bùng nổ

**Analogy:** đôi khi địa hình có vách đá dốc đứng $\to$ gradient đột nhiên _cực lớn_ $\to$ một bước AdamW cũng đủ đẩy trọng số văng đi rất xa, phá hỏng mô hình (loss nhảy lên `NaN`). _Gradient clipping_ giống cái _dây bảo hiểm_: nếu tổng độ lớn (norm) của gradient vượt ngưỡng, ta _co lại_ cho vừa ngưỡng, giữ nguyên _hướng_.

Công thức clip theo chuẩn (norm) toàn cục với ngưỡng $c$:

```text
total_norm = sqrt( sum_i g_i^2 )           # gộp toàn bộ gradient
if total_norm > c:
    g_i <- g_i * (c / total_norm)          # scale ĐỀU, giữ nguyên hướng
```

⚠️ Clip đặt _sau_ `loss.backward()` (đã có `.grad`) và _trước_ `optimizer.step()`. Đặt sai chỗ thì clip vô nghĩa.

---

### 6. Ghép tất cả vào vòng lặp huấn luyện (đúng thứ tự)

```python
import torch

max_grad_norm = 1.0 # ngưỡng clip điển hình cho GPT

for it in range(max_iters):
    # 0) set learning rate for this step (warmup + cosine)
    lr = get_lr(it)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    xb, yb = get_batch('train') # xb, yb shape: (B, T)
    logits = model(xb)          # logits shape: (B, T, vocab_size)
    B, T, V = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.view(B * T, V),  # (B*T, V) - mỗi vị trí là 1 bài phân loại
        yb.view(B * T),         # (B*T,)  - token đúng kế tiếp
    )

    optimizer.zero_grad(set_to_none=True) # xóa gradient cũ
    loss.backward()                       # tính .grad cho mọi tham số

    # clip AFTER backward, BEFORE step -> tránh gradient bùng nổ
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step() # cập nhật trọng số

    if it % 100 == 0:
        print(f"iter {it}: loss {loss.item():.4f}, lr {lr:.2e}, gnorm {grad_norm:.2f}")

# Output mẫu (số minh họa):
# iter 0:   loss 4.1738, lr 1.50e-06, gnorm 1.00
# iter 100: loss 2.6402, lr 1.50e-04, gnorm 0.87
# iter 200: loss 2.4915, lr 3.00e-04, gnorm 0.74
# iter 500: loss 2.2130, lr 2.94e-04, gnorm 0.51
```

Thứ tự “vàng” mỗi bước: _set lr $\to$ forward $\to$ loss $\to$ zero_grad $\to$ backward $\to$ clip $\to$ step_.

---

### Nhìn sâu hơn

- **AdamW tốn bộ nhớ:** vì phải lưu $m$ và $v$ cho _mỗi_ tham số, optimizer state ngốn thêm ~2x kích thước mô hình. Một mô hình 100M tham số fp32 (~400 MB) có thể cần thêm ~800 MB chỉ cho optimizer state - đây là lý do lớn khiến huấn luyện ngốn VRAM hơn inference (nối lại tính toán VRAM ở Bài 2).
- **Không decay lên bias và LayerNorm:** thực hành chuẩn (nanoGPT) là chỉ áp weight decay cho ma trận trọng số 2D (linear, embedding), còn bias và tham số của LayerNorm thì _không_ decay. Ta chia tham số thành hai nhóm `param_groups` với `weight_decay` khác nhau.
- **grad_norm là tín hiệu chẩn đoán:** in `grad_norm` ra. Nếu nó thường xuyên đụng trần (bị clip liên tục) $\to$ lr có thể quá cao hoặc dữ liệu có vấn đề. Nếu nó tăng vọt bất thường $\to$ sắp có sự cố phân kỳ.
- **Vì sao $\beta_2=0.95$:** loss của mô hình ngôn ngữ rất ồn giữa các batch; $\beta_2$ thấp hơn giúp ước lượng $v$ bám nhanh hơn vào thay đổi gần đây, ổn định hơn cho LLM so với 0.999 mặc định.

---

### Thực hành

1. **[Dễ]** Gọi `get_lr` ở trên với `it = 0, 100, 200, 2600, 5000`. In các giá trị và tự kiểm: lr có tăng tuyến tính tới bước 200 rồi giảm mượt về min_lr không?
2. **[Trung bình]** Chia tham số mô hình thành 2 nhóm: nhóm ma trận 2D (`p.dim() >= 2`) có `weight_decay=0.1`, nhóm còn lại (bias, LayerNorm) có `weight_decay=0.0`. Tạo AdamW với hai `param_groups` này rồi so sánh loss với lúc decay đồng đều.
3. **[Thử thách]** Đặt `max_lr=3e-2` (cao gấp 100 lần) và _tắt_ gradient clipping, chạy vài chục bước - quan sát loss bùng nổ/`NaN`. Sau đó bật lại clipping với `max_grad_norm=1.0` và giảm lr - xác nhận huấn luyện ổn định trở lại.

---

### Hiểu lầm thường gặp

- ❌ “Learning rate cứ để cố định là được.” ➔ ✅ Warmup + cosine decay giúp mô hình khởi động ổn định rồi hội tụ tốt hơn hẳn lr phẳng.
- ❌ “AdamW và Adam giống nhau, chỉ đổi tên.” ➔ ✅ AdamW _tách rời_ weight decay khỏi bước adaptive - khác biệt thật, ảnh hưởng rõ tới generalization.
- ❌ “Gradient clipping làm chậm huấn luyện nên bỏ đi.” ➔ ✅ Clipping hầu như chỉ kích hoạt khi gradient bất thường; nó là dây an toàn gần như miễn phí, chống `NaN`.
- ❌ “lr càng lớn thì học càng nhanh.” ➔ ✅ lr quá lớn khiến loss phân kỳ; có một khoảng “vừa đủ” cần dò tìm.
- ❌ “Weight decay nên áp cho mọi tham số kể cả LayerNorm và bias.” ➔ ✅ Thực hành chuẩn là _không_ decay bias và LayerNorm.

---

### Mẹo ghi nhớ

- _AdamW_ = Adam + *W*eight decay tách rời (chữ W nhắc “decoupled weight decay”).
- Thứ tự bước: *z*ero_grad $\to$ *b*ackward $\to$ *c*lip $\to$ *s*tep (nhớ “zbcs”).
- Warmup _lên_, cosine _xuống_ - hình quả đồi nghiêng dốc rồi thoải dần.
- Bộ tham số kinh điển GPT: `lr=3e-4`, `betas=(0.9, 0.95)`, `wd=0.1`, `clip=1.0`.

---

### Tóm tắt trực quan

| Khái niệm             | Một dòng                                        |
| :-------------------- | :---------------------------------------------- |
| **Optimizer**         | Quy tắc cập nhật trọng số từ gradient           |
| **AdamW**             | Adam + weight decay tách rời - chuẩn cho GPT    |
| **Learning rate**     | Độ dài mỗi bước; siêu tham số quan trọng nhất   |
| **Warmup**            | Tăng lr tuyến tính lúc đầu để khởi động ổn định |
| **Cosine decay**      | Giảm lr mượt về gần 0 để hội tụ tinh            |
| **Gradient clipping** | Co gradient về ngưỡng khi quá lớn, giữ hướng    |
| **grad_norm**         | Tín hiệu chẩn đoán sức khỏe huấn luyện          |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [AdamW Optimizer from Scratch in Python](https://www.youtube.com/watch?v=0vgdf_Uhu08)
- AdamW trong PyTorch: [torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- `clip_grad_norm_`: [torch.nn.utils.clip*grad_norm*](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- LR scheduler: [PyTorch Optimization](https://pytorch.org/docs/stable/optim.html)
- [nanoGPT](https://github.com/karpathy/nanoGPT) (cấu hình optimizer + scheduler thực tế của Karpathy)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (mục học tập về warmup schedule)

---

### Tổng kết

- _AdamW_ là optimizer chuẩn cho Transformer: learning rate thích nghi từng tham số + weight decay tách rời.
- _Learning rate_ là siêu tham số quan trọng nhất; kết hợp _warmup + cosine decay_ để vừa khởi động ổn định vừa hội tụ tinh.
- _Gradient clipping_ đặt sau backward và trước step là dây an toàn chống bùng nổ gradient.
- Thứ tự vàng mỗi bước: set lr $\to$ forward $\to$ loss $\to$ zero_grad $\to$ backward $\to$ clip $\to$ step.

Bây giờ mô hình của bạn đã học _ổn định và đúng cách_. Nhưng làm sao biết nó đang học tốt hay đang _học vẹt_ (nhớ dữ liệu huấn luyện mà không tổng quát hóa)? Ở _Bài 28: Theo dõi loss, overfitting và regularization_, mình sẽ theo dõi loss train/validation, phát hiện overfitting, và dùng các kỹ thuật regularization (bao gồm chính weight decay và dropout) để mô hình học _cho ra hồn_. Hẹn gặp ở bài kế tiếp! 🤖
