# Bài 25: Khởi tạo trọng số và ổn định huấn luyện

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 5:** Xây dựng GPT mini

---

### Giới thiệu

Ở _Bài 24_ bạn đã học cách đếm tham số và hiểu kích thước mô hình: một GPT mini với vài triệu tham số là những _ma trận số_ khổng lồ trong `nn.Linear`, `nn.Embedding`, `LayerNorm`… Nhưng có một câu hỏi rất tự nhiên mà chúng ta chưa trả lời: _những con số đó BAN ĐẦU bằng bao nhiêu?_ Trước khi mô hình học được điều gì, ai đó phải điền giá trị khởi đầu vào từng ô của các ma trận đó. Đó chính là _khởi tạo trọng số_ (weight initialization).

Hãy tưởng tượng bạn đổ nước vào một mê cung dốc để nó chảy xuống đáy (điểm loss thấp nhất). Nếu bạn đổ nước từ một vị trí _quá cao và dốc đứng_, nó văng tung tóe (loss nổ - `NaN`). Nếu đổ vào một _mặt phẳng lì_ không có độ dốc, nước đứng yên (mô hình không học được gì). Khởi tạo tốt = chọn điểm xuất phát hợp lý để dòng nước (gradient) chảy đều, ổn định về đáy. Cùng với _dropout_ (một dạng “bịt mắt ngẫu nhiên” để mô hình không học vẹt), đây là hai công cụ giúp huấn luyện _hội tụ ổn định_.

> 🔑 **Một câu để nhớ:** Khởi tạo trọng số quyết định điểm xuất phát của việc học; nếu phương sai tín hiệu bùng nổ hoặc tắt dần qua các tầng, gradient sẽ nổ hoặc biến mất và mô hình không hội tụ.

---

### Bạn sẽ học được gì?

- [ ] Vì sao _không thể_ khởi tạo trọng số bằng 0 hết (đối xứng), và vì sao khởi tạo ngẫu nhiên là bắt buộc.
- [ ] Cách khởi tạo hợp lý: phân phối chuẩn std nhỏ, và các sơ đồ Xavier/Kaiming là gì.
- [ ] Mẹo khởi tạo đặc trưng của GPT (nanoGPT): scale residual theo `1/sqrt(2*n_layer)`.
- [ ] _Dropout_ hoạt động thế nào, đặt ở đâu trong Transformer block, khác nhau khi train/eval.
- [ ] Vì sao khởi tạo + chuẩn hóa (LayerNorm) ảnh hưởng sự hội tụ; cách phát hiện gradient nổ/biến mất.

📌 **Yêu cầu trước:** đã biết cấu trúc Transformer block (attention + feed-forward + residual + LayerNorm), biết `nn.Linear` / `nn.Embedding` (Bài 24), và cross-entropy cho next-token.

---

### 1. Vì sao KHÔNG khởi tạo bằng 0 - vấn đề đối xứng

**Analogy:** nếu tất cả học sinh trong lớp bắt đầu với _cùng một suy nghĩ y hệt_ và nhận _cùng một bài giảng_, chúng sẽ luôn cập nhật giống hệt nhau - cả lớp mãi mãi là một người. Với mạng nơ-ron: nếu mọi trọng số của một tầng bằng 0 (hoặc cùng một hằng số), mọi nơ-ron nhận cùng input, cho cùng output, cùng gradient → chúng cập nhật y hệt nhau và _không bao giờ tách ra học đặc trưng khác nhau_. Đây là _phá vỡ đối xứng_ (symmetry breaking): ta PHẢI khởi tạo ngẫu nhiên.

```python
import torch
import torch.nn as nn

# BAD: all-zero weights -> symmetry, neurons never differentiate
linear_zero = nn.Linear(4, 3)
nn.init.zeros_(linear_zero.weight) # weight shape: (3, 4)
# every output neuron computes the same thing -> gradients identical

# GOOD: small random normal breaks symmetry
linear_rand = nn.Linear(4, 3)
nn.init.normal_(linear_rand.weight, mean=0.0, std=0.02) # (3, 4)
nn.init.zeros_(linear_rand.bias) # bias CAN be zero; weight cannot
print(linear_rand.weight.std().item()) # ~0.02
```

💡 **Lưu ý:** bias khởi tạo 0 là hoàn toàn ổn - đối xứng bị phá bởi weight ngẫu nhiên rồi. Chỉ có weight mới cần ngẫu nhiên.

---

### 2. "Hợp lý" nghĩa là gì - giữ phương sai tín hiệu qua các tầng

**Analogy dây chuyền lắp ráp:** tín hiệu chạy qua nhiều tầng như sản phẩm qua nhiều trạm. Nếu mỗi trạm _phóng đại_ một chút, đến cuối chuyền sản phẩm to đùng (activation nổ → loss `NaN`). Nếu mỗi trạm _thu nhỏ_ một chút, đến cuối chỉ còn hạt bụi (activation $\approx$ 0 → gradient biến mất). Mục tiêu khởi tạo: _giữ phương sai (variance) của tín hiệu xấp xỉ ổn định_ khi đi qua từng tầng, cả chiều tiến (forward) lẫn chiều lùi (backward).

Với một tầng Linear có `fan_in` đầu vào, nếu trọng số có phương sai $\text{Var}(w)$, thì phương sai $\text{output} \approx \text{fan\_in} \times \text{Var}(w) \times \text{Var}(x)$. Muốn giữ nguyên phương sai $\implies$ chọn $\text{Var}(w) \approx 1/\text{fan\_in}$.

| Sơ đồ             | Công thức std                                   | Dùng cho activation                |
| :---------------- | :---------------------------------------------- | :--------------------------------- |
| **Xavier/Glorot** | $\sqrt{2 / (\text{fan\_in} + \text{fan\_out})}$ | tanh, sigmoid (đối xứng)           |
| **Kaiming/He**    | $\sqrt{2 / \text{fan\_in}}$                     | ReLU/GELU (một nửa bị cắt)         |
| **GPT (nanoGPT)** | $\text{std} = 0.02$ cố định                     | Linear/Embedding trong Transformer |

```python
# Kaiming/He init for a ReLU/GELU layer
lin = nn.Linear(512, 512)
nn.init.kaiming_normal_(lin.weight, mode='fan_in', nonlinearity='relu') # (512, 512)

# GPT-style: fixed small std=0.02 (empirically stable for Transformers)
def init_gpt_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02) # weight: (out, in)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02) # (vocab, n_embd)
```

🎯 GPT chọn $\text{std}=0.02$ cố định thay vì tính theo `fan_in` vì thực nghiệm cho thấy nó ổn định cho kiến trúc Transformer, và LayerNorm sẽ chuẩn hóa lại tín hiệu nên yêu cầu về khởi tạo bớt khắt khe hơn.

---

### 3. Mẹo khởi tạo residual của GPT - scale theo số tầng

Mỗi Transformer block CỘNG dồn vào luồng residual: $x = x + \text{attention}(x)$ rồi $x = x + \text{ffn}(x)$. Qua `n_layer` block, tổng phương sai tích tụ tuyến tính theo số lần cộng. Analogy: mỗi tầng góp thêm một chút “ồn”; cộng $2 \times \text{n\_layer}$ lần thì độ lớn tín hiệu phình ra. nanoGPT khắc phục bằng cách _chia nhỏ khởi tạo của lớp chiếu ra (projection) trong residual_ theo $1/\sqrt{2 \times \text{n\_layer}}$:

```python
import math

# Apply after building the model: scale residual-projection weights
for name, p in model.named_parameters():
    # projections that write back into the residual stream
    if name.endswith('c_proj.weight'):
        std = 0.02 / math.sqrt(2 * n_layer) # e.g. n_layer=6 -> 0.02/sqrt(12) ~= 0.00577
        torch.nn.init.normal_(p, mean=0.0, std=std)
```

```text
Residual stream growth (why scale 1/sqrt(2*n_layer)):

  x0 -> [+attn] -> [+ffn] -> [+attn] -> [+ffn] -> ... (n_layer blocks)
         2 adds per block  x  n_layer blocks  =  2*n_layer adds
  Var accumulates ~ linearly -> shrink each proj std by 1/sqrt(2*n_layer)
```

💡 Đây là chi tiết “nhỏ mà có võ”: nhờ nó mô hình sâu (nhiều tầng) không bị hoạt hóa phình dần, giúp huấn luyện ổn định hơn nhiều.

---

### 4. Dropout - chống học vẹt (overfitting)

**Analogy:** khi ôn thi nhóm, nếu mỗi buổi bạn _cố tình cho vắng ngẫu nhiên vài bạn_, cả nhóm buộc phải học đều để không phụ thuộc vào một “cây đa cây đề” duy nhất. `Dropout(p)` làm đúng vậy: trong lúc _huấn luyện_, mỗi nơ-ron bị _tắt ngẫu nhiên với xác suất $p$_ (đặt về 0), buộc mạng không dựa dẫm vào một vài nơ-ron. Các nơ-ron còn lại được _scale lên_ $1/(1-p)$ để giữ kỳ vọng đầu ra không đổi (inverted dropout).

```python
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.ones(1, 8) # shape: (1, 8)
drop = nn.Dropout(p=0.5)

drop.train() # TRAIN mode: randomly zeroes ~half, scales rest by 1/(1-0.5)=2
print(drop(x)) # e.g. tensor([[2., 0., 2., 2., 0., 0., 2., 0.]])

drop.eval()  # EVAL mode: dropout is a NO-OP (identity)
print(drop(x)) # tensor([[1., 1., 1., 1., 1., 1., 1., 1.]])
```

⚠️ **Cực kỳ quan trọng:** dropout chỉ hoạt động ở `model.train()`. Ở `model.eval()` (lúc validate/generate) nó _tắt hẳn_. Quên gọi `model.eval()` khi sinh văn bản → kết quả nhiễu ngẫu nhiên khó lường. Đây là lỗi kinh điển.

Trong Transformer block, dropout thường đặt ở ba chỗ: sau attention weights (attn dropout), sau lớp chiếu ra của attention, và sau feed-forward.

```text
Transformer block với dropout + pre-norm:

  x ---> LayerNorm ---> Multi-Head Attention ---> Dropout ---+
  |                                                          |
  +-------------------- (residual +) ------------------------+---> x'
  x'--> LayerNorm ---> FeedForward(GELU) ---> Dropout -------+
  |                                                          |
  +-------------------- (residual +) ------------------------+---> out
```

---

### 5. LayerNorm - người giữ thăng bằng cho tín hiệu

**Analogy:** LayerNorm như một người điều phối chuẩn hóa lại điểm số về cùng thang (trung bình 0, độ lệch 1) trước mỗi trạm, để không trạm nào nhận input lệch lạc quá mức. Nó chuẩn hóa _trên chiều đặc trưng_ của từng token, rồi cho phép scale/shift học được (`gamma`, `beta`). Nhờ LayerNorm mà việc khởi tạo bớt nhạy cảm: dù tín hiệu lệch một chút, nó được kéo về thang chuẩn ngay đầu mỗi block (_pre-norm_, cách GPT dùng).

```python
ln = nn.LayerNorm(64) # normalize over last dim of size 64
h = torch.randn(2, 10, 64) # shape: (batch=2, seq=10, n_embd=64)
out = ln(h) # (2, 10, 64), mean~0 & std~1 along last dim
print(out.mean().item(), out.std().item()) # ~0.0, ~1.0
```

🎯 **Kết hợp:** khởi tạo $\text{std}=0.02$ + scale residual + LayerNorm pre-norm = công thức ổn định của GPT mini. Ba thứ này bổ trợ nhau chứ không thay thế nhau.

---

### Nhìn sâu hơn

- **Gradient nổ vs biến mất.** Nếu khởi tạo quá lớn hoặc thiếu chuẩn hóa, giá trị hoạt hóa phình theo cấp số nhân qua tầng → gradient nổ → loss thành `inf` / `NaN`. Ngược lại, khởi tạo quá nhỏ → tín hiệu tắt dần → gradient $\approx$ 0 → mô hình “đứng hình”. Ngoài khởi tạo, ta còn dùng _gradient clipping_ (`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`) như một cái phanh: nếu chuẩn (norm) của toàn bộ gradient vượt ngưỡng thì co lại, chống nổ trong các bước đầu.
- **Vì sao AdamW hợp với Transformer.** AdamW (Adam + weight decay tách rời) tự điều chỉnh bước học riêng cho từng tham số dựa trên ước lượng moment bậc 1 và 2, nên bớt nhạy với khởi tạo hơn SGD thuần. Thường kèm _warmup_ (tăng learning rate từ 0 lên đỉnh trong vài trăm bước đầu) rồi _cosine decay_ - giai đoạn warmup đặc biệt quan trọng vì các bước đầu mô hình còn “non”, dễ nổ nếu learning rate cao ngay.
- **Chi phí:** khởi tạo gần như miễn phí (chạy một lần lúc tạo model), dropout không thêm tham số (chỉ thêm phép nhân mask lúc train), LayerNorm thêm $2 \times \text{n\_embd}$ tham số mỗi lớp - rất rẻ so với lợi ích ổn định.

---

### Thực hành

- **[Dễ]** Tạo `nn.Linear(256, 256)`, khởi tạo weight bằng `nn.init.normal_(std=0.02)`, in ra `weight.std()` để xác nhận $\approx 0.02$. Sau đó thử $\text{std}=1.0$ và tự lý giải vì sao dễ gây nổ hơn.
- **[Trung bình]** Viết hàm `init_gpt_weights(module)` như mục 2, rồi gọi `model.apply(init_gpt_weights)` trên GPT mini của bạn. Thêm vòng lặp scale các `c_proj.weight` theo $0.02/\sqrt{2 \times \text{n\_layer}}$. In `named_parameters()` để kiểm tra std của lớp projection nhỏ hơn các lớp khác.
- **[Thử thách]** Thí nghiệm dropout: huấn luyện GPT mini với `dropout=0.0` và `dropout=0.2` trên cùng dữ liệu nhỏ, vẽ đường train-loss và val-loss. Quan sát: `dropout=0.0` thường có train-loss thấp hơn nhưng val-loss cao hơn (dấu hiệu overfitting). Thêm `clip_grad_norm_(..., 1.0)` và thử bỏ warmup để thấy vài bước đầu loss dễ nổ hơn.

---

### Hiểu lầm thường gặp

- ❌ “Khởi tạo tất cả bằng 0 cho gọn.” ➔ ✅ Trọng số 0 gây _đối xứng_, các nơ-ron cập nhật y hệt nhau, không học được. Chỉ bias mới được phép 0.
- ❌ “Khởi tạo càng lớn thì học càng nhanh.” ➔ ✅ Khởi tạo quá lớn làm hoạt hóa/gradient _nổ_ (`NaN`). Cần std nhỏ (vd 0.02) để giữ phương sai ổn định.
- ❌ “Dropout luôn bật, kể cả khi sinh văn bản.” ➔ ✅ Dropout chỉ bật ở `model.train()`; khi `model.eval()` nó là phép đồng nhất (no-op). Quên `eval()` → kết quả nhiễu.
- ❌ “Có LayerNorm rồi thì khởi tạo thế nào cũng được.” ➔ ✅ LayerNorm giúp bớt nhạy, nhưng khởi tạo quá tệ (vd $\text{std}=5$) vẫn có thể làm bước đầu nổ trước khi ổn định.
- ❌ “Dropout xóa nơ-ron vĩnh viễn.” ➔ ✅ Mỗi bước train tắt một tập _ngẫu nhiên KHÁC nhau_; không nơ-ron nào bị bỏ hẳn.

---

### Mẹo ghi nhớ

- _0 = đối xứng = chết_: nhớ luôn khởi tạo weight ngẫu nhiên, bias mới được 0.
- _std nhỏ, phương sai ổn_: GPT dùng 0.02; residual proj chia thêm $1/\sqrt{2 \times \text{n\_layer}}$.
- _train() bật dropout, eval() tắt dropout_ - khắc cốt ghi tâm khi generate.
- Mẹo: “_Đổ nước đúng chỗ_ (init) - _giữ thăng bằng_ (LayerNorm) - _cho vắng ngẫu nhiên_ (dropout) - _có phanh_ (grad clip)”.

---

### Tóm tắt trực quan

| Khái niệm             | Một dòng                                                           |
| :-------------------- | :----------------------------------------------------------------- |
| **Phá vỡ đối xứng**   | Khởi tạo ngẫu nhiên để nơ-ron học đặc trưng khác nhau              |
| **std=0.02**          | Khởi tạo chuẩn của GPT cho Linear/Embedding                        |
| **Xavier vs Kaiming** | Giữ phương sai; Kaiming cho ReLU/GELU                              |
| **Scale residual**    | Chia proj weight cho $\sqrt{2 \times \text{n\_layer}}$ chống phình |
| **Dropout(p)**        | Tắt ngẫu nhiên p nơ-ron lúc train, no-op lúc eval                  |
| **LayerNorm**         | Chuẩn hóa tín hiệu về mean~0/std~1 mỗi block                       |
| **Grad clipping**     | Phanh chống gradient nổ ở bước đầu                                 |
| **AdamW + warmup**    | Tối ưu ổn định, ít nhạy khởi tạo                                   |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Coding a Transformer from scratch on PyTorch, with full explanation, training and inference.](https://www.youtube.com/watch?v=ISNdQcPhsts)
- Khởi tạo trong PyTorch: [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html)
- Dropout & LayerNorm docs: [PyTorch Documentation](https://pytorch.org/docs/stable/)
- nanoGPT (xem `model.py` để thấy `_init_weights` và scale `c_proj`): [nanoGPT on GitHub](https://github.com/karpathy/nanoGPT)
- Attention Is All You Need (kiến trúc Transformer gốc): [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer (trực quan): [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

### Tổng kết

- Khởi tạo weight _ngẫu nhiên_ để phá đối xứng; bias được phép 0.
- Chọn std nhỏ (GPT: 0.02) để giữ phương sai tín hiệu ổn định qua các tầng; Kaiming/He cho ReLU/GELU.
- Scale lớp projection trong residual theo $1/\sqrt{2 \times \text{n\_layer}}$ để mô hình sâu không phình hoạt hóa.
- _Dropout_ chống overfitting, chỉ bật ở `train()`, tắt ở `eval()`; _LayerNorm_ giữ thăng bằng; _grad clipping_ + _AdamW/warmup_ giúp hội tụ ổn định.

Đã có điểm xuất phát tốt và các “trợ thủ ổn định”, giờ là lúc lắp tất cả vào một guồng máy chạy thật. Ở _Bài 26: Vòng lặp huấn luyện đầy đủ_, chúng ta sẽ ghép data loader, forward, cross-entropy, backward, `optimizer.step()`, đánh giá val-loss và checkpoint thành một training loop hoàn chỉnh cho GPT mini. Hẹn gặp ở bài kế tiếp! 🤖
