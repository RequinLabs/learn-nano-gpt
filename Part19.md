# Bài 19: Ghép thành một Transformer block hoàn chỉnh

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 4:** Transformer block

---

### Giới thiệu

Hãy tưởng tượng một dây chuyền lắp ráp ô tô. Mỗi trạm làm _một_ việc: trạm gắn động cơ, trạm lắp bánh, trạm sơn… Chiếc xe đi qua từng trạm và _hoàn thiện dần_. Nếu một trạm lỡ làm hỏng, người ta không vứt cả xe đi mà chỉ _cộng thêm phần sửa_ vào cái đã có sẵn - đó chính là tinh thần của residual connection.

Ở Bài 16-17 bạn đã có _multi-head attention_ (nhiều góc nhìn song song) và _feed-forward_ (xử lý riêng từng token). Ở Bài 18 bạn học hai “chất keo” giữ mạng sâu ổn định: _residual connection_ ($x + f(x)$) và _Layer Normalization_. Bài này là lúc _vặn tất cả lại thành một cụm_: một _Transformer block_ hoàn chỉnh. Rồi ta _xếp chồng_ nhiều block lên nhau - đúng cách GPT thật sự được xây.

> 🔑 **Một câu để nhớ:** Một Transformer block = (attention để token _giao tiếp_) rồi (feed-forward để token _suy nghĩ riêng_), mỗi phần bọc trong $x + \text{sublayer}(\text{LayerNorm}(x))$; xếp chồng $N$ block chính là “chiều sâu” của mô hình.

---

### Bạn sẽ học được gì?

- [ ] Ghép multi-head attention + feed-forward thành _một_ `nn.Module` gọi là `Block`.
- [ ] Hiểu _pre-norm_ ($x + \text{sublayer}(\text{LayerNorm}(x))$) khác _post-norm_ ra sao và vì sao GPT chọn pre-norm.
- [ ] Vì sao có _hai_ kết nối residual và _hai_ LayerNorm trong một block.
- [ ] Theo dõi _shape tensor_ đi xuyên block - đầu ra luôn cùng shape với đầu vào $(B, T, C)$.
- [ ] Xếp chồng nhiều block bằng `nn.Sequential` và hiểu “sâu hơn” nghĩa là gì.

📌 **Yêu cầu trước:** Bài 16 (multi-head attention), Bài 17 (feed-forward), Bài 18 (residual + LayerNorm). Biết `nn.Module`, `nn.LayerNorm`, `nn.Sequential` và tư duy shape $(B, T, C)$.

---

### 1. Bức tranh lớn: một block làm hai việc

**Analogy:** coi một block như _một vòng thảo luận nhóm_. Đầu tiên cả nhóm _trao đổi_ - ai cũng nghe ý người khác để cập nhật hiểu biết của mình (đó là _attention_: token giao tiếp với token). Sau khi nghe xong, _mỗi người tự ngồi ngẫm_ và rút ra kết luận riêng (đó là _feed-forward_: xử lý độc lập từng token). Một block = một vòng “trao đổi rồi ngẫm”.

Nhắc lại điểm dễ lẫn: _attention_ cho các token _trộn thông tin với nhau_ (giao tiếp giữa các vị trí), còn _feed-forward_ biến đổi _mỗi token độc lập_ (không nhìn token khác). Một block cần _cả hai_ thì mới vừa “biết ngữ cảnh” vừa “biến đổi phi tuyến” đủ mạnh.

Đầu vào và đầu ra của block đều có shape $(B, T, C)$ - _không đổi shape_. Chính vì giữ nguyên shape mà ta có thể xếp chồng block này lên block kia thoải mái, giống chồng các viên gạch cùng kích thước.

```text
x  (B, T, C) --> [ Transformer Block ] --> y  (B, T, C)   # cung shape!
```

---

### 2. Vì sao cần residual + LayerNorm quanh mỗi sublayer?

Nhắc lại Bài 18: khi mạng _sâu_ (nhiều block), gradient dễ bị “biến mất” hoặc “bùng nổ” khi lan ngược. Hai công cụ cứu cánh:

- **Residual connection:** thay vì $x = f(x)$, ta viết $x = x + f(x)$. Đường “cộng thẳng” ($+ x$) tạo một _đường cao tốc cho gradient_ chạy ngược không suy giảm. Mỗi sublayer chỉ cần học phần _thay đổi nhỏ_ (residual) so với đầu vào.
- **LayerNorm:** chuẩn hóa mỗi vector token về trung bình 0, phương sai 1 (theo chiều $C$), giữ cho phân phối đầu vào của mỗi sublayer ổn định qua các block.

Có _hai_ thứ cần bọc (attention và feed-forward) $\to$ nên có _hai_ cặp (residual + LayerNorm) trong một block.

💡 Đừng nhầm: LayerNorm chuẩn hóa theo _từng token_ (trên chiều đặc trưng $C$), khác BatchNorm chuẩn hóa theo _cả batch_. Với chuỗi ngôn ngữ độ dài thay đổi, LayerNorm ổn định hơn nhiều.

---

### 3. Pre-norm và post-norm - thứ tự đặt LayerNorm

Có hai cách gắn LayerNorm, khác nhau ở _chỗ đặt norm_:

```text
POST-NORM (Transformer goc 2017):   x = LayerNorm( x + Sublayer(x) )
PRE-NORM  (GPT-2 tro di, nanoGPT):   x = x + Sublayer( LayerNorm(x) )
```

Điểm mấu chốt của _pre-norm_: đường residual $+ x$ là một đường _sạch, không bị chen ngang bởi LayerNorm_ - gradient chạy thẳng từ đầu ra tới đầu vào. Nhờ vậy mạng rất sâu vẫn huấn luyện ổn định mà không cần “learning rate warmup” cầu kỳ. Đây là lý do GPT-2, nanoGPT và hầu hết LLM hiện đại dùng _pre-norm_.

| Tiêu chí                   | Post-norm (gốc)              | Pre-norm (GPT-2, nanoGPT)    |
| :------------------------- | :--------------------------- | :--------------------------- |
| **Công thức**              | `LayerNorm(x + Sublayer(x))` | `x + Sublayer(LayerNorm(x))` |
| **Đường residual**         | Bị LayerNorm “chặn”          | Sạch, gradient chạy thẳng    |
| **Độ ổn định khi rất sâu** | Kém hơn, cần warmup kỹ       | Ổn định hơn                  |
| **Ai dùng**                | Transformer gốc 2017         | GPT-2, GPT-3, nanoGPT…       |

🎯 Trong khóa này ta theo _pre-norm_ cho khớp nanoGPT: $x = x + \text{attention}(\text{LayerNorm}(x))$, rồi $x = x + \text{feedforward}(\text{LayerNorm}(x))$.

---

### 4. Sơ đồ cấu trúc một Transformer block

Đây là bản đồ luồng dữ liệu của một block _pre-norm_ (dấu `(+)` là phép cộng residual):

```text
x  (B, T, C)
        |------------------------------.        # residual duong 1 (giu ban sao cua x)
        v                              |
     LayerNorm (ln1)                   |
        v                              |
  Multi-Head Attention                 |        # token giao tiep voi token
        v                              |
       (+) <---------------------------'        # x = x + attn(ln1(x))
        |
        |------------------------------.        # residual duong 2
        v                              |
     LayerNorm (ln2)                   |
        v                              |
    Feed-Forward (MLP)                 |        # moi token suy nghi rieng
        v                              |
       (+) <---------------------------'        # x = x + ffwd(ln2(x))
        |
        v
        y  (B, T, C)                            # cung shape voi dau vao
```

Chú ý: có đúng _2 LayerNorm_ (`ln1`, `ln2`) và _2 phép cộng residual_. Shape đi vào và đi ra hoàn toàn giống nhau $(B, T, C)$.

---

### 5. Cài đặt module Block

Giả sử ở các bài trước bạn đã có `MultiHeadAttention` và `FeedForward` (mình nhắc lại ngắn để code tự chứa). Điểm nhấn nằm ở lớp `Block`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    # Position-wise MLP: expand by 4x then project back (per-token, independent).
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # (B, T, C) -> (B, T, 4C)
            nn.GELU(),                     # nonlinearity used by GPT
            nn.Linear(4 * n_embd, n_embd), # (B, T, 4C) -> (B, T, C)
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x) # (B, T, C) -> (B, T, C)

class Block(nn.Module):
    # One Transformer block: pre-norm attention + pre-norm feed-forward,
    # each wrapped in a residual connection.
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head # so heads together give back n_embd
        # Reuse the multi-head attention module from a previous lesson.
        self.sa   = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd) # norm BEFORE attention (pre-norm)
        self.ln2  = nn.LayerNorm(n_embd) # norm BEFORE feed-forward

    def forward(self, x):
        # x: (B, T, C)
        x = x + self.sa(self.ln1(x))   # residual 1: x + attn(LN(x))
        x = x + self.ffwd(self.ln2(x)) # residual 2: x + ffwd(LN(x))
        return x # (B, T, C) - same shape as input
```

Đọc kỹ `forward`: chỉ _hai dòng_, nhưng chứa trọn triết lý pre-norm. Dòng 1 là “trao đổi” (attention), dòng 2 là “ngẫm riêng” (feed-forward). Cả hai đều theo mẫu $x = x + \text{sublayer}(\text{LayerNorm}(x))$.

⚠️ Thứ tự _cực kỳ quan trọng_: phải là `self.sa(self.ln1(x))` chứ _không_ phải `self.ln1(self.sa(x))`. Đặt sai chỗ LayerNorm là bạn đã đổi từ pre-norm sang post-norm - mô hình vẫn chạy nhưng khó huấn luyện hơn khi rất sâu.

---

### 6. Xếp chồng nhiều block - "chiều sâu" của mô hình

Vì block giữ nguyên shape $(B, T, C)$, ta chồng $N$ block bằng `nn.Sequential`:

```python
n_embd, n_head, block_size, n_layer = 64, 4, 32, 6

blocks = nn.Sequential(*[
    Block(n_embd, n_head, block_size, dropout=0.1) for _ in range(n_layer) # stack n_layer identical blocks
])

# Fake batch of token embeddings.
B, T = 2, 16
x = torch.randn(B, T, n_embd) # (2, 16, 64)
y = blocks(x)
print(y.shape) # -> torch.Size([2, 16, 64])  (unchanged)
```

Mỗi block _cùng kiến trúc_ nhưng _tham số riêng_ (không chia sẻ trọng số). Càng nhiều block, mô hình càng “sâu” và có thể học các quan hệ trừu tượng hơn - đây chính là ý nghĩa của `n_layer` (GPT-2 nhỏ có 12 block, các bản lớn hơn có 24, 48…).

Sơ đồ shape khi chồng block:

```text
x (B,T,C) -> Block1 (B,T,C) -> Block2 (B,T,C) -> ... -> BlockN (B,T,C) -> y (B,T,C)
```

Ở bài sau ta sẽ đặt `blocks` này vào giữa mô hình GPT hoàn chỉnh: token $\to$ embedding (+ positional) $\to$ _N blocks_ $\to$ LayerNorm cuối $\to$ lớp Linear ra logits.

---

### Nhìn sâu hơn

- **Vì sao feed-forward “phình 4 lần” ($4 \times \text{n\_embd}$)?** Lớp MLP bên trong block mở rộng chiều lên 4 lần rồi thu lại. Không gian ẩn rộng hơn cho phép mỗi token biến đổi phi tuyến “giàu” hơn trước khi ép về $C$. Con số $4\times$ là quy ước từ paper gốc và nanoGPT (không phải luật bất di bất dịch, nhưng là mặc định tốt).
- **Số tham số của một block (ước lượng thô):** Attention có 4 ma trận $C \times C$ ($Q$, $K$, $V$, và projection ghép head) $\to \sim 4C^2$. Feed-forward có $C \times 4C + 4C \times C \to \sim 8C^2$. Tổng $\sim 12C^2$ tham số cho một block (bỏ qua bias và LayerNorm nhỏ). Với $C=64$: $\sim 12 \times 64^2 \approx 49\text{k}$ tham số/block. Nhân với `n_layer` là ra phần lớn tham số của mô hình.
- **Đường cao tốc residual, nói kỹ hơn:** Vì $y = x + f(x)$, đạo hàm $\frac{\partial y}{\partial x} = 1 + f'(x)$. Cái số 1 đó đảm bảo gradient _luôn có một đường không suy giảm_ để chạy ngược qua hàng chục block. Đây là lý do sâu xa vì sao Transformer huấn luyện được mạng rất sâu, điều mà mạng “nối tiếp thuần” (không residual) không làm nổi.

💡 Một block không tự sinh ra thông tin vị trí. Bản thân attention là _hoán vị bất biến_ (đảo thứ tự token thì kết quả cũng đảo theo, không “biết” ai đứng trước ai). Đó là lý do _Bài 20_ phải thêm _positional embedding_ trước khi đưa vào các block.

---

### Thực hành

- **[Dễ]** Đổi `n_layer` từ 6 xuống 2 rồi in lại `y.shape`. Nó vẫn phải là `(2, 16, 64)`. Giải thích vì sao số block _không_ làm đổi shape đầu ra.
- **[Trung bình]** Trong `Block.forward`, đổi thử sang _post-norm_: `x = self.ln1(x + self.sa(x))` và `x = self.ln2(x + self.ffwd(x))`. Chạy lại, kiểm tra shape vẫn đúng. Rồi viết 2-3 câu so sánh: đường residual còn “sạch” không? (Gợi ý: LayerNorm giờ nằm _trên_ đường cộng.)
- **[Thử thách]** Đặt `n_head = 3` với `n_embd = 64`. Chạy thử và quan sát lỗi. Sửa lại cho `n_embd` chia hết cho `n_head` (vì `head_size = n_embd // n_head`). Giải thích vì sao ràng buộc chia hết này quan trọng khi ghép các head lại.

---

### Hiểu lầm thường gặp

- ❌ “Một block chỉ cần attention là đủ, feed-forward là thừa.” ➔ ✅ Sai. Attention lo _giao tiếp_ giữa token; feed-forward lo _biến đổi phi tuyến_ riêng từng token. Thiếu feed-forward, mô hình yếu hẳn.
- ❌ “Post-norm và pre-norm cho kết quả y hệt, chỉ là kiểu viết.” ➔ ✅ Khác nhau ở chỗ đặt LayerNorm. Pre-norm giữ đường residual sạch $\to$ dễ huấn luyện mạng rất sâu; đó là lý do GPT chọn pre-norm.
- ❌ “Xếp chồng nhiều block sẽ đổi shape tensor nên phải chỉnh lại kích thước sau mỗi block.” ➔ ✅ Không. Block giữ nguyên $(B, T, C)$, nên chồng bao nhiêu cũng cùng shape - đó chính là điều làm việc xếp chồng dễ dàng.
- ❌ “Block đã có attention nên tự biết thứ tự vị trí của token.” ➔ ✅ Sai. Attention là hoán vị bất biến; thông tin vị trí phải được thêm riêng bằng positional embedding (Bài 20).

---

### Mẹo ghi nhớ

- Một block = _2 sublayer_ (attention rồi feed-forward), mỗi cái bọc bởi _1 residual + 1 LayerNorm_.
- Thần chú pre-norm: _“cộng x, còn norm nằm bên trong”_ $\to$ `x = x + sub(LN(x))`.
- Shape đi vào bằng shape đi ra $\to$ chồng block như chồng gạch cùng kích thước.
- Ràng buộc: `n_embd % n_head == 0` để `head_size = n_embd // n_head` chia đều.

---

### Tóm tắt trực quan

| Khái niệm             | Một dòng                                                                                |
| :-------------------- | :-------------------------------------------------------------------------------------- |
| **Transformer block** | attention (giao tiếp) + feed-forward (suy nghĩ riêng), mỗi phần có residual + LayerNorm |
| **Pre-norm**          | `x = x + sublayer(LayerNorm(x))` - GPT/nanoGPT dùng, đường residual sạch                |
| **Post-norm**         | `x = LayerNorm(x + sublayer(x))` - Transformer gốc 2017                                 |
| **Hai sublayer**      | attention rồi feed-forward $\to$ hai cặp (residual + LayerNorm)                         |
| **Shape**             | vào $(B, T, C)$, ra $(B, T, C)$ - không đổi                                             |
| **Xếp chồng**         | `nn.Sequential(*[Block(...) for _ in range(n_layer)])`; sâu hơn = nhiều block           |
| **Tham số/block**     | $\sim 12 C^2$ (attention $\sim 4C^2$ + feed-forward $\sim 8C^2$)                        |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Coding a ChatGPT Like Transformer From Scratch in PyTorch](https://www.youtube.com/watch?v=C9QSpl5nmrY)
- PyTorch `nn.Module`, `nn.LayerNorm`, `nn.Sequential`: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT (Andrej Karpathy) - tham chiếu triển khai Block: [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- “Attention Is All You Need” (kiến trúc block gốc): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

---

### Tổng kết

- Một _Transformer block_ ghép _multi-head attention_ (token giao tiếp) với _feed-forward_ (mỗi token suy nghĩ riêng), mỗi phần bọc trong $x + \text{sublayer}(\text{LayerNorm}(x))$ theo kiểu _pre-norm_.
- Block có _2 LayerNorm_ và _2 kết nối residual_; đầu ra luôn cùng shape $(B, T, C)$ với đầu vào - nhờ đó ta _xếp chồng_ $N$ block dễ dàng bằng `nn.Sequential`, và số block chính là “chiều sâu” của mô hình.
- Pre-norm giữ đường residual sạch $\to$ huấn luyện mạng rất sâu ổn định, đó là lý do GPT-2, nanoGPT… đều chọn nó.

Nhưng còn một mảnh ghép quan trọng bị thiếu: _thứ tự vị trí_. Bản thân attention không phân biệt “token đứng đầu” hay “token cuối” - đảo chỗ token thì kết quả cũng đảo theo. Ở _Bài 20: Positional embedding - đưa thông tin vị trí vào mô hình_, chúng ta sẽ thêm thông tin vị trí vào embedding trước khi đưa vào chồng block, để mô hình biết “ai đứng ở đâu”. Hẹn gặp ở bài kế tiếp! 🤖
