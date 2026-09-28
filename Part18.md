# Bài 18: Residual connection và Layer Normalization

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 4:** Transformer block

---

### Giới thiệu

Hãy tưởng tượng bạn viết một bài luận rất dài và nhờ 10 người bạn lần lượt chỉnh sửa: người thứ nhất sửa xong đưa cho người thứ hai, người thứ hai sửa tiếp đưa người thứ ba… Nếu mỗi người _viết lại toàn bộ từ đầu_, thì tới người thứ 10, ý tưởng gốc của bạn có thể đã biến mất sạch. Nhưng nếu mỗi người chỉ _ghi chú sửa đổi bên lề_ rồi cộng vào bản gốc, thì thông tin ban đầu luôn được giữ lại. Đó chính là ý tưởng của _residual connection_ (kết nối tắt): thay vì $x = f(x)$, ta viết $x = x + f(x)$.

Ở Bài 17, chúng ta đã lắp xong _mạng feed-forward_ - khối biến đổi từng vị trí độc lập. Cùng với attention (Bài 12-16), giờ ta đã có hai “công nhân” chính của một Transformer block. Nhưng nếu chỉ xếp chồng chúng lên nhau cho mạng thật sâu, việc huấn luyện sẽ _rất khó_: gradient tan biến, loss không giảm. Bài này giới thiệu hai “chất keo” giúp mạng sâu học được: _residual connection_ và _Layer Normalization_.

> 🔑 **Một câu để nhớ:** `x = x + Sublayer(LN(x))` - LayerNorm chuẩn hóa đầu vào cho ổn định, còn residual (`x +`) tạo một “đường cao tốc” để gradient và thông tin chảy xuyên suốt hàng chục block mà không mất mát.

---

### Bạn sẽ học được gì?

- [ ] Hiểu vì sao mạng sâu khó huấn luyện (vanishing gradient) và residual connection giải quyết thế nào.
- [ ] Nắm công thức _LayerNorm_ và phân biệt nó với BatchNorm.
- [ ] Phân biệt _pre-norm_ vs _post-norm_ và vì sao GPT hiện đại chọn pre-norm.
- [ ] Cài đặt residual + LayerNorm quanh attention và feed-forward bằng PyTorch, theo dõi shape tensor.
- [ ] Kiểm tra được LayerNorm thực sự đưa mean $\approx$ 0, std $\approx$ 1 trên chiều đặc trưng.

📌 **Yêu cầu trước:** đã hiểu attention head (Bài 15-16), mạng feed-forward (Bài 17), và biết `nn.Module`, `nn.Linear`, tensor shape `(B, T, C)`.

---

### 1. Vì sao mạng sâu khó học? Vấn đề vanishing gradient

Khi ta xếp nhiều lớp lên nhau, gradient được lan truyền ngược bằng _quy tắc chuỗi_ (chain rule) - tức là _nhân liên tiếp_ đạo hàm của từng lớp.

**Analogy:** mỗi lớp giống một cái loa nhỏ nói lại thông điệp cho lớp trước. Nếu mỗi loa chỉ truyền được 0.8 độ lớn tín hiệu, thì qua 20 loa còn lại $0.8^{20} \approx 0.012$ - gần như im lặng. Các lớp đầu _gần như không nhận được tín hiệu học_ nên không cập nhật được.

Residual connection giải quyết bằng cách thêm một “đường tắt”: đầu ra = đầu vào _cộng_ với phần biến đổi.

```text
Không residual (khó học khi sâu):
   x ---> [ Sublayer ] ---> out          (gradient phải đi HẾT qua Sublayer)

Có residual (đường cao tốc cho gradient):
   x ---+--> [ Sublayer ] --(+)--> out
        |                     ^
        +---------------------+          out = x + Sublayer(x)
```

Vì đạo hàm của $\text{out} = x + f(x)$ theo $x$ là $1 + f'(x)$, luôn có _số hạng 1_ giúp gradient không bị triệt tiêu dù $f'(x)$ nhỏ. Đây chính là chìa khóa giúp ResNet (thị giác máy tính) và Transformer huấn luyện được hàng chục, thậm chí hàng trăm lớp.

💡 Hãy nhớ: residual _không_ làm mạng “mạnh hơn” theo nghĩa nhiều tham số - nó làm mạng _dễ huấn luyện hơn_ khi sâu. Đó là một khoản đầu tư gần như miễn phí (không thêm tham số).

---

### 2. Layer Normalization: giữ các con số "cùng thang đo"

**Analogy:** trong một lớp học, nếu điểm môn này chấm thang 10, môn kia chấm thang 100, so sánh sẽ lệch. Chuẩn hóa = quy tất cả về cùng một thang (vd trung bình 0, độ lệch chuẩn 1) để mọi thứ công bằng và ổn định.

_LayerNorm_ chuẩn hóa _trên chiều đặc trưng_ (feature) của _từng token_, độc lập với các token khác trong batch. Với một vector đặc trưng $x$ có $C$ phần tử:

```python
mu    = mean(x)                        # trung bình trên C chiều
var   = mean((x - mu)**2)              # phương sai trên C chiều
x_hat = (x - mu) / sqrt(var + eps)     # chuẩn hóa: mean ~ 0, std ~ 1
y     = gamma * x_hat + beta           # gamma, beta là tham số HỌC ĐƯỢC (mỗi cái C phần tử)
```

`gamma` (scale) và `beta` (shift) cho phép mạng “co giãn” và “dời” lại nếu cần, nên LayerNorm không ép cứng mọi thứ về đúng mean 0 std 1. `eps` (vd `1e-5`) tránh chia cho 0.

#### LayerNorm khác BatchNorm thế nào?

```text
Tensor kích hoạt: (B, T, C)  =  (batch, thời gian/token, đặc trưng)

BatchNorm : chuẩn hóa DỌC theo chiều batch (B) cho mỗi đặc trưng  -> phụ thuộc các mẫu KHÁC trong batch
LayerNorm : chuẩn hóa DỌC theo chiều đặc trưng (C) cho MỖI token -> độc lập batch, độc lập độ dài chuỗi
```

| Tiêu chí                          | BatchNorm              | LayerNorm        |
| :-------------------------------- | :--------------------- | :--------------- |
| **Chuẩn hóa theo chiều**          | Batch (B)              | Đặc trưng (C)    |
| **Phụ thuộc kích thước batch?**   | Có (kém khi batch nhỏ) | Không            |
| **Hợp với chuỗi dài/biến thiên?** | Kém                    | Tốt              |
| **Dùng ở đâu**                    | CNN, ảnh               | Transformer, NLP |

🎯 Transformer dùng _LayerNorm_ vì độ dài chuỗi và batch thay đổi liên tục; chuẩn hóa theo từng token giúp huấn luyện ổn định mà không phụ thuộc các câu khác trong batch.

---

### 3. Pre-norm vs Post-norm: đặt LayerNorm ở đâu?

Paper gốc “Attention Is All You Need” (2017) dùng _post-norm_: chuẩn hóa _sau_ khi cộng residual. GPT hiện đại (GPT-2 trở đi, nanoGPT) dùng _pre-norm_: chuẩn hóa _trước_ khi vào sublayer.

```text
POST-NORM (paper gốc):     x = LayerNorm( x + Sublayer(x) )
PRE-NORM (GPT-2/nanoGPT):  x = x + Sublayer( LayerNorm(x) )
```

Sơ đồ luồng của _một_ kết nối con kiểu pre-norm:

```text
x ---+-------------------------------> (+) ---> out
     |                                  ^
     +--> [ LayerNorm ] --> [ Sublayer ]-+
```

_Vì sao pre-norm thắng thế?_ Với pre-norm, đường residual (nhánh `x --->`) là một “đường thẳng sạch” không bị LayerNorm chen vào, nên gradient chảy xuyên suốt dễ hơn $\to$ huấn luyện mạng rất sâu _ổn định hơn_, ít cần warmup cầu kỳ. Post-norm thường cho chất lượng nhỉnh hơn một chút khi hội tụ nhưng khó tune.

| Tiêu chí                | Post-norm       | Pre-norm         |
| :---------------------- | :-------------- | :--------------- |
| **Công thức**           | `LN(x + f(x))`  | `x + f(LN(x))`   |
| **Đường residual**      | Bị LN “chặn”    | Sạch, thông suốt |
| **Ổn định khi rất sâu** | Khó, cần warmup | Dễ hơn           |
| **Ai dùng**             | Paper gốc 2017  | GPT-2, nanoGPT   |

⚠️ Đừng nhầm: _cả hai đều có residual_. Khác biệt chỉ là _vị trí đặt LayerNorm_ so với phép cộng residual, không phải có hay không có residual.

---

### 4. Cài đặt bằng PyTorch

Trước hết, dùng thẳng `nn.LayerNorm` của PyTorch và kiểm tra nó làm đúng việc:

```python
import torch
import torch.nn as nn

B, T, C = 2, 4, 8 # batch=2, tokens=4, features=8
x = torch.randn(B, T, C) * 5 + 3 # cố tình cho mean/std lệch
ln = nn.LayerNorm(C) # chuẩn hóa trên chiều cuối cùng (C=8)
y = ln(x) # shape giữ nguyên: (2, 4, 8)

# LayerNorm chuẩn hóa trên chiều đặc trưng C -> mean ~ 0, std ~ 1 cho MỖI token
print(y.shape) # torch.Size([2, 4, 8])
print(y.mean(dim=-1)) # ~ 0 cho mỗi token
print(y.std(dim=-1, unbiased=False)) # ~ 1 cho mỗi token

# Ví dụ output (xấp xỉ):
# tensor([[ 1.5e-08, -7.4e-09, ...], [...]])   # mean gan nhu 0
# tensor([[1.0000, 1.0000, ...], [...]])       # std ~ 1
```

Giờ ghép residual + pre-norm quanh một sublayer bất kỳ (attention hoặc feed-forward):

```python
import torch.nn.functional as F

class FeedForward(nn.Module):
    # A simple position-wise feed-forward network (from Bai 17)
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # expand: (B,T,C) -> (B,T,4C)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # project back: (B,T,4C) -> (B,T,C)
        )

    def forward(self, x):
        return self.net(x) # shape unchanged: (B, T, C)

class ResidualPreNorm(nn.Module):
    # Pre-norm residual wrapper: x = x + sublayer(LayerNorm(x))
    def __init__(self, n_embd, sublayer):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.sublayer = sublayer

    def forward(self, x):
        # x: (B, T, C). LayerNorm keeps shape; residual adds the clean input back.
        return x + self.sublayer(self.ln(x)) # (B, T, C)

# Quick test
n_embd = 32
block_ff = ResidualPreNorm(n_embd, FeedForward(n_embd))
x = torch.randn(2, 5, n_embd) # (B=2, T=5, C=32)
out = block_ff(x)
print(out.shape) # torch.Size([2, 5, 32])  -> shape KHONG doi
```

Điểm mấu chốt: _residual chỉ cộng được khi đầu ra sublayer cùng shape với đầu vào_ - `(B, T, C)` cộng `(B, T, C)`. Cả attention (`head_size = n_embd` sau khi gộp multi-head + projection) và feed-forward đều được thiết kế để giữ nguyên $C = \text{n\_embd}$, chính vì thế residual mới hoạt động trơn tru.

💡 Trong một Transformer block đầy đủ (Bài 19), ta sẽ gói _hai_ `ResidualPreNorm`: một quanh multi-head attention, một quanh feed-forward. LayerNorm cuối cùng (trước lớp logits) thường thêm ở ngoài block.

---

### Nhìn sâu hơn

- **Chi phí gần như bằng 0:** residual không thêm tham số nào; LayerNorm chỉ thêm $2 \times C$ tham số (`gamma` + `beta`) cho mỗi lần dùng - rất nhỏ so với các lớp Linear. Đổi lại, chúng cho phép mạng sâu học được - một món hời khổng lồ.
- **`eps` không chỉ để tránh chia 0:** nó còn giới hạn độ khuếch đại khi phương sai rất nhỏ, giúp số ổn định (không nổ thành `inf` / `nan`).
- **RMSNorm** (dùng trong LLaMA) là một biến thể của LayerNorm bỏ bước trừ mean, chỉ chia theo căn bậc hai của trung bình bình phương - rẻ hơn một chút và thực nghiệm cho kết quả tương đương. Ý tưởng “chuẩn hóa theo đặc trưng” vẫn giữ nguyên.
- **Thứ tự chuẩn: LN áp lên chiều cuối.** `nn.LayerNorm(C)` chuẩn hóa chiều cuối cùng của tensor. Với `(B, T, C)`, mỗi trong $B \times T$ token được chuẩn hóa độc lập trên $C$ giá trị.

---

### Thực hành

1. **[Dễ]** Tạo `x = torch.randn(3, 6, 16) * 10`. Áp `nn.LayerNorm(16)` rồi in `y.mean(dim=-1)` và `y.std(dim=-1, unbiased=False)`. Xác nhận mean $\approx$ 0, std $\approx$ 1.
2. **[Trung bình]** Viết lớp `ResidualPostNorm` (công thức `x = LayerNorm(x + sublayer(x))`) và so sánh forward với `ResidualPreNorm` trên cùng input. Quan sát output khác nhau ra sao.
3. **[Thử thách]** Xếp chồng 20 khối `ResidualPreNorm(FeedForward)` rồi cho một tensor `(1, 8, 32)` chạy qua; in norm (`x.norm()`) của đầu ra. Thử _bỏ_ residual (chỉ `x = sublayer(ln(x))`) và so sánh xem norm/gradient thay đổi thế nào khi mạng sâu.

---

### Hiểu lầm thường gặp

- ❌ “Pre-norm và post-norm khác nhau ở chỗ có residual hay không.” ➔ ✅ Cả hai đều có residual; khác biệt duy nhất là _vị trí LayerNorm_ so với phép cộng residual.
- ❌ “LayerNorm chuẩn hóa theo chiều batch giống BatchNorm.” ➔ ✅ LayerNorm chuẩn hóa theo _chiều đặc trưng_ của _từng token_, độc lập với các mẫu khác trong batch.
- ❌ “Residual làm mô hình mạnh hơn vì thêm tham số.” ➔ ✅ Residual _không thêm tham số_; nó giúp mạng sâu _dễ huấn luyện_ (gradient không tan biến).
- ❌ “LayerNorm ép đầu ra luôn có mean 0, std 1 cứng nhắc.” ➔ ✅ Nhờ `gamma` / `beta` học được, mạng có thể co giãn/dời lại phân phối nếu cần.

---

### Mẹo ghi nhớ

- Đọc công thức pre-norm như một câu: _“cộng vào chính mình phần biến đổi của bản đã chuẩn hóa”_: `x + Sublayer(LN(x))`.
- Nhớ chữ cái: **L**ayerNorm = **L**ast dimension (chiều cuối = đặc trưng). **B**atchNorm = **B**atch dimension.
- Residual = “note bên lề”, không phải “viết lại từ đầu” $\to$ giữ được thông tin gốc.
- 🎯 Mẹo vàng: nếu training mạng sâu bị `nan` / không hội tụ, việc đầu tiên cần kiểm tra là residual + LayerNorm đã đặt đúng chưa.

---

### Tóm tắt trực quan

| Khái niệm                   | Một dòng                                                                                          |
| :-------------------------- | :------------------------------------------------------------------------------------------------ |
| **Residual connection**     | `out = x + f(x)`; đường tắt giúp gradient không tan biến khi mạng sâu                             |
| **LayerNorm**               | Chuẩn hóa trên chiều đặc trưng của từng token: mean $\approx$ 0, std $\approx$ 1, có gamma / beta |
| **Pre-norm**                | `x + Sublayer(LN(x))`; đường residual sạch, ổn định khi rất sâu (GPT-2, nanoGPT)                  |
| **Post-norm**               | `LN(x + Sublayer(x))`; theo paper gốc 2017, khó tune hơn khi sâu                                  |
| **BatchNorm vs LayerNorm**  | Batch chuẩn hóa theo B; Layer chuẩn hóa theo C (hợp Transformer)                                  |
| **Điều kiện cộng residual** | Đầu ra sublayer phải cùng shape `(B, T, C)` với đầu vào                                           |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Visual Guide to Transformer Neural Networks - (Episode 2) Multi-Head & Self-Attention](https://www.youtube.com/watch?v=mMa2PmYJlCo)
- LayerNorm trong PyTorch: [https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- Tài liệu PyTorch tổng quát: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- nanoGPT (xem cách ghép residual + pre-norm trong `Block`): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Paper gốc “Attention Is All You Need” (mục 3.1, sublayer + LayerNorm): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer (minh họa residual + LayerNorm quanh block): [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

---

### Tổng kết

- Mạng sâu khó học vì _vanishing gradient_; _residual connection_ ($x + f(x)$) tạo đường cao tốc cho gradient, gần như miễn phí (không thêm tham số).
- _LayerNorm_ chuẩn hóa trên _chiều đặc trưng_ của từng token (mean $\approx$ 0, std $\approx$ 1), có `gamma` / `beta` học được; khác BatchNorm ở chỗ _độc lập batch_.
- _Pre-norm_ (`x + Sublayer(LN(x))`) giữ đường residual sạch, ổn định khi mạng rất sâu - lý do GPT-2/nanoGPT chọn nó thay cho post-norm của paper gốc.
- Điều kiện sống còn: sublayer phải _giữ nguyên shape_ `(B, T, C)` để cộng residual được.

Giờ ta đã có đủ toàn bộ linh kiện: attention (Bài 15-16), feed-forward (Bài 17), và bộ keo residual + LayerNorm (bài này). Ở _Bài 19: Ghép thành một Transformer block hoàn chỉnh_, chúng ta sẽ lắp tất cả lại thành một Block chuẩn - hai kết nối con pre-norm quanh attention và feed-forward - viên gạch mà GPT xếp chồng hàng chục lần để tạo nên sức mạnh của mình. Hẹn gặp ở bài sau! 🤖
