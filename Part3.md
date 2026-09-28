# Bài 3: Ôn tập PyTorch cốt lõi - tensor, autograd và nn.Module

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 1:** Môi trường, phần cứng & nền tảng

---

### Giới thiệu

Ở _Bài 2_, mình đã cùng nhau chọn phần cứng: bao nhiêu VRAM, mua card RTX hay thuê GPU trên cloud, cách ước lượng bộ nhớ theo số tham số. Giờ bạn đã có “cỗ máy” rồi, nhưng máy đắt đến mấy mà không biết lái thì cũng chỉ để trưng. Bài này là buổi tập lái: chúng ta ôn lại _ba viên gạch nền tảng của PyTorch_ mà từ đây tới cuối khóa, bài nào bạn cũng sẽ đụng tới.

Hãy tưởng tượng bạn đang học nấu ăn. `tensor` giống như _nguyên liệu_ (rau, thịt, đo bằng số lượng theo từng chiều). `autograd` giống như một _người phụ bếp thần kỳ_: bạn cứ nấu (tính toán) tự nhiên, cậu ấy âm thầm ghi lại mọi bước, và khi bạn hỏi “món này mặn quá là vì bước nào?”, cậu ấy chỉ ra ngay từng gia vị đã ảnh hưởng bao nhiêu. Còn `nn.Module` là _công thức món ăn đóng gói lại_: gói nguyên liệu + các bước thành một khối tái sử dụng được.

Ba thứ này chính là bộ đồ nghề để về sau ta dựng `embedding`, `attention`, Transformer block và cả GPT mini. Hiểu chắc hôm nay, các bài sau bạn sẽ thấy nhẹ tênh.

> 🔑 **Một câu để nhớ:** PyTorch = tensor (dữ liệu) + autograd (tự tính đạo hàm) + nn.Module (đóng gói mô hình) - ba thứ này ghép lại thành mọi mạng nơ-ron, kể cả GPT.

---

### Bạn sẽ học được gì?

- [ ] Tạo và thao tác `tensor`: shape, phép toán element-wise, nhân ma trận, chuyển lên GPU.
- [ ] Hiểu `autograd`: `requires_grad`, `.backward()`, `.grad` và đồ thị tính toán.
- [ ] Tự viết một `nn.Module` đơn giản và chạy `forward`.
- [ ] Ghép mọi thứ thành một vòng huấn luyện (training loop) tối giản với `AdamW`.
- [ ] Đọc và suy luận _shape tensor_ qua từng bước - kỹ năng sống còn cả khóa.

📌 **Yêu cầu trước:** Đã cài Python + PyTorch (Bài 1), biết Python cơ bản và ý niệm gradient/đạo hàm ở mức phổ thông.

---

### 1. Tensor - nguyên liệu của mọi thứ

Một tensor chỉ đơn giản là _một mảng số nhiều chiều_. Số 5 là tensor 0 chiều (scalar), một danh sách `[1, 2, 3]` là tensor 1 chiều (vector), một bảng số là 2 chiều (matrix), và một chồng bảng là 3 chiều… Trong LLM, dữ liệu hầu như luôn có dạng `(batch, sequence_length, embedding_dim)` - tức chồng nhiều câu, mỗi câu nhiều token, mỗi token là một vector.

```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]]) # shape: (2, 3)
print(x.shape) # torch.Size([2, 3])
print(x.dtype) # torch.float32

# Element-wise operations broadcast automatically
y = x * 2 + 1 # shape stays (2, 3)
print(y)
# tensor([[ 3.,  5.,  7.],
#         [ 9., 11., 13.]])
```

Phép quan trọng nhất trong deep learning là _nhân ma trận_ (matmul), viết bằng toán tử `@`. Quy tắc shape: `(a, b) @ (b, c) -> (a, c)`. Chiều “giữa” phải khớp nhau.

```python
A = torch.randn(2, 3) # shape: (2, 3)
B = torch.randn(3, 4) # shape: (3, 4)
C = A @ B             # shape: (2, 4)  -- inner dims 3 == 3 match
print(C.shape)        # torch.Size([2, 4])
```

Sơ đồ shape khi nhân ma trận:

```text
A (2 x 3)        B (3 x 4)        C (2 x 4)
  [ . . . ]        [ . . . . ]      [ . . . . ]
  [ . . . ]   @    [ . . . . ]  ->  [ . . . . ]
                   [ . . . . ]
     ^  ^-------------^  ^                ^--- ket qua
     |    inner = 3 khop nhau
  outer = 2                    outer = 4
```

💡 90% lỗi khi dựng mô hình là _sai shape_. Mỗi khi viết một dòng tensor, hãy tự hỏi: “shape vào là gì, shape ra là gì?” và ghi vào comment. Đây là thói quen của dân chuyên nghiệp.

#### Chuyển lên GPU

Tensor mặc định nằm trên CPU. Để tận dụng GPU (nhanh hơn hàng chục lần cho phép nhân ma trận lớn), ta chuyển nó sang thiết bị `cuda`:

```python
# Pick the best available device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device) # 'cuda' if a GPU is present, else 'cpu'

x = torch.randn(1000, 1000)
x = x.to(device) # move tensor to GPU (or stay on CPU)

# Both operands must be on the SAME device, otherwise it errors
z = x @ x # heavy matmul runs on GPU when device='cuda'
print(z.device) # cuda:0  (or cpu)
```

⚠️ Lỗi kinh điển: `Expected all tensors to be on the same device`. Nếu model ở `cuda` mà dữ liệu ở `cpu` (hoặc ngược lại) sẽ báo lỗi. Nhớ `.to(device)` cho _cả model lẫn dữ liệu_.

---

### 2. Autograd - người phụ bếp ghi lại mọi bước

Huấn luyện mạng nơ-ron thực chất là: tính _loss_ (sai số), rồi tìm xem mỗi tham số nên chỉnh tăng hay giảm để loss nhỏ đi. “Nên chỉnh bao nhiêu” chính là _gradient_ (đạo hàm của loss theo tham số). Tính tay thì cực; PyTorch làm tự động bằng `autograd`.

Cơ chế: khi một tensor có `requires_grad=True`, mọi phép toán trên nó được ghi lại thành một _đồ thị tính toán_ (computational graph). Gọi `.backward()` trên kết quả, PyTorch chạy ngược đồ thị (chain rule) và điền gradient vào `.grad` của từng tensor lá.

```python
import torch

# A leaf tensor we want gradients for
w = torch.tensor(3.0, requires_grad=True)

# Forward pass: build the graph  y = w^2 + 2w
y = w**2 + 2*w # scalar output

# Backward pass: compute dy/dw automatically
y.backward() # dy/dw = 2w + 2 = 2*3 + 2 = 8
print(w.grad) # tensor(8.)
```

Bạn thấy đó: mình không hề viết công thức đạo hàm `2w + 2`, PyTorch tự suy ra. Kết quả đúng chính xác bằng `2*3 + 2 = 8`.

Sơ đồ đồ thị tính toán (forward đi xuôi, gradient chảy ngược lên):

```text
forward  ->
   w  --[ square ]--> w^2  --\
    \                          [ add ] --> y
     \--[ times 2 ]--> 2w  --/

 <-  backward (chain rule)
   dy/dw = d(w^2)/dw + d(2w)/dw = 2w + 2
```

💡 Sau `.backward()`, PyTorch _cộng dồn_ gradient vào `.grad`. Vì vậy mỗi bước huấn luyện phải `optimizer.zero_grad()` để reset về 0, nếu không gradient của các batch sẽ chồng lên nhau sai lệch.

Khi chỉ chạy suy luận (không cần huấn luyện), ta tắt autograd để tiết kiệm bộ nhớ và tăng tốc:

```python
with torch.no_grad(): # do NOT track graph -> less memory, faster
    pred = w * 10     # no gradient recorded here
```

---

### 3. nn.Module - đóng gói mô hình

`nn.Module` là lớp cha để đóng gói một khối mạng: nó tự quản lý tham số (weights/bias), tự biết cách chuyển lên GPU, tự liệt kê tham số cho optimizer. Bạn kế thừa nó, khai báo các lớp con trong `__init__`, và viết luồng tính trong `forward`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden) # weight: (hidden, in_dim)
        self.fc2 = nn.Linear(hidden, out_dim) # weight: (out_dim, hidden)

    def forward(self, x):
        # x shape: (batch, in_dim)
        x = F.relu(self.fc1(x)) # -> (batch, hidden)
        x = self.fc2(x)         # -> (batch, out_dim)  = logits
        return x

model = TinyMLP(in_dim=4, hidden=8, out_dim=3)
inp = torch.randn(5, 4) # batch of 5 samples, each 4 features
out = model(inp)        # calls forward()  -> shape (5, 3)
print(out.shape)        # torch.Size([5, 3])

# nn.Module knows all its parameters
n_params = sum(p.numel() for p in model.parameters())
print(n_params) # 4*8+8 + 8*3+3 = 40 + 27 = 67 params
```

Chú ý ta gọi `model(inp)` chứ _không_ gọi `model.forward(inp)` trực tiếp - PyTorch đã hook thêm việc quản lý ở bước gọi. Đầu ra `out` ở đây là _logits_ (điểm số thô cho mỗi lớp), _chưa_ phải xác suất. Muốn thành xác suất mới đưa qua `softmax`. Phân biệt này rất quan trọng, ta sẽ gặp lại liên tục trong LLM.

Cấu trúc luồng dữ liệu qua module (nền tảng cho Transformer block sau này):

```text
input (batch, in_dim)
      |
   [ Linear fc1 ] -> (batch, hidden)
      |
   [ ReLU ]       -> (batch, hidden)
      |
   [ Linear fc2 ] -> (batch, out_dim)  = logits (chua phai xac suat)
      |
   [ softmax ]    -> (batch, out_dim)  = xac suat (tuy chon)
```

---

### 4. Ghép tất cả: một vòng huấn luyện tối giản

Giờ ta ghép `tensor` + `autograd` + `nn.Module` thành một _training loop_ thu nhỏ. Đây chính là bộ khung mà bài huấn luyện GPT mini về sau sẽ dùng - chỉ khác model to hơn và loss tính trên next-token.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TinyMLP(4, 8, 3).to(device) # move model to device

# AdamW is the standard optimizer for Transformers/LLMs
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

# Fake data: 5 samples, 4 features, target class in {0,1,2}
X = torch.randn(5, 4, device=device)
target = torch.tensor([0, 2, 1, 0, 1], device=device) # shape (5,)

for step in range(100):
    logits = model(X)                      # (5, 3) forward pass
    loss = F.cross_entropy(logits, target) # scalar loss
    optimizer.zero_grad()                  # reset accumulated grads
    loss.backward()                        # autograd fills .grad for every parameter
    optimizer.step()                       # AdamW updates parameters using .grad

    if step % 25 == 0:
        print(step, round(loss.item(), 4))
        # 0   1.23...
        # 25  0.31...
        # 50  0.09...   (loss keeps dropping -> model is learning)
```

Bốn dòng lõi của MỌI training loop trong PyTorch, thuộc lòng luôn nhé:

```text
logits = model(X)           # 1. forward: du lieu -> du doan
loss   = loss_fn(logits, y) # 2. tinh sai so
optimizer.zero_grad()       # 3a. xoa grad cu
loss.backward()             # 3b. autograd tinh grad moi
optimizer.step()            # 4. cap nhat tham so
```

🎯 `F.cross_entropy` nhận thẳng _logits_ (không phải xác suất) và _nhãn lớp dạng số nguyên_ (không phải one-hot). Nó gộp `log_softmax` + `negative log-likelihood` bên trong, ổn định số học hơn. Đây đúng là hàm loss ta dùng cho next-token prediction ở các bài sau.

---

### Nhìn sâu hơn

- **Đồ thị động (dynamic graph):** khác một số framework cũ, PyTorch dựng đồ thị _ngay khi chạy_ (define-by-run). Nghĩa là bạn có thể dùng `if`, `for` trong `forward` bình thường - rất tiện để debug, nhưng cũng vì thế mà đồ thị + gradient chiếm bộ nhớ. Nhớ dùng `torch.no_grad()` khi suy luận.
- **Vì sao gradient cộng dồn?** Thiết kế này cho phép cộng gradient của nhiều mini-batch trước khi cập nhật (gradient accumulation) - mẹo giúp “giả lập” batch lớn khi VRAM có hạn (liên hệ trực tiếp Bài 2!). Cái giá là ta phải nhớ `zero_grad()`.
- **`.item()` vs tensor:** `loss` là một tensor 0-chiều còn dính đồ thị. Khi chỉ muốn in số, dùng `loss.item()` để lấy Python float, tránh vô tình giữ cả đồ thị trong bộ nhớ (rò rỉ VRAM khi log trong vòng lặp dài).
- **`float32` vs `float16`:** huấn luyện lớn thường dùng `float16` / `bfloat16` để tiết kiệm nửa bộ nhớ và chạy nhanh trên Tensor Core - đúng như phần ước lượng VRAM ở Bài 2. Ta sẽ quay lại khi tối ưu huấn luyện.

---

### Thực hành

- **[Dễ]** Tạo `a = torch.arange(6).reshape(2, 3)` và `b = torch.ones(3, 2)`. Tự đoán shape của `a.float() @ b` trước, rồi chạy để kiểm tra. (Đáp án shape: `(2, 2)`).
- **[Trung bình]** Với `w = torch.tensor(2.0, requires_grad=True)`, tính `y = 3*w**2 + w`, gọi `y.backward()` và in `w.grad`. Tự tính tay đạo hàm `6w + 1 = 13` rồi so sánh.
- **[Thử thách]** Sửa `TinyMLP` thành 3 lớp Linear (thêm một lớp ẩn), thêm `F.relu` giữa các lớp, đếm lại số tham số bằng `sum(p.numel() ...)`, và chạy lại training loop. Quan sát loss có giảm không. Thử đổi `lr` từ `1e-2` sang `1.0` xem chuyện gì xảy ra (gợi ý: loss có thể phóng lên `nan`).

---

### Hiểu lầm thường gặp

- ❌ “Gọi `.backward()` một lần là dùng gradient mãi mãi.” ➔ ✅ Gradient bị _cộng dồn_ vào `.grad`; mỗi bước phải `optimizer.zero_grad()` trước, nếu không sẽ sai.
- ❌ “Output của model là xác suất.” ➔ ✅ Output của lớp Linear cuối là _logits_ (điểm thô). Muốn xác suất phải qua `softmax`; và `cross_entropy` lại muốn nhận _logits_, đừng `softmax` trước rồi đưa vào nó.
- ❌ “Cứ để tensor ở CPU cũng chạy GPU được.” ➔ ✅ Phải `.to(device)` cả model lẫn dữ liệu; trộn thiết bị sẽ báo lỗi.
- ❌ “`cross_entropy` cần nhãn one-hot.” ➔ ✅ Nó nhận _nhãn số nguyên_ (chỉ số lớp), tự lo phần còn lại. One-hot là chuyện khác (và tốn bộ nhớ hơn embedding - ta sẽ so sánh ở bài sau).
- ❌ “Gọi `model.forward(x)` cũng như `model(x)`.” ➔ ✅ Luôn gọi `model(x)`; gọi thẳng `.forward()` bỏ qua các hook nội bộ của PyTorch.

---

### Mẹo ghi nhớ

- **Ba viên gạch:** `tensor` (dữ liệu) ➔ `autograd` (đạo hàm tự động) ➔ `nn.Module` (đóng gói).
- **Bốn dòng vàng của training loop:** `forward` ➔ `loss` ➔ `zero_grad` ➔ `backward` ➔ `step`.
- **Luôn nghĩ theo shape:** ghi shape vào comment ngay khi viết dòng tensor.
- Mẹo nhớ thứ tự loop: câu thần chú “_Đoán - Sai - Xóa - Học - Bước_” (predict, loss, zero, backward, step).

---

### Tóm tắt trực quan

| Khái niệm                 | Một dòng để nhớ                                       |
| :------------------------ | :---------------------------------------------------- |
| **tensor**                | Mảng số nhiều chiều; luôn để ý `.shape`               |
| **@ matmul**              | `(a,b) @ (b,c) -> (a,c)`; chiều giữa phải khớp        |
| **.to(device)**           | Chuyển tensor/model lên GPU; phải cùng thiết bị       |
| **requires_grad**         | Bật ghi đồ thị để tính gradient                       |
| **.backward()**           | Chạy ngược đồ thị, điền `.grad` (cộng dồn)            |
| **torch.no_grad()**       | Tắt autograd khi suy luận, tiết kiệm bộ nhớ           |
| **nn.Module**             | Đóng gói mô hình; `__init__` khai báo, `forward` tính |
| **logits vs xác suất**    | Linear ra logits; softmax mới ra xác suất             |
| **AdamW + cross_entropy** | Bộ đôi chuẩn để huấn luyện Transformer/LLM            |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Train a PyTorch neural network step-by-step | PyTorch deep learning (Feb 2023)](https://www.youtube.com/watch?v=v9rputAi9iE)
- PyTorch docs: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials (bắt đầu tại đây): [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT của Andrej Karpathy (đích đến của khóa): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Attention Is All You Need (paper gốc): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

---

### Tổng kết

- `tensor` là dữ liệu nhiều chiều; luôn tư duy theo _shape_ và nhớ quy tắc matmul `(a,b)@(b,c)->(a,c)`.
- `autograd` tự dựng đồ thị và tính gradient khi bạn gọi `.backward()`; gradient _cộng dồn_ nên phải `zero_grad()` mỗi bước.
- `nn.Module` đóng gói mô hình: `__init__` khai báo lớp, `forward` mô tả luồng tính; đầu ra Linear cuối là _logits_.
- Vòng huấn luyện tối giản `forward` ➔ `loss` ➔ `zero_grad` ➔ `backward` ➔ `step` với `AdamW` + `cross_entropy` chính là bộ khung ta sẽ tái dùng cho GPT mini.

Bạn vừa nắm chắc bộ đồ nghề PyTorch. Nhưng đồ nghề để làm gì? Ở _Bài 4: Mô hình ngôn ngữ là gì - dự đoán token tiếp theo_, ta sẽ trả lời câu hỏi cốt lõi: một LLM thực ra chỉ đang làm một việc đơn giản đến bất ngờ - _đoán chữ tiếp theo_. Ta sẽ biến bài toán “đoán token” thành đúng bài toán phân loại + cross_entropy mà bạn vừa học hôm nay. Hẹn gặp lại! 🤖
