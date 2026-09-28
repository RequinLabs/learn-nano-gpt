### Giới thiệu

Chào mừng bạn đến với khóa _“Xây dựng LLM từ đầu”_! 🤖 Hãy tưởng tượng bạn muốn học nấu ăn từ một đầu bếp giỏi. Đầu bếp đó có thể đưa bạn một hộp mì gói (dùng API của OpenAI - nhanh nhưng bạn chẳng hiểu gì bên trong), HOẶC dạy bạn tự tay làm từng công đoạn: chọn nguyên liệu, thái, nêm, canh lửa. Khóa học này chọn cách thứ hai. Chúng ta sẽ _tự tay dựng một mô hình GPT thu nhỏ bằng PyTorch_, đúng tinh thần nanoGPT của Andrej Karpathy - viết từng dòng, hiểu từng phép nhân ma trận.

Nhưng trước khi “vào bếp”, ta cần một căn bếp có đầy đủ dụng cụ. Bài đầu tiên này chính là dựng căn bếp đó: cài _Python_, cài _PyTorch_, và quan trọng nhất - có một chiếc _GPU miễn phí_ để chạy nhanh gấp hàng chục lần. Đừng lo nếu bạn chưa từng cài gì, mình sẽ đi từng bước một, từ số 0.

> 🔑 **Một câu để nhớ:** LLM chỉ là một hàm toán học khổng lồ _đoán token tiếp theo_; và để “chạy” hàm đó đủ nhanh, ta cần Python + PyTorch + GPU.

---

### Bạn sẽ học được gì?

- [ ] Hiểu _bức tranh lớn_: LLM/GPT là gì và cuối khóa bạn tự build được gì
- [ ] Cài _Python_ trên máy của bạn (Windows/macOS/Linux)
- [ ] Cài _PyTorch_ đúng phiên bản (CPU hoặc GPU)
- [ ] Bật _GPU miễn phí_ trên Google Colab và verify bằng code
- [ ] Chạy được phép nhân _tensor trên GPU_ và kiểm tra `torch.cuda.is_available()`

📌 **Yêu cầu trước:** Bạn đã biết Python cơ bản (biến, hàm, vòng lặp) và có khái niệm sơ bộ về Machine Learning (mô hình, tham số, huấn luyện). Không cần biết PyTorch trước - ta học dần.

---

### 1. Bức tranh lớn: LLM và GPT là gì?

- **Analogy:** Khi bạn nhắn tin trên điện thoại, gõ “Hẹn gặp em lúc” thì bàn phím gợi ý chữ tiếp theo: “tối”, “chiều”, “7”. Bàn phím đang _đoán token kế tiếp_ dựa trên những gì bạn đã gõ. Một _LLM (Large Language Model)_ làm chính xác việc đó - nhưng ở quy mô khổng lồ và “thông minh” hơn rất nhiều.
- **Định nghĩa chính xác:** Một LLM là mô hình xác suất `P(token_tiếp_theo | các_token_trước_đó)`. _GPT_ (Generative Pre-trained Transformer) là một họ LLM dùng kiến trúc _Transformer_ và huấn luyện theo kiểu _tự hồi quy_ (autoregressive): sinh từng token một, mỗi token mới lại được đưa ngược vào để đoán token sau.

Cuối khóa này bạn sẽ _tự viết_ toàn bộ các mảnh ghép đó:

```text
token IDs      embedding        N x Transformer block          logits
[7,42,3,...] -> [vector d] --------> [attention + MLP] --------> [vocab_size]
   (T,)          (T, d)                  (T, d)                    (T, V)
                                                                     |
                                                              softmax + sample
                                                                     v
                                                            token tiếp theo
```

Mỗi ô trong sơ đồ trên là một bài học. Hôm nay ta chỉ lo phần “điện nước”: cài môi trường để chạy các phép tính này.

💡 _Đừng nhầm:_ _pretraining_ (học đoán token trên núi văn bản) khác _fine-tuning_ (chỉnh lại cho một nhiệm vụ cụ thể) và khác _RLHF_ (căn chỉnh theo phản hồi của con người). Khóa này tập trung vào phần cốt lõi: kiến trúc + pretraining một GPT mini.

---

### 2. Tại sao cần PyTorch và GPU?

- **Analogy:** Nhân hai ma trận lớn giống như phát bài cho 10.000 người cùng lúc. CPU giống một người chia bài rất khéo nhưng chỉ có 8-16 tay (lõi). GPU giống một sân vận động với hàng nghìn cái tay đơn giản, chia bài song song ào ào. Huấn luyện mạng nơ-ron = hàng tỷ phép nhân ma trận, nên GPU nhanh hơn CPU hàng chục đến hàng trăm lần.

_PyTorch_ cho ta hai thứ vàng ngọc:

1. _Tensor_ - mảng nhiều chiều chạy được trên GPU (như numpy nhưng biết dùng GPU).
2. _Autograd_ - tự động tính đạo hàm (gradient) để huấn luyện, ta không phải đạo hàm bằng tay.

```python
import torch

# A tensor is a multi-dimensional array; here shape (2, 3)
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(x.shape) # torch.Size([2, 3])
print(x.dtype) # torch.float32

# Autograd: track gradients for training
w = torch.randn(3, requires_grad=True) # shape (3,)
y = (x @ w).sum() # matmul (2,3)@(3,) -> (2,); then sum -> scalar
y.backward()      # compute d(y)/d(w) automatically
print(w.grad.shape) # torch.Size([3])  gradient has same shape as w
```

🎯 Bạn không cần hiểu hết đoạn code này ngay. Ý chính: PyTorch cho ta _tensor_ + _đạo hàm tự động_, hai viên gạch để xây mọi mạng nơ-ron.

---

### 3. Cài Python từ số 0

PyTorch chạy trên Python (khuyến nghị _Python 3.9 trở lên_). Kiểm tra xem máy đã có chưa:

```bash
python3 --version # Expected output like: Python 3.11.9
```

Nếu chưa có, tải bản chính thức tại [python.org/downloads](https://www.python.org/downloads/):

| Hệ điều hành | Cách cài                                                                                                         |
| :----------- | :--------------------------------------------------------------------------------------------------------------- |
| **Windows**  | Tải file .exe từ [python.org](https://www.python.org), khi cài _NHỚ tích ô “Add Python to PATH”_ rồi bấm Install |
| **macOS**    | Tải file .pkg từ [python.org](https://www.python.org), hoặc dùng `brew install python` nếu có Homebrew           |
| **Linux**    | Thường có sẵn; nếu thiếu: `sudo apt install python3 python3-pip` (Ubuntu/Debian)                                 |

⚠️ Trên Windows, quên tích _“Add Python to PATH”_ là lỗi phổ biến nhất khiến gõ python báo “không tìm thấy lệnh”. Nếu lỡ quên, cài lại và nhớ tích.

_Mẹo:_ dùng môi trường ảo (virtual environment) để không làm rối máy:

```bash
# Create an isolated environment named .venv
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Activate it (Windows PowerShell)
# .venv\Scripts\Activate.ps1
```

---

### 4. Cài PyTorch

Đây là bước dễ sai nhất, vì PyTorch có _nhiều bản khác nhau_ tùy máy bạn có GPU NVIDIA hay không. Cách an toàn nhất: vào trang chính thức [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), chọn hệ điều hành + có/không CUDA, trang sẽ _sinh ra đúng câu lệnh_ cho bạn.

```bash
# Simplest: CPU-only build (works everywhere, slower)
pip install torch

# If you have an NVIDIA GPU + CUDA, the official site gives a command like:
# pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

Sau khi cài, verify:

```python
import torch

print(torch.__version__)         # e.g. 2.4.1
print(torch.cuda.is_available()) # True if a CUDA GPU is usable, else False
```

💡 Máy bạn _không có_ GPU NVIDIA thì `torch.cuda.is_available()` trả về `False` - hoàn toàn bình thường. Ta sẽ dùng GPU miễn phí trên Colab ở mục tiếp theo.

---

### 5. GPU miễn phí trên Google Colab

Không có GPU xịn? Google tặng bạn một cái - miễn phí. _Google Colab_ ([colab.research.google.com](https://colab.research.google.com/)) là một cuốn sổ notebook chạy trên đám mây, có sẵn Python + PyTorch, và cho dùng GPU không mất tiền (có giới hạn thời gian).

_Các bước bật GPU:_

1. Mở [https://colab.research.google.com/](https://colab.research.google.com/) -> New notebook
2. Menu: `Runtime` -> `Change runtime type`
3. Hardware accelerator: chọn **GPU** (vd T4) -> Save
4. Runtime sẽ khởi động lại với GPU gắn sẵn

Sau đó, trong một ô code (cell) của notebook, chạy đoạn verify - đây là _“Hello GPU”_ của khóa học:

```python
import torch

# 1) Is a CUDA GPU available in this runtime?
print("CUDA available:", torch.cuda.is_available())

# 2) Pick device: use GPU if we have one, else fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device) # e.g. Using device: cuda

if device == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0)) # e.g. Tesla T4

# 3) Do a real matrix multiply ON THE GPU
a = torch.randn(1000, 1000, device=device) # shape (1000, 1000) on GPU
b = torch.randn(1000, 1000, device=device) # shape (1000, 1000) on GPU
c = a @ b # matmul -> shape (1000, 1000)

print("Result shape:", c.shape)   # torch.Size([1000, 1000])
print("Result device:", c.device) # cuda:0  (computed on the GPU!)
```

Kết quả mẫu khi chạy trên Colab GPU:

```text
CUDA available: True
Using device: cuda
GPU name: Tesla T4
Result shape: torch.Size([1000, 1000])
Result device: cuda:0
```

🎯 Thấy `cuda:0` ở dòng device nghĩa là phép nhân _thật sự chạy trên GPU_, không phải CPU. Chúc mừng - căn bếp của bạn đã có lửa! ✅

---

### Nhìn sâu hơn

_Vì sao phải để tensor và mô hình cùng trên một device?_ PyTorch không tự động di chuyển dữ liệu qua lại. Nếu `a` nằm trên GPU còn `b` nằm trên CPU, phép `a @ b` sẽ báo lỗi kiểu `Expected all tensors to be on the same device`. Đây là lỗi số 1 người mới gặp. Cách chuyển:

```python
x = torch.randn(4, 8) # created on CPU by default, shape (4, 8)
x = x.to(device)      # move to GPU (or stay on CPU)

# Whole models are moved the same way:  model = model.to(device)
```

_Chi phí ẩn:_ chuyển dữ liệu CPU <-> GPU tốn thời gian. Mẹo huấn luyện là _đưa dữ liệu lên GPU một lần_ rồi tính nhiều lần trên đó, tránh chuyển đi chuyển lại mỗi bước.

_Colab có gì cần lưu ý:_ phiên GPU miễn phí có thể bị ngắt sau vài giờ hoặc khi để không (idle). Với các bài huấn luyện dài của khóa, ta sẽ lưu checkpoint để không mất công. Colab cũng cho chọn _CPU / GPU / TPU_ - khóa này dùng _GPU_ (TPU cần code khác).

---

### Thực hành

- **[Dễ]** Mở Colab, bật GPU, chạy đoạn “Hello GPU” ở mục 5. Chụp lại dòng in ra `Using device: cuda` và `Result device: cuda:0`.
- **[Trung bình]** So sánh tốc độ: nhân hai ma trận 4000 x 4000 trên `cpu` rồi trên `cuda`, đo bằng `time`. Bạn thấy GPU nhanh hơn bao nhiêu lần?

```python
import torch, time

for dev in ["cpu", "cuda"]:
    if dev == "cuda" and not torch.cuda.is_available():
        continue
    x = torch.randn(4000, 4000, device=dev) # shape (4000, 4000)
    if dev == "cuda":
        torch.cuda.synchronize() # wait so timing is accurate
    t0 = time.time()
    y = x @ x # matmul -> (4000, 4000)
    if dev == "cuda":
        torch.cuda.synchronize() # GPU is async; sync before stopping timer
    print(dev, round(time.time() - t0, 4), "s")
```

- **[Thử thách]** Tạo tensor `x` trên CPU và tensor `w` trên GPU rồi thử `x @ w`. Đọc kỹ thông báo lỗi, sau đó sửa bằng `.to(device)` cho cả hai. Giải thích vì sao PyTorch bắt cùng device.

---

### Hiểu lầm thường gặp

- ❌ “Không có GPU thì không học được khóa này.” ➔ ✅ Vẫn học được: CPU chạy được mọi code, chỉ chậm hơn; và Colab cho GPU _miễn phí_.
- ❌ “`pip install torch` luôn cài bản GPU.” ➔ ✅ Mặc định thường là bản CPU; muốn GPU phải chọn đúng lệnh CUDA từ [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
- ❌ “Cứ tạo tensor là nó nằm trên GPU.” ➔ ✅ Mặc định tensor nằm trên _CPU_; phải `device='cuda'` hoặc `.to(device)` mới lên GPU.
- ❌ “GPU và CPU tensor cộng/nhân trực tiếp được.” ➔ ✅ Phải _cùng device_, nếu không PyTorch báo lỗi.

---

### Mẹo ghi nhớ

- Luôn khai báo một biến `device = "cuda" if torch.cuda.is_available() else "cpu"` ở đầu notebook, rồi dùng nó ở mọi nơi.
- Thấy `cuda:0` trên `.device` = đang chạy trên GPU. Thấy `cpu` = chưa chuyển lên GPU.
- _Mẹo vàng:_ “Data lên device, model lên device, cùng một nhà” - dữ liệu và mô hình phải ở chung một device.

---

### Tóm tắt trực quan

| Khái niệm                     | Một dòng                                       |
| :---------------------------- | :--------------------------------------------- |
| **LLM**                       | Hàm đoán token tiếp theo: `P(next \| context)` |
| **GPT**                       | LLM dùng Transformer, sinh token tự hồi quy    |
| **PyTorch**                   | Thư viện tensor + autograd để dựng mạng nơ-ron |
| **Tensor**                    | Mảng nhiều chiều, chạy được trên CPU hoặc GPU  |
| **torch.cuda.is_available()** | Kiểm tra có GPU CUDA dùng được không           |
| **Google Colab**              | Notebook đám mây, cho GPU miễn phí             |
| **device**                    | Nơi tensor sống: `cpu` hoặc `cuda`             |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Step-by-Step Guide: Installing PyTorch with CUDA GPU Support on Windows!](https://learningvn.com/learn/xay-dung-llm-tu-dau-tu-attention-den-gpt-mini-pytorch/lesson/98ff843d-0f39-4112-8f0c-ec9f74b8669b)
- PyTorch - cài đặt chính thức: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
- PyTorch - tài liệu API: [pytorch.org/docs/stable](https://pytorch.org/docs/stable/)
- PyTorch - tutorials: [pytorch.org/tutorials](https://pytorch.org/tutorials/)
- Python - tải về: [python.org/downloads](https://www.python.org/downloads/)
- Google Colab: [colab.research.google.com](https://colab.research.google.com/)
- nanoGPT (Andrej Karpathy): [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- The Illustrated Transformer: [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- Paper “Attention Is All You Need”: [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

---

### Tổng kết

- _LLM/GPT_ = mô hình đoán token tiếp theo; ta sẽ tự dựng một GPT mini bằng PyTorch.
- Cài _Python 3.9+_, cài _PyTorch_ đúng bản (CPU hay CUDA) từ trang chính thức.
- Không có GPU? Dùng _Google Colab_ miễn phí: `Runtime -> Change runtime type -> GPU`.
- Verify bằng `torch.cuda.is_available()` và một phép _nhân tensor trên GPU_ (thấy `cuda:0`).
- Nhớ quy tắc _cùng device_ cho dữ liệu và mô hình.

Căn bếp đã sẵn sàng, nhưng bạn có thể tự hỏi: _“Nếu muốn huấn luyện mô hình lớn hơn, hay tự host ở nhà thì cần card gì, bao nhiêu VRAM?”_. Đó chính là nội dung _Bài 2: Phần cứng để huấn luyện & tự host LLM - chọn GPU, VRAM, mua card hay dùng cloud?_ - chúng ta sẽ học cách đọc thông số GPU, ước lượng VRAM theo số tham số, và cân nhắc mua card hay thuê cloud. Hẹn gặp bạn ở bài sau! 🚀
