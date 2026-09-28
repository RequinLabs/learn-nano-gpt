# Bài 6: Chuẩn bị dữ liệu - encode văn bản, context window và batch

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 2:** Mô hình ngôn ngữ nơ-ron đầu tiên

---

### Giới thiệu

Ở _Bài 5_, bạn đã tự tay viết một _character-level tokenizer_: từ một chuỗi ký tự, bạn dựng được vocab, hai bảng tra `stoi` / `itos`, và hai hàm `encode` / `decode` để biến _chữ thành số_ rồi trả về được. Nhưng lúc đó ta mới encode vài câu ngắn để nghịch cho vui. Câu hỏi bây giờ là: làm sao _nuôi_ cả một cuốn sách vào mô hình để nó học? 🤖

Hãy tưởng tượng bạn học một bản nhạc dài. Bạn không thể nhìn cả bản một lúc rồi nhớ ngay - bạn _cắt bản nhạc thành từng ô nhịp ngắn_, tập từng ô, lặp lại nhiều lần. Mô hình ngôn ngữ cũng vậy: ta không nhét cả cuốn sách vào một lần (không đủ bộ nhớ, và mô hình cũng chỉ “nhìn” được một cửa sổ hữu hạn). Thay vào đó, ta _encode toàn bộ dataset thành một tensor số nguyên khổng lồ_, rồi cắt nó thành những _cửa sổ ngắn_ (context window) và gom nhiều cửa sổ thành một _batch_ để học song song. Đó chính là toàn bộ nội dung bài hôm nay.

> 🔑 **Một câu để nhớ:** Huấn luyện LLM = lấy vô số cặp (x, y) từ văn bản, trong đó x là một đoạn `block_size` token và y là _chính đoạn đó dịch sang phải 1 vị trí_ - mô hình học đoán token kế tiếp tại mỗi vị trí.

---

### Bạn sẽ học được gì?

- [ ] Encode _toàn bộ_ dataset thành một `torch.tensor` kiểu long (int64) duy nhất.
- [ ] Hiểu _context window_ (`block_size`): mô hình nhìn tối đa bao nhiêu token cùng lúc, và vì sao có giới hạn.
- [ ] Tạo cặp (x, y) cho next-token prediction: y là x dịch phải 1 vị trí.
- [ ] Gom nhiều cửa sổ thành _batch_ với shape (B, T) bằng chỉ số ngẫu nhiên.
- [ ] _Chia train/val_ để phát hiện overfitting một cách trung thực.

📌 **Yêu cầu trước:** đã có `encode` / `decode` và `vocab_size` từ Bài 5; biết `torch.tensor`, indexing, và `torch.randint`.

---

### 1. Encode cả dataset thành MỘT tensor

_Analogy:_ trước khi vào bếp, đầu bếp sơ chế _tất cả_ nguyên liệu một lần rồi để sẵn - đến lúc nấu chỉ việc bốc. Ta cũng encode _một lần_ cả dataset thành số, để vòng lặp huấn luyện sau này chỉ việc “bốc” các lát cắt cực nhanh.

```python
import torch

# Suppose we read a text corpus (e.g. tiny Shakespeare)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# From Bài 5: build the vocabulary and encode function
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # str -> list[int]
decode = lambda l: ''.join(itos[i] for i in l) # list[int] -> str

# Encode the WHOLE dataset once into a single 1D tensor of token ids
data = torch.tensor(encode(text), dtype=torch.long) # shape: (N,), N = number of chars
print(data.shape, data.dtype) # e.g. torch.Size([1115394]) torch.int64
print(data[:20]) # first 20 token ids, e.g. tensor([18, 47, 56, ...])
```

⚠️ Vì sao _`dtype=torch.long` (int64)_? Vì đây là _token id_ - chỉ số nguyên để tra bảng embedding sau này. Lớp `nn.Embedding` yêu cầu chỉ số kiểu số nguyên. Đừng nhầm với `float`: token id _không phải_ một giá trị số học để cộng/trừ, nó chỉ là “địa chỉ ô” trong từ điển.

`data` bây giờ là một dãy số nguyên dài N phần tử. Toàn bộ tri thức văn bản đã nằm gọn trong một tensor 1 chiều.

---

### 2. Context window (block_size): mô hình nhìn được bao xa?

_Analogy:_ khi đoán chữ tiếp theo lúc nhắn tin, bạn dựa vào _vài chữ trước đó_, chứ không đọc lại cả cuộc trò chuyện từ đầu. _Context window_ (`block_size`, còn gọi context length) là số token tối đa mô hình được phép nhìn để đoán token kế tiếp.

```python
block_size = 8 # maximum context length (a.k.a. context window / T)
x = data[:block_size] # the context:      tokens [0 .. 7]
y = data[1:block_size + 1] # the targets:      tokens [1 .. 8]  (x shifted right by 1)

for t in range(block_size):
    context = x[:t + 1] # everything up to and including position t
    target = y[t]       # the token the model should predict next
    print(f"khi thay {context.tolist()} -> du doan {target.item()}")

# One block of length 8 secretly contains 8 training examples:
#   [x0]                -> y0
#   [x0,x1]             -> y1
#   ...
#   [x0,...,x7]         -> y7
```

Điều tuyệt vời: _một cửa sổ dài `block_size` chứa sẵn `block_size` ví dụ huấn luyện_ (dự đoán ở _mỗi_ vị trí), nhờ vào _causal mask_ mà bạn sẽ gặp ở các bài attention. Mô hình học đoán từ ngữ cảnh ngắn (1 token) đến ngữ cảnh đầy đủ (`block_size` token) cùng lúc.

```text
Encode toàn bộ text -> data (N token id)
                        |
         cắt lát ngẫu nhiên dài block_size+1
                        |
        x = data[i : i+T]      (context, T token)
        y = data[i+1 : i+T+1]  (target = x dịch phải 1)
                        |
          gom B lát cắt -> batch (B, T)
                        |
             -> mô hình -> logits (B, T, vocab_size)
```

🎯 `block_size` càng lớn, mô hình càng “nhớ” xa - nhưng chi phí attention tăng theo _bình phương_ độ dài ($O(T^2)$). Vì thế GPT mini để `block_size` nhỏ (vd 8, 32, 256); các LLM lớn mới có context vài nghìn đến vài triệu token.

---

### 3. Cặp (x, y) và ý nghĩa "dịch phải 1"

Nhiệm vụ huấn luyện là _next-token prediction_: tại mỗi vị trí $t$, cho ngữ cảnh $x[:t+1]$, hãy đoán token $y[t]$. Vì $y$ chỉ là $x$ _dịch phải đúng 1 vị trí_, mục tiêu ở vị trí $t$ chính là _token thật ngay sau_ ngữ cảnh đó. Đây là “nhãn miễn phí”: ta không cần con người gán nhãn, chính văn bản tự làm nhãn cho nó - đó là lý do gọi là _self-supervised learning_.

| Khái niệm               | Ý nghĩa                   | Shape ví dụ          |
| :---------------------- | :------------------------ | :------------------- |
| `data`                  | cả dataset đã encode      | `(N,)`               |
| `block_size` (T)        | độ dài context tối đa     | vô hướng, vd 8       |
| `x`                     | context (đầu vào)         | `(B, T)`             |
| `y`                     | target = x dịch phải 1    | `(B, T)`             |
| `logits` (mô hình xuất) | điểm thô cho mỗi token kế | `(B, T, vocab_size)` |

💡 Phân biệt: `logits` _không phải_ xác suất. `logits` là điểm số thực thô; phải qua `softmax` mới thành phân phối xác suất (tổng = 1). Cross-entropy sẽ so `logits` với `y` để tính mất mát.

---

### 4. Tạo batch: gom nhiều cửa sổ để học song song

_Analogy:_ thay vì chấm từng bài kiểm tra một, giáo viên chấm _một chồng_ bài cùng lúc cho nhanh. GPU cũng thích xử lý _cả batch_ cùng lúc thay vì từng ví dụ lẻ. Ta chọn $B$ (batch size) vị trí bắt đầu _ngẫu nhiên_ trong `data`, cắt $B$ lát cắt dài `block_size`, xếp chồng thành tensor $(B, T)$.

```python
torch.manual_seed(1337)
batch_size = 4 # B: how many independent sequences per batch
block_size = 8 # T: context length

def get_batch(data):
    # Pick B random start positions such that i+block_size+1 is still in range
    ix = torch.randint(len(data) - block_size, (batch_size,)) # shape: (B,)
    x = torch.stack([data[i:i + block_size] for i in ix])       # shape: (B, T)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]) # shape: (B, T)
    return x, y

xb, yb = get_batch(data)
print(xb.shape, yb.shape) # torch.Size([4, 8]) torch.Size([4, 8])
print(xb) # each row is one context window of 8 token ids
print(yb) # yb[b, t] is the target for xb[b, :t+1]
```

Sơ đồ shape khi một batch chạy qua toàn bộ pipeline:

```text
ix          : (B,)                <- B chi so bat dau ngau nhien
xb          : (B, T)              <- B cua so context, moi cua so T token
                v  qua nn.Embedding
tok_emb     : (B, T, C)           <- C = so chieu embedding
                v  qua cac Transformer block + lop tuyen tinh cuoi
logits      : (B, T, vocab_size)  <- diem cho token ke tiep tai MOI vi tri
                v  so voi yb (B, T) bang cross-entropy
loss        : vo huong (scalar)
```

Với $B=4, T=8$, một batch có $4 \times 8 = 32$ cặp (context -> target) được học _song song_. Nhân $B$ lên nữa và GPU vẫn xử lý mượt - đó là lý do batch giúp huấn luyện nhanh và ổn định hơn.

---

### 5. Chia train/val: đo lường trung thực

_Analogy:_ ôn thi mà chỉ luyện đúng đề đã biết đáp án thì điểm ảo. Bạn cần _đề lạ_ để biết mình _thật sự_ hiểu. Ta giữ lại một phần dữ liệu (validation) mà mô hình _không được huấn luyện trên đó_, chỉ dùng để đo.

```python
# Simple contiguous split: first 90% train, last 10% validation
n = int(0.9 * len(data))
train_data = data[:n] # shape: (0.9*N,)
val_data = data[n:]   # shape: (0.1*N,)

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])       # (B, T)
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix]) # (B, T)
    return x, y

xb, yb = get_batch('train')
xv, yv = get_batch('val')
```

⚠️ Nếu train loss giảm mà val loss _tăng lên_ -> mô hình đang _overfitting_ (học thuộc train, kém trên dữ liệu mới). Val split chính là chiếc còi báo động đó. Ta cắt _liền khối_ (train trước, val sau) thay vì trộn ngẫu nhiên vì văn bản có tính liên tục theo thời gian, không nên để mô hình “nhìn trộm” đoạn kề của val.

---

### Nhìn sâu hơn

- **Vì sao encode toàn bộ một lần rồi cắt, thay vì đọc từng đoạn từ file?** Vì `torch.randint` + slicing trên tensor trong RAM cực nhanh, không tốn I/O đĩa mỗi bước. Với dataset khổng lồ không nhét hết RAM, người ta dùng `np.memmap` (ánh xạ file, đọc theo trang) hoặc streaming - nhưng _ý tưởng_ vẫn là “cắt lát ngẫu nhiên các cửa sổ dài `block_size`”.
- **Batch size và learning rate đi đôi:** batch lớn cho ước lượng gradient ít nhiễu hơn (mượt hơn) nhưng tốn VRAM hơn; batch quá nhỏ thì nhiễu, hội tụ chậm. Đây là siêu tham số cần thử.
- **`block_size` là _tối đa_, không phải cố định lúc suy luận:** khi sinh văn bản, nếu chuỗi vượt `block_size`, ta phải _cắt bớt phần đầu_ (crop) để không vượt context length mà mô hình từng thấy - nếu không, positional embedding sẽ thiếu vị trí. Bạn sẽ gặp lại điều này ở các bài generate.

---

### Thực hành

- **[Dễ]** In `xb`, `yb` với `batch_size=2`, `block_size=4`. Với `xb[0]`, viết ra 4 cặp (context, target) bằng tay và đối chiếu với `yb[0]`.
- **[Trung bình]** Đổi tỉ lệ chia thành 80/20. In `len(train_data)` và `len(val_data)`. Vì sao lấy lát cắt `data[i:i+block_size]` phải giới hạn $i < \text{len}(data) - block\_size$? (Gợi ý: nghĩ về việc lấy `data[i+block_size]` cho `y`.)
- **[Thử thách]** Viết hàm `estimate_loss()` chạy `get_batch` nhiều lần cho cả `'train'` và `'val'`, đặt mô hình ở `torch.no_grad()`, và trả về mất mát trung bình mỗi split. (Sẽ dùng lại ở bài huấn luyện - nó cho con số val loss ổn định, ít nhiễu.)

---

### Hiểu lầm thường gặp

- ❌ “`y` là câu trả lời do người gán nhãn.” ➔ ✅ `y` chỉ là `x` _dịch phải 1_ - nhãn tự sinh từ văn bản (self-supervised), không cần con người.
- ❌ “Một cửa sổ `block_size=8` chỉ cho 1 ví dụ huấn luyện.” ➔ ✅ Nó cho _8 ví dụ_ (dự đoán ở mỗi vị trí $t$), nhờ causal mask.
- ❌ “Token id nên để kiểu `float`.” ➔ ✅ Token id phải là _số nguyên_ (`torch.long`) vì đó là chỉ số tra bảng `nn.Embedding`, không phải giá trị số học.
- ❌ “`block_size` càng lớn thì luôn tốt.” ➔ ✅ Chi phí attention tăng theo $O(T^2)$ bộ nhớ/thời gian; phải cân bằng với ngân sách phần cứng.
- ❌ “Nên trộn ngẫu nhiên train/val như dữ liệu bảng.” ➔ ✅ Với văn bản liên tục, cắt _liền khối_ để val thực sự “lạ”, tránh rò rỉ ngữ cảnh kề.

---

### Mẹo ghi nhớ

- **“Encode một lần, cắt vô số lần”** - dataset thành 1 tensor `long`, rồi bốc lát cắt ngẫu nhiên.
- **“y = x dịch phải 1”** - nhớ đúng một câu này là hiểu next-token prediction.
- **Shape thần chú:** `x, y : (B, T)` -> `logits : (B, T, vocab_size)`.
- Mẹo nhớ giới hạn chỉ số: cần `data[i+block_size]` cho `y`, nên $i$ chỉ chạy tới `len(data) - block_size`.

---

### Tóm tắt trực quan

| Khái niệm           | Một dòng để nhớ                                         |
| :------------------ | :------------------------------------------------------ |
| **data tensor**     | cả dataset đã encode, kiểu `torch.long`, shape `(N,)`   |
| **block_size (T)**  | context window - số token tối đa mô hình nhìn cùng lúc  |
| **cặp (x, y)**      | `y` là `x` dịch phải 1 -> mục tiêu next-token           |
| **batch (B, T)**    | $B$ cửa sổ ngẫu nhiên xếp chồng để học song song        |
| **train/val split** | val (không huấn luyện) để phát hiện overfitting         |
| **shape logits**    | `(B, T, vocab_size)` - điểm cho token kế tại mỗi vị trí |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Let’s build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- PyTorch docs (tensor, `torch.randint`, `torch.stack`): [pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT của Andrej Karpathy (xem prepare dữ liệu + `get_batch`): [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Paper “Attention Is All You Need”: [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer (trực giác input/embedding): [jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

---

### Tổng kết

- Ta _encode toàn bộ dataset_ thành một tensor số nguyên long duy nhất - làm một lần, dùng mãi.
- _Context window_ (`block_size`) giới hạn số token mô hình nhìn cùng lúc; một cửa sổ dài $T$ chứa $T$ ví dụ huấn luyện.
- Cặp (x, y) với _y = x dịch phải 1_ chính là bài toán next-token prediction, nhãn tự sinh từ văn bản.
- _Batch_ $(B, T)$ gom nhiều cửa sổ ngẫu nhiên để GPU học song song; _train/val split_ giúp đo lường trung thực và bắt overfitting.

👉 **Tiếp theo - Bài 7: “Mô hình bigram bằng PyTorch (embedding + logits)”**: ta sẽ dùng chính `get_batch` vừa viết để nạp batch $(B, T)$ vào mô hình _đơn giản nhất có thể_ - một lớp `nn.Embedding` biến mỗi token id thành `logits`, rồi tính loss bằng cross-entropy. Đây là “Hello World” của mô hình ngôn ngữ, viên gạch đầu tiên trước khi ta thêm attention. Hẹn gặp ở bài sau! 🚀
