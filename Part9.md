# Bài 9: Sinh văn bản (sampling) từ mô hình

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 2:** Mô hình ngôn ngữ nơ-ron đầu tiên

---

### Giới thiệu

Ở _Bài 8_, chúng ta đã dạy cho mô hình bigram của mình biết “đoán chữ tiếp theo”: nó nhận vào một chuỗi token, đưa ra `logits`, rồi ta dùng hàm mất mát cross-entropy so với token đúng để cập nhật trọng số bằng vòng lặp huấn luyện. Nhưng huấn luyện xong rồi thì… nó viết được gì chưa? 🤖

Hãy tưởng tượng bạn đang nhắn tin và bàn phím điện thoại gợi ý từ tiếp theo. Bạn gõ “Chúc mừng sinh…” và nó gợi ý “nhật”. Bàn phím không tự viết cả câu một lúc - nó _đoán một token, chèn vào, rồi đoán token kế tiếp dựa trên tất cả những gì đã có_. Cứ lặp lại như vậy là ra cả câu. Đó chính xác là cách một LLM sinh văn bản: _tự hồi quy (autoregressive)_.

Trong bài này, ta biến “đầu ra thô” của mô hình (logits) thành một token thật sự để viết ra, rồi lặp lại quá trình đó. Đây là lần đầu tiên bạn được “nghe” mô hình mình tự tay huấn luyện nói chuyện.

> 🔑 **Một câu để nhớ:** Sinh văn bản = biến logits -> xác suất bằng softmax -> _lấy mẫu_ ra một token -> nối vào chuỗi -> lặp lại.

---

### Bạn sẽ học được gì?

- [ ] Phân biệt rõ logits và _xác suất_, và vì sao cần softmax ở giữa.
- [ ] Hiểu “lấy mẫu (sampling)” là gì và khác gì với “lấy argmax tham lam”.
- [ ] Viết được vòng lặp `generate()` tự hồi quy đúng cú pháp PyTorch.
- [ ] Đọc và suy luận được shape của tensor qua từng bước sinh.
- [ ] Thấy tận mắt mô hình bigram sinh ra chuỗi ký tự như thế nào.

📌 **Yêu cầu trước:** Đã hiểu bigram + logits + cross-entropy (Bài 7-8), biết `torch`, `torch.nn.functional as F`, và thao tác tensor cơ bản.

---

### 1. Từ logits tới xác suất: vai trò của softmax

Sau bước forward, mô hình đưa ra `logits` - một vector số thực _chưa chuẩn hóa_, mỗi phần tử ứng với một token trong bộ từ vựng. Ví dụ với `vocab_size = 5`, logits có thể là `[2.0, 1.0, -1.0, 0.5, 3.0]`. Con số này _không phải xác suất_: nó có thể âm, có thể lớn tùy ý, và tổng không bằng 1.

**Analogy:** hãy coi logits như “số phiếu bầu thô” cho mỗi ứng viên. Muốn biết “tỷ lệ ủng hộ” (xác suất), ta phải chuẩn hóa cho tổng bằng 100%. `softmax` làm đúng việc đó, và nó dùng hàm mũ để mọi giá trị thành dương rồi chia cho tổng:

$$\text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

💡 Dùng `exp` (không chỉ chia thẳng) vì logits có thể âm; `exp` biến mọi số thành dương và _khuếch đại chênh lệch_ - logit lớn hơn một chút sẽ có xác suất cao hơn hẳn.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, -1.0, 0.5, 3.0]) # shape: (5,) - one score per token
probs = F.softmax(logits, dim=-1)                 # shape: (5,) - now sums to 1
print(probs)       # tensor([0.2312, 0.0851, 0.0115, 0.0516, 0.6285])
print(probs.sum()) # tensor(1.0000) -> valid probability distribution
```

Token cuối (logit 3.0) chiếm ~63% xác suất - nó “được ưu ái” nhất, nhưng các token khác vẫn có cơ hội. Ghi nhớ: _logits là đầu ra thô của mạng; xác suất là logits đã đi qua softmax_. Đừng nhầm hai khái niệm này.

---

### 2. Lấy mẫu (sampling) vs. argmax tham lam

Giờ ta có phân phối xác suất. Chọn token nào để viết ra? Có hai cách:

| Cách                           | Mô tả                                    | Tính chất                                        |
| :----------------------------- | :--------------------------------------- | :----------------------------------------------- |
| **argmax (tham lam - greedy)** | Luôn chọn token có xác suất cao nhất     | Kết quả _cố định_, lặp lại, dễ nhàm/kẹt vòng lặp |
| **Sampling (lấy mẫu)**         | Rút ngẫu nhiên _theo_ phân phối xác suất | Đa dạng, tự nhiên hơn; mỗi lần chạy khác nhau    |

**Analogy:** bạn có một túi bi với 63 viên đỏ, 23 viên xanh… (đúng theo tỷ lệ xác suất). `argmax` = luôn nhặt màu nhiều nhất. `Sampling` = nhắm mắt bốc một viên - màu nhiều thì dễ trúng hơn, nhưng màu ít vẫn có khả năng.

```python
torch.manual_seed(1337) # for reproducible sampling
probs = F.softmax(logits, dim=-1) # shape: (5,)
greedy = torch.argmax(probs)      # shape: () - always index 4
sampled = torch.multinomial(probs, num_samples=1) # shape: (1,) - draws by probability
print(greedy, sampled) # tensor(4) tensor([4]) # here both pick 4, but sampling could pick others
```

`torch.multinomial(probs, 1)` rút một chỉ số (index) theo đúng phân phối `probs`. Đây là trái tim của việc sinh văn bản: nó cho phép mô hình “sáng tạo” thay vì chỉ đọc lại một đường.

🎯 Sau này (Bài về temperature / top-k / top-p) ta sẽ _điều khiển_ độ ngẫu nhiên này. Hôm nay chỉ cần nắm sampling cơ bản.

---

### 3. Sinh tự hồi quy: nối token rồi lặp lại

Tự hồi quy nghĩa là: token vừa sinh ra trở thành _đầu vào_ cho bước tiếp theo. Sơ đồ luồng một bước sinh:

```text
idx (chuỗi hiện tại)
   |  shape: (B, T)         B = batch, T = số token đang có
   v
model(idx) -> logits        shape: (B, T, vocab_size)
   |
   |  lấy bước thời gian CUỐI (dự đoán token kế tiếp)
   v
logits[:, -1, :]            shape: (B, vocab_size)
   |
   v  F.softmax(dim=-1)
probs                       shape: (B, vocab_size)
   |
   v  torch.multinomial
idx_next                    shape: (B, 1)
   |
   v  torch.cat([idx, idx_next], dim=1)
idx (đã dài thêm 1)         shape: (B, T+1)  -> quay lại đầu vòng lặp
```

Với mô hình _bigram_, dự đoán token kế tiếp chỉ phụ thuộc token cuối cùng, nhưng ta vẫn viết vòng lặp tổng quát để dùng lại cho các mô hình mạnh hơn về sau (Transformer nhìn cả ngữ cảnh).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) integers -> logits: (B, T, vocab_size)
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad() # no gradients needed during generation -> saves memory/time
    def generate(self, idx, max_new_tokens):
        # idx: (B, T) -> the current context
        for _ in range(max_new_tokens):
            logits, _ = self(idx)             # logits: (B, T, vocab_size)
            logits = logits[:, -1, :]         # focus on last step -> (B, vocab_size)
            probs = F.softmax(logits, dim=-1) # -> (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1) # -> (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)            # append -> (B, T+1)
        return idx
```

⚠️ _Đừng quên_ `logits[:, -1, :]`: model trả logits cho MỌI vị trí trong chuỗi (shape `(B, T, vocab_size)`), nhưng để đoán token _kế tiếp_ ta chỉ cần dự đoán từ vị trí _cuối cùng_. Lấy nhầm vị trí là sai ngay.

---

### 4. Chạy thử: nghe mô hình bigram nói

Ta khởi tạo mô hình (giả sử `vocab_size = 65` như bộ ký tự tiny Shakespeare), cho một token khởi đầu là số 0, rồi sinh 20 token:

```python
torch.manual_seed(1337)
vocab_size = 65
model = BigramLanguageModel(vocab_size)

# start with a single token (index 0) in one sequence
context = torch.zeros((1, 1), dtype=torch.long) # shape: (B=1, T=1)
out = model.generate(context, max_new_tokens=20) # shape: (1, 21)
print(out.shape)          # torch.Size([1, 21])
print(out[0].tolist())    # e.g. [0, 47, 12, 5, 60, 31, ...] -> integer token ids
```

Mỗi số là một `token id`. Muốn ra chữ, ta cần _decode_ ids thành ký tự (dùng bảng ánh xạ `itos` ta đã dựng ở bài tokenizer):

```python
# itos: dict mapping id -> character, built earlier
decode = lambda ids: ''.join(itos[i] for i in ids)
print(decode(out[0].tolist()))
# Because the model is UNTRAINED, output looks like random gibberish:
# "\nSr?f;wKq.Zx' Yj..."
```

💡 Nếu mô hình _chưa huấn luyện_, kết quả là chuỗi ký tự ngẫu nhiên như “gõ bàn phím lung tung”. Sau khi huấn luyện (Bài 8), bigram bắt đầu ra những cụm ký tự “đỡ vô nghĩa hơn” - có khoảng trắng, có vài âm tiết giống tiếng - dù vẫn còn rất yếu. Đó là lý do Bài 10 sẽ mổ xẻ vì sao bigram còn kém.

---

### Nhìn sâu hơn

- **Vì sao `@torch.no_grad()` khi generate?** Lúc sinh văn bản ta chỉ chạy forward, không cập nhật trọng số, nên không cần lưu đồ thị tính toán cho backprop. Tắt gradient giúp _tiết kiệm bộ nhớ và chạy nhanh hơn_. Đây là thói quen bắt buộc cho inference.
- **Chi phí tính toán:** mỗi bước sinh phải chạy lại forward trên toàn chuỗi `(B, T)`. Chuỗi càng dài, mỗi bước càng tốn. Với bigram thì rẻ (chỉ tra bảng embedding), nhưng với Transformer chi phí tăng theo $T$ - đó là lý do sau này ta học _KV-cache_ để không tính lại phần cũ.
- **Sampling có thể lặp/kẹt:** dùng argmax thuần dễ khiến mô hình kẹt trong vòng lặp ký tự (“the the the…”). Sampling phá vòng lặp nhờ ngẫu nhiên, nhưng nếu quá ngẫu nhiên thì văn bản mất mạch. Cân bằng này chính là bài temperature / top-k sắp tới.
- **`multinomial` cần xác suất hợp lệ:** truyền logits thẳng vào `torch.multinomial` là SAI - nó cần input là phân phối không âm, tổng dương. Luôn softmax trước.

---

### Thực hành

1. **[Dễ]** Với `logits = torch.tensor([0.0, 0.0, 4.0])`, tính tay rồi kiểm bằng `F.softmax(logits, dim=-1)`. Token nào có xác suất cao nhất, khoảng bao nhiêu?
2. **[Trung bình]** Trong `generate()`, đổi `torch.multinomial(probs, 1)` thành `torch.argmax(probs, dim=-1, keepdim=True)`. Chạy nhiều lần: kết quả có còn thay đổi không? Vì sao?
3. **[Thử thách]** Thêm tham số nhiệt độ: chia `logits = logits[:, -1, :] / temperature` trước khi softmax. Thử `temperature = 0.5` và `temperature = 2.0`, quan sát độ đa dạng của chuỗi sinh ra và giải thích theo công thức softmax.

---

### Hiểu lầm thường gặp

- ❌ “Có thể đưa thẳng logits vào `torch.multinomial` để lấy mẫu.” ➔ ✅ Phải softmax trước - multinomial cần _xác suất_ (không âm, tổng dương), không phải logits thô.
- ❌ “Sinh văn bản là chọn token xác suất cao nhất (argmax).” ➔ ✅ Đó chỉ là _greedy_; sinh văn bản thường _lấy mẫu_ theo phân phối để có sự đa dạng tự nhiên.
- ❌ “Mô hình sinh cả câu cùng một lúc.” ➔ ✅ Nó sinh _từng token một_, tự hồi quy: token mới nối vào rồi làm đầu vào cho bước sau.
- ❌ “`logits` chính là xác suất.” ➔ ✅ `logits` là điểm số thô chưa chuẩn hóa; xác suất là kết quả sau `softmax`.

---

### Mẹo ghi nhớ

- Nhớ chuỗi 5 bước sinh: _forward -> lấy bước cuối -> softmax -> multinomial -> cat_.
- Shape đi kèm: `(B, T) -> (B, T, V) -> (B, V) -> (B, V) -> (B, 1) -> (B, T+1)`.
- Mẹo: coi `generate()` như “gợi ý từ tiếp theo” của bàn phím điện thoại - đoán một chữ, chèn vào, đoán tiếp.

---

### Tóm tắt trực quan

| Khái niệm             | Một dòng để nhớ                               |
| :-------------------- | :-------------------------------------------- |
| **logits**            | Điểm số thô, chưa chuẩn hóa, mỗi token một số |
| **softmax**           | Biến logits thành xác suất (dương, tổng = 1)  |
| **argmax (greedy)**   | Luôn chọn token cao nhất -> cố định, dễ lặp   |
| **torch.multinomial** | Lấy mẫu một token theo phân phối xác suất     |
| **Tự hồi quy**        | Nối token vừa sinh vào chuỗi rồi lặp lại      |
| **@torch.no_grad()**  | Tắt gradient khi sinh -> nhanh + nhẹ bộ nhớ   |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Let’s build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- PyTorch `torch.multinomial`: [https://pytorch.org/docs/stable/generated/torch.multinomial.html](https://pytorch.org/docs/stable/generated/torch.multinomial.html)
- PyTorch `torch.nn.functional.softmax`: [https://pytorch.org/docs/stable/nn.functional.html](https://pytorch.org/docs/stable/nn.functional.html)
- nanoGPT (Andrej Karpathy): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

---

### Tổng kết

- Sinh văn bản = _logits -> softmax -> lấy mẫu (multinomial) -> nối -> lặp_ (tự hồi quy).
- `softmax` biến điểm số thô thành xác suất hợp lệ; đừng nhầm logits với xác suất.
- Sampling cho đa dạng tự nhiên, khác với argmax cố định; luôn softmax trước khi multinomial.
- Vòng lặp `generate()` chỉ dùng logits ở _bước cuối_ để đoán token kế tiếp, dùng `@torch.no_grad()` cho gọn nhẹ.
- Bigram chưa huấn luyện sinh ra chuỗi ngẫu nhiên; huấn luyện xong đỡ hơn nhưng vẫn yếu.

Ở _Bài 10: Theo dõi loss, train/val và vì sao mô hình bigram còn yếu_, ta sẽ nhìn vào đường loss trong huấn luyện, tách tập train/val để phát hiện overfitting, và mổ xẻ đúng lý do vì sao chỉ nhìn _một_ token trước là chưa đủ để mô hình “hiểu” ngữ cảnh - mở đường cho attention ở tuần sau. Hẹn gặp lại! 🚀
