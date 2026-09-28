# Bài 7: Mô hình bigram bằng PyTorch (embedding + logits)

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 2:** Mô hình ngôn ngữ nơ-ron đầu tiên

---

### Giới thiệu

Ở _Bài 6_, mình đã biến văn bản thô thành những con số mà máy tính hiểu được: encode từng ký tự thành số nguyên (token id), cắt dữ liệu thành _context window_ rồi gom thành _batch_ những cặp (x, y) - trong đó x là chuỗi token đầu vào và y là chuỗi token “đích” (token kế tiếp mà mô hình phải đoán). Giờ mình đã có “nguyên liệu” rồi. Câu hỏi tiếp theo là: làm sao một mô hình _nhìn vào một token và đoán ra token kế tiếp_?

Hãy tưởng tượng bạn chơi trò “đoán chữ tiếp theo” khi nhắn tin. Bạn gõ “chào bạn kh…”, điện thoại gợi ý “khỏe không”. Mô hình đơn giản NHẤT để làm việc này là _mô hình bigram_: chỉ nhìn _duy nhất 1 ký tự hiện tại_ rồi đoán ký tự kế. Giống như một cuốn “sổ tay thói quen”: sau chữ q thường là chữ u, sau t thường là h… Cuốn sổ này chính là một _bảng tra cứu_ (lookup table). Trong PyTorch, bảng tra cứu đó chính là một lớp `nn.Embedding`.

Bài này bạn sẽ xây dựng mô hình _ngôn ngữ nơ-ron ĐẦU TIÊN_ của cả khóa - nhỏ xíu nhưng đầy đủ tinh thần của một LLM: nhận token vào, chạy _forward pass_, xuất ra _logits_ (điểm số thô cho mỗi token có thể là token kế). Đây là viên gạch nền: các bài sau ta chỉ thay “ruột” bằng attention và Transformer, còn khung sườn (embedding vào -> tính toán -> logits ra) thì y hệt.

> 🔑 **Một câu để nhớ:** Mô hình bigram = một bảng `nn.Embedding` ánh xạ thẳng mỗi token id thành một vector _logits_ dài `vocab_size`, nói “khả năng thô” của từng token trở thành token kế tiếp.

---

### Bạn sẽ học được gì?

- [ ] Hiểu _embedding_ là gì và vì sao nó khác _one-hot_ (và tại sao vẫn tương đương về mặt toán học ở đây).
- [ ] Phân biệt rõ _logits_ và _xác suất_ (softmax mới ra xác suất, logits thì chưa).
- [ ] Viết một lớp `BigramLanguageModel(nn.Module)` với đúng 1 bảng `nn.Embedding`.
- [ ] Chạy _forward pass_ đầu tiên và đọc _shape tensor_ qua từng bước `(B, T) -> (B, T, vocab_size)`.
- [ ] Sinh (generate) vài ký tự từ mô hình chưa huấn luyện để thấy “nó chạy” (dù ra rác).

📌 **Yêu cầu trước:** đã làm Bài 6 (encode/decode, `vocab_size`, batch (x, y) shape `(B, T)`), biết `tensor`, `torch.nn`, và ý tưởng “đoán token kế tiếp”.

---

### 1. Embedding là gì? Analogy "cuốn từ điển tra thẳng"

Hãy tưởng tượng mỗi token id là một _số thứ tự trong danh bạ_. Một bảng `nn.Embedding` giống một _cuốn danh bạ_: đưa số thứ tự vào, nó trả về ngay _cả một hàng thông tin_ (một vector số thực) tương ứng. Không tính toán gì phức tạp - chỉ là _tra cứu theo chỉ số_ (index lookup).

**Định nghĩa chính xác:** `nn.Embedding(num_embeddings, embedding_dim)` là một _ma trận trọng số_ kích thước `(num_embeddings, embedding_dim)`. Khi bạn đưa vào một token id $i$, nó trả về _hàng thứ $i$_ của ma trận đó.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
# Suppose our vocabulary has 5 tokens, each mapped to a vector of size 3
emb = nn.Embedding(num_embeddings=5, embedding_dim=3)
print(emb.weight.shape) # torch.Size([5, 3]) -> the lookup table itself

idx = torch.tensor([0, 3, 3, 1]) # shape (4,): four token ids
out = emb(idx)                   # shape (4, 3): one row per id
print(out.shape)                 # torch.Size([4, 3])
# Note: the two id=3 map to the SAME row -> identical vectors
```

💡 Điểm mấu chốt: `nn.Embedding` _không_ nhân ma trận cồng kềnh - nó chỉ _lấy hàng theo chỉ số_. Nhưng về mặt toán học, nó _tương đương_ với việc nhân một vector _one-hot_ với ma trận trọng số.

#### Embedding vs one-hot: khác gì?

_One-hot_ biểu diễn token id $i$ bằng một vector dài `vocab_size` toàn số 0, chỉ vị trí $i$ là 1. Nhân one-hot với ma trận $W$ `(vocab_size, d)` sẽ lấy đúng hàng $i$ của $W$. Đó CHÍNH XÁC là điều embedding làm - nhưng embedding làm bằng index lookup (nhanh, không tốn bộ nhớ cho vector 0).

```text
one-hot(id=3) @ W:
[0 0 0 1 0] @ [ w0 ]     = w3   (row 3)
              [ w1 ]
              [ w2 ]
              [ w3 ]  <- picked
              [ w4 ]

nn.Embedding(id=3)  ==  W[3]   (same result, no matmul)
```

| Tiêu chí               | One-hot + matmul                | nn.Embedding                                  |
| :--------------------- | :------------------------------ | :-------------------------------------------- |
| **Kết quả**            | Hàng thứ $i$ của $W$            | Hàng thứ $i$ của $W$                          |
| **Cách làm**           | Tạo vector 0/1 rồi nhân ma trận | Tra cứu trực tiếp theo chỉ số                 |
| **Bộ nhớ**             | Tốn (vector dài `vocab_size`)   | Tiết kiệm                                     |
| **Đạo hàm (gradient)** | Vẫn học được                    | Vẫn học được (chỉ hàng được dùng có gradient) |

---

### 2. Ý tưởng bigram: bảng ánh xạ token -> logits của token kế

Mô hình bigram không quan tâm ngữ cảnh dài - nó chỉ nhìn _1 token hiện tại_ để đoán token kế. Vậy nó cần: với mỗi token id, cho ra một “bảng điểm” nói _token nào có khả năng là token kế_. Bảng điểm đó dài đúng `vocab_size`.

Mẹo hay: ta dùng luôn `nn.Embedding(vocab_size, vocab_size)`! Mỗi token id được ánh xạ thành một vector dài `vocab_size` - và ta _diễn giải vector đó chính là logits_ cho token kế tiếp. Không cần lớp Linear riêng.

```text
Token hien tai      Embedding table (vocab_size x vocab_size)      Logits (vocab_size)
   id = 'q'   -->   lookup row 'q'                          -->   [ ..., cao o cot 'u', ... ]
```

#### Logits KHÁC xác suất - đừng nhầm!

_Logits_ là những con số thực _thô, chưa chuẩn hóa_ (có thể âm, có thể lớn). Chúng CHƯA phải xác suất. Muốn ra xác suất (cộng lại bằng 1, đều không âm) ta phải cho qua _softmax_:

$$\text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

```python
logits = torch.tensor([2.0, 1.0, -1.0]) # raw scores, sum != 1
probs = F.softmax(logits, dim=-1)       # -> probabilities, sum == 1
print(probs)       # tensor([0.7054, 0.2595, 0.0351])
print(probs.sum()) # tensor(1.0000)
```

⚠️ Trong huấn luyện ta thường _không_ tự gọi softmax - hàm `F.cross_entropy` nhận thẳng _logits_ (nó tự làm softmax bên trong, ổn định số học hơn). Ta chỉ softmax khi _sinh văn bản_ (generate) để bốc token theo phân phối. Chuyện loss cross-entropy để dành _Bài 8_.

---

### 3. Viết lớp `BigramLanguageModel`

Giờ ghép lại thành một `nn.Module` hoàn chỉnh. Đầu vào `idx` có shape `(B, T)` ($B$ = batch size, $T$ = độ dài context - từ Bài 6). Forward pass chỉ là một lần tra bảng embedding.

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Lookup table: each token id -> a vector of length vocab_size (the logits)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        # idx: (B, T) tensor of token ids
        logits = self.token_embedding_table(idx) # (B, T, vocab_size)
        return logits

vocab_size = 65 # example: 65 unique characters (like tiny Shakespeare)
model = BigramLanguageModel(vocab_size)

# Fake a batch of 4 sequences, each 8 tokens long (Bai 6 style batching)
xb = torch.randint(0, vocab_size, (4, 8)) # (B=4, T=8)
logits = model(xb)
print(logits.shape) # torch.Size([4, 8, 65]) -> (B, T, vocab_size)
```

#### Đọc shape qua từng bước - “tư duy shape tensor”

```text
Input idx           : (B, T)              = (4, 8)   token ids
lookup embedding    : (B, T, vocab_size)  = (4, 8, 65)
                       |  |    |
                       |  |    +-- logits for the NEXT token (one score per possible token)
                       |  +------- position in the context window
                       +---------- which example in the batch
```

Mỗi vị trí $(b, t)$ cho ra một vector 65 chiều = logits dự đoán token kế tiếp _dựa trên token tại vị trí đó_. Vì là bigram, dự đoán ở vị trí $t$ chỉ phụ thuộc token `idx[b, t]`, _hoàn toàn bỏ qua_ các token trước đó. (Chính “trí nhớ ngắn” này là điểm yếu ta sẽ khắc phục bằng _attention_ ở Tuần 3.)

🎯 Ghi nhớ đường đi: `(B, T)` token ids -> qua `nn.Embedding` -> `(B, T, vocab_size)` logits. Đây là bộ khung mọi mô hình trong khóa đều theo, chỉ khác phần “ruột” ở giữa.

---

### 4. Forward pass đầu tiên + sinh văn bản (chưa huấn luyện)

Mô hình vừa tạo có trọng số _ngẫu nhiên_ nên nó chưa biết gì - sinh ra sẽ là rác. Nhưng chạy được là mừng rồi! Hàm `generate` lấy token cuối cùng, tính logits, softmax ra xác suất, rồi _bốc ngẫu nhiên_ (sample) một token theo phân phối đó, nối vào chuỗi, lặp lại.

```python
@torch.no_grad() # no gradients needed when just generating
def generate(model, idx, max_new_tokens, vocab_size):
    # idx: (B, T) current context
    for _ in range(max_new_tokens):
        logits = model(idx)                 # (B, T, vocab_size)
        logits = logits[:, -1, :]           # focus on the LAST time step -> (B, vocab_size)
        probs = F.softmax(logits, dim=-1)   # (B, vocab_size), sums to 1
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) sample one token
        idx = torch.cat((idx, idx_next), dim=1)            # append -> (B, T+1)
    return idx

# Start from a single token (id 0), generate 10 new tokens for 1 sequence
start = torch.zeros((1, 1), dtype=torch.long) # (B=1, T=1)
out = generate(model, start, max_new_tokens=10, vocab_size=vocab_size)
print(out.shape) # torch.Size([1, 11])  -> original 1 + 10 new
# out holds token ids; you would decode() them back to characters (Bai 6)
```

💡 Vì bigram chỉ dùng token cuối, ở đây ta lấy `logits[:, -1, :]` - phần còn lại của context bị lãng phí. Với attention sau này, MỌI vị trí trước đó mới thực sự đóng góp.

---

### Nhìn sâu hơn

_Vì sao `nn.Embedding(vocab_size, vocab_size)`?_ Vì bigram cần ánh xạ _mỗi token -> điểm cho mọi token kế_. Số hàng = số token đầu vào có thể (`vocab_size`), số cột = số token đầu ra có thể (`vocab_size`). Bảng này có `vocab_size * vocab_size` tham số. Với `vocab_size=65`, đó là $65 \times 65 = 4225$ tham số - siêu nhỏ (một GPT thật có hàng triệu tới hàng tỉ).

_Gotcha số học:_ đừng tự softmax rồi lại đưa vào `cross_entropy` - `F.cross_entropy` _muốn logits thô_, tự làm `log_softmax` bên trong ổn định hơn (tránh tràn số khi exp giá trị lớn). Nếu bạn softmax hai lần, loss sẽ sai.

_Phân biệt kỹ:_ ở mô hình lớn sau này, bảng embedding token (`d_model` chiều) và lớp “chiếu ra logits” (`d_model -> vocab_size`) là _hai lớp riêng biệt_ (thậm chí có thể chia sẻ trọng số - _weight tying_). Ở bigram ta gộp làm một chỉ vì `embedding_dim` tình cờ bằng `vocab_size` và ta diễn giải luôn vector đó là logits.

---

### Thực hành

- **[Dễ]** Tạo `nn.Embedding(10, 4)`, đưa vào `torch.tensor([2, 2, 7])`. In shape output. Kiểm chứng hai phần tử `id=2` cho ra vector _giống hệt_ nhau.
- **[Trung bình]** Với `xb` shape `(4, 8)` và `vocab_size=65`, cho qua model rồi lấy `logits[:, -1, :]`. Shape là gì? Áp `F.softmax(..., dim=-1)` và kiểm tra tổng mỗi hàng xấp xỉ 1.0.
- **[Thử thách]** Viết lại forward pass _không dùng_ `nn.Embedding`: tạo `W = torch.randn(vocab_size, vocab_size)`, chuyển `idx` thành one-hot bằng `F.one_hot(idx, vocab_size).float()`, rồi nhân `@ W`. So sánh shape kết quả với bản dùng `nn.Embedding` - chứng minh chúng tương đương.

---

### Hiểu lầm thường gặp

- ❌ “Logits chính là xác suất.” ➔ ✅ Logits là điểm _thô, chưa chuẩn hóa_; phải qua _softmax_ mới thành xác suất (không âm, cộng lại bằng 1).
- ❌ “Embedding luôn phải nhỏ chiều hơn one-hot.” ➔ ✅ Thường thì có, nhưng ở bigram ta cố ý dùng `embedding_dim = vocab_size` để vector chính là logits.
- ❌ “Bigram nhìn cả câu để đoán.” ➔ ✅ Bigram chỉ nhìn _đúng 1 token hiện tại_; các token trước bị bỏ qua (đó là lý do nó yếu, cần attention).
- ❌ “Phải tự softmax rồi mới tính loss.” ➔ ✅ `F.cross_entropy` nhận _logits thô_ và tự softmax bên trong - tự softmax trước là sai.

---

### Mẹo ghi nhớ

- _Embedding = tra bảng theo chỉ số_ (danh bạ), tương đương one-hot @ W nhưng nhanh hơn.
- _Logits -> softmax -> xác suất_: nhớ đúng thứ tự này.
- Đường đi shape bất di bất dịch: `(B, T) --embedding--> (B, T, vocab_size)`.
- 🧠 Mẹo: “Bi-gram = Nhìn-_MỘT_ đoán một”. Chỉ 1 token vào, 1 phân phối ra.

---

### Tóm tắt trực quan

| Khái niệm              | Một dòng                                                         |
| :--------------------- | :--------------------------------------------------------------- |
| **nn.Embedding(n, d)** | Bảng tra `(n, d)`; đưa id ra hàng thứ id đó                      |
| **Bigram model**       | `nn.Embedding(vocab_size, vocab_size)`; token -> logits token kế |
| **Logits**             | Điểm thô, chưa chuẩn hóa, có thể âm                              |
| **Softmax**            | Biến logits thành xác suất (cộng = 1)                            |
| **Forward shape**      | `(B, T) -> (B, T, vocab_size)`                                   |
| **Generate**           | Lấy logits vị trí cuối -> softmax -> multinomial -> nối          |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Let’s build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- `nn.Embedding` (PyTorch docs): [https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- PyTorch tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT (Andrej Karpathy) - model.py khởi đầu chính là bigram: [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Attention Is All You Need (paper gốc): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

---

### Tổng kết

- Mô hình _bigram_ là LLM đơn giản nhất: một bảng `nn.Embedding(vocab_size, vocab_size)` ánh xạ token hiện tại thành _logits_ cho token kế tiếp.
- _Embedding = tra cứu theo chỉ số_, tương đương one-hot nhân ma trận nhưng gọn nhẹ hơn.
- _Logits chưa phải xác suất_ - cần softmax; nhưng khi tính loss thì đưa logits thô vào `cross_entropy`.
- Forward pass đầu tiên biến `(B, T)` thành `(B, T, vocab_size)` - bộ khung mọi mô hình sau đều theo.
- Điểm yếu: bigram chỉ nhìn 1 token, không có “trí nhớ” ngữ cảnh. Ta sẽ khắc phục bằng attention.

Mô hình đã chạy được và sinh ra… rác - vì nó chưa được học! Ở _Bài 8: Hàm mất mát cross-entropy và vòng lặp huấn luyện đầu tiên_, mình sẽ dạy mô hình này thực sự đoán đúng: đo sai số bằng _cross-entropy_, dùng _AdamW_ để cập nhật bảng embedding, và xem loss giảm dần cho tới khi văn bản sinh ra bắt đầu “có vẻ giống thật”. Hẹn gặp ở bài kế! 🤖
