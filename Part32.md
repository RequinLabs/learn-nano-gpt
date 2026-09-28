# Bài 32: Dùng tokenizer thực tế (tiktoken / Hugging Face tokenizers)

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 7:** Tokenization thực tế, tối ưu & tự host

---

### Giới thiệu

Ở _Bài 31_ bạn đã tự tay dựng thuật toán _Byte-Pair Encoding (BPE)_: bắt đầu từ ký tự, đếm cặp xuất hiện nhiều nhất, gộp dần thành các “mảnh từ” (subword). Bạn hiểu _vì sao_ GPT không dùng từ nguyên vẹn cũng không dùng từng ký tự, mà dùng một thứ ở giữa. Tuyệt vời! Nhưng bản BPE tự viết của chúng ta chạy chậm và chỉ để học nguyên lý.

Hãy tưởng tượng bạn đã tự mài được một con dao đá để hiểu “dao là gì” - giờ là lúc lấy con _dao thép công nghiệp_ ra dùng thật. Đó chính là các thư viện tokenizer đã được tối ưu: _tiktoken_ (tokenizer chính hãng của OpenAI/GPT) và _Hugging Face tokenizers_ (dùng cho hàng nghìn mô hình). Chúng làm đúng việc BPE bạn vừa viết, nhưng nhanh hơn hàng trăm lần và đã có sẵn “từ điển” (vocabulary) huấn luyện trên hàng nghìn tỉ ký tự.

> 🔑 **Một câu để nhớ:** Tokenizer là _cây cầu hai chiều_ giữa văn bản (chuỗi ký tự) và số nguyên (token id) - encode đưa chữ thành số cho mô hình ăn, decode đưa số về lại chữ cho người đọc.

---

### Bạn sẽ học được gì?

- [ ] Cài và dùng tiktoken để encode / decode như GPT thật làm.
- [ ] Dùng Hugging Face tokenizers / AutoTokenizer và hiểu điểm khác biệt.
- [ ] Đọc được _token id_ và ánh xạ ngược ra các mảnh subword.
- [ ] Hiểu _độ dài chuỗi tính bằng token_ ảnh hưởng thế nào tới bộ nhớ, context window và chi phí.
- [ ] Nối tokenizer thật vào pipeline GPT mini đã build.

📌 **Yêu cầu trước:** Bài 31 (BPE), biết `pip install`, và khái niệm token id -> embedding từ tuần đầu.

---

### 1. Token, id và độ dài chuỗi - nhắc lại bằng analogy

Hãy coi tokenizer như một _cuốn từ điển đánh số_. Mỗi mục trong từ điển là một mảnh văn bản (một ký tự, một subword, hay cả một từ ngắn), và mỗi mục có một _số thứ tự (id)_ duy nhất.

- `encode("Xin chào")` -> tra từ điển -> ra một _danh sách số nguyên_, ví dụ `[55, 259, 6413]`.
- `decode([55, 259, 6413])` -> tra ngược -> ra lại `"Xin chào"`.

Điểm cực kỳ quan trọng: _số token KHÔNG bằng số ký tự cũng KHÔNG bằng số từ_. Với tiếng Anh, kinh nghiệm thô là _~1 token ≈ 4 ký tự ≈ 0.75 từ_. Với tiếng Việt có dấu hoặc emoji, một “chữ” có thể tách thành nhiều token vì tokenizer làm việc trên _byte UTF-8_.

```text
Văn bản:   "Xin chào các bạn"
              |
          encode()   (tra từ điển đánh số)
              v
Token ids: [ 55, 259, 6413, 998, 2201 ]   # shape: (5,)  - 5 token
              |
          decode()   (tra ngược)
              v
Văn bản:   "Xin chào các bạn"
```

💡 Chính _số token_ (không phải số ký tự) mới là thứ mô hình xử lý và bạn phải trả tiền/tốn bộ nhớ cho. Cửa sổ ngữ cảnh (context window) như “2048 token” nghĩa là 2048 _token id_, không phải 2048 chữ.

---

### 2. `tiktoken` - tokenizer của GPT

`tiktoken` là thư viện BPE cực nhanh (lõi viết bằng Rust) mà OpenAI dùng cho các model GPT.

**Cài đặt:**

```bash
pip install tiktoken
```

Trang chủ: [tiktoken trên GitHub](https://github.com/openai/tiktoken)

```python
import tiktoken

# Load a pretrained BPE encoding (GPT-2 uses "gpt2"; newer models use "cl100k_base")
enc = tiktoken.get_encoding("gpt2")

text = "Xin chao, GPT mini!"
ids = enc.encode(text)
print(ids)        # e.g. [55, 259, 354, 78, 11, 402, 11571, 9927, 0]
print(len(ids))   # number of tokens (NOT characters)

# Decode back to the exact same string
back = enc.decode(ids)
print(back)       # "Xin chao, GPT mini!"
assert back == text # round-trip is lossless for valid text

# Vocabulary size = number of distinct token ids
print(enc.n_vocab) # gpt2 -> 50257
```

_Điểm mấu chốt:_ `n_vocab` (kích thước từ điển) chính là số hàng của bảng embedding mà mô hình cần. GPT-2 dùng 50257; nếu bạn nối tokenizer này vào GPT mini thì `nn.Embedding` phải có `num_embeddings=50257`.

```python
# Peek at how a word splits into subword pieces
ids = enc.encode(" tokenizer") # note the leading space is meaningful!
for i in ids:
    # decode_single_token_bytes shows the raw bytes of each piece
    print(i, enc.decode_single_token_bytes(i))
# e.g.  11241 b' token'    izer -> 7509 b'izer'
```

⚠️ Trong BPE kiểu GPT, _dấu cách ở đầu là một phần của token_ (`" token"` khác `"token"`). Vì thế `encode("token")` và `encode(" token")` cho id khác nhau. Đây là lý do khi nối chuỗi bạn phải cẩn thận với khoảng trắng.

---

### 3. Hugging Face `tokenizers` / `AutoTokenizer`

Hệ sinh thái Hugging Face đi kèm mỗi mô hình một tokenizer. Bạn không cần biết nó dùng BPE, WordPiece hay Unigram - chỉ cần `AutoTokenizer` tự tải đúng loại.

**Cài đặt:**

```bash
pip install transformers tokenizers
```

Trang tài liệu: [Hugging Face Documentation](https://huggingface.co/docs)

```python
from transformers import AutoTokenizer

# Downloads the tokenizer that matches the "gpt2" model
tok = AutoTokenizer.from_pretrained("gpt2")

enc = tok("Xin chao, GPT mini!") # returns a dict-like BatchEncoding
print(enc["input_ids"])          # list[int] token ids
print(enc["attention_mask"])     # list[int] 1s (which tokens are real, not padding)

ids = tok.encode("Hello world")  # shortcut -> list[int]
print(tok.decode(ids))           # "Hello world"
print(tok.vocab_size)            # 50257 for gpt2
print(tok.convert_ids_to_tokens(ids)) # ['Hello', 'Ġworld']  (Ġ marks a leading space)
```

**So sánh hai thư viện:**

| Tiêu chí           | tiktoken                                      | Hugging Face tokenizers             |
| :----------------- | :-------------------------------------------- | :---------------------------------- | --------- | --------- |
| **Mục tiêu chính** | Tokenizer chính hãng các model GPT            | Đa mô hình (BERT, Llama, GPT, T5…)  |
| **Thuật toán**     | BPE (byte-level)                              | BPE / WordPiece / Unigram tùy model |
| **Tốc độ**         | Rất nhanh (lõi Rust)                          | Rất nhanh (tokenizers lõi Rust)     |
| **Special token**  | Ít, tối giản                                  | Đầy đủ (`[CLS]`, `<                 | endoftext | >`, pad…) |
| **Khi nào dùng**   | Đếm token GPT, nanoGPT theo phong cách OpenAI | Khi dùng model có sẵn trên Hub      |

💡 Cả hai thực chất đều là BPE bạn học ở Bài 31, chỉ khác cách đóng gói và bộ từ điển đã huấn luyện sẵn.

---

### 4. Nối tokenizer thật vào GPT mini

Nhớ pipeline forward của GPT mini: _token id -> embedding -> các Transformer block -> logits_. Tokenizer nằm ở _đầu_ và _cuối_ pipeline, ngoài mạng nơ-ron.

```text
"Chào bạn"
    |
 tokenizer.encode()            # text -> ids
    v
tensor([[55, 259]])            # shape: (B=1, T=2)  int64
    |
 nn.Embedding(vocab, d)        # ids -> vectors
    v
(1, 2, d)  float32             # token embeddings + positional embeddings
    |
 Transformer blocks            # attention + feed-forward, giữ nguyên shape
    v
(1, 2, d)
    |
 lm_head: Linear(d, vocab)     # -> logits cho token KẾ TIẾP
    v
(1, 2, vocab)                  # logits
    |
 softmax + sample              # (Bài 33) -> id token mới
    v
 tokenizer.decode()            # ids -> text
```

```python
import torch, tiktoken

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab # 50257 -> must match model's embedding rows

# Encode a prompt into a batch tensor the model can consume
prompt = "Once upon a time"
ids = enc.encode(prompt) # list[int], e.g. length 4
x = torch.tensor(ids, dtype=torch.long).unsqueeze(0) # shape: (1, 4)  int64
print(x.shape) # torch.Size([1, 4])

# ... run model(x) -> logits of shape (1, 4, vocab_size) ...

# After sampling a new id and appending it, decode the whole sequence:
generated_ids = ids + [11241] # pretend the model produced token 11241
text = enc.decode(generated_ids)
print(text) # human-readable output
```

🎯 **Quy tắc vàng khi thay tokenizer:** `nn.Embedding(num_embeddings, d_model)` và `lm_head = nn.Linear(d_model, num_embeddings)` phải dùng _cùng một_ `num_embeddings = vocab_size` của tokenizer. Đổi tokenizer mà quên đổi hai chỗ này -> lỗi shape hoặc id vượt biên (`IndexError` / `CUDA device-side assert`).

---

### 5. Độ dài chuỗi tính bằng token ảnh hưởng gì?

Đây là phần rất thực tế. Vì attention có chi phí _$\mathcal{O}(T^2)$_ theo độ dài chuỗi $T$ (mỗi token nhìn mọi token trước đó), nên _số token quyết định bộ nhớ và tốc độ_.

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

samples = ["Hi", "Xin chào các bạn 👋", "def add(a, b): return a + b"]
for s in samples:
    n = len(enc.encode(s))
    print(f"{n:2d} tokens | {len(s):2d} chars | {s!r}")

# e.g.
#  1 tokens |  2 chars | 'Hi'
#  9 tokens | 17 chars | 'Xin chào các bạn 👋'   # dấu + emoji tách thành nhiều token
#  9 tokens | 27 chars | 'def add(a, b): return a + b'
```

Nhận xét: câu tiếng Việt có dấu và emoji “phồng” số token lên vì chúng thành _nhiều byte UTF-8_ -> nhiều token. Điều này nghĩa là cùng một câu, tiếng Việt thường tốn nhiều token hơn tiếng Anh.

| Hệ quả của “nhiều token hơn” | Vì sao                                      |
| :--------------------------- | :------------------------------------------ |
| **Tốn bộ nhớ hơn**           | Ma trận attention là $T \times T \to T^2$ ô |
| **Chậm hơn**                 | Chi phí tính tăng ~theo $T^2$               |
| **Dễ tràn context window**   | Cửa sổ đo bằng token, không phải ký tự      |
| **Tốn tiền hơn (API)**       | API tính phí theo token                     |

---

### Nhìn sâu hơn

- **Byte-level BPE không bao giờ gặp token “unknown”.** Vì bắt đầu từ 256 byte, mọi văn bản UTF-8 (kể cả emoji, ký tự hiếm) đều mã hóa được - khác với tokenizer từ-vựng-cố-định cũ hay trả `<unk>`.
- **`encode` không phải là ánh xạ 1-1 với ký tự.** Cùng một từ ở giữa câu vs đầu câu (khác dấu cách phía trước) có thể ra id khác nhau. Khi so khớp/nối chuỗi, đừng giả định “một chữ = một token”.
- **Special token** như `<|endoftext|>` (id 50256 trong GPT-2) đánh dấu ranh giới tài liệu khi pretraining. Bạn thêm nó vào cuối mỗi văn bản để mô hình học “khi nào thì dừng”.
- **Tokenizer là cố định, không huấn luyện cùng mạng.** Nó được train riêng (thống kê cặp byte) _trước_, rồi đóng băng; gradient của mô hình không chảy ngược vào tokenizer.

---

### Thực hành

- **[Dễ]** Cài `tiktoken`, encode câu tiếng Việt yêu thích của bạn rồi decode lại. In `len(ids)` và so với `len(text)` - chúng có bằng nhau không? Vì sao?
- **[Trung bình]** Dùng `enc.decode_single_token_bytes(i)` in ra từng mảnh subword của `" transformer"` và `"transformer"` (chú ý dấu cách đầu). Giải thích tại sao hai chuỗi cho id khác nhau.
- **[Thử thách]** Viết hàm `count_tokens(text) -> int` bằng `tiktoken`, rồi tải cùng `AutoTokenizer.from_pretrained("gpt2")` của Hugging Face và kiểm tra hai thư viện cho _cùng số token_ trên 10 câu bất kỳ (chúng phải khớp vì cùng vocab GPT-2). In ra câu nào lệch nếu có.

---

### Hiểu lầm thường gặp

- ❌ “Một token luôn là một từ.” ➔ ✅ Token thường là _subword_; một từ dài có thể tách thành nhiều token, một từ ngắn phổ biến có thể là một token.
- ❌ “Số token bằng số ký tự.” ➔ ✅ Sai hoàn toàn - byte-level BPE gộp/tách theo thống kê; tiếng Việt/emoji thường nhiều token hơn số “chữ”.
- ❌ “`encode('token')` giống `encode(' token')`.” ➔ ✅ Dấu cách đầu là một phần token nên id khác nhau.
- ❌ “Đổi tokenizer thì mô hình vẫn chạy như cũ.” ➔ ✅ Đổi tokenizer là đổi `vocab_size` -> phải khớp `nn.Embedding` và `lm_head`, nếu không sẽ lỗi shape / `IndexError`.
- ❌ “Tokenizer được huấn luyện cùng lúc với mạng nơ-ron.” ➔ ✅ Nó được train riêng trước rồi đóng băng; không có gradient chảy vào nó.

---

### Mẹo ghi nhớ

- _encode = chữ -> số; decode = số -> chữ._ Cây cầu hai chiều.
- _vocab_size là số hàng embedding._ Nhớ khớp ở cả input embedding và `lm_head`.
- _Đếm bằng token, không đếm bằng ký tự_ - mọi giới hạn context/chi phí đều theo token.
- Mẹo: gõ thử tiktoken trên web (Tokenizer của OpenAI) để “nhìn tận mắt” một câu tách thành mấy token trước khi tin trực giác.

---

### Tóm tắt trực quan

| Khái niệm                      | Một dòng                                                        |
| :----------------------------- | :-------------------------------------------------------------- | --------- | ------------------------------ |
| **Tokenizer**                  | Cầu hai chiều văn bản <-> token id (số nguyên)                  |
| **encode**                     | Văn bản -> danh sách id                                         |
| **decode**                     | Danh sách id -> văn bản                                         |
| **tiktoken**                   | BPE byte-level nhanh, chính hãng GPT (`n_vocab=50257` cho gpt2) |
| **Hugging Face AutoTokenizer** | Tải đúng tokenizer theo tên model, đa loại                      |
| **vocab_size**                 | Số token phân biệt = số hàng của bảng embedding                 |
| **Token count**                | Đơn vị đo bộ nhớ, context window, chi phí (không phải ký tự)    |
| **Special token**              | `<                                                              | endoftext | >` đánh dấu ranh giới tài liệu |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Let’s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- tiktoken (OpenAI): [https://github.com/openai/tiktoken](https://github.com/openai/tiktoken)
- Hugging Face docs: [https://huggingface.co/docs](https://huggingface.co/docs)
- nanoGPT (Andrej Karpathy): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- The Illustrated Transformer (Jay Alammar): [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Attention Is All You Need: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- PyTorch docs: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)

---

### Tổng kết

- Tokenizer thực tế (`tiktoken`, Hugging Face `tokenizers`) chính là BPE bạn học ở Bài 31 - nhưng nhanh và có sẵn từ điển huấn luyện trên biển dữ liệu.
- `encode` biến văn bản thành danh sách token id; `decode` đưa ngược lại. Đây là hai đầu của pipeline GPT.
- `vocab_size` phải khớp với `nn.Embedding` và `lm_head`; đổi tokenizer là phải đổi hai chỗ này.
- Số _token_ (không phải ký tự) quyết định bộ nhớ, context window và chi phí - vì attention là $\mathcal{O}(T^2)$.

Giờ mô hình đã “đọc” được văn bản qua tokenizer thật và có thể xuất ra _logits_ cho token kế tiếp. Nhưng làm sao chọn token tiếp theo cho hay - lúc nào cần sáng tạo, lúc nào cần chắc chắn? Ở _Bài 33: Sinh văn bản nâng cao - temperature, top-k, top-p_, chúng ta sẽ học cách biến logits thành lựa chọn thông minh, điều khiển độ “ngẫu hứng” của mô hình. Hẹn gặp bạn ở bài kế tiếp! 🤖
