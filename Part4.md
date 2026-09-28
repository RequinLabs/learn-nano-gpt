# Bài 4: Mô hình ngôn ngữ là gì - dự đoán token tiếp theo

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 1:** Môi trường, phần cứng & nền tảng

---

### Giới thiệu

Ở _Bài 3_ mình đã ôn lại ba trụ cột của PyTorch: tensor để chứa số, autograd để tự tính đạo hàm, và nn.Module để đóng gói mô hình. Bạn đã biết cách tạo một mạng nhỏ, đưa dữ liệu qua nó (forward), rồi để `loss.backward()` tính gradient. Tuyệt vời! Nhưng có một câu hỏi lớn vẫn treo lơ lửng: _một mô hình ngôn ngữ như GPT thực chất đang HỌC cái gì?_

Hãy tưởng tượng lúc bạn nhắn tin trên điện thoại. Bạn gõ “Hôm nay trời rất…” và bàn phím gợi ý ngay “đẹp”, “nóng”, “mưa”. Cái bàn phím đó chính là một mô hình ngôn ngữ tí hon: nó nhìn những chữ đã có rồi _đoán chữ kế tiếp_. GPT làm y hệt, chỉ là ở quy mô khổng lồ và mượt mà hơn rất nhiều.

Điều kỳ diệu là: _chỉ cần dạy máy tính chơi trò “đoán chữ tiếp theo” thật giỏi, nó tự nhiên biết viết văn, trả lời câu hỏi, thậm chí viết code._ Nghe khó tin phải không? Bài này sẽ giải thích vì sao.

> 🔑 **Một câu để nhớ:** Một mô hình ngôn ngữ là một hàm nhận vào các token đã có và trả về _phân phối xác suất trên toàn bộ tập từ vựng_ cho token kế tiếp - học tốt trò chơi đó chính là học ngôn ngữ.

---

### Bạn sẽ học được gì?

- [ ] Hiểu _next-token prediction_ (dự đoán token kế tiếp) là ý tưởng cốt lõi của mọi LLM.
- [ ] Phân biệt rõ _token_, _vocabulary_ (tập từ vựng) và _logits_ với _xác suất_.
- [ ] Biết vì sao chỉ học dự đoán lại tạo ra khả năng _sinh văn bản_ (generation).
- [ ] Viết được đoạn PyTorch biến logits thành xác suất bằng softmax và tính cross-entropy loss.
- [ ] Nắm quy tắc phân rã xác suất chuỗi (chain rule) mà mô hình ngôn ngữ dựa vào.

📌 **Yêu cầu trước:** Đã học Bài 1-3 (môi trường, PyTorch cơ bản: `tensor`, `autograd`, `nn.Module`, `softmax`, `cross-entropy`).

---

### 1. Mô hình ngôn ngữ là gì? Analogy trò đoán chữ

Hãy chơi một trò: mình che chữ cuối trong câu “Con mèo trèo lên cây…” và hỏi bạn chữ tiếp theo là gì. Bạn sẽ nghĩ ngay “cao”, “xanh”, “cột”… và mỗi lựa chọn có một “độ tin” khác nhau. Bạn không trả lời duy nhất một chữ - trong đầu bạn có cả một _bảng xếp hạng khả năng_.

Đó chính xác là điều một _mô hình ngôn ngữ (language model)_ làm. Định nghĩa chính xác:
Một mô hình ngôn ngữ ước lượng xác suất $P(\text{token\_kế\_tiếp} \mid \text{các token trước đó})$.

Nếu chuỗi đầu vào là các token $x_1, x_2, \dots, x_t$, mô hình trả về:
$$P(x_{t+1} \mid x_1, x_2, \dots, x_t)$$

Nghĩa là: cho toàn bộ ngữ cảnh phía trước, mô hình đưa ra _một con số xác suất cho MỖI token có thể_ trong tập từ vựng. Không phải một câu trả lời, mà là cả một phân phối.

💡 GPT là _autoregressive_ (tự hồi quy): nó chỉ nhìn về _quá khứ_ (các token bên trái) để đoán token kế tiếp, không được “nhìn trộm” tương lai. Ta sẽ thực thi điều này bằng _causal mask_ ở các bài sau.

---

### 2. Token và vocabulary - viên gạch của ngôn ngữ

Máy tính không hiểu chữ, chỉ hiểu số. Vậy trước tiên ta phải cắt văn bản thành các mảnh nhỏ gọi là _token_ (có thể là ký tự, mảnh từ, hoặc từ), rồi gán cho mỗi token duy nhất một _số nguyên id_.

Tập tất cả token khác nhau mà mô hình biết gọi là _vocabulary_ (tập từ vựng), kích thước ký hiệu là $V$ (`vocab_size`).

Ví dụ với tokenizer ở mức ký tự (character-level) cho một tập nhỏ:

```python
import torch

text = "hello"
# Build vocabulary from unique characters
chars = sorted(set("helo")) # ['e', 'h', 'l', 'o']
stoi = {c: i for i, c in enumerate(chars)} # char -> id
itos = {i: c for c, i in stoi.items()}       # id -> char
vocab_size = len(chars)                      # V = 4

ids = torch.tensor([stoi[c] for c in text])
print(ids)        # tensor([1, 0, 2, 2, 3])  -> h,e,l,l,o
print(vocab_size) # 4
```

⚠️ Đừng nhầm _token_ với _từ_. Token có thể nhỏ hơn một từ (mảnh subword) hoặc chỉ là một ký tự. Cách cắt token (character-level so với BPE) sẽ học kỹ ở _Bài 5_.

---

### 3. Logits so với xác suất - đừng nhầm hai thứ này

Đây là điểm cực kỳ hay bị lẫn. Khi mô hình xử lý xong ngữ cảnh, lớp cuối cùng cho ra một vector độ dài $V$ gọi là _logits_: các con số thực, có thể âm, dương, chưa chuẩn hóa. _Logits chưa phải xác suất!_

Muốn biến logits thành xác suất (cộng lại bằng 1, đều không âm), ta dùng _softmax_:
$$\text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

Analogy: logits giống như “số phiếu bầu thô” cho mỗi ứng viên (token). Softmax là bước chia tỷ lệ để ra “phần trăm ủng hộ” - tổng luôn 100%.

```python
import torch
import torch.nn.functional as F

# Suppose vocab_size = 4, model output logits for the next token
logits = torch.tensor([2.0, 1.0, 0.1, -1.0]) # shape: (V=4,)
probs = F.softmax(logits, dim=-1)            # shape: (4,)
print(probs)       # tensor([0.6034, 0.2220, 0.0903, 0.0300]) approx
print(probs.sum()) # tensor(1.0000)  -> a valid probability distribution
```

| Khái niệm    | Bản chất                    | Điều kiện                              |
| :----------- | :-------------------------- | :------------------------------------- |
| **logits**   | số thực thô, chưa chuẩn hóa | bất kỳ giá trị nào (âm/dương)          |
| **xác suất** | sau `softmax(logits)`       | mỗi giá trị trong $[0, 1]$, tổng $= 1$ |

🎯 Ghi nhớ: mô hình _sinh ra logits_; ta _áp softmax_ để có xác suất. Khi huấn luyện, PyTorch còn có mẹo gộp cả hai bước lại (xem mục sau) cho ổn định số học.

---

### 4. Sơ đồ luồng: từ token đến phân phối token kế tiếp

Đây là bức tranh tổng thể của một mô hình ngôn ngữ nơ-ron (mà ta sẽ dần lắp ráp qua cả khóa):

```text
Input tokens (ids)        Embeddings           Neural blocks         Logits            Probabilities
 [h, e, l, l]      ->   lookup table    ->   (attention +    ->   linear head  ->   softmax
  shape (T,)            shape (T, d)          MLP layers)         shape (T, V)       shape (T, V)
                                              shape (T, d)
                                                                          |
                                          take last position (t) ---------+--> P(next token | context)
```

Ở đây:

- $T$ = độ dài chuỗi (số token đầu vào),
- $d$ = số chiều embedding (`d_model`),
- $V$ = kích thước vocabulary.

Mỗi token id được tra trong _embedding table_ để thành một vector $d$ chiều (khác hẳn one-hot - xem mục Hiểu lầm). Sau khi qua các block xử lý ngữ cảnh, một _linear head_ chiếu về $V$ chiều cho ra logits. Vị trí cuối cùng của chuỗi cho ta phân phối xác suất của token kế tiếp.

---

### 5. Vì sao "chỉ đoán chữ" lại sinh ra được văn bản?

Đây là phần “ma thuật”. Nếu mô hình dự đoán được $P(x_{t+1} \mid x_1 \dots x_t)$ cho mọi ngữ cảnh, ta có thể _sinh văn bản_ bằng cách lặp lại: đoán một token, nối vào chuỗi, rồi đoán tiếp. Trò này gọi là _generation tự hồi quy_.

Nền tảng toán học là _quy tắc nhân xác suất (chain rule)_: xác suất của cả một câu bằng tích các xác suất token kế tiếp:
$$P(x_1, x_2, \dots, x_n) = P(x_1) \times P(x_2 \mid x_1) \times P(x_3 \mid x_1, x_2) \times \dots \times P(x_n \mid x_1 \dots x_{n-1})$$

Vì thế, học tốt “một bước đoán token kế tiếp” tương đương với học được xác suất của _mọi câu_. Muốn giỏi trò này, mô hình buộc phải ngầm học ngữ pháp, ngữ nghĩa, sự kiện, cả logic - vì tất cả đều giúp đoán chính xác hơn.

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_next(logits, temperature=1.0):
    # logits shape: (V,) -> raw scores for the next token
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1) # shape: (V,)
    next_id = torch.multinomial(probs, num_samples=1) # sample 1 token -> shape (1,)
    return next_id

logits = torch.tensor([2.0, 1.0, 0.1, -1.0])
print(generate_next(logits)) # e.g. tensor([0]) - most often the highest-prob token
```

💡 _Temperature_ điều chỉnh độ “táo bạo”: nhỏ ($< 1$) làm phân phối nhọn hơn $\to$ chọn an toàn, lặp lại; lớn ($> 1$) làm phẳng hơn $\to$ sáng tạo nhưng dễ lung tung. Ta sẽ đào sâu temperature/top-k/top-p ở bài về sinh văn bản.

---

### 6. Huấn luyện bằng cross-entropy - dạy máy đoán đúng

Làm sao “chấm điểm” mô hình? Với mỗi vị trí, ta biết token thật (nhãn) là gì. Ta muốn mô hình gán _xác suất cao cho token đúng_. Hàm mất mát chuẩn cho việc này là _cross-entropy_ (entropy chéo):
$$\text{loss} = -\log(P_{\text{mô hình}}(\text{token đúng}))$$

Đoán đúng với xác suất gần 1 $\to$ loss gần 0. Đoán token đúng với xác suất bé xíu $\to$ loss lớn. Trung bình trên toàn bộ dữ liệu chính là thứ ta tối thiểu hóa bằng gradient descent (`AdamW`).

```python
import torch
import torch.nn.functional as F

# Batch B=2 sequences, T=3 positions, vocab V=4
# logits are the model's raw outputs for each position
logits = torch.randn(2, 3, 4) # shape: (B, T, V)
targets = torch.tensor([[1, 0, 3],
                        [2, 2, 1]]) # shape: (B, T) - true next-token ids

# cross_entropy expects (N, V) and (N,), so flatten batch+time
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)), # shape: (B*T, V) = (6, 4)
    targets.view(-1)                  # shape: (B*T,)   = (6,)
)
print(loss) # a single scalar, e.g. tensor(1.53)
```

🎯 `F.cross_entropy` _tự làm softmax bên trong_ (thực chất là `log_softmax` + `nll_loss`) nên ổn định số học hơn là tự softmax rồi lấy log. Vì vậy ta đưa _logits thô_ vào, KHÔNG đưa xác suất đã softmax.

Một điểm đẹp của next-token prediction: mỗi vị trí trong chuỗi là một bài tập dự đoán. Chuỗi dài $T$ token cho ta _$T-1$ cặp (ngữ cảnh, nhãn)_ miễn phí - dữ liệu tự dán nhãn (self-supervised), không cần con người gán nhãn. Đó là lý do LLM ăn được cả internet.

---

### Nhìn sâu hơn

- **Vì sao thiết kế là dự đoán token kế tiếp mà không phải cả câu một lúc?** Vì phân rã theo chain rule biến bài toán khổng lồ (xác suất trên MỌI câu) thành các bài nhỏ, lặp lại được, và mỗi bước chỉ cần một softmax trên $V$ phần tử. Điều này khiến việc huấn luyện song song hóa được (đoán mọi vị trí cùng lúc khi train) và sinh văn bản tuần tự được (một token mỗi bước khi inference).
- **Chi phí ở đâu?** Lớp cuối `Linear(d, V)` và softmax chạy trên $V$ phần tử. Với $V$ lớn (BPE thường ~50000), riêng ma trận đầu ra đã tốn $d \times V$ tham số - một phần đáng kể bộ nhớ. Đó là một lý do người ta cân nhắc kích thước vocabulary.
- **Pretraining so với fine-tuning so với RLHF:** Trò đoán token kế tiếp trên khối văn bản khổng lồ là _pretraining_ (tiền huấn luyện). Sau đó ta có thể _fine-tuning_ (tinh chỉnh) trên dữ liệu chuyên biệt, rồi _RLHF_ (học tăng cường từ phản hồi con người) để mô hình lịch sự, hữu ích. Cả khóa này ta tập trung vào phần cốt lõi: pretraining bằng next-token prediction.

---

### Thực hành

- **[Dễ]** Tự tay tính `softmax([1.0, 2.0, 3.0])` bằng máy tính bỏ túi rồi so với `F.softmax(torch.tensor([1., 2., 3.]), dim=-1)`. Kiểm tra tổng có bằng 1 không.
- **[Trung bình]** Viết hàm `cross_entropy_manual(logits, target)` cho _một_ vị trí: tự làm `log_softmax` rồi lấy `-logp[target]`. So kết quả với `F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))`.
- **[Thử thách]** Với `logits = torch.tensor([2.0, 1.0, 0.1, -1.0])`, in phân phối xác suất ở `temperature = 0.5, 1.0, 2.0`. Quan sát: nhiệt độ nhỏ làm phân phối _nhọn_ hay _phẳng_ hơn? Giải thích vì sao token có xác suất cao nhất càng được ưu ái khi nhiệt độ giảm.

---

### Hiểu lầm thường gặp

- ❌ “Logits chính là xác suất.” $\to$ ✅ Logits là _số thực thô chưa chuẩn hóa_; phải qua softmax mới thành xác suất (không âm, tổng bằng 1).
- ❌ “Mô hình ngôn ngữ đoán ra _một_ từ đúng duy nhất.” $\to$ ✅ Nó cho ra _một phân phối xác suất trên toàn bộ vocabulary_; sinh văn bản là lấy mẫu/chọn từ phân phối đó.
- ❌ “Embedding chỉ là one-hot đổi tên.” $\to$ ✅ One-hot là vector thưa $V$ chiều toàn 0 chỉ 1 vị trí bằng 1; _embedding_ là vector đặc $d$ chiều _học được_, mang ngữ nghĩa, $d$ nhỏ hơn $V$ rất nhiều.
- ❌ “Chỉ đoán token kế tiếp thì không thể làm việc phức tạp.” $\to$ ✅ Nhờ chain rule, làm tốt bước nhỏ này tương đương học xác suất mọi câu, đủ để sinh văn bản, trả lời, viết code.

---

### Mẹo ghi nhớ

- **L-S-P:** **L**ogits $\to$ **S**oftmax $\to$ **P**robabilities (xác suất). Nhớ đúng thứ tự là hết nhầm.
- Next-token prediction = “đoán chữ trên bàn phím điện thoại”, nhưng học cả internet.
- Khi train, đưa _logits thô_ vào `F.cross_entropy` (nó tự `log_softmax`), đừng softmax trước.
- 🧠 **Mẹo vàng:** “Mô hình không trả lời một token, nó trả lời _một bảng xác suất_”.

---

### Tóm tắt trực quan

| Khái niệm                 | Một dòng để nhớ                                              |
| :------------------------ | :----------------------------------------------------------- |
| **Mô hình ngôn ngữ**      | Hàm ước lượng $P(\text{token kế tiếp} \mid \text{ngữ cảnh})$ |
| **Token / vocabulary**    | Mảnh văn bản được gán id; $V$ = số token khác nhau           |
| **Logits**                | Vector $V$ số thực thô từ lớp cuối, chưa là xác suất         |
| **Softmax**               | Biến logits thành phân phối xác suất (tổng = 1)              |
| **Cross-entropy**         | $-\log P(\text{token đúng})$; hàm mất mát khi huấn luyện     |
| **Next-token prediction** | Học đoán token kế tiếp = học sinh văn bản (chain rule)       |
| **Temperature**           | Điều chỉnh độ nhọn/phẳng của phân phối khi sinh              |

---

### Tài liệu tham khảo

- 🎬 Video hướng dẫn: [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs)
- PyTorch docs (`softmax`, `cross_entropy`): [pytorch.org/docs/stable](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [pytorch.org/tutorials](https://pytorch.org/tutorials/)
- nanoGPT (Andrej Karpathy) - LLM tối giản từ đầu: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- “Attention Is All You Need” (paper Transformer gốc): [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- Hugging Face docs: [huggingface.co/docs](https://huggingface.co/docs)

---

### Tổng kết

- Mô hình ngôn ngữ = hàm ước lượng _xác suất token kế tiếp_ cho trước ngữ cảnh - giống trò đoán chữ trên bàn phím.
- Đầu ra của mô hình là _logits_; qua _softmax_ thành _xác suất_ trên toàn bộ vocabulary (đừng nhầm hai thứ).
- Nhờ _chain rule_, học tốt next-token prediction đủ để _sinh văn bản_ - và dữ liệu tự dán nhãn nên train được ở quy mô khổng lồ.
- Ta huấn luyện bằng _cross-entropy_ ($-\log P(\text{token đúng})$), tối thiểu hóa bằng gradient descent (`AdamW`).

_Bài tiếp theo: Bài 5: Biểu diễn văn bản - token, vocabulary và character-level tokenizer._
