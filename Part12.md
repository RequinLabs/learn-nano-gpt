# Bài 12: Self-attention từ đầu - Query, Key, Value

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 3:** Cơ chế Attention

---

### Giới thiệu

Ở _Bài 11_, chúng ta đã thấy một sự thật khó chịu: khi mô hình chỉ lấy _trung bình_ các vector ngữ cảnh (mỗi token đứng trước đóng góp đúng một phần bằng nhau), nó đối xử với mọi token như nhau. Trong câu _“con mèo đen ngồi trên thảm”_, khi dự đoán token sau _“ngồi”_, chữ _“mèo”_ rõ ràng quan trọng hơn chữ _“đen”_ rất nhiều - nhưng trung bình thì bổ đều, không phân biệt được. Ta cần một cơ chế cho phép mỗi token _tự chọn_ xem nên chú ý tới ai, và chú ý bao nhiêu. Đó chính là _self-attention_.

Hãy hình dung bạn vào thư viện tra cứu. Bạn cầm trong tay một _câu hỏi_ (“mình cần sách về nấu ăn Ý”). Mỗi cuốn sách trên kệ có một _nhãn mô tả_ (“lịch sử”, “ẩm thực Ý”, “vật lý”…). Bạn so câu hỏi của mình với từng nhãn, nhãn nào khớp nhất thì bạn lấy _nội dung_ cuốn đó nhiều nhất. Trong self-attention: câu hỏi là _Query (Q)_, nhãn là _Key (K)_, nội dung là _Value (V)_. Mỗi token phát ra một Query, so với Key của mọi token khác để tính độ khớp, rồi tổng hợp Value theo trọng số khớp đó.

> 🔑 **Một câu để nhớ:** Query hỏi “tôi đang tìm gì?”, Key trả lời “tôi có gì?”, Value là “nội dung tôi mang theo” - attention = tổng hợp Value theo mức khớp giữa Query và Key.

---

### Bạn sẽ học được gì?

- [ ] Hiểu ba vai trò _Query / Key / Value_ bằng analogy tra cứu và định nghĩa chính xác
- [ ] Biết mỗi token biến embedding của mình thành Q, K, V qua ba ma trận trọng số học được
- [ ] Tính được _điểm khớp_ bằng tích vô hướng Q·K, chuẩn hóa bằng _softmax_ thành trọng số attention
- [ ] Tổng hợp _Value_ theo trọng số để ra vector ngữ cảnh mới cho mỗi token
- [ ] Cài đặt bước đầu self-attention bằng PyTorch, theo dõi _shape tensor_ qua từng bước

📌 **Yêu cầu trước:** đã học Bài 11 (hạn chế của trung bình ngữ cảnh), biết embedding, phép nhân ma trận và softmax cơ bản trong PyTorch.

---

### 1. Ba vai trò Q, K, V: từ analogy đến định nghĩa

Mỗi token khi vào lớp attention đã là một _vector embedding_ (ví dụ chiều $C = 32$). Nhưng bản thân embedding chưa nói rõ token đó _đang tìm gì_ hay _có gì để chào mời_. Ta cho token đóng _ba vai_ khác nhau, mỗi vai là một phép chiếu tuyến tính (một ma trận học được):

- _Query (Q)_ = _“Tôi đang tìm loại thông tin nào?”_ - do token hiện tại phát ra.
- _Key (K)_ = _“Tôi mang nhãn gì để người khác tìm thấy tôi?”_ - do mỗi token khác trưng ra.
- _Value (V)_ = _“Nếu được chọn, tôi sẽ đóng góp nội dung gì?”_ - thông tin thực sự được truyền đi.

Ta _không_ dùng thẳng embedding làm cả ba, mà chiếu qua ba ma trận riêng $W_q$, $W_k$, $W_v$ (kích thước $C \times \text{head\_size}$). Nhờ vậy “cách một token đặt câu hỏi” tách khỏi “cách nó tự mô tả” và “nội dung nó mang” - ba việc rất khác nhau nên cần ba không gian riêng.

```text
embedding x_i (chiều C)
                  |
     +------------+------------+
     |            |            |
   x·W_q        x·W_k        x·W_v
     |            |            |
     v            v            v
   Query q_i    Key k_i     Value v_i   (mỗi vector chiều head_size)
```

💡 Query và Key phải cùng chiều $\text{head\_size}$ để có thể lấy tích vô hướng (đo độ khớp). Value có thể cùng hoặc khác chiều, nhưng thường lấy bằng $\text{head\_size}$ cho gọn.

---

### 2. Điểm khớp = tích vô hướng Query · Key

Độ khớp giữa token $i$ (người hỏi) và token $j$ (người bị hỏi) được đo bằng _tích vô hướng_ $q_i \cdot k_j$. Tích vô hướng lớn nghĩa là hai vector “cùng hướng” - câu hỏi và nhãn ăn khớp; nhỏ hoặc âm nghĩa là không liên quan.

Ví dụ tính tay với $\text{head\_size} = 2$:

```text
q_2 = [1.0, 0.5]          # Query của token thứ 2 ("ngồi")
k_0 = [0.9, 0.4]  ("mèo")  -> q2·k0 = 1.0*0.9 + 0.5*0.4 = 1.10
k_1 = [0.1, 0.2]  ("đen")  -> q2·k1 = 1.0*0.1 + 0.5*0.2 = 0.20
k_2 = [0.3, 0.8]  ("ngồi") -> q2·k2 = 1.0*0.3 + 0.5*0.8 = 0.70
```

Điểm khớp thô: `[1.10, 0.20, 0.70]`. Rõ ràng token “ngồi” khớp với “mèo” nhiều nhất - đúng trực giác ngôn ngữ.

Với cả câu có $T$ token, gộp mọi cặp lại thành _ma trận điểm_ $\text{scores} = Q @ K^T$, kích thước $(T, T)$: hàng $i$ là câu hỏi của token $i$, cột $j$ là nhãn của token $j$.

```text
K^T (nhãn của từng token)
        mèo   đen   ngồi
Q mèo [ .    .     .   ]
Q đen [ .    .     .   ]      scores[i][j] = q_i · k_j
Qngồi [1.10  0.20  0.70]  <- hàng của token "ngồi"
```

⚠️ Ở bài này ta _chưa_ chia cho $\sqrt{d_k}$ và _chưa_ áp causal mask - đó là nội dung của _Bài 13 (scaled dot-product attention)_. Ở đây ta tập trung hiểu Q/K/V và luồng dữ liệu trước đã.

---

### 3. Softmax biến điểm khớp thành trọng số

Điểm khớp thô có thể âm, dương, không tổng bằng 1 - chưa dùng làm trọng số được. Ta đưa mỗi hàng qua _softmax_ để thành một phân phối xác suất (dương, tổng = 1):

```text
softmax([1.10, 0.20, 0.70])
 = exp([1.10, 0.20, 0.70]) / sum(...)
 = [3.004, 1.221, 2.014] / 6.239
 ≈ [0.481, 0.196, 0.323]
```

Vậy khi dự đoán ngữ cảnh cho “ngồi”, mô hình rót ~48% chú ý vào “mèo”, ~32% vào chính “ngồi”, ~20% vào “đen”. So với trung bình (mỗi token 33%), attention đã _học cách phân biệt_ - đó là cả một bước nhảy so với Bài 11.

💡 Softmax làm nổi bật điểm lớn nhất (theo hàm mũ) nhưng _không_ loại bỏ hẳn các token khác - vẫn giữ một chút trọng số cho mọi người. Đây là điểm khác biệt với “chọn cứng đúng 1 token”.

---

### 4. Tổng hợp Value theo trọng số

Bước cuối: mỗi token lấy _trung bình có trọng số_ của các Value theo trọng số attention vừa tính. Nếu $w = [0.481, 0.196, 0.323]$ và các Value là $v_0, v_1, v_2$:

```text
out_2 = 0.481 * v_0 + 0.196 * v_1 + 0.323 * v_2
```

Gộp cho cả câu: $\text{out} = \text{attention\_weights} @ V$, với $\text{attention\_weights}$ kích thước $(T, T)$ và $V$ kích thước $(T, \text{head\_size})$, cho ra $\text{out}$ kích thước $(T, \text{head\_size})$. Mỗi token nhận về một vector ngữ cảnh _mới_, đã “hút” thông tin từ những token liên quan nhất.

```text
Luồng đầy đủ (bỏ scale + mask cho bài này):

 x (T,C) --W_q--> Q (T,head_size)
 x (T,C) --W_k--> K (T,head_size) --> scores = Q@K^T  (T,T)
 x (T,C) --W_v--> V (T,head_size)              |
                                             softmax theo hàng
                                                |
                                        weights (T,T)
                                                |
                                   out = weights @ V  (T,head_size)
```

---

### 5. Cài đặt bước đầu bằng PyTorch

Gắn tất cả lại thành một “đầu attention” (single head). Chú ý shape trong comment:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

B, T, C = 1, 3, 32 # batch=1, sequence length=3, embedding dim=32
head_size = 16    # dimension of Q/K/V for this head
x = torch.randn(B, T, C) # input embeddings, shape (B, T, C) = (1, 3, 32)

# Three learnable linear projections (no bias, following nanoGPT style)
W_q = nn.Linear(C, head_size, bias=False)
W_k = nn.Linear(C, head_size, bias=False)
W_v = nn.Linear(C, head_size, bias=False)

q = W_q(x) # Query, shape (B, T, head_size) = (1, 3, 16)
k = W_k(x) # Key,   shape (B, T, head_size) = (1, 3, 16)
v = W_v(x) # Value, shape (B, T, head_size) = (1, 3, 16)

# Match score for every (query_i, key_j) pair via dot product.
# k.transpose(-2, -1) turns (B, T, head_size) into (B, head_size, T).
scores = q @ k.transpose(-2, -1) # shape (B, T, T) = (1, 3, 3)

# Turn each row of scores into a probability distribution (sums to 1).
weights = F.softmax(scores, dim=-1) # shape (B, T, T)

# Weighted sum of Value vectors -> new context vector per token.
out = weights @ v # shape (B, T, head_size) = (1, 3, 16)

print("q shape:", q.shape) # torch.Size([1, 3, 16])
print("scores shape:", scores.shape) # torch.Size([1, 3, 3])
print("weights row sums:", weights.sum(dim=-1)) # tensor([[1., 1., 1.]])
print("out shape:", out.shape) # torch.Size([1, 3, 16])
```

Output mẫu:

```text
q shape: torch.Size([1, 3, 16])
scores shape: torch.Size([1, 3, 3])
weights row sums: tensor([[1., 1., 1.]])
out shape: torch.Size([1, 3, 16])
```

🎯 Điểm mấu chốt: $W_q, W_k, W_v$ là _tham số học được_ - trong quá trình huấn luyện, mô hình tự điều chỉnh chúng để câu hỏi/nhãn/nội dung khớp sao cho dự đoán token tiếp theo tốt nhất. Ta không lập trình “mèo khớp với ngồi” - mô hình học ra điều đó.

---

### Nhìn sâu hơn

_Vì sao tách ba ma trận riêng thay vì dùng chung embedding?_ Vì ba vai trò cần ba “góc nhìn” khác nhau lên cùng một token. Cách một token _hỏi_ (Query) không nhất thiết giống cách nó _tự giới thiệu_ (Key) hay _nội dung nó mang_ (Value). Ba ma trận cho mô hình đủ tự do biểu diễn những khác biệt đó. Nếu ép $Q = K = V = x$, mọi token sẽ khớp mạnh nhất với chính nó (tích vô hướng của vector với chính nó luôn lớn), làm attention gần như vô dụng.

_Chi phí tính toán:_ Ma trận $\text{scores} = Q @ K^T$ có kích thước $(T, T)$, nên bộ nhớ và tính toán tăng _bậc hai theo độ dài câu_ $T$. Với $T = 1024$ đã là hơn 1 triệu ô mỗi head. Đây là lý do vì sao ngữ cảnh dài (long context) tốn kém, và là động lực cho nhiều tối ưu sau này (FlashAttention, KV-cache…). Ở bài này $T$ nhỏ nên chưa lo, nhưng nhớ đặc điểm bậc hai này.

_Q/K/V khác feed-forward thế nào?_ Attention _trộn thông tin GIỮA các token_ (token này nhìn token kia). Còn lớp feed-forward (học ở tuần sau) xử lý _từng token độc lập_, không trộn ngang. Đừng lẫn hai vai trò: attention lo “giao tiếp”, feed-forward lo “suy nghĩ riêng”.

---

### Thực hành

- **[Dễ]** Chạy đoạn code mục 5, đổi $T = 5$ và $\text{head\_size} = 8$. Dự đoán trước shape của `q`, `scores`, `out` rồi in ra kiểm chứng.
- **[Trung bình]** Thay `torch.randn(B, T, C)` bằng một `x` mà token 0 và token 2 gần như giống hệt nhau (copy hàng). In `weights` và giải thích vì sao trọng số attention của token 0 và token 2 lại tương tự.
- **[Thử thách]** Bỏ ba ma trận $W_q/W_k/W_v$ đi, tính $\text{scores} = x @ x.\text{transpose}(-2,-1)$ (dùng thẳng embedding cho cả Q, K, V). So sánh `weights` với bản có ma trận chiếu, và giải thích vì sao mỗi token có xu hướng chú ý mạnh nhất vào _chính nó_.

---

### Hiểu lầm thường gặp

- ❌ “Q, K, V là ba token khác nhau.” ➔ ✅ Chúng là _ba phép chiếu của CÙNG một token_ (cùng embedding $x$), qua ba ma trận khác nhau.
- ❌ “Attention cứ lấy trung bình như Bài 11.” ➔ ✅ Attention lấy _trung bình CÓ TRỌNG SỐ_, trọng số học được qua độ khớp $Q \cdot K$ - đó là điểm khác cốt lõi.
- ❌ “Điểm khớp scores chính là trọng số dùng ngay.” ➔ ✅ Phải qua _softmax_ để thành phân phối dương, tổng 1 mới dùng làm trọng số.
- ❌ “Value cũng chỉ để tính điểm khớp.” ➔ ✅ Điểm khớp chỉ dùng _Q và K_; _V_ là nội dung được tổng hợp ở bước cuối, không tham gia tính điểm.

---

### Mẹo ghi nhớ

- _Q hỏi, K trả, V trao_: Query hỏi, Key trả lời độ khớp, Value trao nội dung.
- Thứ tự bất di bất dịch: _chiếu Q/K/V -> điểm khớp Q·K -> softmax -> nhân V_.
- Shape thần chú: $Q, K, V$ là $(T, \text{head\_size})$; $\text{scores}$ và $\text{weights}$ là $(T, T)$; $\text{out}$ là $(T, \text{head\_size})$.
- 🧠 Mẹo: liên tưởng thư viện - _cầm câu hỏi (Q), quét nhãn sách (K), lấy nội dung (V) theo mức khớp_.

---

### Tóm tắt trực quan

| Khái niệm             | Một dòng để nhớ                                    |
| :-------------------- | :------------------------------------------------- |
| **Query (Q)**         | Câu hỏi token phát ra: “tôi cần gì?”               |
| **Key (K)**           | Nhãn token trưng ra: “tôi có gì?”                  |
| **Value (V)**         | Nội dung token trao đi khi được chọn               |
| **scores = Q @ K^T**  | Ma trận (T, T) đo độ khớp mọi cặp token            |
| **softmax(scores)**   | Biến điểm khớp thành trọng số (dương, tổng 1)      |
| **out = weights @ V** | Vector ngữ cảnh mới = trung bình có trọng số của V |
| **Chi phí**           | Bậc hai theo độ dài câu T                          |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Transformer Self-Attention Explained (Query, Key, Value Math)](https://www.youtube.com/watch?v=cM_8tglxDUE)
- The Illustrated Transformer (giải thích Q/K/V bằng hình): [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- Attention Is All You Need (paper gốc): [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- nanoGPT của Andrej Karpathy (self-attention cài từ đầu): [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- PyTorch documentation (Linear, softmax): [pytorch.org/docs/stable](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [pytorch.org/tutorials](https://pytorch.org/tutorials/)

---

### Tổng kết

- Self-attention cho mỗi token _tự chọn_ chú ý tới token nào, thay vì bổ đều như trung bình ở Bài 11.
- Mỗi token chiếu embedding thành ba vai: _Query_ (hỏi), _Key_ (nhãn), _Value_ (nội dung) qua ba ma trận học được $W_q/W_k/W_v$.
- Quy trình: $\text{scores} = Q @ K^T \to \text{softmax theo hàng} \to \text{out} = \text{weights} @ V$, giữ vững shape $(T, \text{head\_size})$ cho Q/K/V và $(T, T)$ cho ma trận trọng số.
- Trọng số attention chính là _trung bình có trọng số_ thông minh - trái tim của Transformer.

Nhưng bạn có để ý điểm khớp thô $Q \cdot K$ có thể phình to khi $\text{head\_size}$ lớn, khiến softmax bão hòa và gradient teo lại? Ở _Bài 13: Scaled dot-product attention_, chúng ta sẽ chia điểm cho $\sqrt{d_k}$ để ổn định, đồng thời thêm _causal mask_ tam giác để token không được nhìn về tương lai - đưa attention của ta tiến sát công thức chuẩn trong GPT. Hẹn gặp ở bài kế tiếp! 🚀
