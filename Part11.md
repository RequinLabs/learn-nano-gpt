# Bài 11: Vì sao cần Attention - hạn chế của trung bình ngữ cảnh

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 3:** Cơ chế Attention

---

### Giới thiệu

Ở _Bài 10_, chúng ta đã đo loss trên cả tập train lẫn validation và nhìn thẳng vào sự thật: mô hình bigram, dù huấn luyện xong, vẫn “đần”. Lý do gốc rất đơn giản - bigram chỉ nhìn _đúng một token liền trước_ để đoán token kế tiếp. Nó như một người chỉ nhớ được từ cuối cùng bạn vừa nói rồi lập tức quên sạch phần đầu câu. 🤖

Hôm nay chúng ta trả lời câu hỏi lớn: _làm sao để mỗi token “nhìn” được cả những token đứng trước nó?_ Ý tưởng ngây thơ nhất là lấy _trung bình_ tất cả các token trước - kiểu “tóm tắt bằng cách cộng lại chia đều”. Ta sẽ tự tay cài đặt cách trung bình đó, thấy nó hoạt động ra sao, rồi phát hiện điểm yếu chí mạng của nó. Chính điểm yếu này là lý do vì sao ta cần _Attention_ - trái tim của kiến trúc Transformer.

Hãy tưởng tượng bạn đọc câu: _“Con mèo đen ngồi trên thảm, nó đang ngủ.”_ Để hiểu _“nó”_ trỏ về ai, bạn không đối xử mọi từ như nhau. Bạn tự động _chú ý_ nhiều vào “con mèo” và gần như bỏ qua “trên”, “thảm”. Đó chính là attention: mỗi token gán _trọng số quan tâm khác nhau_ cho các token khác, thay vì cào bằng.

> 🔑 **Một câu để nhớ:** trung bình đơn giản coi mọi token quá khứ _quan trọng như nhau_; attention cho phép mỗi token _tự học_ nên chú ý vào token nào _nhiều hay ít_.

---

### Bạn sẽ học được gì?

- [ ] Vì sao bigram thiếu ngữ cảnh, và “trung bình ngữ cảnh” là bước cải tiến đầu tiên tự nhiên
- [ ] Cách biến “trung bình các token trước” thành một phép nhân ma trận với _ma trận tam giác dưới_ (causal)
- [ ] Cài đặt trung bình đó bằng 3 cách trong PyTorch (vòng lặp, ma trận trọng số, `softmax`) và hiểu vì sao chúng tương đương
- [ ] Vì sao trung bình đều _là trường hợp đặc biệt_ của attention (trọng số bằng nhau) và điểm yếu của nó
- [ ] Trực giác về _trọng số học được_ - cầu nối sang self-attention Q/K/V ở bài sau

📌 **Yêu cầu trước:** đã học Bài 9-10 (bigram, `softmax`, `logits`, `loss`), biết `torch`, phép nhân ma trận (`@`), và broadcasting cơ bản.

---

### 1. Vấn đề: bigram chỉ thấy một token

Bigram dự đoán token thứ $t+1$ _chỉ_ dựa vào token thứ $t$. Trong một chuỗi $x_1, x_2, \dots, x_t$, thông tin từ $x_1 \dots x_{t-1}$ bị _vứt đi hoàn toàn_. Nhưng ngôn ngữ thì phụ thuộc xa: chủ ngữ ở đầu câu quyết định động từ ở cuối câu.

Bước cải tiến đơn giản nhất: _để token thứ $t$ “trò chuyện” với tất cả token từ $1$ đến $t$_ (không được nhìn tương lai $t+1, t+2, \dots$ vì lúc sinh văn bản ta chưa có chúng - đây là ràng buộc _causal/nhân quả_). Cách gộp thông tin thô sơ nhất là lấy _trung bình_ các vector token trước đó:

```text
context_t = trung binh cua (x_1, x_2, ..., x_t)
         = (x_1 + x_2 + ... + x_t) / t
```

Mỗi token bây giờ mang theo một “bản tóm tắt” của quá khứ. Đây gọi là mô hình _bag-of-words_ (túi từ): gộp mọi thứ lại, _mất thứ tự và mất mức độ quan trọng_, nhưng ít ra đã có ngữ cảnh.

💡 Ta trung bình các _vector embedding_, KHÔNG phải các chỉ số token (`token id`). Cộng trung bình các id là vô nghĩa (id 5 + id 9 chẳng liên quan gì nhau về ngữ nghĩa); nhưng vector embedding sống trong không gian liên tục nên trung bình chúng cho ra một vector “trung tâm” có ý nghĩa.

---

### 2. Cách 1 - trung bình bằng vòng lặp (dễ hiểu nhưng chậm)

Giả sử ta có một batch chuỗi đã embedding: tensor shape `(B, T, C)` với $B$ = số chuỗi, $T$ = độ dài chuỗi (time), $C$ = số chiều embedding.

```python
import torch

torch.manual_seed(1337)
B, T, C = 2, 4, 3 # batch=2, time(=context)=4, channels(=embed dim)=3
x = torch.randn(B, T, C) # x shape: (2, 4, 3) - fake token embeddings

# xbow[b, t] = average of x[b, 0..t] (bow = bag of words)
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # shape: (t+1, C) - tokens 0..t only, no future
        xbow[b, t] = xprev.mean(dim=0) # average over the time dimension -> (C,)

# xbow shape: (2, 4, 3). xbow[b, 0] == x[b, 0]; xbow[b, 3] == mean of 4 tokens
print(xbow.shape) # torch.Size([2, 4, 3])
```

Để ý: token đầu ($t=0$) chỉ có chính nó nên trung bình = chính nó; token cuối ($t=3$) là trung bình cả 4 token. Đúng ràng buộc causal - _không token nào nhìn về phía tương lai_. Nhưng vòng lặp Python lồng nhau _rất chậm_ trên chuỗi dài. Ta cần cách vector hóa.

---

### 3. Cách 2 - trung bình là một phép NHÂN MA TRẬN 🎯

Đây là cái “aha” quan trọng nhất của bài. Trung bình có trọng số đều _chính là_ nhân với một ma trận tam giác dưới đã chuẩn hóa. Nhìn ma trận trọng số `wei` cho $T=4$:

```text
wei (T x T), moi hang cong = 1, chi nhin qua khu (tam giac duoi):

        t0     t1     t2     t3   <- nguon (token duoc nhin)
 t0 [ 1.00   0      0      0    ]   token 0: chi thay chinh no
 t1 [ 0.50   0.50   0      0    ]   token 1: trung binh t0,t1
 t2 [ 0.33   0.33   0.33   0    ]   token 2: trung binh t0,t1,t2
 t3 [ 0.25   0.25   0.25   0.25 ]   token 3: trung binh ca 4
  ^
 dich (token dang tinh context)
```

Phần _tam giác trên bằng 0_ = _causal mask_: token không được “nhìn” token đứng sau. Khi nhân `wei @ x`, mỗi hàng của `wei` chọn và cộng có trọng số các token trong `x`:

```python
# Build lower-triangular matrix of ones, shape (T, T)
tril = torch.tril(torch.ones(T, T)) # 1 on/below diagonal, 0 above (causal)

# Normalize each row to sum to 1 -> equal-weight average over allowed tokens
wei = tril / tril.sum(dim=1, keepdim=True) # wei shape: (T, T)
print(wei)
# tensor([[1.0000, 0.0000, 0.0000, 0.0000],
#         [0.5000, 0.5000, 0.0000, 0.0000],
#         [0.3333, 0.3333, 0.3333, 0.0000],
#         [0.2500, 0.2500, 0.2500, 0.2500]])

# Matrix multiply: (T, T) @ (B, T, C) broadcasts to (B, T, C)
xbow2 = wei @ x # xbow2 shape: (2, 4, 3)
print(torch.allclose(xbow, xbow2)) # True - same result as the loop, but vectorized
```

💡 Phép nhân `(T, T) @ (B, T, C)`: PyTorch tự _broadcast_ ma trận `(T, T)` cho từng phần tử của batch, cho ra `(B, T, C)`. Toàn bộ vòng lặp Python biến mất - chạy song song trên GPU, nhanh hơn hàng nghìn lần.

---

### 4. Cách 3 - dùng SOFTMAX (dọn đường cho attention) 🔑

Cách 2 dùng `wei` cố định (trọng số đều). Nhưng nếu ta muốn trọng số _học được_, ta cần một dạng linh hoạt hơn. Mẹo: bắt đầu bằng ma trận toàn 0, _che (mask)_ phần tương lai thành `-inf`, rồi cho qua `softmax` theo từng hàng.

```python
import torch.nn.functional as F

tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T)) # start: no preference, all zeros
wei = wei.masked_fill(tril == 0, float('-inf')) # future positions -> -inf (causal mask)
wei = F.softmax(wei, dim=-1) # softmax over each row -> weights sum to 1

# exp(-inf) = 0, so masked spots get weight 0; equal scores -> equal weights
xbow3 = wei @ x # xbow3 shape: (2, 4, 3)
print(torch.allclose(xbow2, xbow3)) # True
```

Vì sao dùng `softmax`? `softmax` biến một hàng số thực _bất kỳ_ (gọi là _scores_ hay _affinities_ - độ hợp) thành _phân phối xác suất_ (không âm, cộng lại bằng 1). Vị trí bị mask nhận `-inf`, mà $\exp(-\infty) = 0$, nên trọng số của chúng thành 0 - token quá khứ được giữ, tương lai bị bỏ. Khi mọi score bằng 0 (như trên), `softmax` cho ra _trọng số đều_ - đúng bằng trung bình. Nhưng nếu các score _khác nhau_ (được tính từ dữ liệu), token nào score cao sẽ được chú ý nhiều hơn - _đó chính là attention!_

```text
Cau chuyen mot dong:

  scores (T x T)  --mask tuong lai=-inf-->  softmax(moi hang)  -->  wei (tong hang=1)  -->  wei @ x
     |                                                                                          |
  score bang nhau  -------------------------------------------------->  trung binh deu (bag-of-words)
  score hoc duoc   -------------------------------------------------->  ATTENTION (bai 12)
```

---

### 5. Vì sao trung bình đều là điểm yếu

Trung bình đối xử _mọi token quá khứ như nhau_. Với câu _“Con mèo đen ngồi trên thảm, nó đang ngủ”_, khi xử lý từ _“nó”_, trung bình đều pha trộn “con”, “mèo”, “đen”, “ngồi”, “trên”, “thảm” với trọng số y hệt - làm loãng tín hiệu quan trọng (“mèo”) giữa đống nhiễu (“trên”, “thảm”). Ta muốn mô hình _tự học_ rằng “nó” nên chú ý mạnh vào “mèo”.

| Tiêu chí                                  | Trung bình đều (bag-of-words)  | Attention (bài sau)                |
| :---------------------------------------- | :----------------------------- | :--------------------------------- |
| **Trọng số**                              | Cố định, bằng nhau ($1/t$)     | Học được, khác nhau từng cặp token |
| **Phụ thuộc dữ liệu**                     | Không                          | Có (tính từ chính nội dung token)  |
| **Giữ token quan trọng**                  | Không (bị pha loãng)           | Có (score cao -> trọng số cao)     |
| **Causal (không nhìn tương lai)**         | Có (nhờ tam giác dưới)         | Có (cùng causal mask)              |
| **Là trường hợp đặc biệt của attention?** | Đúng - khi mọi score bằng nhau | -                                  |

⚠️ Đừng nhầm: trung bình đều _không hề tệ vô dụng_ - nó là bước đệm tuyệt vời để hiểu bộ khung (tam giác dưới + `softmax` + `wei @ x`). Attention _giữ nguyên bộ khung này_, chỉ thay “score bằng nhau” bằng “score học được từ Q và K”. Bạn đã hiểu 80% cơ chế attention rồi đấy!

---

### Nhìn sâu hơn

- **Chi phí tính toán:** Ma trận `wei` có shape `(T, T)`. Với chuỗi dài $T$, riêng ma trận này tốn bộ nhớ và phép tính tỉ lệ với $T^2$. Đây là lý do các mô hình cần _context dài_ (nhiều token) tốn kém - độ phức tạp attention là _$\mathcal{O}(T^2)$_. Trung bình đều cũng chịu chi phí $T^2$ nếu làm bằng ma trận, nhưng nó có thể tính bằng tổng tích lũy (cumulative sum) $\mathcal{O}(T)$ - trong khi attention thật thì bắt buộc $T^2$ vì score phụ thuộc từng cặp token.
- **Vì sao chuẩn hóa (chia cho tổng hàng)?** Nếu chỉ cộng mà không chia, token cuối (cộng 4 vector) sẽ có độ lớn gấp ~4 lần token đầu (chỉ 1 vector). Chia cho tổng trọng số giữ cho các context có _cùng thang độ lớn_ (scale), ổn định huấn luyện. `softmax` tự lo việc này vì đầu ra luôn cộng bằng 1.
- **Gotcha `-inf`:** Phải mask _trước_ khi `softmax`, không phải sau. Nếu `softmax` trước rồi mới đặt 0, các hàng sẽ không còn cộng bằng 1. Ngoài ra, `masked_fill(tril == 0, float('-inf'))` mask theo _giá trị 0 của `tril`_, không phải chỉ số - dễ sai nếu bạn tự viết điều kiện ngược.

---

### Thực hành

- **[Dễ]** Chạy lại 3 cách (vòng lặp, ma trận, `softmax`) với $B, T, C = 1, 3, 2$ và in `xbow`. Kiểm tra `torch.allclose` giữa cả 3 kết quả.
- **[Trung bình]** In ma trận `wei` sau `softmax` cho $T=5$. Xác nhận mỗi hàng cộng bằng 1 (`wei.sum(dim=-1)`) và phần tam giác trên bằng 0. Giải thích vì sao hàng đầu tiên là `[1, 0, 0, 0, 0]`.
- **[Thử thách]** Thay `wei = torch.zeros((T, T))` bằng `wei = torch.randn((T, T))` (score ngẫu nhiên) rồi mask + `softmax`. Quan sát: giờ mỗi token chú ý các token quá khứ với _trọng số khác nhau_. Đây là bản nháp thô của attention - ở Bài 12 bạn sẽ thấy các score này được _tính từ Q và K_ thay vì random.

---

### Hiểu lầm thường gặp

- ❌ “Attention là một cơ chế hoàn toàn mới, khác hẳn trung bình.” ➔ ✅ Attention _chính là_ trung bình có trọng số; trung bình đều chỉ là trường hợp đặc biệt khi mọi trọng số bằng nhau.
- ❌ “Ta trung bình các token id (số nguyên).” ➔ ✅ Ta trung bình các _vector embedding_ (số thực nhiều chiều); trung bình id là vô nghĩa.
- ❌ “Mask tương lai bằng cách đặt trọng số = 0 sau `softmax`.” ➔ ✅ Phải đặt score = `-inf` _trước_ `softmax`; $\exp(-\infty) = 0$ nên trọng số tự thành 0 mà hàng vẫn cộng bằng 1.
- ❌ “Trung bình giữ được thứ tự và mức độ quan trọng của từ.” ➔ ✅ Trung bình đều _làm mất_ cả thứ tự lẫn mức quan trọng (bag-of-words); attention mới khôi phục được mức quan trọng.

---

### Mẹo ghi nhớ

- _Tam giác dưới = quá khứ, tam giác trên = cấm (tương lai)_.
- _`wei @ x`_: `wei` quyết định “chú ý bao nhiêu”, `x` là “nội dung”. Trung bình = `wei` toàn số bằng nhau; attention = `wei` học được.
- _Softmax + mask `-inf`_ là bộ đôi bất ly thân cho mọi attention causal.
- Mẹo một câu: _“Attention = trung bình, nhưng thông minh hơn.”_

---

### Tóm tắt trực quan

| Khái niệm                      | Một dòng                                                    |
| :----------------------------- | :---------------------------------------------------------- |
| **bag-of-words / trung bình**  | Gộp token quá khứ với trọng số bằng nhau, mất thứ tự        |
| **causal mask**                | Tam giác dưới - cấm nhìn token tương lai                    |
| **tril**                       | `torch.tril(torch.ones(T, T))` tạo ma trận tam giác dưới    |
| **wei @ x**                    | Nhân ma trận trọng số với embedding = gộp có trọng số       |
| **softmax(scores, mask=-inf)** | Biến score thành trọng số cộng bằng 1, bỏ tương lai         |
| **Trọng số học được**          | Cầu nối tới attention: score tính từ dữ liệu, không cố định |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- PyTorch docs: [pytorch.org/docs/stable](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [pytorch.org/tutorials](https://pytorch.org/tutorials/)
- nanoGPT (Andrej Karpathy): [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Attention Is All You Need: [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)

---

### Tổng kết

- Bigram chỉ thấy một token; “trung bình ngữ cảnh” (bag-of-words) là bước cải tiến đầu tiên để token nhìn được _cả quá khứ_.
- Trung bình có thể viết gọn thành _`wei @ x`_ với `wei` là ma trận _tam giác dưới đã chuẩn hóa_ - đây là causal mask.
- Dùng `softmax` với mask `-inf` cho ta đúng kết quả trung bình khi score bằng nhau, và _để ngỏ cánh cửa_ cho trọng số học được.
- Điểm yếu của trung bình đều: cào bằng mọi token, làm loãng tín hiệu quan trọng.

Bạn đã nắm trọn bộ khung: `tril` -> mask `-inf` -> `softmax` -> `wei @ x`. Ở _Bài 12: Self-attention từ đầu - Query, Key, Value_, ta sẽ thay “score bằng nhau” bằng score _tính từ Q (câu hỏi) và K (khóa)_ của chính các token - và bạn sẽ tận mắt thấy trung bình biến thành attention thực thụ! 🚀
