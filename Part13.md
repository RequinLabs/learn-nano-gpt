# Bài 13: Scaled dot-product attention

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 3:** Cơ chế Attention

---

### Giới thiệu

Ở Bài 12 bạn đã tự tay dựng self-attention: mỗi token sinh ra ba vector _Query (Q)_, _Key (K)_ và _Value (V)_. Query như câu hỏi “mình đang tìm loại thông tin gì?”, Key như tấm biển “mình chứa loại thông tin này”, còn Value là nội dung thật sự được lấy về. Điểm mấu chốt: một token “chú ý” tới token khác nhiều hay ít được quyết định bằng _tích vô hướng (dot product) giữa Q của nó và K của token kia_.

Nhưng bạn còn nhớ chỗ mình đã “làm tắt” không? Khi tính điểm chú ý, ta lấy `Q @ K.T` rồi cho thẳng qua `softmax`. Bài này sẽ vá đúng một chi tiết nhỏ mà cực kỳ quan trọng: phải _chia cho căn bậc hai của $d_k$_ trước khi `softmax`. Nghe thì lặt vặt, nhưng thiếu nó là mô hình lớn có thể _không học nổi_ vì gradient bị “bóp nghẹt”. Đây chính là công thức nổi tiếng trong paper _Attention Is All You Need_:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Hãy tưởng tượng một lớp học ồn ào: nếu ai cũng hét thật to (điểm số quá lớn), thầy giáo chỉ nghe được một người duy nhất và “phớt lờ” tất cả người còn lại. Chia cho $\sqrt{d_k}$ giống như bảo cả lớp “nói nhỏ vừa đủ nghe” để thầy còn phân bổ sự chú ý một cách hợp lý.

> 🔑 **Một câu để nhớ:** Chia $QK^T$ cho $\sqrt{d_k}$ giữ cho điểm số không quá lớn, nhờ đó softmax không bão hòa và gradient còn chảy được để mô hình học.

---

### Bạn sẽ học được gì?

- [ ] Viết đúng và hiểu từng thành phần của công thức $\text{softmax}(QK^T / \sqrt{d_k}) V$
- [ ] Giải thích _vì sao chia $\sqrt{d_k}$_ bằng lý do phương sai (variance) và bão hòa softmax
- [ ] Theo dõi _shape của tensor_ qua từng bước của scaled dot-product attention
- [ ] Cài đặt hàm attention bằng PyTorch (`torch`, `F.softmax`) chạy được, có ví dụ số
- [ ] Nhận ra vì sao bước này là “trái tim” của mọi Transformer

📌 **Yêu cầu trước:** Bài 12 (Q/K/V và self-attention thô), biết `torch.matmul` / `@`, `softmax`, và tư duy shape `(batch, seq_len, dim)`.

---

### 1. Nhắc lại: điểm chú ý là tích vô hướng

Với mỗi cặp token $(i, j)$, điểm chú ý thô (attention score) là:

$$\text{score}(i, j) = Q_i \cdot K_j \quad (\text{tích vô hướng của 2 vector độ dài } d_k)$$

Tích vô hướng đo mức “khớp” giữa câu hỏi của token $i$ và tấm biển của token $j$: càng lớn nghĩa là càng liên quan. Gom lại thành ma trận, ta có $S = Q @ K^T$ với shape `(seq_len, seq_len)` - mỗi hàng là điểm mà một token chấm cho tất cả token (kể cả chính nó).

```text
Q: (T, d_k)      Kᵀ: (d_k, T)          S = Q @ Kᵀ: (T, T)
                                        ┌ s11 s12 s13 s14 ┐
[ q1 ]                                  │ s21 s22 s23 s24 │
[ q2 ]  @  [ k1 k2 k3 k4 ]  ────────►   │ s31 s32 s33 s34 │
[ q3 ]                                  │ s41 s42 s43 s44 │
[ q4 ]                                  └                 ┘
                                        hàng i = token i chấm điểm mọi token
```

---

### 2. Vấn đề: dot product lớn dần theo $d_k$

Đây là phần cốt lõi của bài. Giả sử các thành phần của $Q$ và $K$ là các số ngẫu nhiên _độc lập, trung bình 0, phương sai 1_. Tích vô hướng của hai vector độ dài $d_k$ là tổng của $d_k$ số hạng:

$$q \cdot k = \sum_{m=1}^{d_k} (q_m \cdot k_m)$$

Mỗi số hạng $q_m \cdot k_m$ có trung bình 0 và _phương sai 1_. Vì các số hạng độc lập, phương sai của tổng bằng tổng các phương sai:

$$\text{Var}(q \cdot k) = d_k \times 1 = d_k \implies \text{độ lệch chuẩn} \approx \sqrt{d_k}$$

Nói cách khác: _$d_k$ càng lớn, điểm số càng “phình” ra_ với độ lớn cỡ $\sqrt{d_k}$. Với $d_k = 64$ thì các điểm có thể xấp xỉ $\pm 8$; với $d_k = 512$ thì cỡ $\pm 22$. Nghe không to lắm? Nhưng softmax cực kỳ nhạy với độ lớn đầu vào.

⚠️ Đây không phải chuyện “cho đẹp”. Điểm quá lớn khiến softmax _bão hòa_: gần như dồn toàn bộ trọng số vào một token, các token khác nhận trọng số $\approx 0$.

---

### 3. Vì sao bão hòa softmax là thảm họa cho gradient

Nhớ lại softmax: nó biến một vector điểm thành phân phối xác suất. Khi một điểm lớn hơn hẳn phần còn lại, softmax cho ra thứ gần như one-hot: `[0.999, 0.0003, 0.0004, ...]`.

Vấn đề nằm ở _đạo hàm_. Với softmax, độ dốc (gradient) tỉ lệ với $p_i \cdot (1 - p_i)$. Khi $p_i \approx 1$ hoặc $p_i \approx 0$, tích này $\approx 0$ - gradient gần như biến mất (vanishing gradient). Mô hình “kẹt cứng”, không cập nhật được trọng số. Chia cho $\sqrt{d_k}$ kéo phương sai của điểm về lại $\approx 1$, giữ softmax ở vùng “mềm”, nơi gradient còn khỏe.

```text
Điểm KHÔNG chia √d_k (d_k lớn):     Điểm CÓ chia √d_k:
  scores = [12.0, 0.3, -8.1, 4.2]     scores = [1.5, 0.04, -1.0, 0.53]
  softmax ≈ [0.9997, ~0, ~0, ~0]      softmax ≈ [0.44, 0.10, 0.036, 0.16]
      → gradient ≈ 0 (kẹt)                → gradient khỏe (học được)
```

---

### 4. Công thức đầy đủ và tư duy shape

Ghép mọi thứ lại, scaled dot-product attention là:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Dõi theo shape từng bước (bỏ qua batch cho gọn, $T = \text{seq\_len}$):

```text
bước 1:  scores  = Q @ Kᵀ            (T, d_k) @ (d_k, T)  -> (T, T)
bước 2:  scaled  = scores / √d_k                          -> (T, T)   (chia từng phần tử)
bước 3:  weights = softmax(scaled, dim=-1)                -> (T, T)   (mỗi hàng cộng lại = 1)
bước 4:  out     = weights @ V       (T, T) @ (T, d_v)    -> (T, d_v)
```

`softmax` lấy `dim=-1` để _chuẩn hóa theo từng hàng_: mỗi token phân bổ 100% sự chú ý của nó cho các token khác. Bước cuối `weights @ V` là _trung bình có trọng số_ các Value - kết quả là biểu diễn mới của mỗi token, đã “trộn” thông tin từ những token nó chú ý nhất.

---

### 5. Cài đặt bằng PyTorch

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    # Q, K: (B, T, d_k) ; V: (B, T, d_v). Here we use d_v == d_k for simplicity.
    d_k = Q.size(-1) # scalar: last-dim size of the query/key vectors

    # Step 1: raw scores. K.transpose(-2, -1): (B, d_k, T)
    scores = Q @ K.transpose(-2, -1) # -> (B, T, T)

    # Step 2: scale by sqrt(d_k) to keep variance ~1 (prevents softmax saturation)
    scores = scores / (d_k ** 0.5) # -> (B, T, T)

    # Step 3: softmax over the LAST dim so each row sums to 1
    weights = F.softmax(scores, dim=-1) # -> (B, T, T)

    # Step 4: weighted sum of values
    out = weights @ V # (B, T, T) @ (B, T, d_v) -> (B, T, d_v)

    return out, weights

# Demo with a batch of 1, sequence length 4, head dim 8
torch.manual_seed(0)
B, T, d_k = 1, 4, 8
Q = torch.randn(B, T, d_k)
K = torch.randn(B, T, d_k)
V = torch.randn(B, T, d_k)

out, w = scaled_dot_product_attention(Q, K, V)
print(out.shape)              # torch.Size([1, 4, 8])
print(w.shape)                # torch.Size([1, 4, 4])
print(w[0].sum(dim=-1))       # tensor([1., 1., 1., 1.])  each row sums to 1
```

💡 `d_k ** 0.5` chính là $\sqrt{d_k}$. Đừng quên bước chia này - nó chỉ là một dòng nhưng là “linh hồn” của attention.

Kết quả mẫu (do có `manual_seed(0)`, con số của bạn sẽ giống): `out.shape = [1, 4, 8]`, `w.shape = [1, 4, 4]`, và tổng mỗi hàng của `w` đúng bằng `1.0`. Đó là bằng chứng weights là một phân phối xác suất hợp lệ trên các token.

---

### 6. Kiểm chứng "vì sao chia" bằng một thí nghiệm nhỏ

```python
import torch

# Compare score variance with a small vs a large d_k
for d_k in (8, 512):
    q = torch.randn(2000, d_k)      # 2000 random query vectors
    k = torch.randn(2000, d_k)      # 2000 random key vectors
    raw = (q * k).sum(dim=-1)       # dot products: (2000,)
    scaled = raw / (d_k ** 0.5)     # scaled version
    print(f"d_k={d_k:>3} | var(raw)={raw.var():7.2f} | var(scaled)={scaled.var():5.2f}")

# Sample output (values close to):
# d_k=  8 | var(raw)=   8.0x | var(scaled)= 1.0x
# d_k=512 | var(raw)= 512.xx | var(scaled)= 1.0x
```

Thấy chưa? Phương sai của điểm thô _bằng đúng $d_k$_ như phần lý thuyết đã dự đoán, còn phương sai sau khi chia luôn quanh 1.0 bất kể $d_k$. Chia cho $\sqrt{d_k}$ là cách “chuẩn hóa” để attention hoạt động ổn định ở mọi kích thước mô hình.

---

### Nhìn sâu hơn

- **$d_k$ là chiều của một head, không phải chiều embedding.** Trong multi-head (bài sau), embedding `d_model` được chia thành `n_head` head, mỗi head có $d_k = \text{d\_model} / \text{n\_head}$. Ví dụ `d_model = 512`, `n_head = 8` $\to d_k = 64 \to$ chia $\sqrt{64} = 8$.
- **Chi phí tính toán.** Ma trận scores có shape `(T, T)`, nên bộ nhớ và phép tính của attention tăng theo $\mathcal{O}(T^2)$ với độ dài chuỗi $T$. Đây là lý do các mô hình context dài rất tốn tài nguyên, và là động lực cho các biến thể như FlashAttention (tính cùng công thức nhưng tiết kiệm bộ nhớ).
- **PyTorch có sẵn.** `F.scaled_dot_product_attention(Q, K, V)` (từ PyTorch 2.0) làm đúng công thức này và tự tối ưu. Tự cài tay như trên là để _hiểu bản chất_; khi làm thật bạn có thể dùng hàm dựng sẵn.
- **Ổn định số học.** Cài softmax “trần” dễ tràn số với điểm lớn; PyTorch trừ đi max mỗi hàng trước khi mũ hóa. Đây là một lý do nữa nên dùng `F.softmax` thay vì tự viết.

---

### Thực hành

- **[Dễ]** Chạy đoạn code mục 5 với $T = 6$, in `w[0]` và xác nhận _mỗi hàng cộng lại bằng 1_. Đổi shape của $V$ thành `(B, T, 16)` (đặt $d_v = 16$) và kiểm tra `out.shape` trở thành `(1, 6, 16)`.
- **[Trung bình]** Bỏ dòng chia `/ (d_k ** 0.5)`, đặt $d_k = 256$, rồi in `w[0]`. Quan sát các hàng trở nên gần one-hot (một trọng số $\approx 1$, còn lại $\approx 0$). Sau đó bật lại dòng chia và so sánh.
- **[Thử thách]** Viết một hàm `attention_stats(d_k)` chạy thí nghiệm mục 6 và trả về `(var_raw, var_scaled)`. Vẽ (hoặc in) `var_raw` theo $d_k \in \{4, 16, 64, 256, 1024\}$ và xác nhận nó tăng tuyến tính theo $d_k$, còn `var_scaled` luôn $\approx 1$.

---

### Hiểu lầm thường gặp

- ❌ “Chia $\sqrt{d_k}$ chỉ để làm số nhỏ cho đẹp.” ➔ ✅ Nó _chuẩn hóa phương sai_ của điểm về $\approx 1$, giữ softmax không bão hòa nên gradient còn chảy - đây là lý do toán học rõ ràng, không phải trang trí.
- ❌ “Chia cho $d_k$ (không lấy căn).” ➔ ✅ Phải chia cho _$\sqrt{d_k}$_ vì độ lệch chuẩn của tổng là $\sqrt{d_k}$. Chia cho $d_k$ sẽ “bóp” điểm quá mức.
- ❌ “`softmax` lấy theo dim nào cũng được.” ➔ ✅ Phải lấy `dim=-1` để mỗi _token (mỗi hàng)_ phân bổ tổng 100% chú ý; lấy sai dim ra kết quả vô nghĩa.
- ❌ “Attention xử lý token tuần tự như RNN.” ➔ ✅ Toàn bộ ma trận $QK^T$ tính _song song_ một lần; attention không lặp theo thời gian.

---

### Mẹo ghi nhớ

- Công thức thần chú: _“soft-max của Q-K-transpose chia căn d-k, rồi nhân V”_.
- Nhớ chuỗi shape: $(T, d_k) \cdot (d_k, T) \to (T, T) \to \text{softmax} \to (T, T) \cdot (T, d_v) \to (T, d_v)$.
- $\sqrt{d_k}$ vì phương sai của tổng $d_k$ số hạng là $d_k$, nên độ lệch chuẩn là $\sqrt{d_k}$.
- 🎯 Mẹo: mỗi khi thấy $QK^T$, phản xạ tự hỏi “đã chia $\sqrt{d_k}$ chưa?”.

---

### Tóm tắt trực quan

| Khái niệm                          | Một dòng cốt lõi                                  |
| :--------------------------------- | :------------------------------------------------ |
| **scores = QKᵀ**                   | Điểm chú ý thô, shape (T, T)                      |
| **d_k**                            | Chiều của vector Q/K (chiều một head)             |
| **Chia √d_k**                      | Kéo phương sai điểm về ≈ 1, chống bão hòa softmax |
| **softmax(dim=-1)**                | Biến mỗi hàng thành phân phối xác suất (cộng = 1) |
| **weights @ V**                    | Trung bình có trọng số của Value → biểu diễn mới  |
| **Chi phí**                        | $\mathcal{O}(T^2)$ theo độ dài chuỗi              |
| **F.scaled_dot_product_attention** | Hàm PyTorch dựng sẵn làm đúng công thức này       |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [The math behind Attention: Keys, Queries, and Values matrices](https://www.youtube.com/watch?v=UPtG_38Oq8o)
- Paper gốc “Attention Is All You Need”: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer (Jay Alammar): [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- PyTorch docs (`torch.nn.functional`): [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- nanoGPT (Andrej Karpathy): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

---

### Tổng kết

- Scaled dot-product attention: $\text{softmax}(QK^T / \sqrt{d_k}) V$ - trái tim của Transformer.
- $QK^T$ cho điểm chú ý, chia $\sqrt{d_k}$ giữ phương sai $\approx 1$, softmax biến thành trọng số, nhân $V$ để trộn thông tin.
- Chia $\sqrt{d_k}$ _không phải tùy chọn_: thiếu nó, với $d_k$ lớn, softmax bão hòa $\to$ gradient biến mất $\to$ mô hình không học.
- Cẩn thận shape: $(T, d_k) \to (T, T) \to (T, d_v)$, và softmax luôn `dim=-1`.

Nhưng khoan - hiện tại mỗi token đang “nhìn thấy” _cả câu_, kể cả các token ở _tương lai_. Với một mô hình sinh văn bản (dự đoán chữ tiếp theo), cho phép nhìn tương lai là _gian lận_! Ở _Bài 14: Causal mask cho mô hình sinh văn bản_, ta sẽ học cách “che” các token phía sau bằng một ma trận tam giác, để mỗi token chỉ được chú ý tới quá khứ và chính nó. Hẹn gặp lại! 🤖
