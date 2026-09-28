# Bài 17: Mạng feed-forward trong Transformer

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 4:** Transformer block

---

### Giới thiệu

Ở _Bài 16: Multi-head attention - nhiều góc nhìn song song_, chúng ta đã để token “nhìn” chuỗi từ nhiều lăng kính cùng lúc: nhiều head chạy song song rồi nối kết quả lại. Sau bước đó, mỗi token đã _thu thập_ được thông tin từ những token quá khứ liên quan tới nó. Nhưng thu thập xong rồi thì sao? Ai sẽ ngồi lại _suy nghĩ_ trên đống thông tin vừa gom được?

Hãy tưởng tượng một buổi họp: phần attention giống lúc bạn đi hỏi han khắp phòng, gom ý kiến của mọi người liên quan. Nhưng sau khi gom xong, bạn cần _về bàn của mình, một mình ngồi ngẫm_ và rút ra kết luận riêng. Chính “khoảnh khắc ngồi ngẫm độc lập” đó là vai trò của _mạng feed-forward (FFN)_ trong mỗi Transformer block. Nó xử lý _từng token một cách độc lập_, không cho token này ngó sang token khác nữa - vì việc giao tiếp đã xong ở tầng attention rồi.

> 🔑 **Một câu để nhớ:** Attention là lúc token _giao tiếp_ với nhau; feed-forward là lúc mỗi token _tự suy nghĩ_ độc lập - một MLP nhỏ Linear → GELU → Linear áp riêng cho từng vị trí.

---

### Bạn sẽ học được gì?

- [ ] Hiểu vai trò “suy nghĩ” của FFN sau khi attention đã “thu thập ngữ cảnh”.
- [ ] Nắm cấu trúc chính xác: `Linear(C → 4C) → activation (GELU/ReLU) → Linear(4C → C)`.
- [ ] Hiểu vì sao gọi là _position-wise_ (áp cùng một mạng cho mọi vị trí, độc lập).
- [ ] Theo dõi _shape tensor_ đi qua FFN: `(B, T, C) → (B, T, 4C) → (B, T, C)`.
- [ ] Cài đặt module `FeedForward` bằng `nn.Sequential` và chạy thử.
- [ ] Phân biệt rõ _attention_ (trộn giữa các token) với _feed-forward_ (xử lý mỗi token riêng).

📌 **Yêu cầu trước:** Bài 15 (một attention head), Bài 16 (multi-head). Biết `nn.Linear`, `nn.Sequential`, hàm kích hoạt ReLU/GELU, và tư duy shape tensor `(B, T, C)`.

---

### 1. Vì sao attention thôi là chưa đủ?

**Analogy:** attention giống một _thư viện dùng chung_ - mỗi token đi mượn thông tin từ các token khác. Nhưng phép attention cốt lõi chỉ là _trung bình có trọng số của các Value_ (`softmax(...) @ V`). Đó là một phép _tuyến tính_ trên các token: nó _trộn_ chứ không _biến đổi phi tuyến_ nội dung của từng token.

Nếu bạn xếp chồng nhiều tầng attention mà không có gì ở giữa, về bản chất bạn vẫn chỉ đang làm các phép trộn tuyến tính lồng nhau - khả năng biểu diễn bị hạn chế. Mô hình cần một chỗ để _biến đổi phi tuyến_ đặc trưng của mỗi token: nhận diện mẫu, kết hợp đặc trưng, “suy luận” cục bộ. Đó chính là feed-forward network.

💡 **Quy tắc trực giác:** _attention = communication (giao tiếp giữa các vị trí)_, _feed-forward = computation (tính toán tại mỗi vị trí)_. Một Transformer block xen kẽ hai pha này.

---

### 2. Cấu trúc chính xác của FFN

Mạng feed-forward trong Transformer gốc (“Attention Is All You Need”) gồm _hai lớp Linear_ với một hàm kích hoạt phi tuyến ở giữa:

$$\text{FFN}(x) = W_2 \cdot \text{activation}(W_1 \cdot x + b_1) + b_2$$

trong đó:

- $W_1$ _nở rộng_ chiều từ $C$ (`n_embd`) lên $d_{ff}$, thường $d_{ff} = 4 \times C$.
- `activation` là _ReLU_ (bản gốc 2017) hoặc _GELU_ (GPT-2 trở đi dùng GELU - mượt hơn, cho gradient tốt hơn).
- $W_2$ _nén_ chiều từ $d_{ff}$ trở lại $C$.

Sơ đồ luồng và shape (với $C = 8, d_{ff} = 4C = 32$):

```text
x                (B, T, C)    = (2, 4, 8)
      | Linear W1 (C -> 4C)
h                (B, T, 4C)   = (2, 4, 32)     # "nở" ra để co dan dac trung
      | GELU (phi tuyen, ap tung phan tu)
h_act            (B, T, 4C)   = (2, 4, 32)     # shape khong doi
      | Linear W2 (4C -> C)
out              (B, T, C)    = (2, 4, 8)      # "nen" ve lai C
```

Điểm mấu chốt: _shape vào bằng shape ra_ (`(B, T, C) → (B, T, C)`). FFN chỉ _biến đổi_ nội dung mỗi token, giữ nguyên kích thước để còn cộng residual (Bài 18) và xếp block tiếp theo.

🎯 **Vì sao nở gấp 4 lần rồi nén lại?** Chiều ẩn rộng ($4C$) cho mạng một “không gian làm việc” lớn để tách và tổ hợp đặc trưng phi tuyến, trước khi ép về lại $C$. Con số 4 là quy ước từ paper gốc; không thiêng liêng nhưng rất phổ biến.

---

### 3. "Position-wise" nghĩa là gì?

Tên đầy đủ trong paper là _position-wise feed-forward network_. “Position-wise” nghĩa là: _cùng một_ cặp trọng số ($W_1, W_2$) được áp cho _mọi vị trí token_ trong chuỗi, _độc lập_ với nhau.

So sánh với attention:

```text
k0    k1    k2    k3           token 0   token 1   token 2   token 3
Attention:  q_i nhin sang cac token khac    <----- CO trao doi ----->
FFN:        moi token di qua CUNG 1 MLP     [MLP]     [MLP]     [MLP]     [MLP]
                                              |         |         |         |
                                          KHONG token nao nhin sang token khac
```

Vì FFN xử lý mỗi vị trí riêng, ta chỉ cần định nghĩa nó cho _một vector $C$ chiều_, rồi PyTorch tự áp song song cho cả $(B, T)$ vị trí nhờ cơ chế broadcasting của `nn.Linear` (Linear luôn tác động lên _chiều cuối cùng_).

⚠️ Đừng nhầm: FFN _không_ phải là nơi token trao đổi thông tin. Nếu token 3 học được điều gì từ token 0, điều đó xảy ra ở tầng _attention_, không phải ở FFN. FFN chỉ “tiêu hóa” thứ mà attention đã mang tới cho từng token.

---

### 4. Cài đặt module FeedForward (PyTorch)

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network of one Transformer block.
    Applied INDEPENDENTLY to each token position.
    """
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        # Expand C -> 4C, apply non-linearity, project 4C -> C back.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # (B, T, C) -> (B, T, 4C)
            nn.GELU(),                     # GPT-2 style non-linearity
            nn.Linear(4 * n_embd, n_embd), # (B, T, 4C) -> (B, T, C)
            nn.Dropout(dropout),           # regularization after projection
        )

    def forward(self, x):
        # x: (B, T, C) -> returns (B, T, C), same shape
        return self.net(x)
```

Chạy thử trên dữ liệu giả để kiểm tra shape:

```python
torch.manual_seed(1337)
B, T, C = 2, 4, 8
ff = FeedForward(n_embd=C, dropout=0.0)

x = torch.randn(B, T, C) # fake token features: (2, 4, 8)
out = ff(x)
print(out.shape)          # -> torch.Size([2, 4, 8]) == (B, T, C)
print(out.shape == x.shape) # -> True (shape preserved)
```

Kiểm chứng tính “position-wise” (mỗi token đi qua cùng một hàm, độc lập). Nếu ta đưa _một token duy nhất_ qua mạng, kết quả phải trùng với đúng vị trí đó khi đưa cả chuỗi:

```python
# Feed only the first token of the first sample: shape (1, 1, C)
single = x[0:1, 0:1, :] # (1, 1, 8)
out_single = ff(single) # (1, 1, 8)

# It must match position 0 of the full-sequence output (positions are independent).
print(torch.allclose(out_single[0, 0], out[0, 0], atol=1e-6))
# -> True # proves FFN treats each position independently
```

Kết quả `True` chứng minh: FFN xử lý từng vị trí độc lập - đổi các token khác trong chuỗi _không_ ảnh hưởng tới đầu ra tại vị trí 0. ✅

---

### 5. GELU và ReLU: chọn hàm kích hoạt nào?

```python
import torch
import torch.nn.functional as F

z = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(F.relu(z)) # -> tensor([0., 0., 0., 1., 3.]) # cut sach phan am
print(F.gelu(z)) # -> approx tensor([-0.0040, -0.1587, 0.0000, 0.8413, 2.9960])
```

- _ReLU_ đặt thẳng $\max(0, x)$ - đơn giản, nhanh, nhưng “gãy khúc” tại 0 và triệt tiêu hẳn tín hiệu âm.
- _GELU_ (Gaussian Error Linear Unit) là phiên bản mượt: nó nhân đầu vào với xác suất giữ lại theo phân phối chuẩn, nên vùng quanh 0 được làm trơn, giữ lại một chút tín hiệu âm nhỏ. GPT-2/GPT-3 dùng GELU vì gradient mượt giúp huấn luyện ổn định hơn.

| Hàm      | Công thức (rút gọn)                    | Đặc điểm                                   |
| :------- | :------------------------------------- | :----------------------------------------- |
| **ReLU** | $\max(0, x)$                           | Nhanh, đơn giản, gãy khúc tại 0            |
| **GELU** | $x \cdot \Phi(x)$ ($\Phi$ = CDF chuẩn) | Mượt, giữ chút tín hiệu âm, dùng trong GPT |

💡 Với GPT mini của chúng ta, dùng `nn.GELU()` là lựa chọn an toàn và “đúng chuẩn GPT”. ReLU vẫn chạy tốt nếu bạn muốn đơn giản hóa.

---

### Nhìn sâu hơn

- **FFN chiếm phần lớn tham số của mô hình.** Với một block, attention có tham số cỡ $\sim 4 \cdot C^2$ (các phép chiếu Q/K/V/output), còn FFN có $\sim 2 \cdot (C \cdot 4C) = 8 \cdot C^2$. Nghĩa là _FFN thường “nặng” gấp đôi attention_ về số tham số. Đây là lý do khi người ta scale mô hình lớn hơn, phần lớn trọng số nằm ở các lớp feed-forward.
- **Vì sao là 4C mà không phải 2C hay 8C?** Con số 4 là kết quả thực nghiệm cân bằng giữa sức mạnh biểu diễn và chi phí. Nở rộng hơn ($8C$) cho mô hình mạnh hơn nhưng tốn bộ nhớ/tính toán; hẹp hơn ($2C$) rẻ nhưng yếu hơn. Nhiều biến thể hiện đại (như dùng SwiGLU) thay đổi cả hàm kích hoạt lẫn tỉ lệ, nhưng $4C$ + GELU là điểm khởi đầu kinh điển.
- **Vì sao FFN không cần positional information?** Vì attention (kèm positional embedding ở đầu mô hình) đã lo phần thứ tự. FFN chỉ nhận vào một vector đặc trưng đã “thấm” ngữ cảnh và vị trí, rồi biến đổi nó. Bản thân FFN hoàn toàn “mù” về vị trí - nó áp y hệt cho mọi token.

🎯 Ghi nhớ đường đi trong một block: $x \to \text{attention (giao tiếp)} \to x \to \text{feed-forward (suy nghĩ)} \to x$. Bài 18 sẽ thêm _residual connection_ (cộng tắt) và _LayerNorm_ quanh hai pha này để mạng sâu vẫn huấn luyện được.

---

### Thực hành

- **[Dễ]** Đổi `n_embd` từ 8 sang 32 rồi in shape các lớp bên trong: `ff.net[0].weight.shape` (phải là `(128, 32)` tức $4C \times C$) và `ff.net[2].weight.shape` (phải là `(32, 128)`). Giải thích vì sao chiều ẩn là $4 \cdot 32 = 128$.
- **[Trung bình]** Thay `nn.GELU()` bằng `nn.ReLU()` và chạy lại đoạn kiểm tra `torch.allclose` ở mục 4. Xác nhận tính position-wise _vẫn đúng_ (kết quả `True`) - chứng tỏ tính chất này đến từ _cấu trúc Linear áp lên chiều cuối_, không phụ thuộc chọn hàm kích hoạt nào.
- **[Thử thách]** Đếm số tham số của `FeedForward(n_embd=64)` bằng `sum(p.numel() for p in ff.parameters())`. So sánh với công thức lý thuyết $2 \cdot (C \cdot 4C) + 4C + C$ (hai ma trận + hai vector bias). Với $C=64$, con số phải khớp $2 \cdot 64 \cdot 256 + 256 + 64 = 33088$.

---

### Hiểu lầm thường gặp

- ❌ “Feed-forward cũng cho các token trao đổi thông tin với nhau như attention.” ➔ ✅ Sai. FFN áp _độc lập_ cho từng vị trí; việc trao đổi giữa các token chỉ xảy ra ở _attention_.
- ❌ “FFN chỉ là một lớp Linear đơn giản.” ➔ ✅ FFN là _hai_ lớp Linear với một _hàm kích hoạt phi tuyến_ ở giữa. Bỏ phi tuyến đi thì hai Linear gộp lại thành một Linear duy nhất - mất sức mạnh.
- ❌ “Chiều ẩn phải bằng C.” ➔ ✅ Không. Chiều ẩn thường là $4C$ (nở rộng rồi nén lại). Đó là điểm cốt lõi khiến FFN mạnh.
- ❌ “GELU và ReLU cho kết quả giống hệt nhau nên chọn cái nào cũng vô nghĩa.” ➔ ✅ Chúng khác nhau: GELU mượt và giữ chút tín hiệu âm, thường ổn định hơn khi huấn luyện GPT; ReLU cắt sạch phần âm.
- ❌ “FFN thay đổi shape của tensor.” ➔ ✅ Shape vào _bằng_ shape ra: `(B, T, C) → (B, T, C)`. Chiều $4C$ chỉ là trạng thái tạm ở giữa.

---

### Mẹo ghi nhớ

- Câu thần chú: _“nở ra 4 lần, uốn cong phi tuyến, nén về lại”_ ($C \to 4C \to \text{GELU} \to C$).
- Nhớ cặp đôi: _attention = communication_, _feed-forward = computation_.
- Nhớ shape: vào $(B, T, C)$, giữa $(B, T, 4C)$, ra $(B, T, C)$ - luôn về lại $C$.
- Nhớ “position-wise” = cùng một MLP cho mọi token, _không_ ai nhìn sang ai.
- Bỏ hàm kích hoạt = hai Linear sụp thành một $\to$ nhớ luôn có phi tuyến ở giữa.

---

### Tóm tắt trực quan

| Khái niệm              | Một dòng                                                      |
| :--------------------- | :------------------------------------------------------------ |
| **Feed-forward (FFN)** | MLP nhỏ áp riêng từng token: Linear → GELU → Linear           |
| **Vai trò**            | “Suy nghĩ” phi tuyến sau khi attention đã “thu thập ngữ cảnh” |
| **Position-wise**      | Cùng một ($W_1, W_2$) cho mọi vị trí, độc lập nhau            |
| **Chiều ẩn**           | Thường $d_{ff} = 4C$ (nở rộng rồi nén lại)                    |
| **Kích hoạt**          | GELU (GPT) hoặc ReLU (bản gốc); phải phi tuyến                |
| **Shape**              | $(B, T, C) \to (B, T, 4C) \to (B, T, C)$ (vào = ra)           |
| **Chi phí**            | FFN thường nặng gấp đôi attention về số tham số ($\sim 8C^2$) |
| **Không làm gì**       | KHÔNG cho token trao đổi thông tin (đó là việc của attention) |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Coding a Transformer from scratch on PyTorch, with full explanation, training and inference.](https://www.youtube.com/watch?v=ISNdQcPhsts)
- PyTorch `nn.Linear`, `nn.Sequential`, `nn.GELU`, `nn.ReLU`: [pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT (Andrej Karpathy) - tham chiếu triển khai FeedForward / Block: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- “Attention Is All You Need” (mục 3.3 Position-wise Feed-Forward Networks): [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

---

### Tổng kết

- Mạng _feed-forward (FFN)_ là pha “suy nghĩ” của Transformer: sau khi attention giúp token _thu thập_ ngữ cảnh, FFN _biến đổi phi tuyến_ đặc trưng của _từng token một cách độc lập_.
- Cấu trúc chuẩn: `Linear(C → 4C) → GELU (hoặc ReLU) → Linear(4C → C)`, với shape đi từ `(B, T, C)` nở lên `(B, T, 4C)` rồi nén về lại `(B, T, C)`.
- “Position-wise” = cùng một MLP áp cho mọi vị trí; FFN _không_ cho token trao đổi thông tin - đó là việc của attention. Phải có hàm kích hoạt phi tuyến ở giữa, nếu không hai Linear sụp thành một.
- FFN thường chiếm phần lớn tham số của mô hình ($\sim 8C^2$), nặng gấp đôi attention.

Giờ ta đã có đủ hai mảnh ghép của một Transformer block: _multi-head attention_ (giao tiếp) và _feed-forward_ (suy nghĩ). Nhưng nếu chỉ xếp chồng chúng lên nhau, mạng sâu sẽ rất khó huấn luyện - gradient dễ biến mất và giá trị dễ trôi lệch. Ở _Bài 18: Residual connection và Layer Normalization_, chúng ta sẽ thêm hai “chất keo” then chốt: đường tắt cộng residual và chuẩn hóa LayerNorm, để ghép attention + feed-forward thành một block hoàn chỉnh, huấn luyện được ở độ sâu lớn. Hẹn gặp ở bài kế tiếp! 🚀
