# Bài 16: Multi-head attention - nhiều góc nhìn song song

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 4:** Transformer block

---

### Giới thiệu

Ở _Bài 15_, bạn đã tự tay cài đặt trọn vẹn _một_ attention head bằng PyTorch: từ token gửi ra ba vector $Q$, $K$, $V$, tính điểm tương đồng $QK^T$, chia cho $\sqrt{d_k}$, che tương lai bằng _causal mask_, softmax rồi nhân với $V$ để ra vector ngữ cảnh. Một cái đầu (head) đó đã khá thông minh - nhưng nó chỉ nhìn câu theo _một góc nhìn duy nhất_.

Hãy tưởng tượng bạn đưa một câu văn cho một hội đồng biên tập. Nếu chỉ có _một_ biên tập viên, người đó sẽ chú ý tới đúng một thứ - ví dụ ngữ pháp. Nhưng nếu bạn có _tám_ biên tập viên đọc _song song_: người soi ngữ pháp, người soi đại từ trỏ về ai, người soi giọng văn, người soi sự kiện… rồi cả nhóm gộp nhận xét lại - bạn sẽ hiểu câu văn sâu và đầy đủ hơn nhiều. Đó chính xác là tinh thần của _multi-head attention_ (chú ý đa đầu): chạy nhiều head _song song_, mỗi head học một kiểu quan hệ, rồi _nối_ (concat) và _chiếu lại_ thành một vector chung.

> 🔑 **Một câu để nhớ:** Multi-head attention = chia không gian embedding thành nhiều “kênh chú ý” nhỏ chạy song song, mỗi head học một loại quan hệ, cuối cùng nối lại và chiếu về không gian ban đầu.

---

### Bạn sẽ học được gì?

- [ ] Vì sao _một_ head là chưa đủ, và đa đầu mạnh hơn ở chỗ nào
- [ ] Cách _chia_ chiều embedding $d_{\text{model}}$ thành $n_{\text{head}}$ head, mỗi head có $d_k = d_{\text{model}} / n_{\text{head}}$
- [ ] Luồng shape tensor đi qua multi-head: split -> attention song song -> concat -> chiếu lại
- [ ] Cài đặt MultiHeadAttention gọn gàng bằng PyTorch (một phép chiếu QKV rồi reshape)
- [ ] Vì sao chi phí tính toán gần _không đổi_ dù thêm head

📌 **Yêu cầu trước:** bạn đã hiểu một attention head đơn (Bài 15): $Q$, $K$, $V$, $\text{softmax}(QK^T / \sqrt{d_k}) V$, causal mask.

---

### 1. Vì sao một head là chưa đủ?

Một head sinh _một_ ma trận trọng số chú ý cho mỗi token. Sau softmax, mỗi token phân bổ toàn bộ “sự chú ý” của nó lên các token khác _theo một cách duy nhất_. Vấn đề: trong ngôn ngữ tự nhiên, một từ thường liên quan tới nhiều từ khác vì _nhiều lý do khác nhau cùng lúc_.

Ví dụ trong câu “Con mèo mà nó nuôi rất dễ thương”: từ “nó” cần chú ý tới “mèo” (đại từ trỏ về danh từ nào), nhưng “dễ thương” lại cần chú ý tới “mèo” vì lý do khác (tính từ mô tả cái gì). Một head phải “nhồi” tất cả loại quan hệ này vào _một_ phân phối trọng số - dễ nhòe. Đa đầu cho phép mỗi loại quan hệ có _kênh riêng_.

💡 **Trực giác:** mỗi head là một “chuyên gia” học một kiểu liên hệ (cú pháp, đại từ, vị trí gần, ngữ nghĩa…). Mạng tự học phân công, ta không lập trình cứng head nào làm gì.

---

### 2. Ý tưởng: chia để trị không gian embedding

Giả sử $d_{\text{model}} = 8$ và ta muốn $n_{\text{head}} = 2$. Thay vì mỗi head làm việc trên cả 8 chiều, ta _chia_ thành 2 head, mỗi head chỉ làm việc trên $d_k = d_{\text{model}} / n_{\text{head}} = 4$ chiều. Mỗi head tự chạy scaled dot-product attention _độc lập_ trên phần chiều của mình, cho ra một vector $d_k$ chiều. Nối 2 vector $d_k$ lại được $n_{\text{head}} \times d_k = d_{\text{model}}$ chiều - đúng bằng kích thước ban đầu.

```text
Multi-head attention (n_head=2, d_model=8, d_k=4):

            x  (T, d_model=8)
            |
   +--------+--------+
   | chiếu Q,K,V rồi chia thành 2 head
   v                 v
 Head 0 (d_k=4)   Head 1 (d_k=4)
   |  attention      |  attention
   |  (mask nhân     |  (mask nhân
   |   quả riêng)    |   quả riêng)
   v                 v
 out0 (T,4)       out1 (T,4)
   |                 |
   +-------concat-----+   -> (T, 8)
            |
     chiếu lại W_o (8x8)
            v
         (T, d_model=8)
```

🎯 **Điểm mấu chốt:** $d_k = d_{\text{model}} / n_{\text{head}}$. Vì mỗi head làm việc trên chiều nhỏ hơn, tổng khối lượng tính toán của tất cả head _xấp xỉ bằng_ một head đơn trên toàn bộ $d_{\text{model}}$ - đa đầu gần như “miễn phí” về chi phí.

---

### 3. Luồng shape tensor qua multi-head

Gọi $B$ = batch size, $T$ = số token (độ dài chuỗi), $C = d_{\text{model}}$, $H = n_{\text{head}}$, $d_k = C / H$. Đây là hành trình shape từng bước:

```text
Input x:                 (B, T, C)
chiếu ra Q, K, V:        mỗi cái (B, T, C)
tách thành H head:       (B, T, C) -> (B, T, H, d_k) -> (B, H, T, d_k)
diem chu y QK^T:         (B, H, T, d_k) @ (B, H, d_k, T) = (B, H, T, T)
chia sqrt(d_k) + mask:   (B, H, T, T)
softmax theo chieu cuoi: (B, H, T, T)
nhan voi V:              (B, H, T, T) @ (B, H, T, d_k) = (B, H, T, d_k)
gop head lai (concat):   (B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, C)
chieu lai W_o:           (B, T, C) @ (C, C) = (B, T, C)
```

Hãy để ý: sau khi gộp head, ta được lại $(B, T, C)$ - _cùng shape với input_. Nhờ vậy multi-head attention có thể xếp chồng nhiều lớp mà shape luôn nhất quán.

---

### 4. Cài đặt MultiHeadAttention bằng PyTorch

Mẹo cài đặt hiệu quả: thay vì tạo $H$ lớp Linear riêng cho mỗi head, ta dùng _một_ phép chiếu `Linear(C, 3*C)` cho $Q$, $K$, $V$, rồi reshape để tách thành head. Cách này chạy tất cả head _song song_ bằng phép nhân ma trận theo batch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_k = d_model // n_head # per-head dimension

        # One Linear projects to Q, K, V at once: (C) -> (3*C)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model) # output projection W_o
        self.dropout = nn.Dropout(dropout)

        # Lower-triangular causal mask (1 = keep, 0 = block future)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape # (B, T, d_model)

        # Project once, then split into Q, K, V each (B, T, C)
        q, k, v = self.qkv(x).split(C, dim=2)

        # Reshape to separate heads: (B, T, C) -> (B, H, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # Scaled dot-product scores: (B, H, T, T)
        att = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Apply causal mask: block attention to future tokens
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1) # normalize over keys, (B, H, T, T)
        att = self.dropout(att)

        # Weighted sum of values: (B, H, T, d_k)
        out = att @ v

        # Merge heads back: (B, H, T, d_k) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection W_o
        return self.proj(out)

# Quick shape sanity check
B, T, d_model, n_head = 2, 5, 8, 2
mha = MultiHeadAttention(d_model, n_head, block_size=16)
x = torch.randn(B, T, d_model) # (2, 5, 8)
y = mha(x)
print(y.shape) # torch.Size([2, 5, 8]) -> same shape as input
```

⚠️ _Chú ý chia $\sqrt{d_k}$ chứ KHÔNG phải $\sqrt{d_{\text{model}}}$._ Vì mỗi head làm việc trên $d_k$ chiều nên tích vô hướng $QK^T$ được scale theo $\sqrt{d_k}$ để giữ phương sai điểm số ổn định trước softmax. Chia sai hằng số scale sẽ làm softmax quá nhọn (gradient nhỏ) hoặc quá phẳng.

---

### 5. Vì sao phải "chiếu lại" bằng W_o?

Sau khi concat, các vector từ các head khác nhau chỉ đơn thuần được nối cạnh nhau - chúng chưa “nói chuyện” với nhau. Phép chiếu cuối $W_o$ (một `Linear(C, C)`) _trộn_ thông tin từ mọi head lại, cho phép mô hình kết hợp góc nhìn của các head thành một biểu diễn thống nhất. Không có $W_o$, đa đầu chỉ là nhiều kênh rời rạc ghép lại.

---

### Nhìn sâu hơn

- **Số tham số:** một MultiHeadAttention với $d_{\text{model}} = C$ có QKV projection $3 \times C \times C$ và output projection $C \times C$, tổng $\sim 4 \times C^2$ tham số (bỏ qua bias) - _không phụ thuộc $n_{\text{head}}$_. Nghĩa là bạn có thể tăng số head mà số tham số \_không đổi_; bạn chỉ đang chia lại cùng lượng tham số thành nhiều kênh nhỏ hơn. Đây là lý do đa đầu “gần như miễn phí”.
- **Chi phí tính toán:** ma trận attention $(B, H, T, T)$ khiến bộ nhớ và tính toán tăng theo $\mathcal{O}(T^2)$ với độ dài chuỗi $T$. Đây là nút thắt cổ chai kinh điển của Transformer với chuỗi dài (nhiều head không làm điều này tệ hơn, vì $H \times d_k = C$). Các kỹ thuật như Flash Attention tối ưu chính bước này.
- **Đánh đổi số head:** nhiều head quá thì $d_k$ quá nhỏ, mỗi head không đủ chiều để biểu diễn quan hệ phong phú; ít head quá thì mất tính đa dạng góc nhìn. GPT-2 dùng $d_{\text{model}}=768, n_{\text{head}}=12$ ($d_k=64$); giá trị $d_k$ khoảng 64 là phổ biến.

💡 Một cài đặt sản xuất thường thay khối `softmax(QK^T/sqrt(d_k)) + mask` thủ công bằng `F.scaled_dot_product_attention(q, k, v, is_causal=True)` của PyTorch - hàm này gọi kernel tối ưu (Flash Attention khi có thể). Nhưng tự viết tay như trên giúp bạn _hiểu_ từng bước.

---

### Thực hành

- **[Dễ]** Chạy đoạn code trên với `d_model=16, n_head=4`. In `mha.d_k` và kiểm tra shape output `y.shape` vẫn là `(B, T, 16)`. Thử `n_head=3` (không chia hết 16) và quan sát assert báo lỗi.
- **[Trung bình]** Sửa `forward` để cũng _trả về_ ma trận `att` (sau softmax), shape `(B, H, T, T)`. In `att[0, 0]` cho một chuỗi ngắn và xác nhận nó là ma trận _tam giác dưới_ (phần trên đường chéo bằng 0 do causal mask).
- **[Thử thách]** So sánh `n_head=1` với `n_head=8` (cùng `d_model=64`) trên một bài toán đồ chơi (vd dự đoán ký tự tiếp theo trên một chuỗi nhỏ). Đếm số tham số bằng `sum(p.numel() for p in mha.parameters())` cho cả hai và xác nhận _bằng nhau_. Quan sát chất lượng học.

---

### Hiểu lầm thường gặp

- ❌ “Nhiều head thì nhiều tham số hơn.” ➔ ✅ Số tham số $\sim 4 \times C^2$ _không đổi_ theo $n_{\text{head}}$; ta chỉ chia lại cùng lượng tham số thành các kênh nhỏ.
- ❌ “Mỗi head chia $\sqrt{d_{\text{model}}}$.” ➔ ✅ Mỗi head chia $\sqrt{d_k}$ với $d_k = d_{\text{model}} / n_{\text{head}}$, vì head làm việc trên chiều $d_k$.
- ❌ “Các head chạy tuần tự.” ➔ ✅ Chúng chạy _song song_ - toàn bộ được gói vào một phép nhân ma trận theo batch trên trục head, không có vòng lặp tuần tự.
- ❌ “Concat xong là xong.” ➔ ✅ Vẫn cần chiếu lại $W_o$ để trộn thông tin giữa các head; thiếu bước này đa đầu chỉ là các kênh rời rạc.

---

### Mẹo ghi nhớ

- _Split -> Attend -> Concat -> Project_: bốn động từ tóm gọn multi-head attention.
- Nhớ hằng số scale là $\sqrt{d_k}$ (chiều _mỗi head_), không phải $\sqrt{d_{\text{model}}}$.
- Mẹo: hình dung `(B, H, T, d_k)` - trục $H$ (head) nằm ngay sau batch, để mọi phép attention chạy song song trên đó như thêm một “batch ảo”.

---

### Tóm tắt trực quan

| Khái niệm                | Một dòng                                                |
| :----------------------- | :------------------------------------------------------ |
| **Multi-head attention** | Nhiều head chạy song song, mỗi head một góc nhìn        |
| **d_k**                  | Chiều mỗi head = $d_{\text{model}} / n_{\text{head}}$   |
| **Split**                | Reshape `(B, T, C) -> (B, H, T, d_k)`                   |
| **Scale**                | Chia điểm $QK^T$ cho $\sqrt{d_k}$                       |
| **Causal mask**          | Che token tương lai (tam giác dưới) trên mỗi head       |
| **Concat**               | Gộp head: `(B, H, T, d_k) -> (B, T, C)`                 |
| **W_o**                  | Chiếu lại để trộn thông tin giữa các head               |
| **Số tham số**           | $\sim 4 \times C^2$, _không đổi_ theo $n_{\text{head}}$ |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Multi Head Attention in Transformer Neural Networks with Code!](https://www.youtube.com/watch?v=HQn1QKQYXVg)
- [Attention Is All You Need (paper gốc)](https://arxiv.org/abs/1706.03762) - mục 3.2.2 mô tả multi-head attention.
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - hình minh họa trực quan các head.
- [nanoGPT (Andrej Karpathy)](https://github.com/karpathy/nanoGPT) - file `model.py`, lớp `CausalSelfAttention`.
- [PyTorch docs - scaled_dot_product_attention](https://pytorch.org/docs/stable/) - kernel attention tối ưu sẵn có.
- [PyTorch tutorials](https://pytorch.org/tutorials/) - hướng dẫn `nn.Module` và tensor.

---

### Tổng kết

- Multi-head attention chạy $n_{\text{head}}$ head _song song_, mỗi head trên $d_k = d_{\text{model}} / n_{\text{head}}$ chiều, mỗi head học một kiểu quan hệ khác nhau.
- Luồng: _split_ thành head -> _attention_ (scale $\sqrt{d_k}$ + causal mask) song song -> _concat_ về `(B, T, C)` -> _chiếu lại_ $W_o$.
- Đa đầu mạnh hơn một đầu nhờ đa dạng góc nhìn, mà số tham số $\sim 4 \times C^2$ _không đổi_ theo số head - gần như miễn phí.
- Cài đặt gọn bằng một `Linear(C, 3*C)` cho QKV rồi reshape để chạy mọi head trong một phép nhân ma trận.

Giờ bạn đã có một khối attention hoàn chỉnh, nhưng attention mới chỉ giúp token “nhìn” nhau. Ở _Bài 17: Mạng feed-forward trong Transformer_, ta sẽ thêm khối feed-forward - nơi mỗi token được xử lý phi tuyến độc lập để “suy nghĩ” về thông tin vừa thu thập được. Hẹn gặp lại! 🤖
