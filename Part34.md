# Bài 34: Tăng tốc sinh văn bản với KV-cache

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 7:** Tokenization thực tế, tối ưu & tự host

---

### Giới thiệu

Ở [Bài 33](https://learningvn.com/learn/xay-dung-llm-tu-dau-tu-attention-den-gpt-mini-pytorch/lesson/cc8f1afd-7f13-4e10-8031-7d7d60b58180#) bạn đã học các chiến lược sinh văn bản nâng cao - temperature, top-k, top-p - để điều khiển mô hình sáng tạo hay chắc chắn. Nhưng bạn có để ý một điều khó chịu không: khi GPT mini sinh một đoạn văn dài, nó _chậm dần theo từng chữ_. Câu đầu ra vèo vèo, nhưng tới token thứ 200 thì cứ ì ạch. Vì sao vậy?

Hãy tưởng tượng bạn chép lại một bức thư dài. Mỗi khi viết thêm _một chữ mới_, thay vì chỉ viết chữ đó, bạn lại _chép lại toàn bộ những gì đã viết từ đầu_ rồi mới thêm chữ mới vào cuối. Chép chữ thứ 1 rất nhanh, nhưng tới chữ thứ 500 bạn phải chép lại 499 chữ cũ mỗi lần! Đó chính xác là cách sinh văn bản tự hồi quy (autoregressive) _ngây thơ_ đang lãng phí.

KV-cache (bộ nhớ đệm Key/Value) là mẹo để bạn _không phải chép lại_ phần cũ: những gì đã tính rồi thì lưu lại, chỉ tính phần mới. Đây là kỹ thuật tăng tốc inference quan trọng bậc nhất của mọi hệ thống LLM thật (ChatGPT, Llama…) đều dùng.

> 🔑 **Một câu để nhớ:** Sinh tự hồi quy ngây thơ tính lại attention cho TOÀN BỘ chuỗi ở mỗi bước; KV-cache lưu Key và Value của các token cũ để mỗi bước chỉ cần tính cho _một token mới_.

---

### Bạn sẽ học được gì?

- [ ] Vì sao sinh tự hồi quy (autoregressive) lại _chậm dần_ theo độ dài chuỗi
- [ ] Chính xác _Key_ và _Value_ nào có thể tái sử dụng, còn _Query_ thì không
- [ ] Cách cài đặt KV-cache trong self-attention có causal mask bằng PyTorch
- [ ] Phân tích _độ phức tạp tính toán_ trước và sau khi có cache
- [ ] Đánh đổi giữa _tốc độ_ và _bộ nhớ_ (VRAM) khi dùng cache

📌 **Yêu cầu trước:** bạn đã hiểu scaled dot-product attention (Bài 12-15), causal mask, và vòng lặp generate (Bài 32-33).

---

### 1. Vấn đề: sinh tự hồi quy ngây thơ tính lại mọi thứ

Nhớ lại cách GPT sinh văn bản: nó sinh _từng token một_. Ở mỗi bước, nó lấy toàn bộ chuỗi hiện có, đưa qua mô hình, lấy logits ở vị trí _cuối cùng_, chọn token tiếp theo, rồi nối vào chuỗi và lặp lại.

```python
import torch

# Naive autoregressive generation (the SLOW way)
# idx shape: (B, T) - B sequences, current length T
def generate_naive(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        # Re-process the ENTIRE sequence every step!
        logits = model(idx) # logits shape: (B, T, vocab_size)
        logits = logits[:, -1, :] # keep only last position: (B, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1) # (B, 1)
        idx = torch.cat([idx, next_id], dim=1) # T grows by 1 each step
    return idx
```

Vấn đề nằm ở dòng `model(idx)`: mỗi bước ta đưa _cả chuỗi độ dài T_ vào mô hình. Nhưng ta chỉ dùng `logits[:, -1, :]` - tức chỉ cần kết quả của _token cuối cùng_! Tất cả tính toán cho $T-1$ token đầu bị _vứt đi rồi tính lại_ ở bước sau.

Hãy nhìn khối lượng công việc phình lên theo từng bước:

```text
Bước sinh   Độ dài chuỗi   attention QK^T tính     (chép lại như bức thư)
---------   ------------   ---------------------   ---------------------
  1              1          1x1                     -
  2              2          2x2                     tính lại token 1
  3              3          3x3                     tính lại token 1,2
  ...            ...        ...                     ...
  N              N          NxN                     tính lại N-1 token cũ!
```

Tổng công việc để sinh $N$ token là $1^2 + 2^2 + \dots + N^2 \approx \mathcal{O}(N^3)$. Càng sinh dài, càng chậm khủng khiếp.

⚠️ Đây KHÔNG phải “attention vốn tuần tự như RNN”. Attention tính song song được; cái chậm ở đây là do _lặp lại tính toán đã làm_ giữa các bước sinh, chứ không phải bản chất attention.

---

### 2. Chìa khóa: Key và Value cũ KHÔNG đổi

Để hiểu vì sao cache được, ta nhìn lại self-attention có causal mask. Với mỗi token, ta tạo 3 vector: _Query (Q)_, _Key (K)_, _Value (V)_. Công thức attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{causal\_mask}\right) V$$

Điểm mấu chốt: token thứ $i$ được tạo ra chỉ dựa vào các token $0..i$ (causal mask che tương lai). Khi ta sinh _token mới_ (vị trí $t$), ta cần tính attention cho _riêng vị trí $t$_ với tất cả các token $0..t$. Cụ thể:

```text
Query: chỉ cần Q của token MỚI (vị trí t)              -> phải tính lại mỗi bước
Key:   cần K của TẤT CẢ token 0..t                     -> K[0..t-1] đã tính ở bước trước!
Value: cần V của TẤT CẢ token 0..t                     -> V[0..t-1] đã tính ở bước trước!
```

Vì các token $0..t-1$ đã cố định (đã sinh xong, không đổi nữa), nên _K và V của chúng cũng không đổi_. Ta chỉ cần lưu (cache) lại K, V cũ, rồi mỗi bước chỉ tính K, V, Q cho _đúng một token mới_, nối K/V mới vào cache.

```text
Sơ đồ attention khi sinh token mới t=3 (đã có cache K,V của token 0,1,2):

           K0   K1   K2   K3(mới)
         +----+----+----+----+
   Q3    | .  | .  | .  | .  |   <- CHỈ cần 1 hàng Query mới
   (mới) +----+----+----+----+       nhân với K0..K3 (dùng cache + K3 mới)

           đã cache ---^    ^--- vừa tính
```

So với ma trận đầy đủ $4 \times 4$ (16 ô) mà cách ngây thơ tính lại, ở đây ta chỉ tính _1 hàng_ (4 ô). Đó chính là chỗ tiết kiệm.

💡 Tại sao Query KHÔNG cache được? Vì mỗi bước ta hỏi một câu hỏi mới (“token tiếp theo là gì dựa trên vị trí hiện tại?”). Query cũ đã dùng xong rồi, không cần lưu.

---

### 3. Cài đặt KV-cache trong self-attention (PyTorch)

Dưới đây là một lớp self-attention nhân quả (causal) hỗ trợ cache. So với bản không cache, điểm khác là nó nhận `kv_cache` cũ và trả về `kv_cache` mới.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttentionKV(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model) # produces Q, K, V together
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        # x shape: (B, T, d_model). During generation with cache, T is usually 1.
        B, T, C = x.shape
        qkv = self.qkv(x) # (B, T, 3*d_model)
        q, k, v = qkv.split(C, dim=2) # each (B, T, d_model)

        # reshape to heads: (B, n_head, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # --- KV-CACHE: prepend previously computed K, V ---
        if kv_cache is not None:
            past_k, past_v = kv_cache # each (B, n_head, T_past, d_k)
            k = torch.cat([past_k, k], dim=2) # (B, n_head, T_past + T, d_k)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v) # return updated cache

        # scaled dot-product attention: softmax(QK^T / sqrt(d_k)) @ V
        att = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5) # (B, n_head, T, T_past+T)

        # NOTE: when T==1 (one new token) and using cache, no explicit causal mask
        # is needed because the new token attends to ALL cached keys (all in its past).
        att = F.softmax(att, dim=-1)
        out = att @ v # (B, n_head, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, T, d_model)
        return self.proj(out), new_cache
```

Điểm cần nhớ: khi sinh với cache, đầu vào `x` chỉ chứa _một token mới_ ($T == 1$). Ta tính Q/K/V cho token đó, _nối_ K/V mới vào K/V cũ trong cache, rồi làm attention. Ma trận `att` có shape `(B, n_head, 1, T_past+1)` - chỉ _một hàng_ thay vì ma trận vuông đầy đủ.

Vòng lặp generate dùng cache:

```python
@torch.no_grad()
def generate_with_cache(model, idx, max_new_tokens):
    # Step 1: prefill - process the whole prompt ONCE, build initial cache
    logits, cache = model(idx, kv_cache=None) # idx shape: (B, T_prompt)
    for _ in range(max_new_tokens):
        last_logits = logits[:, -1, :] # (B, vocab_size)
        probs = torch.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1) # (B, 1)
        idx = torch.cat([idx, next_id], dim=1)

        # Step 2: decode - feed ONLY the new token, reuse cache
        logits, cache = model(next_id, kv_cache=cache) # input shape (B, 1) !
    return idx
```

Hai giai đoạn rất rõ: _prefill_ (nạp prompt một lần, dựng cache) và _decode_ (mỗi bước chỉ đưa vào 1 token mới). Đây đúng là cách các engine inference thật (vLLM, llama.cpp…) chia pha.

---

### 4. Phân tích: nhanh hơn bao nhiêu?

Lập bảng so sánh khối lượng tính attention để sinh $N$ token (sau prompt):

| Phương pháp                | Mỗi bước xử lý      | Attention mỗi bước                         | Tổng để sinh N token   |
| :------------------------- | :------------------ | :----------------------------------------- | :--------------------- |
| **Ngây thơ (không cache)** | cả chuỗi độ dài $t$ | $\mathcal{O}(t^2)$                         | $\sim\mathcal{O}(N^3)$ |
| **KV-cache**               | 1 token mới         | $\mathcal{O}(t)$ (1 hàng $\times$ $t$ cột) | $\sim\mathcal{O}(N^2)$ |

Từ $\mathcal{O}(N^3)$ xuống $\mathcal{O}(N^2)$ - với $N$ lớn (sinh vài trăm/nghìn token) đây là khác biệt trời vực. Cache _không đổi kết quả_ (vẫn ra đúng chuỗi đó nếu dùng cùng seed), chỉ _loại bỏ tính toán thừa_.

🎯 KV-cache không phải “mẹo gần đúng”: kết quả toán học _giống hệt_ bản ngây thơ, chỉ nhanh hơn vì không tính lại.

---

### Nhìn sâu hơn

_Cái giá của tốc độ là bộ nhớ (VRAM)._ Cache lưu K và V của _mọi token đã sinh, ở mọi layer, mọi head_. Ước lượng thô kích thước cache:

$$\text{kv\_cache\_bytes} \approx 2 \ (\text{K và V}) \times n_{\text{layers}} \times \text{seq\_len} \times d_{\text{model}} \times \text{bytes\_per\_elem} \times \text{batch}$$

Ví dụ mô hình có $n_{\text{layers}}=12$, $d_{\text{model}}=768$, sinh $\text{seq\_len}=1024$ token, fp16 (2 byte), $\text{batch}=1$:

$$\approx 2 \times 12 \times 1024 \times 768 \times 2 \text{ bytes} \approx 36\text{ MB (chỉ cho 1 chuỗi)}$$

Với mô hình lớn và batch lớn, cache có thể ngốn hàng GB VRAM - đó là lý do các kỹ thuật như _Multi-Query Attention (MQA)_ và _Grouped-Query Attention (GQA)_ ra đời: chia sẻ K/V giữa các head để cache nhỏ lại. Đó cũng là lý do có _giới hạn context length_ trên sản phẩm thật.

**Gotcha khi cài đặt:**

- Phải dùng _positional embedding đúng vị trí tuyệt đối_ cho token mới (vị trí $T_{\text{past}}$), không phải vị trí 0, nếu không mô hình “quên” mình đang ở đâu.
- Trong pha decode với $T=1$, _không cần_ causal mask tường minh (token mới nhìn thấy toàn bộ quá khứ đã cache). Nhưng ở pha prefill (xử lý cả prompt) thì _vẫn cần_ causal mask.
- Cache là _trạng thái_: mỗi chuỗi sinh mới phải khởi tạo cache riêng, đừng dùng lại cache của lượt trước.

---

### Thực hành

- **[Dễ]** Cho $d_{\text{model}}=768$, $n_{\text{head}}=12$. Hỏi: $d_k$ bằng bao nhiêu? Với một token mới ở bước decode, shape của $q$ sau khi tách head là gì? (Gợi ý: `(B, n_head, T, d_k)` với $T=1$.)
- **[Trung bình]** Chạy cả `generate_naive` và `generate_with_cache` trên cùng GPT mini với cùng `torch.manual_seed(...)`, so sánh: (a) chuỗi sinh ra có _giống hệt_ không? (b) đo thời gian bằng `time.time()` khi sinh 300 token - cache nhanh hơn mấy lần?
- **[Thử thách]** Ước lượng VRAM cho KV-cache của mô hình $n_{\text{layers}}=24$, $d_{\text{model}}=1024$, $\text{seq\_len}=2048$, fp16, $\text{batch}=4$. Rồi thử tưởng tượng nếu dùng MQA (tất cả head chia sẻ 1 K và 1 V) thì cache giảm đi bao nhiêu lần so với 16 head?

---

### Hiểu lầm thường gặp

- ❌ “KV-cache làm mô hình sinh ra kết quả khác (kém chính xác hơn) để đổi lấy tốc độ.” ➔ ✅ Kết quả _giống hệt_ bản không cache; cache chỉ bỏ tính toán trùng lặp, không phải xấp xỉ.
- ❌ “Nên cache luôn cả Query cho tiết kiệm hơn.” ➔ ✅ Query cũ _không dùng lại_. Mỗi bước cần Query MỚI cho token mới; chỉ K và V mới cache được.
- ❌ “Sinh chậm là vì attention vốn tuần tự như RNN.” ➔ ✅ Attention tính song song được. Chậm là do bản ngây thơ _tính lại cả chuỗi mỗi bước_; cache khắc phục điều đó.
- ❌ “Có cache rồi thì độ dài context muốn bao nhiêu cũng được.” ➔ ✅ Cache lớn dần theo độ dài, ăn VRAM tuyến tính - đây chính là ràng buộc giới hạn context length thực tế.

---

### Mẹo ghi nhớ

- _Q = câu hỏi mới mỗi bước (bỏ đi); K, V = trí nhớ (giữ lại)._ Cái gì thuộc quá khứ và không đổi thì cache.
- Nghĩ tới bức thư: đừng chép lại 499 chữ cũ để viết chữ thứ 500.
- Đổi _tốc độ_ lấy _bộ nhớ_: cache nhanh hơn nhưng ăn VRAM theo $2 \times \text{layers} \times \text{seq} \times d_{\text{model}}$.
- 🧠 Mẹo: “*K*eep *V*alues, ask new *Q*uestions” - Keep K & V, ask new Q.

---

### Tóm tắt trực quan

| Khái niệm                    | Một dòng                                                         |
| :--------------------------- | :--------------------------------------------------------------- |
| **Sinh tự hồi quy ngây thơ** | Tính lại cả chuỗi mỗi bước -> $\sim\mathcal{O}(N^3)$             |
| **KV-cache**                 | Lưu K, V cũ, mỗi bước chỉ tính 1 token -> $\sim\mathcal{O}(N^2)$ |
| **Query**                    | KHÔNG cache (mới mỗi bước)                                       |
| **Key & Value**              | CÓ cache (quá khứ cố định)                                       |
| **Prefill**                  | Nạp prompt một lần, dựng cache ban đầu                           |
| **Decode**                   | Đưa vào đúng 1 token mới, tái dùng cache                         |
| **Đánh đổi**                 | Nhanh hơn nhưng tốn VRAM theo độ dài                             |
| **MQA / GQA**                | Chia sẻ K/V giữa head để cache nhỏ lại                           |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Let’s reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- PyTorch docs: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT (Andrej Karpathy) - có hàm generate tham khảo: [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Attention Is All You Need: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Hugging Face docs (KV-cache, generation): [https://huggingface.co/docs](https://huggingface.co/docs)

---

### Tổng kết

- Sinh tự hồi quy _ngây thơ_ tính lại attention cho toàn bộ chuỗi mỗi bước -> chậm dần, tổng $\sim\mathcal{O}(N^3)$.
- _KV-cache_ lưu Key và Value của các token đã sinh (cố định, không đổi), nên mỗi bước chỉ cần tính Q/K/V cho _một token mới_ -> tổng $\sim\mathcal{O}(N^2)$, nhanh hơn nhiều mà kết quả _giống hệt_.
- Query không cache được; K và V thì có. Sinh chia 2 pha: _prefill_ (nạp prompt) và _decode_ (từng token).
- Cái giá là _VRAM_: cache lớn dần theo độ dài $\times$ số layer $\times$ $d_{\text{model}}$; đây là gốc rễ giới hạn context length.

Bạn đã biết cách làm GPT mini sinh văn bản nhanh và tiết kiệm. Ở _Bài 35: Tự host mô hình của bạn - chạy inference cục bộ trên GPU_, ta sẽ nối lại với bài phần cứng: đưa mô hình đã huấn luyện lên GPU cục bộ, chạy inference thật (có dùng KV-cache), và đóng gói một API sinh văn bản đơn giản. Hẹn gặp ở bài kế tiếp! 🤖
