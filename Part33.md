# Bài 33: Sinh văn bản nâng cao - temperature, top-k, top-p

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 7:** Tokenization thực tế, tối ưu & tự host

---

### Giới thiệu

Ở _Bài 32_, mình đã học cách dùng tokenizer thực tế (`tiktoken`, Hugging Face `tokenizers`) để biến văn bản thành chuỗi token id và ngược lại. Giờ model đã “biết đọc, biết viết” bằng subword như GPT thật. Nhưng có một câu hỏi lớn còn bỏ ngỏ: _khi model đã đưa ra logits cho token tiếp theo, ta CHỌN token nào?_

Hãy tưởng tượng bạn hỏi một người bạn: “Kể cho tớ một câu chuyện.” Nếu bạn ấy lúc nào cũng chọn từ _an toàn nhất, dễ đoán nhất_, câu chuyện sẽ nhàm chán, lặp đi lặp lại (“và rồi… và rồi… và rồi…”). Nếu bạn ấy chọn từ _quá ngẫu hứng, bạ gì nói nấy_, câu chuyện sẽ lảm nhảm, mất mạch lạc. Nghệ thuật sinh văn bản chính là _điều khiển núm vặn “độ sáng tạo”_ giữa hai thái cực đó.

Trong bài này, mình sẽ mổ xẻ ba núm vặn nổi tiếng nhất: _temperature_ (làm phân phối “nóng” hay “lạnh”), _top-k_ (chỉ giữ k ứng viên hàng đầu) và _top-p / nucleus sampling_ (giữ đủ ứng viên cho tới khi gom đủ xác suất p). Tất cả đều tác động lên đúng một chỗ: _phân phối xác suất trên từ vựng ngay trước khi lấy mẫu (sampling)_.

> 🔑 **Một câu để nhớ:** temperature, top-k và top-p đều là các phép _biến đổi phân phối xác suất trên logits trước khi lấy mẫu_ - chúng không đổi kiến thức của model, chỉ đổi cách model _chọn_ từ tiếp theo.

---

### Bạn sẽ học được gì?

- [ ] Hiểu vì sao logits → softmax → xác suất, và sinh văn bản = lấy mẫu lặp lại từ phân phối này.
- [ ] Nắm _temperature_ làm phân phối sắc nét (T thấp) hay tù (T cao) như thế nào, kèm công thức.
- [ ] Cài đặt _top-k_: chỉ giữ k logits lớn nhất, phần còn lại gán `-inf`.
- [ ] Cài đặt _top-p (nucleus sampling)_: giữ tập nhỏ nhất gom đủ xác suất tích lũy $\ge p$.
- [ ] Kết hợp cả ba trong một hàm `generate()` PyTorch chạy được và hiểu đánh đổi _đa dạng ↔ mạch lạc_.

📌 **Yêu cầu trước:** bạn đã có GPT mini huấn luyện xong (Bài 24-31), biết `softmax`, `logits`, và tokenizer từ Bài 32.

---

### 1. Từ logits tới lựa chọn: bức tranh lớn

Ở mỗi bước sinh, model nhận chuỗi token hiện tại và trả về _logits_ cho token _cuối cùng_ - một vector số thực dài bằng kích thước từ vựng `vocab_size`. Logits _chưa phải_ xác suất (có thể âm, không cộng lại bằng 1). Ta biến nó thành xác suất bằng softmax, rồi _lấy mẫu_ một token, nối vào chuỗi, và lặp lại.

```text
token ids ──> [GPT mini] ──> logits[vocab_size]  (số thực, chưa chuẩn hóa)
                                   │
                    biến đổi (temperature / top-k / top-p)
                                   │
                              softmax
                                   ▼
                           probs[vocab_size]  (cộng = 1)
                                   │
                         torch.multinomial  (lấy mẫu 1 token)
                                   ▼
                            next_token id  ──> nối vào chuỗi, lặp lại
```

Điểm mấu chốt: _mọi kỹ thuật trong bài chỉ can thiệp vào bước “biến đổi” trước softmax/sampling_. Model không hề được huấn luyện lại - ta chỉ đổi _chiến lược chọn_.

💡 Nếu luôn chọn token có xác suất cao nhất (`argmax`), ta gọi là _greedy decoding_: mạch lạc nhưng cứng nhắc, dễ lặp. Lấy mẫu (sampling) đưa yếu tố ngẫu nhiên vào để văn bản đa dạng hơn.

---

### 2. Temperature: núm vặn "nóng - lạnh"

**Analogy:** hãy nghĩ tới độ tự tin của một người. Người “lạnh lùng, chắc chắn” luôn nói ra đáp án chắc nhất (phân phối dồn hết vào một, vài lựa chọn). Người “phiêu, ngẫu hứng” thì rải đều cơ hội cho nhiều lựa chọn. Temperature $T$ chính là núm vặn độ phiêu đó.

Công thức: chia logits cho $T$ _trước khi_ softmax:

$$\text{probs} = \text{softmax}\left(\frac{\text{logits}}{T}\right)$$

- $T < 1$ (vd 0.7): logits bị _phóng đại_ khoảng cách → phân phối _sắc nét_ hơn → thiên về token có xác suất cao → mạch lạc, ít bất ngờ.
- $T = 1$: giữ nguyên phân phối gốc của model.
- $T > 1$ (vd 1.5): logits bị _nén_ lại gần nhau → phân phối _tù, phẳng_ hơn → đa dạng, sáng tạo, nhưng dễ lảm nhảm.
- $T \to 0$: xấp xỉ `argmax` (greedy).

Ví dụ tính tay với 3 token, logits `[2.0, 1.0, 0.1]`:

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1]) # shape: [3]

# T = 1.0 (original distribution)
p1 = F.softmax(logits / 1.0, dim=-1) # -> [0.659, 0.242, 0.099]

# T = 0.5 (sharper - mass shifts toward the top token)
p2 = F.softmax(logits / 0.5, dim=-1) # -> [0.844, 0.114, 0.019]

# T = 2.0 (flatter - more uniform)
p3 = F.softmax(logits / 2.0, dim=-1) # -> [0.506, 0.307, 0.187]

print(p1) # tensor([0.6590, 0.2424, 0.0986])
print(p2) # tensor([0.8438, 0.1142, 0.0190])
print(p3) # tensor([0.5062, 0.3072, 0.1866])
```

⚠️ **Sai lầm phổ biến:** nghĩ “temperature cao luôn tốt vì sáng tạo hơn”. Thực tế $T$ quá cao (vd > 1.5) thường làm văn bản mất mạch lạc, rơi vào token hiếm vô nghĩa. Đa số hệ thống thực tế dùng $T$ trong khoảng 0.7-1.0.

---

### 3. Top-k: chỉ giữ k ứng viên hàng đầu

**Analogy:** khi chọn món ăn, thay vì cân nhắc cả 500 món trong menu (kể cả những món bạn chẳng bao giờ gọi), bạn chỉ nhìn _5 món ngon nhất_ rồi chọn ngẫu nhiên trong đó. Đó chính là top-k.

Cơ chế: giữ lại $k$ logits _lớn nhất_, gán phần còn lại bằng `-inf` (để sau softmax chúng thành xác suất 0), rồi mới lấy mẫu.

```python
def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    # logits shape: [vocab_size]
    v, _ = torch.topk(logits, k) # v: k largest values, shape [k]
    threshold = v[-1]            # threshold = smallest value inside top-k
    logits = logits.clone()
    logits[logits < threshold] = float('-inf') # drop everything below threshold
    return logits # shape: [vocab_size]

logits = torch.tensor([2.0, 1.0, 0.1, -0.5, 3.0]) # shape: [5]
filtered = top_k_filter(logits, k=2) # keep 2 largest tokens: 3.0 and 2.0
# filtered -> [2.0, -inf, -inf, -inf, 3.0]

probs = F.softmax(filtered, dim=-1) # -> [0.269, 0, 0, 0, 0.731]
```

- $k$ nhỏ (vd 5-10): rất an toàn, mạch lạc, nhưng có thể bỏ lỡ token hay ho.
- $k$ lớn (vd 50-100): đa dạng hơn, gần với phân phối gốc.

🎯 **Nhược điểm cố hữu của top-k:** $k$ là con số _cố định_ bất kể phân phối. Có lúc model rất chắc chắn (một token áp đảo), giữ tới $k=50$ token là thừa và nguy hiểm; có lúc model phân vân (nhiều token gần bằng nhau), $k=50$ lại có thể còn quá ít. Đây chính là lý do top-p ra đời.

---

### 4. Top-p (nucleus sampling): giữ theo xác suất tích lũy

**Analogy:** thay vì cố định “luôn lấy 5 món”, bạn nói: “cứ gom các món ngon nhất _cho tới khi chúng chiếm 90% khẩu vị của tôi_ thì dừng”. Số món giữ lại _tự co giãn_ theo tình huống. Đó là top-p (còn gọi _nucleus sampling_, giới thiệu trong bài báo “The Curious Case of Neural Text Degeneration”).

Cơ chế: sắp xếp xác suất giảm dần, cộng dồn (cumulative), giữ tập _nhỏ nhất_ có tổng $\ge p$, loại phần còn lại.

```python
def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    # logits shape: [vocab_size]
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1) # cumulative probability, shape [vocab_size]

    # mark tokens that go BEYOND the threshold p (to be removed)
    remove = cumprobs > p
    # shift right by one so we ALWAYS keep at least the first token
    remove[1:] = remove[:-1].clone()
    remove[0] = False

    sorted_logits[remove] = float('-inf')

    # map back to the original token order via index
    logits = torch.full_like(logits, float('-inf'))
    logits[sorted_idx] = sorted_logits
    return logits # shape: [vocab_size]

logits = torch.tensor([2.0, 1.0, 0.1, -0.5, 3.0]) # shape: [5]
filtered = top_p_filter(logits, p=0.9)
probs = F.softmax(filtered, dim=-1)
# Example: keep 3.0, 2.0, 1.0 (they gather ~0.9 probability), drop 0.1 and -0.5
```

**So sánh nhanh ba kỹ thuật:**

| Kỹ thuật        | Điều khiển gì                | Số ứng viên              | Ưu điểm                 | Nhược điểm                     |
| :-------------- | :--------------------------- | :----------------------- | :---------------------- | :----------------------------- |
| **temperature** | độ “nóng” của phân phối      | không cắt (giữ hết)      | mượt, dễ hiểu           | không loại được token rác đuôi |
| **top-k**       | giữ k token lớn nhất         | _cố định_ = k            | đơn giản, chặn đuôi rác | k cứng nhắc, không thích nghi  |
| **top-p**       | giữ tới khi tích lũy $\ge p$ | _thay đổi_ theo ngữ cảnh | thích nghi độ chắc chắn | cài đặt phức tạp hơn chút      |

💡 Thực tế thường _kết hợp_: áp temperature trước, rồi top-k (hoặc top-p), rồi mới softmax + lấy mẫu. GPT thật hay dùng bộ như $T \approx 0.8$, $\text{top\_p} \approx 0.9$.

---

### 5. Ráp lại: hàm `generate()` hoàn chỉnh

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    # idx shape: [B, T] - current sequence of token ids
    model.eval()
    for _ in range(max_new_tokens):
        # crop context if it exceeds block_size (max context length)
        idx_cond = idx[:, -model.block_size:] # [B, <=block_size]
        logits, _ = model(idx_cond)           # logits: [B, T, vocab_size]
        logits = logits[:, -1, :]             # keep only last step: [B, vocab_size]

        # 1) temperature (divide before softmax)
        logits = logits / max(temperature, 1e-8)

        # 2) top-k (if provided)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # 3) top-p (if provided)
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            # scatter the mask back to original token positions
            mask = remove.scatter(1, sorted_idx, remove)
            logits[mask] = float('-inf')

        probs = F.softmax(logits, dim=-1)                   # [B, vocab_size], sums to 1
        next_id = torch.multinomial(probs, num_samples=1)   # [B, 1] - sample one token
        idx = torch.cat((idx, next_id), dim=1)              # [B, T+1]

    return idx

# Example usage:
# out = generate(model, start_ids, max_new_tokens=100, temperature=0.8, top_k=40)
# text = tokenizer.decode(out[0].tolist())
```

🎯 Chú ý `@torch.no_grad()`: khi sinh văn bản ta _không cần gradient_ → tiết kiệm bộ nhớ và nhanh hơn. Và `model.eval()` để tắt dropout.

---

### Nhìn sâu hơn

- **Vì sao chia cho $T$ chứ không nhân?** Vì logits nằm trong hàm mũ của softmax: $\exp(z_i / T)$. Chia cho $T$ nhỏ ($T < 1$) làm khoảng cách giữa các $\exp$ lớn hơn theo cấp số nhân → sắc nét. Đây là cùng dạng với “nhiệt độ” trong phân phối Boltzmann của vật lý, nơi tên gọi bắt nguồn.
- **Top-k và top-p không loại trừ nhau.** Bạn có thể áp cả hai: top-k chặn cứng số ứng viên tối đa (an toàn), top-p tiếp tục co lại khi model chắc chắn. Nhiều thư viện ([Hugging Face](https://huggingface.co/docs) `generate`) cho bật đồng thời.
- **Greedy dễ lặp vòng.** Nếu chỉ `argmax`, model dễ rơi vào vòng lặp “the the the” vì mỗi bước luôn chọn token an toàn nhất. Một chút ngẫu nhiên (temperature + sampling) phá vỡ vòng lặp đó - đây là quan sát trung tâm của nucleus sampling.
- **Chi phí tính toán** của cả ba đều rất rẻ so với một lần forward của model: chỉ là các thao tác trên vector `vocab_size` (topk/sort/cumsum). Nút thắt tốc độ thật sự nằm ở việc chạy lại toàn bộ chuỗi mỗi bước - và đó chính là vấn đề _Bài 34_ sẽ giải bằng KV-cache.

---

### Thực hành

- **[Dễ]** Lấy `logits = torch.tensor([3.0, 2.0, 1.0, 0.0])`. Tính bằng tay (hoặc code) softmax với $T=0.5$, $T=1.0$, $T=2.0$. Quan sát xác suất token đầu tăng/giảm ra sao.
- **[Trung bình]** Sửa hàm `generate()` để in ra _số token còn sống sót_ sau bước top-p ở mỗi bước sinh. Chạy với một câu “chắc chắn” và một câu “phân vân”, xem con số co giãn thế nào.
- **[Thử thách]** Sinh 200 token từ GPT mini của bạn với 3 cấu hình: (a) greedy ($T \to 0$), (b) $T=1.0, \text{top\_k}=40$, (c) $T=0.9, \text{top\_p}=0.92$. So sánh định tính độ mạch lạc và đa dạng, rồi thử tìm bộ tham số bạn thấy “vừa tay” nhất cho dữ liệu của mình.

---

### Hiểu lầm thường gặp

- ❌ “Temperature cao luôn tốt hơn vì sáng tạo hơn.” ➔ ✅ $T$ quá cao làm phân phối phẳng, model chọn token hiếm vô nghĩa → mất mạch lạc. Thường 0.7-1.0 là dải hợp lý.
- ❌ “Top-k và top-p làm model _thông minh hơn_.” ➔ ✅ Chúng không đổi tri thức của model, chỉ thay đổi _cách chọn_ token từ cùng một phân phối logits.
- ❌ “Top-p với $p=0.9$ nghĩa là luôn giữ 90% số token.” ➔ ✅ $p$ là _xác suất tích lũy_, không phải tỉ lệ số token. Số token giữ lại co giãn: có khi chỉ 1-2 token đã gom đủ 0.9.
- ❌ “Logits chính là xác suất.” ➔ ✅ Logits là số thực chưa chuẩn hóa; phải qua softmax mới thành xác suất cộng bằng 1.

---

### Mẹo ghi nhớ

- _temperature = độ nóng_: $T$ thấp → lạnh, chắc chắn; $T$ cao → nóng, phiêu.
- _top-k = số cố định_ món trong menu; _top-p = gom tới đủ p%_ khẩu vị (số món co giãn).
- Thứ tự áp dụng trong code: `logits / T` → cắt `top-k/top-p` → `softmax` → `multinomial`.
- 🧠 Mẹo: “k là *k*ount (đếm cố định), p là *p*ercent tích lũy (co giãn)”.

---

### Tóm tắt trực quan

| Khái niệm             | Một dòng                                                       |
| :-------------------- | :------------------------------------------------------------- |
| **logits**            | vector số thực `[vocab_size]`, chưa chuẩn hóa                  |
| **softmax**           | biến logits thành xác suất cộng bằng 1                         |
| **temperature T**     | chia logits cho T trước softmax; T thấp = sắc nét, cao = phẳng |
| **top-k**             | giữ k logits lớn nhất, phần còn lại `-inf`                     |
| **top-p (nucleus)**   | giữ tập nhỏ nhất có xác suất tích lũy $\ge p$                  |
| **torch.multinomial** | lấy mẫu token theo phân phối xác suất                          |
| **đánh đổi**          | mạch lạc ↔ đa dạng (sáng tạo)                                  |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Temperature, Top-K & Top-P Explained in 10 Minutes | LLM Sampling Made Simple](https://www.youtube.com/watch?v=YMIQrH9BQK0)
- PyTorch docs: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT của Andrej Karpathy (hàm generate tham khảo): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- “Attention Is All You Need”: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Hugging Face docs (text generation strategies): [https://huggingface.co/docs](https://huggingface.co/docs)

---

### Tổng kết

- Sinh văn bản = lặp lại: forward → _biến đổi logits_ → softmax → lấy mẫu → nối token.
- _Temperature_ điều chỉnh độ sắc nét của phân phối (chia logits cho $T$ trước softmax).
- _Top-k_ giữ số ứng viên _cố định_; _top-p (nucleus)_ giữ số ứng viên _co giãn_ theo xác suất tích lũy.
- Cả ba là công cụ điều khiển đánh đổi _mạch lạc ↔ đa dạng_; thường kết hợp $T \approx 0.8$ với `top_k` hoặc $\text{top\_p} \approx 0.9$.
- Chúng chỉ đổi _cách chọn_, không đổi tri thức của model.

Ở _Bài 34: Tăng tốc sinh văn bản với KV-cache_, mình sẽ giải quyết nút thắt tốc độ mà bạn vừa thấy trong hàm `generate()`: mỗi bước ta đang chạy lại toàn bộ chuỗi từ đầu, tính lại Key/Value cho mọi token cũ một cách lãng phí. KV-cache sẽ _nhớ lại_ các Key/Value đã tính để mỗi bước sinh nhanh hơn nhiều. Hẹn gặp bạn ở bài kế! 🚀
