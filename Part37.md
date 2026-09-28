# Bài 37: Scaling laws - mô hình lớn hơn, dữ liệu nhiều hơn

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 8:** Từ pretraining đến trợ lý & dự án cuối

---

### Giới thiệu

Ở _Bài 35_, bạn đã đưa GPT mini tự huấn luyện lên GPU cục bộ, chạy inference và bọc nó thành một API sinh văn bản. Cảm giác “mô hình của mình đang chạy thật” rất đã. Nhưng chắc chắn bạn cũng nhận ra: GPT mini viết được vài câu tiếng Anh nghe hay hay, nhưng còn xa mới thông minh như ChatGPT. Vậy khoảng cách đó nằm ở đâu? Câu trả lời ngắn gọn: _quy mô_ (scale) - nhiều tham số hơn, nhiều dữ liệu hơn, nhiều phép tính hơn.

Hãy tưởng tượng bạn học ngoại ngữ. Một người đọc 10 cuốn sách sẽ giỏi hơn người đọc 1 cuốn; một người có “bộ não” ghi nhớ nhiều hơn (nhiều nơ-ron hơn) sẽ nắm được nhiều quy tắc tinh tế hơn. LLM cũng vậy: cho nó _lớn hơn_ (nhiều tham số) và _đọc nhiều hơn_ (nhiều token dữ liệu), năng lực thường tăng lên một cách _đều đặn và dự đoán được_. Điều bất ngờ là sự tăng đó không hỗn loạn mà tuân theo một quy luật toán học gọi là _scaling laws_ (quy luật mở rộng quy mô).

> 🔑 **Một câu để nhớ:** Với LLM, chất lượng (đo bằng loss) giảm _đều đặn theo quy luật lũy thừa_ khi ta tăng số tham số, số token dữ liệu và lượng tính toán - và ta có thể _dự đoán_ mức cải thiện trước khi train.

---

### Bạn sẽ học được gì?

- [ ] Vì sao mô hình/dữ liệu lớn hơn thường tốt hơn (trực giác + bằng chứng thực nghiệm)
- [ ] Scaling laws là gì: quan hệ lũy thừa giữa loss với N (tham số), D (dữ liệu), C (tính toán)
- [ ] Cách ước lượng chi phí tính toán bằng công thức thô $C \approx 6 \times N \times D$
- [ ] “Bài học Chinchilla”: cân bằng tham số và dữ liệu, đừng chỉ phình mô hình
- [ ] Giới hạn thực tế khi tự train ở nhà: VRAM, thời gian, tiền điện, dữ liệu

📌 **Yêu cầu trước:** đã hiểu training loop, cross-entropy loss (Bài 27-28), số tham số của Transformer (Bài 20), và tự host inference (Bài 35).

---

### 1. Trực giác: vì sao "lớn hơn" lại tốt hơn?

**Analogy:** hãy coi việc train LLM như _nén tri thức của Internet vào một bộ trọng số_. Mô hình càng có nhiều tham số thì “kho chứa” càng lớn, nén được nhiều mẫu hình ngôn ngữ tinh tế hơn. Dữ liệu càng nhiều thì mô hình càng gặp nhiều tình huống, học được quy luật tổng quát thay vì học vẹt vài câu.

Ba “nút vặn” chính quyết định chất lượng:

| Ký hiệu | Ý nghĩa                                             | Ví dụ đơn vị                |
| :------ | :-------------------------------------------------- | :-------------------------- |
| **N**   | Số _tham số_ (parameters) của mô hình               | 124 triệu, 7 tỷ…            |
| **D**   | Số _token dữ liệu_ dùng để train                    | 10 tỷ, 300 tỷ token…        |
| **C**   | Lượng _tính toán_ (compute), thường tính bằng FLOPs | $10^{18}$, $10^{23}$ FLOPs… |

💡 **Phân biệt:** _tham số_ (parameter) là trọng số học được, cố định sau khi train. _Token_ là đơn vị dữ liệu đầu vào (một mẩu văn bản). Đừng nhầm “mô hình 7 tỷ” (7 tỷ _tham số_) với “train trên 7 tỷ _token_” - hai con số khác hẳn nhau.

---

### 2. Scaling laws là gì?

Năm 2020, nhóm nghiên cứu của OpenAI (Kaplan và cộng sự) phát hiện: khi bạn vẽ _loss_ theo N, D, hoặc C trên _trục log-log_, các điểm rơi gần như thành _đường thẳng_. Đường thẳng trên log-log nghĩa là quan hệ _lũy thừa_ (power law):

```text
Loss(N) ≈ (Nc / N)^αN        <- loss giảm khi N tăng
Loss(D) ≈ (Dc / D)^αD        <- loss giảm khi D tăng
```

Trong đó $\alpha_N$, $\alpha_D$ là các số mũ nhỏ (thường ~0.05-0.1), $N_c$, $D_c$ là hằng số. Điều kỳ diệu: _loss có thể dự đoán được_. Nếu bạn train 3 mô hình nhỏ và đo loss, bạn vẽ được đường thẳng và _ngoại suy_ loss của mô hình lớn hơn 100 lần - trước khi tốn tiền train nó.

```text
loss
 (log)
  ^
  |  x                     <- mô hình nhỏ: loss cao
  |    x
  |      x                  <- các điểm gần như thẳng hàng
  |        x
  |          x              <- mô hình lớn: loss thấp
  +------------------------> compute C (log)
```

🎯 Đây là lý do các phòng lab dám đổ hàng chục triệu đô vào một lần train: họ đã “đo trước” và tin loss sẽ về mức mong muốn.

---

### 3. Công thức chi phí tính toán: `C ≈ 6 * N * D`

Đây là công thức thô cực hữu ích để ước lượng compute cần thiết:

```text
C (FLOPs)  ≈  6 * N (tham số) * D (token)
```

Hệ số 6 đến từ: mỗi tham số làm ~2 phép (nhân + cộng) ở _forward pass_, và ~4 phép ở _backward pass_ (tính gradient tốn gấp đôi forward), tổng ~6 FLOPs cho mỗi cặp (tham số, token).

Hãy tính thử với code:

```python
def estimate_train_flops(n_params: int, n_tokens: int) -> float:
    # Rough compute estimate for training a Transformer LM.
    # C ~= 6 * N * D  (2 for forward, 4 for backward).
    return 6 * n_params * n_tokens

# GPT mini của bạn: ~10M params, train trên ~1M token
small = estimate_train_flops(10_000_000, 1_000_000)
print(f"GPT mini: {small:.2e} FLOPs") # GPT mini: 6.00e+13 FLOPs

# GPT-2 medium-ish: ~350M params, ~10B token
big = estimate_train_flops(350_000_000, 10_000_000_000)
print(f"GPT-2 medium: {big:.2e} FLOPs") # GPT-2 medium: 2.10e+19 FLOPs

print(f"Tỷ lệ: {big / small:.1e} lần") # Tỷ lệ: 3.5e+05 lần
```

GPT-2 medium tốn nhiều hơn GPT mini của bạn khoảng _350.000 lần_ về tính toán! Đó là lý do bạn không thể train nó trên laptop trong vài phút.

💡 **Muốn quy FLOPs ra thời gian:** chia cho throughput của GPU. Ví dụ một GPU đạt ~$10^{13}$ FLOPs/giây thực tế (sau khi trừ hao hụt) thì $2.1 \times 10^{19} / 10^{13} \approx 2.1 \times 10^6$ giây $\approx$ _24 ngày_ chạy liên tục. Con số này chỉ là ước lượng bậc độ lớn.

---

### 4. Bài học Chinchilla: cân bằng N và D

Năm 2022, DeepMind (paper Chinchilla, Hoffmann và cộng sự) chỉ ra một điều quan trọng: các mô hình khổng lồ thời đó (như GPT-3 với 175 tỷ tham số) bị _train thiếu dữ liệu_ (undertrained). Với một ngân sách compute C cố định, bạn nên _cân bằng_ giữa N và D chứ đừng chỉ phình N.

Kết luận thực nghiệm nổi tiếng: _train khoảng 20 token cho mỗi tham số_ thì hiệu quả compute là tốt nhất.

| Chiến lược                | Tham số N | Token D      | Ghi chú                                       |
| :------------------------ | :-------- | :----------- | :-------------------------------------------- |
| **Chỉ phình mô hình**     | Rất lớn   | Ít (D/N nhỏ) | Lãng phí: mô hình “đói” dữ liệu, học chưa hết |
| **Cân bằng (Chinchilla)** | Vừa phải  | ~20\*N token | Dùng compute hiệu quả nhất                    |
| **Chỉ thêm dữ liệu**      | Nhỏ       | Rất lớn      | Mô hình “nhỏ đầu”, không đủ năng lực chứa     |

```python
def chinchilla_tokens(n_params: int, tokens_per_param: int = 20) -> int:
    # Chinchilla rule of thumb: ~20 tokens per parameter for
    # compute-optimal training under a fixed budget.
    return n_params * tokens_per_param

n = 1_000_000_000 # 1B params
print(f"Nên train ~{chinchilla_tokens(n):.2e} token") # Nên train ~2.00e+10 token
```

⚠️ Con số “20 token/tham số” là quy tắc ngón tay cái từ một bối cảnh cụ thể, không phải hằng số vũ trụ. Ngày nay nhiều mô hình cố ý train _vượt xa_ mức này (nhiều token hơn) để mô hình _nhỏ mà chạy rẻ khi inference_. Đánh đổi: tốn nhiều compute train hơn để tiết kiệm chi phí phục vụ về sau.

---

### 5. Emergence: khả năng "bất ngờ xuất hiện"

Một hiện tượng thú vị: một số khả năng (như làm toán nhiều bước, suy luận theo chuỗi) gần như _không xuất hiện_ ở mô hình nhỏ, rồi _đột ngột tốt lên_ khi mô hình vượt một ngưỡng quy mô. Trực giác: giống như đứa trẻ, có những kỹ năng chỉ “bừng sáng” khi não đủ lớn và đã đọc đủ nhiều. (Lưu ý: cộng đồng vẫn tranh luận emergence “thật” hay chỉ là hiệu ứng của cách đo - nên coi đây là quan sát thú vị, không phải định luật cứng.)

---

### Nhìn sâu hơn

- **Vì sao loss cứ giảm mãi mà không chạm đáy 0?** Vì ngôn ngữ có _entropy nội tại_ - luôn tồn tại độ bất định không thể loại bỏ (không ai đoán đúng 100% từ tiếp theo). Scaling laws thường có thêm một số hạng “không thể giảm” (irreducible loss) mà mô hình tiệm cận nhưng không vượt qua.
- **Giới hạn dữ liệu.** Lượng văn bản chất lượng cao trên Internet là _hữu hạn_. Khi các mô hình lớn đã “ăn” gần hết dữ liệu công khai, việc scale D bị nghẽn - dẫn tới nghiên cứu về dữ liệu tổng hợp, lọc chất lượng, và huấn luyện nhiều epoch.
- **Bộ nhớ mới là rào cản thật khi tự train.** Không chỉ FLOPs, mà _VRAM_ giới hạn N. Nhớ lại Bài 2: mỗi tham số fp32 cần 4 byte; khi train với AdamW còn cần thêm gradient + 2 trạng thái optimizer (momentum $m$ và variance $v$), cộng activation. Ước lượng thô cho fp32:

```text
VRAM train (thô) ~= N * 4 byte (weights)
                  + N * 4 byte (gradients)
                  + N * 8 byte (AdamW: m và v)
                 ~= N * 16 byte  (chưa kể activation!)
```

```python
def vram_train_gb(n_params: int, bytes_per_param: int = 16) -> float:
    # Very rough: weights + grads + AdamW states in fp32 ~= 16 bytes/param.
    # Activations add more on top (depends on batch size, seq len).
    return n_params * bytes_per_param / (1024 ** 3)

print(f"124M model: ~{vram_train_gb(124_000_000):.1f} GB") # 124M model: ~1.8 GB
print(f"7B model:   ~{vram_train_gb(7_000_000_000):.1f} GB") # 7B model:   ~104.3 GB
```

Một card game phổ thông thường có 8-24 GB VRAM. Train mô hình 7 tỷ tham số bằng fp32 cần ~100 GB _chỉ cho trạng thái_ - vượt xa một card đơn. Đó là lý do phải dùng mixed precision (fp16/bf16), nhiều GPU, gradient checkpointing, hoặc fine-tuning tiết kiệm như LoRA thay vì train từ đầu.

---

### Thực hành

- **[Dễ]** Dùng `estimate_train_flops`, tính compute cho một mô hình 124M tham số train trên 300 tỷ token (cỡ GPT-2 gốc). So sánh với GPT mini của bạn. Gợi ý: kết quả cỡ $2.2 \times 10^{20}$ FLOPs.
- **[Trung bình]** Viết hàm `flops_to_days(flops, gpu_flops_per_sec)` chia FLOPs cho throughput rồi đổi giây ra ngày. Thử với `gpu_flops_per_sec=5e13` và compute của mô hình 124M/300B token ở trên. Ước lượng bao nhiêu ngày trên một GPU?
- **[Thử thách]** Theo quy tắc Chinchilla (~20 token/tham số), nếu bạn chỉ có ngân sách để train trên _2 tỷ token_, số tham số “tối ưu compute” nên là bao nhiêu? Ngược lại, nếu bạn có một mô hình 500M tham số, cần bao nhiêu token cho “đủ no”? Viết vài dòng code trả lời.

---

### Hiểu lầm thường gặp

- ❌ “Cứ tăng tham số là mô hình thông minh hơn.” ➔ ✅ Chỉ đúng nếu _đồng thời tăng dữ liệu và compute_ cân đối. Phình N mà D ít -> mô hình undertrained, lãng phí (bài học Chinchilla).
- ❌ “Scaling laws hứa loss sẽ về 0 nếu đủ lớn.” ➔ ✅ Có _irreducible loss_ do entropy của ngôn ngữ; loss tiệm cận một sàn chứ không về 0.
- ❌ “Có FLOPs là train được, VRAM không quan trọng.” ➔ ✅ _VRAM là rào cản cứng_: không đủ bộ nhớ thì đơn giản là không nạp nổi mô hình + optimizer state, dù bạn có bao nhiêu thời gian.
- ❌ “Mô hình lớn luôn đáng tiền hơn.” ➔ ✅ Mô hình lớn tốn kém khi _inference_ (mỗi lần chạy). Nhiều khi mô hình nhỏ train kỹ (nhiều token) lại rẻ và đủ dùng hơn.

---

### Mẹo ghi nhớ

- Ba nút vặn: _N_ (tham số) - _D_ (dữ liệu/token) - _C_ (compute). Nhớ _$C \approx 6 \times N \times D$_.
- Quy tắc Chinchilla: _~20 token cho mỗi tham số_.
- 🧠 Mẹo: “Bộ não to (N) mà đọc ít sách (D) thì vẫn dốt.” Cân bằng mới hiệu quả.

---

### Tóm tắt trực quan

| Khái niệm            | Một dòng                                                           |
| :------------------- | :----------------------------------------------------------------- |
| **Scaling laws**     | Loss giảm theo _lũy thừa_ khi tăng N, D, C - dự đoán được          |
| **N / D / C**        | Tham số / token dữ liệu / lượng tính toán                          |
| **C ≈ 6\*N\*D**      | Ước lượng thô FLOPs để train (2 forward + 4 backward)              |
| **Chinchilla**       | Cân bằng N và D; ~20 token/tham số cho compute tối ưu              |
| **Irreducible loss** | Sàn loss do entropy ngôn ngữ; không về 0                           |
| **VRAM train**       | ~16 byte/tham số (fp32) cho weights+grad+AdamW, chưa kể activation |
| **Emergence**        | Vài khả năng “bừng sáng” khi vượt ngưỡng quy mô                    |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [[1hr Talk] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)
- [Attention Is All You Need (kiến trúc Transformer nền tảng)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer (trực quan hóa)](https://jalammar.github.io/illustrated-transformer/)
- [nanoGPT của Andrej Karpathy (train GPT thật, quy mô nhỏ)](https://github.com/karpathy/nanoGPT)
- [PyTorch documentation](https://pytorch.org/docs/stable/)
- [Hugging Face docs (mô hình lớn, kỹ thuật train hiệu quả)](https://huggingface.co/docs)

---

### Tổng kết

- _Lớn hơn thường tốt hơn_, nhưng theo một quy luật _đo được_: scaling laws cho thấy loss giảm lũy thừa theo N, D, C.
- Dùng _$C \approx 6 \times N \times D$_ để ước lượng chi phí train; nhớ _cân bằng_ N và D (Chinchilla ~20 token/tham số).
- Có _giới hạn thật_: irreducible loss, dữ liệu hữu hạn, và trên hết là _VRAM/thời gian/tiền_ khi tự train ở nhà - đây là lý do ta thường fine-tune mô hình có sẵn thay vì train từ số 0.

Bạn đã hiểu “vì sao mô hình lớn mạnh” và “tự train tốn kém thế nào”. Nhưng train từ đầu (pretraining) chỉ là _một nửa_ câu chuyện - làm sao biến một mô hình “biết đoán chữ” thành một _trợ lý biết nghe lời_? Ở _Bài 38: Pretraining và fine-tuning - bức tranh tổng thể_, chúng ta sẽ ghép mọi mảnh lại: từ pretraining trên dữ liệu thô, tới fine-tuning và các bước biến GPT mini thành một trợ lý hữu ích. Hẹn gặp ở bài kế! 🤖
