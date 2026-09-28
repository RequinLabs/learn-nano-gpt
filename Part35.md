# Bài 35: Tự host mô hình của bạn - chạy inference cục bộ trên GPU

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 7:** Tokenization thực tế, tối ưu & tự host

---

### Giới thiệu

Ở _Bài 34_, bạn đã tăng tốc sinh văn bản bằng _KV-cache_ - thay vì tính lại key/value cho toàn bộ câu ở mỗi bước, ta lưu lại và chỉ tính cho token mới. Mô hình chạy nhanh hơn hẳn. Nhưng cho tới giờ, mọi thứ vẫn nằm trong một notebook - bạn gõ `model.generate(...)` rồi nhìn chữ hiện ra. Câu hỏi tự nhiên tiếp theo là: _làm sao để người khác dùng được mô hình của bạn?_

Hãy tưởng tượng bạn nấu ăn rất ngon 🍳. Nấu cho mình ăn là một chuyện; _mở một quán và phục vụ khách_ lại là chuyện khác - bạn cần một cái bếp cố định (GPU), một thực đơn (API), và phải lo chuyện nồi có đủ lớn không (VRAM), khách chờ bao lâu (tốc độ). Bài này chính là lúc bạn “mở quán”: đưa GPT mini đã huấn luyện lên _GPU cục bộ_ (chiếc card bạn đã tìm hiểu ở Bài 2), chạy _inference_ ổn định, và bọc nó trong một _API sinh văn bản_ đơn giản để bất kỳ ai cũng gọi được.

> 🔑 **Một câu để nhớ:** Tự host = nạp trọng số lên GPU MỘT LẦN, giữ mô hình ở chế độ `eval()` + `torch.no_grad()`, rồi phục vụ nhiều yêu cầu sinh văn bản qua một API - tối ưu bộ nhớ (VRAM) và tốc độ (token/giây).

---

### Bạn sẽ học được gì?

- [ ] Phân biệt _training_ (cần gradient, optimizer) và _inference_ (chỉ forward, không cần gradient)
- [ ] Nạp checkpoint đã lưu và đưa mô hình lên GPU đúng cách (`.to(device)`, `eval()`, `no_grad()`)
- [ ] Ước lượng _VRAM_ thô cho inference của GPT mini và dùng float16 để tiết kiệm bộ nhớ
- [ ] Bọc mô hình trong một _API HTTP_ đơn giản (FastAPI) để sinh văn bản
- [ ] Hiểu các nút thắt về _bộ nhớ và tốc độ_ khi tự host, và cách giảm bớt

📌 **Yêu cầu trước:** Bạn đã có một checkpoint GPT mini đã huấn luyện (Tuần 5-6), hiểu vòng lặp generate với temperature/top-k (Bài 33), và KV-cache (Bài 34). Đã đọc Bài 2 về GPU/VRAM.

---

### 1. Inference khác training thế nào?

**Analogy:** Học lái xe (training) bạn cần thầy ngồi cạnh, phanh phụ, ghi chú sửa lỗi - tốn chỗ, tốn công. Còn khi đã có bằng và chỉ _lái đi làm_ (inference), bạn chỉ cần chiếc xe - bỏ hết phanh phụ, thầy, sổ ghi chú. Inference gọn nhẹ hơn training rất nhiều.

Trong PyTorch, sự khác biệt quan trọng nhất là: inference _không cần gradient_. Không lưu đồ thị tính toán (computation graph) để backprop nghĩa là tiết kiệm rất nhiều bộ nhớ activation.

| Khía cạnh                   | Training               | Inference (tự host)      |
| :-------------------------- | :--------------------- | :----------------------- |
| **Gradient**                | Có (backward)          | KHÔNG                    |
| **Optimizer state (AdamW)** | Có (~2x số tham số)    | KHÔNG                    |
| **Chế độ module**           | `model.train()`        | `model.eval()`           |
| **Dropout / BatchNorm**     | Hoạt động (train mode) | Tắt / dùng thống kê chạy |
| **Bộ nhớ activation**       | Lớn (giữ để backward)  | Nhỏ (giải phóng ngay)    |
| **Lệnh then chốt**          | `loss.backward()`      | `with torch.no_grad():`  |

⚠️ Quên `model.eval()` là lỗi kinh điển: dropout vẫn bật khiến kết quả sinh _ngẫu nhiên bất thường_ giữa các lần chạy. Quên `torch.no_grad()` thì không sai kết quả nhưng _phí VRAM_ vì PyTorch dựng đồ thị backward vô ích.

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device) # 'cuda' nếu có GPU, ngược lại 'cpu'

# eval() + no_grad() là bộ đôi bắt buộc cho inference
model.eval() # turn off dropout, use running stats
with torch.no_grad(): # do not build the backward graph -> save VRAM
    logits, _ = model(idx) # idx shape: (B, T) -> logits shape: (B, T, vocab_size)
```

---

### 2. Nạp checkpoint và đưa mô hình lên GPU

**Analogy:** Checkpoint giống như _bản thiết kế + vật liệu_ để dựng lại đúng ngôi nhà. Bạn phải dựng lại đúng khung nhà (kiến trúc GPTMini với đúng siêu tham số) rồi mới lắp “nội thất” (trọng số) vào.

```python
# Recreate the SAME architecture with the SAME hyper-parameters as training
model = GPTMini(vocab_size=vocab_size, n_embd=384, n_head=6,
                n_layer=6, block_size=256)

# Load only the learned weights (state_dict), not the whole Python object
ckpt = torch.load("gpt_mini.pt", map_location=device) # dict of tensors
model.load_state_dict(ckpt["model"])

model.to(device) # move all parameters to GPU
model.eval()     # inference mode

n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params/1e6:.1f}M") # e.g. Params: 10.7M
```

💡 Luôn lưu _state_dict_ (chỉ tensor trọng số) thay vì `torch.save(model)` (đóng gói cả class). `state_dict` di động hơn: bạn nạp được kể cả khi refactor code, chỉ cần giữ đúng kiến trúc.

**Luồng dữ liệu khi inference** - từ token vào tới token mới ra:

```text
prompt (chuỗi ký tự)
        |  encode
        v
  idx  (B, T)  int64          <- token ids
        |  token_embedding + position_embedding
        v
  x    (B, T, n_embd)         <- (B, T, 384)
        |  N Transformer blocks (attention + feed-forward, mỗi block có residual + LayerNorm)
        v
  x    (B, T, n_embd)
        |  final LayerNorm -> lm_head (Linear n_embd -> vocab_size)
        v
  logits (B, T, vocab_size)   <- (B, T, V)
        |  lấy bước cuối logits[:, -1, :] -> temperature -> softmax
        v
  probs (B, vocab_size)  --sample--> token mới (B, 1)  --> nối vào idx, lặp lại
```

---

### 3. Ước lượng VRAM cho inference

**Analogy:** Trước khi mở quán bạn phải biết _cái tủ lạnh chứa được bao nhiêu_ để không nhận quá nhiều nguyên liệu. VRAM là “tủ lạnh” của GPU.

Công thức thô cho _trọng số_:

$$\text{VRAM}_{\text{weights}} \approx \text{số\_tham\_số} \times \text{số\_byte\_mỗi\_tham\_số}$$

- float32 = 4 byte/tham số
- float16 (hoặc bfloat16) = 2 byte/tham số

Ví dụ GPT mini _~10.7M tham số_:

```text
float32:  10.7e6 × 4 byte ≈ 42.8 MB
float16:  10.7e6 × 2 byte ≈ 21.4 MB   (giảm một nửa 🎯)
```

Khi inference, tổng VRAM $\approx$ _trọng số + KV-cache + activation tạm thời + phần đệm của CUDA_. Với mô hình nhỏ như GPT mini, con số vài chục MB là rất nhẹ - chạy được cả trên GPU khiêm tốn. Nhưng KV-cache _lớn dần theo độ dài sinh_ và _theo số yêu cầu song song_ (batch), nên đừng quên nó khi phục vụ nhiều người.

🔑 So với _training_, inference tiết kiệm VRAM lớn: training còn cần gradient (~1x số tham số nữa) + optimizer state của AdamW (~2x: momentum và variance) + activation giữ cho backward. Đó là lý do bạn _suy luận_ được mô hình mà không _huấn luyện_ nổi nó trên cùng chiếc card.

```python
# Cast weights to float16 to halve the weight memory (great for pure inference)
model.half() # all params -> float16
input_ids = input_ids.to(device) # keep token ids as int64 (indices)

with torch.no_grad():
    logits, _ = model(input_ids) # internally runs in float16
# Note: on CPU, prefer float32; float16 is meant for GPU (CUDA) speedups.
```

---

### 4. Bọc mô hình trong một API sinh văn bản

**Analogy:** Cho tới giờ mô hình như một _món ăn tự nấu tự ăn_. API là _cửa sổ order_: khách gửi “prompt”, quán trả “văn bản sinh ra”. Ai cũng gọi được mà không cần biết trong bếp có gì.

Ý tưởng cốt lõi khi tự host: _nạp mô hình MỘT LẦN lúc khởi động_ (đắt, chỉ làm 1 lần), rồi _mỗi request chỉ chạy forward + sample_ (rẻ, làm nhiều lần). Dưới đây là một API tối giản với FastAPI.

```python
# app.py  --  run:  uvicorn app:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI
from pydantic import BaseModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load ONCE at startup (expensive, done a single time) ---
model = GPTMini(vocab_size=vocab_size, n_embd=384, n_head=6,
                n_layer=6, block_size=256)
model.load_state_dict(torch.load("gpt_mini.pt", map_location=device)["model"])
model.to(device).eval() # GPU + inference mode

app = FastAPI()

class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8 # <1 -> "chắc chắn" hơn, >1 -> "sáng tạo"/hỗn loạn hơn
    top_k: int = 50

@torch.no_grad() # no gradients for every request -> save VRAM
def generate_text(req: GenRequest) -> str:
    ids = torch.tensor([encode(req.prompt)], dtype=torch.long, device=device) # (1, T)
    out = model.generate(ids, max_new_tokens=req.max_new_tokens,
                         temperature=req.temperature, top_k=req.top_k) # (1, T+max_new)
    return decode(out[0].tolist())

@app.post("/generate")
def generate(req: GenRequest):
    return {"text": generate_text(req)}
```

Gọi thử bằng curl (client bất kỳ đều gọi được endpoint HTTP này):

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Ngày xửa ngày xưa", "max_new_tokens": 80, "temperature": 0.8}'
# -> {"text": "Ngày xửa ngày xưa ... (văn bản do GPT mini sinh ra) ..."}
```

💡 `temperature` và `top_k` là _tham số của request_, không phải của mô hình. Cùng một mô hình phục vụ nhiều “khẩu vị” khác nhau: temperature thấp cho câu trả lời ổn định, cao cho sáng tạo hơn.

---

### Nhìn sâu hơn

_Bộ nhớ và tốc độ là hai nút thắt chính khi tự host._ Vài điểm thực tế đáng nhớ:

1. _Tốc độ đo bằng token/giây._ Với KV-cache (Bài 34), sinh 1 token mới chỉ cần forward cho token đó thay vì cả câu - đây là đòn bẩy tốc độ lớn nhất. Không có KV-cache, chi phí sinh câu dài tăng theo bình phương độ dài.
2. _Batching request._ Nếu nhiều người gọi cùng lúc, gộp (batch) nhiều prompt vào một forward tận dụng GPU tốt hơn - nhưng KV-cache cho cả batch _ăn VRAM nhiều hơn_. Đây là đánh đổi thông lượng (throughput) và bộ nhớ.
3. _float16 / bfloat16_ giảm một nửa VRAM trọng số và thường _nhanh hơn_ trên GPU có Tensor Core (nhớ lại Bài 2). Với mô hình lớn, đây gần như bắt buộc.
4. _Chi phí nạp một lần._ Nạp checkpoint + chuyển lên GPU tốn thời gian; đừng làm việc này trong mỗi request. Nạp lúc khởi động server và giữ mô hình sống trong bộ nhớ.
5. _Đo VRAM thật_ bằng `torch.cuda.memory_allocated()` để biết mình đang dùng bao nhiêu, thay vì đoán:

```python
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=50)
peak = torch.cuda.max_memory_allocated() / 1024 ** 2 # MB
print(f"Peak VRAM: {peak:.1f} MB") # e.g. Peak VRAM: 68.3 MB
```

---

### Thực hành

- **[Dễ]** Nạp checkpoint GPT mini của bạn, in số tham số bằng `sum(p.numel() for p in model.parameters())`, và tính tay VRAM trọng số cho cả float32 và float16. Kiểm tra `torch.cuda.is_available()`.
- **[Trung bình]** Viết endpoint `/generate` với FastAPI như mục 4. Chạy `uvicorn app:app --port 8000` rồi gửi 3 prompt với temperature lần lượt 0.2, 0.8, 1.5. Quan sát: temperature thấp cho văn bản lặp/ổn định, cao cho hỗn loạn hơn.
- **[Thử thách]** Đo token/giây (chia `max_new_tokens` cho thời gian sinh) _có và không có KV-cache_. Dùng `torch.cuda.max_memory_allocated()` để so sánh VRAM đỉnh khi sinh 50 vs 500 token. Giải thích vì sao VRAM tăng khi sinh dài hơn (gợi ý: KV-cache).

---

### Hiểu lầm thường gặp

- ❌ “Chạy inference cứ để `model.train()` cũng được.” ➔ ✅ Phải `model.eval()`: nếu không, dropout bật khiến output khác nhau mỗi lần chạy, không ổn định.
- ❌ “Không cần `torch.no_grad()`, chỉ chậm chút thôi.” ➔ ✅ Thiếu `no_grad()` khiến PyTorch dựng đồ thị backward, _lãng phí VRAM_ và có thể tràn bộ nhớ khi câu dài.
- ❌ “Mua GPU chơi game là đủ VRAM cho mọi mô hình.” ➔ ✅ VRAM là giới hạn cứng; mô hình lớn cần nhiều VRAM hơn card game phổ thông. Ước lượng trước (mục 3) rồi mới quyết.
- ❌ “`float16` luôn tốt hơn `float32`.” ➔ ✅ `float16` tiết kiệm bộ nhớ và nhanh trên GPU có Tensor Core, nhưng trên _CPU_ thường nên dùng `float32`; và độ chính xác thấp hơn đôi khi gây sai số.
- ❌ “Nạp mô hình trong mỗi request cho chắc.” ➔ ✅ Nạp một lần lúc khởi động; nạp lại mỗi request làm API chậm kinh khủng và tốn tài nguyên.

---

### Mẹo ghi nhớ

- _Bộ đôi bất ly thân của inference:_ `model.eval()` + `with torch.no_grad():`.
- _Nạp một lần, phục vụ nhiều lần_ - đó là tinh thần của tự host.
- _VRAM $\approx$ tham số $\times$ byte_ (4 cho fp32, 2 cho fp16) + KV-cache + đệm.
- Mẹo nhớ nhanh: _“eval, no_grad, half, serve”_ - bốn từ tóm gọn cả bài. 🎯

---

### Tóm tắt trực quan

| Khái niệm            | Một dòng để nhớ                                               |
| :------------------- | :------------------------------------------------------------ |
| **Inference**        | Chỉ forward, không gradient, không optimizer                  |
| **model.eval()**     | Tắt dropout, dùng thống kê chạy - bắt buộc                    |
| **torch.no_grad()**  | Không dựng đồ thị backward → tiết kiệm VRAM                   |
| **state_dict**       | Chỉ tensor trọng số, di động khi nạp lại                      |
| **VRAM trọng số**    | $\approx$ số tham số $\times$ 4 (fp32) hoặc $\times$ 2 (fp16) |
| **float16**          | Giảm nửa bộ nhớ, nhanh hơn trên GPU Tensor Core               |
| **API sinh văn bản** | Nạp mô hình 1 lần lúc start, mỗi request forward + sample     |
| **Nút thắt tự host** | Bộ nhớ (VRAM) và tốc độ (token/giây)                          |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [How to setup NVIDIA GPU for PyTorch on Windows 10/11](https://www.youtube.com/watch?v=r7Am-ZGMef8)
- PyTorch docs: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- PyTorch tutorials (saving & loading models, inference): [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT của Andrej Karpathy (ví dụ generate/inference thực tế): [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Hugging Face docs (khái niệm inference/serving): [https://huggingface.co/docs](https://huggingface.co/docs)

---

### Tổng kết

- _Inference_ khác training ở chỗ không cần gradient/optimizer $\to$ dùng `model.eval()` + `torch.no_grad()` để tiết kiệm VRAM và chạy ổn định.
- Nạp checkpoint qua _state_dict_, dựng lại đúng kiến trúc, rồi `.to(device)` đưa lên GPU.
- _Ước lượng VRAM_ thô = số tham số $\times$ byte mỗi tham số; `float16` giảm một nửa bộ nhớ trọng số.
- Bọc mô hình trong một _API_ (FastAPI) - nạp một lần lúc khởi động, mỗi request chỉ forward + sample; temperature / top_k là tham số của request.
- Hai nút thắt tự host là _bộ nhớ (VRAM)_ và _tốc độ (token/giây)_ - KV-cache, batching và float16 là các đòn bẩy chính.

Giờ bạn đã tự host được GPT mini của mình trên GPU cục bộ - mô hình nhỏ nhưng “chạy được thật” và phục vụ được người khác. Ở _Bài 36: Tải và tự host một model pretrained từ Hugging Face_, chúng ta sẽ đi tiếp bước tiếp theo: lấy một mô hình đã được pretrain hàng tỉ token trên mạng về máy và host nó theo chuẩn công nghiệp. Hẹn gặp lại! 🤖
