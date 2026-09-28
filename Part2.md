### Giới thiệu

Ở _Bài 1_, bạn đã cài Python, PyTorch và bật được GPU miễn phí trên Google Colab, rồi verify bằng `torch.cuda.is_available()`. Tuyệt vời - nhưng chắc bạn đang thắc mắc: _“Cái GPU đó mạnh cỡ nào? Mình có nên mua hẳn một card về nhà tự huấn luyện không, hay cứ thuê cloud cho khỏe?”_.

Hãy tưởng tượng GPU như một _nhà bếp trong nhà hàng_. CPU giống một đầu bếp giỏi làm được mọi món nhưng chỉ có _vài tay_ (vài nhân) - làm tuần tự rất nhanh cho từng món. GPU thì như một bếp có _hàng nghìn phụ bếp_ làm song song những việc lặp lại y hệt nhau (thái hành, đảo chảo). Huấn luyện LLM về bản chất là _nhân ma trận khổng lồ lặp đi lặp lại_ - đúng kiểu việc mà “nghìn phụ bếp” làm cực nhanh. Còn _VRAM_ (bộ nhớ trên card) chính là _diện tích mặt bàn bếp_: bàn nhỏ thì không bày hết nguyên liệu, mô hình lớn sẽ “tràn bàn” và báo lỗi hết bộ nhớ.

Bài này không có nhiều code kiến trúc, nhưng là bài _cực kỳ thực tế_: hiểu sai phần cứng, bạn sẽ đốt tiền vô ích hoặc suốt ngày gặp lỗi `CUDA out of memory`.

> 🔑 **Một câu để nhớ:** Với LLM, _VRAM (dung lượng nhớ) thường là nút thắt trước cả tốc độ tính toán_ - card có bao nhiêu GB VRAM quyết định mô hình lớn tới đâu bạn nhét vừa.

---

### Bạn sẽ học được gì?

- [ ] Vì sao GPU quan trọng với LLM, và _VRAM_ khác _tốc độ tính toán_ thế nào
- [ ] Cách đọc thông số một card đồ họa: VRAM, CUDA core, _Tensor Core_, băng thông bộ nhớ
- [ ] So sánh _mua card tự host_ (dòng NVIDIA RTX tiêu dùng) và _thuê cloud_ (Colab, các dịch vụ thuê GPU)
- [ ] Cách _ước lượng VRAM thô_ theo số tham số và kiểu dữ liệu (fp32/fp16) - cho cả inference lẫn training
- [ ] Khi nào nên tự host, khi nào nên thuê cloud - và cách tính chi phí sơ bộ

📌 **Yêu cầu trước:** Đã làm xong Bài 1 (biết mở Colab, chạy được `torch.cuda.is_available()`). Không cần đã sở hữu GPU rời.

---

### 1. Vì sao LLM "nghiện" GPU?

Huấn luyện và chạy một mô hình Transformer chủ yếu là _phép nhân ma trận_ (matmul): mỗi lớp attention và feed-forward đều là những phép nhân ma trận rất lớn, lặp lại hàng triệu lần. CPU làm tuần tự thì chậm; GPU có hàng nghìn nhân xử lý _song song_ nên nhanh hơn hàng chục đến hàng trăm lần cho loại việc này.

Có _hai đại lượng_ bạn phải phân biệt rạch ròi:

| Đại lượng                 | Là gì (analogy)              | Đơn vị thường gặp | Ảnh hưởng                              |
| :------------------------ | :--------------------------- | :---------------- | :------------------------------------- |
| **Compute** (tốc độ tính) | Số phụ bếp × tốc độ tay      | TFLOPS            | Mô hình chạy/huấn luyện _nhanh_ cỡ nào |
| **VRAM** (dung lượng nhớ) | Diện tích mặt bàn bếp        | GB                | Mô hình _lớn tới đâu_ thì nhét vừa     |
| **Băng thông nhớ**        | Tốc độ bưng đồ từ kho ra bàn | GB/s              | Bơm dữ liệu vào nhân nhanh cỡ nào      |

💡 Với người tự học LLM, _VRAM thường là nút thắt xuất hiện TRƯỚC_. Bạn hay gặp lỗi “hết bộ nhớ” (mô hình không vừa) sớm hơn nhiều so với chuyện “chờ hơi lâu” (thiếu tốc độ).

Một mẹo kiểm tra nhanh thông số card GPU bạn đang có trong PyTorch:

```python
import torch

# Report GPU name and total VRAM in GB
if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx) # e.g. "Tesla T4"
    total_bytes = torch.cuda.get_device_properties(idx).total_memory
    total_gb = total_bytes / (1024 ** 3) # bytes -> GiB
    print(f"GPU: {name} | VRAM: {total_gb:.1f} GB")
else:
    print("No CUDA GPU available; running on CPU.")

# Example output on a free Colab T4:
# GPU: Tesla T4 | VRAM: 14.7 GB
```

---

### 2. Đọc thông số một card đồ họa

Khi nhìn spec một card NVIDIA, hãy chú ý các dòng sau (giải thích bằng lời trước, thuật ngữ sau):

- **VRAM (GB):** mặt bàn bếp. Ví dụ dòng tiêu dùng thường thấy 8 GB, 12 GB, 16 GB, 24 GB. _Con số quan trọng nhất_ với người tập tự host LLM.
- **CUDA core:** các nhân đa dụng để tính song song.
- **Tensor Core:** nhân _chuyên biệt_ để nhân ma trận ở độ chính xác thấp (fp16/bf16), tăng tốc rất mạnh phần matmul của Transformer. Card có nhiều Tensor Core + hỗ trợ _mixed precision_ sẽ huấn luyện nhanh và tiết kiệm VRAM hơn.
- **Băng thông bộ nhớ (GB/s):** tốc độ chuyển dữ liệu giữa VRAM và nhân. Cao thì nhân ít bị “đói dữ liệu”.
- **Kiến trúc / thế hệ:** thế hệ mới thường hỗ trợ định dạng số hiệu quả hơn (như `bf16`) và tối ưu attention tốt hơn.

⚠️ Đừng chỉ nhìn tên card cho oách. Một card game đời mới _nhưng chỉ 8 GB VRAM_ có thể _thua_ một card cũ hơn _có 24 GB VRAM_ cho việc chạy mô hình lớn - vì mô hình không vừa vào 8 GB thì tốc độ nhanh cũng vô nghĩa.

---

### 3. Ước lượng VRAM theo số tham số (phần quan trọng nhất)

Đây là công thức “nằm lòng”. Trước hết: mỗi _tham số_ (parameter) là một con số thực được lưu bằng một kiểu dữ liệu.

| Kiểu dữ liệu             | Số byte / tham số |
| :----------------------- | :---------------- |
| **fp32** (float 32-bit)  | **4 byte**        |
| **fp16 / bf16** (16-bit) | **2 byte**        |
| **int8** (lượng tử hóa)  | **1 byte**        |

_Bộ nhớ chỉ để CHỨA trọng số_ (weights):

```text
VRAM_weights ≈ (số tham số) × (số byte mỗi tham số)
```

Ví dụ tính từng bước cho mô hình _7 tỷ tham số (7B)_:

```text
7B ở fp16: 7e9 params × 2 byte = 14e9 byte ≈ 14 GB   (chỉ riêng weights, để inference)
7B ở fp32: 7e9 params × 4 byte = 28e9 byte ≈ 28 GB
```

Nhưng _huấn luyện tốn nhiều hơn nạp để chạy rất nhiều_, vì ngoài weights còn phải giữ:

- **Gradient:** cùng kích thước với weights (mỗi tham số 1 gradient).
- **Optimizer state:** với _AdamW_ cần lưu 2 mô-men (momentum + variance) cho mỗi tham số $\to$ thêm ~2× kích thước weights (thường giữ ở fp32).
- **Activation:** các giá trị trung gian của mỗi lớp, phụ thuộc `batch_size` và độ dài chuỗi (`seq_len`) - phần này dao động mạnh.

Quy tắc ngón tay cái cho training với AdamW (bỏ qua activation): _~16 byte / tham số_ (4 weight fp32 + 4 gradient fp32 + 8 optimizer state) - hoặc nhẹ hơn nếu dùng mixed precision. Nghĩa là huấn luyện mô hình 7B “chay” có thể cần _~112 GB_ chỉ cho weights + gradient + optimizer, chưa kể activation $\to$ _vượt xa_ một card tiêu dùng.

```python
def estimate_vram_gb(num_params, mode="inference", dtype_bytes=2):
    # Rough VRAM estimate (GB). Ignores activations, which vary with batch/seq.
    weights = num_params * dtype_bytes # bytes for weights
    if mode == "inference":
        total = weights
    else:
        # training with AdamW: weights + grad + 2 optimizer moments (~16 B/param at fp32)
        total = num_params * 16
    return total / (1024 ** 3) # bytes -> GiB

for p in [1e7, 1e8, 7e9]: # 10M, 100M, 7B params
    inf = estimate_vram_gb(p, "inference", dtype_bytes=2)
    tr  = estimate_vram_gb(p, "training")
    print(f"{p:>10.0e} params | inference(fp16) ~ {inf:6.2f} GB | training ~ {tr:7.2f} GB")

# Example output:
#      1e+07 params | inference(fp16) ~   0.02 GB | training ~    0.15 GB
#      1e+08 params | inference(fp16) ~   0.19 GB | training ~    1.49 GB
#      7e+09 params | inference(fp16) ~  13.04 GB | training ~  104.31 GB
```

🎯 Tin vui cho khóa này: _GPT mini_ ta sẽ xây có cỡ chỉ vài triệu đến vài chục triệu tham số $\to$ chạy thoải mái ngay trên _GPU miễn phí của Colab_ (khoảng 15 GB). Bạn KHÔNG cần mua card đắt tiền để hoàn thành khóa.

---

### 4. Mua card tự host vs thuê cloud

Sơ đồ quyết định thô:

```text
Bạn cần GPU cho LLM?
        |
        v
Dùng thường xuyên (hàng ngày, nhiều giờ) & có mô hình vừa VRAM card tiêu dùng?
        |                                    |
       Có                                   Không / thỉnh thoảng
        |                                    |
        v                                    v
  Cân nhắc MUA card tự host            Dùng CLOUD (Colab / dịch vụ thuê GPU)
  (RTX tiêu dùng, ưu tiên VRAM cao)    trả theo giờ, đổi loại GPU linh hoạt
```

Bảng so sánh (nguyên tắc chung - _giá và tên sản phẩm thay đổi theo thời điểm_, chỉ minh họa tương đối):

| Tiêu chí               | Tự host (mua card, ví dụ RTX 30/40 series) | Cloud (Colab, dịch vụ thuê GPU)           |
| :--------------------- | :----------------------------------------- | :---------------------------------------- |
| **Chi phí ban đầu**    | Cao (mua card + máy + điện)                | Gần như 0                                 |
| **Chi phí dài hạn**    | Rẻ dần nếu dùng liên tục                   | Trả theo giờ, cộng dồn lớn nếu chạy nhiều |
| **Linh hoạt loại GPU** | Cố định 1 card đã mua                      | Đổi từ nhỏ đến rất lớn tùy nhu cầu        |
| **Sẵn sàng**           | Luôn có, không giới hạn thời lượng         | Có thể bị ngắt phiên, giới hạn quota      |
| **Dữ liệu riêng tư**   | Nằm hoàn toàn trên máy bạn                 | Nằm trên hạ tầng nhà cung cấp             |
| **Bảo trì**            | Bạn tự lo (driver, tản nhiệt)              | Nhà cung cấp lo                           |

💡 Với _RTX tiêu dùng_, hãy ưu tiên _VRAM cao_ (ví dụ bản 16 GB hoặc 24 GB) hơn là chạy theo tốc độ, vì VRAM là nút thắt. Card 24 GB cho phép nghịch mô hình lớn hơn và batch lớn hơn nhiều so với card 8-12 GB dù cùng thế hệ.

---

### Nhìn sâu hơn

_Vì sao mixed precision (fp16/bf16) tiết kiệm VRAM và nhanh hơn?_ Lưu trọng số/tính toán ở 16-bit thay vì 32-bit $\to$ _cắt đôi_ bộ nhớ cho weights và activation, đồng thời Tensor Core xử lý 16-bit nhanh hơn. PyTorch có `torch.cuda.amp` (Automatic Mixed Precision) để tự động làm việc này an toàn:

```python
import torch

scaler = torch.cuda.amp.GradScaler() # scales loss to avoid fp16 underflow

# Inside the training loop (pseudo-outline):
optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits = model(x) # forward in fp16 where safe
    loss = loss_fn(logits, y)
scaler.scale(loss).backward() # scaled backward pass
scaler.step(optimizer)        # unscale + AdamW step
scaler.update()
```

_Các mẹo cắt VRAM khi bị `CUDA out of memory`:_ giảm `batch_size`; giảm độ dài chuỗi `seq_len`; bật _gradient accumulation_ (cộng dồn gradient qua nhiều micro-batch để “giả lập” batch lớn mà không tốn VRAM); dùng _gradient checkpointing_ (đánh đổi tính lại activation lấy ít bộ nhớ hơn); và với inference thì dùng _lượng tử hóa_ (int8/4-bit) để nhét mô hình lớn vào card nhỏ.

⚠️ Đừng quên _activation_: khi training, activation có thể chiếm bộ nhớ _ngang hoặc hơn_ weights, đặc biệt với `seq_len` dài và batch lớn. Công thức “byte/tham số” chỉ là chặn dưới; luôn để lại khoảng đệm VRAM.

---

### Thực hành

- **[Dễ]** Chạy đoạn code ở Mục 1 trên Colab (bật runtime GPU) để in tên GPU và VRAM. Ghi lại card bạn được cấp và số GB.
- **[Trung bình]** Dùng hàm `estimate_vram_gb` ở Mục 3, tính VRAM cần để _inference_ một mô hình _1.3B_ tham số ở fp16, rồi ở fp32. Cùng mô hình đó, ước lượng VRAM để _training_ với AdamW. Nó có vừa card 16 GB không?
- **[Thử thách]** Giả sử bạn có card _12 GB VRAM_. Hãy lập bảng: với fp16, mô hình _inference_ lớn nhất (theo số tham số) mà bạn nhét vừa là bao nhiêu nếu chừa 2 GB đệm cho activation/hệ thống? Viết một hàm `max_params_for(vram_gb, reserve_gb=2, dtype_bytes=2)` trả về số tham số tối đa.

---

### Hiểu lầm thường gặp

- ❌ “Cứ mua GPU chơi game là đủ VRAM cho mọi mô hình.” $\to$ ✅ Nhiều card game mạnh về tốc độ _nhưng VRAM thấp (8 GB)_; mô hình lớn không vừa. _VRAM_ mới là con số quyết định mô hình lớn tới đâu.
- ❌ “Tốc độ tính (TFLOPS) là thứ duy nhất cần nhìn.” $\to$ ✅ Với LLM, _VRAM và băng thông bộ nhớ_ thường là nút thắt trước; TFLOPS cao mà VRAM nhỏ thì vẫn kẹt.
- ❌ “Training và inference tốn VRAM như nhau.” $\to$ ✅ Training tốn _nhiều hơn hẳn_: ngoài weights còn gradient + optimizer state (AdamW) + activation, có thể gấp nhiều lần.
- ❌ “Phải mua GPU đắt tiền mới học được LLM.” $\to$ ✅ GPT mini của khóa này chạy tốt trên _GPU miễn phí Colab_; chỉ khi cần chạy liên tục/mô hình lớn mới tính chuyện tự host.
- ❌ “fp16 làm mất chính xác nên đừng dùng.” $\to$ ✅ _Mixed precision_ dùng đúng cách (có `GradScaler`) vừa nhanh vừa tiết kiệm VRAM mà chất lượng gần như không đổi - là chuẩn thực tế hiện nay.

---

### Mẹo ghi nhớ

- _VRAM = mặt bàn bếp_ (chứa được bao nhiêu), _Compute = số phụ bếp_ (làm nhanh cỡ nào).
- Nhớ nhanh byte/tham số: _fp32 = 4, fp16 = 2, int8 = 1_.
- Ước lượng thô: _inference ≈ params × byte/param_; _training với AdamW ≈ params × 16 byte_ (chưa tính activation).
- Gặp `CUDA out of memory`? Nghĩ ngay: _giảm batch $\to$ giảm seq_len $\to$ gradient accumulation $\to$ mixed precision $\to$ quantization_.

---

### Tóm tắt trực quan

| Khái niệm                 | Một dòng để nhớ                                           |
| :------------------------ | :-------------------------------------------------------- |
| **VRAM**                  | Dung lượng nhớ trên card - quyết định mô hình lớn tới đâu |
| **Compute (TFLOPS)**      | Tốc độ nhân ma trận - quyết định nhanh cỡ nào             |
| **Tensor Core**           | Nhân chuyên biệt tăng tốc matmul ở fp16/bf16              |
| **byte/param**            | fp32=4, fp16/bf16=2, int8=1                               |
| **VRAM inference**        | ≈ params × byte/param                                     |
| **VRAM training (AdamW)** | ≈ params × 16 byte (+ activation)                         |
| **Tự host**               | Rẻ dài hạn nếu dùng nhiều; ưu tiên VRAM cao               |
| **Cloud**                 | Không tốn đầu tư; linh hoạt; trả theo giờ                 |
| **Mixed precision**       | Cắt đôi VRAM, tăng tốc; dùng torch.cuda.amp               |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [How to setup NVIDIA GPU for PyTorch on Windows 10/11](https://www.youtube.com/watch?v=r7Am-ZGMef8)
- PyTorch - CUDA semantics & bộ nhớ: [https://pytorch.org/docs/stable/notes/cuda.html](https://pytorch.org/docs/stable/notes/cuda.html)
- PyTorch - Automatic Mixed Precision (AMP): [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- nanoGPT (Andrej Karpathy) - dự án tham chiếu của khóa: [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- The Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- Hugging Face Docs (tối ưu bộ nhớ, quantization): [https://huggingface.co/docs](https://huggingface.co/docs)

---

### Tổng kết

- _VRAM_ là nút thắt số một khi tự học LLM: nó quyết định mô hình lớn tới đâu bạn nhét vừa.
- Đọc spec card: nhìn _VRAM_ trước, rồi _Tensor Core / mixed precision_, rồi băng thông và tốc độ.
- Ước lượng VRAM: _inference ≈ params × byte/param_; _training với AdamW ≈ params × 16 byte_ (chưa kể activation).
- _Tự host_ hợp lý khi dùng liên tục và ưu tiên VRAM cao; _cloud/Colab_ hợp lý khi mới học, chạy thỉnh thoảng, cần linh hoạt.
- GPT mini của khóa này _chạy tốt trên GPU miễn phí_ - bạn không cần mua card để đi hết chặng đường.

Giờ ta đã có bức tranh phần cứng đầy đủ. Ở _Bài 3: Ôn tập PyTorch cốt lõi - tensor, autograd và nn.Module_, ta sẽ xắn tay vào chính công cụ dựng nên mọi thứ: cách tạo và thao tác `tensor`, cơ chế `autograd` tự tính đạo hàm khi `backward()`, và cách gói mô hình trong `nn.Module`. Đây là nền tảng để về sau ta lắp ráp attention và cả một GPT mini. Hẹn gặp ở bài sau! 🚀
