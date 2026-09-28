# Bài 36: Tải và tự host một model pretrained từ Hugging Face

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 7:** Tokenization thực tế, tối ưu & tự host

---

### Giới thiệu

Hãy tưởng tượng bạn muốn nấu một bữa ăn ngon. Bạn có thể tự trồng lúa, xay bột, ủ men suốt nhiều tháng - hoặc ra siêu thị mua sẵn nguyên liệu chất lượng cao rồi chỉ việc chế biến. Việc dùng lại model pretrained cũng giống hệt vậy: thay vì tự huấn luyện một GPT khổng lồ tốn hàng nghìn giờ GPU và hàng đống tiền điện, bạn _tải sẵn weights_ mà cộng đồng đã dày công train về máy và chạy ngay.

Ở _Bài 35_, bạn đã tự host CHÍNH mô hình mình train - đưa GPT mini lên GPU cục bộ, chạy inference, đóng gói một API sinh văn bản. Tuyệt vời, nhưng GPT mini của bạn chỉ vài triệu tham số nên chất lượng còn khiêm tốn. Bài này ta lật ngược góc nhìn: thay vì host model của MÌNH, ta sẽ _tải WEIGHTS của model người khác đã train sẵn_ (ví dụ GPT-2 do OpenAI công bố) rồi cũng tự host nó. Cùng một kỹ năng inference, nhưng nguồn weights hoàn toàn khác.

> 🔑 **Một câu để nhớ:** Model pretrained = weights có sẵn của người khác, tải về là chạy được ngay - bạn không phải train lại từ đầu.

---

### Bạn sẽ học được gì?

- [ ] Hiểu vì sao nên dùng lại model pretrained thay vì tự train mọi thứ
- [ ] Biết Hugging Face Hub là gì: model card, license, kích thước/số tham số
- [ ] Cài transformers và tải + chạy inference một model thật (`gpt2`/`distilgpt2`) bằng PyTorch
- [ ] Phân biệt rõ 3 cách serve: transformers trực tiếp vs Ollama vs vLLM
- [ ] Biết khi nào nên TỰ TRAIN từ đầu vs DÙNG LẠI model pretrained
- [ ] Lưu ý về LICENSE, VRAM (nối Bài 2) và an toàn/đạo đức khi tự host

📌 **Yêu cầu trước:** đã hoàn thành Bài 35 (tự host inference), biết chạy PyTorch trên GPU/CPU và hiểu khái niệm tham số (Bài 24) + VRAM (Bài 2).

---

### 1. Vì sao dùng lại model pretrained?

Tự train một GPT cỡ lớn từ đầu là chuyện rất đắt đỏ: bạn cần _dữ liệu khổng lồ_ (hàng trăm tỷ token), _hàng trăm/hàng nghìn GPU-giờ_, và kinh nghiệm điều chỉnh siêu tham số. GPT mini của bạn ở các bài trước chạy tốt để HỌC nguyên lý, nhưng để có một mô hình sinh văn bản mạch lạc thì cần quy mô lớn hơn nhiều.

May mắn là cộng đồng đã chia sẻ _hàng nghìn model mở_ trên Hugging Face Hub. Bạn chỉ cần tải weights về là dùng được ngay - giống như cài một thư viện có sẵn thay vì viết lại từ số 0.

```text
PHÂN BIỆT RÕ:

  Bài 35:  [Dữ liệu của bạn] -> [BẠN train] -> [weights của BẠN] -> host

  Bài 36:  [Người khác đã train sẵn] -> [tải weights có sẵn] -> host
                     (Hugging Face Hub)
```

💡 Đây KHÁC với fine-tuning: bài này chỉ tải weights về chạy nguyên trạng (inference). Việc chỉnh weights cho tác vụ riêng của bạn (fine-tuning/LoRA) là chủ đề của khóa chuyên sâu sẽ nhắc ở cuối bài.

---

### 2. Hugging Face Hub là gì?

Hugging Face Hub là một “kho model” khổng lồ - hãy hình dung nó như một App Store dành cho model AI. Mỗi model có một trang gọi là _model card_ cho biết:

- _Tên model_ (ví dụ `gpt2`, `distilgpt2`) - dùng chuỗi này để tải.
- _License_ - giấy phép cho biết bạn được dùng model vào việc gì (thương mại hay không, có điều kiện gì).
- _Kích thước / số tham số_ - ví dụ GPT-2 bản gốc có ~124 triệu tham số (còn có các bản lớn hơn 355M, 774M, 1.5B).
- _Cách dùng mẫu_ và các cảnh báo (bias, giới hạn).

Để tải và chạy model, ta cài thư viện `transformers` của Hugging Face:

```bash
# Install the Hugging Face libraries + PyTorch
pip install transformers torch accelerate
```

- `transformers`: cung cấp lớp `AutoModelForCausalLM`, `AutoTokenizer`, `pipeline`.
- `torch`: backend tính toán (bạn đã dùng suốt khóa).
- `accelerate`: giúp đặt model lên đúng thiết bị (CPU/GPU) và quản lý bộ nhớ.

---

### 3. Tải + chạy inference một model nhỏ bằng transformers

Ta dùng _distilgpt2_ - một bản GPT-2 nhỏ, được “chưng cất” (distilled) nên nhẹ, tải nhanh, chạy được cả trên CPU. Đây là model CÓ THẬT trên Hub.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "distilgpt2" # a real, small public model on the Hub

# Download weights + tokenizer the first time; cached afterwards
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval() # inference mode: no dropout, no gradient tracking needed

prompt = "In a distant galaxy, a small robot learned to"
# Encode text -> token ids. inputs['input_ids'] shape: (1, seq_len)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad(): # no autograd during generation -> saves memory
    output_ids = model.generate(
        **inputs,
        max_new_tokens=30, # how many NEW tokens to produce
        do_sample=True,    # sample instead of greedy
        temperature=0.8,   # softens/sharpens the distribution
        top_k=50,          # keep only top-50 tokens each step
    )

# output_ids shape: (1, seq_len + max_new_tokens)
text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)

# Sample output (varies each run because do_sample=True):
# In a distant galaxy, a small robot learned to speak with the stars,
# and it began to dream of a world beyond the machines...
```

Luồng dữ liệu giống hệt cái bạn đã tự viết ở khóa này - chỉ khác là weights nay đến từ HF Hub:

```text
text -> tokenizer.encode -> input_ids (1, seq_len)
            |
            v
  model.generate (forward nhiều lần, mỗi lần đoán token kế)
            |
            v
  output_ids (1, seq_len + N) -> tokenizer.decode -> text sinh ra
```

Nếu chỉ muốn nhanh gọn, `pipeline` gói cả tokenizer + model + decode vào một hàm:

```python
from transformers import pipeline

# pipeline bundles tokenizer + model + decoding into one callable
gen = pipeline("text-generation", model="distilgpt2")

result = gen(
    "The secret to good code is",
    max_new_tokens=25,
    do_sample=True,
    temperature=0.7,
)
print(result[0]["generated_text"]) # result is a list of dicts, each with key "generated_text"
```

🎯 Chú ý `max_new_tokens` (số token MỚI sinh thêm) khác `max_length` (tổng độ dài kể cả prompt). Dùng `max_new_tokens` để tránh nhầm lẫn khi prompt dài ngắn khác nhau.

---

### 4. So sánh model pretrained với GPT mini bạn tự train

Bây giờ hãy đặt cạnh nhau. GPT mini của bạn (giả sử vài triệu tham số) và GPT-2 124M do OpenAI train. Ước lượng VRAM thô cho inference ở fp16: mỗi tham số ~2 byte (nối lại cách ước lượng ở _Bài 2_).

| Tiêu chí                  | GPT mini (bạn tự train)     | GPT-2 124M (pretrained)          |
| :------------------------ | :-------------------------- | :------------------------------- |
| **Số tham số**            | ~vài triệu (đếm như Bài 24) | ~124 triệu                       |
| **Dữ liệu train**         | nhỏ, do bạn chuẩn bị        | ~40GB WebText (OpenAI)           |
| **Chất lượng văn bản**    | học nguyên lý, còn ngô nghê | mạch lạc hơn nhiều               |
| **VRAM inference (fp16)** | vài chục MB                 | ~124M × 2 byte ≈ 0.25 GB weights |
| **Chi phí bạn bỏ ra**     | công train của bạn          | 0 (tải về miễn phí)              |
| **Mục đích**              | HỌC cách build              | DÙNG được ngay                   |

⚠️ Con số VRAM trên chỉ tính riêng weights. Khi chạy thực tế còn cộng thêm bộ nhớ cho activation, KV-cache và overhead của framework - luôn để dư một khoảng an toàn (xem lại Bài 2).

---

### 5. Cách SERVE model để tự host thành dịch vụ

Chạy đoạn Python ở mục 3 là inference một lần. Để _tự host thành một dịch vụ_ mà app khác gọi được, có 3 hướng phổ biến:

**(a) `transformers` trực tiếp** - bạn tự viết một web server (ví dụ FastAPI, giống Bài 35), nạp model một lần rồi phục vụ nhiều request. Linh hoạt nhất, hợp khi bạn cần kiểm soát hoàn toàn hoặc đang thử nghiệm.

**(b) Ollama** - công cụ chạy LLM cục bộ cực đơn giản. Bạn chỉ cần một lệnh:

```bash
# Ollama: run a model locally with one command
ollama run llama3.2
```

Ollama tự lo tải model, quản lý bộ nhớ và cung cấp một API cục bộ. Rất hợp để chạy thử trên máy cá nhân, không cần viết code phục vụ.

**(c) vLLM** - một máy chủ inference tối ưu cho hiệu năng cao (batching thông minh, tận dụng KV-cache). vLLM cung cấp một API _tương thích chuẩn OpenAI_, nên các client viết cho OpenAI có thể trỏ sang server vLLM của bạn gần như không đổi code. Hợp cho môi trường production, nhiều request đồng thời.

```text
KHI NÀO DÙNG GÌ?

  transformers trực tiếp -> thử nghiệm, tùy biến sâu, học tập
  Ollama                 -> chạy nhanh cục bộ trên máy cá nhân, ít cấu hình
  vLLM                   -> production, throughput cao, API kiểu OpenAI
```

💡 Cả ba đều “tự host”: model chạy trên máy/hạ tầng của bạn, dữ liệu không rời khỏi hệ thống - khác với gọi API của bên thứ ba.

---

### 6. Khi nào nên TỰ TRAIN vs DÙNG LẠI model pretrained?

Đây là quyết định quan trọng. Checklist đơn giản:

| Tình huống                                              | Nên chọn                         |
| :------------------------------------------------------ | :------------------------------- |
| Cần một trợ lý ngôn ngữ chung chung, tiếng phổ thông    | DÙNG LẠI pretrained              |
| Học nguyên lý, làm đồ án, hiểu bên trong Transformer    | TỰ TRAIN (như khóa này)          |
| Dữ liệu/ngôn ngữ rất đặc thù, riêng tư, không sẵn model | Cân nhắc fine-tune từ pretrained |
| Ngân sách nhỏ, ít GPU, cần kết quả nhanh                | DÙNG LẠI pretrained              |
| Cần kiểm soát tuyệt đối dữ liệu train từ đầu            | TỰ TRAIN (đắt)                   |

🎯 Nguyên tắc thực dụng: mặc định hãy DÙNG LẠI model pretrained; chỉ tự train từ đầu khi có lý do rõ ràng (học tập, dữ liệu quá đặc thù, yêu cầu kiểm soát toàn bộ). Đứng giữa hai thái cực là _fine-tuning_ - lấy pretrained rồi chỉnh nhẹ theo dữ liệu của bạn.

---

### 7. Lưu ý: LICENSE, VRAM và an toàn khi tự host

- _Đọc và tôn trọng LICENSE_: mỗi model có giấy phép riêng. Một số cho dùng thương mại, một số chỉ cho nghiên cứu, một số cấm vài mục đích. Luôn đọc trên model card TRƯỚC khi đưa vào sản phẩm.
- _VRAM đủ cho model_ (nối Bài 2): trước khi tải một model lớn, ước lượng bộ nhớ (số tham số × số byte/tham số + overhead). Model quá lớn so với VRAM sẽ báo lỗi hết bộ nhớ (OOM) hoặc chạy cực chậm.
- _An toàn/đạo đức_: model pretrained có thể sinh nội dung sai lệch, thiên kiến hoặc độc hại. Khi tự host và mở cho người khác dùng, hãy có lớp kiểm duyệt/giới hạn phù hợp và minh bạch về giới hạn của model.

---

### Nhìn sâu hơn

Một điểm hay: `AutoModelForCausalLM` và `AutoTokenizer` là các lớp “tự động” - chúng đọc file cấu hình đi kèm model để tự chọn ĐÚNG kiến trúc (GPT-2, LLaMA, v.v.). Nhờ vậy bạn đổi tên model là chạy được kiến trúc khác mà gần như không đổi code. “Causal” ở đây nghĩa là causal mask (mặt nạ tam giác dưới) mà bạn đã học ở phần attention - model chỉ được nhìn về quá khứ khi đoán token kế.

Về bộ nhớ: lần đầu `from_pretrained` sẽ tải file weights về thư mục cache (thường trong `~/.cache/huggingface`), lần sau dùng lại không tải nữa. Với model lớn, có thể nạp ở fp16/bfloat16 (`torch_dtype=torch.float16`) để giảm một nửa VRAM so với fp32 - đúng như cách ước lượng ở Bài 2. Đây cũng là cửa ngõ dẫn tới _quantization_ (nén xuống 8-bit/4-bit) mà khóa chuyên sâu sẽ đào sâu.

⚠️ Gotcha thường gặp: quên `model.eval()` và `torch.no_grad()` khi inference - vừa tốn bộ nhớ (theo dõi gradient vô ích) vừa có thể sai hành vi (dropout còn bật). Luôn bật hai thứ này khi chỉ sinh văn bản.

---

### Thực hành

- **[Dễ]** Chạy đúng đoạn code mục 3 với `distilgpt2`. Đổi prompt sang một câu tiếng Anh khác và quan sát văn bản sinh ra. Ghi lại `output_ids.shape` và giải thích tại sao chiều thứ hai lớn hơn `input_ids.shape`.
- **[Trung bình]** Thí nghiệm siêu tham số: chạy `model.generate` với `temperature` lần lượt 0.2, 0.8, 1.5 (giữ nguyên prompt và seed). Mô tả sự khác nhau về độ “an toàn/sáng tạo” của văn bản. Sau đó thử đặt `do_sample=False` (greedy) và so sánh.
- **[Thử thách]** Dựa trên Bài 35, viết một endpoint FastAPI `/generate` nhận `{"prompt": "..."}`, nạp `distilgpt2` MỘT LẦN lúc khởi động (không nạp lại mỗi request), trả về văn bản sinh ra. Đo thời gian phản hồi và giải thích vì sao nạp model ngoài hàm xử lý request lại quan trọng.

---

### Hiểu lầm thường gặp

- ❌ “Dùng model pretrained tức là đang fine-tune nó.” ➔ ✅ Không. Tải weights về chạy nguyên trạng là INFERENCE. Fine-tuning là CHỈNH weights theo dữ liệu của bạn - một bước khác hẳn.
- ❌ “Tên model gõ sao cũng được.” ➔ ✅ Phải đúng tên trên Hub (ví dụ `gpt2`, `distilgpt2`). Gõ sai tên sẽ báo lỗi không tìm thấy model.
- ❌ “Model mở thì muốn dùng vào việc gì cũng được.” ➔ ✅ Phải đọc LICENSE. Một số model giới hạn mục đích thương mại hoặc cấm vài trường hợp dùng.
- ❌ “Ollama, vLLM và transformers là ba thứ cạnh tranh, chỉ chọn một cái đúng.” ➔ ✅ Chúng phục vụ mục đích khác nhau: transformers để tùy biến/học, Ollama để chạy cục bộ nhanh, vLLM cho production throughput cao.

---

### Mẹo ghi nhớ

- _Auto = tự nhận kiến trúc_: `AutoModelForCausalLM` + `AutoTokenizer` đọc config để chọn đúng lớp.
- _max_new_tokens_ kiểm soát số token SINH THÊM, không tính prompt.
- _eval() + no_grad()_ khi inference: tiết kiệm bộ nhớ, tắt dropout.
- Mẹo nhớ 3 cách serve: _“Transformers để Tùy biến, Ollama để Ổn (nhanh gọn), vLLM để Vận hành production”_.

---

### Tóm tắt trực quan

| Khái niệm                | Một dòng                                                           |
| :----------------------- | :----------------------------------------------------------------- |
| **Model pretrained**     | weights có sẵn của người khác, tải về là chạy                      |
| **Hugging Face Hub**     | kho model mở, mỗi model có model card + license                    |
| **AutoModelForCausalLM** | tự nạp đúng kiến trúc model sinh văn bản                           |
| **AutoTokenizer**        | tự nạp tokenizer khớp với model                                    |
| **model.generate**       | sinh token kế tiếp (dùng `max_new_tokens`, `temperature`, `top_k`) |
| **Ollama**               | chạy LLM cục bộ một lệnh, ít cấu hình                              |
| **vLLM**                 | server inference tối ưu, API kiểu OpenAI, cho production           |
| **License**              | giấy phép quyết định bạn được dùng model vào việc gì               |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Run LLMs Locally for free with Hugging Face Transformers](https://www.youtube.com/watch?v=Xsn5hV2b7qQ)
- Tài liệu transformers (Hugging Face): [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- Model card GPT-2 trên Hub: [https://huggingface.co/gpt2](https://huggingface.co/gpt2)
- Ollama: [https://ollama.com](https://ollama.com)
- vLLM docs: [https://docs.vllm.ai](https://docs.vllm.ai)
- PyTorch docs: [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)

---

### Tổng kết

- Dùng lại model pretrained giúp bạn có mô hình chất lượng cao mà không phải train từ đầu - tải weights từ Hugging Face Hub về là chạy.
- `AutoModelForCausalLM` + `AutoTokenizer` (hoặc `pipeline`) cho phép tải + inference một model thật như `distilgpt2` chỉ trong vài dòng.
- Ba cách serve: transformers trực tiếp (tùy biến), Ollama (cục bộ nhanh), vLLM (production, API kiểu OpenAI).
- Mặc định nên DÙNG LẠI pretrained; tự train từ đầu chỉ khi có lý do rõ ràng. Luôn đọc LICENSE và ước lượng VRAM (Bài 2).

Muốn đi xa hơn - fine-tune model pretrained cho tác vụ riêng, nén model bằng quantization, hay serve production với LoRA/Ollama/vLLM - hãy tham khảo khóa chuyên sâu có sẵn trên nền tảng: _“Fine-tuning & Tự host LLM (LoRA, Ollama, vLLM)”_. Đó là bước tiếp theo tự nhiên sau khi bạn đã nắm cách tải và chạy model pretrained.

Ở _Bài 37: Scaling laws - mô hình lớn hơn, dữ liệu nhiều hơn_, ta sẽ trả lời một câu hỏi lớn: nếu tăng số tham số và dữ liệu thì chất lượng model cải thiện theo quy luật nào? Đây là chìa khóa để hiểu vì sao các model như GPT-2, GPT-3 ngày càng khổng lồ. Hẹn gặp bạn ở bài kế tiếp! 🤖
