# Bài 5: Biểu diễn văn bản - token, vocabulary và character-level tokenizer

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 1:** Môi trường, phần cứng & nền tảng

---

### Giới thiệu

Ở _Bài 4_, bạn đã hiểu bản chất của mô hình ngôn ngữ: chơi trò “đoán token tiếp theo” dựa trên phân phối xác suất. Nhưng mạng nơ-ron chỉ tính toán được trên ma trận số thực; nó hoàn toàn không hiểu chữ cái, từ ngữ hay câu văn.

Để đưa văn bản vào mô hình, ta cần một cầu nối chuyển đổi giữa ngôn ngữ tự nhiên và các con số. Chiếc cầu nối đó gọi là **Tokenizer**. Bài học này sẽ giúp bạn hiểu cặn kẽ cách biến văn bản thành các đơn vị cơ bản (**tokens**), xây dựng bảng tra cứu từ vựng (**vocabulary**), và tự tay hiện thực một bộ **Character-level Tokenizer** hoàn chỉnh bằng Python và PyTorch.

> 🔑 **Một câu để nhớ:** Tokenizer là cổng vào và cổng ra của LLM: nó mã hóa (encode) chuỗi ký tự thành các số nguyên `token_id`, và giải mã (decode) các con số dự đoán ngược lại thành văn bản con người đọc được.

---

### Bạn sẽ học được gì?

- [ ] Phân biệt được 3 cấp độ tokenization: **Character-level**, **Subword-level (BPE)**, và **Word-level**.
- [ ] Hiểu rõ cấu trúc và vai trò của **Vocabulary** (tập từ vựng) cùng hai phép ánh xạ `stoi` và `itos`.
- [ ] Tự tay code lớp `CharTokenizer` hỗ trợ hai hàm cốt lõi: `encode()` và `decode()`.
- [ ] Chuyển đổi chuỗi token ID thành **PyTorch Tensor** sẵn sàng đưa vào mô hình.
- [ ] Nắm được ưu và nhược điểm khi dùng tokenizer mức ký tự cho mô hình GPT mini.

📌 **Yêu cầu trước:** Đã hoàn thành Bài 1-4 (cơ bản về Python dictionary, chuỗi string và `torch.tensor`).

---

### 1. Token và Tokenization là gì?

- **Khái niệm:** **Token** là đơn vị văn bản nhỏ nhất mà mô hình ngôn ngữ xử lý độc lập. Quá trình chia nhỏ văn bản thô thành danh sách các token được gọi là **Tokenization**.
- **Token ID:** Mỗi token duy nhất trong tập từ vựng sẽ được gán một số nguyên không âm $0, 1, 2, \dots, V - 1$ (với $V$ là kích thước từ vựng `vocab_size`).

Có 3 trường phái chia token chính:

| Cấp độ                  | Ví dụ với từ "learning"                    | Kích thước Vocabulary ($V$) | Ưu điểm                                     | Nhược điểm                                               |
| :---------------------- | :----------------------------------------- | :-------------------------- | :------------------------------------------ | :------------------------------------------------------- |
| **Word-level**          | `["learning"]`                             | Rất lớn ($10^5 - 10^6$)     | Giữ trọn vẹn ngữ nghĩa của từ               | Rất dễ gặp từ lạ ngoài từ điển (Out-of-Vocabulary - OOV) |
| **Character-level**     | `['l', 'e', 'a', 'r', 'n', 'i', 'n', 'g']` | Rất nhỏ ($50 - 256$)        | Không bao giờ bị lỗi OOV; cực kỳ đơn giản   | Chuỗi token bị dài ra, mô hình tốn ngữ cảnh để học       |
| **Subword-level (BPE)** | `["learn", "ing"]`                         | Vừa phải ($32k - 100k$)     | Cân bằng hoàn hảo độ dài chuỗi và ngữ nghĩa | Thuật toán xây dựng từ vựng phức tạp hơn                 |

💡 Với mục tiêu học bản chất kiến trúc ở Tuần 1 và dựng mô hình nanoGPT mini, **Character-level Tokenizer** là lựa chọn lý tưởng vì nó trong sáng, không phụ thuộc thư viện ngoài và giúp bạn kiểm soát 100% dữ liệu.

---

### 2. Xây dựng Vocabulary: stoi và itos

Để biến đổi qua lại giữa ký tự và số nguyên, ta chỉ cần hai bảng ánh xạ:

1. `stoi` (**s**tring **to** **i**nteger): Tra từ ký tự ra chỉ số ID để đưa vào model.
2. `itos` (**i**nteger **to** **s**tring): Tra từ ID ra lại ký tự khi sinh văn bản.

```python
# Giả sử tập văn bản mẫu dùng để huấn luyện
corpus = "hello world! xin chao the gioi"

# 1. Trích xuất toàn bộ ký tự phân biệt và sắp xếp có thứ tự
unique_chars = sorted(list(set(corpus)))
vocab_size = len(unique_chars)

print("Các ký tự:", unique_chars)
print("Kích thước vocab (V):", vocab_size)

# 2. Xây dựng hai từ điển tra cứu
stoi = {ch: i for i, ch in enumerate(unique_chars)}
itos = {i: ch for i, ch in enumerate(unique_chars)}

# Thử nghiệm tra cứu
sample_char = 'h'
sample_id = stoi[sample_char]
print(f"Ký tự '{sample_char}' -> ID: {sample_id} -> Ký tự: '{itos[sample_id]}'")
```

---

### 3. Tự viết lớp CharacterTokenizer hoàn chỉnh

Chúng ta đóng gói toàn bộ logic vào một class Python tái sử dụng được:

```python
import torch

class CharTokenizer:
    def __init__(self, text: str):
        # Trích xuất tập từ vựng từ văn bản mẫu
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Tạo bảng tra cứu ánh xạ 2 chiều
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        """Chuyển chuỗi văn bản thành danh sách token ID."""
        return [self.stoi[c] for c in s]

    def decode(self, indices: list[int]) -> str:
        """Chuyển danh sách token ID trở lại thành văn bản."""
        return ''.join([self.itos[i] for i in indices])

    def encode_to_tensor(self, s: str, device: str = 'cpu') -> torch.Tensor:
        """Mã hóa văn bản thành PyTorch Tensor kiểu int64 (torch.long)."""
        token_ids = self.encode(s)
        return torch.tensor(token_ids, dtype=torch.long, device=device)
```

Kiểm tra hoạt động thực tế của Tokenizer:

```python
# Khởi tạo dữ liệu mẫu
raw_text = "PyTorch cho LLM"
tokenizer = CharTokenizer(raw_text)

# Thử nghiệm Encode
text_to_test = "cho LLM"
encoded_ids = tokenizer.encode(text_to_test)
print("Văn bản gốc:", text_to_test)
print("Danh sách Token ID:", encoded_ids)

# Thử nghiệm Tensor
tensor_ids = tokenizer.encode_to_tensor(text_to_test)
print("Tensor đầu ra:", tensor_ids)
print("Shape của Tensor:", tensor_ids.shape) # torch.Size([7])

# Thử nghiệm Decode
decoded_text = tokenizer.decode(encoded_ids)
print("Sau khi giải mã:", decoded_text)
assert decoded_text == text_to_test, "Chuỗi giải mã phải trùng khớp 100% với chuỗi gốc!"
```

---

### 4. Sơ đồ xử lý dữ liệu đầu vào của mô hình

Khi bạn đưa một câu văn vào mô hình GPT, luồng biến đổi dữ liệu diễn ra như sau:

```text
Văn bản thô (String):
   "cat"
     |
     v  [Tokenizer.encode]
Danh sách ID:
   [2, 0, 19]
     |
     v  [torch.tensor]
Tensor chỉ số (1D Tensor):
   tensor([2, 0, 19])               shape: (sequence_length,) = (3,)
     |
     v  [nn.Embedding lookup table]
Ma trận Embedding đặc:
   [[ 0.12, -0.45,  0.89, ...],     shape: (sequence_length, d_model)
    [-0.03,  0.72, -0.11, ...],
    [ 0.65,  0.31, -0.24, ...]]
```

💡 Chú ý rằng các ID số nguyên (`0, 1, 2...`) chỉ mang tính chất định danh (index danh mục), không hề có tính chất lớn bé hay khoảng cách số học. Lớp **Embedding** ở tuần sau sẽ chịu trách nhiệm biến các ID này thành các vector ngữ nghĩa.

---

### Nhìn sâu hơn

- **Chi phí ngữ cảnh của Character-level:** Vì mỗi token chỉ là một ký tự, một từ tiếng Anh/Việt trung bình mất từ 4 - 6 token. Điều này khiến cửa sổ ngữ cảnh (Context Window) bị đầy rất nhanh so với khi dùng BPE.
- **Lỗi Out-of-Vocabulary (OOV):** Nếu bạn gặp một ký tự chưa từng có trong lúc khởi tạo `stoi`, hàm `encode()` sẽ ném lỗi `KeyError`. Ở các tokenizer thực tế, người ta thường dành riêng một token đặc biệt gọi là `<unk>` (Unknown) để gán cho các ký tự lạ.
- **Kiểu dữ liệu của Token Tensor:** PyTorch luôn yêu cầu các tensor dùng làm chỉ mục tra cứu embedding hoặc nhãn target của `nn.CrossEntropyLoss` phải là số nguyên 64-bit: `torch.long` (hoặc `torch.int64`), tuyệt đối không dùng `torch.float32`.

---

### Thực hành

- **[Dễ]** Hãy khởi tạo `CharTokenizer` từ chuỗi `"Xay dung GPT mini tu dau"`. In ra `vocab_size` và kiểm tra xem ký tự dấu cách (`' '`) có được gán ID hay không.
- **[Trung bình]** Nâng cấp hàm `encode()` của class `CharTokenizer` để nếu gặp một ký tự không có trong `stoi`, chương trình không bị crash mà tự động bỏ qua ký tự đó hoặc thay thế bằng ký tự đầu tiên trong bảng từ vựng.
- **[Thử thách]** Tải một file văn bản nhỏ (ví dụ đoạn thơ hoặc file `input.txt` trong kho lưu trữ `nanoGPT`). Đọc file, khởi tạo tokenizer, mã hóa toàn bộ file thành một tensor duy nhất, sau đó chia thành hai tập `train_data` ($90\%$) và `val_data` ($10\%$).

---

### Hiểu lầm thường gặp

- ❌ _“Giá trị số của token_id càng lớn thì token đó càng quan trọng.”_ ➔ ✅ `token_id` chỉ đơn thuần là số thứ tự trong bảng tra cứu (index của hàng trong ma trận Embedding), không mang ý nghĩa độ lớn toán học.
- ❌ _“Tokenizer là một phần bên trong mạng nơ-ron và được cập nhật gradient.”_ ➔ ✅ Tokenizer là thuật toán tiền xử lý chạy bằng CPU thuần, đứng độc lập bên ngoài và không có tham số hay gradient nào để huấn luyện bằng `backward()`.
- ❌ _“Mọi mô hình LLM đều dùng Character-level Tokenizer.”_ ➔ ✅ Các LLM sản xuất thực tế (GPT-4, Llama, Claude) đều dùng thuật toán **Byte-Pair Encoding (BPE)** trên mức byte để cân bằng tối ưu giữa kích thước từ vựng và độ dài văn bản.

---

### Mẹo ghi nhớ

- **Bắt cặp kinh điển:** `stoi` là đường đi vào mô hình (`encode`), `itos` là đường đi ra thế giới con người (`decode`).
- **Ép kiểu nguyên:** Đưa token vào PyTorch luôn luôn đặt `dtype=torch.long`.
- **Nguyên lý bảo toàn:** `decode(encode("text")) == "text"`. Nếu không bằng nhau, tokenizer của bạn đã bị lỗi.

---

### Tóm tắt trực quan

| Thuật ngữ        | Ý nghĩa kỹ thuật                              | Dạng dữ liệu                      |
| :--------------- | :-------------------------------------------- | :-------------------------------- |
| **Token**        | Một phần tử ký tự/từ nhỏ nhất sau khi cắt nhỏ | `str`                             |
| **Token ID**     | Mã số nguyên đại diện cho token               | `int`                             |
| **Vocabulary**   | Danh sách tất cả token duy nhất được hỗ trợ   | `set` / `list`                    |
| **`stoi`**       | Bảng tra: Ký tự $\to$ Token ID                | `dict[str, int]`                  |
| **`itos`**       | Bảng tra: Token ID $\to$ Ký tự                | `dict[int, str]`                  |
| **Tensor Input** | Chuỗi ID đóng gói sẵn sàng cho model          | `torch.Tensor (dtype=torch.long)` |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Let's build GPT: from scratch, in code, spelled out - Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- nanoGPT repository: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- PyTorch Documentation - `torch.nn.Embedding`: [pytorch.org/docs/stable/generated/torch.nn.Embedding.html](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- Hugging Face Tokenizers Concept Guide: [huggingface.co/docs/transformers/tokenizer_summary](https://huggingface.co/docs/transformers/tokenizer_summary)

---

### Tổng kết

- Bạn đã hoàn thành toàn bộ **Tuần 1: Môi trường, phần cứng & nền tảng** với 5 bài học nền tảng.
- Máy tính không hiểu chữ, chỉ hiểu số $\to$ Tokenizer là thành phần chuyển hóa văn bản thành chuỗi các số nguyên `token_id`.
- Bộ đôi `stoi` và `itos` hình thành nên xương sống của bảng từ vựng `vocabulary`.
- Dữ liệu sau khi mã hóa được chuyển thành `torch.Tensor` kiểu `torch.long` để nạp vào mô hình.

Chúc mừng bạn đã hoàn thành Tuần 1! Ở **Tuần 2: Mô hình ngôn ngữ nơ-ron đầu tiên**, chúng ta sẽ chính thức bắt tay vào xây dựng lớp `Embedding`, cơ chế lấy mẫu dữ liệu theo Batch/Context window, và huấn luyện mô hình Bigram Neural Network đầu tiên! 🚀
