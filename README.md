# 🧠 Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch

> **"Không gì giúp bạn hiểu sâu một kiến trúc AI bằng việc tự tay viết lại nó từ những phép toán ma trận cơ bản nhất."**  
> Dự án được xây dựng dựa trên triết lý _nanoGPT_ và tư duy kỹ thuật thực chiến: bóc tách "chiếc hộp đen" của Large Language Models (LLMs) thành từng dòng mã PyTorch rõ ràng, dễ hiểu và tối ưu.

---

## 📌 1. Tổng quan dự án (Overview)

Trong làn sóng bùng nổ của Trí tuệ Nhân tạo Tạo sinh (Generative AI), hầu hết các lập trình viên tiếp cận LLM thông qua các API đóng gói sẵn (OpenAI, Anthropic) hoặc thư viện bậc cao (Hugging Face `transformers`). Cách tiếp cận này giúp tạo prototype nhanh nhưng để lại một khoảng trống lớn về kiến thức nền tảng:

- _Cơ chế Attention thực sự tính toán điều gì trên GPU?_
- _Ma trận Q, K, V tương tác như thế nào để mô hình nắm bắt được ngữ cảnh ngữ pháp và ý nghĩa?_
- _Tại sao mô hình lại sinh từ tuần tự (autoregressive) và kỹ thuật KV-cache tối ưu tốc độ ra sao?_
- _Làm thế nào để huấn luyện một mô hình ngôn ngữ ổn định mà không bị nổ gradient?_

Kho tài liệu này là một chương trình thực chiến toàn diện kéo dài **8 tuần gồm 41 phần học (Part 1 -> Part 41)**. Bạn sẽ đi từ bước chuẩn bị môi trường phần cứng, xử lý dữ liệu thô, thiết kế từng module Attention, lắp ghép thành Transformer Block, huấn luyện mô hình GPT mini trên tập dữ liệu thực tế, cho đến các kỹ thuật tối ưu inference (KV-cache, BPE tokenizer) và triển khai API tự host trên GPU cá nhân.

---

## 🎯 2. Mục đích cốt lõi (Purpose & Objectives)

Dự án được thiết kế nhằm đạt được các mục tiêu giáo dục và kỹ thuật trọng tâm:

1. **Xoá bỏ "hộp đen" LLM (De-blackboxing AI)**:
   - Thay vì gọi các hàm trừu tượng như `model.generate()` hay `AutoModelForCausalLM`, bạn sẽ tự tay cài đặt phép nhân `Q @ K.T`, causal mask, softmax, residual connection và LayerNorm.
2. **Làm chủ tư duy Tensor & PyTorch**:
   - Hiểu sâu về tensor shape `(B, T, C)`, cơ chế broadcasting, autograd, forward-backward pass và quản lý VRAM trên GPU.
3. **Hiểu rõ vòng đời trọn vẹn của một LLM**:
   - Từ dữ liệu văn bản thô $\rightarrow$ Tokenization $\rightarrow$ Embedding $\rightarrow$ Transformer blocks $\rightarrow$ Logits $\rightarrow$ Cross-entropy loss $\rightarrow$ Backpropagation $\rightarrow$ Autoregressive generation.
4. **Cầu nối giữa lý thuyết và kỹ thuật thực tế (Production Engineering)**:
   - Không chỉ dừng ở đồ chơi (toy example), dự án đưa vào các kỹ thuật hiện đại như Byte-Pair Encoding (BPE), Cosine Decay Learning Rate Scheduler, Gradient Clipping, KV-cache và REST API serving với FastAPI.

---

## 🏆 3. Kết quả đạt được (Key Outcomes & Deliverables)

Sau khi hoàn thành lộ trình, bạn sở hữu một sản phẩm hoàn chỉnh và năng lực kỹ thuật thực thụ:

### 🧩 Sản phẩm phần mềm

- **Mô hình GPT mini thuần PyTorch**: Tự thiết kế và cài đặt 100% không phụ thuộc vào lớp có sẵn của thư viện ngoài.
- **Hệ thống Tokenizer linh hoạt**:
  - Character-level tokenizer cho mục đích nghiên cứu và trực quan hóa.
  - Byte-Pair Encoding (BPE) chuẩn OpenAI/Hugging Face cho bài toán thực tế.
- **Training Pipeline hoàn chỉnh**:
  - Hỗ trợ Mixed Precision (`torch.cuda.amp`), AdamW optimizer, gradient clipping, evaluation loop định kỳ và checkpointing (lưu/tải trọng số).
- **Inference Engine tối ưu**:
  - Hỗ trợ các chiến lược sampling đa dạng: Temperature scaling, Top-$k$, Top-$p$ (Nucleus sampling).
  - Tích hợp **KV-Cache** giúp giảm độ phức tạp tính toán từ $O(T^2)$ xuống $O(T)$ trong quá trình sinh từ liên tục.
- **Microservice API sẵn sàng triển khai**:
  - FastAPI server đóng gói mô hình GPT mini, hỗ trợ streaming response / batch generation phục vụ web demo.

### 💡 Năng lực chuyên môn

- Đọc hiểu sâu sắc các bài báo khoa học nền tảng (_"Attention Is All You Need"_, _GPT-2/3_, _Chinchilla Scaling Laws_).
- Tự tin debug các lỗi phổ biến khi train mạng nơ-ron sâu: nổ/biến mất gradient, overfitting, rò rỉ context, OOM (Out Of Memory) GPU.
- Có khả năng đánh giá, so sánh và đưa ra quyết định khi áp dụng LLM mã nguồn mở (Llama, Mistral, Qwen) vào bài toán doanh nghiệp.

---

## 🗺️ 4. Cấu trúc lộ trình học tập (Curriculum Roadmap)

Toàn bộ nội dung được tổ chức thành 8 tuần học với 41 bài học chi tiết:

| Tuần       | Chủ đề chính                         | Các phần                                    | Nội dung trọng tâm                                                                                                                                                    |
| :--------- | :----------------------------------- | :------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tuần 1** | **Môi trường & Nền tảng**            | [Part 1](Part1.md) – [Part 5](Part5.md)     | Thiết lập GPU, kiến trúc phần cứng, PyTorch Tensor/Autograd/nn.Module, khái niệm LLM và Character-level Tokenizer.                                                    |
| **Tuần 2** | **Mô hình ngôn ngữ nơ-ron đầu tiên** | [Part 6](Part6.md) – [Part 10](Part10.md)   | Chuẩn bị dữ liệu chuỗi, Bigram Language Model, Cross-Entropy Loss, Text Sampling và cơ chế đánh giá Validation Loss.                                                  |
| **Tuần 3** | **Cơ chế Attention cốt lõi**         | [Part 11](Part11.md) – [Part 15](Part15.md) | Từ Context Averaging đến Self-Attention, cơ chế Query-Key-Value, Scaled Dot-Product, Causal Masking và hoàn thiện một Causal Attention Head.                          |
| **Tuần 4** | **Transformer Block hoàn chỉnh**     | [Part 16](Part16.md) – [Part 20](Part20.md) | Multi-Head Attention song song, Feed-Forward Network, Residual Connections, Pre-LayerNorm và Positional Embeddings.                                                   |
| **Tuần 5** | **Xây dựng kiến trúc GPT mini**      | [Part 21](Part21.md) – [Part 25](Part25.md) | Ghép nối các khối Transformer, định nghĩa lớp mô hình `GPTLanguageModel`, đếm tham số, kỹ thuật khởi tạo trọng số (weight initialization).                            |
| **Tuần 6** | **Huấn luyện mô hình GPT mini**      | [Part 26](Part26.md) – [Part 30](Part30.md) | Vòng lặp huấn luyện chuẩn, AdamW, Learning Rate Scheduler, chống Overfitting, huấn luyện trên tập Tiny Shakespeare và Checkpoint Management.                          |
| **Tuần 7** | **Tokenization thực tế & Tự host**   | [Part 31](Part31.md) – [Part 35](Part35.md) | Thuật toán Byte-Pair Encoding (BPE), thư viện `tiktoken`, kỹ thuật Temperature/Top-k/Top-p, tăng tốc với KV-Cache và viết API FastAPI host trên GPU.                  |
| **Tuần 8** | **Mở rộng, Tinh chỉnh & Đồ án cuối** | [Part 36](Part36.md) – [Part 41](Part41.md) | Khám phá Pretrained Models (Hugging Face), Định luật mở rộng (Scaling Laws), Pretraining vs Fine-tuning, RLHF/DPO, đạo đức AI và Đồ án tốt nghiệp (Capstone Project). |

---

## 🚀 5. Hướng dẫn bắt đầu nhanh (Quickstart)

### 5.1. Yêu cầu môi trường

- **Python**: $\ge 3.10$
- **PyTorch**: $\ge 2.0$ (Khuyến nghị có card đồ họa NVIDIA với CUDA hoặc Apple Silicon với MPS)

```bash
# Cài đặt các thư viện cần thiết
pip install torch torchvision torchaudio
pip install tiktoken fastapi uvicorn matplotlib tqdm
```

### 5.2. Chạy thử nghiệm

Mỗi bài học được thiết kế độc lập kèm theo code mẫu hoàn chỉnh. Bạn có thể mở trực tiếp các file từ `Part1.md` đến `Part41.md` để học lý thuyết và chạy code thực hành theo từng bước.

---

## 📚 6. Tài liệu tham khảo nổi bật

- **Attention Is All You Need** (Vaswani et al., 2017)
- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019 - GPT-2)
- **nanoGPT** by Andrej Karpathy ([GitHub Repository](https://github.com/karpathy/nanoGPT))
- **Hugging Face Transformers Documentation & Courses**

---

_Chúc bạn có một hành trình khám phá và làm chủ công nghệ lõi của LLM thật nhiều cảm hứng và thu hoạch giá trị!_
