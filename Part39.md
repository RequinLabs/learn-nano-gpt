# Bài 39: Instruction tuning và RLHF - cách ChatGPT/Claude được căn chỉnh

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 8:** Từ pretraining đến trợ lý & dự án cuối

---

### Giới thiệu

Ở _Bài 37_ và _Bài 38_, bạn đã thấy bức tranh tổng thể: _pretraining_ dạy mô hình “nối chữ” trên hàng núi văn bản (dự đoán token tiếp theo bằng cross-entropy), còn _fine-tuning_ thì tinh chỉnh mô hình cho một việc cụ thể. Nhưng có một điều lạ: một mô hình pretrained thuần túy (như GPT mini của chúng ta, chỉ được huấn luyện next-token) khi bạn hỏi “Thủ đô của Pháp là gì?” thường KHÔNG trả lời “Paris” - nó có thể… viết tiếp thành một bài kiểm tra địa lý, hoặc lặp lại câu hỏi. Vì sao vậy?

Hãy tưởng tượng bạn có một người bạn đọc ngấu nghiến _cả thư viện_: người đó thuộc lòng cách các câu văn nối tiếp nhau, nhưng chưa ai dạy họ “khi có người HỎI thì phải TRẢ LỜI”. Họ chỉ biết “đoán chữ tiếp theo” theo thói quen của văn bản đã đọc. _Instruction tuning_ chính là bước dạy người đó: “Đây là một mệnh lệnh, và đây là cách phản hồi hữu ích.” Còn _RLHF_ (Reinforcement Learning from Human Feedback) là bước tinh tế hơn: dạy họ trả lời sao cho con người _thích hơn_ - hữu ích, trung thực và an toàn.

Đây chính là bí mật biến một mô hình “nối chữ” khô khan thành trợ lý như ChatGPT hay Claude.

> 🔑 **Một câu để nhớ:** Pretraining dạy mô hình _biết ngôn ngữ_; instruction tuning dạy nó _nghe lệnh_; RLHF dạy nó _cư xử theo sở thích con người_ (hữu ích, trung thực, an toàn).

---

### Bạn sẽ học được gì?

- [ ] Hiểu vì sao mô hình pretrained thuần túy không phải là một trợ lý biết nghe lệnh
- [ ] Nắm cách _instruction tuning (SFT)_ hoạt động: định dạng dữ liệu prompt-response và cách tính loss
- [ ] Hiểu _RLHF_ ở mức khái niệm: reward model + tối ưu chính sách (PPO) và phương án đơn giản hơn _DPO_
- [ ] Phân biệt rõ ba giai đoạn: _pretraining -> SFT -> alignment (RLHF/DPO)_
- [ ] Biết ba trụ cột “HHH”: _Helpful, Honest, Harmless_ và vì sao Claude/ChatGPT an toàn hơn

📌 **Yêu cầu trước:** đã hiểu next-token prediction + cross-entropy (Bài 37-38), biết cấu trúc GPT (embedding -> Transformer block -> logits) và cách generate với temperature/top-k.

---

### 1. Vì sao "nối chữ" chưa đủ để thành trợ lý?

**Analogy:** một ca sĩ hát karaoke thuộc cả nghìn bài, nhưng nếu bạn nói “cho tôi xin ly nước”, họ… hát tiếp câu tiếp theo trong bài đang dở. Đó là mô hình pretrained: nó tối đa hóa $P(\text{token tiếp theo} \mid \text{các token trước})$ theo _phân phối của Internet_, không phải theo ý định trả lời bạn.

Mục tiêu pretraining chính xác là cực tiểu hóa cross-entropy của next-token:

$$\mathcal{L}_{\text{pretrain}} = -\frac{1}{N} \sum_{t=1}^{N} \log P(\text{token}_t \mid \text{token}_1, \dots, \text{token}_{t-1})$$

Nó học _thống kê ngôn ngữ_, không học _hành vi trợ lý_. Ta cần thêm một bước dạy “khi thấy mệnh lệnh thì tạo phản hồi hữu ích”.

```text
PRETRAINING              INSTRUCTION TUNING (SFT)          RLHF / DPO
   +--------------------+     +-----------------------------+   +------------------+
   | "nối chữ" trên      | --> | prompt -> response tốt       |-->| tối ưu theo sở   |
   | Internet (next-tok) |     | (do người viết mẫu)          |   | thích con người  |
   +--------------------+     +-----------------------------+   +------------------+
   biết NGÔN NGỮ               biết NGHE LỆNH                    biết CƯ XỬ ĐÚNG
```

💡 Ba giai đoạn dùng CHUNG một kiến trúc GPT (không đổi cấu trúc). Chỉ thay _dữ liệu_ và _mục tiêu huấn luyện_.

---

### 2. Instruction tuning (SFT) - dạy mô hình nghe lệnh

**Analogy:** giống như đào tạo nhân viên mới bằng “kịch bản mẫu”: mỗi tình huống (khách hỏi gì) đi kèm câu trả lời chuẩn. Nhân viên bắt chước cho tới khi thành phản xạ.

_SFT (Supervised Fine-Tuning)_ = fine-tuning có giám sát trên bộ dữ liệu gồm cặp _(instruction, response)_ do con người viết mẫu. Bản chất vẫn là _next-token prediction_ như pretraining, nhưng dữ liệu bây giờ có định dạng hội thoại rõ ràng.

Một mẫu dữ liệu điển hình (định dạng chat):

```text
<|user|> Thủ đô của Pháp là gì?
<|assistant|> Thủ đô của Pháp là Paris.
```

_Điểm mấu chốt:_ khi tính loss, ta thường _chỉ tính cross-entropy trên phần response_ (phần assistant), còn phần prompt bị _mask_ (đặt loss = 0). Vì ta muốn mô hình học _cách trả lời_, không phải học _cách sinh ra câu hỏi_.

```python
import torch
import torch.nn.functional as F

# Toy example: 1 sequence, 8 tokens.
# Positions 0..3 = prompt (mask), positions 4..7 = response (train).
# logits shape: (batch, seq_len, vocab_size)
vocab_size = 50
logits = torch.randn(1, 8, vocab_size) # (1, 8, 50)
targets = torch.randint(0, vocab_size, (1, 8)) # (1, 8) next-token ids

# loss_mask = 0 on prompt tokens, 1 on response tokens
loss_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.float32) # (1, 8)

# Flatten for cross_entropy: (N, vocab) and (N,)
logits_flat = logits.view(-1, vocab_size) # (8, 50)
targets_flat = targets.view(-1)           # (8,)

# per-token loss, then apply the mask (prompt tokens contribute 0)
per_tok = F.cross_entropy(logits_flat, targets_flat, reduction='none') # (8,)
mask_flat = loss_mask.view(-1) # (8,)
loss = (per_tok * mask_flat).sum() / mask_flat.sum() # scalar, averaged over RESPONSE tokens only
print(loss.item()) # e.g. ~4.1 (random init) -> giảm dần khi train
```

Sau SFT, mô hình đã “biết nghe lệnh”: hỏi thủ đô Pháp thì nó trả lời Paris, thay vì viết tiếp một đề thi. Nhưng SFT vẫn có giới hạn: nó chỉ _bắt chước_ các câu trả lời mẫu, chưa biết cách _so sánh_ câu trả lời nào TỐT HƠN khi có nhiều lựa chọn hợp lý.

⚠️ SFT khác pretraining ở _dữ liệu + loss mask_, KHÔNG khác ở kiến trúc. Vẫn là cross-entropy next-token, chỉ mask phần prompt.

---

### 3. RLHF - căn chỉnh theo sở thích con người

**Analogy:** dạy một đầu bếp không chỉ bằng công thức (SFT), mà bằng cách để thực khách _nếm hai món và chọn món ngon hơn_. Đầu bếp dần điều chỉnh để nấu ra món mà mọi người _thích hơn_, dù không có công thức tuyệt đối.

Vấn đề: “câu trả lời tốt” rất khó viết thành một hàm toán học. Nhưng con người lại _dễ so sánh_: cho hai câu trả lời A và B, người ta chỉ ra cái nào tốt hơn. RLHF khai thác đúng điều đó, qua _hai bước_:

#### Bước 1: Huấn luyện Reward Model (mô hình phần thưởng)

Thu thập dữ liệu _so sánh sở thích_: với cùng một prompt, con người xếp hạng “response A tốt hơn response B”. Một reward model $r(\text{prompt}, \text{response})$ được huấn luyện để cho _điểm cao hơn_ cho câu người ta thích, dùng hàm mất mát dạng Bradley-Terry:

$$\mathcal{L}_{\text{RM}} = -\log \sigma\left(r(x, y_{\text{chosen}}) - r(x, y_{\text{rejected}})\right)$$

```python
import torch
import torch.nn.functional as F

# Reward model outputs a scalar score per (prompt, response).
# Human said: y_chosen is BETTER than y_rejected.
r_chosen   = torch.tensor([2.3, 1.1, 0.7])  # (batch,) scores for preferred answers
r_rejected = torch.tensor([0.5, 1.4, -0.2]) # (batch,) scores for rejected answers

# We want r_chosen > r_rejected. Bradley-Terry / pairwise logistic loss:
loss_rm = -F.logsigmoid(r_chosen - r_rejected).mean() # scalar
print(loss_rm.item()) # decreases as the gap (r_chosen - r_rejected) grows
```

#### Bước 2: Tối ưu chính sách bằng PPO

Dùng reward model làm “giám khảo”, ta tinh chỉnh mô hình ngôn ngữ (gọi là _policy_) bằng học tăng cường (thuật toán _PPO_ - Proximal Policy Optimization): sinh câu trả lời, chấm điểm bằng reward model, cập nhật để tăng điểm. Có thêm một số hạng _phạt KL-divergence_ để mô hình không đi quá xa mô hình SFT gốc (tránh “hack” giám khảo bằng văn bản vô nghĩa nhưng điểm cao):

$$\text{Objective}_{\text{PPO}} = \mathbb{E}[r(x, y)] - \beta \cdot D_{\text{KL}}(\pi(y \mid x) \parallel \pi_{\text{SFT}}(y \mid x))$$

```text
Objective_PPO = E[ r(x, y) ]  -  beta * KL( policy(y|x) || policy_SFT(y|x) )
                (tăng thưởng)         (phạt nếu lệch quá xa bản gốc SFT)
```

🎯 Số hạng KL rất quan trọng: nếu bỏ đi, mô hình dễ “reward hacking” - tạo văn bản kỳ dị làm reward model chấm điểm cao nhưng thực ra vô dụng với người dùng.

---

### 4. DPO - lối tắt gọn hơn của RLHF

RLHF-PPO mạnh nhưng _phức tạp và tốn kém_ (phải huấn luyện thêm reward model + vòng lặp RL không ổn định). Gần đây, _DPO (Direct Preference Optimization)_ trở nên phổ biến: nó tối ưu _trực tiếp_ trên dữ liệu so sánh sở thích, _bỏ qua_ reward model và RL riêng biệt, gói mọi thứ vào một hàm mất mát dạng phân loại.

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \left[\log \frac{\pi(y_{\text{chosen}} \mid x)}{\pi_{\text{ref}}(y_{\text{chosen}} \mid x)} - \log \frac{\pi(y_{\text{rejected}} \mid x)}{\pi_{\text{ref}}(y_{\text{rejected}} \mid x)}\right]\right)$$

| Tiêu chí                | RLHF (PPO)           | DPO                          |
| :---------------------- | :------------------- | :--------------------------- |
| **Reward model riêng**  | Cần                  | Không cần                    |
| **Vòng lặp RL**         | Có (khó ổn định)     | Không (chỉ như phân loại)    |
| **Độ phức tạp cài đặt** | Cao                  | Thấp hơn nhiều               |
| **Dữ liệu**             | Cặp so sánh sở thích | Cặp so sánh sở thích (giống) |

💡 Với người tự build LLM nhỏ, _SFT + DPO_ là con đường thực tế nhất để có một trợ lý biết nghe lệnh mà không cần hạ tầng RL phức tạp.

---

### 5. Ba trụ cột HHH - vì sao Claude/ChatGPT hữu ích và an toàn

Căn chỉnh (alignment) không chỉ là “trả lời hay”, mà nhắm tới ba mục tiêu thường gọi là _HHH_:

| Trụ cột                 | Ý nghĩa                                    | Ví dụ                          |
| :---------------------- | :----------------------------------------- | :----------------------------- |
| **Helpful** (hữu ích)   | Hiểu đúng ý định, trả lời trọng tâm        | Tóm tắt đúng điều được hỏi     |
| **Honest** (trung thực) | Không bịa; nói “không chắc” khi không biết | Thừa nhận giới hạn kiến thức   |
| **Harmless** (an toàn)  | Từ chối yêu cầu gây hại                    | Không hướng dẫn chế tạo vũ khí |

Instruction tuning + RLHF/DPO chính là cơ chế “khắc” ba giá trị này vào mô hình: dữ liệu mẫu và so sánh sở thích đều được thiết kế để ưu tiên câu trả lời hữu ích, trung thực, an toàn. Đó là lý do Claude/ChatGPT khác hẳn một mô hình pretrained thô.

---

### Nhìn sâu hơn

- **Alignment tax:** đôi khi mô hình sau alignment kém hơn một chút ở vài benchmark thuần “kiến thức” so với mô hình base, đổi lại nó _hữu ích và an toàn hơn nhiều_ trong thực tế. Đây là đánh đổi có chủ đích.
- **Chất lượng dữ liệu > số lượng:** nghiên cứu như LIMA cho thấy chỉ _~1.000 mẫu SFT chất lượng cao_ đã có thể tạo trợ lý khá tốt - vì “kiến thức” đã nằm sẵn trong pretraining, SFT chỉ _mở khóa hành vi_.
- **Reward hacking:** không có KL-penalty, policy sẽ tìm cách “lừa” reward model. KL neo mô hình vào bản SFT gốc.
- **RLHF không sửa được kiến thức sai:** nếu pretraining chưa từng thấy một sự thật, alignment không tự sinh ra nó. Alignment định hình _hành vi_, không bơm thêm _kiến thức_.
- **Constitutional AI:** hướng của Anthropic dùng một bộ “nguyên tắc” để mô hình tự phê bình và sửa câu trả lời, giảm phụ thuộc vào nhãn người ở khâu an toàn.

---

### Thực hành

- **[Dễ]** Viết 3 cặp (instruction, response) theo định dạng `<|user|> ... <|assistant|> ...`. Với mỗi cặp, chỉ ra token nào nên có `loss_mask = 0` và token nào `= 1`.
- **[Trung bình]** Sửa đoạn code SFT ở Mục 2: đổi `loss_mask` thành `[[0, 0, 1, 1, 1, 1, 1, 1]]` (prompt ngắn hơn) và giải thích loss thay đổi ra sao khi số token response tính vào mẫu số tăng lên.
- **[Thử thách]** Với đoạn reward model ở Mục 3, hãy tự tay tính `loss_rm` cho một cặp: $r_{\text{chosen}} = 2.0$, $r_{\text{rejected}} = 0.0$. (Gợi ý: $-\log(\sigma(2.0))$, $\sigma(2.0) \approx 0.881 \implies \text{loss} \approx 0.127$). Rồi đổi thành $r_{\text{chosen}} = 0.0$, $r_{\text{rejected}} = 2.0$ và giải thích vì sao loss tăng vọt (mô hình đang chấm ngược sở thích của người).

---

### Hiểu lầm thường gặp

- ❌ “RLHF dạy mô hình thêm kiến thức mới.” ➔ ✅ RLHF định hình _hành vi/sở thích_, kiến thức chủ yếu đến từ _pretraining_.
- ❌ “Instruction tuning đổi kiến trúc mô hình.” ➔ ✅ Vẫn là _GPT y hệt_; chỉ đổi _dữ liệu + loss mask_ (next-token trên phần response).
- ❌ “Reward model chấm điểm tuyệt đối cho từng câu.” ➔ ✅ Nó học từ _so sánh cặp_ (A tốt hơn B), điểm chỉ có ý nghĩa _tương đối_.
- ❌ “Có RLHF là mô hình không bao giờ sai/không bao giờ độc hại.” ➔ ✅ Alignment _giảm_ rủi ro chứ không _loại bỏ hoàn toàn_; vẫn có jailbreak và hallucination.
- ❌ “DPO và RLHF là hai thứ hoàn toàn khác nhau.” ➔ ✅ Cùng mục tiêu (tối ưu theo sở thích); DPO chỉ là cách _gọn hơn_, bỏ reward model + vòng RL.

---

### Mẹo ghi nhớ

- _P -> S -> A_: *P*retraining (biết ngôn ngữ) -> *S*FT (nghe lệnh) -> *A*lignment/RLHF (cư xử theo sở thích).
- _SFT = pretraining + loss mask_ (chỉ tính loss trên response).
- _RLHF = reward model + PPO (có phạt KL)_; _DPO = RLHF không cần reward model_.
- Mẹo: nhớ ba chữ _HHH_ = *H*elpful, *H*onest, *H*armless - kim chỉ nam của căn chỉnh.

---

### Tóm tắt trực quan

| Khái niệm                    | Một dòng                                                      |
| :--------------------------- | :------------------------------------------------------------ |
| **Pretraining**              | Next-token trên Internet -> biết ngôn ngữ                     |
| **Instruction tuning (SFT)** | Next-token trên cặp (lệnh, đáp) + loss mask -> biết nghe lệnh |
| **Reward model**             | Học từ so sánh cặp: chấm điểm câu người thích cao hơn         |
| **RLHF (PPO)**               | Tối ưu policy theo reward + phạt KL -> theo sở thích người    |
| **DPO**                      | Tối ưu trực tiếp trên cặp sở thích, bỏ reward model + RL      |
| **HHH**                      | Helpful, Honest, Harmless - mục tiêu căn chỉnh                |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [RLHF - The secret sauce of ChatGPT | Arvind Nagraj](https://learningvn.com/learn/xay-dung-llm-tu-dau-tu-attention-den-gpt-mini-pytorch/lesson/69fa75d7-6615-41ad-979c-ca382b436fb8#)
- [Attention Is All You Need (nền tảng Transformer)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [nanoGPT (Andrej Karpathy)](https://github.com/karpathy/nanoGPT)
- [PyTorch documentation](https://pytorch.org/docs/stable/)
- [Hugging Face documentation (TRL/alignment)](https://huggingface.co/docs)

---

### Tổng kết

- Mô hình pretrained thuần túy chỉ “nối chữ”; nó cần _instruction tuning (SFT)_ để biết nghe lệnh và _RLHF/DPO_ để cư xử theo sở thích con người.
- _SFT_ = next-token có giám sát trên cặp (lệnh, đáp), _mask phần prompt_ trong loss.
- _RLHF_ = huấn luyện _reward model_ từ so sánh cặp, rồi tối ưu policy bằng _PPO_ với _phạt KL_; _DPO_ là lối tắt gọn hơn.
- Ba trụ cột _HHH_ (Helpful, Honest, Harmless) giải thích vì sao Claude/ChatGPT hữu ích và an toàn hơn nhiều so với mô hình base.

Tiếp theo, ở _Bài 40: Hạn chế, chi phí, an toàn và đạo đức khi xây/tự host LLM_, chúng ta sẽ nhìn thẳng vào mặt trái: mô hình vẫn có thể bịa (hallucination), tốn kém tính toán, và những cân nhắc đạo đức/an toàn khi bạn tự xây và tự host một LLM. Hẹn gặp bạn ở bài kế tiếp! 🤖
