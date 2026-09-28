# Bài 31: Subword tokenization - Byte-Pair Encoding (BPE)

**Khóa học:** Xây dựng LLM từ đầu: Từ Attention đến GPT mini bằng PyTorch  
**Tuần 7:** Tokenization thực tế, tối ưu & tự host

---

### Giới thiệu

Ở _Bài 30_, GPT mini của bạn đã biết lưu/tải checkpoint và sinh văn bản. Nhưng để ý một điều: suốt cả khóa tới giờ, mô hình vẫn “nhìn” văn bản ở mức _từng ký tự (character-level)_ - mỗi chữ cái, dấu cách, dấu câu là một token. Đơn giản, dễ code, nhưng có một cái giá đắt: câu càng dài thì chuỗi token càng dài kinh khủng, và mô hình phải học lại từ đầu rằng “t-h-e” đi liền nhau tạo thành chữ “the”.

Hãy tưởng tượng bạn đọc sách mà _phải đánh vần từng chữ cái một_ thay vì đọc cả từ: “c-o-n” … “m-è-o”. Chậm và mệt! Ngược lại, nếu ghi nhớ nguyên cả từ điển thì kho từ (vocabulary) khổng lồ, gặp từ lạ (tên riêng, teencode, emoji) là “bó tay”. _Byte-Pair Encoding (BPE)_ là lối đi ở giữa: nó _gộp các cặp ký tự/byte hay đi cùng nhau_ thành “mảnh từ” (subword). Từ phổ biến trở thành 1 token; từ hiếm bị chẻ thành vài mảnh - không bao giờ “kẹt”. Đây chính là cách GPT-2/GPT-3 tokenize thật sự.

> 🔑 **Một câu để nhớ:** BPE bắt đầu từ các byte/ký tự đơn lẻ, rồi _lặp lại việc gộp cặp xuất hiện nhiều nhất_ thành một token mới, cho tới khi đạt kích thước vocabulary mong muốn - cân bằng giữa “chuỗi ngắn” và “vocab nhỏ”.

---

### Bạn sẽ học được gì?

- [ ] Vì sao character-level đủ dùng để học nhưng _không hiệu quả_ khi scale lên
- [ ] Ý tưởng cốt lõi của _BPE_: gộp cặp thường gặp, lặp lại nhiều lần
- [ ] Tự cài _BPE tối giản_ bằng Python (đếm cặp -> gộp -> lặp) và encode/decode
- [ ] Đánh đổi khi chọn _kích thước vocabulary_ (vocab lớn vs chuỗi ngắn)
- [ ] Phân biệt rõ _character-level vs word-level vs subword (BPE)_

📌 **Yêu cầu trước:** Bài 30 (bạn đã có GPT mini chạy được ở mức ký tự) và biết Python cơ bản (dict, vòng lặp).

---

### 1. Vì sao character-level "chưa đủ ngon"?

Nhớ lại: ở character-level, ta lấy tập ký tự khác nhau trong dữ liệu làm vocabulary. Với tiếng Anh cơ bản, vocab chỉ ~65-100 ký tự - rất nhỏ. Nghe hay đấy, nhưng vấn đề nằm ở _độ dài chuỗi_.

Câu “Xây dựng LLM” (giả sử ASCII) dài 11 ký tự -> _11 token_. Trong khi Transformer có chi phí attention _bậc hai theo độ dài_ (nhớ Bài về attention: ma trận $QK^T$ kích thước $(T, T)$). Chuỗi dài gấp 4 lần -> chi phí tính toán tăng ~16 lần! Character-level làm chuỗi dài ra rất nhanh.

Vậy dùng _word-level_ (mỗi từ = 1 token) cho ngắn? Cũng có bẫy:

| Cách tách           | Kích thước vocab    | Độ dài chuỗi | Từ ngoài từ điển (OOV)                   |
| :------------------ | :------------------ | :----------- | :--------------------------------------- |
| **Character-level** | Rất nhỏ (~100)      | Rất dài      | Không bao giờ (mọi ký tự đều có)         |
| **Word-level**      | Rất lớn (>1 triệu)  | Ngắn         | Gặp từ lạ = token `<unk>`, mất thông tin |
| **Subword (BPE)**   | Vừa phải (~30k-50k) | Vừa phải     | Không bao giờ (tệ nhất là chẻ về byte)   |

💡 BPE lấy phần tốt của cả hai: _vocab vừa đủ_ như word-level nhưng _không bao giờ bí từ lạ_ như character-level. Từ phổ biến (“the”, “ing”) thành 1 token; từ hiếm (“antidisestablishmentarianism”) bị chẻ thành nhiều mảnh quen thuộc.

---

### 2. Ý tưởng BPE: gộp cặp thường gặp, lặp lại

**Analogy:** hình dung bạn có một bảng chữ cái và một cuốn sách. Bạn quét cả sách, thấy _cặp “e” + “r” xuất hiện nhiều nhất_ -> phát minh một “chữ mới” gọi là `er`, thay mọi `e r` bằng `er`. Rồi quét lại, thấy cặp `er` + `.` … cứ thế. Mỗi lần gộp, bảng chữ cái phình thêm 1 ký hiệu, còn văn bản ngắn lại. Đó chính xác là BPE.

Thuật toán (huấn luyện tokenizer) gồm đúng 4 bước lặp:

```text
BPE training (mỗi vòng lặp thêm 1 token vào vocab):

  1. Đếm tần suất mọi cặp (a, b) liền kề trong toàn bộ corpus
  2. Chọn cặp (a, b) có tần suất CAO NHẤT
  3. Tạo token mới ab, ghi luật gộp (a, b) -> ab vào "merge list"
  4. Thay MỌI (a, b) trong corpus bằng ab

  Lặp lại tới khi đạt số merge mong muốn (=> vocab_size).
```

⚠️ Phân biệt: _BPE khác character-level ở chỗ vocab được HỌC từ dữ liệu_, không cố định. Merge list học một lần trên corpus lớn, sau đó dùng lại y hệt để encode văn bản mới. Thứ tự các luật gộp là _quan trọng_ - phải áp dụng đúng thứ tự đã học.

---

### 3. Ví dụ tính tay từng bước

Cho corpus tí hon (đã tách theo từ, mỗi từ ta thêm ký hiệu kết thúc `_` để không gộp qua ranh giới từ):

```text
Từ và tần suất xuất hiện:
  "low"    x5   -> l o w _
  "lower"  x2   -> l o w e r _
  "newest" x6   -> n e w e s t _
  "widest" x3   -> w i d e s t _

Vòng 1: đếm cặp liền kề trên TẤT CẢ (nhân theo tần suất từ)
  (e, s): xuất hiện trong newest(6) + widest(3) = 9   <-- cao nhất
  (s, t): 6 + 3 = 9  (chọn 1 trong 2; giả sử chọn e,s trước)
  => MERGE (e, s) -> "es"

Vòng 2: sau khi có "es"
  (es, t): newest(6) + widest(3) = 9   <-- cao nhất
  => MERGE (es, t) -> "est"

Vòng 3:
  (o, w): low(5) + lower(2) = 7   <-- cao nhất
  => MERGE (o, w) -> "ow"
  ...
```

Sau vài vòng, vocabulary ban đầu `{l, o, w, e, r, n, s, t, i, d, _}` được bổ sung thêm `es`, `est`, `ow`, … Những “mảnh” này chính là subword hữu ích: `est` (đuôi so sánh nhất), `ow` (hay gặp trong “low/lower”)!

---

### 4. Cài BPE tối giản bằng Python

Ta hiện thực đúng 4 bước ở trên. Code là Python thuần (BPE là chuyện đếm/gộp, không cần PyTorch):

```python
from collections import Counter

# Corpus: map each word (split into symbols) to its frequency.
# We add an end-of-word marker '_' so merges don't cross word boundaries.
corpus = {
    ("l", "o", "w", "_"): 5,
    ("l", "o", "w", "e", "r", "_"): 2,
    ("n", "e", "w", "e", "s", "t", "_"): 6,
    ("w", "i", "d", "e", "s", "t", "_"): 3,
}

def count_pairs(corpus):
    # Count every adjacent symbol pair, weighted by word frequency.
    pairs = Counter()
    for symbols, freq in corpus.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge_pair(corpus, pair):
    # Replace all occurrences of `pair` with the merged symbol a+b.
    a, b = pair
    new_corpus = {}
    for symbols, freq in corpus.items():
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b) # join into one new symbol
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        new_corpus[tuple(merged)] = freq
    return new_corpus

NUM_MERGES = 6
merges = [] # ordered list of learned rules (a, b)

for step in range(NUM_MERGES):
    pairs = count_pairs(corpus)
    if not pairs:
        break
    best = max(pairs, key=pairs.get) # most frequent adjacent pair
    merges.append(best)
    corpus = merge_pair(corpus, best)
    print(f"merge {step+1}: {best} -> '{best[0]+best[1]}'  (count={pairs[best]})")

# Example output:
# merge 1: ('e', 's') -> 'es'   (count=9)
# merge 2: ('es', 't') -> 'est' (count=9)
# merge 3: ('o', 'w') -> 'ow'   (count=7)
# merge 4: ('l', 'ow') -> 'low' (count=7)
# ...
print("Learned merges (in order):", merges)
```

Để _encode một từ mới_, ta áp dụng lại merge list theo đúng thứ tự đã học:

```python
def encode_word(word, merges):
    # Start from characters + end marker, then apply learned merges IN ORDER.
    symbols = list(word) + ["_"]
    for a, b in merges: # order matters!
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols

print(encode_word("lowest", merges))
# e.g. ['low', 'est', '_']  -> unseen word split into KNOWN subwords, no <unk>!
print(encode_word("slowly", merges))
# e.g. ['s', 'low', 'l', 'y', '_'] -> rare parts fall back to characters
```

🎯 Điểm mấu chốt: _“lowest” chưa từng xuất hiện trong corpus_ nhưng vẫn được chẻ thành `low` + `est` - hai mảnh mô hình đã quen. Đây là siêu năng lực của BPE: _không bao giờ_ `<unk>`, trường hợp tệ nhất chỉ là quay về ký tự/byte đơn lẻ.

---

### 5. Byte-level BPE: vì sao "byte" chứ không chỉ "ký tự"?

GPT-2 dùng _byte-level BPE_: đơn vị khởi đầu không phải ký tự Unicode mà là _256 byte_ (0-255). Lý do: mọi văn bản trên đời (tiếng Việt có dấu, emoji 🤖, ký tự lạ) đều mã hóa được thành byte UTF-8, nên vocabulary cơ sở chỉ cần _đúng 256 phần tử_ là _phủ mọi thứ, không bao giờ OOV_.

```python
text = "Xin chào 🤖"
raw_bytes = text.encode("utf-8") # bytes object
print(list(raw_bytes)[:12])
# e.g. [88, 105, 110, 32, 99, 104, 195, 160, 111, 32, 240, 159]
# 'à' -> 2 bytes (195,160); '🤖' -> 4 bytes. Base vocab = 256 possible byte values.
```

Sau đó BPE gộp các cặp byte thường gặp y như trên. GPT-2 học ~50.000 merge -> `vocab_size = 50257` (256 byte + các merge + 1 token đặc biệt `<|endoftext|>`).

---

### 6. Nối lại GPT mini: đổi vocab, phần còn lại giữ nguyên

Tin vui: chuyển từ character-level sang BPE _không phá vỡ kiến trúc_! Bạn chỉ cần thay `vocab_size` và bảng token. Toàn bộ luồng attention/block/logits không đổi:

```text
Character-level:                BPE (subword):
  vocab_size = 65                 vocab_size = 50257
  "the cat" -> 7 tokens           "the cat" -> ~2 tokens  (ngắn hơn nhiều!)

Luồng dữ liệu (GIỐNG HỆT, chỉ khác kích thước):

  ids (B, T) --[Embedding vocab_size x d]--> x (B, T, d)
      |
      +--> [Transformer blocks: attention + feed-forward] --> (B, T, d)
      |
      +--> [LayerNorm] --> [Linear d x vocab_size] --> logits (B, T, vocab_size)
      |
      +--> softmax --> phân phối xác suất token kế tiếp
```

```python
import torch.nn as nn

# The ONLY things that change when switching char-level -> BPE:
vocab_size = 50257 # was ~65 for char-level
d_model = 384
token_embedding = nn.Embedding(vocab_size, d_model) # (50257, 384) table
lm_head = nn.Linear(d_model, vocab_size)            # projects to (B, T, 50257)

# Everything else (blocks, attention, mask, training loop) stays identical.
```

💡 Chuỗi ngắn hơn -> chi phí attention $\mathcal{O}(T^2)$ giảm mạnh -> huấn luyện/sinh văn bản nhanh hơn nhiều với cùng lượng văn bản. Đó là lý do LLM thật đều dùng subword, không dùng character-level.

---

### Nhìn sâu hơn

- **Đánh đổi kích thước vocab:** Vocab lớn (nhiều merge) -> chuỗi ngắn hơn nhưng bảng embedding `(vocab_size, d)` và `lm_head` to ra, tốn bộ nhớ + tham số. Vocab nhỏ -> chuỗi dài, attention đắt. GPT-2 chọn ~50k như một điểm cân bằng thực nghiệm.
- **Thứ tự merge là bất biến:** Encode phải áp merge _đúng thứ tự đã học_; đảo thứ tự cho ra token khác. Vì thế tokenizer lưu cả vocab lẫn merges (nhớ file `merges.txt` của GPT-2).
- **Chi phí huấn luyện tokenizer:** Bước `count_pairs` chạy trên corpus lớn tốn thời gian; thư viện thật (`tiktoken`, `HF tokenizers`) viết bằng Rust để nhanh - ta chỉ tự cài bản tối giản để hiểu.
- **Ranh giới từ & khoảng trắng:** GPT-2 gắn khoảng trắng vào token (ví dụ `Ġthe` = " the") nên " the" và “the” là hai token khác nhau - chi tiết này ảnh hưởng cách đếm token.

---

### Thực hành

- **[Dễ]** Chạy đoạn code mục 4 với `NUM_MERGES = 3` rồi `= 10`. In ra `merges` và so sánh `encode_word("newest", merges)` giữa hai trường hợp: merge càng nhiều thì chuỗi càng ngắn thế nào?
- **[Trung bình]** Sửa `encode_word` để trả về danh sách _id số nguyên_ thay vì chuỗi: xây dict ánh xạ mỗi symbol -> một số nguyên (id), rồi viết `decode(ids)` ghép ngược lại thành văn bản.
- **[Thử thách]** Chạy `"Xin chào 🤖".encode("utf-8")` rồi cài BPE _trên danh sách byte_ (số 0-255) thay vì ký tự. Xác nhận: dù có emoji/tiếng Việt, thuật toán vẫn chạy được mà không bao giờ gặp token lạ.

---

### Hiểu lầm thường gặp

- ❌ “BPE gộp các từ có nghĩa giống nhau.” ➔ ✅ BPE _chỉ đếm tần suất cặp ký tự/byte liền kề_, hoàn toàn dựa trên thống kê - nó không hiểu nghĩa, `est` được gộp vì hay đi cùng nhau, không phải vì “nghĩa”.
- ❌ “1 token = 1 từ.” ➔ ✅ Với BPE, một từ có thể là 1 token (nếu phổ biến) hoặc nhiều token (nếu hiếm); ngược lại " the" (có dấu cách) có thể là 1 token. _Token không đồng nghĩa với từ._
- ❌ “Vocab càng lớn càng tốt.” ➔ ✅ Vocab lớn làm chuỗi ngắn nhưng phình bảng embedding + tham số và cần nhiều dữ liệu để học tốt mỗi token hiếm; phải _cân bằng_.
- ❌ “Character-level không bao giờ dùng được.” ➔ ✅ Character-level vẫn tốt để _học/thử nghiệm_ (như cả khóa tới giờ); nó chỉ _kém hiệu quả khi scale_ chứ không sai.

---

### Mẹo ghi nhớ

- BPE = _Bắt cặp, Phổ biến, Ép lại_ (gộp cặp xuất hiện nhiều nhất, lặp lại).
- Nhớ 4 bước: _đếm cặp -> chọn cặp nhiều nhất -> ghi luật gộp -> thay thế_.
- Mẹo: hình dung BPE như “nén file” cho văn bản - cụm hay lặp thì được viết tắt thành một ký hiệu ngắn.

---

### Tóm tắt trực quan

| Khái niệm           | Một dòng                                            |
| :------------------ | :-------------------------------------------------- |
| **Character-level** | 1 token = 1 ký tự; vocab nhỏ, chuỗi rất dài         |
| **Word-level**      | 1 token = 1 từ; vocab khổng lồ, dễ dính `<unk>`     |
| **BPE (subword)**   | Gộp cặp thường gặp thành mảnh; vocab vừa, không OOV |
| **Merge list**      | Danh sách luật gộp có THỨ TỰ, học từ corpus         |
| **Byte-level BPE**  | Bắt đầu từ 256 byte -> phủ mọi văn bản/emoji        |
| **vocab_size**      | Chỉnh nó thôi; kiến trúc Transformer giữ nguyên     |
| **Đánh đổi**        | Vocab lớn = chuỗi ngắn nhưng embedding to hơn       |

---

### Tài liệu tham khảo

- 🎬 _Video hướng dẫn:_ [Lecture 8: The GPT Tokenizer: Byte Pair Encoding](https://www.youtube.com/watch?v=fKd8s29e-l4)
- [nanoGPT (Andrej Karpathy)](https://github.com/karpathy/nanoGPT) - có ví dụ BPE/tokenizer
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - kiến trúc Transformer
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Hugging Face - tài liệu về tokenizers](https://huggingface.co/docs)
- [PyTorch documentation](https://pytorch.org/docs/stable/)

---

### Tổng kết

- Character-level _đúng nhưng không hiệu quả_: chuỗi quá dài -> attention $\mathcal{O}(T^2)$ đắt đỏ.
- _BPE_ cân bằng: gộp cặp thường gặp thành subword; vocab vừa phải, _không bao giờ OOV_ (tệ nhất quay về byte).
- Thuật toán BPE = _đếm cặp -> chọn cặp nhiều nhất -> ghi luật gộp -> thay thế_, lặp tới khi đủ `vocab_size`.
- _Byte-level BPE_ (GPT-2) khởi đầu từ 256 byte nên phủ mọi văn bản, kể cả tiếng Việt có dấu và emoji.
- Đổi sang BPE _không phá kiến trúc_: chỉ thay `vocab_size` và bảng token; attention/block/logits giữ nguyên.

Bây giờ bạn đã hiểu _bên trong_ BPE hoạt động ra sao. Nhưng tự cài lại tokenizer cho corpus lớn thì chậm và dễ sai - trong thực tế ta dùng các thư viện đã tối ưu. Ở _Bài 32: Dùng tokenizer thực tế (tiktoken / Hugging Face tokenizers)_, chúng ta sẽ dùng chính tokenizer của GPT-2 chỉ với vài dòng code, encode/decode văn bản thật và cắm thẳng vào GPT mini. Hẹn gặp ở bài kế tiếp! 🚀
