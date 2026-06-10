# PIPELINE NCKH BIGFIVE — Phiên bản V3.2

> \\\*\\\*Tích hợp MediaPipe · Symmetric Bidirectional Cross-Attention · LoRA · Hybrid Loss\\\*\\\*
>
> \\\*\\\*Baseline-complete:\\\*\\\* Dữ liệu \\\& Nhãn · Xử lý lỗi MediaPipe · Padding mask · LR Scheduler · Early Stopping · Biểu đồ CCC Loss

\---

## Mục lục

0. [Giai đoạn 0 — Dữ liệu, Nhãn \& Phân chia Tập](#giai-đoạn-0)
1. [Giai đoạn 1 — Tiền xử lý \& Phân đoạn](#giai-đoạn-1)
2. [Giai đoạn 2 — Trích xuất Đặc trưng](#giai-đoạn-2)

   * [2.1 Luồng Âm thanh (WavLM)](#21-luồng-âm-thanh-wavlm)
   * [2.2 Luồng Hình ảnh (MediaPipe Holistic)](#22-luồng-hình-ảnh-mediapipe-holistic)
   * [2.3 Luồng Ngôn ngữ (RoBERTa)](#23-luồng-ngôn-ngữ-roberta)
3. [Giai đoạn 3 — Mạng Dung hợp Đa phương thức](#giai-đoạn-3)

   * [3.1 Ánh xạ không gian — GLU Projection](#31-ánh-xạ-không-gian--glu-projection)
   * [3.2 Đồng bộ hóa thời gian — Positional Encoding](#32-đồng-bộ-hóa-thời-gian--positional-encoding)
   * [3.3 Chú ý chéo hai chiều đối xứng](#33-chú-ý-chéo-hai-chiều-đối-xứng)
   * [3.4 Cổng dung hợp thích ứng — GMU](#34-cổng-dung-hợp-thích-ứng--gmu)
   * [3.5 Attention Pooling — Giảm N × 256 → 1 × 256](#35-attention-pooling--giảm-n--256--1--256)
   * [3.6 Mạng Đa nhiệm BigFive — Giảm 1 × 256 → 1 × 5](#36-mạng-đa-nhiệm-bigfive--giảm-1--256--1--5)
4. [Giai đoạn 4 — Cải tiến Kiến trúc V3](#giai-đoạn-4)

   * [4.1 Hybrid Loss + Gradient Accumulation](#41-hybrid-loss--gradient-accumulation)
   * [4.2 LoRA — Tinh chỉnh tham số hiệu quả](#42-lora--tinh-chỉnh-tham-số-hiệu-quả)
   * [4.3 Mạng Đa nhiệm Động + PIMC Loss *(Phase 2)*](#43-mạng-đa-nhiệm-động--pimc-loss-phase-2)
5. [Giai đoạn 5 — Khởi tạo \& Quy trình Huấn luyện](#giai-đoạn-5)
6. [Giai đoạn 6 — Theo dõi \& Trực quan hóa CCC Loss](#giai-đoạn-6)

\---

## Giai đoạn 0 — Dữ liệu, Nhãn \& Phân chia Tập {#giai-đoạn-0}

> \\\*\\\*Lỗ hổng baseline cần vá trước tiên:\\\*\\\* Pipeline chỉ có ý nghĩa khi biết rõ dữ liệu đầu vào trông như thế nào, nhãn được lưu ở đâu và mô hình được đánh giá theo quy ước nào.

### 0.1 Nguồn dữ liệu và Cấu trúc thư mục

Bộ dữ liệu sử dụng là **ChaLearn First Impressions V2** (hoặc tương đương), gồm các video ngắn 15 giây quay nghiệm thể nói chuyện trực tiếp vào camera. Mỗi video có một nhãn điểm số BigFive tương ứng.

Cấu trúc thư mục chuẩn:

```
dataset/
├── train/          ← thư mục train (có thể chia nhỏ: train-1, train-2, ...)
│   ├── video\\\_001.mp4
│   ├── video\\\_002.mp4
│   └── ...
├── val/            ← tập validation (dùng để theo dõi loss, early stopping)
├── test/           ← tập test (chỉ đụng vào khi báo cáo kết quả cuối)
└── labels/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

### 0.2 Chuẩn hóa Nhãn (Label Normalization)

File `train.csv` có cấu trúc:

```
video\\\_id, openness, conscientiousness, extraversion, agreeableness, neuroticism
video\\\_001, 0.72, 0.51, 0.83, 0.60, 0.41
...
```

Thang điểm gốc thường nằm trong $\[0, 1]$ (đã chuẩn hóa sẵn bởi ChaLearn). Nếu dữ liệu tự thu thập với thang Likert $\[1, 5]$, áp dụng Min-Max:

$$
y\_{\\text{norm}} = \\frac{y - y\_{\\min}}{y\_{\\max} - y\_{\\min}} \\in \[0, 1]
$$

**Kiểm tra phân phối nhãn:** Trước khi huấn luyện, vẽ histogram 5 cột để phát hiện nhãn lệch (skewed labels) — nếu một tính cách tập trung quá ở một đầu (>80% mẫu trong $\[0.4, 0.6]$), CCC sẽ khó tính toán ổn định, cần xem xét augmentation nhãn.

### 0.3 Phân chia Train / Val / Test

Tỷ lệ phân chia tiêu chuẩn cho baseline:

|Tập|Tỷ lệ|Vai trò|
|-|-|-|
|Train|70%|Cập nhật trọng số qua Gradient Descent|
|Val|15%|Theo dõi loss, điều chỉnh hyperparameter, Early Stopping|
|Test|15%|**Chỉ chạy một lần duy nhất** khi báo cáo kết quả cuối|

> \\\*\\\*Nguyên tắc tuyệt đối:\\\*\\\* Không bao giờ dùng thông tin từ Test để điều chỉnh kiến trúc hoặc chọn checkpoint. Mọi quyết định kỹ thuật (chọn learning rate, số epoch) phải dựa trên Val.

Phân chia ngẫu nhiên với `random\\\_seed = 42` để đảm bảo tái lập (reproducibility).

### 0.4 Chiến lược Đánh giá (Evaluation Protocol)

**Metric chính:** CCC trung bình trên tập Val, tính độc lập cho từng tính cách:

$$
\\overline{\\text{CCC}}*{\\text{val}} = \\frac{1}{5}\\sum*{k \\in {O,C,E,A,N}} \\rho\_c^{(k)}(\\hat{\\mathbf{y}}\_k^{\\text{val}},, \\mathbf{y}\_k^{\\text{val}})
$$

**Checkpoint tốt nhất:** Lưu trọng số của epoch có $\\overline{\\text{CCC}}\_{\\text{val}}$ cao nhất — không phải epoch cuối.

\---



### Bước 1 · Chuẩn hóa đầu vào (FFmpeg)

Đưa file video `.mp4` qua **FFmpeg** để tách biệt hai luồng dữ liệu song song:

* **Luồng âm thanh:** xuất file `.wav` chuẩn 16 kHz, mono, 16-bit PCM.
* **Luồng hình ảnh:** giải nén toàn bộ khung hình ra ổ đĩa ở chuẩn **30 FPS**, định dạng `.jpg`.

### Bước 2 · Cắt lời tự động (Whisper ASR)

Nạp file `.wav` vào mô hình **Whisper** để sinh ra:

* Kịch bản chép lời (transcript) toàn bộ đoạn video.
* **Nhãn thời gian cấp câu** (sentence-level timestamp): `(start\\\_câu, end\\\_câu)`.
* **Nhãn thời gian cấp từ** (word-level timestamp): `(start\\\_từ, end\\\_từ)` cho từng từ trong từng câu.

### Bước 3 · Mở rộng lề theo ngữ cảnh (Contextual Padding)

Nhằm không bỏ sót các tín hiệu phi ngôn ngữ xuất hiện ngay trước hoặc sau lời nói (gật đầu chuẩn bị phát ngôn, hơi thở dài sau câu), mỗi từ được nới rộng một khoảng dự phòng $p = 0.1\\ \\text{s}$ về hai phía.

Gọi $d$ là khoảng trống thực tế giữa điểm kết thúc từ thứ $n$ và điểm bắt đầu từ thứ $n+1$:

$$d = \\text{start}\_{n+1} - \\text{end}\_n \\geq 0$$

Điều kiện kiểm soát cốt lõi phải so sánh $d$ với $\\mathbf{2p}$ — không phải $p$ — vì mỗi bên cần $p$, tổng cộng cần $2p$ khoảng trống để cả hai đầu được mở rộng cực đại mà không chồng lấn:

$$
\\text{end}*n^\* = \\begin{cases}
\\text{end}n + p \& \\text{nếu } d \\geq 2p \\\[6pt]
\\text{end}n + \\dfrac{d}{2} \& \\text{nếu } 0 < d < 2p \\\[6pt]
\\text{end}n \& \\text{nếu } d = 0
\\end{cases}
\\qquad
\\text{start}{n+1}^\* = \\begin{cases}
\\text{start}{n+1} - p \& \\text{nếu } d \\geq 2p \\\[6pt]
\\text{start}{n+1} - \\dfrac{d}{2} \& \\text{nếu } 0 < d < 2p \\\[6pt]
\\text{start}*{n+1} \& \\text{nếu } d = 0
\\end{cases}
$$

> \\\*\\\*Tại sao ngưỡng là $2p$ chứ không phải $p$?\\\*\\\* Nếu dùng $d \\\\geq p$, trường hợp $d = p$ (vừa đủ một đệm) sẽ bị cả hai đầu cùng mở rộng tối đa: $\\\\text{end}\\\_n + p$ và $\\\\text{start}\\\_{n+1} - p$, tổng chiếm $2p = d$ — \\\*\\\*cửa sổ hai từ chạm nhau chính xác tại cùng một điểm thời gian\\\*\\\*, gây chồng lấn biên. Dùng ngưỡng $2p$ đảm bảo điều kiện mở rộng cực đại chỉ kích hoạt khi khoảng trống thực sự đủ chỗ cho cả hai đệm, triệt tiêu hoàn toàn sự nhân đôi cửa sổ thời gian.

**Ví dụ minh họa ba trường hợp** với $p = 0.1\\text{s}$:

|Khoảng trống $d$|Điều kiện|$\\text{end}\_n^\*$|$\\text{start}\_{n+1}^\*$|Kết quả|
|:-:|:-:|:-:|:-:|-|
|$0.25\\text{s}$|$d \\geq 2p$|$\\text{end}\_n + 0.1$|$\\text{start}\_{n+1} - 0.1$|Mở rộng tối đa, còn $0.05\\text{s}$ đệm giữa|
|$0.15\\text{s}$|$0 < d < 2p$|$\\text{end}\_n + 0.075$|$\\text{start}\_{n+1} - 0.075$|Chia đôi khoảng trống, hai cửa sổ chạm nhau|
|$0\\text{s}$|$d = 0$|$\\text{end}\_n$|$\\text{start}\_{n+1}$|Không mở rộng|

**Ví dụ áp dụng:** Từ *"Play"* nằm trong khoảng $\[1.0\\text{s},\\ 1.3\\text{s}]$, từ tiếp theo bắt đầu ở $1.55\\text{s}$ → $d = 0.25\\text{s} \\geq 2p = 0.2\\text{s}$. Sau khi mở rộng: cửa sổ *"Play"* trở thành $\[0.9\\text{s},\\ 1.4\\text{s}]$, từ tiếp theo co về $\[1.45\\text{s},\\ \\ldots]$ — khoảng trống còn lại $0.05\\text{s}$, không chồng lấn.

\---

## Giai đoạn 2 — Trích xuất Đặc trưng {#giai-đoạn-2}

### 2.1 Luồng Âm thanh (WavLM)

> \\\*\\\*Chiến lược tránh OOM — Sentence-level Chunking:\\\*\\\* Thay vì nạp toàn bộ file âm thanh nguyên khối vào WavLM (khiến Self-Attention phải tính ma trận $T \\\\times T$ với $T$ rất lớn, dễ gây \\\*\\\*CUDA Out of Memory\\\*\\\* với video dài), hệ thống sử dụng \\\*\\\*nhãn thời gian cấp câu\\\*\\\* của Whisper để cắt file audio thành từng đoạn câu ngắn (\\\*\\\*Sentence-level\\\*\\\*). Mỗi đoạn câu được nạp lần lượt qua WavLM để trích xuất đặc trưng $\\\\mathbf{M}\\\_{\\\\text{audio}}$, sau đó mới ghép nối lại. Việc này \\\*\\\*triệt tiêu 100% rủi ro tràn VRAM\\\*\\\* do cơ chế Self-Attention khi gặp video dài.

#### Bước 1 · Cắt file audio theo nhãn thời gian câu \& Nạp vào WavLM

Sử dụng **nhãn thời gian cấp câu mở rộng** `(start\\\_câu\\\*, end\\\_câu\\\*)` để cắt file `.wav` thành từng đoạn ngắn. Biên câu mở rộng **không lấy trực tiếp từ Whisper** mà được suy ra từ nhãn từ đã padding (Giai đoạn 1) theo quy tắc:

$$
\\text{start}^*{\\text{câu}} = \\min{w \\in \\text{câu}}, \\text{start}^w, \\qquad \\text{end}^\**{\\text{câu}} = \\max\_{w \\in \\text{câu}}, \\text{end}^\*\_w
$$

Cách tính này đảm bảo chunk audio chứa đầy đủ phần lề của từ đầu câu và từ cuối câu, tránh tình huống `r\\\_start` hoặc `r\\\_end` bị âm hoặc vượt biên $T\_s$ ở Bước 2. Với mỗi đoạn câu thứ $s$, nạp đoạn âm thanh tương ứng vào **WavLM Base** (hoặc Large). Mô hình trượt qua tín hiệu với bước nhảy $20\\ \\text{ms}$, tức **50 cửa sổ mỗi giây**.

**Đầu ra cho mỗi câu $s$:** ma trận hai chiều cục bộ

$$
\\mathbf{M}\_{\\text{audio}}^{(s)} \\in \\mathbb{R}^{T\_s \\times 768}
$$

Trong đó:

* $T\_s$ = số bước thời gian của riêng câu $s$. Với đoạn dài $k\_s$ giây: $T\_s = k\_s \\times 50$.
* $768$ = số chiều đặc trưng tại mỗi bước (WavLM Large cho $1024$).

**Ví dụ:** đoạn câu dài $3\\text{s}$ → $\\mathbf{M}\_{\\text{audio}}^{(s)} \\in \\mathbb{R}^{150 \\times 768}$.

#### Bước 2 · Chiếu nhãn thời gian từ lên ma trận cục bộ

Quy đổi cửa sổ thời gian $\[\\text{start}^*\_{\\text{từ}}, \\text{end}^*\_{\\text{từ}}]$ (đã mở rộng lề) sang chỉ số dòng **tương đối trong đoạn câu**:

$$
r\_{\\text{start}} = \\lfloor (\\text{start}^*\_{\\text{từ}} - \\text{start}^{\\text{câu}}) \\times 50 \\rfloor, \\qquad r*{\\text{end}} = \\lfloor (\\text{end}^*\_{\\text{từ}} - \\text{start}^*\_{\\text{câu}}) \\times 50 \\rfloor
$$

Sau đó **clip về biên hợp lệ** để phòng sai số dấu phẩy động nhỏ đẩy nhẹ vào biên:

$$
r\_{\\text{start}} \\leftarrow \\max(0,, r\_{\\text{start}}), \\qquad r\_{\\text{end}} \\leftarrow \\min(T\_s,, r\_{\\text{end}})
$$

**Ví dụ:** từ ở $\[0.9\\text{s},\\ 1.4\\text{s}]$ trong câu bắt đầu $0.5\\text{s}$ → dòng $20$ đến $45$ trong $\\mathbf{M}\_{\\text{audio}}^{(s)}$.

> **⚠️ Edge case — Từ quá ngắn (`r_start >= r_end`):** Nếu từ ngắn hơn 1 bước WavLM (< 20 ms), sau khi clip có thể xảy ra `r_end <= r_start` → slice rỗng → Mean Pool chia cho 0 → **NaN**. Bắt buộc kiểm tra:
>
> ```python
> if r_end <= r_start:
>     f_audio = M_audio_s[r_start]          # fallback: lấy đúng 1 frame
> else:
>     f_audio = M_audio_s[r_start:r_end].mean(dim=0)
> ```

Trích đoạn con:

$$
\\mathbf{S}*{\\text{audio}} = \\mathbf{M}*{\\text{audio}}^{(s)}\[r\_{\\text{start}} : r\_{\\text{end}},\\ :] \\in \\mathbb{R}^{(r\_{\\text{end}} - r\_{\\text{start}}) \\times 768}
$$

#### Bước 3 · Mean Pooling theo trục thời gian

Lấy trung bình cộng trên trục dòng (trục thời gian) để thu về một vector đại diện cho **từng từ**:

$$
\\mathbf{f}*{\\text{audio}} = \\frac{1}{r*{\\text{end}} - r\_{\\text{start}}} \\sum\_{t=r\_{\\text{start}}}^{r\_{\\text{end}}-1} \\mathbf{M}\_{\\text{audio}}^{(s)}\[t,\\ :] \\in \\mathbb{R}^{768}
$$

#### Kết quả cuối

Sau khi xử lý lần lượt toàn bộ câu và ghép nối các vector từ lại:

$$
\\boxed{\\mathbf{A} \\in \\mathbb{R}^{a \\times 768}}
$$

Hàng $i$ chứa đặc trưng âm học của từ thứ $i$. **Không có thêm chiều nào (không dùng pause).**

\---

### 2.2 Luồng Hình ảnh (MediaPipe Holistic) {#22-luồng-hình-ảnh-mediapipe-holistic}

> \\\*\\\*Lý do thay thế Swin Transformer:\\\*\\\* Swin-T xử lý điểm ảnh RGB thô, dễ bị lấn át bởi nhiễu nền, màu da, ánh sáng — những yếu tố không liên quan đến tâm lý. \\\*\\\*MediaPipe Holistic\\\*\\\* trích xuất trực tiếp hệ tọa độ xương khớp — thông tin hình học thuần của nghiệm thể — đặc hiệu hơn nhiều cho bài toán ngôn ngữ cơ thể và đồng điệu cảm xúc.

#### Bước 1 · Trích xuất khung hình

Với cửa sổ thời gian $\[\\text{start}^*, \\text{end}^*]$ của từng từ (đã mở rộng lề), tính chỉ số khung hình:

$$
f\_{\\text{start}} = \\lfloor \\text{start}^\* \\times \\text{FPS} \\rfloor, \\qquad f\_{\\text{end}} = \\lfloor \\text{end}^\* \\times \\text{FPS} \\rfloor
$$

Với FPS = $30$, từ *"Play"* ở $\[0.9\\text{s},\\ 1.4\\text{s}]$ → khung $27$ đến $42$ (tổng $F = 15$ khung).

#### Bước 2 · Trích xuất điểm tọa độ thô (MediaPipe Holistic)

Chạy **MediaPipe Holistic** trên từng khung hình. Mô hình trả về hai nhóm điểm riêng biệt:

* **Cụm Pose:** 33 điểm xương khớp toàn thân (**0-indexed: 0 → 32** theo MediaPipe API; hip trái = landmark 23, hip phải = landmark 24), mỗi điểm có 3 tọa độ $(x, y, z)$ → vector $\\mathbf{p}\_{\\text{pose}}(t) \\in \\mathbb{R}^{99}$.
* **Cụm Face Mesh:** 468 điểm lưới khuôn mặt (**0-indexed: 0 → 467**; điểm đỉnh mũi dùng làm gốc Face Mesh = landmark **1**, không phải 0), mỗi điểm có 3 tọa độ $(x, y, z)$ → vector $\\mathbf{p}\_{\\text{face}}(t) \\in \\mathbb{R}^{1404}$.

**Hai cụm này được xử lý chuẩn hóa độc lập ở Bước 2.5 trước khi ghép lại.** Tuyệt đối không gộp chung 501 điểm rồi mới chuẩn hóa, vì gốc tọa độ trục $z$ của hông (Pose) và tâm đầu (Face Mesh) nằm ở hai mặt phẳng hoàn toàn khác nhau — phép trừ chung một điểm mốc sẽ gây lệch metric không gian nghiêm trọng.

> \\\*\\\*⚠️ Xử lý lỗi — MediaPipe không phát hiện được landmarks:\\\*\\\*
> Trong thực tế, MediaPipe thất bại ở một số khung hình do: góc quay bất lợi, ánh sáng yếu, khuôn mặt bị che, hoặc khoảng cách camera không phù hợp. Chiến lược xử lý theo thứ tự ưu tiên:
>
> 1. \\\*\\\*Carry-forward từ khung trước hợp lệ:\\\*\\\* Nếu khung $t$ thất bại nhưng $t-1$ có dữ liệu, dùng $\\\\mathbf{pos}(t) = \\\\mathbf{pos}(t-1)$ → vector vận tốc $\\\\mathbf{v}(t) = \\\\mathbf{0}$ (không có chuyển động — đúng về mặt vật lý).
> 2. \\\*\\\*Nếu toàn bộ cửa sổ từ thất bại:\\\*\\\* Điền $\\\\bar{\\\\mathbf{v}} = \\\\mathbf{0}\\\_{1503}$ và đánh dấu `video\\\_pose\\\_valid = False` để loại khỏi tính Loss của batch đó.
> 3. \\\*\\\*Ngưỡng loại mẫu:\\\*\\\* Nếu $> 30\\\\%$ khung hình trong video thất bại, loại toàn bộ video khỏi dataset ngay từ bước tiền xử lý.

#### Bước 2.5 · Chuẩn hóa Hệ quy chiếu Phân tách (Decoupled Reference-frame Normalization)

Thay vì chuẩn hóa thống nhất bằng một điểm mốc chung (Nose-centric cho toàn bộ 1503 chiều), hệ thống thực hiện **hai phép tịnh tiến độc lập**, mỗi phép tôn trọng không gian metric riêng của từng module:

\---

**Nhánh 1 — Cụm Pose (33 điểm → gốc Trung điểm Hông):**

Lấy **Trung điểm Hông** (Hip Midpoint) làm mốc tịnh tiến — điểm này nằm ở thân dưới, ổn định nhất trong toàn chuỗi chuyển động, đảm bảo bất biến toàn cục tốt hơn Mũi (Mũi thuộc đầu, dao động nhiều khi gật gù hay xoay cổ):

$$
\\mathbf{m}*{\\text{hip}}(t) = \\frac{1}{2}\\left(\\mathbf{p}*{\\text{left\_hip}}(t) + \\mathbf{p}\_{\\text{right\_hip}}(t)\\right) \\in \\mathbb{R}^3
$$

Tịnh tiến tất cả 33 điểm Pose về gốc Trung điểm Hông:

$$
\\tilde{p}*{j}^{\\text{pose}}(t) = p*{j}^{\\text{pose}}(t) - \\mathbf{m}\_{\\text{hip}}(t), \\qquad j = 1, \\ldots, 33
$$

Chải phẳng thành vector:

$$
\\mathbf{pos}\_{\\text{pose}}(t) = \\left\[\\tilde{p}*1^{\\text{pose}}(t),, \\ldots,, \\tilde{p}*{33}^{\\text{pose}}(t)\\right]^\\top \\in \\mathbb{R}^{99}
$$

\---

**Nhánh 2 — Cụm Face Mesh (468 điểm → gốc Nose Face Mesh):**

Lấy **điểm Nose trung tâm của lưới khuôn mặt** (điểm số 1 trong danh sách 468 điểm của Face Mesh — đây là điểm đỉnh mũi nằm ở chính tâm lưới, hoàn toàn khác với điểm Nose số 0 của Pose) làm mốc tịnh tiến. Việc dùng điểm thuộc chính lưới mặt giữ nguyên vẹn hình thái vi mô của 468 điểm, loại bỏ hoàn toàn chuyển động tịnh tiến đầu mà không bị ảnh hưởng bởi chuyển động thân người:

$$
\\mathbf{m}\_{\\text{nose\_face}}(t) = p\_1^{\\text{face}}(t) \\in \\mathbb{R}^3
$$

Tịnh tiến tất cả 468 điểm Face Mesh về gốc Nose Face Mesh:

$$
\\tilde{p}*{j}^{\\text{face}}(t) = p*{j}^{\\text{face}}(t) - \\mathbf{m}\_{\\text{nose\_face}}(t), \\qquad j = 1, \\ldots, 468
$$

Chải phẳng thành vector:

$$
\\mathbf{pos}\_{\\text{face}}(t) = \\left\[\\tilde{p}*1^{\\text{face}}(t),, \\ldots,, \\tilde{p}*{468}^{\\text{face}}(t)\\right]^\\top \\in \\mathbb{R}^{1404}
$$

\---

**Ghép hai cụm đã chuẩn hóa:**

Sau khi mỗi cụm đã được tịnh tiến về gốc tọa độ nội tại riêng, ghép nối thành vector tổng hợp:

$$
\\mathbf{pos}*{\\text{norm}}(t) = \\left\[\\mathbf{pos}*{\\text{pose}}(t)^\\top \\mid \\mathbf{pos}\_{\\text{face}}(t)^\\top\\right]^\\top \\in \\mathbb{R}^{1503}
$$

> \\\*\\\*Lý do phân tách:\\\*\\\* Nếu chuẩn hóa chung bằng một điểm mốc (ví dụ: Nose của Pose), các tọa độ $z$ của cụm Face Mesh — vốn được đo từ tâm đầu — sẽ bị dịch chuyển sai lệch so với tọa độ $z$ của cụm Pose — vốn được đo từ hông. Kết quả là vector 1503 chiều mang ý nghĩa hình học không nhất quán. Phân tách chuẩn hóa đảm bảo mỗi chiều trong vector cuối đều có \\\*\\\*nghĩa vật lý rõ ràng và nhất quán\\\*\\\* trong không gian metric của vùng cơ thể tương ứng.

#### Bước 3 · Xây dựng Ma trận Vận tốc Có Hướng (Directional Velocity Matrix)

Thay vì dùng hiệu bình phương element-wise (làm mất thông tin phương hướng và gây Vanishing Gradient với chuyển động nhỏ), hệ thống tính **sai phân bậc nhất có hướng** kết hợp với hằng số khuếch đại $K$ để giữ nguyên khả năng phân biệt trái/phải, lên/xuống và tránh tín hiệu bị dập tắt:

$$
\\mathbf{v}(t) = \\begin{cases}
\\mathbf{0}*{1503} \& t = 1 \\quad \\text{(khởi tạo zero, không có thông tin giả tạo)} \\\[6pt]
K \\cdot \\left(\\mathbf{pos}*{\\text{norm}}(t) - \\mathbf{pos}\_{\\text{norm}}(t-1)\\right) \& t \\geq 2
\\end{cases}
$$

Trong đó $K = \\text{FPS} = 30$ — nhân với tốc độ khung hình để chuyển đổi đơn vị sai phân sang **vận tốc vật lý** (đơn vị: đơn vị tọa độ / giây), đồng thời khuếch đại biên độ về dải số hợp lý cho hàm kích hoạt mạng nơ-ron (tránh Vanishing Gradient).

> \\\*\\\*Lý do không dùng hiệu bình phương:\\\*\\\* Phép bình phương element-wise $({\\\\Delta \\\\mathbf{p}})^2$ triệt tiêu hoàn toàn thông tin \\\*\\\*phương hướng\\\*\\\* (dấu âm/dương) — mất đi tín hiệu quan trọng như "tay giơ lên" hay "đầu cúi xuống". Ngoài ra, với chuyển động nhỏ $|\\\\Delta p\\\_j| \\\\ll 1$, bình phương tạo ra giá trị cực nhỏ $(|\\\\Delta p\\\_j|^2 \\\\approx 0)$ gây \\\*\\\*Vanishing Gradient\\\*\\\* nghiêm trọng trong quá trình huấn luyện. Sai phân × $K$ bảo toàn cả dấu lẫn biên độ, là đại lượng metric tuyến tính chuẩn trong cơ học.

Ma trận vận tốc có hướng:

$$
\\mathbf{V} = \\begin{bmatrix} \\mathbf{v}(1)^\\top \\ \\mathbf{v}(2)^\\top \\ \\vdots \\ \\mathbf{v}(F)^\\top \\end{bmatrix} \\in \\mathbb{R}^{F \\times 1503}
$$

Mỗi hàng $\\mathbf{v}(t)$ biểu diễn **vận tốc tương đối có hướng** của 501 điểm (33 Pose + 468 Face) từ khung $t-1$ sang khung $t$, trong hệ quy chiếu đã tịnh tiến về gốc nội tại của từng cụm.

**Nhánh phụ tùy chọn — Năng lượng Cục bộ L2 (501 chiều):**

Nếu muốn một biểu diễn vô hướng bổ sung về **mức độ kích động tổng thể** (đặc biệt hữu ích cho Hướng ngoại E và Nhiễu loạn thần kinh N), tính chuẩn khoảng cách Euclidean cho từng điểm $i$ trong số 501 điểm:

$$
E\_i(t) = \\left|\\Delta\\mathbf{p}\_i(t)\\right|\_2 = \\sqrt{\\Delta x\_i^2 + \\Delta y\_i^2 + \\Delta z\_i^2}, \\qquad i = 1, \\ldots, 501
$$

trong đó $\\Delta\\mathbf{p}\_i(t) = \\mathbf{p}\_i^{\\text{norm}}(t) - \\mathbf{p}\_i^{\\text{norm}}(t-1)$ là vector dịch chuyển 3D của điểm thứ $i$.

Thu được ma trận năng lượng cục bộ (thay thế vector 1503 chiều bằng 501 chiều vô hướng):

$$
\\mathbf{E}(t) = \\left\[E\_1(t),, E\_2(t),, \\ldots,, E\_{501}(t)\\right]^\\top \\in \\mathbb{R}^{501}
$$

> Ưu điểm: triệt tiêu nhiễu định hướng, giảm chiều từ $1503$ xuống $501$, mỗi chiều mang ý nghĩa vật lý rõ ràng — "khớp $i$ di chuyển bao nhiêu trong một khung hình". Đặc biệt phù hợp để ghép nối với biên độ cường độ Audio Energy trong cơ chế PIMC Loss.

Trong pipeline chính, **sử dụng vector vận tốc có hướng $\\mathbf{v}(t) \\in \\mathbb{R}^{1503}$**. Vector năng lượng $\\mathbf{E}(t) \\in \\mathbb{R}^{501}$ được tính song song và lưu lại để phục vụ PIMC Loss (Giai đoạn 4.3).

#### Bước 4 · Mean Pooling theo trục thời gian

Vì $\\mathbf{v}(1) = \\mathbf{0}\_{1503}$ là vector khởi tạo (không mang thông tin chuyển động thực), phép pooling **bỏ qua frame đầu** và chỉ lấy trung bình trên $t = 2 \\ldots F$:

$$
\\bar{\\mathbf{v}} = \\begin{cases}
\\mathbf{0}*{1503} \& F = 1 \\quad \\text{(cửa sổ chỉ có 1 frame — không có sai phân)} \\\[6pt]
\\dfrac{1}{F - 1} \\displaystyle\\sum*{t=2}^{F} \\mathbf{v}(t) \& F \\geq 2
\\end{cases} \\quad \\in \\mathbb{R}^{1503}
$$

> \\\*\\\*Lý do:\\\*\\\* Nếu gộp $\\\\mathbf{v}(1) = \\\\mathbf{0}$ vào trung bình, với cửa sổ ngắn ($F = 3$–5 frame), frame zero chiếm $20$–33% trọng số, gây sai lệch đáng kể. Bỏ frame đầu đảm bảo $\\\\bar{\\\\mathbf{v}}$ phản ánh chuyển động thực tế.

**Nhánh phụ (nếu dùng):** $\\mathbf{E}(t)$ được tính từ $t = 2$ (sai phân thực sự), không có giá trị giả tạo tại $t = 1$, nên pooling bình thường:

$$
\\bar{\\mathbf{E}} = \\frac{1}{F - 1} \\sum\_{t=2}^{F} \\mathbf{E}(t) \\in \\mathbb{R}^{501} \\qquad (F \\geq 2)
$$

#### Kết quả cuối

$$
\\boxed{\\mathbf{V}\_{\\text{out}} \\in \\mathbb{R}^{a \\times 1503}}
$$

Hàng $i$ chứa vector vận tốc có hướng (đã khuếch đại $\\times K$) của tất cả 501 điểm xương khớp + khuôn mặt, trong hệ quy chiếu phân tách theo cụm, đại diện cho từ thứ $i$. **Luồng này bắt buộc đi qua GLU ở Giai đoạn 3 để nén từ $1503$ xuống $256$ chiều.**

\---

### 2.3 Luồng Ngôn ngữ (RoBERTa) {#23-luồng-ngôn-ngữ-roberta}

#### Bước 1 · Nạp từng cặp câu

Đưa lần lượt **từng cặp câu liên tiếp** vào **RoBERTa Base**. Nếu số câu lẻ, lần cuối chỉ đưa một câu. Cấu trúc đầu vào:

```
\\\[CLS] câu\\\_1 \\\[SEP] câu\\\_2 \\\[SEP]
```

> \\\*\\\*⚠️ Giới hạn độ dài:\\\*\\\* RoBERTa Base chấp nhận tối đa \\\*\\\*512 token\\\*\\\* (bao gồm `\\\[CLS]` và 2 `\\\[SEP]`). Nếu tổng token của hai câu vượt 510, tokenizer phải cắt bớt (`truncation=True, max\\\_length=512`). Khi bị cắt, ánh xạ `word\\\_ids()` không trả về ID cho các token bị loại — cần bẫy trường hợp `word\\\_id is None` hoặc thiếu so với danh sách từ Whisper. Khúyen nghị: log cảnh báo nếu xảy ra để xém xét chia nhỏ cặp câu.

#### Bước 2 · Tokenization (BPE)

RoBERTa sử dụng thuật toán **Byte-Pair Encoding (BPE)**. Một từ phức tạp bị băm thành nhiều sub-word token. Ký tự `Ġ` đánh dấu sự bắt đầu của một từ mới:

|Từ Whisper|Tokens RoBERTa|
|-|-|
|`playing`|`\\\[Ġplay, ing]`|
|`Thuyết trình`|`\\\[ĠThuy, ết, Ġtrình]`|

#### Bước 3 · Self-Attention (12 lớp)

Thuật toán duyệt qua tất cả token, tính trọng số chú ý theo ngữ cảnh để nắm bắt cấu trúc ngữ pháp, sự mỉa mai, và ý đồ câu nói.

#### Bước 4 · Last Hidden State

Đầu ra là ma trận $N \\times 768$ với $N$ = tổng số token (bao gồm `\\\[CLS]`, `\\\[SEP]`). Bỏ hàng `\\\[CLS]` và `\\\[SEP]`, giữ lại mảng giữa.

#### Bước 5 · Xử lý lệch Tokenization (Fast Tokenizer + word\_ids)

Gọi hàm `word\\\_ids()` từ **Fast Tokenizer** để nhận mảng ánh xạ sub-word → từ gốc. Các token cùng `word\\\_id` được **Mean Pooling** về thành một vector duy nhất đồng bộ với nhãn thời gian Whisper:

**Ví dụ:** câu `"Tôi thích playing"`:

```
Tokens:   \\\[ĠTôi,  Ġthích,  Ġplay,  ing]
word\\\_ids: \\\[  0,      1,       2,     2 ]
```

$$
\\mathbf{h}*{\\text{playing}} = \\frac{1}{2}\\left(\\mathbf{h}*{\\text{Ġplay}} + \\mathbf{h}\_{\\text{ing}}\\right)
$$

#### Kết quả cuối

$$
\\boxed{\\mathbf{T} \\in \\mathbb{R}^{a \\times 768}}
$$

Hàng $i$ là vector ngữ nghĩa của từ thứ $i$, đồng bộ hoàn toàn với nhãn thời gian Whisper.

\---

### Tổng kết Giai đoạn 2

|Luồng|Mô hình|Kích thước đầu ra|Ghi chú|
|-|-|-|-|
|Âm thanh|WavLM (Sentence-level chunks)|$a \\times 768$|Tránh OOM; không có pause|
|Hình ảnh|MediaPipe Holistic|$a \\times 1503$|Pose chuẩn hóa gốc Hip; Face chuẩn hóa gốc Nose Mesh; vận tốc có hướng × K; chưa qua GLU|
|Ngôn ngữ|RoBERTa Base|$a \\times 768$|Sub-word đã gộp về từ|

Điều kiện bất biến: với mọi $i \\in {1, \\ldots, a}$, **hàng thứ $i$ của cả 3 ma trận đều ứng với từ thứ $i$** theo nhãn thời gian Whisper.

> \\\*\\\*⚠️ Lỗ hổng baseline — Độ dài chuỗi khác nhau giữa các video:\\\*\\\* Mỗi video có số từ $a$ khác nhau (video 15s có thể có 30 từ, video khác có 80 từ). Khi ghép thành batch 8 video, các ma trận không có cùng chiều $a$ → cần \\\*\\\*padding và attention mask\\\*\\\*.

### Chiến lược Padding và Attention Mask cho Batch

**Bước 1 — Tìm độ dài chuỗi lớn nhất trong batch:**

$$
a\_{\\max} = \\max\_{i=1}^{8} a\_i
$$

**Bước 2 — Padding bằng vector zero:**

Với mỗi video $i$ có $a\_i < a\_{\\max}$, thêm $(a\_{\\max} - a\_i)$ hàng zero vào cuối cả 3 ma trận:

$$
\\mathbf{A}*i^{\\text{padded}} \\in \\mathbb{R}^{a*{\\max} \\times 768}, \\quad
\\mathbf{V}*i^{\\text{padded}} \\in \\mathbb{R}^{a*{\\max} \\times 1503}, \\quad
\\mathbf{T}*i^{\\text{padded}} \\in \\mathbb{R}^{a*{\\max} \\times 768}
$$

**Bước 3 — Tạo Attention Mask:**

$$
\\text{mask}\_i\[j] = \\begin{cases} 1 \& \\text{nếu } j < a\_i \\quad \\text{(vị trí thực)} \\ 0 \& \\text{nếu } j \\geq a\_i \\quad \\text{(vị trí padding)} \\end{cases}
$$

Ma trận mask cho batch: $\\mathbf{M}*{\\text{batch}} \\in {0, 1}^{8 \\times a*{\\max}}$.

**Bước 4 — Áp dụng mask trong Cross-Attention:**

Trước khi tính Softmax trong Cross-Attention, cộng thêm $-\\infty$ vào các vị trí padding:

$$
\\text{score}*{ij} \\leftarrow \\text{score}*{ij} + (1 - \\text{mask}\_j) \\times (-10^9)
$$

Điều này đảm bảo Softmax trả về weight $\\approx 0$ cho các vị trí padding — không có thông tin giả tạo từ zero vector ảnh hưởng đến dung hợp.

Tương tự, trong **Attention Pooling** (Giai đoạn 3.5), áp dụng mask vào vector điểm $\\mathbf{e}$ trước Softmax để chỉ các từ thực mới được tính trọng số.

\---



Ký hiệu toàn bộ giai đoạn này: $N = a$ (số từ, dùng $N$ cho nhất quán với ký hiệu Transformer).

### 3.1 Ánh xạ không gian — GLU Projection {#31-ánh-xạ-không-gian--glu-projection}

**Mục tiêu:** đưa cả 3 luồng về không gian $256$ chiều thống nhất, đồng thời **cấp quyền truy cập bình đẳng vào không gian phi tuyến** cho cả 3 phương thức thông qua màng lọc GLU.

> \\\*\\\*Công bằng Biểu diễn (Fair Representation):\\\*\\\* Để đảm bảo cả 3 luồng có cơ hội học biểu diễn phi tuyến như nhau, \\\*\\\*tất cả 3 luồng đều bắt buộc đi qua GLU\\\*\\\* — kể cả Luồng Hình ảnh. Không luồng nào được ưu tiên hay bỏ qua cơ chế lọc này.

Thay vì chiếu tuyến tính đơn thuần, dùng **Gated Linear Unit (GLU)** hoạt động như một "màng lọc thông minh": tự học đặc trưng nào hữu ích, đặc trưng nào là nhiễu. **Khởi tạo toàn bộ trọng số $\\mathbf{W}$ tại GLU bằng Xavier Uniform** để tránh lỗi phân phối (Covariance Shift) — phân phối đầu ra đầu vào được bảo toàn phương sai qua lớp tuyến tính.

\---

#### GLU — Luồng Âm thanh \& Luồng Ngôn ngữ ($768 \\to 256$)

Với ma trận đầu vào $\\mathbf{X} \\in \\mathbb{R}^{N \\times 768}$ (audio hoặc text):

> \\\*\\\*Lưu ý ký hiệu:\\\*\\\* Nánh Value dùng $\\\\mathbf{U}$ (không dùng $\\\\mathbf{V}$) để tránh xung đột với ma trận luồng hình ảnh $\\\\mathbf{V}\\\_{\\\\text{out}}$ và output Cross-Attention $\\\\mathbf{V}'$.

**Nhánh tạo giá trị (Value branch):**

$$
\\mathbf{U} = \\mathbf{X},\\mathbf{W}\_1 + \\mathbf{b}\_1, \\qquad \\mathbf{W}\_1 \\in \\mathbb{R}^{768 \\times 256},\\ \\mathbf{b}\_1 \\in \\mathbb{R}^{256},\\ \\mathbf{U} \\in \\mathbb{R}^{N \\times 256}
$$

**Nhánh cổng lọc (Gate branch):**

$$
\\mathbf{G} = \\sigma!\\left(\\mathbf{X},\\mathbf{W}\_2 + \\mathbf{b}\_2\\right), \\qquad \\mathbf{W}\_2 \\in \\mathbb{R}^{768 \\times 256},\\ \\mathbf{b}\_2 \\in \\mathbb{R}^{256},\\ \\mathbf{G} \\in \\mathbb{R}^{N \\times 256}
$$

Với $\\sigma$ là hàm sigmoid: $\\sigma(x) = \\frac{1}{1+e^{-x}}$, do đó $G\_{ij} \\in (0, 1)$.

**Kết quả (Hadamard product — tích từng phần tử):**

$$
\\mathbf{Y} = \\mathbf{U} \\odot \\mathbf{G} \\in \\mathbb{R}^{N \\times 256}
$$

Phần tử $Y\_{ij} = U\_{ij} \\cdot G\_{ij}$: cổng $G\_{ij}$ gần $1$ → giữ lại đặc trưng; gần $0$ → triệt tiêu.

Khởi tạo: $\\mathbf{W}\_1, \\mathbf{W}\_2$ dùng **Xavier Uniform**; $\\mathbf{b}\_2 = \\mathbf{0}$ để cổng trung lập ban đầu.

\---

#### GLU — Luồng Hình ảnh ($1503 \\to 256$)

Luồng Hình ảnh với $\\mathbf{V}\_{\\text{out}} \\in \\mathbb{R}^{N \\times 1503}$ **BẮT BUỘC** đi qua màng lọc GLU riêng để nén từ $1503$ xuống $256$ chiều:

**Nhánh tạo giá trị:**

$$
\\mathbf{V}*{\\text{vid}} = \\mathbf{V}*{\\text{out}},\\mathbf{W}*{v1} + \\mathbf{b}*{v1}, \\qquad \\mathbf{W}\_{v1} \\in \\mathbb{R}^{1503 \\times 256}
$$

**Nhánh cổng lọc:**

$$
\\mathbf{G}*{\\text{vid}} = \\sigma!\\left(\\mathbf{V}*{\\text{out}},\\mathbf{W}*{v2} + \\mathbf{b}*{v2}\\right), \\qquad \\mathbf{W}\_{v2} \\in \\mathbb{R}^{1503 \\times 256}
$$

**Kết quả:**

$$
\\mathbf{V}' = \\mathbf{V}*{\\text{vid}} \\odot \\mathbf{G}*{\\text{vid}} \\in \\mathbb{R}^{N \\times 256}
$$

Khởi tạo: $\\mathbf{W}*{v1}, \\mathbf{W}*{v2}$ dùng **Xavier Uniform** để tránh Covariance Shift — đặc biệt quan trọng vì số chiều đầu vào $1503$ lớn hơn nhiều so với Audio/Text; $\\mathbf{b}\_{v2} = \\mathbf{0}$ để cổng trung lập ban đầu.

\---

**Sau bước này, toàn bộ 3 luồng đều ở chiều $N \\times 256$:**

$$
\\mathbf{A}' \\in \\mathbb{R}^{N \\times 256}, \\quad \\mathbf{V}' \\in \\mathbb{R}^{N \\times 256}, \\quad \\mathbf{T}' \\in \\mathbb{R}^{N \\times 256}
$$

\---

### 3.2 Đồng bộ hóa thời gian — Positional Encoding {#32-đồng-bộ-hóa-thời-gian--positional-encoding}

Trước khi đưa vào Cross-Attention, mỗi vector được cộng thêm **mã hóa vị trí lượng giác (Sinusoidal PE)** để mô hình biết từ thứ $i$ nằm ở đâu trong chuỗi thời gian:

$$
\\text{PE}(\\text{pos},, 2i) = \\sin!\\left(\\frac{\\text{pos}}{10000^{2i/d\_{\\text{model}}}}\\right)
$$

$$
\\text{PE}(\\text{pos},, 2i+1) = \\cos!\\left(\\frac{\\text{pos}}{10000^{2i/d\_{\\text{model}}}}\\right)
$$

Trong đó:

* $\\text{pos} \\in {0, 1, \\ldots, N-1}$: vị trí của từ trong chuỗi.
* $d\_{\\text{model}} = 256$: số chiều vector đặc trưng.
* $i \\in {0, 1, \\ldots, 127}$: chỉ số chiều con.

Xây dựng ma trận $\\mathbf{PE} \\in \\mathbb{R}^{N \\times 256}$ rồi cộng trực tiếp vào cả 3 ma trận:

$$
\\mathbf{A}' \\leftarrow \\mathbf{A}' + \\mathbf{PE}, \\quad \\mathbf{V}' \\leftarrow \\mathbf{V}' + \\mathbf{PE}, \\quad \\mathbf{T}' \\leftarrow \\mathbf{T}' + \\mathbf{PE}
$$

\---

### 3.3 Chú ý chéo hai chiều đối xứng {#33-chú-ý-chéo-hai-chiều-đối-xứng}

Cơ chế **Symmetric Bidirectional Cross-Attention** chạy hai luồng song song thay vì neo một chiều.

Công thức tổng quát của một lớp Cross-Attention:

$$
\\text{CrossAttn}(\\mathbf{Q},, \\mathbf{K},, \\mathbf{V}) = \\text{Softmax}!\\left(\\frac{\\mathbf{Q},\\mathbf{K}^\\top}{\\sqrt{d\_k}}\\right)\\mathbf{V}
$$

với $d\_k = 256$ (chiều của key).

\---

#### Luồng 1 — Text-Anchored Stream (Text làm Query)

**Sub-luồng T→A (Text-to-Audio Fusion):**

$$\\mathbf{H}\_{TA} = \\text{CrossAttn}!\\left(\\mathbf{Q} = \\mathbf{T}',; \\mathbf{K} = \\mathbf{A}',; \\mathbf{V} = \\mathbf{A}'\\right) \\in \\mathbb{R}^{N \\times 256}$$

Tích $\\mathbf{Q}*{\\text{text}} \\cdot \\mathbf{K}*{\\text{audio}}^\\top$ tạo ma trận trọng số thể hiện: *"Với ngữ nghĩa này, âm điệu nào quyết định nhất?"* — khai thác sắc thái mỉa mai, do dự.

**Sub-luồng T→V (Text-to-Video Fusion):**

$$\\mathbf{H}\_{TV} = \\text{CrossAttn}!\\left(\\mathbf{Q} = \\mathbf{T}',; \\mathbf{K} = \\mathbf{V}',; \\mathbf{V} = \\mathbf{V}'\\right) \\in \\mathbb{R}^{N \\times 256}$$

Tìm kiếm sự tương đồng giữa từ ngữ và cử chỉ cơ thể của người nói.

\---

#### Luồng 2 — Audio-Anchored Stream (Audio làm Query)

**Sub-luồng A→T (Audio-to-Text Fusion):**

$$\\mathbf{H}\_{AT} = \\text{CrossAttn}!\\left(\\mathbf{Q} = \\mathbf{A}',; \\mathbf{K} = \\mathbf{T}',; \\mathbf{V} = \\mathbf{T}'\\right) \\in \\mathbb{R}^{N \\times 256}$$

Phát hiện sự khác biệt giữa giọng điệu và nội dung lời nói.

**Sub-luồng A→V (Audio-to-Video Fusion):**

$$\\mathbf{H}\_{AV} = \\text{CrossAttn}!\\left(\\mathbf{Q} = \\mathbf{A}',; \\mathbf{K} = \\mathbf{V}',; \\mathbf{V} = \\mathbf{V}'\\right) \\in \\mathbb{R}^{N \\times 256}$$

Đo lường sự đồng bộ giữa biến thiên giọng nói và nghiệm thể cơ thể (ví dụ: giọng run rẩy khớp với ngón tay run).

\---

#### Tổng hợp cục bộ 2 nhánh neo (Per-Anchor Branch Compression)

Thay vì gộp chung cả 4 luồng cross-attention, ta nén **cục bộ từng nhánh neo** về $256$ chiều bằng phép nối + chiếu tuyến tính:

**Nhánh neo Văn bản (Text-Anchored):** gộp 2 sub-luồng $\\mathbf{H}*{TA}$ và $\\mathbf{H}*{TV}$:

$$
\\mathbf{H}*{\\text{Text\_Anchor}} = \\text{Linear}!\\left(\\text{Concat}!\\left\[\\mathbf{H}*{TA},, \\mathbf{H}\_{TV}\\right]\\right) \\in \\mathbb{R}^{N \\times 256}
$$

trong đó $\\text{Concat}\[\\mathbf{H}*{TA}, \\mathbf{H}*{TV}] \\in \\mathbb{R}^{N \\times 512}$ và lớp Linear chiếu $512 \\to 256$.

**Nhánh neo Âm thanh (Audio-Anchored):** gộp 2 sub-luồng $\\mathbf{H}*{AT}$ và $\\mathbf{H}*{AV}$:

$$
\\mathbf{H}*{\\text{Audio\_Anchor}} = \\text{Linear}!\\left(\\text{Concat}!\\left\[\\mathbf{H}*{AT},, \\mathbf{H}\_{AV}\\right]\\right) \\in \\mathbb{R}^{N \\times 256}
$$

trong đó $\\text{Concat}\[\\mathbf{H}*{AT}, \\mathbf{H}*{AV}] \\in \\mathbb{R}^{N \\times 512}$ và lớp Linear chiếu $512 \\to 256$.

> \\\*\\\*Lý do:\\\*\\\* Mỗi nhánh neo giữ nguyên "góc nhìn" của mình (Text làm Query hoặc Audio làm Query), phép nén cục bộ bảo toàn cấu trúc ngữ nghĩa riêng trước khi đưa vào GMU để cân bằng.

\---

### 3.4 Cổng dung hợp thích ứng — GMU {#34-cổng-dung-hợp-thích-ứng--gmu}

**Mục tiêu:** Học một cổng tỷ lệ $\\mathbf{Z}$ để hệ thống tự phân bổ mức độ tin tưởng giữa **Nhánh neo Văn bản** ($\\mathbf{H}*{\\text{Text\_Anchor}}$) và **Nhánh neo Âm thanh** ($\\mathbf{H}*{\\text{Audio\_Anchor}}$). GMU đóng vai trò "trọng tài" cân bằng giữa hai góc nhìn.

**Bước 1 — Tính cổng tỷ lệ $\\mathbf{Z}$:**

$$
\\mathbf{Z} = \\sigma!\\left(\\mathbf{H}*{\\text{Text\_Anchor}},\\mathbf{W}*{T} + \\mathbf{H}*{\\text{Audio\_Anchor}},\\mathbf{W}*{A} + \\mathbf{b}\\right) \\in \\mathbb{R}^{N \\times 256}
$$

với $\\mathbf{W}*{T},\\ \\mathbf{W}*{A} \\in \\mathbb{R}^{256 \\times 256}$, $\\mathbf{b} \\in \\mathbb{R}^{256}$. Khởi tạo $\\mathbf{b} = \\mathbf{0}$ để $\\mathbf{Z} \\approx 0.5$ (trung lập) lúc đầu.

**Bước 2 — Nội suy tuyến tính (Linear Interpolation):**

$$
\\mathbf{H}*{\\text{final}} = \\mathbf{Z} \\odot \\mathbf{H}*{\\text{Text\_Anchor}} + (\\mathbf{1} - \\mathbf{Z}) \\odot \\mathbf{H}\_{\\text{Audio\_Anchor}} \\in \\mathbb{R}^{N \\times 256}
$$

Tại mỗi phần tử $(i, j)$: nếu $Z\_{ij} = 0.9$ → lấy $90%$ từ nhánh neo Văn bản và $10%$ từ nhánh neo Âm thanh. Khi giọng nói bị nhiễu (nhánh Audio-Anchored kém tin cậy), $Z\_{ij}$ tự tăng lên, hệ thống tự động chuyển sang tin tưởng nhánh Text-Anchored nhiều hơn — và ngược lại.

\---

### 3.5 Attention Pooling — Giảm $N \\times 256$ → $1 \\times 256$ {#35-attention-pooling--giảm-n--256--1--256}

**Mục tiêu:** tổng hợp thông tin toàn chuỗi $N$ từ về một vector $256$ chiều duy nhất, trong đó các từ quan trọng về mặt tâm lý được đóng góp nhiều hơn.

**Bước 1 — Chấm điểm từng từ:**

$$
\\mathbf{e} = \\mathbf{H}\_{\\text{final}},\\mathbf{w}\_a + b\_a \\in \\mathbb{R}^{N \\times 1}
$$

với $\\mathbf{w}\_a \\in \\mathbb{R}^{256 \\times 1}$, $b\_a \\in \\mathbb{R}$.

**Bước 2 — Áp dụng Mask rồi Softmax:**

Trước khi Softmax, cộng $-\\infty$ vào các vị trí padding để trọng số của chúng → 0:

$$
e_i \\leftarrow \\begin{cases} e_i & \\text{nếu } \\text{mask}_i = 1 \\\\ -10^9 & \\text{nếu } \\text{mask}_i = 0 \\end{cases}
$$

$$
\\boldsymbol{\\alpha} = \\text{Softmax}(\\mathbf{e}) \\in \\mathbb{R}^{N \\times 1}, \\qquad \\sum_{i:\\,\\text{mask}_i=1} \\alpha_i \\approx 1
$$

**Bước 3 — Tổng có trọng số (Weighted Sum):**

$$
\\mathbf{c} = \\sum\_{i=1}^{N} \\alpha\_i, \\mathbf{H}\_{\\text{final}}\[i,, :] \\in \\mathbb{R}^{1 \\times 256}
$$

Chiều $N$ bị triệt tiêu hoàn toàn; chỉ còn vector tổng hợp $\\mathbf{c} \\in \\mathbb{R}^{256}$.

\---

### 3.6 Mạng Đa nhiệm BigFive — Giảm $1 \\times 256$ → $1 \\times 5$ {#36-mạng-đa-nhiệm-bigfive--giảm-1--256--1--5}

Vector $\\mathbf{c} \\in \\mathbb{R}^{256}$ chứa toàn bộ "dấu vết tâm lý" của đoạn video. Mạng không cắt nhỏ vector này mà **nhân bản (clone)** thành 5 bản sao giống hệt, mỗi bản đi vào một đầu dự đoán độc lập. Toàn bộ 5 nhánh đều nhìn thấy bức tranh toàn cảnh 256 chiều.

> Ký hiệu: $k \\\\in \\\\{O, C, E, A, N\\\\}$.

**Bước 1 — Lớp ẩn (256 → 64):**

$$
\\mathbf{h}*k = \\text{ReLU}!\\left(\\mathbf{c},\\mathbf{W}^{(k)}*{\\text{hidden}} + \\mathbf{b}^{(k)}*{\\text{hidden}}\\right), \\qquad \\mathbf{W}^{(k)}*{\\text{hidden}} \\in \\mathbb{R}^{256 \\times 64},\\ \\mathbf{h}\_k \\in \\mathbb{R}^{64}
$$

Sau ReLU: **Dropout($p = 0.3$)** — ngẫu nhiên tắt $30%$ trong 64 đặc trưng, ép các nơ-ron phải hỗ trợ lẫn nhau, triệt tiêu học vẹt cục bộ.

Khởi tạo $\\mathbf{W}^{(k)}\_{\\text{hidden}}$ bằng **Kaiming Initialization** (bù đắp $\\approx 50%$ dữ liệu bị ReLU triệt tiêu).

**Bước 2 — Lớp xuất (64 → 1):**

$$
s\_k = \\mathbf{h}*k,\\mathbf{w}^{(k)}*{\\text{out}} + b^{(k)}*{\\text{out}} \\in \\mathbb{R}, \\qquad \\mathbf{w}^{(k)}*{\\text{out}} \\in \\mathbb{R}^{64 \\times 1}
$$

$s\_k$ là **logit thô** — có thể âm hoặc dương bất kỳ ($-2.5$ hoặc $4.1$, v.v.).

Khởi tạo $\\mathbf{w}^{(k)}\_{\\text{out}}$ bằng **Xavier (Glorot) Initialization** (lớp này đi qua Sigmoid).

**Bước 3 — Chuẩn hóa thang đo (Sigmoid):**

$$
\\hat{y}\_k = \\sigma(s\_k) = \\frac{1}{1 + e^{-s\_k}} \\in (0,, 1)
$$

Ép kết quả về khoảng $\[0, 1]$ khớp với thang điểm BigFive được chuẩn hóa.

**Bước 4 — Nối và xuất kết quả (Concatenation):**

$$
\\hat{\\mathbf{y}} = \\text{Concat}!\\left\[\\hat{y}\_O,, \\hat{y}\_C,, \\hat{y}\_E,, \\hat{y}\_A,, \\hat{y}\_N\\right] \\in \\mathbb{R}^{1 \\times 5}
$$

Đây là **vector đầu ra cuối cùng** của toàn bộ hệ thống, sẵn sàng đối chiếu với nhãn thực tế (Ground Truth) để tính hàm mất mát.

\---

## Giai đoạn 4 — Cải tiến Kiến trúc V3 {#giai-đoạn-4}

### 4.1 Hybrid Loss + Gradient Accumulation {#41-hybrid-loss--gradient-accumulation}

#### Gradient Accumulation

Giữ **Batch Size vật lý = 8** để tránh tràn RAM, nhưng cộng dồn gradient qua **16 bước** trước khi gọi `optimizer.step()`:

$$
\\text{Batch Size hiệu dụng} = 8 \\times 16 = 128
$$

Batch lớn giúp hàm CCC tính toán phương sai ổn định (CCC yêu cầu phương sai đủ lớn mới có ý nghĩa thống kê).

```python
accumulation\\\_steps = 16
optimizer.zero\\\_grad()

for step, batch in enumerate(dataloader):
    loss = criterion(model(batch), labels) / accumulation\\\_steps
    loss.backward()  # cộng dồn gradient, không xóa

    if (step + 1) % accumulation\\\_steps == 0:
        optimizer.step()       # cập nhật trọng số sau 16 bước
        optimizer.zero\\\_grad()  # xóa VRAM
```

#### Tham số Epsilon $\\varepsilon$ trong CCC

Công thức CCC chuẩn:

$$
\\rho\_c = \\frac{2,\\sigma\_{xy}}{\\sigma\_x^2 + \\sigma\_y^2 + (\\mu\_x - \\mu\_y)^2}
$$

Khi mẫu số gần về $0$ (phương sai nhỏ, hai phân phối giống nhau), kết quả tính toán trở thành `NaN`. Thêm $\\varepsilon = 10^{-8}$:

$$
\\rho\_c = \\frac{2,\\sigma\_{xy}}{\\sigma\_x^2 + \\sigma\_y^2 + (\\mu\_x - \\mu\_y)^2 + \\varepsilon}
$$

#### Hàm Hybrid Loss

Trong những epoch đầu, phương sai dự đoán gần $0$ khiến CCC mất ổn định. Dùng **MSE làm mỏ neo** kéo dự đoán về sát thực tế:

$$
L\_{\\text{CCC}} = 1 - \\rho\_c
$$

$$
\\boxed{L\_{\\text{final}} = \\lambda \\cdot L\_{\\text{CCC}} + (1 - \\lambda) \\cdot \\text{MSE}(\\hat{\\mathbf{y}},, \\mathbf{y})}
$$

với $\\lambda \\in \[0, 1]$ là trọng số. Ban đầu $\\lambda$ nhỏ (ưu tiên MSE), tăng dần về $1$ khi mô hình hội tụ (để CCC điều khiển hoàn toàn).

\---

#### Hàm `ccc_loss()` — Định nghĩa Python

> **⚠️ Blocker đã vá:** Hàm này bắt buộc phải định nghĩa trước khi chạy bất kỳ vòng lặp huấn luyện nào.

```python
import torch

def ccc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Concordance Correlation Coefficient Loss.
    Trả về scalar tensor = 1 - rho_c, trong khoảng [0, 2].
    pred, target: 1-D tensor cùng kích thước (batch_size,).
    """
    pred_mean   = pred.mean()
    target_mean = target.mean()
    pred_var    = pred.var(unbiased=False)
    target_var  = target.var(unbiased=False)
    covariance  = ((pred - pred_mean) * (target - target_mean)).mean()
    ccc = (2.0 * covariance) / (
        pred_var + target_var + (pred_mean - target_mean) ** 2 + eps
    )
    return 1.0 - ccc
```

---

### 4.2 LoRA — Tinh chỉnh tham số hiệu quả {#42-lora--tinh-chỉnh-tham-số-hiệu-quả}

Thay vì **Hard Freezing** toàn bộ WavLM và RoBERTa, hệ thống chèn thêm các ma trận **LoRA (Low-Rank Adaptation)** với $r = 16$ vào các lớp Self-Attention $(\\mathbf{W}\_Q, \\mathbf{W}\_K, \\mathbf{W}\_V)$ của cả hai mô hình.

**Cơ chế:**

$$
\\mathbf{W}' = \\mathbf{W}*{\\text{frozen}} + \\Delta\\mathbf{W} = \\mathbf{W}*{\\text{frozen}} + \\mathbf{B},\\mathbf{A}
$$

trong đó:

* $\\mathbf{W}\_{\\text{frozen}} \\in \\mathbb{R}^{d \\times d}$: trọng số gốc, **đóng băng hoàn toàn** (không cần gradient).
* $\\mathbf{A} \\in \\mathbb{R}^{r \\times d}$: ma trận hạng thấp A, **được huấn luyện**.
* $\\mathbf{B} \\in \\mathbb{R}^{d \\times r}$: ma trận hạng thấp B, **được huấn luyện**.
* $r = 16 \\ll d$: rank (ví dụ $d = 768$, rank = 16 ≈ 2% số tham số).

**Lợi ích:** Chỉ thêm $\\approx 1\\text{–}2%$ tham số tính toán nhưng cho phép mô hình thích nghi sâu (Domain Adaptation) vào các vi mô cảm xúc mà pre-trained weights không học được.

**MediaPipe** không có trọng số Self-Attention cần điều chỉnh — LoRA **không áp dụng** cho luồng hình ảnh.

\---

### 4.3 Mạng Đa nhiệm Động + PIMC Loss *(Phase 2 — Không dùng cho Baseline)* {#43-mạng-đa-nhiệm-động--pimc-loss-phase-2}

> \\\*\\\*⚠️ Ghi chú Baseline:\\\*\\\* Dynamic Task Weighting (GradNorm / Uncertainty) và PIMC Loss là các kỹ thuật nâng cao phức tạp, \\\*\\\*không phù hợp để triển khai trong baseline đầu tiên\\\*\\\*. Baseline sử dụng \\\*\\\*tổng có trọng số đồng đều\\\*\\\* của 5 nhánh Loss thay thế:
> $$L\\\_{\\\\text{baseline}} = \\\\frac{1}{5}\\\\sum\\\_{k \\\\in \\\\{O,C,E,A,N\\\\}} L\\\_k$$
> Kỹ thuật Phase 2 được tích hợp sau khi baseline đã hội tụ và CCC ổn định.

#### Dynamic Task Weighting (Uncertainty-weighted Loss)

Thay vì cộng trung bình đơn giản 5 nhánh Loss:

$$
L\_{\\text{total}}^{\\text{cũ}} = \\frac{1}{5}\\sum\_{k=1}^{5} L\_k
$$

Dùng **Uncertainty-weighted Loss** tự động phân bổ gradient:

$$
L\_{\\text{total}} = \\sum\_{k=1}^{5} \\frac{1}{2\\sigma\_k^2}, L\_k + \\log \\sigma\_k
$$

Trong đó $\\sigma\_k$ là độ không chắc chắn (uncertainty) của nhiệm vụ $k$ — được học như một tham số của mô hình. Khi nhánh $k$ khó hội tụ, $\\sigma\_k$ nhỏ, gradient của $L\_k$ được nhân lên $\\frac{1}{2\\sigma\_k^2}$ — tức được **bơm gradient lớn hơn** tự động.

#### PIMC Loss (Psychology-Informed Modality Correlation Loss)

Ép mô hình học theo quy luật tâm lý học: phạt nặng nếu mô hình dự đoán đúng nhưng vì lý do sai về phương thức thông tin (modality).

$$
L\_{\\text{PIMC}} = \\gamma \\cdot \\sum\_{(m,, t), \\in, \\mathcal{C}} \\max!\\left(0,, \\rho\_{\\min} - \\text{corr}(\\mathbf{a}\_m,, \\mathbf{a}\_t)\\right)
$$

Trong đó:

* $\\mathcal{C}$: tập các cặp ràng buộc tâm lý học, ví dụ:

  * `(Audio Energy, Hướng ngoại E)` — người hướng ngoại nói to, năng lượng âm thanh cao.
  * `(Text sentiment, Dễ chịu A)` — người dễ chịu dùng từ ngữ tích cực.
* $\\mathbf{a}\_m \\in \\mathbb{R}^N$: **vector trọng số Cross-Attention trung bình theo đầu (head-averaged)** được trích từ sub-luồng tương ứng — cụ thể:

  * Nếu $m = \\text{Audio}$: lấy attention weight từ sub-luồng **A→T** (ma trận $\\text{Softmax}(\\mathbf{Q}\_A \\mathbf{K}*T^\\top / \\sqrt{d})$, trung bình các hàng cho ra $\\mathbf{a}*{\\text{audio}} \\in \\mathbb{R}^N$).
  * Nếu $m = \\text{Text}$: lấy từ sub-luồng **T→A** tương tự.
  * $\\mathbf{a}\_t \\in \\mathbb{R}^N$: vector trọng số tương ứng cho phương thức mục tiêu $t$ (ví dụ **đầu ra scalar** của Attention Pooling cho nhánh $t$, được tính tương quan với $\\mathbf{a}\_m$).
* $\\text{corr}(\\cdot, \\cdot)$: hệ số tương quan Pearson.
* $\\rho\_{\\min}$: ngưỡng tương quan tối thiểu bắt buộc (ví dụ $0.3$).
* $\\gamma$: hệ số phạt.

**Hàm Loss tổng hợp cuối cùng:**

$$
\\boxed{L = \\sum\_{k=1}^{5} \\frac{1}{2\\sigma\_k^2},\\left\[\\lambda\_k \\cdot L\_{\\text{CCC},k} + (1 - \\lambda\_k) \\cdot \\text{MSE}*k\\right] + \\sum*{k=1}^{5}\\log\\sigma\_k + \\gamma, L\_{\\text{PIMC}}}
$$

\---

## Giai đoạn 5 — Khởi tạo \& Quy trình Huấn luyện {#giai-đoạn-5}

### 5.0 Class `BigFiveModel` — Skeleton PyTorch

> **⚠️ Blocker đã vá:** Toàn bộ vòng lặp huấn luyện gọi `model(batch)`. Class dưới đây gom tất cả các khối kiến trúc đã mô tả ở Giai đoạn 3 vào một `nn.Module` duy nhất.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BigFiveModel(nn.Module):
    """Pipeline BigFive V3.2 — Baseline."""

    def __init__(self, d: int = 256, dropout: float = 0.3):
        super().__init__()
        # ── 3.1 GLU Projections ───────────────────────────────────
        self.glu_a_val  = nn.Linear(768,  d); self.glu_a_gate = nn.Linear(768,  d)
        self.glu_t_val  = nn.Linear(768,  d); self.glu_t_gate = nn.Linear(768,  d)
        self.glu_v_val  = nn.Linear(1503, d); self.glu_v_gate = nn.Linear(1503, d)

        # ── 3.3 Cross-Attention projections (4 sub-streams × Q/K/V) ──
        self.ca = nn.ModuleDict({
            s: nn.ModuleDict({m: nn.Linear(d, d) for m in ("wq", "wk", "wv")})
            for s in ("ta", "tv", "at", "av")
        })

        # ── Per-anchor compression 512→256 ────────────────────────
        self.compress_text  = nn.Linear(d * 2, d)
        self.compress_audio = nn.Linear(d * 2, d)

        # ── 3.4 GMU ───────────────────────────────────────────────
        self.gmu_wt = nn.Linear(d, d)
        self.gmu_wa = nn.Linear(d, d)

        # ── 3.5 Attention Pooling ─────────────────────────────────
        self.pool_w = nn.Linear(d, 1)

        # ── 3.6 Multi-task Heads (O, C, E, A, N) ─────────────────
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 1), nn.Sigmoid()
            ) for _ in range(5)
        ])
        self._init_weights()

    # ── Helpers ───────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for head in self.heads:   # Kaiming cho lớp ẩn có ReLU
            nn.init.kaiming_uniform_(head[0].weight, nonlinearity="relu")

    @staticmethod
    def _glu(x, val_layer, gate_layer):
        return val_layer(x) * torch.sigmoid(gate_layer(x))

    @staticmethod
    def _sinusoidal_pe(N, d, device):
        pe  = torch.zeros(N, d, device=device)
        pos = torch.arange(N, device=device).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # (N, d)

    @staticmethod
    def _cross_attn(Q, K, V, mask=None):
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        return torch.bmm(F.softmax(scores, dim=-1), V)  # (B, N, d)

    # ── Forward ───────────────────────────────────────────────────
    def forward(self, batch: dict, device: str = "cuda"):
        """
        batch keys:
          'audio' : (B, N, 768)   — WavLM features đã padding
          'video' : (B, N, 1503)  — MediaPipe velocity đã padding
          'text'  : (B, N, 768)   — RoBERTa features đã padding
          'mask'  : (B, N)        — 1 = thực, 0 = padding
        Returns: (B, 5) — điểm BigFive trong (0, 1)
        """
        A    = batch["audio"].to(device)   # (B, N, 768)
        V    = batch["video"].to(device)   # (B, N, 1503)
        T    = batch["text"].to(device)    # (B, N, 768)
        mask = batch["mask"].to(device)    # (B, N)
        B, N, _ = A.shape

        # 3.1 GLU
        Ap = self._glu(A, self.glu_a_val, self.glu_a_gate)  # (B,N,256)
        Vp = self._glu(V, self.glu_v_val, self.glu_v_gate)
        Tp = self._glu(T, self.glu_t_val, self.glu_t_gate)

        # 3.2 Sinusoidal PE
        pe = self._sinusoidal_pe(N, Ap.size(-1), device)
        Ap = Ap + pe;  Vp = Vp + pe;  Tp = Tp + pe

        # 3.3 Symmetric Cross-Attention
        def ca(src, ctx, s):
            Q  = self.ca[s]["wq"](src)
            K  = self.ca[s]["wk"](ctx)
            Vv = self.ca[s]["wv"](ctx)
            return self._cross_attn(Q, K, Vv, mask)

        H_TA = ca(Tp, Ap, "ta");  H_TV = ca(Tp, Vp, "tv")
        H_AT = ca(Ap, Tp, "at");  H_AV = ca(Ap, Vp, "av")

        H_text  = self.compress_text( torch.cat([H_TA, H_TV], dim=-1))  # (B,N,256)
        H_audio = self.compress_audio(torch.cat([H_AT, H_AV], dim=-1))

        # 3.4 GMU
        Z       = torch.sigmoid(self.gmu_wt(H_text) + self.gmu_wa(H_audio))
        H_final = Z * H_text + (1 - Z) * H_audio                        # (B,N,256)

        # 3.5 Attention Pooling (có mask)
        e     = self.pool_w(H_final).squeeze(-1)         # (B, N)
        e     = e.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(e, dim=-1).unsqueeze(-1)       # (B, N, 1)
        ctx   = (alpha * H_final).sum(dim=1)             # (B, 256)

        # 3.6 Multi-task Heads
        scores = [head(ctx) for head in self.heads]      # 5 × (B, 1)
        return torch.cat(scores, dim=-1)                 # (B, 5)
```

---

### 5.1 Chiến lược Khởi tạo Trọng số

|Khối|Lớp|Chiến lược|Lý do|
|-|-|-|-|
|WavLM|Tất cả|Pre-trained + LoRA A/B|Tận dụng biểu diễn âm học phổ quát|
|RoBERTa|Tất cả|Pre-trained + LoRA A/B|Tận dụng biểu diễn ngữ nghĩa phổ quát|
|MediaPipe|—|Không có trọng số|Bộ điều khiển hình học cố định|
|GLU Video — Nhánh Value ($\\mathbf{W}\_{v1}$)|Linear|Xavier Uniform|Nén 1503→256; bảo toàn phương sai đầu vào lớn|
|GLU Video — Nhánh Gate ($\\mathbf{W}\_{v2}$)|Linear + Sigmoid|Xavier Uniform; $\\mathbf{b}\_{v2} = \\mathbf{0}$|Tránh Covariance Shift; cổng trung lập ban đầu|
|GLU Audio/Text — Nhánh Value ($\\mathbf{W}\_1$)|Linear|Xavier Uniform|Biến đổi tuyến tính, bảo toàn phương sai|
|GLU Audio/Text — Nhánh Gate ($\\mathbf{W}\_2$)|Linear + Sigmoid|Xavier Uniform; $\\mathbf{b}\_2 = \\mathbf{0}$|Sigmoid cần phương sai vừa đủ; b=0 → cổng trung lập ban đầu|
|Cross-Attention ($\\mathbf{W}\_Q, \\mathbf{W}\_K, \\mathbf{W}\_V$)|Linear|Xavier Uniform|Tích $\\mathbf{Q}\\mathbf{K}^\\top$ không bùng nổ trước Softmax|
|Nén nhánh neo ($512 \\to 256$)|Linear|Xavier Uniform|Chiếu nén cục bộ từng nhánh neo|
|GMU ($\\mathbf{W}*{T}, \\mathbf{W}*{A}$)|Linear|Xavier Uniform; $\\mathbf{b} = \\mathbf{0}$|$\\mathbf{Z} \\approx 0.5$ ban đầu — cân bằng hai nhánh neo|
|Multi-task Head — Lớp ẩn|Linear + ReLU|Kaiming Initialization|Bù đắp 50% bị ReLU triệt tiêu|
|Multi-task Head — Lớp xuất|Linear + Sigmoid|Xavier (Glorot)|Đi qua Sigmoid|
|$\\sigma\_k$ (uncertainty)|Scalar|Khởi tạo $= 1$|$\\frac{1}{2\\sigma\_k^2} = 0.5$ — bắt đầu cân bằng|

### 5.2 Quy trình Học tập (The Learning Cycle)

#### Bước 1 · Forward Pass

Chạy toàn bộ pipeline trên batch vật lý 8 video, thu được:

$$
\\hat{\\mathbf{Y}} \\in \\mathbb{R}^{8 \\times 5}
$$

#### Bước 2 · Tính Hybrid Loss có Dynamic Weighting

**Bước 2.1** — Tách ma trận $8 \\times 5$ thành 5 cột theo từng tính cách.

**Bước 2.2** — Với mỗi nhánh $k$, tính CCC Loss (có $\\varepsilon$):

$$
L\_{\\text{CCC},k} = 1 - \\frac{2,\\sigma\_{\\hat{y}*k y\_k}}{\\sigma*{\\hat{y}*k}^2 + \\sigma*{y\_k}^2 + (\\mu\_{\\hat{y}*k} - \\mu*{y\_k})^2 + \\varepsilon}
$$

**Bước 2.3** — Hybrid Loss của từng nhánh:

$$
L\_k = \\lambda\_k \\cdot L\_{\\text{CCC},k} + (1 - \\lambda\_k) \\cdot \\frac{1}{8}\\sum\_{i=1}^{8}(\\hat{y}*{k,i} - y*{k,i})^2
$$

**Bước 2.4** — Uncertainty-weighted Total Loss:

$$
L\_{\\text{total}} = \\sum\_{k=1}^{5}\\left(\\frac{1}{2\\sigma\_k^2}, L\_k + \\log\\sigma\_k\\right) + \\gamma, L\_{\\text{PIMC}}
$$

#### Bước 3 · Backward Pass (Gradient Accumulation)

```python
loss\\\_scaled = L\\\_total / 16   # chuẩn hóa theo số bước dồn
loss\\\_scaled.backward()        # cộng dồn gradient vào buffer

if (step + 1) % 16 == 0:
    optimizer.step()
    optimizer.zero\\\_grad()
```

Đạo hàm lan truyền ngược từ $L\_{\\text{total}}$ qua: Multi-task Heads → Attention Pooling → GMU → Cross-Attention → GLU → LoRA layers (trong WavLM, RoBERTa) → Chiếu tăng chiều (trong MediaPipe stream).

#### Bước 4 · Optimizer Step (AdamW)

$$
\\mathbf{W} \\leftarrow \\mathbf{W} - \\eta \\cdot \\hat{\\mathbf{m}} / (\\sqrt{\\hat{\\mathbf{v}}} + \\varepsilon) - \\eta, \\lambda\_{\\text{wd}}, \\mathbf{W}
$$

với $\\eta$ là learning rate, $\\hat{\\mathbf{m}}$ / $\\hat{\\mathbf{v}}$ là moment bậc 1 / bậc 2 hiệu chỉnh, $\\lambda\_{\\text{wd}}$ là hệ số Weight Decay (chống Overfitting).

#### Bước 4b · Learning Rate Scheduler (Warmup + Cosine Decay)

> \\\*\\\*⚠️ Lỗ hổng baseline — thiếu scheduler:\\\*\\\* Không có scheduler, learning rate cố định toàn bộ quá trình huấn luyện gây hai rủi ro: (1) epoch đầu learning rate quá cao → gradient bùng nổ trước khi mô hình ổn định; (2) epoch cuối learning rate vẫn cao → dao động xung quanh điểm tối ưu, không hội tụ sâu.

Chiến lược **Linear Warmup + Cosine Decay**:

$$
\\eta\_t = \\begin{cases}
\\eta\_{\\max} \\cdot \\dfrac{t}{T\_{\\text{warmup}}} \& t \\leq T\_{\\text{warmup}} \\\[8pt]
\\eta\_{\\min} + \\dfrac{1}{2}(\\eta\_{\\max} - \\eta\_{\\min})\\left(1 + \\cos\\dfrac{\\pi,(t - T\_{\\text{warmup}})}{T\_{\\text{total}} - T\_{\\text{warmup}}}\\right) \& t > T\_{\\text{warmup}}
\\end{cases}
$$

Tham số khuyến nghị cho baseline:

* $\\eta\_{\\max} = 2 \\times 10^{-4}$, $\\eta\_{\\min} = 1 \\times 10^{-6}$
* $T\_{\\text{warmup}} = 3$ epochs (warmup tuyến tính qua 3 epoch đầu)
* $T\_{\\text{total}} = 50$ epochs

```python
from torch.optim.lr\\\_scheduler import CosineAnnealingLR
from torch.optim.lr\\\_scheduler import LinearLR
from torch.optim.lr\\\_scheduler import SequentialLR

warmup = LinearLR(optimizer, start\\\_factor=0.01, end\\\_factor=1.0, total\\\_iters=3)
cosine = CosineAnnealingLR(optimizer, T\\\_max=47, eta\\\_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=\\\[warmup, cosine], milestones=\\\[3])
```

#### Bước 4c · Lịch trình λ cho Hybrid Loss (Lambda Schedule)

> \\\*\\\*⚠️ Lỗ hổng baseline — λ không có lịch trình cụ thể:\\\*\\\* Pipeline mô tả λ tăng dần nhưng không định nghĩa hàm tăng → không thể tái lập.

Dùng lịch trình **tuyến tính theo epoch**:

$$
\\lambda\_e = \\lambda\_{\\text{init}} + \\frac{e}{E\_{\\text{total}}} \\cdot (\\lambda\_{\\text{final}} - \\lambda\_{\\text{init}})
$$

Với $\\lambda\_{\\text{init}} = 0.3$, $\\lambda\_{\\text{final}} = 0.95$, $E\_{\\text{total}} = 50$:

```python
lambda\\\_ccc = 0.3 + (epoch / 50) \\\* (0.95 - 0.3)   # tăng tuyến tính
L\\\_k = lambda\\\_ccc \\\* L\\\_CCC\\\_k + (1 - lambda\\\_ccc) \\\* MSE\\\_k
```

#### Hàm `evaluate_val()` — Định nghĩa Python

> **⚠️ Blocker đã vá:** Hàm này được gọi cuối mỗi epoch để tính CCC trên tập Val và quyết định Early Stopping.

```python
def evaluate_val(model: nn.Module, val_loader, device: str = "cuda") -> dict:
    """Tính CCC trên tập Val cho từng tính cách.
    Returns: dict {'O': float, 'C': float, 'E': float, 'A': float, 'N': float}
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            preds = model(batch, device=device)   # (B, 5)
            all_preds.append(preds.cpu())
            all_labels.append(batch["labels"].cpu())

    preds  = torch.cat(all_preds,  dim=0)  # (N_val, 5)
    labels = torch.cat(all_labels, dim=0)  # (N_val, 5)

    result = {}
    for i, trait in enumerate(TRAITS):
        ccc_val = 1.0 - ccc_loss(preds[:, i], labels[:, i]).item()
        result[trait] = float(ccc_val)
    return result
```

---

#### Bước 5 · Ghi file trọng số \& Early Stopping

Sau mỗi epoch, **tính $\\overline{\\text{CCC}}\_{\\text{val}}$ trên tập Val**. Nếu là cao nhất từ trước đến nay, ghi file:

```python
val\\\_ccc = evaluate\\\_val(model, val\\\_loader)   # trả về mean CCC trên 5 traits

if val\\\_ccc > best\\\_val\\\_ccc:
    best\\\_val\\\_ccc = val\\\_ccc
    torch.save(model.state\\\_dict(), f"{run\\\_name}\\\_best.pth")
    patience\\\_counter = 0
else:
    patience\\\_counter += 1
    if patience\\\_counter >= patience:      # patience = 10 epochs
        print(f"Early stopping tại epoch {epoch}")
        break
```

> \\\*\\\*Early Stopping:\\\*\\\* Nếu $\\\\overline{\\\\text{CCC}}\\\_{\\\\text{val}}$ không cải thiện trong \\\*\\\*10 epoch liên tiếp\\\*\\\*, dừng huấn luyện và tải lại checkpoint tốt nhất. Điều này tránh Overfitting và tiết kiệm thời gian tính toán.

Tên file `.pth` phản ánh thư mục train đã dùng:

|Thư mục train|File khởi tạo|Tên file .pth|
|-|-|-|
|`train-1, train-2, train-3`|*(không có)*|`123\\\_best.pth`|
|`train-1, train-2, train-3`|`789.pth`|`123\\\_789\\\_best.pth`|

### 5.3 File cấu hình Huấn luyện

```json
{
  "batch\\\_size\\\_physical": 8,
  "accumulation\\\_steps": 16,
  "effective\\\_batch\\\_size": 128,
  "epochs": 50,
  "learning\\\_rate\\\_max": 2e-4,
  "learning\\\_rate\\\_min": 1e-6,
  "warmup\\\_epochs": 3,
  "weight\\\_decay": 1e-2,
  "lambda\\\_hybrid\\\_init": 0.3,
  "lambda\\\_hybrid\\\_final": 0.95,
  "lambda\\\_schedule": "linear",
  "lora\\\_rank": 16,
  "epsilon\\\_ccc": 1e-8,
  "dropout\\\_heads": 0.3,
  "early\\\_stopping\\\_patience": 10,
  "random\\\_seed": 42,
  "mediapipe\\\_failure\\\_threshold": 0.30,
  "velocity\\\_scale\\\_K": 30,
  "train\\\_dirs": \\\[1, 2, 3],
  "init\\\_weights": "789.pth",
  "phase2\\\_dynamic\\\_weighting": false,
  "phase2\\\_pimc\\\_loss": false
}
```

### 5.4 Giám sát Huấn luyện (TensorBoard)

Ghi sau mỗi batch (trước backward):

* **6 đường Loss:** $L\_O$, $L\_C$, $L\_E$, $L\_A$, $L\_N$, $L\_{\\text{total}}$.
* **5 giá trị uncertainty** $\\sigma\_k$: quan sát nhánh nào đang được ưu tiên gradient.
* **Gradient norm** của mỗi nhóm tham số: phát hiện Vanishing / Exploding Gradient sớm.

\---

## Giai đoạn 6 — Theo dõi \& Trực quan hóa CCC Loss {#giai-đoạn-6}

### 6.1 Chiến lược Ghi Log theo Batch

Mỗi lần `optimizer.step()` được gọi (tức sau mỗi 16 bước gradient accumulation = 1 **effective step**), ghi lại:

* $L\_{\\text{CCC},k}$ cho cả 5 tính cách $k \\in {O, C, E, A, N}$
* $L\_{\\text{total}}$ (hybrid loss tổng hợp)
* $\\overline{\\text{CCC}}\_{\\text{val}}$ (tính cuối mỗi epoch, không phải mỗi batch)
* Learning rate hiện tại $\\eta\_t$

```python
log = {
    "step":      \\\[],     # effective step (sau accumulation)
    "epoch":     \\\[],
    "loss\\\_O":    \\\[], "loss\\\_C": \\\[], "loss\\\_E": \\\[],
    "loss\\\_A":    \\\[], "loss\\\_N": \\\[],
    "loss\\\_total": \\\[],
    "lr":        \\\[],
    # val (ghi cuối epoch, lặp lại giá trị cho toàn bộ step trong epoch đó)
    "val\\\_ccc\\\_O": \\\[], "val\\\_ccc\\\_C": \\\[], "val\\\_ccc\\\_E": \\\[],
    "val\\\_ccc\\\_A": \\\[], "val\\\_ccc\\\_N": \\\[], "val\\\_ccc\\\_mean": \\\[],
}
```

### 6.2 Vòng lặp Huấn luyện có Ghi Log

```python
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

TRAITS   = \\\["O", "C", "E", "A", "N"]
COLORS   = {"O": "#4C72B0", "C": "#DD8452", "E": "#55A868",
            "A": "#C44E52", "N": "#8172B2"}
PATIENCE = 10

log          = {k: \\\[] for k in \\\["step","epoch","loss\\\_total","lr",
                                 \\\*\\\[f"loss\\\_{t}" for t in TRAITS],
                                 \\\*\\\[f"val\\\_ccc\\\_{t}" for t in TRAITS], "val\\\_ccc\\\_mean"]}
best\\\_val\\\_ccc = -1.0
patience\\\_ctr = 0
global\\\_step  = 0

for epoch in range(1, 51):
    model.train()
    optimizer.zero\\\_grad()

    for local\\\_step, batch in enumerate(train\\\_loader):
        # ── Forward ────────────────────────────────────────────
        y\\\_pred = model(batch)           # (B, 5)
        y\\\_true = batch\\\["labels"]        # (B, 5)

        lambda\\\_ccc = 0.3 + (epoch / 50) \\\* 0.65   # tăng tuyến tính 0.3 → 0.95

        losses = {}
        for i, trait in enumerate(TRAITS):
            l\\\_ccc = ccc\\\_loss(y\\\_pred\\\[:, i], y\\\_true\\\[:, i])   # 1 - rho\\\_c
            l\\\_mse = F.mse\\\_loss(y\\\_pred\\\[:, i], y\\\_true\\\[:, i])
            losses\\\[trait] = lambda\\\_ccc \\\* l\\\_ccc + (1 - lambda\\\_ccc) \\\* l\\\_mse

        # Baseline: tổng đồng đều (Phase 2 thay bằng Uncertainty-weighted)
        L\\\_total = sum(losses.values()) / 5

        # ── Backward (gradient accumulation) ──────────────────
        (L\\\_total / 16).backward()

        if (local\\\_step + 1) % 16 == 0:
            optimizer.step()
            optimizer.zero\\\_grad()
            global\\\_step += 1

            # Ghi log theo effective step
            # scheduler.get\\\_last\\\_lr() trả về LR của epoch hiện tại
            # (chưa đổi bín trong epoch này vì scheduler.step() cuối epoch)
            log\\\["step"].append(global\\\_step)
            log\\\["epoch"].append(epoch)
            log\\\["loss\\\_total"].append(L\\\_total.item())
            log\\\["lr"].append(scheduler.get\\\_last\\\_lr()\\\[0])
            for t in TRAITS:
                log\\\[f"loss\\\_{t}"].append(losses\\\[t].item())
            # val được điền sau (placeholder)
            for t in TRAITS:
                log\\\[f"val\\\_ccc\\\_{t}"].append(None)
            log\\\["val\\\_ccc\\\_mean"].append(None)

    # ── Flush gradient còn dư (nếu len(dataloader) không chia hết cho 16) ──
    remainder = len(train\\\_loader) % 16
    if remainder != 0:
        optimizer.step()      # áp dụng gradient các batch cuối
        optimizer.zero\\\_grad()
        global\\\_step += 1
        log\\\["step"].append(global\\\_step)
        log\\\["epoch"].append(epoch)
        log\\\["loss\\\_total"].append(L\\\_total.item())
        log\\\["lr"].append(scheduler.get\\\_last\\\_lr()\\\[0])
        for t in TRAITS:
            log\\\[f"loss\\\_{t}"].append(losses\\\[t].item())
        for t in TRAITS:
            log\\\[f"val\\\_ccc\\\_{t}"].append(None)
        log\\\["val\\\_ccc\\\_mean"].append(None)

    # ── Validation cuối epoch ──────────────────────────────────
    scheduler.step()
    model.eval()
    val\\\_ccc\\\_per\\\_trait = evaluate\\\_val(model, val\\\_loader)   # dict {trait: ccc\\\_value}
    val\\\_mean = np.mean(\\\[val\\\_ccc\\\_per\\\_trait\\\[t] for t in TRAITS])  # giữ đúng thứ tự TRAITS

    # Cập nhật ngược các bước trong epoch vừa xong
    steps\\\_this\\\_epoch = len(\\\[s for s in log\\\["epoch"] if s == epoch])
    for idx in range(-steps\\\_this\\\_epoch, 0):
        for t in TRAITS:
            log\\\[f"val\\\_ccc\\\_{t}"]\\\[idx] = val\\\_ccc\\\_per\\\_trait\\\[t]
        log\\\["val\\\_ccc\\\_mean"]\\\[idx] = val\\\_mean

    # ── Early stopping ─────────────────────────────────────────
    if val\\\_mean > best\\\_val\\\_ccc:
        best\\\_val\\\_ccc = val\\\_mean
        torch.save(model.state\\\_dict(), "best\\\_model.pth")
        patience\\\_ctr = 0
    else:
        patience\\\_ctr += 1
        if patience\\\_ctr >= PATIENCE:
            print(f"Early stopping tại epoch {epoch} | best val CCC = {best\\\_val\\\_ccc:.4f}")
            break
```

### 6.3 Biểu đồ CCC Loss theo Batch (Matplotlib)

Sau khi huấn luyện xong (hoặc có thể vẽ real-time sau mỗi epoch), gọi hàm `plot\\\_ccc\\\_dashboard()`:

```python
def plot\\\_ccc\\\_dashboard(log: dict, save\\\_path: str = "ccc\\\_loss\\\_dashboard.png"):
    """
    Vẽ bảng điều khiển 3×2 gồm:
      \\\[0,0] Train Loss tổng (5 nhánh + total)
      \\\[0,1] Train CCC-Loss từng tính cách riêng
      \\\[1,0] Val CCC theo epoch — 5 tính cách
      \\\[1,1] Val CCC Mean vs Train Loss Total
      \\\[2,0] Learning Rate schedule
      \\\[2,1] Bảng điểm CCC cuối cùng (bar chart)
    """
    df = pd.DataFrame(log).dropna(subset=\\\["val\\\_ccc\\\_mean"])

    # Làm mịn đường train bằng rolling mean (cửa sổ 20 effective steps)
    W = 20
    for col in \\\[f"loss\\\_{t}" for t in TRAITS] + \\\["loss\\\_total"]:
        df\\\[f"{col}\\\_smooth"] = df\\\[col].rolling(W, min\\\_periods=1).mean()

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("CCC Loss Dashboard — BigFive Pipeline V3.2",
                 fontsize=15, fontweight="bold", y=1.01)

    steps  = df\\\["step"].values
    epochs = sorted(df\\\["epoch"].unique())

    # ─── \\\[0,0] Train Total Loss ─────────────────────────────────────────────
    ax = axes\\\[0, 0]
    ax.plot(steps, df\\\["loss\\\_total\\\_smooth"], color="#2d2d2d", lw=2, label="Total (smoothed)")
    ax.plot(steps, df\\\["loss\\\_total"],        color="#2d2d2d", lw=0.5, alpha=0.3)
    ax.set\\\_title("Train — Total Hybrid Loss")
    ax.set\\\_xlabel("Effective Step")
    ax.set\\\_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    \\\_add\\\_epoch\\\_lines(ax, df)

    # ─── \\\[0,1] Train CCC-Loss mỗi tính cách ────────────────────────────────
    ax = axes\\\[0, 1]
    for t in TRAITS:
        ax.plot(steps, df\\\[f"loss\\\_{t}\\\_smooth"], color=COLORS\\\[t], lw=1.8,
                label=t, alpha=0.9)
    ax.set\\\_title("Train — CCC Loss từng tính cách (smoothed)")
    ax.set\\\_xlabel("Effective Step")
    ax.set\\\_ylabel("L\\\_CCC = 1 − ρ\\\_c")
    ax.legend(title="Trait", fontsize=9)
    ax.grid(True, alpha=0.3)
    \\\_add\\\_epoch\\\_lines(ax, df)

    # ─── \\\[1,0] Val CCC theo epoch ───────────────────────────────────────────
    ax = axes\\\[1, 0]
    # Lấy giá trị val duy nhất theo epoch (cuối epoch)
    val\\\_df = df.groupby("epoch").last().reset\\\_index()
    for t in TRAITS:
        ax.plot(val\\\_df\\\["epoch"], val\\\_df\\\[f"val\\\_ccc\\\_{t}"], color=COLORS\\\[t],
                lw=2, marker="o", markersize=4, label=t)
    ax.set\\\_title("Validation — CCC từng tính cách theo Epoch")
    ax.set\\\_xlabel("Epoch")
    ax.set\\\_ylabel("CCC (ρ\\\_c) ↑ tốt hơn")
    ax.legend(title="Trait", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set\\\_xlim(left=1)

    # ─── \\\[1,1] Val CCC Mean vs Train Loss Total ─────────────────────────────
    ax = axes\\\[1, 1]
    ax2 = ax.twinx()
    val\\\_df\\\_e = df.groupby("epoch").last().reset\\\_index()
    ax.plot(val\\\_df\\\_e\\\["epoch"], val\\\_df\\\_e\\\["val\\\_ccc\\\_mean"],
            color="#1a7abf", lw=2.5, marker="D", markersize=5, label="Val CCC Mean")
    ax2.plot(val\\\_df\\\_e\\\["epoch"], val\\\_df\\\_e\\\["loss\\\_total\\\_smooth"],
             color="#d62728", lw=1.5, linestyle="--", alpha=0.7, label="Train Loss (epoch mean)")
    ax.set\\\_title("Val CCC Mean vs. Train Loss")
    ax.set\\\_xlabel("Epoch")
    ax.set\\\_ylabel("Val CCC Mean ↑", color="#1a7abf")
    ax2.set\\\_ylabel("Train Loss ↓", color="#d62728")
    ax.tick\\\_params(axis="y", labelcolor="#1a7abf")
    ax2.tick\\\_params(axis="y", labelcolor="#d62728")
    lines1, labels1 = ax.get\\\_legend\\\_handles\\\_labels()
    lines2, labels2 = ax2.get\\\_legend\\\_handles\\\_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # ─── \\\[2,0] Learning Rate Schedule ───────────────────────────────────────
    ax = axes\\\[2, 0]
    ax.plot(steps, df\\\["lr"], color="#8c564b", lw=2)
    ax.set\\\_title("Learning Rate Schedule (Warmup + Cosine Decay)")
    ax.set\\\_xlabel("Effective Step")
    ax.set\\\_ylabel("η (Learning Rate)")
    ax.yaxis.set\\\_major\\\_formatter(ticker.FormatStrFormatter("%.2e"))
    ax.grid(True, alpha=0.3)
    \\\_add\\\_epoch\\\_lines(ax, df)

    # ─── \\\[2,1] Bar chart CCC cuối cùng ──────────────────────────────────────
    ax = axes\\\[2, 1]
    final\\\_val = val\\\_df\\\_e.iloc\\\[-1]
    ccc\\\_values = \\\[final\\\_val\\\[f"val\\\_ccc\\\_{t}"] for t in TRAITS]
    bars = ax.bar(TRAITS, ccc\\\_values,
                  color=\\\[COLORS\\\[t] for t in TRAITS], edgecolor="white", width=0.55)
    for bar, val in zip(bars, ccc\\\_values):
        ax.text(bar.get\\\_x() + bar.get\\\_width() / 2, bar.get\\\_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=final\\\_val\\\["val\\\_ccc\\\_mean"], color="black", linestyle="--",
               lw=1.5, label=f"Mean = {final\\\_val\\\['val\\\_ccc\\\_mean']:.3f}")
    ax.set\\\_title(f"Val CCC cuối — Epoch {int(final\\\_val\\\['epoch'])}")
    ax.set\\\_xlabel("BigFive Trait")
    ax.set\\\_ylabel("CCC (ρ\\\_c)")
    ax.set\\\_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight\\\_layout()
    plt.savefig(save\\\_path, dpi=150, bbox\\\_inches="tight")
    plt.show()
    print(f"Đã lưu biểu đồ: {save\\\_path}")


def \\\_add\\\_epoch\\\_lines(ax, df, alpha=0.15):
    """Vẽ đường dọc mờ tại ranh giới mỗi epoch."""
    epoch\\\_starts = df.groupby("epoch")\\\["step"].first()
    for step in epoch\\\_starts.values\\\[1:]:
        ax.axvline(x=step, color="gray", linewidth=0.8, alpha=alpha)
```

### 6.4 Kết quả Kỳ vọng của Biểu đồ

|Ô|Kỳ vọng nếu training đúng|Dấu hiệu cần xem xét|
|-|-|-|
|**\[0,0] Train Total Loss**|Giảm đều, có thể dao động nhẹ ở đầu (MSE cao do warmup)|Tăng đột ngột → learning rate quá cao; phẳng từ sớm → underfitting|
|**\[0,1] CCC Loss từng trait**|5 đường cùng giảm; N và C thường chậm hơn O và E|Một đường đứng yên hoặc tăng → nhánh đó cần điều chỉnh|
|**\[1,0] Val CCC theo epoch**|Tăng dần và plateau → dừng ở early stopping|Giảm sau khi tăng → overfitting; cần tăng dropout hoặc weight decay|
|**\[1,1] Val Mean vs Train Loss**|Hai đường song hành ban đầu, sau tách dần khi overfitting|Khoảng cách lớn giữa train loss thấp và val CCC thấp → mô hình quá phức tạp|
|**\[2,0] Learning Rate**|Tăng tuyến tính 3 epoch, giảm cosine đều đặn|—|
|**\[2,1] Bar CCC cuối**|O và E thường cao nhất ($>0.5$); N thấp nhất|Tất cả $< 0.3$ → pipeline có lỗi nghiêm trọng cần debug|





Video (.mp4)
    │
    ├─ FFmpeg ──────────────────────────────────────────────────────────────────┐
    │                                                                           │
    ▼                                                                           ▼
.wav (16kHz)                                                            Frames (30 FPS)
    │                                                                           │
    ├─ Whisper ASR ─────────────────┐                                           │
    │  (word timestamps)            │                                           │
    │                               ▼                                           │
    │                   Contextual Padding (ngưỡng 2p)                         │
    │                               │                                           │
    ▼                               │                                           ▼
WavLM + LoRA(r=16)          RoBERTa + LoRA(r=16)               MediaPipe Holistic
    │                               │                          ┌────────────────┤
    │  \\\[T\\\_s × 768] per sentence     \\\[N × 768] (BPE → word)    │                │
    │  chunk → concat               │                   33 Pose pts      468 Face pts
    │                               │                   Hip-centric      Nose#1-centric
    │  Mean Pool               Mean Pool                (pos\\\_pose)       (pos\\\_face)
    │                               │                          └────────┬───────┘
    ▼                               ▼                                   │ concat
a × 768                         a × 768              pos\\\_norm(t) ∈ ℝ¹⁵⁰³
                                                      v(t) = K·(pos\\\_norm(t)−pos\\\_norm(t−1))
                                                      v(1) = 0; K = FPS = 30
                                                      ─────────────────────
                                                       \\\[nhánh phụ] E\\\_i(t) = ‖Δp\\\_i‖₂ ∈ ℝ⁵⁰¹
                                                        → dùng cho PIMC Loss
                                                      Mean Pool theo trục thời gian
                                                                │
    │                               │                           ▼
    │                               │                       a × 1503
    │                               │                           │
    └──── GLU (768→256, Xavier) ────┘              GLU (1503→256, Xavier)
                 │                                                     │
                 ▼                                                     ▼
             a × 256                                              a × 256
                 │                                                     │
                 └─────────────── + PE (Sinusoidal) ─────────────────┘
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
              Text-Anchored Stream           Audio-Anchored Stream
              T→A (N×256), T→V (N×256)       A→T (N×256), A→V (N×256)
                          │                               │
              Linear(Concat\\\[H\\\_TA,H\\\_TV])     Linear(Concat\\\[H\\\_AT,H\\\_AV])
                   512→256                         512→256
                          │                               │
                          ▼                               ▼
                H\\\_Text\\\_Anchor (N×256)       H\\\_Audio\\\_Anchor (N×256)
                          │                               │
                          └───── GMU (Gated Interp.) ─────┘
                                          │
                                      N × 256
                                          │
                               Attention Pooling (N→1)
                                          │
                                      1 × 256
                                          │
                      ┌────┬────┬────┬────┴────┐
                      ▼    ▼    ▼    ▼         ▼
                     \\\[O]  \\\[C]  \\\[E]  \\\[A]       \\\[N]   ← 5 heads độc lập
                      │    │    │    │         │
                   256→64→1 (ReLU + Dropout + Sigmoid)
                      │    │    │    │         │
                      └────┴────┴────┴─────────┘
                                  │
                             1 × 5  (BigFive scores)
                                  │
                    \\\[Baseline] L = (1/5)·Σ L\\\_k  (đồng đều 5 nhánh)
                    \\\[Phase 2]  Uncertainty-weighted Loss + PIMC Loss
                    (PIMC dùng E\\\_i(t) ∈ ℝ⁵⁰¹ từ nhánh phụ)
```

\---

*Phiên bản V3.2 — Baseline-complete. Bổ sung: (0) Giai đoạn 0 — Dữ liệu, nhãn, phân chia Train/Val/Test 70/15/15, evaluation protocol CCC; (1) Contextual Padding sửa ngưỡng $d \\geq 2p$; (2) Chuẩn hóa hệ quy chiếu phân tách — Pose gốc Hip Midpoint, Face Mesh gốc Nose #1; (3) Sai phân có hướng × K=30 thay hiệu bình phương; (4) MediaPipe failure fallback — carry-forward + ngưỡng loại 30%; (5) Padding/Attention Mask cho batch độ dài khác nhau; (6) LR Scheduler Warmup+Cosine; (7) Lambda schedule tuyến tính; (8) Early Stopping patience=10; (9) Giai đoạn 6 — CCC Loss Dashboard 3×2 (matplotlib) ; (10) PIMC Loss và Dynamic Weighting đánh dấu Phase 2 — không dùng cho baseline.*

