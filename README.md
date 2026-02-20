# 🌟 EdenTTS with Qwen3 Audio Tokenizer

Dự án này là một phiên bản cải tiến sâu rộng của mô hình **EdenTTS** (một mô hình Non-Autoregressive Text-to-Speech). Thay vì dự đoán các đặc trưng Mel-spectrogram liên tục như bản gốc, mô hình này đã được thiết kế lại toàn bộ phần Acoustic Model để dự đoán các **Discrete Audio Tokens** (Audio Codebook rời rạc) sử dụng bộ Tokenizer 16 tầng RVQ (Residual Vector Quantization) của **Qwen3**, với kích thước từ điển là 2048 class.

Hướng tiếp cận này mang mô hình tiến gần hơn đến các kiến trúc sinh âm thanh State-of-the-Art (SOTA) hiện nay như VALL-E hay SoundStorm, kết hợp ưu điểm **sinh âm thanh song song tốc độ cao (NAR)** với **chất lượng âm thanh trung thực của Discrete Codebook**.

## ✨ Những Cải Tiến Kiến Trúc Nổi Bật

1. **Từ Liên Tục Sang Rời Rạc (Continuous to Discrete):** * Thay thế toàn bộ quá trình xử lý Mel-spectrogram bằng mảng Token IDs (Số nguyên từ 0 - 2047).
* Tích hợp `nn.Embedding` để biểu diễn các Token IDs thành không gian liên tục bên trong mạng học đặc trưng.


2. **Kiến Trúc Decoder Phân Cấp (Coarse-to-Fine Decoder):**
* Lấy cảm hứng từ **SoundStorm**, Decoder không dự đoán đồng thời 16 tầng (tránh hiện tượng Conditional Independence gây nhiễu).
* **Bước 1 (Coarse):** Dự đoán Layer 0 (Ngữ âm cốt lõi) từ Text.
* **Bước 2 (Fine):** Dự đoán Layer 1 đến 15 dựa trên Text và đặc trưng của Layer 0, đảm bảo tính đồng nhất phân cấp của bộ mã hóa RVQ.


3. **Alignment Khử Nhiễu (Denoised Attention):**
* Module `MelEncoder` tìm kiếm sự gióng hàng (Alignment) giữa Text và Audio chỉ dựa trên thông tin cốt lõi của **Layer 0**, loại bỏ hoàn toàn nhiễu từ các tầng RVQ sâu, giúp ma trận Attention hội tụ cực kỳ nhanh và ổn định.


4. **Weighted Cross-Entropy Loss:**
* Sử dụng hàm Loss CrossEntropy với trọng số giảm dần (`0.9 ** i`) cho 16 tầng RVQ, tập trung tối đa "sự chú ý" của gradient vào các tầng quyết định ngữ âm (Layer 0, 1, 2) và giảm bớt áp lực ở các tầng chứa tiếng ồn vi mô.



## 🚀 Cài Đặt (Installation)

Yêu cầu môi trường có cài đặt PyTorch và CUDA.

```bash
# Clone repository
git clone https://github.com/vandoan2k5/edentts-with-qwen3-tokenizer.git
cd edentts-with-qwen3-tokenizer

# Cài đặt các thư viện phụ thuộc (tuỳ chỉnh theo file requirements.txt của bạn)
pip install torch numpy librosa ...

```

*Lưu ý:* Cần có mã nguồn giải mã của **Qwen3 Audio Tokenizer** để chạy inference (chuyển đổi Token ID thành sóng âm Waveform).

## 🧠 Huấn Luyện (Training)

Mô hình hiện tại được thiết lập để huấn luyện bộ dữ liệu chuẩn (ví dụ: LJSpeech đã được lượng tử hóa thành Token IDs).

1. Đảm bảo bạn đã config đúng đường dẫn dữ liệu trong `config/eden.json`.
2. Chạy lệnh huấn luyện:

```bash
python train_cls.py

```

*Log huấn luyện theo dõi chỉ số `acc_L0` (Độ chính xác của Tầng 0). Khi `acc_L0` đạt mốc ổn định (thường từ 65% - 80%), mô hình đã sẵn sàng để inference.*

## 🔊 Sinh Âm Thanh (Inference)

Quá trình inference bao gồm 2 giai đoạn:

1. **Acoustic Model:** Sinh ra các mảng Token IDs từ văn bản.
2. **Vocoder (Qwen3):** Giải mã Token IDs thành file âm thanh `.wav`.

Sử dụng script inference đã được cung cấp:

```bash
python inference.py -t "This is a test sentence for discrete audio generation."

```

Đoạn mã sẽ tạo ra tệp `.npy` chứa mảng ID với shape `[1, Time, 16]`. Bạn cần truyền mảng này vào hàm `decode` của Qwen3 để nghe kết quả.

## 📁 Cấu Trúc Thư Mục Chính

* `models/components.py`: Chứa kiến trúc hạt nhân (TextEncoder, Coarse-to-Fine Decoder, DurationPredictor).
* `models/edenTTS.py`: Mô hình tổng, xử lý luồng forward và inference.
* `models/loss.py`: Định nghĩa hàm CrossEntropy theo trọng số tầng RVQ.
* `train_cls.py`: Vòng lặp huấn luyện chính (Classification thay vì Regression).
* `inference.py`: Script chạy thử nghiệm dự đoán Token IDs.

## 🙏 Lời Cảm Ơn (Acknowledgements)

* Kiến trúc cơ sở được xây dựng dựa trên [EdenTTS](https://www.google.com/search?q=%23) (edenmyn).
* Lấy cảm hứng từ các nghiên cứu âm thanh rời rạc như VALL-E (Microsoft), SoundStorm (Google).
* Sử dụng bộ Audio Tokenizer mạnh mẽ từ Qwen3.