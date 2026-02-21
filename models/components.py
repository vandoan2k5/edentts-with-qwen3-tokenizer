"""
@author: edenmyn
@email: edenmyn
@time: 2022/10/16 10:54
@DESC: 

"""
from .layers import *
import torch.nn as nn
from transformer.Layers import FFTBlock
from models.layers import TokenEmbedding
from transformer.Models import get_sinusoid_encoding_table
from collections import OrderedDict
from models.modules import Conv, Linear
import torch
import torch.nn as nn

class TextEncoder(torch.nn.Module):
    """
     this is the text encoder adapted from fastspeech
     we add make a small modification on the position embedding
    """
    def __init__(self, encoder_layer=5,
        encoder_head =2,
        encoder_hidden=256,
        conv_filter_size=1024,
        conv_kernel_size=[9, 1],
        encoder_dropout=0.2,
        n_channels=512,
        vocab_size=365):
        super().__init__()
        max_seq_len = 1000
        n_position = max_seq_len + 1
        d_word_vec = n_channels
        n_layers = encoder_layer
        n_head = encoder_head
        d_k = d_v = (
                encoder_hidden
                // encoder_head
        )
        d_model = encoder_hidden
        d_inner = conv_filter_size
        kernel_size = conv_kernel_size
        dropout = encoder_dropout

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.src_word_emb = TokenEmbedding(hidden_size=d_word_vec, padding_idx=0, vocab_size=vocab_size)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        self.pre_linear = torch.nn.Linear(in_features=n_channels, out_features=encoder_hidden)
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.linear_key = nn.Linear(encoder_hidden, n_channels)
        self.linear_value = nn.Linear(encoder_hidden, n_channels)

    def forward(self, src_seq, text_lengths, return_attns=False):
        mask = get_mask_from_lengths(text_lengths).to(src_seq.device)
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        # use abosolute positonal embedding, same as the original fastspeech FFT
        if hp.pos_embed_scheme == "absolute":
            # -- Forward
            if not self.training and src_seq.shape[1] > self.max_seq_len:
                enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                    src_seq.shape[1], self.d_model
                )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                    src_seq.device
                )
            else:
                enc_output = self.src_word_emb(src_seq) + self.position_enc[
                    :, :max_len, :
                ].expand(batch_size, -1, -1)
        else:
        # do not use the position embedding or use relative position embedding
        # leads to better performance especially for longer sentences
            enc_output = self.src_word_emb(src_seq)
        enc_output = self.pre_linear(enc_output)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        text_key = self.linear_key(enc_output)
        text_value = self.linear_value(enc_output)
        return text_key, text_value

    def inference(self, phone_ids: torch.Tensor):
        text_lens = torch.Tensor([phone_ids.size(1)]).long()
        text_key, text_value = self.forward(phone_ids, text_lens)
        return text_value


class MelEncoder(torch.nn.Module):
    def __init__(self, n_mels, n_channels, nonlinear_activation, nonlinear_activation_params,
                 dropout_rate, n_mel_encoder_layer, k_size, use_weight_norm,
                 dilations=None, vocab_size=2048, num_base_layers=4):
        super().__init__()
        self.num_base_layers = num_base_layers
        
        # 1. Khởi tạo 4 lớp Embedding độc lập cho 4 layer (0, 1, 2, 3)
        # Việc dùng embedding riêng giúp mô hình phân biệt được ý nghĩa của token ở layer 0 khác với layer 1
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(vocab_size, n_channels) for _ in range(num_base_layers)
        ])
        
        # 2. Thêm LayerNorm theo yêu cầu để ổn định phương sai sau khi cộng gộp
        self.layer_norm = torch.nn.LayerNorm(n_channels)
        
        # 3. Mạng Conv1D chính
        self.mel_encoder = ResConvBlock(
            num_layers=n_mel_encoder_layer, n_channels=n_channels, k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate, use_weight_norm=use_weight_norm, dilations=dilations
        )

    def forward(self, speech):
        # speech ban đầu có dạng [Batch, Time, 16]
        # Trích xuất 4 layer đầu tiên: [Batch, Time, 4]
        speech_base = speech[:, :, :self.num_base_layers] 
        
        # Tính embedding cho từng layer và cộng dồn lại (Summation)
        mel_h = 0
        for i in range(self.num_base_layers):
            # Lấy token của layer thứ i và nhúng nó
            emb_i = self.embeddings[i](speech_base[:, :, i]) # [Batch, Time, n_channels]
            mel_h = mel_h + emb_i
            
        # Đi qua LayerNorm để chuẩn hóa các giá trị sau khi cộng
        mel_h = self.layer_norm(mel_h)
        
        # Đổi chiều [Batch, Time, Channels] thành [Batch, Channels, Time] cho mạng Conv1D
        mel_h = mel_h.transpose(1, 2) 
        mel_h = self.mel_encoder(mel_h)
        
        # Trả về kích thước gốc [Batch, Time, Channels]
        return mel_h.transpose(1, 2)


class Decoder(torch.nn.Module):
    def __init__(self, idim, encoder_hidden, n_decoder_layer, k_size,
                 nonlinear_activation, nonlinear_activation_params,
                 dropout_rate, use_weight_norm, n_mels, vocab_size=2048, dialations=None):
        super().__init__()
        self.n_mels = n_mels
        self.vocab_size = vocab_size
        
        # --- BƯỚC 1: COARSE DECODER (Dự đoán Tầng 0) ---
        self.coarse_pre_linear = torch.nn.Linear(idim, encoder_hidden)
        self.coarse_decoder = ResConvBlock(
            num_layers=n_decoder_layer, n_channels=encoder_hidden, k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate, use_weight_norm=use_weight_norm, dilations=dialations
        )
        self.coarse_output = torch.nn.Linear(encoder_hidden, vocab_size)

        # --- BƯỚC 2: CASCADED FINE DECODER ---
        self.fine_pre_linear = torch.nn.Linear(idim, encoder_hidden)
        self.fine_decoder = ResConvBlock(
            num_layers=n_decoder_layer, n_channels=encoder_hidden, k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate, use_weight_norm=use_weight_norm, dilations=dialations
        )
        
        # MỘI TẦNG SỬ DỤNG EMBEDDING VÀ OUTPUT HEAD RIÊNG BIỆT
        self.layer_embs = nn.ModuleList([
            torch.nn.Embedding(vocab_size, encoder_hidden) for _ in range(n_mels)
        ])
        
        self.fine_outputs = nn.ModuleList([
            torch.nn.Linear(encoder_hidden, vocab_size) for _ in range(n_mels - 1)
        ])

    def forward(self, text_value_expanded, targets=None):
        # 1. Dự đoán Tầng 0 (Coarse)
        x_coarse = self.coarse_pre_linear(text_value_expanded)
        x_coarse = self.coarse_decoder(x_coarse.transpose(1, 2))
        logits_coarse = self.coarse_output(x_coarse.transpose(1, 2)) # [B, Time, 2048]
        
        logits_list = [logits_coarse.unsqueeze(2)]
        
        # 2. Dự đoán Tầng 1 đến 15 (Cascaded)
        # Khởi tạo hidden state ban đầu bằng Text Content
        current_hidden = self.fine_pre_linear(text_value_expanded)
        
        # Xác định token khởi nguồn (Tầng 0)
        if targets is not None:
            current_tokens = targets[:, :, 0] # Training: Dùng nhãn thật (Teacher Forcing)
        else:
            current_tokens = torch.argmax(logits_coarse, dim=-1) # Inference: Dùng token vừa sinh ra
            
        # 3. Chạy vòng lặp truyền trạng thái nối tiếp nhau
        for i in range(self.n_mels - 1): 
            # Bơm thông tin token của tầng hiện tại vào hidden state
            emb_i = self.layer_embs[i](current_tokens)
            current_hidden = current_hidden + emb_i 
            
            # Xử lý Conv Block
            h = self.fine_decoder(current_hidden.transpose(1, 2)).transpose(1, 2)
            
            # Dự đoán Tầng tiếp theo (i+1) thông qua Head riêng
            logits_i = self.fine_outputs[i](h)
            logits_list.append(logits_i.unsqueeze(2))
            
            # Chuẩn bị Token cho vòng lặp kế tiếp
            if targets is not None:
                current_tokens = targets[:, :, i+1]
            else:
                current_tokens = torch.argmax(logits_i, dim=-1)
                
        # Gộp tất cả 16 tầng lại
        logits = torch.cat(logits_list, dim=2) # Shape: [B, Time, 16, 2048]
        return logits




class DecoderV2(torch.nn.Module):
    def __init__(self, idim, encoder_hidden, n_decoder_layer, k_size,
                 nonlinear_activation, nonlinear_activation_params,
                 dropout_rate, use_weight_norm, n_mels, vocab_size=2048, dialations=None):
        super().__init__()
        self.n_mels = n_mels
        self.vocab_size = vocab_size
        
        # --- BƯỚC 1: COARSE DECODER (Dự đoán Tầng 0 - Ngữ nghĩa chính) ---
        self.coarse_pre_linear = torch.nn.Linear(idim, encoder_hidden)
        self.coarse_decoder = ResConvBlock(
            num_layers=n_decoder_layer, n_channels=encoder_hidden, k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate, use_weight_norm=use_weight_norm, dilations=dialations
        )
        self.coarse_output = torch.nn.Linear(encoder_hidden, vocab_size)

        # --- BƯỚC 2: CASCADED FINE DECODER (Dự đoán Tầng 1 -> 15) ---
        self.fine_pre_linear = torch.nn.Linear(idim, encoder_hidden)
        
        # TỐI ƯU: Fine Decoder nên mỏng hơn (VD: giảm một nửa số layer) vì nó phải chạy lặp 15 lần
        self.fine_decoder = ResConvBlock(
            num_layers=max(1, n_decoder_layer // 2), n_channels=encoder_hidden, k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate, use_weight_norm=use_weight_norm, dilations=dialations
        )
        
        # SỬA LỖI BUG: Chỉ cần 15 (n_mels - 1) Embeddings và Heads cho Tầng 1 đến 15
        self.layer_embs = nn.ModuleList([
            torch.nn.Embedding(vocab_size, encoder_hidden) for _ in range(n_mels - 1)
        ])
        
        self.fine_outputs = nn.ModuleList([
            torch.nn.Linear(encoder_hidden, vocab_size) for _ in range(n_mels - 1)
        ])

    def sample_tokens(self, logits, temperature=0.8, top_k=5):
        """Hàm phụ trợ: Lấy mẫu chống giật cục cho Inference"""
        logits = logits / temperature
        top_v, _ = torch.topk(logits, top_k, dim=-1)
        logits[logits < top_v[..., [-1]]] = -float('Inf')
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        shape = probs.shape
        probs_flat = probs.view(-1, shape[-1])
        sampled_flat = torch.multinomial(probs_flat, 1)
        return sampled_flat.view(*shape[:-1])

    def forward(self, text_value_expanded, targets=None, temperature=0.8, top_k=5, teacher_forcing_ratio=0.75):
        # 1. Dự đoán Tầng 0 (Coarse)
        x_coarse = self.coarse_pre_linear(text_value_expanded)
        x_coarse = self.coarse_decoder(x_coarse.transpose(1, 2))
        logits_coarse = self.coarse_output(x_coarse.transpose(1, 2)) # [B, Time, 2048]
        
        logits_list = [logits_coarse.unsqueeze(2)]
        
        # 2. Chuẩn bị cho Tầng 1 đến 15 (Cascaded)
        current_hidden = self.fine_pre_linear(text_value_expanded)
        
        # XÁC ĐỊNH TOKEN KHỞI NGUỒN (TẦNG 0)
        # Bổ sung Scheduled Sampling: Đôi khi tự lấy dự đoán của chính mình để huấn luyện
        if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
            current_tokens = targets[:, :, 0] # Dùng nhãn thật (Teacher Forcing)
        else:
            if targets is None: # Mode Inference
                current_tokens = self.sample_tokens(logits_coarse, temperature, top_k)
            else: # Mode Training nhưng cố tình cho model tự đoán (Scheduled Sampling)
                current_tokens = torch.argmax(logits_coarse, dim=-1).detach()
                
        # 3. Chạy vòng lặp truyền trạng thái nối tiếp nhau
        for i in range(self.n_mels - 1): 
            # Bơm thông tin token của tầng hiện tại vào hidden state
            emb_i = self.layer_embs[i](current_tokens)
            current_hidden = current_hidden + emb_i 
            
            # Xử lý Conv Block
            h = self.fine_decoder(current_hidden.transpose(1, 2)).transpose(1, 2)
            
            # Dự đoán Tầng tiếp theo (i+1) thông qua Head riêng
            logits_i = self.fine_outputs[i](h)
            logits_list.append(logits_i.unsqueeze(2))
            
            # Chuẩn bị Token cho vòng lặp kế tiếp (Trừ vòng lặp cuối cùng)
            if i < self.n_mels - 2:
                if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    current_tokens = targets[:, :, i+1]
                else:
                    if targets is None:
                        current_tokens = self.sample_tokens(logits_i, temperature, top_k)
                    else:
                        current_tokens = torch.argmax(logits_i, dim=-1).detach()
                
        # Gộp tất cả 16 tầng lại
        logits = torch.cat(logits_list, dim=2) # Shape: [B, Time, 16, 2048]
        return logits

class DecoderV3(torch.nn.Module):
    def __init__(self, idim, encoder_hidden, n_decoder_layer, k_size,
                 nonlinear_activation, nonlinear_activation_params,
                 dropout_rate, use_weight_norm, n_mels, vocab_size=2048, dialations=None):
        super().__init__()
        self.n_mels = n_mels
        self.vocab_size = vocab_size
        self.hidden_size = encoder_hidden
        
        # --- TẦNG 0: COARSE (Mỏ neo ngữ nghĩa) ---
        self.coarse_pre_linear = torch.nn.Linear(idim, encoder_hidden)
        self.coarse_decoder = ResConvBlock(
            num_layers=n_decoder_layer, n_channels=encoder_hidden, k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate, use_weight_norm=use_weight_norm, dilations=dialations
        )
        self.coarse_output = torch.nn.Linear(encoder_hidden, vocab_size)

        # --- TẦNG 1, 2, 3: DEDICATED BLOCKS (Tập trung giải quyết sụp đổ) ---
        # Mỗi tầng quan trọng có 1 block riêng biệt để học Acoustic Texture
        self.priority_decoders = nn.ModuleList([
            ResConvBlock(
                num_layers=n_decoder_layer, n_channels=encoder_hidden, k_size=k_size,
                nonlinear_activation=nonlinear_activation,
                nonlinear_activation_params=nonlinear_activation_params,
                dropout_rate=dropout_rate, use_weight_norm=use_weight_norm, dilations=dialations
            ) for _ in range(3) # Dành cho L1, L2, L3
        ])
        
        # --- TẦNG 4 -> 15: SHARED BLOCK (Tiết kiệm tài nguyên) ---
        self.fine_decoder_shared = ResConvBlock(
            num_layers=max(1, n_decoder_layer // 2), n_channels=encoder_hidden, k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate, use_weight_norm=use_weight_norm, dilations=dialations
        )

        # --- EMBEDDINGS & HEADS ---
        self.layer_embs = nn.ModuleList([
            torch.nn.Embedding(vocab_size, encoder_hidden) for _ in range(n_mels - 1)
        ])
        
        # Lớp Linear để hòa trộn (Fusion) thay vì cộng thô
        self.fusion_layers = nn.ModuleList([
            torch.nn.Linear(encoder_hidden * 2, encoder_hidden) for _ in range(n_mels - 1)
        ])

        self.fine_outputs = nn.ModuleList([
            torch.nn.Linear(encoder_hidden, vocab_size) for _ in range(n_mels - 1)
        ])

    def sample_tokens(self, logits, temperature=0.8, top_k=5):
        logits = logits / temperature
        top_v, _ = torch.topk(logits, top_k, dim=-1)
        logits[logits < top_v[..., [-1]]] = -float('Inf')
        probs = torch.nn.functional.softmax(logits, dim=-1)
        shape = probs.shape
        probs_flat = probs.view(-1, shape[-1])
        sampled_flat = torch.multinomial(probs_flat, 1)
        return sampled_flat.view(*shape[:-1])

    def forward(self, text_value_expanded, targets=None, temperature=0.8, top_k=5, teacher_forcing_ratio=0.75):
        B, T, _ = text_value_expanded.shape
        
        # 1. DỰ ĐOÁN TẦNG 0
        x_coarse = self.coarse_pre_linear(text_value_expanded)
        x_coarse = self.coarse_decoder(x_coarse.transpose(1, 2)).transpose(1, 2)
        logits_coarse = self.coarse_output(x_coarse)
        logits_list = [logits_coarse.unsqueeze(2)]
        
        # Trạng thái ẩn hiện tại (bắt đầu từ output của Coarse Decoder)
        current_hidden = x_coarse 
        
        # Lấy token khởi đầu
        if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
            current_tokens = targets[:, :, 0]
        else:
            if targets is None:
                current_tokens = self.sample_tokens(logits_coarse, temperature, top_k)
            else:
                current_tokens = torch.argmax(logits_coarse, dim=-1).detach()

        # 2. DỰ ĐOÁN CASCADED (L1 -> L15)
        for i in range(self.n_mels - 1):
            # Hòa trộn Text Context và Audio Token hiện tại
            emb_i = self.layer_embs[i](current_tokens)
            # Fusion: Concatenate rồi đưa về hidden_size
            gate_input = torch.cat([current_hidden, emb_i], dim=-1)
            current_hidden = self.fusion_layers[i](gate_input)
            
            # Chọn Decoder: 3 tầng đầu dùng não riêng, các tầng sau dùng chung
            if i < 3:
                h = self.priority_decoders[i](current_hidden.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.fine_decoder_shared(current_hidden.transpose(1, 2)).transpose(1, 2)
            
            # Dự đoán tầng tiếp theo
            logits_i = self.fine_outputs[i](h)
            logits_list.append(logits_i.unsqueeze(2))
            
            # Cập nhật hidden state cho vòng lặp sau
            current_hidden = h
            
            # Chuẩn bị Token cho tầng kế tiếp (Teacher Forcing hoặc Sample)
            if i < self.n_mels - 2:
                if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    current_tokens = targets[:, :, i+1]
                else:
                    if targets is None:
                        current_tokens = self.sample_tokens(logits_i, temperature, top_k)
                    else:
                        current_tokens = torch.argmax(logits_i, dim=-1).detach()
                
        return torch.cat(logits_list, dim=2)
    
    
class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, idim, filter_size=256, ksize=3, dropout=0.1, offset=1):
        super(DurationPredictor, self).__init__()
        self.input_size = idim
        self.filter_size = filter_size
        self.kernel = ksize
        self.conv_output_size = filter_size
        self.dropout = dropout
        self.offset = offset

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("relu_1", nn.ReLU()),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("relu_2", nn.ReLU()),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        # predict log(d_target + offset)
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        return out

    def inference(self, encoder_output):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        out = torch.clamp(out.exp() - self.offset, min=0)
        return out
