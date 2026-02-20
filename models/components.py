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
                 dilations=None, vocab_size=2048):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, n_channels)
        self.mel_encoder = ResConvBlock(
            num_layers=n_mel_encoder_layer, n_channels=n_channels, k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate, use_weight_norm=use_weight_norm, dilations=dilations
        )

    def forward(self, speech):
        # CẢI TIẾN 1: speech có dạng [Batch, Time, 16]
        # Chỉ lấy Tầng 0 (chứa thông tin âm vị cốt lõi) để học Alignment, loại bỏ nhiễu
        speech_layer_0 = speech[:, :, 0] 
        
        mel_h = self.embedding(speech_layer_0) # [Batch, Time, n_channels]
        mel_h = mel_h.transpose(1, 2) 
        mel_h = self.mel_encoder(mel_h)
        
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

        # --- BƯỚC 2: FINE DECODER (Dự đoán Tầng 1 đến 15) ---
        self.layer0_emb = torch.nn.Embedding(vocab_size, encoder_hidden)
        self.fine_pre_linear = torch.nn.Linear(idim, encoder_hidden)
        self.fine_decoder = ResConvBlock(
            num_layers=n_decoder_layer, n_channels=encoder_hidden, k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate, use_weight_norm=use_weight_norm, dilations=dialations
        )
        self.fine_output = torch.nn.Linear(encoder_hidden, (n_mels - 1) * vocab_size)

    def forward(self, text_value_expanded, target_layer0=None):
        # text_value_expanded: [Batch, Time, idim]
        
        # 1. Dự đoán Tầng 0
        x_coarse = self.coarse_pre_linear(text_value_expanded)
        x_coarse = self.coarse_decoder(x_coarse.transpose(1, 2))
        logits_coarse = self.coarse_output(x_coarse.transpose(1, 2)) # [B, Time, 2048]
        
        # 2. Chuẩn bị thông tin Tầng 0 cho Decoder sau
        if target_layer0 is not None:
            # Training: Dùng "Teacher Forcing" bằng nhãn thật
            layer0_tokens = target_layer0
        else:
            # Inference: Lấy kết quả vừa dự đoán
            layer0_tokens = torch.argmax(logits_coarse, dim=-1)
            
        l0_emb = self.layer0_emb(layer0_tokens) # [B, Time, encoder_hidden]
        
        # 3. Dự đoán Tầng 1 đến 15 dựa trên (Text + Tầng 0)
        x_fine = self.fine_pre_linear(text_value_expanded)
        x_fine = x_fine + l0_emb # Ép sự phụ thuộc: Fine token phải khớp với Coarse token
        
        x_fine = self.fine_decoder(x_fine.transpose(1, 2))
        logits_fine = self.fine_output(x_fine.transpose(1, 2)) # [B, Time, 15 * 2048]
        
        # 4. Gộp kết quả
        B, T, _ = logits_fine.shape
        logits_fine = logits_fine.view(B, T, self.n_mels - 1, self.vocab_size)
        logits_coarse = logits_coarse.unsqueeze(2) # [B, Time, 1, 2048]
        
        # Output: [B, Time, 16, 2048]
        logits = torch.cat([logits_coarse, logits_fine], dim=2)
        
        return logits

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
