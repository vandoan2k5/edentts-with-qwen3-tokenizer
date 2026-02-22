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
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
from hparams import hparams


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



# --- 1. SETTINGS & HELPERS ---

@dataclass
class ModelArgs:
    dim: int = 1024
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: Optional[int] = 2
    vocab_size: int = 2048
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048

@dataclass
# Lấy giá trị trực tiếp từ hparams bạn đã khởi tạo
@dataclass
class DecoderConfig:
    # Lấy từ hparams.text_encoder_hidden (384)
    idim: int = hparams.text_encoder_hidden   
    # Lấy từ hparams.decoder_hidden (512)
    hidden_size: int = hparams.decoder_hidden 
    # Lấy từ hparams.vocab_size (365 - đây là token text, 
    # nhưng nếu bạn dùng cho audio RVQ thì thường là 1024 hoặc 2048)
    vocab_size: int = 2048 
    # Lấy từ hparams.num_mels (16 layers audio)
    num_codebooks: int = hparams.num_mels      
    
    # Layer settings (tùy chỉnh theo kiến trúc Transformer mới)
    num_layers_coarse: int = hparams.decoder_layers # 6
    num_layers_priority: int = 4
    num_layers_shared: int = 2

def sample_token(logits: torch.Tensor, topk: int = 5, temperature: float = 1.0) -> torch.Tensor:
    # Thêm hàm sample đơn giản để code không báo lỗi
    logits = logits / max(temperature, 1e-5)
    v, _ = torch.topk(logits, min(topk, logits.size(-1)))
    pivot = v[..., [-1]]
    logits[logits < pivot] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(*probs.shape[:-1])

# --- 2. LLAMA BLOCKS (FROM SCRATCH) ---

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(3))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Lặp lại các KV heads để khớp với số lượng Q heads (Cần thiết cho Llama GQA).
    Shape input: (B, T, n_kv_heads, head_dim)
    Shape output: (B, T, n_heads, head_dim)
    """
    if n_rep == 1:
        return x
    bs, sl, n_kv, hd = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, sl, n_kv, n_rep, hd)
        .reshape(bs, sl, n_kv * n_rep, hd)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        # Tỷ lệ lặp lại đầu (ví dụ: 16 heads / 2 kv_heads = 8 lần lặp)
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # FIX: Lặp lại KV để khớp với Q trước khi nhân ma trận
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Chuyển về (B, Heads, T, Dim) để matmul
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        output = torch.matmul(scores, xv) # (B, Heads, T, Dim)
        return self.wo(output.transpose(1, 2).contiguous().view(bsz, seqlen, -1))

class FeedForward(nn.Module):
    def __init__(self, dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(8 * dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1, self.w2, self.w3 = nn.Linear(dim, hidden_dim, bias=False), nn.Linear(hidden_dim, dim, bias=False), nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args.dim, args.multiple_of)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    def forward(self, x, freqs_cis):
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        return x + self.feed_forward(self.ffn_norm(x))

class LlamaTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len))

    def forward(self, x):
        # x is embedding if tok_embeddings is Identity
        freqs_cis = self.freqs_cis[:x.shape[1]].to(x.device)
        for layer in self.layers:
            x = layer(x, freqs_cis)
        return self.norm(x)

def get_llama3_2_from_scratch(num_layers, num_heads, embed_dim):
    args = ModelArgs(n_layers=num_layers, n_heads=num_heads, dim=embed_dim)
    return LlamaTransformer(args)

def prepare_transformer(model):
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, model.args.dim

class TransformerDecoderV3(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        # Thêm Projection Layer để khớp số chiều idim (n_channels) với hidden_size của Llama
        self.input_proj = nn.Linear(config.idim, config.hidden_size) if config.idim != config.hidden_size else nn.Identity()
        
        # Khởi tạo các Llama blocks như cũ
        self.coarse_decoder, _ = prepare_transformer(
            get_llama3_2_from_scratch(num_layers=config.num_layers_coarse, num_heads=16, embed_dim=config.hidden_size)
        )
        self.coarse_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.priority_decoders = nn.ModuleList([
            prepare_transformer(get_llama3_2_from_scratch(num_layers=config.num_layers_priority, num_heads=8, embed_dim=config.hidden_size))[0]
            for _ in range(3)
        ])
        
        self.shared_fine_decoder, _ = prepare_transformer(
            get_llama3_2_from_scratch(num_layers=config.num_layers_shared, num_heads=8, embed_dim=config.hidden_size)
        )

        self.audio_embeddings = nn.Embedding(config.vocab_size * config.num_codebooks, config.hidden_size)
        self.register_buffer("audio_shifts", (config.vocab_size * torch.arange(config.num_codebooks)).view(1, 1, -1))

        self.fine_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_size) for _ in range(config.num_codebooks - 1)])
        self.fusion_layers = nn.ModuleList([nn.Linear(config.hidden_size * 2, config.hidden_size) for _ in range(config.num_codebooks - 1)])

    def _get_layered_embedding(self, tokens: torch.Tensor, layer_idx: int):
        shifted_tokens = tokens + (layer_idx * self.config.vocab_size)
        return self.audio_embeddings(shifted_tokens)

    # Thêm tham số top_k vào hàm forward
    def forward(self, text_features: torch.Tensor, targets: Optional[torch.Tensor] = None, temperature: float = 0.8, top_k: int = 5):
        text_features = self.input_proj(text_features)
        B, T, _ = text_features.shape
        
        logits_list = []
        sampled_tokens_list = [] # THÊM LIST NÀY: Để lưu token đã lấy mẫu
        
        h_coarse = self.coarse_decoder(text_features) 
        logits_0 = self.coarse_head(h_coarse)
        logits_list.append(logits_0.unsqueeze(2)) 

        current_tokens = targets[:, :, 0] if targets is not None else sample_token(logits_0, topk=top_k, temperature=temperature)
        sampled_tokens_list.append(current_tokens.unsqueeze(2)) # Lưu lại
        current_hidden = h_coarse

        for i in range(self.config.num_codebooks - 1):
            emb_prev = self._get_layered_embedding(current_tokens, layer_idx=i)
            current_hidden = self.fusion_layers[i](torch.cat([current_hidden, emb_prev], dim=-1))

            h = self.priority_decoders[i](current_hidden) if i < 3 else self.shared_fine_decoder(current_hidden)
            
            logits_i = self.fine_heads[i](h)
            logits_list.append(logits_i.unsqueeze(2))
            
            # Xử lý cẩn thận cho token tiếp theo
            if targets is not None:
                if i < self.config.num_codebooks - 2:
                    current_tokens = targets[:, :, i+1]
            else:
                # Lúc inference, sample và lưu lại ngay lập tức
                current_tokens = sample_token(logits_i, topk=top_k, temperature=temperature)
                sampled_tokens_list.append(current_tokens.unsqueeze(2))
            
            current_hidden = h

        # NẾU LÀ INFERENCE: Trả về trực tiếp Tokens (chứ không phải logits)
        if targets is None:
            return torch.cat(sampled_tokens_list, dim=2) 
            
        # NẾU LÀ TRAIN: Trả về Logits để tính Loss
        return torch.cat(logits_list, dim=2)