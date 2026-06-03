import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# --- Custom & External Imports ---
from transformer.Layers import FFTBlock
from transformer.Models import get_sinusoid_encoding_table
from collections import OrderedDict
from models.modules import Conv

from utils.log_util import get_logger
try:
    from utils.net_utils import get_padding
except ImportError:
    def get_padding(kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

logging = get_logger(__name__)

# Import Global params và các Config class
from hparams import hparams as hp, ModelArgs, DecoderConfig



class LayerNorm(nn.LayerNorm):
    """Layer normalization hỗ trợ đa chiều (transpose)"""
    def __init__(self, nout, dim=-1):
        super().__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super().forward(x)
        return super().forward(x.transpose(1, -1)).transpose(1, -1)

class ResConv1d(nn.Module):
    """Khối Residual Convolution 1D tiêu chuẩn"""
    def __init__(self, n_channels=512, k_size=5, nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.1}, dropout_rate=0.1, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=k_size, 
                      padding=get_padding(k_size, dilation), dilation=dilation),
            getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

    def forward(self, x):
        return x + self.conv(x)

class ResConvBlock(nn.Module):
    """Chuỗi các khối ResConv1d với tùy chọn Weight Norm"""
    def __init__(self, num_layers, n_channels=512, k_size=5, use_weight_norm=True, dilations=None):
        super().__init__()
        layers = []
        if dilations is not None:
            for d in dilations:
                layers.append(ResConv1d(n_channels, k_size, dilation=d))
        else:
            for _ in range(num_layers):
                layers.append(ResConv1d(n_channels, k_size))
        
        self.layers = nn.Sequential(*layers)
        if use_weight_norm:
            self.apply_weight_norm()

    def apply_weight_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.utils.weight_norm(m)

    def forward(self, x):
        return self.layers(x)

class TokenEmbedding(nn.Module):
    """Lớp Embedding cho Phonemes/Text"""
    def __init__(self, hidden_size=384, padding_idx=0, vocab_size=365):
        super().__init__()
        self.phone_embed_layer = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

    def forward(self, phone_ids):
        return self.phone_embed_layer(phone_ids)

# ==========================================
# 1. UTILS & HELPERS
# ==========================================

def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids >= lengths.unsqueeze(1)).bool()
    return mask

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

def sample_token(logits: torch.Tensor, topk: int = 5, temperature: float = 1.0) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    v, _ = torch.topk(logits, min(topk, logits.size(-1)))
    pivot = v[..., [-1]]
    logits[logits < pivot] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(*probs.shape[:-1])


# ==========================================
# 3. TEXT ENCODER
# ==========================================

class TextEncoder(torch.nn.Module):
    """
    Text encoder adapted from FastSpeech
    """
    def __init__(self, encoder_layer=5, encoder_head=2, encoder_hidden=256,
                 conv_filter_size=1024, conv_kernel_size=[9, 1], encoder_dropout=0.2,
                 n_channels=512, vocab_size=365):
        super().__init__()
        max_seq_len = 1000
        n_position = max_seq_len + 1
        d_word_vec = n_channels
        n_layers = encoder_layer
        n_head = encoder_head
        d_k = d_v = (encoder_hidden // encoder_head)
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
        self.layer_stack = nn.ModuleList([
            FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.linear_key = nn.Linear(encoder_hidden, n_channels)
        self.linear_value = nn.Linear(encoder_hidden, n_channels)

    def forward(self, src_seq, text_lengths, return_attns=False):
        mask = get_mask_from_lengths(text_lengths).to(src_seq.device)
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        if hp.pos_embed_scheme == "absolute":
            if not self.training and src_seq.shape[1] > self.max_seq_len:
                enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                    src_seq.shape[1], self.d_model
                )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.device)
            else:
                enc_output = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        else:
            enc_output = self.src_word_emb(src_seq)
            
        enc_output = self.pre_linear(enc_output)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
                
        text_key = self.linear_key(enc_output)
        text_value = self.linear_value(enc_output)
        return text_key, text_value

    def inference(self, phone_ids: torch.Tensor):
        text_lens = torch.Tensor([phone_ids.size(1)]).long()
        text_key, text_value = self.forward(phone_ids, text_lens)
        return text_value


# ==========================================
# 4. CIF MIDDLEWARE
# ==========================================

class CifMiddleware(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cif_threshold = cfg.cif_threshold
        self.cif_output_dim = cfg.cif_embedding_dim
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.produce_weight_type = cfg.produce_weight_type
        self.conv_cif_width = cfg.conv_cif_width
        self.conv_cif_dropout = cfg.conv_cif_dropout
        self.apply_scaling = cfg.apply_scaling
        self.apply_tail_handling = cfg.apply_tail_handling
        self.tail_handling_firing_threshold = cfg.tail_handling_firing_threshold

        if self.produce_weight_type == "dense":
            self.dense_proj = Linear(self.encoder_embed_dim, self.encoder_embed_dim).cuda()
            self.weight_proj = Linear(self.encoder_embed_dim, 1).cuda()
        elif self.produce_weight_type == "conv":
            self.conv = torch.nn.Conv1d(
                self.encoder_embed_dim, self.encoder_embed_dim, self.conv_cif_width,
                stride=1, padding=int(self.conv_cif_width / 2),
                dilation=1, groups=1, bias=True, padding_mode='zeros'
            ).cuda()
            self.conv_dropout = torch.nn.Dropout(p=self.conv_cif_dropout).cuda()
            self.weight_proj = Linear(self.encoder_embed_dim, 1).cuda()
        else:
            self.weight_proj = Linear(self.encoder_embed_dim, 1).cuda()

        if self.cif_output_dim != self.encoder_embed_dim:
            self.cif_output_proj = Linear(self.encoder_embed_dim, self.cif_output_dim, bias=False).cuda()

    def forward(self, encoder_outputs, target_lengths):
        encoder_raw_outputs = encoder_outputs["encoder_raw_out"]
        encoder_padding_mask = encoder_outputs["encoder_padding_mask"]
        device = encoder_raw_outputs.device

        if self.produce_weight_type == "dense":
            proj_out = self.dense_proj(encoder_raw_outputs)
            act_proj_out = torch.relu(proj_out)
            sig_input = self.weight_proj(act_proj_out)
            weight = torch.sigmoid(sig_input)
        elif self.produce_weight_type == "conv":
            conv_input = encoder_raw_outputs.permute(0, 2, 1)
            conv_out = self.conv(conv_input)
            proj_input = conv_out.permute(0, 2, 1)
            proj_input = self.conv_dropout(proj_input)
            sig_input = self.weight_proj(proj_input)
            weight = torch.sigmoid(sig_input)
        else:
            sig_input = self.weight_proj(encoder_raw_outputs)
            weight = torch.sigmoid(sig_input)

        not_padding_mask = ~encoder_padding_mask
        weight = torch.squeeze(weight, dim=-1) * not_padding_mask.int()
        org_weight = weight

        if self.training and self.apply_scaling and target_lengths is not None:
            weight_sum = weight.sum(-1)
            normalize_scalar = torch.unsqueeze(target_lengths / weight_sum, -1)
            weight = weight * normalize_scalar

        batch_size = encoder_raw_outputs.size(0)
        max_length = encoder_raw_outputs.size(1)
        encoder_embed_dim = encoder_raw_outputs.size(2)
        padding_start_id = not_padding_mask.sum(-1)

        accumulated_weights = torch.zeros(batch_size, max_length, device=device)
        accumulated_states = torch.zeros(batch_size, max_length, encoder_embed_dim, device=device)
        fired_states = torch.zeros(batch_size, max_length, encoder_embed_dim, device=device)

        # Track per-position contributions to each fired frame.
        # contributions[b, i, j] = how much of text position i ends up in audio frame j.
        # We don't know T_aud yet, so we allocate the upper bound [B, T_txt, T_txt]
        # and slice at the end. We also keep, per-batch, a list of (pos, weight)
        # pairs that are currently in the accumulation pool - this lets us
        # attribute the prev_accumulated part of each fire to the contributing
        # text positions proportionally to their weight. When the pool is empty
        # (the previous position just fired), the prev_accumulated is the
        # residual produced by that previous fire, so we attribute it back to
        # the previous firing position via `last_firing_pos`.
        contributions = torch.zeros(batch_size, max_length, max_length, device=device)
        accumulated_lists = [[] for _ in range(batch_size)]
        fire_counters = [0] * batch_size
        last_firing_pos = [-1] * batch_size

        for i in range(max_length):
            prev_accumulated_weight = torch.zeros([batch_size], device=device) if i == 0 else accumulated_weights[:, i - 1]
            prev_accumulated_state = torch.zeros([batch_size, encoder_embed_dim], device=device) if i == 0 else accumulated_states[:, i - 1, :]

            cur_is_fired = ((prev_accumulated_weight + weight[:, i]) >= self.cif_threshold).unsqueeze(dim=-1)
            cur_weight = torch.unsqueeze(weight[:, i], -1)
            prev_accumulated_weight_unsq = torch.unsqueeze(prev_accumulated_weight, -1)
            remained_weight = torch.ones_like(prev_accumulated_weight_unsq).to(device) - prev_accumulated_weight_unsq

            cur_accumulated_weight = torch.where(
                cur_is_fired, cur_weight - remained_weight, cur_weight + prev_accumulated_weight_unsq)
            cur_accumulated_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                (cur_weight - remained_weight) * encoder_raw_outputs[:, i, :],
                prev_accumulated_state + cur_weight * encoder_raw_outputs[:, i, :])
            cur_fired_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                prev_accumulated_state + remained_weight * encoder_raw_outputs[:, i, :],
                torch.zeros([batch_size, encoder_embed_dim]).cuda())

            if (not self.training) and self.apply_tail_handling:
                cur_fired_state = torch.where(
                    i == padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                    torch.where(
                        cur_accumulated_weight.repeat([1, encoder_embed_dim]) <= self.tail_handling_firing_threshold,
                        torch.zeros([batch_size, encoder_embed_dim]).cuda(),
                        cur_accumulated_state / (cur_accumulated_weight + 1e-10)
                    ), cur_fired_state)

            cur_fired_state = torch.where(
                torch.full([batch_size, encoder_embed_dim], i).cuda() >
                padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                torch.zeros([batch_size, encoder_embed_dim]).cuda(), cur_fired_state)

            accumulated_weights[:, i] = cur_accumulated_weight.squeeze(-1) # Chú ý shape
            accumulated_states[:, i, :] = cur_accumulated_state
            fired_states[:, i, :] = cur_fired_state

            # Per-position contribution tracking.
            for b in range(batch_size):
                if cur_is_fired[b, 0].item():
                    j = fire_counters[b]
                    if j < max_length:
                        # Position i itself contributes `remained_weight` to this fire.
                        contributions[b, i, j] = remained_weight[b, 0].item()
                        # Every position that was in the accumulation pool also
                        # contributes its full weight (it is consumed by this fire).
                        for (pos, w) in accumulated_lists[b]:
                            contributions[b, pos, j] = w
                    fire_counters[b] += 1
                    last_firing_pos[b] = i
                    # The residual cur_accumulated_weight is the leftover weight
                    # that wasn't consumed by this fire. Carry it forward as an
                    # attributed contribution from position i so the pool stays
                    # consistent with prev_accumulated at the next step.
                    residual = cur_accumulated_weight[b, 0].item()
                    accumulated_lists[b] = [(i, residual)] if residual > 0 else []
                else:
                    accumulated_lists[b].append((i, weight[b, i].item()))

        fired_marks = (torch.abs(fired_states).sum(-1) != 0.0).int()
        fired_utt_length = fired_marks.sum(-1)
        fired_max_length = fired_utt_length.max().int()
        cif_outputs_list = [] 

        def dynamic_partition(data: torch.Tensor, partitions: torch.Tensor, num_partitions=None):
            assert len(partitions.shape) == 1, "Only one dimensional partitions supported"
            assert (data.shape[0] == partitions.shape[0]), "Partitions requires the same size as data"
            if num_partitions is None:
                num_partitions = max(torch.unique(partitions))
            return [data[partitions == index] for index in range(num_partitions)]

        for j in range(batch_size):
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]
            cur_utt_outputs = dynamic_partition(cur_utt_fired_state, cur_utt_fired_mark, 2)
            cur_utt_output = cur_utt_outputs[1]
            cur_utt_length = cur_utt_output.size(0)
            pad_length = fired_max_length - cur_utt_length
            
            # Tối ưu: Dùng torch.zeros với device thay vì .cuda() cứng
            pad_tensor = torch.zeros([pad_length, encoder_embed_dim], dtype=cur_utt_output.dtype, device=device)
            cur_utt_output = torch.cat((cur_utt_output, pad_tensor), dim=0)
            
            cur_utt_output = torch.unsqueeze(cur_utt_output, 0)
            
            # Thêm vào list thay vì cat trực tiếp
            cif_outputs_list.append(cur_utt_output)

        # 2. CHỈ THỰC HIỆN CAT 1 LẦN DUY NHẤT NGOÀI VÒNG LẶP
        cif_outputs = torch.cat(cif_outputs_list, dim=0)

        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).int()
        # --- KẾT THÚC ĐOẠN CODE SỬA ---

        if self.training:
            quantity_out = org_weight.sum(-1)
        else:
            quantity_out = weight.sum(-1)

        if self.cif_output_dim != encoder_embed_dim:
            cif_outputs = self.cif_output_proj(cif_outputs)

        # Slice the contribution matrix to the actual number of fires per batch.
        # We use the max so all batches share the same shape (padded with zeros).
        max_fires = max(fire_counters) if fire_counters else 0
        contributions = contributions[:, :, :max_fires]

        return {
            "cif_out": cif_outputs,
            "cif_out_padding_mask": cif_out_padding_mask,
            "quantity_out": quantity_out,
            "fired_marks": fired_marks,
            "accumulated_weights": accumulated_weights,
            "contributions": contributions,
        }


# ==========================================
# 5. LLAMA & DECODER BLOCKS
# ==========================================

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

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        output = torch.matmul(scores, xv)
        return self.wo(output.transpose(1, 2).contiguous().view(bsz, seqlen, -1))

class FeedForward(nn.Module):
    def __init__(self, dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(8 * dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
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
        
        self.input_proj = nn.Linear(config.idim, config.hidden_size) if config.idim != config.hidden_size else nn.Identity()
        
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

    def forward(self, text_features: torch.Tensor, targets: Optional[torch.Tensor] = None, temperature: float = 0.8, top_k: int = 5):
        text_features = self.input_proj(text_features)
        
        logits_list = []
        sampled_tokens_list = [] 
        
        h_coarse = self.coarse_decoder(text_features) 
        logits_0 = self.coarse_head(h_coarse)
        logits_list.append(logits_0.unsqueeze(2)) 

        current_tokens = targets[:, :, 0] if targets is not None else sample_token(logits_0, topk=top_k, temperature=temperature)
        sampled_tokens_list.append(current_tokens.unsqueeze(2))
        current_hidden = h_coarse

        for i in range(self.config.num_codebooks - 1):
            emb_prev = self._get_layered_embedding(current_tokens, layer_idx=i)
            current_hidden = self.fusion_layers[i](torch.cat([current_hidden, emb_prev], dim=-1))

            h = self.priority_decoders[i](current_hidden) if i < 3 else self.shared_fine_decoder(current_hidden)
            
            logits_i = self.fine_heads[i](h)
            logits_list.append(logits_i.unsqueeze(2))
            
            if targets is not None:
                if i < self.config.num_codebooks - 2:
                    current_tokens = targets[:, :, i+1]
            else:
                current_tokens = sample_token(logits_i, topk=top_k, temperature=temperature)
                sampled_tokens_list.append(current_tokens.unsqueeze(2))
            
            current_hidden = h

        if targets is None:
            return torch.cat(sampled_tokens_list, dim=2) 
            
        return torch.cat(logits_list, dim=2)