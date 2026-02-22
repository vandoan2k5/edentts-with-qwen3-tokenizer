#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# edenmyn 2022/09/22 -> Modified for Classification 2026/02/20

import torch
import torch.nn as nn
from typing import Dict, Tuple
from models.abtract_model import AbstractModel
from .components import *
from utils.net_utils import parameter_count, make_non_pad_mask
from hparams import Hparams
from utils.log_util import get_logger

logging = get_logger(__name__)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# edenmyn 2022/09/22 -> Modified for Classification 2026/02/20

import torch
import torch.nn as nn
from typing import Dict, Tuple
from models.abtract_model import AbstractModel
from .components import *
from utils.net_utils import parameter_count, make_non_pad_mask
from hparams import Hparams
from utils.log_util import get_logger

logging = get_logger(__name__)

class EdenTTS(AbstractModel):
    def __init__(
        self,
        h: Hparams,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        duration_offset=1.0,
    ):
        super().__init__()
        self.duration_offset = duration_offset
        self.delta = h.delta
        self.n_mels = h.num_mels  # 16
        self.vocab_size = 2048    
        self.audio_emb = nn.Embedding(self.vocab_size, h.n_channels)

        self.text_encoder = TextEncoder(n_channels=h.n_channels,
                                        encoder_layer=h.text_encoder_layers,
                                        encoder_hidden=h.text_encoder_hidden,
                                        encoder_dropout=h.text_encoder_dropout,
                                        vocab_size=h.vocab_size)

        self.mel_encoder = MelEncoder(n_mels=h.n_channels,
                                      n_channels=h.n_channels,
                                      nonlinear_activation=nonlinear_activation,
                                      nonlinear_activation_params=nonlinear_activation_params,
                                      dropout_rate=h.mel_encoder_dropout,
                                      n_mel_encoder_layer=h.mel_encoder_layers,
                                      k_size=h.mel_encoder_ksize,
                                      use_weight_norm=use_weight_norm,
                                      dilations=h.mel_encoder_dilation)

        self.duration_predictor = DurationPredictor(
            idim=h.n_channels,
            filter_size=h.duration_predictor_filter_zie,
            ksize=h.duration_predictor_ksize,
            dropout=h.duration_predicotr_dropout
        )

        # --- NÂNG CẤP: DÙNG DUY NHẤT 1 DECODER CASCADED ---
        # DecoderV2 sẽ tự động dự đoán Tầng 0, sau đó lấy Tầng 0 làm input dự đoán Tầng 1, v.v... đến 15.
        decoder_config = DecoderConfig(
            idim=h.n_channels,                 # Khớp với output của Text Encoder (sau khi align)
            hidden_size=h.decoder_hidden,      # Kích thước ẩn của Llama blocks
            vocab_size=self.vocab_size,        # 2048
            num_codebooks=h.num_mels,          # 16 layers RVQ
            num_layers_coarse=4,
            num_layers_priority=2,
            num_layers_shared=2
            )
        self.decoder = TransformerDecoderV3(config=decoder_config)
        
        # Đã xóa bỏ refine_cond_proj vì DecoderV2 tự xử lý nội bộ

        self.postnet = None

        te = parameter_count(self.text_encoder)
        de = parameter_count(self.decoder)
        du = parameter_count(self.duration_predictor)
        logging.info(f"tol_params: {parameter_count(self)}, "
                     f"text_encoder:{te},"
                     f"decoder:{de},"
                     f"dur:{du},"
                     f"tol_infer:{te + de + du}")

    def forward(
        self,
        phone_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor, # targets [Batch, Time, 16]
        mel_lens: torch.Tensor,
        e_weight=None
    ) -> Tuple:
        if self.training:
            self.step += 1
        
        device = phone_ids.device
        mel_mask = ~make_non_pad_mask(mel_lens).to(device)
        
        text_key, text_value = self.text_encoder(phone_ids, text_lengths)
        mel_h = self.mel_encoder(speech)

        alpha = scaled_dot_attention(key=text_key, key_lens=text_lengths, query=mel_h,
                                     query_lens=mel_lens, e_weight=e_weight)
        
        dur0 = torch.sum(alpha, dim=-1)
        e = torch.cumsum(dur0, dim=-1)
        e = e - dur0/2
        reconst_alpha = reconstruct_align_from_aligned_position(e, mel_lens=mel_lens,
                                                                text_lens=text_lengths,
                                                                delta=self.delta)

        log_dur_pred = self.duration_predictor(text_value)
        log_dur_target = torch.log(dur0.detach() + self.duration_offset)

        text_value_expanded = torch.bmm(text_value.transpose(1, 2), reconst_alpha)
        _tmp_mask_2 = mel_mask.unsqueeze(1).repeat(1, text_value.size(2), 1)
        text_value_expanded = text_value_expanded.masked_fill(_tmp_mask_2, 0.0)

        # --- CHẠY DECODER V2 ---
        # Truyền toàn bộ speech (16 layers) vào để làm Teacher Forcing
        # DecoderV2 sẽ trả về Logits có shape [Batch, Time, 16, 2048]
        mel_pred = self.decoder(text_value_expanded.transpose(1, 2), targets=speech)
        
        return log_dur_pred, log_dur_target, mel_pred, alpha, reconst_alpha

    def inference(self, phone_ids: torch.Tensor, delta=None, d_control=1, temperature=0.8, top_k=1):
        if delta is None: delta = self.delta
        self.eval()
        with torch.no_grad():
            text_value = self.text_encoder.inference(phone_ids)
            print(f"Text Encoder output shape: {text_value.shape}")  # Debug: Kiểm tra shape của text_value
            dur = self.duration_predictor.inference(text_value)*d_control
            
            if torch.sum(dur)/dur.size(0) < 1:
                dur = 4*torch.ones_like(dur)
                
            e = torch.cumsum(dur, dim=1) - dur/2
            alpha = reconstruct_align_from_aligned_position(e, mel_lens=None, text_lens=None, delta=delta)
            
            text_value_expanded = torch.bmm(text_value.transpose(1, 2), alpha)
            print(f"Expanded Text Value shape: {text_value_expanded.shape}")  # Debug: Kiểm tra shape sau khi align
            
            # --- CHẠY DECODER V2 INFERENCE ---
            # DO targets=None, hàm forward bây giờ sẽ trực tiếp trả về mel_pred_ids [Batch, Time, 16]
            mel_pred_ids = self.decoder(
                text_value_expanded.transpose(1, 2), 
                targets=None, 
                temperature=temperature, 
                top_k=top_k
            )
            
        self.train()
        return mel_pred_ids