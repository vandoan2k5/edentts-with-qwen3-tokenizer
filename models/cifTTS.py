import torch
import torch.nn as nn

from .components import (
    TextEncoder, 
    CifMiddleware, 
    TransformerDecoderV3, 
    get_mask_from_lengths
)

# Import Configs từ file hparams
from hparams import CifConfig, DecoderConfig
import torch
from pathlib import Path
from typing import Union


class AbstractModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
    
    def load(self, path: Union[str, Path]):
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device), strict=False)

    def save(self, path: Union[str, Path]):
        torch.save(self.state_dict(), path, _use_new_zipfile_serialization=False)

    def get_step(self):
        return self.step.data.item()

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

class CifTTSModel(AbstractModel):
    def __init__(self, text_cfg: dict, cif_cfg: CifConfig, decoder_cfg: DecoderConfig):
        super().__init__()
        
        # 1. Khởi tạo Text Encoder
        self.text_encoder = TextEncoder(
            encoder_layer=text_cfg.get('encoder_layer', 4),
            encoder_head=text_cfg.get('encoder_head', 2),
            encoder_hidden=text_cfg.get('encoder_hidden', 256),
            conv_filter_size=text_cfg.get('conv_filter_size', 1024),
            conv_kernel_size=text_cfg.get('conv_kernel_size', [9, 1]),
            encoder_dropout=text_cfg.get('encoder_dropout', 0.2),
            n_channels=text_cfg.get('encoder_hidden', 256), # Output dimension
            vocab_size=text_cfg.get('vocab_size', 365)
        )
        
        # 2. Khởi tạo CIF Middleware
        self.cif = CifMiddleware(cif_cfg)
        
        # 3. Khởi tạo Llama-based RVQ Decoder
        self.decoder = TransformerDecoderV3(decoder_cfg)

    def forward(
        self, 
        phone_ids: torch.Tensor, 
        phone_lengths: torch.Tensor, 
        target_audio_lengths: torch.Tensor = None, 
        target_audio_tokens: torch.Tensor = None, 
        temperature: float = 0.8, 
        top_k: int = 5,
        teacher_forcing_ratio: float = 0.75 
    ):
        """
        Quá trình forward chung cho cả Train và Inference.
        - Nếu có target_audio_lengths và target_audio_tokens: Mô hình chạy ở chế độ Training.
        - Nếu truyền None cho 2 tham số trên: Mô hình chạy ở chế độ Inference.
        """
        
        # --- BƯỚC 1: TEXT ENCODING ---
        _, text_value = self.text_encoder(phone_ids, phone_lengths)
        
        # --- BƯỚC 2: CIF MIDDLEWARE ---
        encoder_padding_mask = get_mask_from_lengths(phone_lengths, max_len=phone_ids.size(1))

        encoder_outputs = {
            "encoder_raw_out": text_value,
            "encoder_padding_mask": encoder_padding_mask # Truyền trực tiếp, không đảo ngược nữa
        }
        
        cif_ret = self.cif(encoder_outputs, target_audio_lengths)
        cif_out = cif_ret["cif_out"]
        quantity_out = cif_ret["quantity_out"]


        if target_audio_tokens is not None:
            target_len = target_audio_tokens.size(1)
            cif_len = cif_out.size(1)
            
            if cif_len < target_len:
                # Nếu CIF ngắn hơn, đệm thêm zeros
                pad_tensor = torch.zeros(
                    cif_out.size(0), target_len - cif_len, cif_out.size(2), 
                    dtype=cif_out.dtype, device=cif_out.device
                )
                cif_out = torch.cat([cif_out, pad_tensor], dim=1)
            elif cif_len > target_len:
                # Nếu CIF dài hơn, cắt bớt phần dư
                cif_out = cif_out[:, :target_len, :]
                
        # --- BƯỚC 3: DECODING ---
        decoder_out = self.decoder(
            text_features=cif_out,
            targets=target_audio_tokens,
            temperature=temperature,
            top_k=top_k
        )
        
        return {
            "decoder_out": decoder_out,
            "quantity_out": quantity_out,
            "cif_lengths": (cif_ret["cif_out_padding_mask"]).sum(dim=1) 
        }