import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ==========================================
# 1. CONFIG SCHEMAS (Dành cho việc Import)
# ==========================================

@dataclass
class CifConfig:
    cif_threshold: float = 0.999
    cif_embedding_dim: int = 512
    encoder_embed_dim: int = 384
    produce_weight_type: str = "conv"
    conv_cif_width: int = 3
    conv_cif_dropout: float = 0.1
    apply_scaling: bool = True
    apply_tail_handling: bool = True
    tail_handling_firing_threshold: float = 0.5

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: Optional[int] = 2
    vocab_size: int = 2048
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048

@dataclass
class DecoderConfig:
    idim: int = 512
    hidden_size: int = 384
    vocab_size: int = 2048
    num_codebooks: int = 16
    num_layers_coarse: int = 4
    num_layers_priority: int = 2
    num_layers_shared: int = 1

# ==========================================
# 2. MAIN HPARAMS CLASS
# ==========================================

class Hparams(object):
    def __init__(self, data=None):
        self.config_file = "config/eden.json"
        if data is not None:
            self.config_file = data
            
        # Giá trị mặc định
        self.vocab_size = 365 
        self.text_encoder_hidden = 384
        self.n_channels = 512
        self.decoder_hidden = 512
        self.num_mels = 16 
        self.pos_embed_scheme = "none"
        self.text_encoder_layers = 6
        self.text_encoder_dropout = 0.2
        self.data_path = os.path.join("./", r'data')
        self.base_path = r'./'
        self.token_type = "char"



        self.sample_rate = 22050
        self.n_fft = 1024
        self.fft_bins = self.n_fft // 2 + 1
        self.num_mels = 80
        self.hop_length = 256 # 12.5ms - in line with Tacotron 2 paper
        self.win_length = 1024   # 50ms - same reason as above
        self.fmin = 0
        self.fmax = 8000
        self.bits = 16


        self.tts_max_steps = 50000  # you may stop at around 100_000 for ljspeech dataset for acceptable speech quality
        self.tts_max_mel_len = 100
        self.tts_bin_lengths = True   # bins the spectrogram lengths before sampling in data loader - speeds up training
        self.tts_checkpoint_every = 5000  # checkpoints the model every X steps
        self.tts_show_info_every = 2   # print tran status every X steps
        self.tts_eval_every = 30
        self.lr = 1e-4
        self.batch_size = 64

        # Đọc file config (eden.json)
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding="utf-8") as f:
                config_data = json.load(f)
            for key, value in config_data.items():
                setattr(self, key, value)
            print(f"✅ Loaded config from {self.config_file}")
        
        self.tts_model_id = "cif_tts"

    def get_cif_config(self):
        return CifConfig(
            cif_embedding_dim=self.n_channels,        # Thường là 512
            encoder_embed_dim=self.text_encoder_hidden # Thường là 384
        )

    def get_decoder_config(self):
        return DecoderConfig(
            idim=self.n_channels,                     # Đầu vào khớp với đầu ra CIF
            hidden_size=self.decoder_hidden,           # Thường là 512
            num_codebooks=self.num_mels,
            num_layers_coarse=getattr(self, 'decoder_layers', 6)
        )

hparams = Hparams()