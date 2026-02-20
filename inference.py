"""
@author: edenmyn
@email: edenmyn
@time: 2022/10/1 13:40
@DESC: Modified for Qwen3 Audio Tokenizer
"""

from models.edenTTS import EdenTTS
from hparams import hparams as hp
from utils.paths import Paths
import torch

import time
from utils.dsp import save_wav
import os
from utils.log_util import get_logger
import numpy as np
from text.en_util import text_to_sequence
import argparse

# TODO: IMPORT QWEN3 AUDIO DECODER CỦA BẠN VÀO ĐÂY
# Ví dụ: from qwen_audio import QwenAudioDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = get_logger(__name__)

def m_inference(tts_model, out_path, texts):
    # Khởi tạo Qwen3 Decoder (Giả định)
    # qwen_decoder = QwenAudioDecoder.from_pretrained(...)
    # qwen_decoder.to(device)

    for text in texts:
        log.info(f"processing text: {text}")
        
        # 1. Chuẩn bị Text (Phone IDs)
        phones = text_to_sequence(text)
        phones = torch.tensor(phones).long().unsqueeze(0).to(device)
        
        # 2. Sinh Audio Tokens bằng EdenTTS
        s1 = time.time()
        with torch.no_grad():
            # mel_pred_ids có shape: [1, Time, 16] chứa các Token ID từ 0-2047
            mel_pred_ids = tts_model.inference(phones)
        log.info(f"acoustic model inference time: {time.time() - s1:.3f}s")
        
        # 3. Lưu lại numpy array của Token IDs phòng trường hợp cần dùng
        token_file = os.path.join(out_path, f'{text[:20].replace(" ", "_")}_tokens.npy')
        np.save(token_file, mel_pred_ids.cpu().numpy())
        log.info(f"Saved Token IDs to: {token_file}")

        # =========================================================
        # 4. GIẢI MÃ TOKENS THÀNH AUDIO (DÙNG QWEN3)
        # =========================================================
        '''
        # BẠN CẦN THAY THẾ ĐOẠN NÀY BẰNG HÀM DECODE THỰC TẾ CỦA QWEN3
        # Đa số các RVQ Decoder yêu cầu shape: [Batch, 16, Time]
        qwen_input = mel_pred_ids.transpose(1, 2) 
        
        with torch.no_grad():
            audio = qwen_decoder.decode(qwen_input) # Trả về sóng âm waveform
            
        file = os.path.join(out_path, f'{text[:20].replace(" ", "_")}.wav')
        wav = audio.squeeze().cpu().numpy()
        
        # Chuẩn hóa âm lượng (Normalize)
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak
            
        save_wav(wav, file)
        log.info(f"Synthesized wave saved at: {file}")
        '''

def inference(texts):
    if type(texts) == str:
        texts = [texts]
        
    tts_model = EdenTTS(hp).to(device)
    tts_model_id = hp.tts_model_id
    paths = Paths(hp.data_path, tts_model_id)
    
    tts_model_path = paths.tts_latest_weights
    if not os.path.exists(tts_model_path):
        log.error(f"{tts_model_path} does not exist. Vui lòng huấn luyện mô hình trước!")
        return
        
    out_path = paths.tts_output
    os.makedirs(out_path, exist_ok=True)
    
    log.info(f"Loading weights from {tts_model_path}")
    tts_model.load(tts_model_path)
    tts_model.eval() # Đảm bảo mô hình ở chế độ evaluation
    
    m_inference(tts_model, out_path, texts)


if __name__ == "__main__":
    assert hp.speaker == "ljs", "Hiện tại mô hình chỉ hỗ trợ speaker 'ljs'"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--text", type=str, required=True, help="Input text để tổng hợp giọng nói"
    )
    args = parser.parse_args()
    
    inference(args.text)