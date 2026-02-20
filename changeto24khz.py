import os
import librosa
import soundfile as sf
from tqdm import tqdm

def resample_folder(input_dir, output_dir, target_sr=24000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    for filename in tqdm(files, desc="Processing audio"):
        file_path = os.path.join(input_dir, filename)
        
        # Load audio và resample
        # librosa.load tự động resample nếu truyền target_sr
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        # Xuất file (mặc định là 16-bit PCM nếu không chỉnh)
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, audio, target_sr, subtype='PCM_16')

# Sử dụng
resample_folder('/kaggle/edentts-with-qwen3-tokenizer/LJSpeech-1.1/wavs', '/kaggle/edentts-with-qwen3-tokenizer/LJSpeech-1.1/wavs_24khz')