import torch
from hparams import hparams
from models.cifTTS import CifTTSModel
def print_model_summary(model):
    print("\n" + "="*50)
    print(f"{'Component':<25} | {'Parameters':<15}")
    print("-"*50)
    
    # 1. Text Encoder
    text_params = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
    print(f"{'1. Text Encoder':<25} | {text_params:>15,}")
    
    # 2. CIF Middleware
    cif_params = sum(p.numel() for p in model.cif.parameters() if p.requires_grad)
    print(f"{'2. CIF Middleware':<25} | {cif_params:>15,}")
    
    # 3. Decoder (Llama-based)
    # Tách nhỏ các phần trong Decoder
    dec = model.decoder
    coarse = sum(p.numel() for p in dec.coarse_decoder.parameters() if p.requires_grad)
    priority = sum(p.numel() for p in dec.priority_decoders.parameters() if p.requires_grad)
    shared = sum(p.numel() for p in dec.shared_fine_decoder.parameters() if p.requires_grad)
    heads = sum(p.numel() for p in dec.fine_heads.parameters() if p.requires_grad)
    embs = dec.audio_embeddings.weight.numel()
    
    decoder_total = coarse + priority + shared + heads + embs + sum(p.numel() for p in dec.input_proj.parameters())
    
    print(f"{'3. Decoder Total':<25} | {decoder_total:>15,}")
    print(f"{'   - Coarse Llama':<25} | {coarse:>15,}")
    print(f"{'   - Priority Llama (x3)':<25} | {priority:>15,}")
    print(f"{'   - Shared Fine Llama':<25} | {shared:>15,}")
    print(f"{'   - Audio Embeddings':<25} | {embs:>15,}")
    print(f"{'   - Output Heads':<25} | {heads:>15,}")
    
    print("-"*50)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'TOTAL TRAINABLE PARAMS':<25} | {total:>15,}")
    print("="*50 + "\n")

def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cif_cfg = hparams.get_cif_config()
    decoder_cfg = hparams.get_decoder_config()
    
    text_cfg = {
        'encoder_layer': hparams.text_encoder_layers,
        'encoder_head': 2,
        'encoder_hidden': hparams.text_encoder_hidden,
        'encoder_dropout': hparams.text_encoder_dropout,
        'vocab_size': hparams.vocab_size,
    }

    model = CifTTSModel(text_cfg, cif_cfg, decoder_cfg).to(device)
    print_model_summary(model)
    model.train() # Quan trọng để kích hoạt Scaling
    
    # --- THIẾT LẬP THEO THỰC TẾ CỦA BẠN ---
    B = 2
    T_txt = 100  # Text dài (100 tokens)
    T_aud = 40   # Audio ngắn (40 frames)
    # ---------------------------------------

    ids = torch.randint(1, hparams.vocab_size, (B, T_txt)).to(device)
    l_txt = torch.tensor([T_txt, T_txt - 10]).to(device)
    l_aud = torch.tensor([T_aud, T_aud - 5]).to(device)
    tokens = torch.randint(0, 2048, (B, T_aud, 16)).to(device)
    
    print(f"Testing with Text({T_txt}) > Audio({T_aud})...")
    
    try:
        out = model(ids, l_txt, l_aud, tokens)
        print("✅ Success!")
        print(f"Input Text Length: {T_txt}")
        print(f"Target Audio Length: {T_aud}")
        print(f"CIF Output Shape: {out['decoder_out'].shape} (Kỳ vọng chiều thứ 1 là {T_aud})")
    except Exception as e:
        print(f"❌ Vẫn lỗi: {e}")

if __name__ == "__main__":
    run_test()