import torch
from torch import optim
from utils.display import *
from utils.dataset import get_tts_datasets
from torch.utils.data import DataLoader
from utils.paths import Paths
from models.cifTTS import CifTTSModel # Đổi sang model mới
import time
import wandb
from utils.display import stats_str, save_stats
from utils.checkpoints import save_checkpoint, restore_checkpoint
from utils.log_util import get_logger
from models.loss import CifTTSLoss # Sử dụng class loss mới
from hparams import hparams as hp

log = get_logger(__name__)

def np_now(x: torch.Tensor): return x.detach().cpu().numpy()

# torch.autograd.set_detect_anomaly(True)

def main():
    paths = Paths(hp.data_path, hp.tts_model_id, speaker=hp.speaker)
    device = torch.device('cuda')
    log.info(f"🚀 Train: {hp.tts_model_id} | Batch: {hp.batch_size} | Device: {device}")
    
    # 1. Khởi tạo cấu hình cho Model mới
    cif_cfg = hp.get_cif_config()
    decoder_cfg = hp.get_decoder_config()
    text_cfg = {
        'encoder_layer': hp.text_encoder_layers,
        'encoder_head': 2,
        'encoder_hidden': hp.text_encoder_hidden,
        'encoder_dropout': hp.text_encoder_dropout,
        'vocab_size': hp.vocab_size
    }

    log.info('\nInitialising CifTTS Model...\n')
    model = CifTTSModel(text_cfg, cif_cfg, decoder_cfg).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = CifTTSLoss(delta=hp.delta) # Loss class mới

    restore_checkpoint('tts', paths, model, optimizer, create_if_missing=True)
    
    # Giữ nguyên get_tts_datasets cũ
    train_set, val_set, test_set, attn_example = get_tts_datasets(paths.data, hp.batch_size)
    log.info(f"Attention example ID: {attn_example}")
    
    tts_train(paths, model, optimizer, criterion, train_set, hp.lr, hp.tts_max_steps, attn_example)
    log.info('\n\n training completed!')

def tts_train(paths, model, optimizer, criterion, train_set, lr, train_steps, attn_example):
    wandb_run = wandb.init(project="cifTTS", config=vars(hp))
    device = next(model.parameters()).device 
    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(1, epochs+1):
        start = time.time()
        
        for i, batch in enumerate(train_set, 1):
            # 1. CHUẨN BỊ DỮ LIỆU (Giữ nguyên unpack batch của bạn)
            # Dataloader cũ trả về: phones, text_lens, m, mel_lens, e_mask, ids
            phones, text_lens, m, mel_lens, e_mask = [v.to(device) for v in batch[:-1]]
            ids = batch[-1]
            
            # Chuyển Target về dạng Token Long cho CrossEntropy
            # m thường là [B, 16, T], ta cần [B, T, 16]
            m_target = m.transpose(1, 2).long() 

            model.train()
            optimizer.zero_grad()

            # 2. FORWARD MODEL (Mẫu mới nhận target_audio_lengths và target_audio_tokens)
            outputs = model(
                phone_ids=phones, 
                phone_lengths=text_lens, 
                target_audio_lengths=mel_lens, 
                target_audio_tokens=m_target
            )
            step = model.get_step()
            
            # 3. TÍNH LOSS (Sử dụng class Loss mới đã tối ưu)
            loss_dict = criterion(outputs, m_target, mel_lens)
            loss = loss_dict["total_loss"]

            # --- PHẦN MỞ RỘNG: CONTINUOUS LOSS (Nếu bạn muốn giữ lại từ code cũ) ---
            # Chỉ thực hiện nếu model có layer audio_embeddings (tên biến tùy thuộc vào components.py)
            continuous_loss = torch.tensor(0.0).to(device)
            if hasattr(model.decoder, 'audio_embeddings'):
                emb_weight = model.decoder.audio_embeddings.weight.detach()
                valid_mask = ~get_mask_from_lengths(mel_lens, max_len=outputs["decoder_out"].shape[1]).to(device)
                
                # Tính Soft-Embedding cho Layer 0
                pred_L0 = outputs["decoder_out"][:, :, 0, :] 
                probs_L0 = torch.nn.functional.softmax(pred_L0[valid_mask], dim=-1)
                soft_emb_pred = torch.matmul(probs_L0, emb_weight[:2048]) # Vocab size 2048
                
                # Hard-Embedding thực tế
                # Chú ý: audio_embeddings trong code mới thường offset theo layer, 
                # ở đây ta chỉ lấy layer 0 nên tokens giữ nguyên
                hard_emb_target = model.decoder.audio_embeddings(m_target[:, :, 0][valid_mask])
                continuous_loss = torch.nn.functional.mse_loss(soft_emb_pred, hard_emb_target)
                
                loss += continuous_loss # Cộng thêm vào loss tổng

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 4. TÍNH TOÁN STATS (Accuracy L0 -> L3)
            stats = dict()
            with torch.no_grad():
                mel_pred = outputs["decoder_out"]
                valid_mask = ~get_mask_from_lengths(mel_lens, max_len=mel_pred.shape[1]).to(device)
                total_valid_frames = valid_mask.sum().item()
                
                if total_valid_frames > 0:
                    for layer in range(4): # Tính L0, L1, L2, L3
                        pred_ids = torch.argmax(mel_pred[:, :, layer, :], dim=-1)
                        target_layer = m_target[:, :, layer]
                        correct = (pred_ids[valid_mask] == target_layer[valid_mask]).sum().item()
                        stats[f"acc_L{layer}"] = correct / total_valid_frames

                    # Accuracy tổng quát
                    pred_all = torch.argmax(mel_pred, dim=-1)
                    mask_all = valid_mask.unsqueeze(-1).expand(-1, -1, 16)
                    stats["acc"] = (pred_all[mask_all] == m_target[mask_all]).sum().item() / (total_valid_frames * 16)
                else:
                    stats["acc_L0"], stats["acc"] = 0.0, 0.0

            stats["loss"] = loss.item()
            stats["ce"] = loss_dict["ce_loss"].item()
            stats["quant"] = loss_dict["quantity_loss"].item()
            stats["cont"] = continuous_loss.item()

            # Log WandB
            wandb_run.log(stats)
            save_stats(stats, paths, step)

            # 5. HIỂN THỊ & CHECKPOINT
            speed = i / (time.time() - start)
            if step % hp.tts_show_info_every == 0:
                msg = f'| Ep: {e} | Step: {step} | {stats_str(stats)} | {speed:#.2} st/s'
                log.info(msg)

            if step % hp.tts_checkpoint_every == 0:
                save_checkpoint('tts', paths, model, optimizer, name=f'step_{step}', is_silent=True)

            # Visualization (Giữ nguyên logic cũ)
            if attn_example in ids:
                idx = ids.index(attn_example)
                mel_len = mel_lens[idx].item()
                p_mel_ids = torch.argmax(mel_pred[idx, :mel_len, 0, :], dim=-1) # Plot layer 0
                save_spectrogram(np_now(p_mel_ids.float() / 2048.0), paths.tts_mel_plot / f'{step}', 600)

            if step >= train_steps: break
        if step >= train_steps: break

    save_checkpoint('tts', paths, model, optimizer, is_silent=False)

def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None: max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids >= lengths.unsqueeze(1))
    return mask

if __name__ == "__main__":
    main()