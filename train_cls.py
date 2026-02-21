from torch import optim
from utils.display import *
from utils.dataset import get_tts_datasets
from torch.utils.data import DataLoader
from utils.paths import Paths
from models.edenTTS import EdenTTS
import time
import wandb
from utils.display import stats_str, save_stats
from utils.checkpoints import save_checkpoint, restore_checkpoint
from utils.log_util import get_logger
from models.loss import *
from hparams import hparams as hp

log = get_logger(__name__)

def np_now(x: torch.Tensor): return x.detach().cpu().numpy()

torch.autograd.set_detect_anomaly(True)

def main():
    paths = Paths(hp.data_path, hp.tts_model_id, speaker=hp.speaker)
    device = torch.device('cuda')
    log.info(f"train :{hp.tts_model_id}, checkpoint path:{paths.tts_checkpoints}, batch_size:{hp.batch_size}, Using device:{device}")
    
    batch_size = hp.batch_size
    training_steps = hp.tts_max_steps
    lr = hp.lr

    log.info('\nInitialising  Model...\n')
    model = EdenTTS(hp).to(device)

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint('tts', paths, model, optimizer, create_if_missing=True)
    
    train_set, val_set, test_set, attn_example = get_tts_datasets(paths.data, batch_size)
    log.info(f"atten exmaple is: {attn_example}")
    tts_train(paths, model, optimizer, train_set, lr, training_steps, attn_example)
    log.info('\n\n training completed!')

def tts_train(paths: Paths, model: EdenTTS, optimizer, train_set: DataLoader, lr, train_steps, attn_example):
    wandb_run = wandb.init(project="edenTTS", config=vars(hp))
    device = next(model.parameters()).device 
    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(1, epochs+1):
        start = time.time()
        running_loss = 0
        ave_mel_loss = 0
        
        for i, batch in enumerate(train_set, 1):
            # 1. Chuẩn bị dữ liệu
            phones, text_lens, m, mel_lens, e_mask = [v.to(device) for v in batch[:-1]]
            m=m.transpose(1, 2) # Chuyển thành [Batch, Time, 16]
            m_target = torch.round(m).long() if m.dtype != torch.long else m
            ids = batch[-1]
            
            # CHUYỂN ĐỔI TARGET: Nếu m là float [0,1], đưa về Long [0,2047]
            # Nếu bạn đã chạy lại Preprocessing không chia 2048, dòng round này không cần thiết.
            
            # m_target = m_target.transpose(1, 2) # [Batch, Time, 16]

            # 2. Forward model (nhận về mel_pred là Logits [B, T, 16, 2048])
            outputs = model(phones, text_lens, m_target, mel_lens, e_mask)
            step = model.get_step()
            
            # Bóc tách output (EdenTTS mới trả về 5 giá trị)
            log_dur_pred, log_dur_target, mel_pred, alpha, reconst_alpha = outputs

            # 3. Tính toán Loss (Sử dụng CrossEntropy cho mel_loss)
            mel_loss = mel_loss_func(mel_pred, m_target, mel_lens)
            dur_loss = duration_loss_func(log_dur_pred, log_dur_target, text_lens)
            attn_loss = guided_atten_loss_func(alpha, text_lens, mel_lens)
            # Cân bằng trọng số: CrossEntropy thường lớn hơn MSE, có thể để 1.0 thay vì 5.0
            loss = dur_loss + mel_loss + attn_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 4. Tính toán Stats và Accuracy (giống tư tưởng voxtream)
            stats = dict()
            with torch.no_grad():
                # Lấy mask cơ bản [Batch, Time] để lọc padding
                valid_mask = ~get_mask_from_lengths(mel_lens, max_len=mel_pred.shape[1]).to(device)
                total_valid_frames = valid_mask.sum().item()
                
                if total_valid_frames > 0:
                    # --- A. ACCURACY TẦNG 0 (Nội dung chính) ---
                    # Lấy dự đoán và target của RIÊNG tầng đầu tiên (index 0)
                    pred_ids_L0 = torch.argmax(mel_pred[:, :, 0, :], dim=-1) # [B, T]
                    target_L0 = m_target[:, :, 0] # [B, T]
                    
                    correct_L0 = (pred_ids_L0[valid_mask] == target_L0[valid_mask]).sum().item()
                    stats["acc_L0"] = correct_L0 / total_valid_frames
                    
                    # --- B. TOP-5 ACCURACY (Cho toàn bộ 16 tầng) ---
                    # Lấy 5 giá trị có xác suất cao nhất: [B, T, 16, 5]
                    _, top5_preds = torch.topk(mel_pred, k=5, dim=-1)
                    
                    # Mở rộng chiều target để so sánh: [B, T, 16, 1]
                    target_expanded = m_target.unsqueeze(-1)
                    
                    # Kiểm tra target có nằm trong 5 dự đoán này không (trả về True/False) [B, T, 16]
                    correct_top5 = (top5_preds == target_expanded).any(dim=-1)
                    
                    # Expand mask để đếm trên 16 tầng
                    mask_16 = valid_mask.unsqueeze(-1).expand(-1, -1, 16)
                    top5_correct = correct_top5[mask_16].sum().item()
                    stats["top5"] = top5_correct / (total_valid_frames * 16)

                    # --- C. ACCURACY CŨ (Exact match cho cả 16 tầng - để tham khảo) ---
                    pred_ids_all = torch.argmax(mel_pred, dim=-1)
                    correct_all = (pred_ids_all[mask_16] == m_target[mask_16]).sum().item()
                    stats["acc"] = correct_all / (total_valid_frames * 16)
                else:
                    stats["acc_L0"], stats["top5"], stats["acc"] = 0.0, 0.0, 0.0

            ave_mel_loss += mel_loss.item()
            stats["mel"] = mel_loss.item()
            stats["dur"] = dur_loss.item()
            stats["attn"] = attn_loss.item()
            running_loss += loss.item()
            stats["loss"] = loss.item()
            wandb_run.log({
                "loss": loss.item(),
                "mel_loss": mel_loss.item(),
                "dur_loss": dur_loss.item(),
                "attn_loss": attn_loss.item(),
                "acc_L0": stats["acc_L0"],
                "top5_acc": stats["top5"],
                "acc_all": stats["acc"]
            })
            speed = i / (time.time() - start)
            save_stats(stats, paths, step)

            # 5. Checkpoint và Visualization
            if step % hp.tts_checkpoint_every == 0:
                save_checkpoint('tts', paths, model, optimizer, name=f'step_{step}', is_silent=True)

            if attn_example in ids:
                idx = ids.index(attn_example)
                mel_len = mel_lens[idx].item()
                
                # SỬA LỖI VẼ: p_mel phải là IDs [0-2047], không phải Logits
                p_mel_ids = torch.argmax(mel_pred[idx, :mel_len, :], dim=-1)
                # Vẽ ảnh bằng cách đưa về dải 0-1 để quan sát
                save_spectrogram(np_now(p_mel_ids.float() / 2048.0), paths.tts_mel_plot / f'{step}', 600)
                
                # Lưu attention
                atten = alpha[idx]
                text_len = text_lens[idx].item()
                save_attention(np_now(atten[:text_len, :mel_len]), paths.tts_attention / f'{step}s')

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters})|{stats_str(stats)} {speed:#.2} steps/s'
            if step % hp.tts_show_info_every == 0:
                log.info(msg)

            if step >= train_steps:
                save_checkpoint('tts', paths, model, optimizer, is_silent=False)
                break

        save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        model.log(paths.tts_log, msg)

if __name__ == "__main__":
    main()