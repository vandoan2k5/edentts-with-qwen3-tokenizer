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

            # 2. Forward model (nhận về mel_pred là Logits [B, T, 16, 2048])
            outputs = model(phones, text_lens, m_target, mel_lens, e_mask)
            step = model.get_step()
            
            # Bóc tách output
            log_dur_pred, log_dur_target, mel_pred, alpha, reconst_alpha = outputs

            # 3. Tính toán Loss Tách biệt
            # Lấy 4 layer đầu cho Base và 12 layer sau cho Refine
            mel_pred_base = mel_pred[:, :, :4, :]
            m_target_base = m_target[:, :, :4]
            
            mel_pred_refine = mel_pred[:, :, 4:, :]
            m_target_refine = m_target[:, :, 4:]
            
            # Tính loss riêng lẻ
            # Tính loss riêng lẻ (Cross Entropy)
            mel_loss_refine = mel_loss_func(mel_pred_refine, m_target_refine, mel_lens)
            mel_loss_base = mel_loss_func(mel_pred_base, m_target_base, mel_lens)

            # ==================================================
            # THÊM CONTINUOUS EMBEDDING LOSS (ĐÃ SỬA LỖI MASK)
            # ==================================================
            emb_weight = model.audio_emb.weight.detach() # [2048, hidden_dim]

            # 1. Tạo mask để CHỈ lấy các frame âm thanh thực tế, bỏ qua padding
            valid_mask = ~get_mask_from_lengths(mel_lens, max_len=mel_pred.shape[1]).to(device)

            # 2. Trích xuất layer 0
            pred_L0 = mel_pred_base[:, :, 0, :] # [B, T, 2048]
            target_L0 = m_target_base[:, :, 0]  # [B, T]

            # 3. Lọc qua mask: Ép thành mảng 2D chỉ chứa các frame hợp lệ
            valid_pred_L0 = pred_L0[valid_mask]   # [N_valid_frames, 2048]
            valid_target_L0 = target_L0[valid_mask] # [N_valid_frames]

            # 4. Tính xác suất (Softmax) và Soft-Embedding
            probs_L0 = torch.nn.functional.softmax(valid_pred_L0, dim=-1)
            soft_emb_pred = torch.matmul(probs_L0, emb_weight) # [N_valid_frames, hidden_dim]

            # 5. Lấy Hard-Embedding thực tế
            hard_emb_target = model.audio_emb(valid_target_L0) # [N_valid_frames, hidden_dim]

            # 6. Tính độ lệch MSE (Chỉ trên các frame có ý nghĩa)
            continuous_loss = torch.nn.functional.mse_loss(soft_emb_pred, hard_emb_target)

            # ==================================================
            # GỘP TỔNG LOSS
            # ==================================================
            # Ở đây mình nâng nhẹ trọng số của Refine lên 1.5 để mô hình khôi phục chi tiết sóng âm tốt hơn
            if e <= 15:  # Trong 10 epoch đầu, tập trung vào Base để mô hình học tốt cấu trúc chung
                mel_loss = mel_loss_base + 1.0 * continuous_loss + 0.1 * mel_loss_refine
            else:  # Sau đó, tăng dần trọng số của Refine để cải thiện
                mel_loss = 0.5 * mel_loss_base + 2.0 * continuous_loss + 2.0 * mel_loss_refine
            
            dur_loss = duration_loss_func(log_dur_pred, log_dur_target, text_lens)
            attn_loss = guided_atten_loss_func(alpha, text_lens, mel_lens)
            
            loss = dur_loss + mel_loss + attn_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 4. Tính toán Stats và Accuracy
            stats = dict()
            with torch.no_grad():
                valid_mask = ~get_mask_from_lengths(mel_lens, max_len=mel_pred.shape[1]).to(device)
                total_valid_frames = valid_mask.sum().item()
                
                if total_valid_frames > 0:
                    # --- A. ACCURACY TẦNG 0 ---
                    pred_ids_L0 = torch.argmax(mel_pred[:, :, 0, :], dim=-1)
                    target_L0 = m_target[:, :, 0]
                    correct_L0 = (pred_ids_L0[valid_mask] == target_L0[valid_mask]).sum().item()
                    stats["acc_L0"] = correct_L0 / total_valid_frames

                    pred_ids_L1 = torch.argmax(mel_pred[:, :, 1, :], dim=-1)
                    target_L1 = m_target[:, :, 1]
                    correct_L1 = (pred_ids_L1[valid_mask] == target_L1[valid_mask]).sum().item()
                    stats["acc_L1"] = correct_L1 / total_valid_frames

                    pred_ids_L2 = torch.argmax(mel_pred[:, :, 2, :], dim=-1)
                    target_L2 = m_target[:, :, 2]
                    correct_L2 = (pred_ids_L2[valid_mask] == target_L2[valid_mask]).sum().item()
                    stats["acc_L2"] = correct_L2 / total_valid_frames   

                    pred_ids_L3 = torch.argmax(mel_pred[:, :, 3, :], dim=-1)
                    target_L3 = m_target[:, :, 3]
                    correct_L3 = (pred_ids_L3[valid_mask] == target_L3[valid_mask]).sum().item()
                    stats["acc_L3"] = correct_L3 / total_valid_frames
                    
                    # --- B. TOP-5 ACCURACY ---
                    _, top5_preds = torch.topk(mel_pred, k=5, dim=-1)
                    target_expanded = m_target.unsqueeze(-1)
                    correct_top5 = (top5_preds == target_expanded).any(dim=-1)
                    mask_16 = valid_mask.unsqueeze(-1).expand(-1, -1, 16)
                    top5_correct = correct_top5[mask_16].sum().item()
                    stats["top5"] = top5_correct / (total_valid_frames * 16)

                    # --- C. ACCURACY CŨ ---
                    pred_ids_all = torch.argmax(mel_pred, dim=-1)
                    correct_all = (pred_ids_all[mask_16] == m_target[mask_16]).sum().item()
                    stats["acc"] = correct_all / (total_valid_frames * 16)
                else:
                    stats["acc_L0"], stats["top5"], stats["acc"] = 0.0, 0.0, 0.0

            ave_mel_loss += mel_loss.item()
            stats["mel"] = mel_loss.item()
            # LƯU Ý THÊM: Log thêm 2 loss con
            stats["mel_base"] = mel_loss_base.item()
            stats["mel_refine"] = mel_loss_refine.item()
            stats["continuous_emb"] = continuous_loss.item()
            stats["dur"] = dur_loss.item()
            stats["attn"] = attn_loss.item()
            running_loss += loss.item()
            stats["loss"] = loss.item()
            
            wandb_run.log({
                "loss": loss.item(),
                "mel_loss": mel_loss.item(),
                "mel_loss_base": mel_loss_base.item(),
                "mel_loss_refine": mel_loss_refine.item(),
                "dur_loss": dur_loss.item(),
                "attn_loss": attn_loss.item(),
                "acc_L0": stats["acc_L0"],
                "acc_L1": stats["acc_L1"],
                "acc_L2": stats["acc_L2"],
                "acc_L3": stats["acc_L3"],
                "top5_acc": stats["top5"],
                "acc_all": stats["acc"],
                "continuous_emb_loss": stats["continuous_emb"]
            })
            speed = i / (time.time() - start)
            save_stats(stats, paths, step)

            # 5. Checkpoint và Visualization
            if step % hp.tts_checkpoint_every == 0:
                save_checkpoint('tts', paths, model, optimizer, name=f'step_{step}', is_silent=True)

            if attn_example in ids:
                idx = ids.index(attn_example)
                mel_len = mel_lens[idx].item()
                
                p_mel_ids = torch.argmax(mel_pred[idx, :mel_len, :], dim=-1)
                save_spectrogram(np_now(p_mel_ids.float() / 2048.0), paths.tts_mel_plot / f'{step}', 600)
                
                atten = alpha[idx]
                text_len = text_lens[idx].item()
                save_attention(np_now(atten[:text_len, :mel_len]), paths.tts_attention / f'{step}s')

            # Cập nhật hiển thị msg để print cả mel_base và mel_refine
            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters})| {stats_str(stats)} speed:{speed:#.2} steps/s'
            if step % hp.tts_show_info_every == 0:
                log.info(msg)

            if step >= train_steps:
                save_checkpoint('tts', paths, model, optimizer, is_silent=False)
                break

        save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        model.log(paths.tts_log, msg)

if __name__ == "__main__":
    main()