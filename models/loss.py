import torch
from utils.tools import get_mask_from_lengths
import torch.nn.functional as F


def format_Wnt(N, T, g=0.2):
    n_items = torch.arange(0, N, device=N.device)/N
    t_items = torch.arange(0, T, device=N.device)/T
    w = 1 - torch.exp(-(n_items.unsqueeze(1) - t_items.unsqueeze(0))**2/(2*g**2))
    # 中间的值weighth很小，让loss不会注意到他们，非中间的值weight较大
    return w


def guided_atten_loss_func(a, text_lens, mel_lens):
    loss = 0.0
    for i, (N, T) in enumerate(zip(text_lens, mel_lens)):
        w = format_Wnt(N, T).to(a.device)
        loss += torch.sum(a[i, :N, :T]*w)
    return loss/sum(mel_lens)


def duration_loss_func(d_pred, d_target, ilens):
    duration_masks = ~get_mask_from_lengths(ilens).to(d_pred.device)
    d_outs = d_pred.masked_select(duration_masks)
    ds = d_target.masked_select(duration_masks)
    return F.l1_loss(d_outs, ds)


def pe_loss_func(d_pred, d_target, ilens):
    duration_masks = ~get_mask_from_lengths(ilens).to(d_pred.device)
    d_target.requires_grad = False
    d_outs = d_pred.masked_select(duration_masks)
    ds = d_target.masked_select(duration_masks)
    return F.mse_loss(d_outs, ds)


def mel_loss_func(logits, targets, mel_lens):
    """
    logits: [Batch, Time, N_Layers, Vocab_Size] 
    targets: [Batch, Time, N_Layers] 
    mel_lens: [Batch] 
    """
    # 1. Lấy số lượng layer thực tế (Ví dụ: 4 cho base, 12 cho refine, hoặc 16 cho gốc)
    n_layers = logits.shape[2]
    vocab_size = logits.shape[-1]
    
    # 2. Tạo mask và lọc padding
    mel_masks = ~get_mask_from_lengths(mel_lens, max_len=logits.shape[1]).to(logits.device)
    
    # valid_logits: [N_frames, N_Layers, Vocab_Size]
    # valid_targets: [N_frames, N_Layers]
    valid_logits = logits[mel_masks] 
    valid_targets = targets[mel_masks]
    
    # 3. Tính loss riêng cho từng token (reduction='none')
    loss_per_token = F.cross_entropy(
        valid_logits.view(-1, vocab_size), 
        valid_targets.reshape(-1).long(),
        reduction='none'
    )
    
    # 4. Reshape lại theo đúng số layer TRUYỀN VÀO (thay vì fix cứng 16)
    loss_per_layer = loss_per_token.view(-1, n_layers) # <--- ĐÃ SỬA LỖI Ở ĐÂY
    
    # 5. Đánh trọng số cho các tầng
    weights = torch.ones(n_layers, device=logits.device) # <--- ĐÃ SỬA LỖI CHẬM HỘI TỤ Ở ĐÂY
    
    # 6. Nhân trọng số và tính trung bình
    weighted_loss = loss_per_layer * weights
    return weighted_loss.mean()
