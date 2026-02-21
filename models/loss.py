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
    logits: [Batch, Time, 16, 2048] 
    targets: [Batch, Time, 16] 
    mel_lens: [Batch] 
    """
    # 1. Tạo mask và lọc padding
    mel_masks = ~get_mask_from_lengths(mel_lens, max_len=logits.shape[1]).to(logits.device)
    
    # Lọc lấy các token hợp lệ (Tự động gộp Batch và Time)
    # valid_logits: [N_frames, 16, 2048]
    # valid_targets: [N_frames, 16]
    valid_logits = logits[mel_masks] 
    valid_targets = targets[mel_masks]
    
    # 2. Tính loss riêng cho từng token (reduction='none')
    loss_per_token = F.cross_entropy(
        valid_logits.view(-1, 2048), 
        valid_targets.reshape(-1).long(),
        reduction='none'
    )
    
    # 3. Reshape lại để tách 16 tầng ra: [N_frames, 16]
    loss_per_layer = loss_per_token.view(-1, 16)
    
    # 4. Tạo bộ trọng số giảm dần cho 16 tầng (Tầng 1 quan trọng nhất)
    weights = torch.tensor([0.9 ** i for i in range(16)], device=logits.device)
    
    # 5. Nhân trọng số và tính trung bình
    weighted_loss = loss_per_layer * weights
    return weighted_loss.mean()
