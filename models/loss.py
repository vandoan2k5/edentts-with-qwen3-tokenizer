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
    logits: [Batch, Time, 16, 2048] - Đầu ra từ Decoder mới
    targets: [Batch, Time, 16] - Token IDs nguyên bản (0-2047)
    mel_lens: [Batch] - Độ dài thực của từng câu
    """
    # 1. Tạo mask để loại bỏ phần padding (giống code cũ của bạn)
    mel_masks = ~get_mask_from_lengths(mel_lens, max_len=logits.shape[1]).to(logits.device)
    
    # 2. Lọc lấy các giá trị không phải padding
    # logits[mel_masks] sẽ có shape [Tổng_số_frame_thực, 16, 2048]
    # targets[mel_masks] sẽ có shape [Tổng_số_frame_thực, 16]
    valid_logits = logits[mel_masks] 
    valid_targets = targets[mel_masks]
    
    # 3. Tính Cross Entropy cho toàn bộ 16 tầng RVQ cùng lúc
    # Ta flatten chiều thời gian và chiều 16 tầng để tính cho nhanh
    return F.cross_entropy(
        valid_logits.view(-1, 2048), 
        valid_targets.view(-1).long(),
        label_smoothing=0.1 # Gợi ý: Thêm chút smoothing để mô hình mượt hơn
    )
