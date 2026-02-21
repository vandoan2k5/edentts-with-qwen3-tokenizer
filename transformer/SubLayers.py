import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Tăng cường liên kết Temporal (Lấy cảm hứng từ Conformer Block) """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Lớp chiếu mở rộng
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=1 # Pointwise
        )
        
        # NÂNG CẤP 2: Depthwise Convolution để liên kết các Time-Frames mạnh mẽ
        # Ưu tiên kernel_size lớn hơn (vd config gốc bạn đặt 9, 17...)
        dw_kernel = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
        self.depthwise_conv = nn.Conv1d(
            d_hid,
            d_hid,
            kernel_size=dw_kernel,
            padding=(dw_kernel - 1) // 2,
            groups=d_hid # Depthwise
        )

        # Lớp chiếu thu hẹp
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=1 # Pointwise
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU() # Swish hoạt động tốt hơn ReLU cho Audio

    def forward(self, x):
        residual = x
        
        # Chuyển đổi sang (Batch, Channel, Time) cho Conv1d
        output = x.transpose(1, 2)
        
        output = self.w_1(output)
        output = self.activation(output)
        
        output = self.depthwise_conv(output)
        output = self.activation(output)
        
        output = self.w_2(output)
        output = output.transpose(1, 2) # Trả lại (Batch, Time, Channel)
        
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output