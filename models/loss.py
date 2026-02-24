import torch
import torch.nn as nn
import torch.nn.functional as F

class CifTTSLoss(nn.Module):
    def __init__(self, delta=0.2):
        super().__init__()
        self.delta = delta # Trọng số cho Quantity Loss (từ eden.json)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1) # Giả sử -1 là padding cho audio tokens

    def forward(self, model_outputs, target_audio_tokens, target_audio_lengths):
        """
        Args:
            model_outputs: Dictionary từ CifTTSModel.forward
            target_audio_tokens: [B, T_aud, Num_layers]
            target_audio_lengths: [B] - Độ dài thực tế của audio
        """
        
        # 1. CROSS ENTROPY LOSS (Cho Decoder)
        # decoder_out: [B, T_aud, Num_layers, Vocab_size]
        logits = model_outputs["decoder_out"]
        B, T, L, V = logits.shape
        
        # Flatten để tính CE Loss
        # Logits: [B * T * L, V]
        # Targets: [B * T * L]
        flat_logits = logits.reshape(-1, V)
        flat_targets = target_audio_tokens.reshape(-1)
        
        loss_ce = self.ce_loss(flat_logits, flat_targets)

        # 2. QUANTITY LOSS (Cho CIF)
        # quantity_out: Tổng trọng số CIF dự đoán cho mỗi sample [B]
        # target_audio_lengths: Độ dài audio thực tế [B]
        quantity_out = model_outputs["quantity_out"]
        
        # Chúng ta muốn quantity_out xấp xỉ target_audio_lengths
        loss_quantity = self.mse_loss(quantity_out, target_audio_lengths.float())

        # 3. TOTAL LOSS
        total_loss = loss_ce + self.delta * loss_quantity

        return {
            "total_loss": total_loss,
            "ce_loss": loss_ce,
            "quantity_loss": loss_quantity
        }