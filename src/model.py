import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from .utils import mean_pool


class LogSLMNC(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        projection_dim: int = 128,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.mlm = AutoModelForMaskedLM.from_pretrained(backbone_name)

        hidden_size = self.mlm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
        )

        if freeze_backbone:
            for p in self.mlm.base_model.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = out.hidden_states[-1]
        pooled = mean_pool(hidden, attention_mask)

        z = self.projector(pooled)
        z = F.normalize(z, p=2, dim=-1)

        return {
            "mlm_loss": out.loss,
            "logits": out.logits,
            "embedding": z,
        }
