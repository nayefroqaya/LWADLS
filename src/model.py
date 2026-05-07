from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM


class LogSLMNC(nn.Module):
    """
    Small Language Model for normality modeling.

    Outputs:
        mlm_loss:
            Batch-level MLM loss used for training.

        per_sample_mlm_loss:
            Individual MLM loss per sequence.
            This is better for anomaly scoring.

        embedding:
            Sequence embedding used for center/prototype distance.
    """

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

        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
            nn.Tanh(),
        )

        if freeze_backbone:
            for param in self.mlm.base_model.parameters():
                param.requires_grad = False

    def mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean pooling over non-padding tokens.
        """
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def compute_per_sample_mlm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MLM loss for each sequence separately.

        labels:
            -100 means ignore token.
        """

        vocab_size = logits.size(-1)

        loss_fct = nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=-100,
        )

        token_loss = loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1),
        ).view(labels.size())

        active_mask = (labels != -100).float()

        loss_sum = (token_loss * active_mask).sum(dim=1)
        token_count = active_mask.sum(dim=1).clamp(min=1.0)

        per_sample_loss = loss_sum / token_count

        return per_sample_loss

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
    ):
        out = self.mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = out.hidden_states[-1]
        pooled = self.mean_pool(last_hidden, attention_mask)
        embedding = self.projection(pooled)

        if labels is not None:
            per_sample_mlm_loss = self.compute_per_sample_mlm_loss(
                logits=out.logits,
                labels=labels,
            )
            mlm_loss = per_sample_mlm_loss.mean()
        else:
            per_sample_mlm_loss = None
            mlm_loss = None

        return {
            "mlm_loss": mlm_loss,
            "per_sample_mlm_loss": per_sample_mlm_loss,
            "embedding": embedding,
            "logits": out.logits,
        }