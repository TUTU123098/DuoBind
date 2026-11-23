from typing import Optional

import torch
import torch.nn as nn
from torchmetrics import (
    MeanMetric,
    Accuracy,
    AUROC,
    AveragePrecision,
    F1Score,
)

from loguru import logger
from .base import BaseLightningModule, MetricCollection
from .esmc.modeling_esmc import (
    ESMCModel,
    load_weights_from_esm,
)
from .esmc.configuration_esmc import ESMCConfig

from .utils import FocalLoss


class CrossAttentionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pooled_dim = hidden_size * 3
        self.fused_dim = hidden_size * 6

        def make_ffn():
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout),
            )

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        # Change 1: peptide queries protein, keeping the original direction.
                        "pep_attn": nn.MultiheadAttention(
                            embed_dim=hidden_size,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "pep_norm1": nn.LayerNorm(hidden_size),
                        "pep_ffn": make_ffn(),
                        "pep_norm2": nn.LayerNorm(hidden_size),
                        # Change 1: protein queries peptide, adding the reverse direction.
                        "prot_attn": nn.MultiheadAttention(
                            embed_dim=hidden_size,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "prot_norm1": nn.LayerNorm(hidden_size),
                        "prot_ffn": make_ffn(),
                        "prot_norm2": nn.LayerNorm(hidden_size),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        else:
            act = nn.GELU()
        self.act = act

        self.gate = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_size * 2),  # Change 3: input changed from hidden_size * 4 to hidden_size * 6.
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        prot_embeds: torch.Tensor,
        pep_embeds: torch.Tensor,
        prot_mask: Optional[torch.Tensor] = None,
        pep_mask: Optional[torch.Tensor] = None,
        return_residue_features: bool = False,
        return_attention: bool = False,
    ):
        B, Lp, H = prot_embeds.shape
        B2, Ls, H2 = pep_embeds.shape
        assert B == B2 and H == H2, "prot/pep batch or hidden mismatch"

        prot_key_padding_mask = None
        if prot_mask is not None:
            assert prot_mask.shape == (B, Lp)
            prot_key_padding_mask = prot_mask == 0

        pep_key_padding_mask = None
        if pep_mask is not None:
            assert pep_mask.shape == (B, Ls)
            pep_key_padding_mask = pep_mask == 0

        x_pep = pep_embeds
        x_prot = prot_embeds
        pep_attn_weights = None
        prot_attn_weights = None

        for layer in self.layers:
            pep_attn_out, pep_attn_weights = layer["pep_attn"](
                query=x_pep,
                key=x_prot,
                value=x_prot,
                key_padding_mask=prot_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x_pep = layer["pep_norm1"](x_pep + pep_attn_out)
            pep_ffn_out = layer["pep_ffn"](x_pep)
            x_pep = layer["pep_norm2"](x_pep + pep_ffn_out)

            prot_attn_out, prot_attn_weights = layer["prot_attn"](
                query=x_prot,
                key=x_pep,
                value=x_pep,
                key_padding_mask=pep_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x_prot = layer["prot_norm1"](x_prot + prot_attn_out)
            prot_ffn_out = layer["prot_ffn"](x_prot)
            x_prot = layer["prot_norm2"](x_prot + prot_ffn_out)

        # Change 2: protein side three-way pooling now uses bidirectionally updated x_prot.
        if prot_mask is not None:
            prot_mask_expanded = prot_mask.unsqueeze(-1).float()
            x_prot_masked_for_max = x_prot.masked_fill(
                prot_mask_expanded == 0, float("-inf")
            )
            protein_max = torch.max(x_prot_masked_for_max, dim=1).values
            protein_sum = (x_prot * prot_mask_expanded).sum(dim=1)
            protein_count = prot_mask_expanded.sum(dim=1).clamp(min=1e-8)
            protein_mean = protein_sum / protein_count
        else:
            protein_max = torch.max(x_prot, dim=1).values
            protein_mean = torch.mean(x_prot, dim=1)
        protein_scores = pep_attn_weights.mean(dim=(1, 2))
        if prot_mask is not None:
            protein_scores = protein_scores.masked_fill(prot_mask == 0, 0.0)
        protein_scores = protein_scores / (protein_scores.sum(dim=-1, keepdim=True) + 1e-8)
        protein_attn = torch.sum(x_prot * protein_scores.unsqueeze(-1), dim=1)
        protein_pool = torch.cat([protein_max, protein_mean, protein_attn], dim=-1)
        assert protein_pool.size(-1) == self.pooled_dim

        # Change 2: replace the old peptide_pool with interaction_pool from x_pep.
        if pep_mask is not None:
            pep_mask_expanded = pep_mask.unsqueeze(-1).float()
            x_pep_masked_for_max = x_pep.masked_fill(
                pep_mask_expanded == 0, float("-inf")
            )
            inter_max = torch.max(x_pep_masked_for_max, dim=1).values
            inter_sum = (x_pep * pep_mask_expanded).sum(dim=1)
            inter_count = pep_mask_expanded.sum(dim=1).clamp(min=1e-8)
            inter_mean = inter_sum / inter_count
        else:
            inter_max = torch.max(x_pep, dim=1).values
            inter_mean = torch.mean(x_pep, dim=1)
        inter_scores = prot_attn_weights.mean(dim=(1, 2))
        if pep_mask is not None:
            inter_scores = inter_scores.masked_fill(pep_mask == 0, 0.0)
        inter_scores = inter_scores / (inter_scores.sum(dim=-1, keepdim=True) + 1e-8)
        inter_attn = torch.sum(x_pep * inter_scores.unsqueeze(-1), dim=1)
        interaction_pool = torch.cat([inter_max, inter_mean, inter_attn], dim=-1)
        assert interaction_pool.size(-1) == self.pooled_dim

        # Change 3: gate is driven by inter_mean, and fused is [B, hidden_size * 6].
        gate = self.gate(inter_mean)
        gated_fusion = gate * protein_pool + (1 - gate) * interaction_pool
        fused = torch.cat([gated_fusion, interaction_pool], dim=-1)
        assert fused.size(-1) == self.fused_dim

        cls_out = self.classification_head(fused).squeeze(-1)
        if return_residue_features and return_attention:
            return cls_out, x_prot, prot_attn_weights
        if return_residue_features:
            return cls_out, x_prot
        if return_attention:
            return cls_out, prot_attn_weights
        return cls_out


class PepLDM2(BaseLightningModule):
    def __init__(
        self,
        model_args: dict = {},
        esm_args: dict = {},
        tokenizer=None,
        load_from_esmc: bool = False,
        esmc_weights_path: Optional[str] = None,
        strict_esmc_load: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.esm_args = esm_args
        config = ESMCConfig(**esm_args)
        self.prot_encoder = ESMCModel(config)
        self.pep_encoder = ESMCModel(config)
        self.affinity_head = CrossAttentionHead(
            config.hidden_size,
            num_layers=1,
        )

        for param in self.prot_encoder.parameters():
            param.requires_grad = False
        for param in self.pep_encoder.parameters():
            param.requires_grad = False
        for param in self.affinity_head.parameters():
            param.requires_grad = True
        if load_from_esmc:
            try:
                load_weights_from_esm(self.prot_encoder, esmc_weights_path)
                load_weights_from_esm(self.pep_encoder, esmc_weights_path)
            except Exception as exc:
                message = (
                    "Failed to load pretrained ESMC weights. This usually means "
                    "the local ESMC weights path is wrong, or the Hugging Face "
                    "model is not cached locally and the current environment "
                    "cannot reach huggingface.co."
                )
                if strict_esmc_load:
                    raise RuntimeError(message) from exc
                logger.warning(f"{message} Original error: {exc}")

        self._init_metrics()
        self.criterion = FocalLoss(task="binary", alpha=0.83, gamma=2)

    def _init_metrics(self):
        self._val_has_binary_labels = False
        self._test_has_binary_labels = False
        self.val_logits = []
        self.val_labels = []
        self.test_logits = []
        self.test_labels = []
        self.metrics: MetricCollection = MetricCollection.from_metrics(
            {
                "train_loss": MeanMetric(sync_on_compute=True),
                "val_loss": MeanMetric(sync_on_compute=True),
                "val_acc": Accuracy(sync_on_compute=True, task="binary"),
                "val_auc": AUROC(sync_on_compute=True, task="binary"),
                "val_prauc": AveragePrecision(sync_on_compute=True, task="binary"),
                "val_f1": F1Score(sync_on_compute=True, task="binary"),
                "test_loss": MeanMetric(sync_on_compute=True),
                "test_acc": Accuracy(sync_on_compute=True, task="binary"),
                "test_auc": AUROC(sync_on_compute=True, task="binary"),
                "test_prauc": AveragePrecision(sync_on_compute=True, task="binary"),
                "test_f1": F1Score(sync_on_compute=True, task="binary"),
            }
        )

    def forward(
        self,
        prot_input_ids,
        prot_attention_mask,
        pep_input_ids,
        pep_attention_mask,
        labels,
    ):

        prot_embeds = self.prot_encoder(
            prot_input_ids, prot_attention_mask
        ).last_hidden_state
        pep_embeds = self.pep_encoder(
            pep_input_ids, pep_attention_mask
        ).last_hidden_state

        cls_out = self.affinity_head(
            prot_embeds,
            pep_embeds,
            prot_attention_mask,
            pep_attention_mask,
        )
        loss = self.criterion(cls_out, labels.float())
        return {
            "loss": loss,
            "preds": cls_out,
            "labels": labels,
        }

    def training_step(self, batch, batch_idx):
        outputs = self(
            prot_input_ids=batch["prot_input_ids"],
            prot_attention_mask=batch["prot_attention_mask"],
            pep_input_ids=batch["pep_input_ids"],
            pep_attention_mask=batch["pep_attention_mask"],
            labels=batch["label"],
        )
        self.log_step(
            "train_loss_step",
            outputs["loss"],
        )
        self.metrics.update("train_loss", outputs["loss"])
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self(
            prot_input_ids=batch["prot_input_ids"],
            prot_attention_mask=batch["prot_attention_mask"],
            pep_input_ids=batch["pep_input_ids"],
            pep_attention_mask=batch["pep_attention_mask"],
            labels=batch["label"],
        )

        metric_preds = outputs["preds"].detach()
        metric_labels = outputs["labels"].detach().long()
        self.val_logits.append(metric_preds.cpu())
        self.val_labels.append(metric_labels.cpu())

        self._val_has_binary_labels = True
        self.metrics.update("val_acc", metric_preds, metric_labels)
        self.metrics.update("val_auc", metric_preds, metric_labels)
        self.metrics.update("val_prauc", metric_preds, metric_labels)
        self.metrics.update("val_f1", metric_preds, metric_labels)
        self.metrics.update("val_loss", outputs["loss"].detach())

    def test_step(self, batch, batch_idx):
        outputs = self(
            prot_input_ids=batch["prot_input_ids"],
            prot_attention_mask=batch["prot_attention_mask"],
            pep_input_ids=batch["pep_input_ids"],
            pep_attention_mask=batch["pep_attention_mask"],
            labels=batch["label"],
        )

        metric_preds = outputs["preds"].detach()
        metric_labels = outputs["labels"].detach().long()
        self.test_logits.append(metric_preds.cpu())
        self.test_labels.append(metric_labels.cpu())

        self._test_has_binary_labels = True
        self.metrics.update("test_acc", metric_preds, metric_labels)
        self.metrics.update("test_auc", metric_preds, metric_labels)
        self.metrics.update("test_prauc", metric_preds, metric_labels)
        self.metrics.update("test_f1", metric_preds, metric_labels)
        self.metrics.update("test_loss", outputs["loss"].detach())

    def on_train_epoch_end(self):
        self.log_epoch(
            "train_loss",
            self.metrics.compute("train_loss"),
        )
        self.metrics.reset_all(prefix="train")

    def on_validation_epoch_end(self):
        self.log_epoch(
            "val_loss",
            self.metrics.compute("val_loss"),
        )
        if self._val_has_binary_labels:
            self.log_epoch(
                "val_acc",
                self.metrics.compute("val_acc"),
            )
            self.log_epoch(
                "val_auc",
                self.metrics.compute("val_auc"),
            )
            self.log_epoch(
                "val_prauc",
                self.metrics.compute("val_prauc"),
            )
            self.log_epoch(
                "val_f1",
                self.metrics.compute("val_f1"),
            )
            logits = torch.cat(self.val_logits).view(-1)
            labels = torch.cat(self.val_labels).view(-1)
            probs = torch.sigmoid(logits)

            best_f1 = 0.0
            best_threshold = 0.5
            best_acc = 0.0

            for threshold in torch.linspace(0.05, 0.95, 91):
                preds = (probs >= threshold).long()
                tp = ((preds == 1) & (labels == 1)).sum().item()
                fp = ((preds == 1) & (labels == 0)).sum().item()
                fn = ((preds == 0) & (labels == 1)).sum().item()
                tn = ((preds == 0) & (labels == 0)).sum().item()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(threshold)
                    best_acc = acc

            self.log_epoch("val_best_f1", best_f1)
            self.log_epoch("val_best_threshold", best_threshold)
            self.log_epoch("val_acc_at_best_threshold", best_acc)
        self.metrics.reset_all(prefix="val")
        self.val_logits.clear()
        self.val_labels.clear()
        self._val_has_binary_labels = False

    def on_test_epoch_end(self):
        self.log_epoch(
            "test_loss",
            self.metrics.compute("test_loss"),
        )
        if self._test_has_binary_labels:
            self.log_epoch(
                "test_acc",
                self.metrics.compute("test_acc"),
            )
            self.log_epoch(
                "test_auc",
                self.metrics.compute("test_auc"),
            )
            self.log_epoch(
                "test_prauc",
                self.metrics.compute("test_prauc"),
            )
            self.log_epoch(
                "test_f1",
                self.metrics.compute("test_f1"),
            )
            logits = torch.cat(self.test_logits).view(-1)
            labels = torch.cat(self.test_labels).view(-1)
            probs = torch.sigmoid(logits)

            best_f1 = 0.0
            best_threshold = 0.5
            best_acc = 0.0

            for threshold in torch.linspace(0.05, 0.95, 91):
                preds = (probs >= threshold).long()
                tp = ((preds == 1) & (labels == 1)).sum().item()
                fp = ((preds == 1) & (labels == 0)).sum().item()
                fn = ((preds == 0) & (labels == 1)).sum().item()
                tn = ((preds == 0) & (labels == 0)).sum().item()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(threshold)
                    best_acc = acc

            self.log_epoch("test_best_f1", best_f1)
            self.log_epoch("test_best_threshold", best_threshold)
            self.log_epoch("test_acc_at_best_threshold", best_acc)
        self.metrics.reset_all(prefix="test")
        self.test_logits.clear()
        self.test_labels.clear()
        self._test_has_binary_labels = False
