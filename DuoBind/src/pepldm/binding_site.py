from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torchmetrics import (
    Accuracy,
    AUROC,
    AveragePrecision,
    F1Score,
    MatthewsCorrCoef,
    MeanMetric,
    Precision,
    Recall,
    Specificity,
)

from loguru import logger

from .base import BaseLightningModule, MetricCollection
from .esmc.configuration_esmc import ESMCConfig
from .esmc.modeling_esmc import ESMCModel, load_weights_from_esm
from .lightning import CrossAttentionHead
from .local_context import LOCAL_CONTEXT_NAMES, build_local_context
from .site_heads import EVIDENCE_FEATURE_NAMES, SITE_HEAD_NAMES, build_site_head
from .tokenizer import ESMSeqTokenizer


class BindingSiteModel(BaseLightningModule):
    """Protein-residue binding-site classifier conditioned on a peptide."""

    def __init__(
        self,
        model_args: dict = {},
        esm_args: dict = {},
        tokenizer: Optional[ESMSeqTokenizer] = None,
        load_from_esmc: bool = False,
        esmc_weights_path: Optional[str] = None,
        strict_esmc_load: bool = False,
        pretrained_ckpt: Optional[str] = None,
        pos_weight: float = 1.0,
        threshold: float = 0.5,
        prediction_dir: Optional[str] = None,
        site_head: str = "mlp",
        site_head_dropout: float = 0.3,
        attention_aggregate: str = "max",
        use_cached_embeddings: bool = False,
        embedding_cache_dir: Optional[str] = None,
        attention_loss_weight: float = 0.0,
        interaction_source: str = "attention",
        evidence_features: str = "max,topk,entropy",
        evidence_fusion: str = "concat",
        evidence_loss_weight: float = 0.0,
        evidence_topk: int = 3,
        local_context: str = "none",
        local_context_width: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.model_args = model_args
        self.esm_args = esm_args
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.prediction_dir = Path(prediction_dir) if prediction_dir else None
        self.site_head_name = site_head.lower()
        self.attention_aggregate = attention_aggregate
        self.use_cached_embeddings = use_cached_embeddings
        self.embedding_cache_dir = embedding_cache_dir
        self.attention_loss_weight = float(attention_loss_weight)
        self.interaction_source = interaction_source
        self.evidence_feature_names = tuple(
            feature.strip().lower()
            for feature in evidence_features.split(",")
            if feature.strip()
        )
        self.evidence_fusion = evidence_fusion
        self.evidence_loss_weight = float(evidence_loss_weight)
        self.evidence_topk = int(evidence_topk)
        self.local_context_name = str(local_context).lower()
        self.local_context_width = int(local_context_width)
        if self.local_context_name not in LOCAL_CONTEXT_NAMES:
            raise ValueError(
                f"Unknown local_context {local_context!r}; "
                f"expected {LOCAL_CONTEXT_NAMES}"
            )
        if self.site_head_name not in SITE_HEAD_NAMES:
            raise ValueError(
                f"Unknown site head {site_head!r}; expected one of {SITE_HEAD_NAMES}"
            )
        if self.attention_loss_weight < 0:
            raise ValueError("attention_loss_weight must be non-negative")
        if self.evidence_loss_weight < 0:
            raise ValueError("evidence_loss_weight must be non-negative")
        if self.interaction_source != "attention":
            raise ValueError(
                "Only interaction_source='attention' is implemented; "
                "pre-softmax logits are not exposed by the current attention path"
            )
        unknown_features = sorted(
            set(self.evidence_feature_names) - set(EVIDENCE_FEATURE_NAMES)
        )
        if unknown_features:
            raise ValueError(
                f"Unknown evidence features {unknown_features}; "
                f"expected {EVIDENCE_FEATURE_NAMES}"
            )
        if self.attention_loss_weight > 0:
            logger.warning(
                "attention_loss_weight is accepted for experiment metadata, but "
                "no auxiliary attention supervision is applied"
            )

        config = ESMCConfig(**esm_args)
        self.hidden_size = config.hidden_size
        self.prot_encoder = None if use_cached_embeddings else ESMCModel(config)
        self.pep_encoder = (
            None if use_cached_embeddings else ESMCModel(ESMCConfig(**esm_args))
        )
        self.affinity_head = CrossAttentionHead(self.hidden_size, num_layers=1)
        self.site_head = build_site_head(
            self.site_head_name,
            self.hidden_size,
            dropout=site_head_dropout,
            attention_aggregate=attention_aggregate,
            evidence_fusion=evidence_fusion,
            evidence_features=self.evidence_feature_names,
            evidence_topk=self.evidence_topk,
        )
        self.local_context = build_local_context(
            mode=self.local_context_name,
            hidden_dim=self.hidden_size,
            bottleneck_dim=self.local_context_width,
            dropout=site_head_dropout,
        )

        if self.prot_encoder is not None:
            for parameter in self.prot_encoder.parameters():
                parameter.requires_grad = False
        if self.pep_encoder is not None:
            for parameter in self.pep_encoder.parameters():
                parameter.requires_grad = False
        for name, parameter in self.affinity_head.named_parameters():
            if name.startswith("classification_head."):
                parameter.requires_grad = False

        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight), dtype=torch.float32),
            persistent=True,
        )
        self.site_criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight,
            reduction="none",
        )
        self._init_metrics()

        if load_from_esmc and not use_cached_embeddings:
            try:
                load_weights_from_esm(self.prot_encoder, esmc_weights_path)
                load_weights_from_esm(self.pep_encoder, esmc_weights_path)
            except Exception as exc:
                message = (
                    "Failed to load pretrained ESMC weights. Check "
                    "esmc_weights_path or the local model cache."
                )
                if strict_esmc_load:
                    raise RuntimeError(message) from exc
                logger.warning(f"{message} Original error: {exc}")

        if pretrained_ckpt:
            self.load_affinity_checkpoint(pretrained_ckpt)

    def _init_metrics(self) -> None:
        metrics = {
            "train_loss": MeanMetric(sync_on_compute=True),
            "train_evidence_loss": MeanMetric(sync_on_compute=True),
        }
        for prefix in ("val", "test"):
            metrics.update(
                {
                    f"{prefix}_site_loss": MeanMetric(sync_on_compute=True),
                    f"{prefix}_evidence_loss": MeanMetric(sync_on_compute=True),
                    f"{prefix}_site_auc": AUROC(
                        task="binary",
                        sync_on_compute=True,
                    ),
                    f"{prefix}_site_prauc": AveragePrecision(
                        task="binary",
                        sync_on_compute=True,
                    ),
                    f"{prefix}_site_f1": F1Score(
                        task="binary",
                        threshold=self.threshold,
                        sync_on_compute=True,
                    ),
                    f"{prefix}_site_mcc": MatthewsCorrCoef(
                        task="binary",
                        threshold=self.threshold,
                        sync_on_compute=True,
                    ),
                    f"{prefix}_site_precision": Precision(
                        task="binary",
                        threshold=self.threshold,
                        sync_on_compute=True,
                    ),
                    f"{prefix}_site_recall": Recall(
                        task="binary",
                        threshold=self.threshold,
                        sync_on_compute=True,
                    ),
                    f"{prefix}_site_specificity": Specificity(
                        task="binary",
                        threshold=self.threshold,
                        sync_on_compute=True,
                    ),
                    f"{prefix}_site_accuracy": Accuracy(
                        task="binary",
                        threshold=self.threshold,
                        sync_on_compute=True,
                    ),
                }
            )
        self.metrics = MetricCollection.from_metrics(metrics)
        self.val_records: list[dict[str, Any]] = []
        self.test_records: list[dict[str, Any]] = []

    def load_affinity_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            raise ValueError(f"No state_dict found in {checkpoint_path}")
        transfer_prefix = "affinity_head.layers."
        model_state = self.state_dict()
        expected_keys = sorted(
            key for key in model_state if key.startswith(transfer_prefix)
        )
        transfer_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(transfer_prefix)
        }
        transfer_keys = sorted(transfer_state)

        missing = sorted(set(expected_keys) - set(transfer_keys))
        unexpected = sorted(set(transfer_keys) - set(expected_keys))
        shape_mismatches = sorted(
            (
                key,
                tuple(model_state[key].shape),
                tuple(transfer_state[key].shape),
            )
            for key in set(expected_keys) & set(transfer_keys)
            if tuple(model_state[key].shape) != tuple(transfer_state[key].shape)
        )
        if missing or unexpected or shape_mismatches:
            raise RuntimeError(
                "Interaction transfer checkpoint is incompatible. "
                f"missing={missing}; unexpected={unexpected}; "
                f"shape_mismatches={shape_mismatches}"
            )

        with torch.no_grad():
            for key in transfer_keys:
                model_state[key].copy_(transfer_state[key].to(model_state[key].device))

        parameter_count = sum(int(transfer_state[key].numel()) for key in transfer_keys)
        logger.info("Interaction transfer source checkpoint: {}", checkpoint_path)
        logger.info("Loaded interaction key count: {}", len(transfer_keys))
        logger.info("Loaded interaction parameter count: {}", parameter_count)
        logger.info("Missing count: {}", len(missing))
        logger.info("Unexpected count: {}", len(unexpected))
        logger.info("Shape mismatch count: {}", len(shape_mismatches))
        logger.info("affinity_head.gate.* = SKIPPED")
        logger.info("affinity_head.classification_head.* = SKIPPED")
        logger.info("encoder parameters = SKIPPED")

    def forward(
        self,
        prot_input_ids: Optional[torch.Tensor] = None,
        prot_attention_mask: Optional[torch.Tensor] = None,
        pep_input_ids: Optional[torch.Tensor] = None,
        pep_attention_mask: Optional[torch.Tensor] = None,
        protein_embeddings: Optional[torch.Tensor] = None,
        peptide_embeddings: Optional[torch.Tensor] = None,
        site_labels: Optional[torch.Tensor] = None,
        site_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if prot_attention_mask is None or pep_attention_mask is None:
            raise ValueError("Both protein and peptide attention masks are required")
        if self.use_cached_embeddings:
            if protein_embeddings is None or peptide_embeddings is None:
                raise ValueError(
                    "Cached mode requires protein_embeddings and peptide_embeddings"
                )
            prot_embeds = protein_embeddings
            pep_embeds = peptide_embeddings
        else:
            if prot_input_ids is None or pep_input_ids is None:
                raise ValueError(
                    "Non-cached mode requires prot_input_ids and pep_input_ids"
                )
            if self.prot_encoder is None or self.pep_encoder is None:
                raise RuntimeError("Protein/peptide encoders are not initialized")
            prot_embeds = self.prot_encoder(
                prot_input_ids,
                prot_attention_mask,
            ).last_hidden_state
            pep_embeds = self.pep_encoder(
                pep_input_ids,
                pep_attention_mask,
            ).last_hidden_state

        # x_prot is the peptide-conditioned residue representation before pooling.
        need_evidence = self.site_head_name in (
            "attention_guided",
            "interaction_evidence",
            "evidence_only",
        )
        need_attention = (
            need_evidence
            or self.attention_loss_weight > 0
            or self.evidence_loss_weight > 0
        )
        head_output = self.affinity_head(
            prot_embeds,
            pep_embeds,
            prot_attention_mask,
            pep_attention_mask,
            return_residue_features=True,
            return_attention=need_attention,
        )
        if need_attention:
            _, residue_features, attention = head_output
        else:
            _, residue_features = head_output
            attention = None
        if self.use_cached_embeddings:
            residue_site_features = residue_features
            protein_evidence_mask = prot_attention_mask
            peptide_evidence_mask = pep_attention_mask
            peptide_site_features = pep_embeds
            residue_lengths = None
        else:
            residue_lengths = (
                prot_attention_mask.to(torch.long).sum(dim=-1) - 2
            ).clamp_min(0)
            peptide_lengths = (
                pep_attention_mask.to(torch.long).sum(dim=-1) - 2
            ).clamp_min(0)
            max_residue_length = int(residue_lengths.max().item())
            max_peptide_length = int(peptide_lengths.max().item())
            residue_site_features = residue_features[
                :, 1 : 1 + max_residue_length, :
            ]
            peptide_site_features = pep_embeds[
                :, 1 : 1 + max_peptide_length, :
            ]
            positions_protein = torch.arange(
                max_residue_length,
                device=prot_attention_mask.device,
            ).unsqueeze(0)
            positions_peptide = torch.arange(
                max_peptide_length,
                device=pep_attention_mask.device,
            ).unsqueeze(0)
            protein_evidence_mask = positions_protein < residue_lengths.unsqueeze(1)
            peptide_evidence_mask = positions_peptide < peptide_lengths.unsqueeze(1)
            if attention is not None:
                attention = attention[
                    :,
                    :,
                    1 : 1 + max_residue_length,
                    1 : 1 + max_peptide_length,
                ]
        if self.local_context is not None:
            residue_site_features = self.local_context(
                residue_site_features,
                protein_evidence_mask,
            )
        site_logits = self.site_head(
            residue_site_features,
            x_pep=peptide_site_features,
            prot_mask=protein_evidence_mask,
            pep_mask=peptide_evidence_mask,
            attention=attention,
            attention_aggregate=self.attention_aggregate,
            protein_evidence_mask=protein_evidence_mask,
            peptide_evidence_mask=peptide_evidence_mask,
            return_aux=need_evidence or self.evidence_loss_weight > 0,
            evidence_features=self.evidence_feature_names,
            evidence_topk=self.evidence_topk,
        )
        evidence = None
        evidence_logits = None
        evidence_details = None
        if isinstance(site_logits, tuple):
            site_logits, evidence, evidence_logits, evidence_details = site_logits
        residue_logits = site_logits
        if self.use_cached_embeddings:
            site_logits = residue_logits
        else:
            site_logits = torch.zeros(
                (
                    residue_logits.shape[0],
                    prot_attention_mask.shape[1],
                ),
                dtype=residue_logits.dtype,
                device=residue_logits.device,
            )
            if residue_logits.shape[1] > 0:
                site_logits[:, 1 : 1 + residue_logits.shape[1]] = residue_logits
        output = {"site_logits": site_logits}
        if attention is not None:
            output["attention"] = attention
        if evidence is not None:
            output["evidence"] = evidence
        if evidence_logits is not None:
            output["evidence_logits"] = evidence_logits
        if evidence_details is not None:
            output["evidence_details"] = evidence_details

        if site_labels is not None:
            if site_mask is None:
                raise ValueError("site_mask is required with site_labels")
            if site_logits.shape != site_labels.shape:
                raise ValueError(
                    f"site_logits {tuple(site_logits.shape)} != "
                    f"site_labels {tuple(site_labels.shape)}"
                )
            token_loss = self.site_criterion(site_logits, site_labels.float())
            output["loss"] = (
                token_loss * site_mask.float()
            ).sum() / site_mask.float().sum().clamp_min(1.0)
            if self.evidence_loss_weight > 0:
                if evidence_logits is None:
                    raise ValueError(
                        "evidence_loss_weight requires a head with evidence logits"
                    )
                if self.use_cached_embeddings:
                    evidence_labels = site_labels.float()
                    evidence_loss_mask = site_mask.float()
                else:
                    evidence_labels = site_labels[
                        :, 1 : 1 + evidence_logits.shape[1]
                    ].float()
                    evidence_loss_mask = site_mask[
                        :, 1 : 1 + evidence_logits.shape[1]
                    ].float()
                evidence_token_loss = self.site_criterion(
                    evidence_logits,
                    evidence_labels,
                )
                evidence_loss = (
                    evidence_token_loss * evidence_loss_mask
                ).sum() / evidence_loss_mask.sum().clamp_min(1.0)
                output["evidence_loss"] = evidence_loss
                output["loss"] = output["loss"] + self.evidence_loss_weight * evidence_loss
            else:
                output["evidence_loss"] = output["loss"].detach() * 0
        return output

    def _step(self, batch: dict[str, Any], prefix: str) -> torch.Tensor:
        output = self(
            prot_input_ids=batch.get("prot_input_ids"),
            prot_attention_mask=batch["prot_attention_mask"],
            pep_input_ids=batch.get("pep_input_ids"),
            pep_attention_mask=batch["pep_attention_mask"],
            protein_embeddings=batch.get("protein_embeddings"),
            peptide_embeddings=batch.get("peptide_embeddings"),
            site_labels=batch["site_labels"],
            site_mask=batch["site_mask"],
        )
        valid = batch["site_mask"].bool()
        logits = output["site_logits"][valid]
        labels = batch["site_labels"][valid].long()
        probabilities = torch.sigmoid(logits)

        if prefix == "train":
            self.metrics.update("train_loss", output["loss"].detach())
            self.metrics.update("train_evidence_loss", output["evidence_loss"].detach())
        else:
            self.metrics.update(f"{prefix}_site_loss", output["loss"].detach())
            self.metrics.update(
                f"{prefix}_evidence_loss",
                output["evidence_loss"].detach(),
            )
            for name in (
                "auc",
                "prauc",
                "f1",
                "mcc",
                "precision",
                "recall",
                "specificity",
                "accuracy",
            ):
                self.metrics.update(
                    f"{prefix}_site_{name}",
                    probabilities.detach(),
                    labels.detach(),
                )
            if prefix == "val":
                self._collect_records(self.val_records, batch, logits)
            elif prefix == "test":
                self._collect_records(self.test_records, batch, logits)
        return output["loss"]

    @staticmethod
    def _collect_records(
        records: list[dict[str, Any]],
        batch: dict[str, Any],
        valid_logits: torch.Tensor,
    ) -> None:
        probabilities = torch.sigmoid(valid_logits).detach().cpu()
        masks = batch["site_mask"].bool().detach().cpu()
        labels = batch["site_labels"].long().detach().cpu()
        offset = 0
        for row_index, sequence in enumerate(batch["protein_sequence"]):
            length = int(masks[row_index].sum().item())
            row_probabilities = probabilities[offset : offset + length]
            row_labels = labels[row_index][masks[row_index]]
            offset += length
            records.append(
                {
                    "sample_id": batch["sample_id"][row_index],
                    "protein_id": batch["protein_id"][row_index],
                    "protein_sequence": sequence,
                    "peptide_sequence": batch["peptide_sequence"][row_index],
                    "labels": row_labels.tolist(),
                    "probabilities": row_probabilities.tolist(),
                }
            )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, "train")
        self.log_step("train_site_loss_step", loss)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._step(batch, "test")

    def on_train_epoch_end(self) -> None:
        self.log_epoch("train_loss", self.metrics.compute("train_loss"))
        self.log_epoch(
            "train_evidence_loss",
            self.metrics.compute("train_evidence_loss"),
        )
        self.metrics.reset_all(prefix="train")

    def _log_site_metrics(self, prefix: str) -> None:
        for name in (
            "site_loss",
            "evidence_loss",
            "site_auc",
            "site_prauc",
            "site_f1",
            "site_mcc",
            "site_precision",
            "site_recall",
            "site_specificity",
            "site_accuracy",
        ):
            self.log_epoch(
                f"{prefix}_{name}",
                self.metrics.compute(f"{prefix}_{name}"),
            )

    def on_validation_epoch_end(self) -> None:
        self._log_site_metrics("val")
        self.metrics.reset_all(prefix="val")
        self.val_records.clear()

    def on_test_epoch_end(self) -> None:
        self._log_site_metrics("test")
        if self.prediction_dir and getattr(self.trainer, "is_global_zero", True):
            self._save_predictions(self.test_records)
        self.metrics.reset_all(prefix="test")
        self.test_records.clear()

    def _save_predictions(self, records: list[dict[str, Any]]) -> None:
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        residue_rows = []
        protein_rows = []
        for record in records:
            predicted = [
                index
                for index, probability in enumerate(record["probabilities"])
                if probability >= self.threshold
            ]
            true_indices = [
                index for index, label in enumerate(record["labels"]) if label == 1
            ]
            for index, (label, probability) in enumerate(
                zip(record["labels"], record["probabilities"])
            ):
                residue_rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "protein_id": record["protein_id"],
                        "peptide_sequence": record["peptide_sequence"],
                        "residue_index": index,
                        "amino_acid": record["protein_sequence"][index],
                        "label": int(label),
                        "probability": float(probability),
                        "prediction": int(probability >= self.threshold),
                    }
                )
            protein_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "protein_id": record["protein_id"],
                    "peptide_sequence": record["peptide_sequence"],
                    "protein_length": len(record["protein_sequence"]),
                    "true_binding_count": len(true_indices),
                    "predicted_binding_count": len(predicted),
                    "true_binding_indices": json.dumps(
                        true_indices,
                        separators=(",", ":"),
                    ),
                    "predicted_binding_indices": json.dumps(
                        predicted,
                        separators=(",", ":"),
                    ),
                    "mean_probability": sum(record["probabilities"])
                    / max(len(record["probabilities"]), 1),
                    "max_probability": max(record["probabilities"], default=0.0),
                }
            )
        _write_csv(self.prediction_dir / "per_residue_predictions.csv", residue_rows)
        _write_csv(self.prediction_dir / "per_protein_predictions.csv", protein_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
