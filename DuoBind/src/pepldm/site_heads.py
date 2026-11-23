from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


SITE_HEAD_NAMES = (
    "mlp",
    "local_conv",
    "attention_guided",
    "gated",
    "multiscale",
    "evidence_only",
    "interaction_evidence",
)

EVIDENCE_FEATURE_NAMES = ("max", "topk", "entropy")


def _masked_mean(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=1)
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _attention_prior(
    attention: torch.Tensor,
    pep_mask: Optional[torch.Tensor],
    aggregate: str = "max",
) -> torch.Tensor:
    """Return [B, Lprotein, 1] from [B, heads, Lprotein, Lpeptide]."""
    if attention.ndim != 4:
        raise ValueError(
            "protein-to-peptide attention must have shape "
            "[B, heads, Lprotein, Lpeptide]"
        )
    if pep_mask is not None:
        if pep_mask.shape != (attention.shape[0], attention.shape[-1]):
            raise ValueError("pep_mask does not match attention source length")
        valid = pep_mask.bool()[:, None, None, :]
        masked = attention.masked_fill(~valid, 0.0)
    else:
        valid = None
        masked = attention

    aggregate = aggregate.lower()
    if aggregate == "max":
        prior = masked.max(dim=-1).values
    elif aggregate == "topk":
        valid_count = (
            pep_mask.bool().sum(dim=-1)
            if pep_mask is not None
            else torch.full(
                (attention.shape[0],),
                attention.shape[-1],
                dtype=torch.long,
                device=attention.device,
            )
        )
        k = max(1, min(3, attention.shape[-1]))
        values = masked.topk(k=k, dim=-1).values
        rank_mask = (
            torch.arange(k, device=attention.device)[None, None, None, :]
            < valid_count[:, None, None, None].clamp_max(k)
        )
        prior = (values * rank_mask.to(values.dtype)).sum(dim=-1) / (
            rank_mask.sum(dim=-1).clamp_min(1).to(values.dtype)
        )
    elif aggregate == "entropy":
        probabilities = masked.clamp_min(1e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        if pep_mask is None:
            normalizer = torch.log(
                torch.tensor(
                    float(attention.shape[-1]),
                    device=attention.device,
                    dtype=attention.dtype,
                )
            ).clamp_min(1e-8)
            prior = 1.0 - entropy / normalizer
        else:
            valid_count = pep_mask.bool().sum(dim=-1).to(attention.dtype)
            normalizer = valid_count.clamp_min(2.0).log()
            prior = torch.where(
                valid_count > 1,
                1.0 - entropy / normalizer[:, None, None],
                torch.ones_like(entropy),
            )
    else:
        raise ValueError(
            f"Unknown attention aggregate {aggregate!r}; "
            "expected max, entropy, or topk"
        )
    return prior.mean(dim=1).unsqueeze(-1)


class InteractionEvidenceExtractor(nn.Module):
    """Extract compact residue-level evidence from protein-to-peptide A.

    ``attention`` is normalized softmax attention with shape
    [B, heads, Lprotein, Lpeptide]. The full attention sum over peptide
    positions is always approximately one and is therefore deliberately not
    used as an evidence feature. We first restrict A to peptide residues and
    renormalize that restricted distribution, then compute concentration
    statistics.

    Pre-softmax compatibility logits Z are not available from the current
    ``nn.MultiheadAttention`` call without reimplementing its projection path.
    This extractor therefore implements the explicit-A variant only.
    """

    def __init__(
        self,
        feature_names: tuple[str, ...] | list[str] = EVIDENCE_FEATURE_NAMES,
        topk: int = 3,
        eps: float = 1e-8,
    ):
        super().__init__()
        normalized = tuple(str(name).strip().lower() for name in feature_names)
        unknown = sorted(set(normalized) - set(EVIDENCE_FEATURE_NAMES))
        if unknown:
            raise ValueError(
                f"Unknown evidence features {unknown}; "
                f"expected {EVIDENCE_FEATURE_NAMES}"
            )
        if not normalized:
            raise ValueError("At least one evidence feature is required")
        if topk < 1:
            raise ValueError("topk must be at least one")
        self.feature_names = normalized
        self.topk = int(topk)
        self.eps = float(eps)
        self.output_dim = len(self.feature_names) * 3

    def forward(
        self,
        attention: torch.Tensor,
        protein_mask: torch.Tensor,
        peptide_mask: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if attention.ndim != 4:
            raise ValueError(
                "attention must have shape [B, heads, Lprotein, Lpeptide]"
            )
        batch, heads, protein_length, peptide_length = attention.shape
        if protein_mask.shape != (batch, protein_length):
            raise ValueError("protein_mask does not match attention query length")
        if peptide_mask.shape != (batch, peptide_length):
            raise ValueError("peptide_mask does not match attention key length")

        peptide_valid = peptide_mask.bool()[:, None, None, :]
        protein_valid = protein_mask.bool()[:, None, :, None]
        restricted = attention.float().masked_fill(~peptide_valid, 0.0)
        denominator = restricted.sum(dim=-1, keepdim=True)
        normalized_attention = restricted / denominator.clamp_min(self.eps)
        valid_count = peptide_mask.bool().sum(dim=-1).to(attention.dtype)

        per_head: dict[str, torch.Tensor] = {}
        for feature_name in self.feature_names:
            if feature_name == "max":
                values = normalized_attention.masked_fill(
                    ~peptide_valid,
                    float("-inf"),
                ).max(dim=-1).values
                values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
            elif feature_name == "topk":
                k = min(self.topk, peptide_length)
                values = normalized_attention.masked_fill(
                    ~peptide_valid,
                    float("-inf"),
                ).topk(k=k, dim=-1).values
                rank = torch.arange(k, device=attention.device)
                rank_mask = rank[None, None, None, :] < valid_count[:, None, None, None].clamp_max(k)
                values = (
                    values * rank_mask.to(values.dtype)
                ).sum(dim=-1) / rank_mask.sum(dim=-1).clamp_min(1).to(values.dtype)
            elif feature_name == "entropy":
                entropy = -(
                    normalized_attention
                    * normalized_attention.clamp_min(self.eps).log()
                ).sum(dim=-1)
                normalizer = valid_count.clamp_min(2.0).log()
                values = torch.where(
                    valid_count[:, None, None] > 1,
                    entropy / normalizer[:, None, None].clamp_min(self.eps),
                    torch.zeros_like(entropy),
                )
            else:
                raise AssertionError(f"Unhandled evidence feature {feature_name}")
            values = values * protein_valid.squeeze(-1).to(values.dtype)
            per_head[feature_name] = values

        aggregate_parts = []
        for feature_name in self.feature_names:
            values = per_head[feature_name]
            aggregate_parts.extend(
                [
                    values.mean(dim=1),
                    values.max(dim=1).values,
                    values.std(dim=1, unbiased=False),
                ]
            )
        evidence = torch.stack(aggregate_parts, dim=-1)
        evidence = evidence * protein_mask.to(evidence.dtype).unsqueeze(-1)
        if not return_details:
            return evidence
        return evidence, {
            "per_head": torch.stack(
                [per_head[name] for name in self.feature_names],
                dim=-1,
            ),
            "normalized_attention": normalized_attention,
            "feature_names": self.feature_names,
        }


class SiteMLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        inner_dim = max(1, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, 1),
        )

    def forward(
        self,
        x_prot: torch.Tensor,
        x_pep: Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None,
        pep_mask: Optional[torch.Tensor] = None,
        attention: Optional[torch.Tensor] = None,
        attention_aggregate: str = "max",
        protein_evidence_mask: Optional[torch.Tensor] = None,
        peptide_evidence_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        evidence_features: tuple[str, ...] = EVIDENCE_FEATURE_NAMES,
        evidence_topk: int = 3,
    ) -> torch.Tensor:
        del (
            x_pep,
            prot_mask,
            pep_mask,
            attention,
            attention_aggregate,
            protein_evidence_mask,
            peptide_evidence_mask,
            return_aux,
            evidence_features,
            evidence_topk,
        )
        return self.classifier(x_prot).squeeze(-1)


class LocalConvSiteHead(SiteMLP):
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__(hidden_dim, dropout)
        self.local = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding="same"),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_prot: torch.Tensor,
        x_pep: Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None,
        pep_mask: Optional[torch.Tensor] = None,
        attention: Optional[torch.Tensor] = None,
        attention_aggregate: str = "max",
        protein_evidence_mask: Optional[torch.Tensor] = None,
        peptide_evidence_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        evidence_features: tuple[str, ...] = EVIDENCE_FEATURE_NAMES,
        evidence_topk: int = 3,
    ) -> torch.Tensor:
        del (
            x_pep,
            pep_mask,
            attention,
            attention_aggregate,
            protein_evidence_mask,
            peptide_evidence_mask,
            return_aux,
            evidence_features,
            evidence_topk,
        )
        update = self.local(x_prot.transpose(1, 2)).transpose(1, 2)
        features = self.norm(x_prot + update)
        if prot_mask is not None:
            features = features * prot_mask.to(features.dtype).unsqueeze(-1)
        return self.classifier(features).squeeze(-1)


class AttentionGuidedSiteHead(SiteMLP):
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__(hidden_dim, dropout)
        self.prior_projection = nn.Sequential(
            nn.Linear(1, max(1, hidden_dim // 4)),
            nn.GELU(),
            nn.Linear(max(1, hidden_dim // 4), hidden_dim),
        )
        self.guided_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_prot: torch.Tensor,
        x_pep: Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None,
        pep_mask: Optional[torch.Tensor] = None,
        attention: Optional[torch.Tensor] = None,
        attention_aggregate: str = "max",
        protein_evidence_mask: Optional[torch.Tensor] = None,
        peptide_evidence_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        evidence_features: tuple[str, ...] = EVIDENCE_FEATURE_NAMES,
        evidence_topk: int = 3,
    ) -> torch.Tensor:
        del (
            protein_evidence_mask,
            peptide_evidence_mask,
            return_aux,
            evidence_features,
            evidence_topk,
        )
        if attention is None:
            raise ValueError("attention_guided site head requires attention weights")
        prior = _attention_prior(attention, pep_mask, attention_aggregate)
        features = self.guided_norm(x_prot + self.prior_projection(prior))
        if prot_mask is not None:
            features = features * prot_mask.to(features.dtype).unsqueeze(-1)
        return self.classifier(features).squeeze(-1)


class GatedSiteHead(SiteMLP):
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__(hidden_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_prot: torch.Tensor,
        x_pep: Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None,
        pep_mask: Optional[torch.Tensor] = None,
        attention: Optional[torch.Tensor] = None,
        attention_aggregate: str = "max",
        protein_evidence_mask: Optional[torch.Tensor] = None,
        peptide_evidence_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        evidence_features: tuple[str, ...] = EVIDENCE_FEATURE_NAMES,
        evidence_topk: int = 3,
    ) -> torch.Tensor:
        del (
            attention,
            attention_aggregate,
            protein_evidence_mask,
            peptide_evidence_mask,
            return_aux,
            evidence_features,
            evidence_topk,
        )
        if x_pep is None:
            raise ValueError("gated site head requires peptide features")
        peptide_context = _masked_mean(x_pep, pep_mask).unsqueeze(1)
        gate = self.gate(torch.cat([x_prot, peptide_context.expand_as(x_prot)], dim=-1))
        features = x_prot + gate * x_prot
        if prot_mask is not None:
            features = features * prot_mask.to(features.dtype).unsqueeze(-1)
        return self.classifier(features).squeeze(-1)


class MultiscaleSiteHead(SiteMLP):
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__(hidden_dim, dropout)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=size, padding="same")
                for size in (3, 5, 7)
            ]
        )
        self.fuse = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_prot: torch.Tensor,
        x_pep: Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None,
        pep_mask: Optional[torch.Tensor] = None,
        attention: Optional[torch.Tensor] = None,
        attention_aggregate: str = "max",
        protein_evidence_mask: Optional[torch.Tensor] = None,
        peptide_evidence_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        evidence_features: tuple[str, ...] = EVIDENCE_FEATURE_NAMES,
        evidence_topk: int = 3,
    ) -> torch.Tensor:
        del (
            x_pep,
            pep_mask,
            attention,
            attention_aggregate,
            protein_evidence_mask,
            peptide_evidence_mask,
            return_aux,
            evidence_features,
            evidence_topk,
        )
        transposed = x_prot.transpose(1, 2)
        branches = [conv(transposed).transpose(1, 2) for conv in self.convs]
        features = self.norm(x_prot + self.dropout(self.fuse(torch.cat(branches, dim=-1))))
        if prot_mask is not None:
            features = features * prot_mask.to(features.dtype).unsqueeze(-1)
        return self.classifier(features).squeeze(-1)


class EvidenceOnlySiteHead(nn.Module):
    """SE baseline: residue evidence only, without H_prot."""

    def __init__(
        self,
        evidence_dim: int,
        dropout: float = 0.3,
        evidence_features: tuple[str, ...] = EVIDENCE_FEATURE_NAMES,
        evidence_topk: int = 3,
    ):
        super().__init__()
        inner_dim = max(1, evidence_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(evidence_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, 1),
        )
        self.evidence_predictor = nn.Linear(evidence_dim, 1)
        self.extractor = InteractionEvidenceExtractor(
            evidence_features,
            topk=evidence_topk,
        )

    def forward(
        self,
        x_prot: torch.Tensor,
        x_pep: Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None,
        pep_mask: Optional[torch.Tensor] = None,
        attention: Optional[torch.Tensor] = None,
        attention_aggregate: str = "max",
        protein_evidence_mask: Optional[torch.Tensor] = None,
        peptide_evidence_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        evidence_features: tuple[str, ...] = EVIDENCE_FEATURE_NAMES,
        evidence_topk: int = 3,
    ):
        del x_pep, prot_mask, pep_mask, attention_aggregate
        if attention is None:
            raise ValueError("evidence_only requires protein-to-peptide attention")
        if protein_evidence_mask is None or peptide_evidence_mask is None:
            raise ValueError("Evidence masks are required")
        if (
            tuple(evidence_features) != self.extractor.feature_names
            or evidence_topk != self.extractor.topk
        ):
            raise ValueError(
                "Evidence feature configuration must match the constructed head"
            )
        evidence, details = self.extractor(
            attention,
            protein_evidence_mask,
            peptide_evidence_mask,
            return_details=True,
        )
        logits = self.classifier(evidence).squeeze(-1)
        if not return_aux:
            return logits
        return (
            logits,
            evidence,
            self.evidence_predictor(evidence).squeeze(-1),
            details,
        )


class InteractionEvidenceSiteHead(nn.Module):
    """S2/S3: fuse H_prot with compact interaction evidence."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.3,
        evidence_fusion: str = "concat",
        evidence_features: tuple[str, ...] = EVIDENCE_FEATURE_NAMES,
        evidence_topk: int = 3,
    ):
        super().__init__()
        if evidence_fusion not in ("concat", "gate"):
            raise ValueError("evidence_fusion must be concat or gate")
        self.evidence_fusion = evidence_fusion
        self.extractor = InteractionEvidenceExtractor(
            evidence_features,
            topk=evidence_topk,
        )
        evidence_dim = self.extractor.output_dim
        self.evidence_dim = evidence_dim
        if evidence_fusion == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim + evidence_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(evidence_dim, max(1, hidden_dim // 4)),
                nn.GELU(),
                nn.Linear(max(1, hidden_dim // 4), 1),
                nn.Sigmoid(),
            )
        inner_dim = max(1, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, 1),
        )
        self.evidence_predictor = nn.Linear(evidence_dim, 1)

    def forward(
        self,
        x_prot: torch.Tensor,
        x_pep: Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None,
        pep_mask: Optional[torch.Tensor] = None,
        attention: Optional[torch.Tensor] = None,
        attention_aggregate: str = "max",
        protein_evidence_mask: Optional[torch.Tensor] = None,
        peptide_evidence_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        evidence_features: tuple[str, ...] = EVIDENCE_FEATURE_NAMES,
        evidence_topk: int = 3,
    ):
        del x_pep, attention_aggregate
        if attention is None:
            raise ValueError("interaction_evidence requires protein-to-peptide attention")
        if protein_evidence_mask is None:
            protein_evidence_mask = prot_mask
        if peptide_evidence_mask is None:
            peptide_evidence_mask = pep_mask
        if protein_evidence_mask is None or peptide_evidence_mask is None:
            raise ValueError("Evidence masks are required")
        if tuple(evidence_features) != self.extractor.feature_names or evidence_topk != self.extractor.topk:
            raise ValueError(
                "Evidence feature configuration must match the constructed head"
            )
        evidence, details = self.extractor(
            attention,
            protein_evidence_mask,
            peptide_evidence_mask,
            return_details=True,
        )
        if self.evidence_fusion == "concat":
            features = self.fusion(torch.cat([x_prot, evidence], dim=-1))
        else:
            gate = self.fusion(evidence)
            features = x_prot * (1.0 + gate)
        logits = self.classifier(features).squeeze(-1)
        if not return_aux:
            return logits
        return (
            logits,
            evidence,
            self.evidence_predictor(evidence).squeeze(-1),
            details,
        )


def build_site_head(
    name: str,
    hidden_dim: int,
    dropout: float = 0.3,
    attention_aggregate: str = "max",
    evidence_fusion: str = "concat",
    evidence_features: tuple[str, ...] = EVIDENCE_FEATURE_NAMES,
    evidence_topk: int = 3,
) -> nn.Module:
    del attention_aggregate
    normalized = name.lower()
    if normalized not in SITE_HEAD_NAMES:
        raise ValueError(
            f"Unknown site head {name!r}; expected one of {SITE_HEAD_NAMES}"
        )
    if normalized == "evidence_only":
        head = EvidenceOnlySiteHead(
            evidence_dim=len(evidence_features) * 3,
            dropout=dropout,
            evidence_features=evidence_features,
            evidence_topk=evidence_topk,
        )
        return head
    if normalized == "interaction_evidence":
        return InteractionEvidenceSiteHead(
            hidden_dim,
            dropout=dropout,
            evidence_fusion=evidence_fusion,
            evidence_features=evidence_features,
            evidence_topk=evidence_topk,
        )
    classes = {
        "mlp": SiteMLP,
        "local_conv": LocalConvSiteHead,
        "attention_guided": AttentionGuidedSiteHead,
        "gated": GatedSiteHead,
        "multiscale": MultiscaleSiteHead,
    }
    return classes[normalized](hidden_dim, dropout=dropout)
