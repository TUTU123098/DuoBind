from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


LOCAL_CONTEXT_NAMES = ("none", "single", "multiscale", "ffn")


class ResidualLocalContextBlock(nn.Module):
    """Small residue-local refinement applied after cross-attention.

    ``single`` and ``multiscale`` mix neighboring residues with depthwise
    convolutions in a 256-dimensional bottleneck. ``ffn`` is a
    parameter-matched position-wise control and never mixes sequence
    positions.
    """

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int = 256,
        mode: str = "single",
        dropout: float = 0.3,
    ):
        super().__init__()
        mode = str(mode).lower()
        if mode not in LOCAL_CONTEXT_NAMES[1:]:
            raise ValueError(
                f"Unknown local-context mode {mode!r}; "
                f"expected {LOCAL_CONTEXT_NAMES[1:]}"
            )
        if hidden_dim < 1 or bottleneck_dim < 1:
            raise ValueError("hidden_dim and bottleneck_dim must be positive")

        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.mode = mode
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_projection = nn.Linear(hidden_dim, bottleneck_dim)

        if mode == "single":
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        bottleneck_dim,
                        bottleneck_dim,
                        kernel_size=3,
                        dilation=1,
                        padding=1,
                        groups=bottleneck_dim,
                    )
                ]
            )
        elif mode == "multiscale":
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        bottleneck_dim,
                        bottleneck_dim,
                        kernel_size=3,
                        dilation=dilation,
                        padding=dilation,
                        groups=bottleneck_dim,
                    )
                    for dilation in (1, 2, 4)
                ]
            )
        else:
            self.convs = nn.ModuleList()

        self.activation = nn.GELU()
        self.output_projection = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _apply_mask(
        values: torch.Tensor,
        residue_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if residue_mask is None:
            return values
        if residue_mask.ndim != 2 or residue_mask.shape != values.shape[:2]:
            raise ValueError(
                "residue_mask must have shape [B, L] matching local features; "
                f"got {tuple(residue_mask.shape)} for {tuple(values.shape)}"
            )
        return values * residue_mask.to(dtype=values.dtype).unsqueeze(-1)

    def forward(
        self,
        hidden: torch.Tensor,
        residue_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(
                f"Local context expects [B, L, D], got {tuple(hidden.shape)}"
            )
        if hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Local context hidden dimension mismatch: "
                f"expected {self.hidden_dim}, got {hidden.shape[-1]}"
            )
        if residue_mask is not None and (
            residue_mask.ndim != 2
            or residue_mask.shape != hidden.shape[:2]
        ):
            raise ValueError(
                "residue_mask must have shape [B, L] matching hidden; "
                f"got {tuple(residue_mask.shape)} for {tuple(hidden.shape)}"
            )

        mask = residue_mask
        normalized = self._apply_mask(self.norm(hidden), mask)
        bottleneck = self._apply_mask(
            self.input_projection(normalized),
            mask,
        )

        if self.mode == "ffn":
            update = self.activation(bottleneck)
        else:
            transposed = bottleneck.transpose(1, 2)
            branches = [conv(transposed) for conv in self.convs]
            if self.mode == "single":
                mixed = branches[0]
            else:
                mixed = torch.stack(branches, dim=0).mean(dim=0)
            mixed = self._apply_mask(mixed.transpose(1, 2), mask)
            update = self.activation(mixed)

        update = self.output_projection(update)
        update = self.dropout(update)
        update = self._apply_mask(update, mask)
        return self._apply_mask(hidden + update, mask)


def build_local_context(
    mode: str,
    hidden_dim: int,
    bottleneck_dim: int = 256,
    dropout: float = 0.3,
) -> Optional[ResidualLocalContextBlock]:
    normalized = str(mode).lower()
    if normalized == "none":
        return None
    return ResidualLocalContextBlock(
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        mode=normalized,
        dropout=dropout,
    )
