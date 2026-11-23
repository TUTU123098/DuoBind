import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence, Tuple, Union, List
from einops import rearrange, repeat
from transformers import (
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from loguru import logger

from .configuration_esmc import ESMCConfig



### Rotary Embeddings
def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    _inplace: bool = False,
) -> torch.Tensor:
    """Apply rotary embeddings to input based on cos and sin."""
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    cos = repeat(cos, "s d -> s 1 (2 d)")
    sin = repeat(sin, "s d -> s 1 (2 d)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class RotaryEmbedding(torch.nn.Module):
    """Rotary position embeddings.

    Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"

    Args:
        dim: Dimension of the embedding
        base: Base for computing angular frequencies
        interleaved: Whether to use interleaved rotations
        scale_base: Base for scaling
        scaling_factor: Factor for scaling positions
        pos_idx_in_fp32: Whether to compute position indices in fp32
        device: Computation device
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        scaling_factor: float = 1.0,
        pos_idx_in_fp32: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.device = device

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the embedding."""
        inv_freq = self._compute_inv_freq(self.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        arange = torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32)
        scale = (
            (arange + 0.4 * self.dim) / (1.4 * self.dim)
            if self.scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute inverse frequency bands."""
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(
        self,
        seqlen: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Update the cached cosine and sine values."""
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)

            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.

        Args:
            q: Query tensor of shape (batch, seqlen, nheads, headdim)
            k: Key tensor of shape (batch, seqlen, nheads, headdim)

        Returns:
            Tuple of rotated query and key tensors
        """
        self._update_cos_sin_cache(q.shape[1], device=q.device, dtype=q.dtype)
        assert self._cos_cached is not None
        assert self._sin_cached is not None
        if self.scale is None:
            return (
                apply_rotary_emb_torch(
                    q,
                    self._cos_cached,
                    self._sin_cached,
                    self.interleaved,
                    True,  # inplace=True
                ),
                apply_rotary_emb_torch(
                    k,
                    self._cos_cached,
                    self._sin_cached,
                    self.interleaved,
                    True,  # inplace=True
                ),
            )  # type: ignore
        else:
            assert False


### Feedforward Network Components
def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    """Compute corrected dimension for SwiGLU."""
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float) -> nn.Sequential:
    """Create SwiGLU feedforward network with layer normalization."""
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=False
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=False),
    )


### Attention
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=False)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_ln = nn.LayerNorm(d_model, bias=False)
        self.k_ln = nn.LayerNorm(d_model, bias=False)

        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights

        Returns:
            Output tensor after self attention, and optionally attention weights
        """
        attn_weights = None
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)
        query_BHLD, key_BHLD, value_BHLD = map(
            self.reshaper, (query_BLD, key_BLD, value_BLD)
        )

        if output_attentions:  # Manual attention computation
            b, h, l, d = query_BHLD.shape
            scale = 1 / math.sqrt(d)
            attn_bias = torch.zeros(
                b, h, l, l, dtype=query_BLD.dtype, device=query_BLD.device
            )
            if attention_mask is not None:
                attn_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
            attn_weights = torch.matmul(query_BHLD, key_BHLD.transpose(-2, -1)) * scale
            attn_weights += attn_bias
            attn_weights = F.softmax(attn_weights, dim=-1)
            context_BHLD = torch.matmul(attn_weights, value_BHLD)
        else:
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD, attention_mask
            )

        context_BLD = rearrange(context_BHLD, "b h s d -> b s (h d)")
        output = self.out_proj(context_BLD)
        return output, attn_weights


class UnifiedTransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        residue_scaling_factor: float = 1,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = swiglu_ln_ffn(d_model, expansion_ratio)
        self.scaling_factor = residue_scaling_factor
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights

        Returns:
            Output tensor after transformer block, and optionally attention weights
        """
        attn_output, attn_weights = self.attn(x, attention_mask, output_attentions)
        x = x + self.dropout(attn_output) / self.scaling_factor
        x = x + self.dropout(self.ffn(x)) / self.scaling_factor
        return x, attn_weights


### Model Outputs
@dataclass
class TransformerOutput(ModelOutput):
    """Output type for transformer encoder."""

    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None


### Transformer Stack
class TransformerStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    residue_scaling_factor=math.sqrt(n_layers / 36),
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> TransformerOutput:

        batch_size, seq_len, _ = x.shape
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None

        if attention_mask is not None:
            attention_mask = (
                attention_mask[:, None, None, :]
                .expand(batch_size, 1, seq_len, seq_len)
                .bool()
            )

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, attn_weights = self._gradient_checkpointing_func(
                    block.__call__,
                    x,
                    attention_mask,
                    output_attentions,
                )
            else:
                x, attn_weights = block(x, attention_mask, output_attentions)

            if attentions is not None:
                attentions += (attn_weights,)

            if output_hidden_states:
                assert hidden_states is not None
                hidden_states += (x,)

        return TransformerOutput(
            last_hidden_state=self.norm(x),
            hidden_states=hidden_states,
            attentions=attentions,
        )


### Support for embedding datasets with low code
class ESMCModel(PreTrainedModel):
    config_class = ESMCConfig
    base_model_prefix = "esmc"
    supports_gradient_checkpointing = True
    all_tied_weights_keys = []

    def __init__(self, config: ESMCConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.all_tied_weights_keys = []
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(self.vocab_size, config.hidden_size)
        self.transformer = TransformerStack(
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.dropout,
        )
        self.init_weights()

    def tie_weights(self, *args, **kwargs):
        return

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def _embed(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.transformer(
            x, attention_mask, output_hidden_states=False, output_attentions=False
        ).last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,  # to play nice with HF adjacent packages
        **kwargs,
    ) -> TransformerOutput:
        """Forward pass for masked language modeling.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            inputs_embeds: Optional precomputed embeddings
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights

        Returns:
            TransformerOutput containing last hidden state and optionally all hidden states and attention weights
        """
        if inputs_embeds is None:
            x = self.embed(input_ids)
        else:
            x = inputs_embeds
        return self.transformer(
            x, attention_mask, output_hidden_states, output_attentions
        )


def load_weights_from_esm(new_model: ESMCModel, weights_path=None):
    if weights_path is not None:
        from pathlib import Path
        weights_path = Path(weights_path).expanduser()
        logger.info(f"Loading local ESMC weights from {weights_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"ESMC weights file not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        missing_keys, unexpected_keys = new_model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Loaded local ESMC weights. Missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
        return new_model

    logger.info(f"Loading pretrained ESM weights from esmc_600m")

    from esm.models.esmc import ESMC

    esm_model = ESMC.from_pretrained("esmc_600m")
    esm_state_dict = esm_model.state_dict()

    # 2. 获取当前模型的 state_dict（target）
    model_state_dict = new_model.state_dict()

    # 3. 筛选能映射的 key（通常你要做 key 重命名或子模块筛选）
    filtered_state_dict = {}
    for k in esm_state_dict:
        if (
            k in model_state_dict
            and esm_state_dict[k].shape == model_state_dict[k].shape
        ):
            filtered_state_dict[k] = esm_state_dict[k]

    # 4. 加载（strict=False）防止报错
    missing_keys, unexpected_keys = new_model.load_state_dict(
        filtered_state_dict, strict=False
    )

    logger.info(f"Loaded {len(filtered_state_dict)} weights from ESM.")
    if missing_keys:
        logger.warning(f"Missing keys in target model: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in pretrained model: {unexpected_keys}")
    del esm_model
    torch.cuda.empty_cache()


def _torch_load_local(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict_local(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _strip_known_prefixes_local(key: str) -> str:
    prefixes = ("module.", "model.", "esmc.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def load_weights_from_esm(new_model: ESMCModel, weights_path=None):
    logger.info("Loading pretrained ESM weights from esmc_600m")

    if weights_path is None:
        from esm.models.esmc import ESMC

        esm_model = ESMC.from_pretrained("esmc_600m")
        esm_state_dict = esm_model.state_dict()
    else:
        from pathlib import Path

        weights_path = Path(weights_path).expanduser()
        logger.info(f"Loading local ESMC weights from {weights_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"ESMC weights file not found: {weights_path}")
        esm_model = None
        esm_state_dict = _extract_state_dict_local(_torch_load_local(weights_path))

    model_state_dict = new_model.state_dict()
    filtered_state_dict = {}
    for key, value in esm_state_dict.items():
        target_key = _strip_known_prefixes_local(key)
        if target_key in model_state_dict and value.shape == model_state_dict[target_key].shape:
            filtered_state_dict[target_key] = value

    missing_keys, unexpected_keys = new_model.load_state_dict(filtered_state_dict, strict=False)

    logger.info(f"Loaded {len(filtered_state_dict)} weights from ESM.")
    if missing_keys:
        logger.warning(f"Missing keys in target model: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in pretrained model: {unexpected_keys}")
    if esm_model is not None:
        del esm_model
    torch.cuda.empty_cache()
