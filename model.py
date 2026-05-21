from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: int = 4
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (self.hidden_size // self.num_heads) % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if int(self.mlp_ratio) != self.mlp_ratio or self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be a positive integer")
        self.mlp_ratio = int(self.mlp_ratio)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def intermediate_size(self) -> int:
        return self.hidden_size * self.mlp_ratio


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed.to(dtype=x.dtype) * self.weight).to(dtype=x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :].to(dtype=dtype)
        sin = emb.sin()[None, None, :, :].to(dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dtype != q.dtype:
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class ChunkedAttention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout

    def forward(self, x: torch.Tensor, weights: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        q_w, k_w, v_w, o_w = weights.unbind(dim=0)

        q = F.linear(x, q_w).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = F.linear(x, k_w).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = F.linear(x, v_w).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return F.linear(attn_output, o_w)


class MLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class ChunkedMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.r = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w1_rcc, w2_rcc = torch.split(weights, [self.r, self.r], dim=0)
        w1 = w1_rcc.reshape(self.r * self.hidden_size, self.hidden_size)
        w2 = w2_rcc.reshape(self.r * self.hidden_size, self.hidden_size).T

        x = F.linear(x, w1)
        x = F.silu(x)
        x = self.dropout(x)
        x = F.linear(x, w2)
        return self.dropout(x)


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class ChunkedBlock(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ChunkedAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = ChunkedMLP(config)

    def forward(self, x: torch.Tensor, weights: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        attn_weights, mlp_weights = torch.split(weights, [4, 2 * self.mlp.r], dim=0)
        x = x + self.self_attn(self.input_layernorm(x), attn_weights, cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x), mlp_weights)
        return x


class ChunkedAttentionBlock(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ChunkedAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, weights: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), weights, cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class ChunkedMlpBlock(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = ChunkedMLP(config)

    def forward(self, x: torch.Tensor, weights: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x), weights)
        return x


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.head_dim, base=config.rope_theta)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Sequence length exceeds max_position_embeddings")
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(x.shape[1], device=x.device, dtype=x.dtype)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return {"logits": logits, "loss": loss}


class ChunkedLlamaForCausalLMBase(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        *,
        block_cls: type[nn.Module],
        num_matrix: int,
        init_chunk_weights: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_matrix = num_matrix

        hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)
        self.rotary_emb = RotaryEmbedding(config.head_dim, base=config.rope_theta)
        self.layers = nn.ModuleList([block_cls(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)

        total_chunks = config.num_layers * num_matrix
        chunk_weights = torch.empty(total_chunks, hidden_size, hidden_size)
        if init_chunk_weights:
            chunk_weights.normal_(mean=0.0, std=hidden_size**-0.5)
        self.chunk_weights = nn.Parameter(chunk_weights)
        self.chunk_affine1 = nn.Parameter(torch.zeros(total_chunks, hidden_size, 1))
        self.chunk_affine2 = nn.Parameter(torch.zeros(total_chunks, 1, hidden_size))

        self.apply(self._init_weights)
        if init_chunk_weights:
            self._init_chunk_weights()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_chunk_weights(self) -> None:
        with torch.no_grad():
            weights = self.chunk_weights.data.to(dtype=torch.float32)
            q, _ = torch.linalg.qr(weights)
            self.chunk_weights.data.copy_(q.to(dtype=self.chunk_weights.dtype))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Sequence length exceeds max_position_embeddings")
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(x.shape[1], device=x.device, dtype=x.dtype)

        chunk_affine = self.chunk_affine1 + self.chunk_affine2 + 1
        block_weights = (self.chunk_weights * chunk_affine).reshape(
            len(self.layers), self.num_matrix, self.hidden_size, self.hidden_size
        )

        for layer, weights in zip(self.layers, block_weights):
            x = layer(x, weights, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return {"logits": logits, "loss": loss}


class ChunkedLlamaForCausalLM(ChunkedLlamaForCausalLMBase):
    def __init__(self, config: LlamaConfig, *, init_chunk_weights: bool = True) -> None:
        super().__init__(
            config,
            block_cls=ChunkedBlock,
            num_matrix=4 + 2 * config.mlp_ratio,
            init_chunk_weights=init_chunk_weights,
        )


class ChunkedAttentionLlamaForCausalLM(ChunkedLlamaForCausalLMBase):
    def __init__(self, config: LlamaConfig, *, init_chunk_weights: bool = True) -> None:
        super().__init__(
            config,
            block_cls=ChunkedAttentionBlock,
            num_matrix=4,
            init_chunk_weights=init_chunk_weights,
        )


class ChunkedMlpLlamaForCausalLM(ChunkedLlamaForCausalLMBase):
    def __init__(self, config: LlamaConfig, *, init_chunk_weights: bool = True) -> None:
        super().__init__(
            config,
            block_cls=ChunkedMlpBlock,
            num_matrix=2 * config.mlp_ratio,
            init_chunk_weights=init_chunk_weights,
        )


def build_model(
    config: LlamaConfig,
    orthogonal_type: str = "none",
    *,
    init_chunk_weights: bool = True,
) -> nn.Module:
    if orthogonal_type == "none":
        return LlamaForCausalLM(config)
    if orthogonal_type == "mlp":
        return ChunkedMlpLlamaForCausalLM(config, init_chunk_weights=init_chunk_weights)
    if orthogonal_type == "atten":
        return ChunkedAttentionLlamaForCausalLM(config, init_chunk_weights=init_chunk_weights)
    if orthogonal_type == "all":
        return ChunkedLlamaForCausalLM(config, init_chunk_weights=init_chunk_weights)
    raise ValueError(f"Unsupported orthogonal_type {orthogonal_type}")
