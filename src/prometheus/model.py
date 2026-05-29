from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from prometheus.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        return self.projection(attended)


def _valid_head_count(embedding_dim: int, preferred_heads: int) -> int:
    candidate = min(preferred_heads, embedding_dim)
    while candidate > 1:
        if embedding_dim % candidate == 0:
            return candidate
        candidate -= 1
    return 1


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, mlp_ratio: int, dropout: float):
        super().__init__()
        hidden_dim = embedding_dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(embedding_dim, num_heads, dropout)
        self.feedforward_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForward(embedding_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feedforward(self.feedforward_norm(x))
        return x


@dataclass(slots=True)
class ForwardOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class LanguageModelBase(nn.Module):
    def __init__(self, config: ModelConfig, sequence_length: int):
        super().__init__()
        if not isinstance(config.vocab_size, int):
            raise ValueError("Model vocab_size must be resolved to an integer before construction.")
        self.config = config
        self.sequence_length = sequence_length
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embeddings = nn.Embedding(sequence_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        _, sequence_length = tokens.shape
        if sequence_length > self.sequence_length:
            raise ValueError("Input sequence length exceeds the configured model context window.")
        positions = torch.arange(0, sequence_length, device=tokens.device)
        x = self.token_embeddings(tokens) + self.position_embeddings(positions)[None, :, :]
        return self.dropout(x)

    def _finalize(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> ForwardOutput:
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return ForwardOutput(logits=logits, loss=loss)


class DenseTransformerLM(LanguageModelBase):
    def __init__(self, config: ModelConfig, sequence_length: int):
        super().__init__(config, sequence_length)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self._initialize_weights()
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> ForwardOutput:
        x = self._embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self._finalize(x, targets)


class StaticRouter(nn.Module):
    def __init__(self, num_groups: int, topology: str, top_k: int | None):
        super().__init__()
        self.num_groups = num_groups
        self.topology = topology
        self.top_k = top_k
        self.routing_logits = nn.Parameter(torch.zeros(num_groups, num_groups))
        self.register_buffer("routing_mask", self._build_mask(num_groups, topology), persistent=False)

    @staticmethod
    def _build_mask(num_groups: int, topology: str) -> torch.Tensor:
        mask = torch.zeros(num_groups, num_groups, dtype=torch.bool)
        for destination in range(num_groups):
            for source in range(num_groups):
                if topology == "dense":
                    allowed = True
                elif topology == "local":
                    allowed = source == destination or abs(source - destination) == 1
                elif topology == "small_world":
                    stride = max(1, num_groups // 2)
                    allowed = source == destination or abs(source - destination) == 1 or source == (destination + stride) % num_groups
                else:
                    raise ValueError(f"Unsupported routing topology: {topology}")
                if allowed:
                    mask[destination, source] = True
        return mask

    def forward(self, summaries: torch.Tensor) -> torch.Tensor:
        scores = self.routing_logits.masked_fill(~self.routing_mask, float("-inf"))
        if self.top_k is not None and self.top_k < self.num_groups:
            top_values, top_indices = torch.topk(scores, k=self.top_k, dim=-1)
            del top_values
            top_mask = torch.zeros_like(self.routing_mask)
            top_mask.scatter_(1, top_indices, True)
            scores = scores.masked_fill(~top_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("gh,bhc->bgc", weights, summaries)


class ModularStage(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_ratio: int, dropout: float, group_count: int, depth: int, topology: str, top_k: int | None):
        super().__init__()
        if embedding_dim % group_count != 0:
            raise ValueError("embedding_dim must be divisible by each group count in the stage schedule")
        group_dim = embedding_dim // group_count
        local_heads = _valid_head_count(group_dim, num_heads)
        self.group_count = group_count
        self.group_dim = group_dim
        self.local_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TransformerBlock(
                            embedding_dim=group_dim,
                            num_heads=local_heads,
                            mlp_ratio=mlp_ratio,
                            dropout=dropout,
                        )
                        for _ in range(depth)
                    ]
                )
                for _ in range(group_count)
            ]
        )
        self.router = StaticRouter(group_count, topology, top_k)
        self.router_norm = nn.LayerNorm(group_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape
        grouped = x.view(batch_size, sequence_length, self.group_count, self.group_dim)
        outputs = []
        for group_index, blocks in enumerate(self.local_blocks):
            group_state = grouped[:, :, group_index, :]
            for block in blocks:
                group_state = block(group_state)
            outputs.append(group_state)
        merged = torch.stack(outputs, dim=2)
        summaries = self.router_norm(merged.mean(dim=1))
        routed = self.router(summaries)
        merged = merged + routed[:, None, :, :]
        return merged.reshape(batch_size, sequence_length, embedding_dim)


class ModularTransformerLM(LanguageModelBase):
    def __init__(self, config: ModelConfig, sequence_length: int):
        super().__init__(config, sequence_length)
        group_schedule = config.stage_groups or [2, 1]
        depth_schedule = config.stage_depths or [1 for _ in group_schedule]
        if len(group_schedule) != len(depth_schedule):
            raise ValueError("stage_groups and stage_depths must have the same length")
        self.stages = nn.ModuleList(
            [
                ModularStage(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    group_count=group_count,
                    depth=depth,
                    topology=config.routing_topology,
                    top_k=config.routing_top_k,
                )
                for group_count, depth in zip(group_schedule, depth_schedule, strict=True)
            ]
        )
        self._initialize_weights()
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> ForwardOutput:
        x = self._embed(tokens)
        for stage in self.stages:
            x = stage(x)
        return self._finalize(x, targets)


def build_model(config: ModelConfig, sequence_length: int) -> LanguageModelBase:
    if config.architecture == "dense":
        return DenseTransformerLM(config, sequence_length)
    if config.architecture == "modular":
        return ModularTransformerLM(config, sequence_length)
    raise ValueError(f"Unsupported architecture: {config.architecture}")
