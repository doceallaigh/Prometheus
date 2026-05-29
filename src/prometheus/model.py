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


class DenseTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig, sequence_length: int):
        super().__init__()
        if not isinstance(config.vocab_size, int):
            raise ValueError("Model vocab_size must be resolved to an integer before construction.")
        self.sequence_length = sequence_length
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embeddings = nn.Embedding(sequence_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
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
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> ForwardOutput:
        _, sequence_length = tokens.shape
        if sequence_length > self.sequence_length:
            raise ValueError("Input sequence length exceeds the configured model context window.")
        positions = torch.arange(0, sequence_length, device=tokens.device)
        x = self.token_embeddings(tokens) + self.position_embeddings(positions)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return ForwardOutput(logits=logits, loss=loss)
