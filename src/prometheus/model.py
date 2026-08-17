from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from prometheus.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention for causal language modeling."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        """Initialize the projection layers and attention head geometry.

        Args:
            embedding_dim: Width of the model hidden state.
            num_heads: Number of attention heads.
            dropout: Dropout probability applied inside attention.
        """

        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention to a batch of token embeddings.

        Args:
            x: Input tensor shaped ``(batch, sequence, embedding_dim)``.

        Returns:
            torch.Tensor: Attention-updated hidden states.
        """

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
    """Choose the largest head count up to the preference that divides the width.

    Args:
        embedding_dim: Width of the tensor that will be split into heads.
        preferred_heads: Desired head count before divisibility adjustment.

    Returns:
        int: Valid head count that evenly divides the embedding dimension.
    """

    candidate = min(preferred_heads, embedding_dim)
    while candidate > 1:
        if embedding_dim % candidate == 0:
            return candidate
        candidate -= 1
    return 1


class FeedForward(nn.Module):
    """Transformer MLP block with GELU activation and dropout."""

    def __init__(self, embedding_dim: int, mlp_ratio: int, dropout: float):
        """Create the two-layer feedforward network for one transformer block.

        Args:
            embedding_dim: Width of the input and output activations.
            mlp_ratio: Expansion factor used for the hidden layer.
            dropout: Dropout probability after the projection back to model width.
        """

        super().__init__()
        hidden_dim = embedding_dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform token embeddings through the block MLP.

        Args:
            x: Input tensor of token embeddings.

        Returns:
            torch.Tensor: Feedforward-transformed embeddings.
        """

        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention and feedforward residual paths."""

    def __init__(self, embedding_dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        """Build the normalization, attention, and feedforward submodules.

        Args:
            embedding_dim: Width of the block hidden state.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion factor for the feedforward hidden width.
            dropout: Dropout probability used inside the block.
        """

        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(embedding_dim, num_heads, dropout)
        self.feedforward_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForward(embedding_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and feedforward updates with residual connections.

        Args:
            x: Input tensor of hidden states.

        Returns:
            torch.Tensor: Updated hidden states after the residual block.
        """

        x = x + self.attention(self.attention_norm(x))
        x = x + self.feedforward(self.feedforward_norm(x))
        return x


@dataclass(slots=True)
class ForwardOutput:
    """Model outputs consisting of logits and an optional training loss."""

    logits: torch.Tensor
    loss: torch.Tensor | None


@dataclass(slots=True)
class ModularStageSpec:
    """Resolved width and routing geometry for one modular stage."""

    group_count: int
    group_dim: int
    stage_dim: int
    depth: int


@dataclass(slots=True)
class ModularLayoutConfig:
    """Normalized modular-layout settings shared by modular-like architectures."""

    group_schedule: list[int]
    depth_schedule: list[int]
    fixed_group_size: int | None
    routing_topology: str
    routing_top_k: int | None
    recombination_mode: str
    unit_label: str


def _resolve_depth_schedule(requested_depths: list[int] | None, stage_count: int) -> list[int]:
    """Resolve per-stage depths, optionally repeating a single provided value."""

    if requested_depths is None:
        return [1 for _ in range(stage_count)]
    if len(requested_depths) == 1:
        return [requested_depths[0] for _ in range(stage_count)]
    if len(requested_depths) != stage_count:
        raise ValueError("stage_groups and stage_depths must have the same length")
    return requested_depths


def _estimate_modular_parameter_count(config: ModelConfig, sequence_length: int, group_schedule: list[int], depth_schedule: list[int]) -> int:
    """Estimate exact parameter count by instantiating a temporary explicit layout."""

    explicit_config = replace(
        config,
        column_counts=group_schedule,
        column_depths=depth_schedule,
        column_input_count=None,
        column_branching_factor=None,
        target_parameter_count=None,
        max_column_stages=None,
    )
    return sum(parameter.numel() for parameter in ModularTransformerLM(explicit_config, sequence_length).parameters())


def _resolve_fractal_column_schedule(config: ModelConfig, sequence_length: int) -> tuple[list[int], list[int]]:
    """Generate a powers-of-N cortical-column schedule up to a parameter target."""

    if config.column_input_count is None or config.column_branching_factor is None or config.target_parameter_count is None:
        raise ValueError("fractal cortical_columns requires column_input_count, column_branching_factor, and target_parameter_count")
    if not isinstance(config.vocab_size, int):
        raise ValueError("fractal cortical_columns requires vocab_size to be resolved before model construction")
    if config.column_input_count <= 0:
        raise ValueError("column_input_count must be a positive integer")
    if config.column_branching_factor < 2:
        raise ValueError("column_branching_factor must be at least 2")
    if config.target_parameter_count <= 0:
        raise ValueError("target_parameter_count must be a positive integer")
    if config.fixed_column_size is None:
        raise ValueError("fractal cortical_columns currently requires fixed_column_size")

    max_column_stages = config.max_column_stages or 8
    if max_column_stages <= 0:
        raise ValueError("max_column_stages must be a positive integer when provided")

    group_schedule = [config.column_input_count]
    depth_schedule = _resolve_depth_schedule(config.column_depths, len(group_schedule))
    current_parameter_count = _estimate_modular_parameter_count(config, sequence_length, group_schedule, depth_schedule)
    while len(group_schedule) < max_column_stages:
        next_group_schedule = [*group_schedule, group_schedule[-1] * config.column_branching_factor]
        next_depth_schedule = _resolve_depth_schedule(config.column_depths, len(next_group_schedule))
        next_parameter_count = _estimate_modular_parameter_count(config, sequence_length, next_group_schedule, next_depth_schedule)
        if next_parameter_count >= config.target_parameter_count:
            if abs(next_parameter_count - config.target_parameter_count) <= abs(current_parameter_count - config.target_parameter_count):
                return next_group_schedule, next_depth_schedule
            return group_schedule, depth_schedule
        group_schedule = next_group_schedule
        depth_schedule = next_depth_schedule
        current_parameter_count = next_parameter_count
    return group_schedule, depth_schedule


class LanguageModelBase(nn.Module):
    """Shared embedding and output-head utilities for Prometheus language models."""

    def __init__(self, config: ModelConfig, sequence_length: int):
        """Initialize token embeddings, positions, normalization, and LM head.

        Args:
            config: Resolved model hyperparameters.
            sequence_length: Maximum supported context length.
        """

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
        """Initialize linear and embedding weights with small normal noise.

        Returns:
            None: Parameters are initialized in place.
        """

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed token ids and add learned positional embeddings.

        Args:
            tokens: Token ids shaped ``(batch, sequence)``.

        Returns:
            torch.Tensor: Embedded token representations with positions added.
        """

        _, sequence_length = tokens.shape
        if sequence_length > self.sequence_length:
            raise ValueError("Input sequence length exceeds the configured model context window.")
        positions = torch.arange(0, sequence_length, device=tokens.device)
        x = self.token_embeddings(tokens) + self.position_embeddings(positions)[None, :, :]
        return self.dropout(x)

    def _finalize(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> ForwardOutput:
        """Project hidden states to logits and optionally compute cross-entropy loss.

        Args:
            x: Hidden states produced by the model body.
            targets: Optional next-token targets for loss computation.

        Returns:
            ForwardOutput: Logits and optional cross-entropy loss.
        """

        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return ForwardOutput(logits=logits, loss=loss)


class DenseTransformerLM(LanguageModelBase):
    """Standard dense transformer baseline used as the experiment control."""

    def __init__(self, config: ModelConfig, sequence_length: int):
        """Construct the configured stack of dense transformer blocks.

        Args:
            config: Resolved model hyperparameters.
            sequence_length: Maximum supported context length.
        """

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
        """Run dense transformer blocks over token inputs and return model outputs.

        Args:
            tokens: Input token ids.
            targets: Optional next-token targets for loss computation.

        Returns:
            ForwardOutput: Dense-model logits and optional loss.
        """

        x = self._embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self._finalize(x, targets)


class RecurrentLoopTransformerLM(LanguageModelBase):
    """Shared-weight recurrent transformer loop for iterative hidden-state refinement."""

    def __init__(self, config: ModelConfig, sequence_length: int):
        """Construct a shared transformer trunk that is unrolled for multiple recurrent steps.

        Args:
            config: Resolved model hyperparameters.
            sequence_length: Maximum supported context length.
        """

        super().__init__(config, sequence_length)
        if config.num_layers <= 0:
            raise ValueError("recurrent_loop requires num_layers to be at least 1")
        if config.recurrent_steps is None or config.recurrent_steps <= 0:
            raise ValueError("recurrent_loop requires recurrent_steps to be a positive integer")
        if config.recurrent_state_blend is None or not 0.0 <= config.recurrent_state_blend <= 1.0:
            raise ValueError("recurrent_loop requires recurrent_state_blend in the inclusive range [0.0, 1.0]")

        self.recurrent_steps = config.recurrent_steps
        self.recurrent_state_blend = config.recurrent_state_blend
        self.shared_blocks = nn.ModuleList(
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
        """Run the shared recurrent loop over token inputs and return model outputs.

        Args:
            tokens: Input token ids.
            targets: Optional next-token targets for loss computation.

        Returns:
            ForwardOutput: Recurrent-loop logits and optional loss.
        """

        base_state = self._embed(tokens)
        state = base_state
        for _ in range(self.recurrent_steps):
            loop_state = state + base_state
            for block in self.shared_blocks:
                loop_state = block(loop_state)
            state = self.recurrent_state_blend * state + (1.0 - self.recurrent_state_blend) * loop_state
        return self._finalize(state, targets)


class DenseRingMemoryTransformerLM(LanguageModelBase):
    """Dense transformer trunk with a ring-fractal sidecar fused into the fast path."""

    def __init__(self, config: ModelConfig, sequence_length: int):
        super().__init__(config, sequence_length)
        if config.num_layers <= 0:
            raise ValueError("dense_ring_memory requires num_layers to be at least 1")
        if config.cluster_copies is None:
            raise ValueError("cluster_copies must be provided for dense_ring_memory")
        if config.cluster_bridge_percent is None:
            raise ValueError("cluster_bridge_percent must be provided for dense_ring_memory")
        if config.cluster_base_embedding_dim is None or config.cluster_base_embedding_dim <= 0:
            raise ValueError("cluster_base_embedding_dim must be a positive integer for dense_ring_memory")
        if config.memory_fusion_blend is None or not 0.0 <= config.memory_fusion_blend <= 1.0:
            raise ValueError("dense_ring_memory requires memory_fusion_blend in the inclusive range [0.0, 1.0]")

        self.memory_fusion_blend = config.memory_fusion_blend
        self.memory_update_interval = 1 if config.memory_update_interval is None else config.memory_update_interval
        if self.memory_update_interval <= 0:
            raise ValueError("dense_ring_memory requires memory_update_interval to be a positive integer")

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
        self.cluster_levels, self.cluster_top_count = resolve_ring_fractal_cluster_shape(config, sequence_length)
        self.cluster_base_embedding_dim = config.cluster_base_embedding_dim
        self.memory_input_projection = nn.Identity() if config.cluster_base_embedding_dim == config.embedding_dim else nn.Linear(config.embedding_dim, config.cluster_base_embedding_dim)
        self.memory_ring = _build_ring_cluster_group(
            stage_dim=config.cluster_base_embedding_dim,
            num_heads=_valid_head_count(config.cluster_base_embedding_dim, config.num_heads),
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            depth=1,
            child_count=self.cluster_top_count,
            branch_count=config.cluster_copies,
            levels_remaining=self.cluster_levels,
            cluster_bridge_percent=config.cluster_bridge_percent,
        )
        self.memory_output_projection = nn.Identity() if config.cluster_base_embedding_dim == config.embedding_dim else nn.Linear(config.cluster_base_embedding_dim, config.embedding_dim)
        self._initialize_weights()
        self.lm_head.weight = self.token_embeddings.weight

    def _fuse_memory(self, hidden_state: torch.Tensor) -> torch.Tensor:
        memory_state = self.memory_input_projection(hidden_state)
        memory_state = self.memory_ring(memory_state)
        memory_state = self.memory_output_projection(memory_state)
        return hidden_state + self.memory_fusion_blend * memory_state

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> ForwardOutput:
        x = self._embed(tokens)
        for index, block in enumerate(self.blocks, start=1):
            x = block(x)
            if index % self.memory_update_interval == 0:
                x = self._fuse_memory(x)
        if len(self.blocks) % self.memory_update_interval != 0:
            x = self._fuse_memory(x)
        return self._finalize(x, targets)


class ClusterBridgeStage(nn.Module):
    """One neighbor-bridged cluster stage over a hidden-state tensor."""

    def __init__(self, stage_dim: int, num_heads: int, mlp_ratio: int, dropout: float, depth: int, cluster_copies: int, cluster_bridge_percent: float, cluster_wrap_neighbors: bool):
        """Construct parallel dense copies with pairwise bridge channels."""

        super().__init__()
        if cluster_copies < 2:
            raise ValueError("cluster_copies must be at least 2 for clustered_dense")
        if cluster_bridge_percent <= 0 or cluster_bridge_percent > 100:
            raise ValueError("cluster_bridge_percent must be in the interval (0, 100]")
        self.stage_dim = stage_dim
        self.cluster_copies = cluster_copies
        self.cluster_wrap_neighbors = cluster_wrap_neighbors
        self.depth = depth
        self.bridge_width = max(1, math.ceil(stage_dim * (cluster_bridge_percent / 100.0)))
        self.bridge_slice = slice(stage_dim - self.bridge_width, stage_dim)
        self.cluster_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TransformerBlock(
                            embedding_dim=stage_dim,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            dropout=dropout,
                        )
                        for _ in range(depth)
                    ]
                )
                for _ in range(self.cluster_copies)
            ]
        )
        pair_count = self.cluster_copies if self.cluster_wrap_neighbors else self.cluster_copies - 1
        self.left_to_right_bridges = nn.ModuleList(
            [nn.ModuleList([nn.Linear(self.bridge_width, self.bridge_width, bias=False) for _ in range(pair_count)]) for _ in range(depth)]
        )
        self.right_to_left_bridges = nn.ModuleList(
            [nn.ModuleList([nn.Linear(self.bridge_width, self.bridge_width, bias=False) for _ in range(pair_count)]) for _ in range(depth)]
        )

    def _apply_neighbor_bridges(self, states: list[torch.Tensor], layer_index: int) -> list[torch.Tensor]:
        """Exchange bridge-channel updates between neighboring cluster copies."""

        bridge_deltas = [torch.zeros_like(state[:, :, self.bridge_slice]) for state in states]
        pair_count = self.cluster_copies if self.cluster_wrap_neighbors else self.cluster_copies - 1
        for pair_index in range(pair_count):
            left_index = pair_index
            right_index = (pair_index + 1) % self.cluster_copies
            left_state = states[left_index][:, :, self.bridge_slice]
            right_state = states[right_index][:, :, self.bridge_slice]
            bridge_deltas[right_index] = bridge_deltas[right_index] + self.left_to_right_bridges[layer_index][pair_index](left_state)
            bridge_deltas[left_index] = bridge_deltas[left_index] + self.right_to_left_bridges[layer_index][pair_index](right_state)

        updated_states: list[torch.Tensor] = []
        for state, delta in zip(states, bridge_deltas, strict=True):
            updated_state = state.clone()
            updated_state[:, :, self.bridge_slice] = updated_state[:, :, self.bridge_slice] + delta
            updated_states.append(updated_state)
        return updated_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run dense copies in parallel and recombine them after neighbor bridging."""

        cluster_states = [x.clone() for _ in range(self.cluster_copies)]
        for layer_index in range(self.depth):
            cluster_states = [blocks[layer_index](state) for blocks, state in zip(self.cluster_blocks, cluster_states, strict=True)]
            cluster_states = self._apply_neighbor_bridges(cluster_states, layer_index)
        return torch.stack(cluster_states, dim=0).mean(dim=0)


class LeafClusterProcessor(nn.Module):
    """One leaf cluster that performs local dense computation."""

    def __init__(self, stage_dim: int, num_heads: int, mlp_ratio: int, dropout: float, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=stage_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class RingClusterGroup(nn.Module):
    """A ring of child clusters or child rings with bridge-channel communication."""

    def __init__(self, child_modules: nn.ModuleList, stage_dim: int, cluster_bridge_percent: float):
        super().__init__()
        if len(child_modules) < 2:
            raise ValueError("ring groups must contain at least 2 children")
        self.child_modules = child_modules
        self.stage_dim = stage_dim
        self.child_count = len(child_modules)
        self.bridge_width = max(1, math.ceil(stage_dim * (cluster_bridge_percent / 100.0)))
        self.bridge_slice = slice(stage_dim - self.bridge_width, stage_dim)
        self.left_to_right_bridges = nn.ModuleList([nn.Linear(self.bridge_width, self.bridge_width, bias=False) for _ in range(self.child_count)])
        self.right_to_left_bridges = nn.ModuleList([nn.Linear(self.bridge_width, self.bridge_width, bias=False) for _ in range(self.child_count)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        child_states = [child(x.clone()) for child in self.child_modules]
        bridge_deltas = [torch.zeros_like(state[:, :, self.bridge_slice]) for state in child_states]
        for index in range(self.child_count):
            right_index = (index + 1) % self.child_count
            left_state = child_states[index][:, :, self.bridge_slice]
            right_state = child_states[right_index][:, :, self.bridge_slice]
            bridge_deltas[right_index] = bridge_deltas[right_index] + self.left_to_right_bridges[index](left_state)
            bridge_deltas[index] = bridge_deltas[index] + self.right_to_left_bridges[index](right_state)

        merged_children: list[torch.Tensor] = []
        for state, delta in zip(child_states, bridge_deltas, strict=True):
            updated = state.clone()
            updated[:, :, self.bridge_slice] = updated[:, :, self.bridge_slice] + delta
            merged_children.append(updated)
        return torch.stack(merged_children, dim=0).mean(dim=0)


def _build_ring_cluster_group(stage_dim: int, num_heads: int, mlp_ratio: int, dropout: float, depth: int, child_count: int, branch_count: int, levels_remaining: int, cluster_bridge_percent: float) -> RingClusterGroup:
    """Recursively build a ring-of-rings cluster hierarchy."""

    if levels_remaining <= 0:
        raise ValueError("levels_remaining must be positive")
    if levels_remaining == 1:
        children = nn.ModuleList([LeafClusterProcessor(stage_dim, num_heads, mlp_ratio, dropout, depth) for _ in range(child_count)])
    else:
        children = nn.ModuleList(
            [
                _build_ring_cluster_group(
                    stage_dim=stage_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    depth=depth,
                    child_count=branch_count,
                    branch_count=branch_count,
                    levels_remaining=levels_remaining - 1,
                    cluster_bridge_percent=cluster_bridge_percent,
                )
                for _ in range(child_count)
            ]
        )
    return RingClusterGroup(children, stage_dim, cluster_bridge_percent)


def _estimate_fractal_clustered_parameter_count(config: ModelConfig, sequence_length: int, level_count: int) -> int:
    """Estimate exact parameter count for an explicit recursive clustered layout."""

    explicit_config = replace(
        config,
        cluster_levels=level_count,
        cluster_target_parameter_count=None,
        cluster_max_levels=None,
    )
    return sum(parameter.numel() for parameter in FractalClusteredDenseTransformerLM(explicit_config, sequence_length).parameters())


def _estimate_ring_fractal_clustered_parameter_count(config: ModelConfig, sequence_length: int, level_count: int, top_count: int) -> int:
    """Estimate exact parameter count for an explicit ring-fractal clustered layout."""

    explicit_config = replace(
        config,
        cluster_levels=level_count,
        cluster_top_count=top_count,
        cluster_target_parameter_count=None,
        cluster_max_levels=None,
    )
    return sum(parameter.numel() for parameter in RingFractalClusteredDenseTransformerLM(explicit_config, sequence_length).parameters())


def resolve_fractal_cluster_levels(config: ModelConfig, sequence_length: int) -> int:
    """Resolve recursive cluster depth from an explicit count or a parameter target."""

    if config.cluster_levels is not None:
        if config.cluster_levels <= 0:
            raise ValueError("cluster_levels must be a positive integer")
        return config.cluster_levels
    if config.cluster_target_parameter_count is None:
        raise ValueError("fractal_clustered_dense requires cluster_levels or cluster_target_parameter_count")
    if config.cluster_target_parameter_count <= 0:
        raise ValueError("cluster_target_parameter_count must be a positive integer")
    max_levels = config.cluster_max_levels or 8
    if max_levels <= 0:
        raise ValueError("cluster_max_levels must be a positive integer when provided")
    current_level = 1
    current_params = _estimate_fractal_clustered_parameter_count(config, sequence_length, current_level)
    while current_level < max_levels:
        next_level = current_level + 1
        next_params = _estimate_fractal_clustered_parameter_count(config, sequence_length, next_level)
        if next_params >= config.cluster_target_parameter_count:
            if abs(next_params - config.cluster_target_parameter_count) <= abs(current_params - config.cluster_target_parameter_count):
                return next_level
            return current_level
        current_level = next_level
        current_params = next_params
    return current_level


def resolve_ring_fractal_cluster_shape(config: ModelConfig, sequence_length: int) -> tuple[int, int]:
    """Resolve recursive ring depth and top ring size from explicit values or a parameter target."""

    if config.cluster_copies is None or config.cluster_copies < 2:
        raise ValueError("cluster_copies must be at least 2 for ring_fractal_clustered_dense")
    if config.cluster_levels is not None and config.cluster_top_count is not None:
        if config.cluster_levels <= 0:
            raise ValueError("cluster_levels must be a positive integer")
        if config.cluster_top_count < 2 or config.cluster_top_count > config.cluster_copies:
            raise ValueError("cluster_top_count must be in the interval [2, cluster_copies]")
        return config.cluster_levels, config.cluster_top_count
    if config.cluster_target_parameter_count is None:
        raise ValueError("ring_fractal_clustered_dense requires cluster_levels and cluster_top_count or cluster_target_parameter_count")
    if config.cluster_target_parameter_count <= 0:
        raise ValueError("cluster_target_parameter_count must be a positive integer")
    max_levels = config.cluster_max_levels or 8
    if max_levels <= 0:
        raise ValueError("cluster_max_levels must be a positive integer when provided")
    best_shape = (1, config.cluster_copies)
    best_distance: int | None = None
    for levels in range(1, max_levels + 1):
        for top_count in range(2, config.cluster_copies + 1):
            parameter_total = _estimate_ring_fractal_clustered_parameter_count(config, sequence_length, levels, top_count)
            distance = abs(parameter_total - config.cluster_target_parameter_count)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_shape = (levels, top_count)
    return best_shape


class ClusteredDenseTransformerLM(LanguageModelBase):
    """Dense baseline replicated into neighbor-bridged clusters."""

    def __init__(self, config: ModelConfig, sequence_length: int):
        """Construct multiple dense copies and connect bridge channels between neighbors.

        Args:
            config: Resolved model hyperparameters.
            sequence_length: Maximum supported context length.
        """

        super().__init__(config, sequence_length)
        self.cluster_copies = config.cluster_copies
        self.cluster_wrap_neighbors = config.cluster_wrap_neighbors
        if config.cluster_copies is None:
            raise ValueError("cluster_copies must be provided for clustered_dense")
        if config.cluster_bridge_percent is None:
            raise ValueError("cluster_bridge_percent must be provided for clustered_dense")
        self.cluster_stage = ClusterBridgeStage(
            stage_dim=config.embedding_dim,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            depth=config.num_layers,
            cluster_copies=config.cluster_copies,
            cluster_bridge_percent=config.cluster_bridge_percent,
            cluster_wrap_neighbors=config.cluster_wrap_neighbors,
        )
        self.bridge_width = self.cluster_stage.bridge_width
        self._initialize_weights()
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> ForwardOutput:
        """Run dense copies in parallel and recombine them after neighbor bridging."""

        base_state = self._embed(tokens)
        merged_state = self.cluster_stage(base_state)
        return self._finalize(merged_state, targets)


class FractalClusteredDenseTransformerLM(LanguageModelBase):
    """Recursive neighbor-bridged clusters repeated up to a target parameter count."""

    def __init__(self, config: ModelConfig, sequence_length: int):
        super().__init__(config, sequence_length)
        if config.cluster_copies is None:
            raise ValueError("cluster_copies must be provided for fractal_clustered_dense")
        if config.cluster_bridge_percent is None:
            raise ValueError("cluster_bridge_percent must be provided for fractal_clustered_dense")
        if config.cluster_base_embedding_dim is None or config.cluster_base_embedding_dim <= 0:
            raise ValueError("cluster_base_embedding_dim must be a positive integer for fractal_clustered_dense")
        self.cluster_levels = resolve_fractal_cluster_levels(config, sequence_length)
        self.cluster_base_embedding_dim = config.cluster_base_embedding_dim
        self.input_projection = nn.Identity() if config.cluster_base_embedding_dim == config.embedding_dim else nn.Linear(config.embedding_dim, config.cluster_base_embedding_dim)
        self.fractal_stages = nn.ModuleList(
            [
                ClusterBridgeStage(
                    stage_dim=config.cluster_base_embedding_dim,
                    num_heads=_valid_head_count(config.cluster_base_embedding_dim, config.num_heads),
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    depth=config.num_layers,
                    cluster_copies=config.cluster_copies,
                    cluster_bridge_percent=config.cluster_bridge_percent,
                    cluster_wrap_neighbors=config.cluster_wrap_neighbors,
                )
                for _ in range(self.cluster_levels)
            ]
        )
        self.output_projection = nn.Identity() if config.cluster_base_embedding_dim == config.embedding_dim else nn.Linear(config.cluster_base_embedding_dim, config.embedding_dim)
        self._initialize_weights()
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> ForwardOutput:
        x = self._embed(tokens)
        x = self.input_projection(x)
        for stage in self.fractal_stages:
            x = stage(x)
        x = self.output_projection(x)
        return self._finalize(x, targets)


class RingFractalClusteredDenseTransformerLM(LanguageModelBase):
    """Recursive ring-of-rings clustered dense model."""

    def __init__(self, config: ModelConfig, sequence_length: int):
        super().__init__(config, sequence_length)
        if config.cluster_copies is None:
            raise ValueError("cluster_copies must be provided for ring_fractal_clustered_dense")
        if config.cluster_bridge_percent is None:
            raise ValueError("cluster_bridge_percent must be provided for ring_fractal_clustered_dense")
        if config.cluster_base_embedding_dim is None or config.cluster_base_embedding_dim <= 0:
            raise ValueError("cluster_base_embedding_dim must be a positive integer for ring_fractal_clustered_dense")
        self.cluster_levels, self.cluster_top_count = resolve_ring_fractal_cluster_shape(config, sequence_length)
        self.cluster_base_embedding_dim = config.cluster_base_embedding_dim
        self.input_projection = nn.Identity() if config.cluster_base_embedding_dim == config.embedding_dim else nn.Linear(config.embedding_dim, config.cluster_base_embedding_dim)
        self.ring_root = _build_ring_cluster_group(
            stage_dim=config.cluster_base_embedding_dim,
            num_heads=_valid_head_count(config.cluster_base_embedding_dim, config.num_heads),
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            depth=config.num_layers,
            child_count=self.cluster_top_count,
            branch_count=config.cluster_copies,
            levels_remaining=self.cluster_levels,
            cluster_bridge_percent=config.cluster_bridge_percent,
        )
        self.output_projection = nn.Identity() if config.cluster_base_embedding_dim == config.embedding_dim else nn.Linear(config.cluster_base_embedding_dim, config.embedding_dim)
        self._initialize_weights()
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> ForwardOutput:
        x = self._embed(tokens)
        x = self.input_projection(x)
        x = self.ring_root(x)
        x = self.output_projection(x)
        return self._finalize(x, targets)


def _linear_connectivity_stats(layer: nn.Linear) -> tuple[int, int, int]:
    """Return connection counts, output units, and max fan-in for one linear layer."""

    return layer.in_features * layer.out_features, layer.out_features, layer.in_features


def _block_connectivity_stats(block: TransformerBlock) -> tuple[int, int, int]:
    """Accumulate hidden connectivity stats for one transformer block."""

    layers = [
        block.attention.qkv,
        block.attention.projection,
        block.feedforward.net[0],
        block.feedforward.net[2],
    ]
    total_connections = 0
    total_outputs = 0
    max_fan_in = 0
    for layer in layers:
        connections, outputs, layer_max = _linear_connectivity_stats(layer)
        total_connections += connections
        total_outputs += outputs
        max_fan_in = max(max_fan_in, layer_max)
    return total_connections, total_outputs, max_fan_in


def resolve_modular_layout(config: ModelConfig, sequence_length: int | None = None) -> ModularLayoutConfig:
    """Normalize modular and cortical-column configs to one layout schema."""

    if config.architecture == "modular":
        if any(
            value is not None
            for value in (
                config.column_counts,
                config.fixed_column_size,
                config.column_input_count,
                config.column_branching_factor,
                config.target_parameter_count,
                config.max_column_stages,
                config.column_depths,
                config.column_recombination,
                config.column_routing_topology,
                config.column_routing_top_k,
            )
        ):
            raise ValueError("column_* fields are only valid when architecture is cortical_columns")
        group_schedule = config.stage_groups or [2, 1]
        depth_schedule = config.stage_depths or [1 for _ in group_schedule]
        fixed_group_size = config.fixed_group_size
        routing_topology = config.routing_topology
        routing_top_k = config.routing_top_k
        recombination_mode = "summary_router"
        unit_label = "Modular group"
    elif config.architecture == "cortical_columns":
        if any(value is not None for value in (config.stage_groups, config.fixed_group_size, config.stage_depths)):
            raise ValueError("Use column_* fields instead of stage_* fields for cortical_columns")
        if config.column_counts is None:
            if sequence_length is None:
                raise ValueError("sequence_length is required to resolve fractal cortical_columns layouts")
            group_schedule, depth_schedule = _resolve_fractal_column_schedule(config, sequence_length)
        else:
            group_schedule = config.column_counts
            depth_schedule = _resolve_depth_schedule(config.column_depths, len(group_schedule))
        fixed_group_size = config.fixed_column_size
        recombination_mode = config.column_recombination or "summary_router"
        if recombination_mode not in {"summary_router", "binary_tree"}:
            raise ValueError("column_recombination must be one of: summary_router, binary_tree")
        routing_topology = config.column_routing_topology or config.routing_topology
        routing_top_k = config.column_routing_top_k if config.column_routing_top_k is not None else config.routing_top_k
        if recombination_mode == "binary_tree":
            routing_topology = "dense"
            routing_top_k = None
        unit_label = "Cortical column"
    else:
        raise ValueError(f"Unsupported architecture: {config.architecture}")

    if len(group_schedule) != len(depth_schedule):
        raise ValueError("stage_groups and stage_depths must have the same length")
    if fixed_group_size is not None and fixed_group_size <= 0:
        raise ValueError("fixed_group_size must be a positive integer")
    return ModularLayoutConfig(
        group_schedule=group_schedule,
        depth_schedule=depth_schedule,
        fixed_group_size=fixed_group_size,
        routing_topology=routing_topology,
        routing_top_k=routing_top_k,
        recombination_mode=recombination_mode,
        unit_label=unit_label,
    )


def resolve_modular_stage_specs(config: ModelConfig, sequence_length: int | None = None) -> list[ModularStageSpec]:
    """Resolve modular stage schedules into per-stage width specifications."""

    layout = resolve_modular_layout(config, sequence_length=sequence_length)

    stage_specs: list[ModularStageSpec] = []
    for group_count, depth in zip(layout.group_schedule, layout.depth_schedule, strict=True):
        if layout.fixed_group_size is None:
            if config.embedding_dim % group_count != 0:
                raise ValueError("embedding_dim must be divisible by each group count in the stage schedule")
            group_dim = config.embedding_dim // group_count
            stage_dim = config.embedding_dim
        else:
            group_dim = layout.fixed_group_size
            stage_dim = group_count * group_dim
        stage_specs.append(ModularStageSpec(group_count=group_count, group_dim=group_dim, stage_dim=stage_dim, depth=depth))
    return stage_specs


def structural_connectivity_summary(model: LanguageModelBase) -> dict[str, float | int | None]:
    """Summarize hidden linear connectivity for dense and modular variants."""

    total_connections = 0
    total_outputs = 0
    max_fan_in = 0
    if isinstance(model, DenseTransformerLM):
        for block in model.blocks:
            connections, outputs, layer_max = _block_connectivity_stats(block)
            total_connections += connections
            total_outputs += outputs
            max_fan_in = max(max_fan_in, layer_max)
    elif isinstance(model, DenseRingMemoryTransformerLM):
        for block in model.blocks:
            connections, outputs, layer_max = _block_connectivity_stats(block)
            total_connections += connections
            total_outputs += outputs
            max_fan_in = max(max_fan_in, layer_max)
    elif isinstance(model, RecurrentLoopTransformerLM):
        for block in model.shared_blocks:
            connections, outputs, layer_max = _block_connectivity_stats(block)
            total_connections += connections
            total_outputs += outputs
            max_fan_in = max(max_fan_in, layer_max)
    elif isinstance(model, ClusteredDenseTransformerLM):
        for blocks in model.cluster_stage.cluster_blocks:
            for block in blocks:
                connections, outputs, layer_max = _block_connectivity_stats(block)
                total_connections += connections
                total_outputs += outputs
                max_fan_in = max(max_fan_in, layer_max)
        for bridge_layers in (model.cluster_stage.left_to_right_bridges, model.cluster_stage.right_to_left_bridges):
            for bridge_group in bridge_layers:
                for bridge in bridge_group:
                    connections, outputs, layer_max = _linear_connectivity_stats(bridge)
                    total_connections += connections
                    total_outputs += outputs
                    max_fan_in = max(max_fan_in, layer_max)
    elif isinstance(model, FractalClusteredDenseTransformerLM):
        for projection in (model.input_projection, model.output_projection):
            if isinstance(projection, nn.Linear):
                connections, outputs, layer_max = _linear_connectivity_stats(projection)
                total_connections += connections
                total_outputs += outputs
                max_fan_in = max(max_fan_in, layer_max)
        for stage in model.fractal_stages:
            for blocks in stage.cluster_blocks:
                for block in blocks:
                    connections, outputs, layer_max = _block_connectivity_stats(block)
                    total_connections += connections
                    total_outputs += outputs
                    max_fan_in = max(max_fan_in, layer_max)
            for bridge_layers in (stage.left_to_right_bridges, stage.right_to_left_bridges):
                for bridge_group in bridge_layers:
                    for bridge in bridge_group:
                        connections, outputs, layer_max = _linear_connectivity_stats(bridge)
                        total_connections += connections
                        total_outputs += outputs
                        max_fan_in = max(max_fan_in, layer_max)
    elif isinstance(model, RingFractalClusteredDenseTransformerLM):
        for projection in (model.input_projection, model.output_projection):
            if isinstance(projection, nn.Linear):
                connections, outputs, layer_max = _linear_connectivity_stats(projection)
                total_connections += connections
                total_outputs += outputs
                max_fan_in = max(max_fan_in, layer_max)
        for module in model.modules():
            if isinstance(module, LeafClusterProcessor):
                for block in module.blocks:
                    connections, outputs, layer_max = _block_connectivity_stats(block)
                    total_connections += connections
                    total_outputs += outputs
                    max_fan_in = max(max_fan_in, layer_max)
            elif isinstance(module, RingClusterGroup):
                for bridge in (*module.left_to_right_bridges, *module.right_to_left_bridges):
                    connections, outputs, layer_max = _linear_connectivity_stats(bridge)
                    total_connections += connections
                    total_outputs += outputs
                    max_fan_in = max(max_fan_in, layer_max)
    elif isinstance(model, ModularTransformerLM):
        for projection in model.stage_projections:
            if isinstance(projection, nn.Linear):
                connections, outputs, layer_max = _linear_connectivity_stats(projection)
                total_connections += connections
                total_outputs += outputs
                max_fan_in = max(max_fan_in, layer_max)
        for stage in model.stages:
            for group_blocks in stage.local_blocks:
                for block in group_blocks:
                    connections, outputs, layer_max = _block_connectivity_stats(block)
                    total_connections += connections
                    total_outputs += outputs
                    max_fan_in = max(max_fan_in, layer_max)
        if isinstance(model.output_projection, nn.Linear):
            connections, outputs, layer_max = _linear_connectivity_stats(model.output_projection)
            total_connections += connections
            total_outputs += outputs
            max_fan_in = max(max_fan_in, layer_max)
    average_hidden_fan_in = total_connections / total_outputs if total_outputs else None
    return {
        "average_hidden_fan_in": average_hidden_fan_in,
        "max_hidden_fan_in": max_fan_in if total_outputs else None,
    }


class StaticRouter(nn.Module):
    """Fixed routing layer that mixes module summaries under a topology mask."""

    def __init__(self, num_groups: int, topology: str, top_k: int | None):
        """Create routing logits and the allowed communication mask.

        Args:
            num_groups: Number of groups participating in routing.
            topology: Name of the allowed communication pattern.
            top_k: Optional number of inbound routes to keep per group.
        """

        super().__init__()
        self.num_groups = num_groups
        self.topology = topology
        self.top_k = top_k
        self.routing_logits = nn.Parameter(torch.zeros(num_groups, num_groups))
        self.register_buffer("routing_mask", self._build_mask(num_groups, topology), persistent=False)

    @staticmethod
    def _build_mask(num_groups: int, topology: str) -> torch.Tensor:
        """Build the boolean adjacency mask implied by the selected topology.

        Args:
            num_groups: Number of groups participating in routing.
            topology: Name of the allowed communication pattern.

        Returns:
            torch.Tensor: Boolean adjacency mask over source and destination groups.
        """

        mask = torch.zeros(num_groups, num_groups, dtype=torch.bool)
        cluster_size = max(2, int(num_groups**0.5))
        for destination in range(num_groups):
            for source in range(num_groups):
                if topology == "dense":
                    allowed = True
                elif topology == "local":
                    allowed = source == destination or abs(source - destination) == 1
                elif topology == "small_world":
                    stride = max(1, num_groups // 2)
                    allowed = source == destination or abs(source - destination) == 1 or source == (destination + stride) % num_groups
                elif topology == "cluster_graph":
                    destination_cluster = destination // cluster_size
                    source_cluster = source // cluster_size
                    allowed = source_cluster == destination_cluster
                    if not allowed:
                        cluster_count = max(1, (num_groups + cluster_size - 1) // cluster_size)
                        allowed = source == ((destination_cluster + 1) % cluster_count) * cluster_size
                else:
                    raise ValueError(f"Unsupported routing topology: {topology}")
                if allowed:
                    mask[destination, source] = True
        return mask

    def forward(self, summaries: torch.Tensor) -> torch.Tensor:
        """Mix group summaries according to learned weights constrained by the mask.

        Args:
            summaries: Group summary tensor shaped ``(batch, groups, channels)``.

        Returns:
            torch.Tensor: Routed summaries with the same shape as the input.
        """

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
    """One modular processing stage with local blocks and configurable recombination."""

    def __init__(self, stage_dim: int, group_count: int, group_dim: int, num_heads: int, mlp_ratio: int, dropout: float, depth: int, topology: str, top_k: int | None, recombination_mode: str = "summary_router"):
        """Partition channels into groups, attach local blocks, and configure routing.

        Args:
            stage_dim: Total width of the hidden state processed by this stage.
            group_count: Number of channel groups in the stage.
            group_dim: Width of each channel group in the stage.
            num_heads: Preferred attention head count for local blocks.
            mlp_ratio: Expansion factor for local feedforward layers.
            dropout: Dropout probability used inside local blocks.
            depth: Number of local transformer blocks per group.
            topology: Routing topology used between group summaries.
            top_k: Optional number of routes to keep per destination group.
            recombination_mode: Strategy used to recombine per-group summaries.
        """

        super().__init__()
        if stage_dim != group_count * group_dim:
            raise ValueError("stage_dim must equal group_count * group_dim")
        local_heads = _valid_head_count(group_dim, num_heads)
        self.stage_dim = stage_dim
        self.group_count = group_count
        self.group_dim = group_dim
        self.recombination_mode = recombination_mode
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
        self.summary_norm = nn.LayerNorm(group_dim)
        self.router = StaticRouter(group_count, topology, top_k) if recombination_mode == "summary_router" else None

    @staticmethod
    def _binary_tree_group_context(summaries: torch.Tensor) -> torch.Tensor:
        """Build hierarchical pairwise context over groups in powers of two."""

        propagated = torch.zeros_like(summaries)
        active_nodes = [summaries[:, index, :] for index in range(summaries.size(1))]
        spans = [(index, index + 1) for index in range(summaries.size(1))]
        while len(active_nodes) > 1:
            next_nodes: list[torch.Tensor] = []
            next_spans: list[tuple[int, int]] = []
            for index in range(0, len(active_nodes), 2):
                left_node = active_nodes[index]
                left_span = spans[index]
                if index + 1 < len(active_nodes):
                    right_node = active_nodes[index + 1]
                    right_span = spans[index + 1]
                    combined = 0.5 * (left_node + right_node)
                    span = (left_span[0], right_span[1])
                else:
                    combined = left_node
                    span = left_span
                propagated[:, span[0]:span[1], :] += combined[:, None, :]
                next_nodes.append(combined)
                next_spans.append(span)
            active_nodes = next_nodes
            spans = next_spans
        return propagated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run grouped local computation, summarize groups, and route updates.

        Args:
            x: Hidden states entering the modular stage.

        Returns:
            torch.Tensor: Hidden states after local processing and routed mixing.
        """

        batch_size, sequence_length, stage_dim = x.shape
        if stage_dim != self.stage_dim:
            raise ValueError("Input width does not match the configured stage width")
        grouped = x.view(batch_size, sequence_length, self.group_count, self.group_dim)
        outputs = []
        for group_index, blocks in enumerate(self.local_blocks):
            group_state = grouped[:, :, group_index, :]
            for block in blocks:
                group_state = block(group_state)
            outputs.append(group_state)
        merged = torch.stack(outputs, dim=2)
        summaries = self.summary_norm(merged.mean(dim=1))
        if self.recombination_mode == "summary_router":
            if self.router is None:
                raise ValueError("summary_router recombination requires a configured router")
            recombined = self.router(summaries)
        elif self.recombination_mode == "binary_tree":
            recombined = self._binary_tree_group_context(summaries)
        else:
            raise ValueError(f"Unsupported recombination mode: {self.recombination_mode}")
        merged = merged + recombined[:, None, :, :]
        return merged.reshape(batch_size, sequence_length, self.stage_dim)


class ModularTransformerLM(LanguageModelBase):
    """Hierarchical modular transformer variant with fixed inter-group routing."""

    def __init__(self, config: ModelConfig, sequence_length: int):
        """Construct the configured sequence of modular processing stages.

        Args:
            config: Resolved model hyperparameters.
            sequence_length: Maximum supported context length.
        """

        super().__init__(config, sequence_length)
        self.layout = resolve_modular_layout(config, sequence_length=sequence_length)
        self.stage_specs = resolve_modular_stage_specs(config, sequence_length=sequence_length)
        self.stage_projections = nn.ModuleList()
        self.stages = nn.ModuleList(
            [
                ModularStage(
                    stage_dim=stage_spec.stage_dim,
                    group_count=stage_spec.group_count,
                    group_dim=stage_spec.group_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    depth=stage_spec.depth,
                    topology=self.layout.routing_topology,
                    top_k=self.layout.routing_top_k,
                    recombination_mode=self.layout.recombination_mode,
                )
                for stage_spec in self.stage_specs
            ]
        )
        previous_dim = config.embedding_dim
        for stage_spec in self.stage_specs:
            if previous_dim == stage_spec.stage_dim:
                self.stage_projections.append(nn.Identity())
            else:
                self.stage_projections.append(nn.Linear(previous_dim, stage_spec.stage_dim))
            previous_dim = stage_spec.stage_dim
        self.output_projection = nn.Identity() if previous_dim == config.embedding_dim else nn.Linear(previous_dim, config.embedding_dim)
        self._initialize_weights()
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> ForwardOutput:
        """Run modular stages over token inputs and return model outputs.

        Args:
            tokens: Input token ids.
            targets: Optional next-token targets for loss computation.

        Returns:
            ForwardOutput: Modular-model logits and optional loss.
        """

        x = self._embed(tokens)
        for projection, stage in zip(self.stage_projections, self.stages, strict=True):
            x = stage(projection(x))
        x = self.output_projection(x)
        return self._finalize(x, targets)


def build_model(config: ModelConfig, sequence_length: int) -> LanguageModelBase:
    """Instantiate the requested dense or modular language-model variant.

    Args:
        config: Resolved model hyperparameters.
        sequence_length: Maximum supported context length.

    Returns:
        LanguageModelBase: Instantiated dense or modular model.
    """

    if config.architecture == "dense":
        return DenseTransformerLM(config, sequence_length)
    if config.architecture == "dense_ring_memory":
        return DenseRingMemoryTransformerLM(config, sequence_length)
    if config.architecture == "recurrent_loop":
        return RecurrentLoopTransformerLM(config, sequence_length)
    if config.architecture == "clustered_dense":
        return ClusteredDenseTransformerLM(config, sequence_length)
    if config.architecture == "fractal_clustered_dense":
        return FractalClusteredDenseTransformerLM(config, sequence_length)
    if config.architecture == "ring_fractal_clustered_dense":
        return RingFractalClusteredDenseTransformerLM(config, sequence_length)
    if config.architecture in {"modular", "cortical_columns"}:
        return ModularTransformerLM(config, sequence_length)
    raise ValueError(f"Unsupported architecture: {config.architecture}")
