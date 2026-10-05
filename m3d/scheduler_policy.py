"""Compact permutation-equivariant graph/Transformer net scheduler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .scheduler_features import (
    EncodedState, GLOBAL_FEATURE_DIM, NET_FEATURE_DIM, batch_states, encode_observation,
)


@dataclass(frozen=True)
class NetDistribution:
    """A categorical distribution whose labels are actual Python integer net IDs."""

    net_ids: tuple[int, ...]
    logits: torch.Tensor

    @property
    def probs(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    def sample(self, generator: torch.Generator | None = None) -> int:
        # CPU draws deliberately separate the action RNG from model-device RNG.
        index = int(torch.multinomial(self.probs.detach().to("cpu"), 1, generator=generator).item())
        return self.net_ids[index]

    def mode(self) -> int:
        return self.net_ids[int(self.logits.argmax().item())]

    def log_prob(self, net_id: int) -> torch.Tensor:
        if type(net_id) is not int or net_id not in self.net_ids:
            raise ValueError(f"unknown integer net ID: {net_id!r}")
        return torch.log_softmax(self.logits, dim=-1)[self.net_ids.index(net_id)]


class _MeanGraphLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.update = nn.Linear(2 * hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()

    def forward(self, tokens: torch.Tensor, adjacency: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        weights = adjacency.to(dtype=tokens.dtype)
        mean_neighbors = torch.bmm(weights, tokens) / weights.sum(-1, keepdim=True).clamp_min(1)
        updated = self.norm(tokens + self.activation(self.update(torch.cat((tokens, mean_neighbors), dim=-1))))
        return updated.masked_fill(padding_mask.unsqueeze(-1), 0)


class SchedulerPolicy(nn.Module):
    def __init__(self, hidden_size: int = 192, heads: int = 6,
                 ff_size: int = 1024, transformer_layers: int = 4,
                 graph_layers: int = 2):
        super().__init__()
        self.architecture = dict(hidden_size=hidden_size, heads=heads, ff_size=ff_size,
                                 transformer_layers=transformer_layers, graph_layers=graph_layers)
        self.net_embedding = nn.Sequential(nn.Linear(NET_FEATURE_DIM, hidden_size), nn.GELU(),
                                           nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))
        self.global_embedding = nn.Sequential(nn.Linear(GLOBAL_FEATURE_DIM, hidden_size), nn.GELU(),
                                              nn.Linear(hidden_size, hidden_size))
        self.graph = nn.ModuleList(_MeanGraphLayer(hidden_size) for _ in range(graph_layers))
        # No positional embeddings or net-ID embeddings: tokens are a set.
        self.transformer = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=heads,
                                       dim_feedforward=ff_size, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(transformer_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.score = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1))

    def forward(self, batch: Mapping) -> torch.Tensor:
        """Return raw finite logits [B,N]; callers mask eligibility themselves."""
        features = batch["net_features"]
        padding = batch["padding_mask"].bool()
        if features.shape[1] == 0:
            return features.new_zeros(features.shape[:2])
        # Mask BEFORE embedding and aggregation, including poisoned padding input.
        features = features.masked_fill(padding.unsqueeze(-1), 0)
        tokens = self.net_embedding(features) + self.global_embedding(batch["global_features"]).unsqueeze(1)
        tokens = tokens.masked_fill(padding.unsqueeze(-1), 0)
        allowed = ~padding
        adjacency = batch["adjacency"].bool() & allowed.unsqueeze(1) & allowed.unsqueeze(2)
        for layer in self.graph:
            tokens = layer(tokens, adjacency, padding)
        # Empty-net rows have no attention tokens. Excluding them prevents an
        # all-masked softmax while preserving the usual mixed-size batch path.
        active_rows = allowed.any(dim=1)
        if bool(active_rows.all()):
            for layer in self.transformer:
                tokens = layer(tokens, src_key_padding_mask=padding)
                tokens = tokens.masked_fill(padding.unsqueeze(-1), 0)
        elif bool(active_rows.any()):
            active = tokens[active_rows]
            active_padding = padding[active_rows]
            for layer in self.transformer:
                active = layer(active, src_key_padding_mask=active_padding)
                active = active.masked_fill(active_padding.unsqueeze(-1), 0)
            tokens = tokens.clone()
            tokens[active_rows] = active
        logits = self.score(self.output_norm(tokens)).squeeze(-1)
        return logits.masked_fill(padding, 0)

    def distribution(self, observation: Mapping | EncodedState) -> NetDistribution:
        state = observation if isinstance(observation, EncodedState) else encode_observation(observation)
        if not bool(state.eligible.any()):
            raise ValueError("cannot select a net when no net is eligible")
        device = next(self.parameters()).device
        batch = batch_states([state], device=device)
        logits = self(batch)[0].masked_fill(~batch["eligible"][0], float("-inf"))
        return NetDistribution(state.net_ids, logits)

    def act(self, observation: Mapping | EncodedState, deterministic: bool = False,
            generator: torch.Generator | None = None) -> int:
        distribution = self.distribution(observation)
        return distribution.mode() if deterministic else distribution.sample(generator)

    def log_prob(self, observation: Mapping | EncodedState, action: int) -> torch.Tensor:
        return self.distribution(observation).log_prob(action)
