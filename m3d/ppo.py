"""Conflict-graph actor/critic and PPO primitives for net-order selection.

PyTorch is optional: only neural training/inference imports this module.
The model observes all nets; its action mask permits only pending reroutes.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .rl_policy import candidate_features, FEATURE_DIM, FEATURE_NAMES, FEATURE_VERSION


class AttentionBlock(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(),
                                          nn.Linear(width * 2, width))

    def forward(self, x, graph, present):
        batch, nodes, width = x.shape
        qkv = self.qkv(x).reshape(batch, nodes, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Self-loops also make padded query rows well-defined; pooling excludes them.
        allowed = graph | torch.eye(nodes, device=x.device, dtype=torch.bool)[None]
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None], dropout_p=0.)
        attention = attention.transpose(1, 2).reshape(batch, nodes, width)
        x = self.norm1(x + self.projection(attention))
        x = self.norm2(x + self.feed_forward(x))
        return x * present.unsqueeze(-1)


class GraphActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.SiLU())
        self.blocks = nn.ModuleList([AttentionBlock(), AttentionBlock()])
        self.actor = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2 ** .5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.)

    def forward(self, features, graph, eligible, present=None):
        if present is None:
            present = torch.ones_like(eligible)
        eligible = eligible & present
        if not eligible.any(dim=1).all():
            raise ValueError('each observation must have an eligible action')
        x = self.encoder(features) * present.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, graph, present)
        mean = x.sum(dim=1) / present.sum(dim=1, keepdim=True).clamp_min(1)
        maximum = x.masked_fill(~present.unsqueeze(-1), -torch.inf).max(dim=1).values
        logits = self.actor(torch.cat((x, mean[:, None].expand_as(x)), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~eligible, -torch.inf)
        values = self.critic(torch.cat((mean, maximum), dim=-1)).squeeze(-1)
        return logits, values


def pack_observations(observations):
    batch = len(observations)
    nodes = max(len(obs['features']) for obs in observations)
    features = np.zeros((batch, nodes, FEATURE_DIM), dtype=np.float32)
    graph = np.zeros((batch, nodes, nodes), dtype=bool)
    eligible = np.zeros((batch, nodes), dtype=bool)
    present = np.zeros((batch, nodes), dtype=bool)
    for i, obs in enumerate(observations):
        n = len(obs['features'])
        features[i, :n] = obs['features']
        graph[i, :n, :n] = obs['graph']
        eligible[i, :n] = obs['eligible']
        present[i, :n] = True
    return tuple(torch.from_numpy(array) for array in (features, graph, eligible, present))


def routing_observation(router, eligible, pressure, iteration, cache=None):
    ids = sorted(router.routes)
    indices = {nid: i for i, nid in enumerate(ids)}
    features = np.asarray(candidate_features(router, ids, pressure, iteration, cache), dtype=np.float32)
    graph = np.eye(len(ids), dtype=bool)
    for owners in router.owners_v.values():
        if len(owners) > 1:
            positions = [indices[nid] for nid in owners]
            graph[np.ix_(positions, positions)] = True
    mask = np.zeros(len(ids), dtype=bool)
    mask[[indices[nid] for nid in eligible]] = True
    return {'features': features, 'graph': graph, 'eligible': mask}, ids


class NeuralSelector:
    def __init__(self, model, training=False, seed=42, record=False):
        self.model = model.eval()
        self.training = training
        self.record = record
        self.generator = torch.Generator().manual_seed(seed)
        self.cache = {}
        self.steps = []

    def __call__(self, router, eligible, pressure, iteration):
        if len(eligible) == 1:
            return eligible[0]
        observation, ids = routing_observation(router, eligible, pressure, iteration, self.cache)
        with torch.inference_mode():
            logits, value = self.model(*pack_observations([observation]))
            logp = F.log_softmax(logits[0], dim=-1)
            action = (torch.multinomial(logp.exp(), 1, generator=self.generator).item()
                      if self.training else logits[0].argmax().item())
        if self.record:
            self.steps.append(dict(observation, action=action, logp=logp[action].item(),
                                   value=value[0].item()))
        return ids[action]


class ExpertSelector:
    def __init__(self):
        self.cache = {}
        self.steps = []

    def __call__(self, router, eligible, pressure, iteration):
        if len(eligible) > 1:
            observation, ids = routing_observation(router, eligible, pressure, iteration, self.cache)
            self.steps.append(dict(observation, action=ids.index(eligible[0])))
        return eligible[0]


def episode_advantages(values, terminal_reward):
    # Complete trajectories, gamma=lambda=1: GAE telescopes to R_terminal - V(s).
    return [terminal_reward - value for value in values], [terminal_reward] * len(values)


def clipped_policy_loss(logp, old_logp, advantages, clip=.2):
    ratio = (logp - old_logp).exp()
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()


def ppo_update(model, optimizer, episodes, entropy=.01, epochs=4, minibatch=128,
               generator=None, target_kl=.02):
    steps, advantages, returns = [], [], []
    for episode in episodes:
        trajectory = episode['steps']
        a, r = episode_advantages([s['value'] for s in trajectory], episode['row']['reward'])
        steps.extend(trajectory)
        advantages.extend(a)
        returns.extend(r)
    if not steps:
        return {'decisions': 0, 'minibatches': 0, 'kl': 0.}
    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    returns = torch.tensor(returns, dtype=torch.float32)
    old_logp = torch.tensor([s['logp'] for s in steps], dtype=torch.float32)
    actions = torch.tensor([s['action'] for s in steps], dtype=torch.long)
    model.train()
    batches, last_kl, stop = 0, 0., False
    for _ in range(epochs):
        permutation = torch.randperm(len(steps), generator=generator)
        for start in range(0, len(steps), minibatch):
            indices = permutation[start:start + minibatch]
            logits, values = model(*pack_observations([steps[i] for i in indices.tolist()]))
            distribution = torch.distributions.Categorical(logits=logits)
            logp = distribution.log_prob(actions[indices])
            logratio = logp - old_logp[indices]
            with torch.no_grad():
                last_kl = ((logratio.exp() - 1) - logratio).mean().item()
            if last_kl > target_kl:
                stop = True
                break
            loss = clipped_policy_loss(logp, old_logp[indices], advantages[indices])
            loss = loss + .5 * F.mse_loss(values, returns[indices]) - entropy * distribution.entropy().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            batches += 1
        if stop:
            break
    return {'decisions': len(steps), 'minibatches': batches, 'kl': last_kl, 'kl_stopped': stop}


def state_bytes(model):
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return stream.getvalue()


def save_checkpoint(path, model, metadata=None):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save({'format': 'm3d-ppo-policy', 'feature_version': FEATURE_VERSION,
                'features': FEATURE_NAMES, 'state_dict': model.state_dict(),
                'metadata': metadata or {}}, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if (data.get('format') != 'm3d-ppo-policy' or data.get('feature_version') != FEATURE_VERSION
            or data.get('features') != FEATURE_NAMES):
        raise ValueError('incompatible neural policy checkpoint')
    model = GraphActorCritic()
    model.load_state_dict(data['state_dict'])
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('checkpoint contains non-finite parameters')
    return model.eval(), data.get('metadata', {})
