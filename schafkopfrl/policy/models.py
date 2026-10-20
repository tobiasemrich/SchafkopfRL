from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.modules import ActorValueOperator, ProbabilisticActor, ValueOperator

from schafkopfrl.environment.pettingzoo_env import INFO_VECTOR_SIZE, MAX_ACTIONS, NUM_ACTIONS
from schafkopfrl.policy.lstm_encoder import LSTMHistoryEncoder
from schafkopfrl.policy.transformer_encoder import TransformerHistoryEncoder

# Name of the TorchRL agent group the networks read from / write to.
GROUP: str = "agents"

OBSERVATION_KEYS: tuple[str, ...] = ("player_hand", "info_vector", "action_history", "action_history_len")


class ObservationEncoder(nn.Module):
    """Concatenate hand, info vector and encoded action history into one feature vector.

    Accepts any number of leading batch dimensions (including none).
    """

    def __init__(self, history_encoder: nn.Module, history_dim: int) -> None:
        super().__init__()
        self.history_encoder = history_encoder
        self.output_dim: int = 32 + INFO_VECTOR_SIZE + history_dim

    def forward(self, player_hand: torch.Tensor, info_vector: torch.Tensor,
                action_history: torch.Tensor, action_history_len: torch.Tensor) -> torch.Tensor:
        lead = player_hand.shape[:-1]
        hand = player_hand.reshape(-1, 32).float()
        info = info_vector.reshape(-1, INFO_VECTOR_SIZE).float()
        history = action_history.reshape(-1, MAX_ACTIONS, 2)
        lengths = action_history_len.reshape(-1).long()
        encoded = self.history_encoder(history, lengths)
        return torch.cat([hand, info, encoded], dim=-1).reshape(*lead, -1)


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
    return net


class MaskedPolicyHead(nn.Module):
    """MLP producing action logits with illegal actions pushed to the dtype minimum."""

    def __init__(self, in_dim: int, hidden: Sequence[int]) -> None:
        super().__init__()
        self.net = _mlp(in_dim, hidden, NUM_ACTIONS)

    def forward(self, features: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        logits = self.net(features)
        mask = action_mask.bool()
        # Non-acting agents have an all-zero mask; keep their logits finite so nothing becomes NaN.
        mask = mask | ~mask.any(dim=-1, keepdim=True)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class ValueHead(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int]) -> None:
        super().__init__()
        self.net = _mlp(in_dim, hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def make_history_encoder(kind: str, dim: int, num_layers: int, num_heads: int) -> nn.Module:
    if kind == "lstm":
        return LSTMHistoryEncoder(NUM_ACTIONS, emb_size_actions=32, emb_size_player=4,
                                  lstm_hidden_size=dim, num_layers=num_layers)
    if kind == "transformer":
        return TransformerHistoryEncoder(NUM_ACTIONS, emb_size_actions=32, emb_size_player=4,
                                         transformer_dim=dim, num_heads=num_heads, num_layers=num_layers)
    raise ValueError(f"Unknown history encoder {kind!r}")


def build_actor_critic(
    encoder: str = "lstm",
    hidden: Sequence[int] = (128, 128),
    history_dim: int = 128,
    num_layers: int = 1,
    num_heads: int = 4,
    group: str = GROUP,
) -> ActorValueOperator:
    """Build the shared-encoder actor-critic used for both PPO and behavioural cloning.

    The observation encoder is shared; policy and value heads are separate MLPs. Use
    ``get_policy_operator()`` for acting/BC and ``get_value_operator()`` for the critic.
    """
    obs_encoder = ObservationEncoder(
        make_history_encoder(encoder, history_dim, num_layers, num_heads), history_dim
    )
    common = TensorDictModule(
        obs_encoder,
        in_keys=[(group, "observation", k) for k in OBSERVATION_KEYS],
        out_keys=[(group, "hidden")],
    )
    policy = ProbabilisticActor(
        module=TensorDictModule(
            MaskedPolicyHead(obs_encoder.output_dim, hidden),
            in_keys=[(group, "hidden"), (group, "action_mask")],
            out_keys=[(group, "logits")],
        ),
        in_keys=[(group, "logits")],
        out_keys=[(group, "action")],
        distribution_class=Categorical,
        return_log_prob=True,
        log_prob_key=(group, "sample_log_prob"),
    )
    value = ValueOperator(
        ValueHead(obs_encoder.output_dim, hidden),
        in_keys=[(group, "hidden")],
        out_keys=[(group, "state_value")],
    )
    return ActorValueOperator(common, policy, value)
