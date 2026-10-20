from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from schafkopfrl.policy.models import GROUP

_PPO_KEYS = ("observation", "action_mask", "action", "sample_log_prob", "state_value", "advantage", "value_target")


def make_gae(value_operator, gamma: float = 1.0, lmbda: float = 1.0, group: str = GROUP) -> GAE:
    """GAE over the (time, agent) grid.

    Every seat has a value at every step (observations are ego-relative and all agents get the
    final payoff), so advantages are computed for all seats and the acting ones are picked out later.
    """
    # shifted=True avoids vmap, which cannot trace the history encoders' data-dependent control flow
    gae = GAE(gamma=gamma, lmbda=lmbda, value_network=value_operator, average_gae=False, shifted=True)
    gae.set_keys(
        advantage=(group, "advantage"), value_target=(group, "value_target"), value=(group, "state_value"),
        reward=(group, "reward"), done=(group, "done"), terminated=(group, "terminated"),
    )
    return gae


def make_ppo_loss(actor_critic, clip_epsilon: float = 0.2, entropy_coeff: float = 0.01,
                  critic_coeff: float = 1.0, group: str = GROUP) -> ClipPPOLoss:
    loss = ClipPPOLoss(
        actor_network=actor_critic.get_policy_operator(),
        critic_network=actor_critic.get_value_operator(),
        clip_epsilon=clip_epsilon,
        entropy_coeff=entropy_coeff,
        critic_coeff=critic_coeff,
        normalize_advantage=True,
    )
    loss.set_keys(
        advantage=(group, "advantage"), value_target=(group, "value_target"), value=(group, "state_value"),
        sample_log_prob=(group, "sample_log_prob"), action=(group, "action"),
    )
    return loss


def acting_transitions(td: TensorDictBase, group: str = GROUP) -> TensorDictBase:
    """Flatten a collected batch to the transitions where an agent actually moved.

    ``td[group]`` has an agent dimension; in a turn-based game exactly one agent per step has
    ``mask=True``. The result has batch size ``[N]`` and the agent dimension removed.
    """
    mask = td.get((group, "mask"))
    idx = mask.nonzero(as_tuple=True)
    agents = td.get(group).select(*_PPO_KEYS)
    return TensorDict({group: agents[idx]}, batch_size=[idx[0].numel()])
