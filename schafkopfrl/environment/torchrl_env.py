from __future__ import annotations

from typing import Optional

from torchrl.envs import PettingZooWrapper
from torchrl.envs.utils import MarlGroupMapType

from .pettingzoo_env import SchafkopfAECEnv


def make_torchrl_env(
    group_map: Optional[MarlGroupMapType | dict[str, list[str]]] = None,
    render_mode: Optional[str] = None,
) -> PettingZooWrapper:
    """Wrap ``SchafkopfAECEnv`` as a TorchRL env.

    Defaults to one group per player; pass ``MarlGroupMapType.ALL_IN_ONE_GROUP`` (or an explicit
    dict) to share a policy between seats. ``use_mask=True`` is required for turn-based envs:
    ``(group, "mask")`` marks the acting agent and ``(group, "action_mask")`` the legal actions.
    """
    return PettingZooWrapper(
        env=SchafkopfAECEnv(render_mode=render_mode),
        use_mask=True,
        categorical_actions=True,
        group_map=group_map,
    )
