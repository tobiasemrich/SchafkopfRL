from __future__ import annotations

import json
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from tensordict import TensorDict

from schafkopfrl.policy.models import GROUP, OBSERVATION_KEYS


def obs_tensordict(obs: Mapping[str, Any], group: str = GROUP) -> TensorDict:
    """Wrap an observation (``SchafkopfAECEnv.observe`` output or a batch of them) for the actor-critic.

    Leading dimensions of the values become the batch size of the result.
    """
    mask = torch.as_tensor(np.asarray(obs["action_mask"])).bool()
    return TensorDict(
        {group: {
            "observation": {k: torch.as_tensor(np.asarray(obs[k])) for k in OBSERVATION_KEYS},
            "action_mask": mask,
        }},
        batch_size=mask.shape[:-1],
    )


def load_expert_shard(path: str, max_rows: int | None = None) -> dict[str, torch.Tensor]:
    """Load one expert-data JSONL shard into dense tensors (``obs`` fields plus ``actions``)."""
    rows: dict[str, list] = {k: [] for k in (*OBSERVATION_KEYS, "action_mask", "actions")}
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            sample = json.loads(line)
            for k in (*OBSERVATION_KEYS, "action_mask"):
                rows[k].append(sample["obs"][k])
            rows["actions"].append(sample["actions"])
    return {
        "player_hand": torch.tensor(rows["player_hand"], dtype=torch.int8),
        "info_vector": torch.tensor(rows["info_vector"], dtype=torch.float32),
        "action_history": torch.tensor(rows["action_history"], dtype=torch.int32),
        "action_history_len": torch.tensor(rows["action_history_len"], dtype=torch.int32).reshape(-1, 1),
        "action_mask": torch.tensor(rows["action_mask"], dtype=torch.bool),
        "actions": torch.tensor(rows["actions"], dtype=torch.long),
    }


def iter_batches(data: dict[str, torch.Tensor], batch_size: int, shuffle: bool = True) -> Iterator[tuple[TensorDict, torch.Tensor]]:
    """Yield ``(obs_tensordict, actions)`` mini-batches from a loaded shard."""
    n = data["actions"].shape[0]
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield obs_tensordict({k: v[idx] for k, v in data.items() if k != "actions"}), data["actions"][idx]
