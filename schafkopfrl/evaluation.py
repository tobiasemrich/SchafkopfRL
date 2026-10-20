from __future__ import annotations

from typing import Callable, Optional

import torch
from torchrl.envs.utils import set_exploration_type
from torchrl.envs.utils import ExplorationType

from schafkopfrl.data import iter_batches, load_expert_shard, obs_tensordict
from schafkopfrl.environment.pettingzoo_env import SchafkopfAECEnv
from schafkopfrl.policy.models import GROUP
from schafkopfrl.policy.pimc_policy import PIMCPolicy
from schafkopfrl.policy.rulebased_policy import RuleBasedPolicy

# A player maps the env (acting agent = env.agent_selection) to a discrete action index.
Player = Callable[[SchafkopfAECEnv], int]


class RawStatePlayer:
    """Adapter for heuristic policies (rule-based, PIMC) that work on the raw game state."""

    def __init__(self, policy: RuleBasedPolicy | PIMCPolicy) -> None:
        self.policy = policy

    def __call__(self, env: SchafkopfAECEnv) -> int:
        return self.policy.act(env.raw_state())


class NetworkPlayer:
    """Plays with the policy half of a TorchRL actor-critic (sampling, or argmax if ``deterministic``)."""

    def __init__(self, actor_critic, deterministic: bool = False) -> None:
        self.policy = actor_critic.get_policy_operator()
        self.deterministic = deterministic

    @torch.no_grad()
    def __call__(self, env: SchafkopfAECEnv) -> int:
        device = next(self.policy.parameters()).device
        td = obs_tensordict(env.observe(env.agent_selection)).to(device)
        mode = ExplorationType.DETERMINISTIC if self.deterministic else ExplorationType.RANDOM
        with set_exploration_type(mode):
            self.policy(td)
        return int(td[GROUP, "action"])


def play_game(env: SchafkopfAECEnv, players: list[Player], seed: Optional[int] = None) -> list[float]:
    """Play one game and return the final reward of each seat."""
    env.reset(seed=seed)
    while not env.terminations[env.agent_selection]:
        seat = env.possible_agents.index(env.agent_selection)
        env.step(players[seat](env))
    return [env.rewards[a] for a in env.possible_agents]


def evaluate_vs_rulebased(player: Player, n_games: int, opponent: Optional[Player] = None) -> float:
    """Average reward per seat of ``player`` (seats 1 and 3) against the rule-based policy (seats 0 and 2)."""
    env = SchafkopfAECEnv()
    opponent = opponent or RawStatePlayer(RuleBasedPolicy())
    total = 0.0
    for seed in range(n_games):
        rewards = play_game(env, [opponent, player, opponent, player], seed=seed)
        total += rewards[1] + rewards[3]
    return total / n_games / 2


@torch.no_grad()
def validation_accuracy(actor_critic, shard_path: str, max_samples: Optional[int] = None,
                        batch_size: int = 4096) -> float:
    """Fraction of expert actions in a shard that the policy predicts (argmax)."""
    policy = actor_critic.get_policy_operator()
    device = next(policy.parameters()).device
    data = load_expert_shard(shard_path, max_samples)
    correct = 0
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for td, actions in iter_batches(data, batch_size, shuffle=False):
            td = td.to(device)
            policy(td)
            correct += (td[GROUP, "action"].cpu() == actions).sum().item()
    return correct / data["actions"].shape[0]
