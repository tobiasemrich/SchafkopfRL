from typing import Tuple
from gymnasium.spaces import Discrete
from ray.rllib.algorithms import Algorithm
from ray.rllib.core import Columns
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.utils.typing import ResultDict

from schafkopfrl.environment.multi_agent_env import SchafkopfMultiAgentEnv
import torch
from torch.distributions import Categorical
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from schafkopfrl.policy.rulebased_policy import RuleBasedRLModule

class TournamentEvaluation:
    def __init__(self, rl_module_name: str, n_rounds: int):
        self.rl_module_name = rl_module_name
        self.n_episodes = n_rounds

    def rulebased_tournament_eval_fn(self, algorithm: Algorithm, eval_workers: EnvRunnerGroup) -> Tuple[ResultDict, int, int]:

        env = SchafkopfMultiAgentEnv()
        linear_policy = algorithm.env_runner.module._rl_modules[self.rl_module_name]
        rulebased_policy = RuleBasedRLModule()  

        total_rewards = {"policy": 0.0, "rulebased": 0.0}

        for i in range(self.n_episodes):
            obs, info = env.reset(seed=i)
            done = False
            rewards = {"player_0": 0.0, "player_1": 0.0, "player_2": 0.0, "player_3": 0.0}
            while not done:
                actions = {}
                for player_id, pobs in obs.items():
                    if player_id in ["player_1", "player_3"]:
                        tensor_obs = self.convert_obs_dict_to_tensor(pobs)

                        # Inference with the RLModule
                        logits = linear_policy.forward_inference({"obs": tensor_obs})[Columns.ACTION_DIST_INPUTS]
                        dist = Categorical(logits=logits)
                        actions[player_id] = torch.tensor([dist.sample().item()])
                    else:
                        actions[player_id] = rulebased_policy.forward_inference(info[player_id])
                obs, rew, terminateds, _, info = env.step(actions)
                done = terminateds["__all__"]
                for k in rewards:
                    rewards[k] += rew.get(k, 0.0)

            total_rewards["policy"] += rewards["player_1"] + rewards["player_3"]
            total_rewards["rulebased"] += rewards["player_0"] + rewards["player_2"]
        

        print(env.env.render())
        mean_reward = total_rewards["policy"] / self.n_episodes / 2
        print("avg_reward_against_rulebased_policy", mean_reward)
        algorithm.metrics.log_value("avg_reward_against_rulebased_policy", mean_reward, window=1)
        return {"avg_reward_against_rulebased_policy": mean_reward}, self.n_episodes, self.n_episodes

    def convert_obs_dict_to_tensor(self, obs_dict, device="cpu"):
        return {k: torch.tensor(v, dtype=torch.int32, device=device).unsqueeze(0) for k, v in obs_dict.items()} # unsqueezing produces a batch
