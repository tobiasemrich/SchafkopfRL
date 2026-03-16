import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from torch.distributions import Categorical
from schafkopfrl.environment.multi_agent_env import SchafkopfMultiAgentEnv
from schafkopfrl.policy.pimcmodule import PIMCModule
from schafkopfrl.policy.rulebased_policy import RuleBasedRLModule
import torch
from typing import Any


def main() -> None:
  """Run a round-robin tournament between MCTS and rule-based policies."""

  pimc_policy: PIMCModule = PIMCModule(10, 40)

  rulebased_policy: RuleBasedRLModule = RuleBasedRLModule()


  participants: list[RLModule] = [pimc_policy, rulebased_policy]

  number_of_games: int = 10

  for i in range(len(participants)):
    for j in range(i+1, len(participants)):
      p1 = participants[i]
      p2 = participants[j]

      cummulative_reward: list[float] = [0, 0, 0, 0]
      for k in range(2): #run the same tournament twice with differen positions of players
        print(' ')
        schafkopf_env: SchafkopfMultiAgentEnv = SchafkopfMultiAgentEnv()
        if k == 0:
          players = [p1, p1, p2, p2]
        else:
          players = [p2, p2, p1, p1]
          cummulative_reward.reverse()

        # tournament loop
        for game_nr in range(1, number_of_games+1):
          obs, info = schafkopf_env.reset(seed=game_nr)
          done: bool = False
          rewards: dict[str, float] = {"player_0": 0.0, "player_1": 0.0, "player_2": 0.0, "player_3": 0.0}
          while not done:
              actions: dict[str, Any] = {}
              player_id, pobs = obs.popitem()
              player_index: int = int(player_id[-1])
              player_rlmodule: RLModule = players[player_index]
              if isinstance(player_rlmodule, PIMCModule) or isinstance(player_rlmodule, RuleBasedRLModule):
                actions[player_id] = player_rlmodule.forward_inference(info[player_id])
              else:
                  tensor_obs = convert_obs_dict_to_tensor(pobs)
                  # Inference with the RLModule
                  logits = player_rlmodule.forward_inference({"obs": tensor_obs})[Columns.ACTION_DIST_INPUTS]
                  dist = Categorical(logits=logits)
                  actions[player_id] = torch.tensor([dist.sample().item()])
                  
              obs, rew, terminateds, _, info = schafkopf_env.step(actions)
              done = terminateds["__all__"]
              for k in rewards:
                  rewards[k] += rew.get(k, 0.0)

          cummulative_reward = [cummulative_reward[m] + rewards["player_"+str(m)] for m in range(4)]

          if game_nr % 100 == 0:
            print('.', end = '')
          #schafkopf_env.print_game()

      print("player "+str(i)+" vs. player "+str(j)+" = " + str((cummulative_reward[2] + cummulative_reward[3]) / (2*2*number_of_games)) + " to " +str((cummulative_reward[0] + cummulative_reward[1]) / (2*2*number_of_games)))


def convert_obs_dict_to_tensor(obs_dict: dict[str, Any], device: str = "cpu") -> dict[str, torch.Tensor]:
        """Convert a numpy observation dict to batched PyTorch tensors.

        Parameters
        ----------
        obs_dict : dict[str, Any]
            Observation dict with numpy arrays.
        device : str, optional
            Target device, by default ``"cpu"``.

        Returns
        -------
        dict[str, torch.Tensor]
            Tensors with an added batch dimension.
        """
        return {k: torch.tensor(v, dtype=torch.int32, device=device).unsqueeze(0) for k, v in obs_dict.items()} # unsqueezing produces a batch


if __name__ == '__main__':
  main()