import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
from os import listdir
from copy import deepcopy
import torch
from sqlitedict import SqliteDict
import sys

from schafkopfrl.environment.multi_agent_env import SchafkopfMultiAgentEnv
from tensorboard import program
import numpy as np

from schafkopfrl.environment.rules import Rules
import json
import pandas as pd
from typing import Any

rules: Rules = Rules()


def main() -> None:
  """Parse game transcripts and write sharded expert data to JSONL for BC training."""
  SHARD_SIZE: int = 20_000  # games per shard
  output_dir: str = 'data'
  os.makedirs(output_dir, exist_ok=True)

  count: int = 0
  total_rows: int = 0
  shard_idx: int = 0
  shard_file = None
  shard_row_count: int = 0

  with open('data/normal_games.json', 'r') as file:
    games = json.load(file)

    for game_id, g in enumerate(games):
      game = g
      if len(game["sonderregeln"]) == 0:
        # Open a new shard file when needed
        if shard_file is None:
          shard_path = os.path.join(output_dir, f'expert_data_shard_{shard_idx:03d}.jsonl')
          shard_file = open(shard_path, 'w')
          shard_row_count = 0
          print(f"Writing shard {shard_idx}: {shard_path}")

        count += 1
        final_rows: list[dict[str, Any]] = get_states_actions(game, game_id=game_id)
        for row in final_rows:
          row_serializable: Any = convert_to_serializable(row)
          json.dump(row_serializable, shard_file)
          shard_file.write('\n')
        shard_row_count += len(final_rows)
        total_rows += len(final_rows)

        if count % 1000 == 0:
          print(f"Read {count} normal games ({total_rows} rows total, shard {shard_idx}: {shard_row_count} rows)")

        # Close shard and start a new one after SHARD_SIZE games
        if count % SHARD_SIZE == 0:
          shard_file.close()
          print(f"Finished shard {shard_idx} with {shard_row_count} rows")
          shard_file = None
          shard_idx += 1

  # Close the last shard if it's still open
  if shard_file is not None:
    shard_file.close()
    print(f"Finished shard {shard_idx} with {shard_row_count} rows")

  print(f"Saved {total_rows} rows across {shard_idx + 1} shards from {count} games")


def get_states_actions(game_transcript: dict[str, Any], game_id: int) -> list[dict[str, Any]]:
  """Replay a game transcript through the environment and collect transitions.

  Parameters
  ----------
  game_transcript : dict[str, Any]
      Raw game transcript with player hands, bidding, and course of game.
  game_id : int
      Unique identifier for this game episode.

  Returns
  -------
  list[dict[str, Any]]
      List of ``(obs, new_obs, actions, rewards, ...)`` dicts per agent step.
  """
  env: SchafkopfMultiAgentEnv = SchafkopfMultiAgentEnv()

  # initialize with fixed cards from transcript (convert JSON lists to tuples)
  obs_dict, _ = env.reset_with_fixed_cards(
      [[(c[0], c[1]) for c in game_transcript["player_hands"][str(i)]] for i in range(4)])
  # initialize current agent/obs from obs_dict (env returns the acting agent's obs as sole entry)
  current_agent_id, current_obs = deepcopy(next(iter(obs_dict.items())))
  
  # store simple (obs, action, reward, agent_id) tuples during stepping
  step_sequence: list[tuple[dict, int, float, str]] = []
  final_rewards: dict[str, float] = {'player_0':0, 'player_1':0, 'player_2':0, 'player_3':0}
  # ------------------ BIDDING STAGE ------------------
  game_player: int | None = None
  game_type: tuple[int | None, int | None] | None = None

  # Determine who bid and which game type (if any)
  if len(game_transcript["bidding_round"]) != 4:  # not all said weiter
    player_bidding: str | None = None
    for i in range(1, 5):
      if "Vortritt" not in game_transcript["bidding_round"][-i]:
        player_bidding = game_transcript["bidding_round"][-i]
        break

    if player_bidding.startswith("Ex-Sauspieler"):
      game_player = game_transcript["player_dict"][player_bidding.split(" ")[0] + " " + player_bidding.split(" ")[1]]
    else:
      game_player = game_transcript["player_dict"][player_bidding.split(" ")[0]]

    # remove player name in case it contains one of the following words
    player_bidding = player_bidding.split(' ', 1)[1]
    if "Hundsgfickte" in player_bidding:
      game_type = (0, 0)
    elif "Blaue" in player_bidding:
      game_type = (2, 0)
    elif "Alte" in player_bidding:
      game_type = (3, 0)
    elif "Schelle" in player_bidding:
      game_type = (0, 2)
    elif "Herz" in player_bidding:
      game_type = (1, 2)
    elif "Gras" in player_bidding:
      game_type = (2, 2)
    elif "Eichel" in player_bidding:
      game_type = (3, 2)
    elif "Wenz" in player_bidding:
      game_type = (None, 1)

  # four bidding actions (weiter or selected game)
  for i in range(4):
    action: tuple[int | None, int | None] = (None, None)
    if game_player is not None and i == game_player:
      action = game_type

    action_idx: int = int(preprocess_action(Rules.BIDDING, action).item())

    agent_id: str = current_agent_id
    obs: dict = current_obs

    next_obs_dict, rewards, terminateds, truncateds, _ = env.step({agent_id: action_idx})

    # reward is only non-zero at terminal
    reward_value: float = rewards.get(agent_id, 0.0)
    if not terminateds.get("__all__", False):
      reward_value = 0.0

    # store simple tuple during stepping
    step_sequence.append((deepcopy(obs), action_idx, float(reward_value), agent_id))

    if terminateds.get("__all__", False):
      break

    # update current obs/agent for next decision using obs_dict
    current_agent_id, current_obs = deepcopy(next(iter(next_obs_dict.items())))

  # ------------------ CONTRA / RETOUR ------------------
  if len(game_transcript["bidding_round"]) != 4:  # only if a game was announced
    con_ret: list[int] = [game_transcript["player_dict"][p] for p in game_transcript["kontra"]]

    # CONTRA stage (4 decisions)
    for i in range(4):
      action = False
      if len(con_ret) > 0 and i == con_ret[0]:
        action = True

      action_idx = int(preprocess_action(Rules.CONTRA, action).item())

      agent_id = current_agent_id
      obs = current_obs

      next_obs_dict, rewards, terminateds, truncateds, _ = env.step({agent_id: action_idx})

      reward_value = rewards.get(agent_id, 0.0)
      if not terminateds.get("__all__", False):
        reward_value = 0.0

      step_sequence.append((deepcopy(obs), action_idx, float(reward_value), agent_id))

      if terminateds.get("__all__", False):
        break

      current_agent_id, current_obs = deepcopy(next(iter(next_obs_dict.items())))

    # RETOUR stage (4 decisions)
    if len(con_ret) > 0:
      for i in range(4):
        action = False
        if len(con_ret) == 2 and i == con_ret[1]:
          action = True

        action_idx = int(preprocess_action(Rules.RETOUR, action).item())

        agent_id = current_agent_id
        obs = current_obs

        next_obs_dict, rewards, terminateds, truncateds, _ = env.step({agent_id: action_idx})

        reward_value = rewards.get(agent_id, 0.0)
        if not terminateds.get("__all__", False):
          reward_value = 0.0

        step_sequence.append((deepcopy(obs), action_idx, float(reward_value), agent_id))

        if terminateds.get("__all__", False):
          break

        current_agent_id, current_obs = deepcopy(next(iter(next_obs_dict.items())))

    # ------------------ TRICK STAGE ------------------
    for trick in range(8):
      for c in range(4):
        action = tuple(game_transcript["course_of_game"][trick][c])
        action_idx = int(preprocess_action(Rules.TRICK, action).item())

        agent_id = current_agent_id
        obs = current_obs

        next_obs_dict, rewards, terminateds, truncateds, _ = env.step({agent_id: action_idx})

        # save final rewards for later
        final_rewards = rewards

        #set final rewards later
        step_sequence.append((deepcopy(obs), action_idx, 0.0, agent_id))

        if terminateds.get("__all__", False):
          break

        current_agent_id, current_obs = deepcopy(next(iter(next_obs_dict.items())))

  # finished full transcript; build rows from step_sequence
  rows_by_agent: dict[str, list[tuple[dict, int, float]]] = {agent: [] for agent in env.agents}
  
  # group steps by agent
  for obs, action, reward, agent_id in step_sequence:
    rows_by_agent[agent_id].append((obs, action, reward))
  
  # build final rows with next_obs and terminateds/truncateds
  final_rows: list[dict[str, Any]] = []
  for agent in env.agents:
    agent_steps: list[tuple[dict, int, float]] = rows_by_agent[agent]
    for i, (obs, action, reward) in enumerate(agent_steps):
      # next_obs is the next obs for the same agent, or a copy of current obs for the last step
      next_obs: dict = agent_steps[i + 1][0] if i < len(agent_steps) - 1 else obs
      
      # terminateds is True only for the last step of this agent's sequence
      is_last_step: bool = (i == len(agent_steps) - 1)
      
      # Use final_rewards for last step, otherwise use stored reward (or 0 for trick stage)
      step_reward: float = final_rewards.get(agent, 0.0) if is_last_step else reward
      
      final_rows.append({
        "obs": {
          "player_hand": obs["player_hand"],
          "info_vector": obs["info_vector"],
          "action_history": obs["action_history"],
          "action_history_len": obs["action_history_len"],
          "action_mask": obs["action_mask"],
        },
        "new_obs": {
          "player_hand": next_obs["player_hand"],
          "info_vector": next_obs["info_vector"],
          "action_history": next_obs["action_history"],
          "action_history_len": next_obs["action_history_len"],
          "action_mask": next_obs["action_mask"],
        },
        "actions": action,
        "rewards": step_reward,
        "terminateds": is_last_step,
        "truncateds": False,
        "agent_id": agent,
        "eps_id": game_id,
      })
  
  return final_rows

def convert_to_serializable(obj: Any) -> Any:
  """Recursively convert numpy arrays and torch tensors to JSON-serializable types.

  Parameters
  ----------
  obj : Any
      Object to convert.

  Returns
  -------
  Any
      JSON-serializable representation.
  """
  if isinstance(obj, dict):
    return {key: convert_to_serializable(value) for key, value in obj.items()}
  elif isinstance(obj, (list, tuple)):
    return [convert_to_serializable(item) for item in obj]
  elif isinstance(obj, np.ndarray):
    return obj.tolist()
  elif isinstance(obj, torch.Tensor):
    return obj.detach().cpu().numpy().tolist()
  elif isinstance(obj, (np.integer, np.floating)):
    return obj.item()
  elif isinstance(obj, (int, float, str, bool, type(None))):
    return obj
  else:
    return str(obj)


def preprocess_action(stage: int, action: Any) -> torch.Tensor:
  """Convert a game action to its discrete index tensor.

  Parameters
  ----------
  stage : int
      Current game stage (``Rules.BIDDING``, ``Rules.CONTRA``, etc.).
  action : Any
      The action (game type, bool, or card).

  Returns
  -------
  torch.Tensor
      Scalar long tensor with the action index.
  """
  index: int | None = None
  if stage == Rules.BIDDING:
    index = rules.games.index(action)
  elif stage == Rules.CONTRA or stage == Rules.RETOUR:
    if action == True:
      index = 10
    else:
      index = 9
  else:  # trick stage
    index = 11 + rules.cards.index(action)
  action_representation: np.ndarray = np.zeros(43)
  action_representation[index] = 1
  #return torch.tensor(action_representation).float()
  return torch.tensor(index, dtype=torch.long)


if __name__ == '__main__':
  main()