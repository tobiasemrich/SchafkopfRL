import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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

rules = Rules()


def main():
  all_rows = []
  # load and preprocess database
  count = 0
  with open('data/normal_games.json', 'r') as file:
    games = json.load(file)

    with open('data/expert_data.jsonl', 'w') as f:

      for game_id, g in enumerate(games):
        game = g # GameTranscript.from_dict(g)
        if len(game["sonderregeln"]) == 0:
          count += 1
          final_rows = get_states_actions(game, game_id=game_id)
          all_rows += final_rows
          if count % 1000 == 0:
            print("Read " + str(count) + " normal games")
            print(len(all_rows))
          # For testing, limit to 100 games
          #if count >= 1000:
          #  break
          for row in final_rows:
            # Convert entire row recursively
            row_serializable = convert_to_serializable(row)
            json.dump(row_serializable, f)
            f.write('\n')
      #df = pd.DataFrame(all_rows)
      #parquet_path = '/home/git/SchafkopfRL/expert_data.parquet'
      #df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')

  print(f"Saved {len(all_rows)} rows to expert_data.jsonl")


def get_states_actions(game_transcript, game_id):
  env = SchafkopfMultiAgentEnv()

  # initialize with fixed cards from transcript
  obs_dict, _ = env.reset_with_fixed_cards([game_transcript["player_hands"][str(i)] for i in range(4)])
  # initialize current agent/obs from obs_dict (env returns the acting agent's obs as sole entry)
  current_agent_id, current_obs = deepcopy(next(iter(obs_dict.items())))
  
  # store simple (obs, action, reward, agent_id) tuples during stepping
  step_sequence = []
  final_rewards = {'player_0':0, 'player_1':0, 'player_2':0, 'player_3':0}
  # ------------------ BIDDING STAGE ------------------
  game_player = None
  game_type = None

  # Determine who bid and which game type (if any)
  if len(game_transcript["bidding_round"]) != 4:  # not all said weiter
    player_bidding = None
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
      game_type = [0, 0]
    elif "Blaue" in player_bidding:
      game_type = [2, 0]
    elif "Alte" in player_bidding:
      game_type = [3, 0]
    elif "Schelle" in player_bidding:
      game_type = [0, 2]
    elif "Herz" in player_bidding:
      game_type = [1, 2]
    elif "Gras" in player_bidding:
      game_type = [2, 2]
    elif "Eichel" in player_bidding:
      game_type = [3, 2]
    elif "Wenz" in player_bidding:
      game_type = [None, 1]

  # four bidding actions (weiter or selected game)
  for i in range(4):
    action = [None, None]
    if game_player is not None and i == game_player:
      action = game_type

    action_idx = int(preprocess_action(Rules.BIDDING, action).item())

    agent_id = current_agent_id
    obs = current_obs

    next_obs_dict, rewards, terminateds, truncateds, _ = env.step({agent_id: action_idx})

    # reward is only non-zero at terminal
    reward_value = rewards.get(agent_id, 0.0)
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
    con_ret = [game_transcript["player_dict"][p] for p in game_transcript["kontra"]]

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
        action = game_transcript["course_of_game"][trick][c]
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
  rows_by_agent = {agent: [] for agent in env.agents}
  
  # group steps by agent
  for obs, action, reward, agent_id in step_sequence:
    rows_by_agent[agent_id].append((obs, action, reward))
  
  # build final rows with next_obs and terminateds/truncateds
  final_rows = []
  for agent in env.agents:
    agent_steps = rows_by_agent[agent]
    for i, (obs, action, reward) in enumerate(agent_steps):
      # next_obs is the next obs for the same agent, or a copy of current obs for the last step
      next_obs = agent_steps[i + 1][0] if i < len(agent_steps) - 1 else obs
      
      # terminateds is True only for the last step of this agent's sequence
      is_last_step = (i == len(agent_steps) - 1)
      
      # Use final_rewards for last step, otherwise use stored reward (or 0 for trick stage)
      step_reward = final_rewards.get(agent, 0.0) if is_last_step else reward
      
      final_rows.append({
        "obs": {
          "player_hand": obs["player_hand"],
          "action_history": obs["action_history"],
          "action_history_len": obs["action_history_len"],
          "action_mask": obs["action_mask"],
        },
        "new_obs": {
          "player_hand": next_obs["player_hand"],
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

def convert_to_serializable(obj):
  """Recursively convert non-serializable objects (numpy arrays, torch tensors) to JSON-serializable formats."""
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


def preprocess_action(stage, action):
  index = None
  if stage == Rules.BIDDING:
    index = rules.games.index(action)
  elif stage == Rules.CONTRA or stage == Rules.RETOUR:
    if action == True:
      index = 10
    else:
      index = 9
  else:  # trick stage
    index = 11 + rules.cards.index(action)
  action_representation = np.zeros(43)
  action_representation[index] = 1
  #return torch.tensor(action_representation).float()
  return torch.tensor(index, dtype=torch.long)


if __name__ == '__main__':
  main()