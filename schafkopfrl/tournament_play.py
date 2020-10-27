import time

import torch

from models.actor_critic_lstm import ActorCriticNetworkLSTM
from models.hand_predictor import HandPredictor
from players.mcts_player import MCTSPlayer
from players.random_coward_player import RandomCowardPlayer
from players.random_player import RandomPlayer
from players.rl_player import RlPlayer
from players.rule_based_player import RuleBasedPlayer
from players.smart_mcts_player import SmartMCTSPlayer
from schafkopf_env import SchafkopfEnv
from settings import Settings


def main():
  mcts_player_1 = MCTSPlayer(5, 20, RandomPlayer())


  policy = ActorCriticNetworkLSTM().to(Settings.device)
  policy.load_state_dict(torch.load("../policies/pretrained/lstm-policy.pt"))
  rl_player = RlPlayer(policy, action_shaping=False, eval=True)

  mcts_player_2 = MCTSPlayer(10, 40, RandomPlayer())

  smart_mcts_player = SmartMCTSPlayer(5, 20, RandomPlayer(), HandPredictor().to(Settings.device))

  #players = [mcts_player_2,mcts_player_1, mcts_player_2,mcts_player_1]
  #players = [RuleBasedPlayer(),RuleBasedPlayer(), RuleBasedPlayer(),mcts_player]
  #players = [MCTSPlayer(10, 40, RandomPlayer()), MCTSPlayer(10, 40, RandomPlayer()), RuleBasedPlayer(), RuleBasedPlayer()]
  players = [mcts_player_1, mcts_player_2, smart_mcts_player,  rl_player]

  # create a game simulation
  schafkopf_env = SchafkopfEnv(None)

  i_episode = 0
  cummulative_reward = [0, 0, 0, 0]
  # tournament loop
  for _ in range(0, 1000):

    # play a bunch of games
    t0 = time.time()
    state, reward, terminal = schafkopf_env.reset()
    while not terminal:
      action, prob = players[state["game_state"].current_player].act(state)
      state, reward, terminal = schafkopf_env.step(action, prob)

    i_episode += 1
    cummulative_reward = [cummulative_reward[i] + reward[i] for i in range(4)]

    t1 = time.time()



    schafkopf_env.print_game()

    print("--------Episode: " + str(i_episode) + " game simulation (s) = " + str(t1 - t0))
    print("--------Cummulative reward: " + str(cummulative_reward))
    print("--------per game reward: " + str([i /i_episode for i in cummulative_reward] ))
    #print("--------MCTS rewards: " + str(((cummulative_reward[1] + cummulative_reward[3]) / i_episode)/2))
    #print("--------MCTS rewards: " + str(((cummulative_reward[1] + cummulative_reward[3]) / i_episode)/2))


if __name__ == '__main__':
  main()