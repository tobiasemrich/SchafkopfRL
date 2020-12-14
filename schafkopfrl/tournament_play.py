import time

import torch

from models.actor_critic_lstm import ActorCriticNetworkLSTM
from models.hand_predictor import HandPredictor
from players.pimc_player import PIMCPlayer
from players.random_coward_player import RandomCowardPlayer
from players.random_player import RandomPlayer
from players.rl_player import RlPlayer
from players.rule_based_player import RuleBasedPlayer
from players.hp_pimc_player import HPPIMCPlayer
from schafkopf_env import SchafkopfEnv
from settings import Settings


def main():

  pimc_player = PIMCPlayer(10, 40, RandomPlayer())

  policy = ActorCriticNetworkLSTM().to(Settings.device)
  policy.load_state_dict(torch.load("../policies/pretrained/lstm-policy.pt"))
  rl_player = RlPlayer(policy, action_shaping=False, eval=True)

  hp = HandPredictor().to(Settings.device)
  hp.load_state_dict(torch.load("../policies/pretrained/hand-predictor.pt"))
  smart_pimc_player = HPPIMCPlayer(30, 100, RandomPlayer(), HandPredictor().to(Settings.device))

  ip = ActorCriticNetworkLSTM().to(Settings.device)
  ip.load_state_dict(torch.load("../policies/pretrained/immitation-policy.pt"))
  immitation_player = RlPlayer(ip, action_shaping=False, eval=True)

  participants = [rl_player, immitation_player, smart_pimc_player, pimc_player, RuleBasedPlayer()]

  number_of_games = 1000

  for i in range(len(participants)):
    for j in range(i+1, len(participants)):

      p1 = participants[i]
      p2 = participants[j]

      cummulative_reward = [0, 0, 0, 0]
      for k in range(2): #run the same tournament twice with differen positions of players
        schafkopf_env = SchafkopfEnv(seed=1)
        if k == 0:
          players = [p1, p1, p2, p2]
        else:
          players = [p2, p2, p1, p1]
          cummulative_reward.reverse()

        # tournament loop
        for game_nr in range(1, number_of_games+1):
          state, reward, terminal = schafkopf_env.reset()
          while not terminal:
            action, prob = players[state["game_state"].current_player].act(state)
            state, reward, terminal = schafkopf_env.step(action, prob)

          cummulative_reward = [cummulative_reward[m] + reward[m] for m in range(4)]
          #schafkopf_env.print_game()

      print("player "+i+" vs. player "+j+" = " + str((cummulative_reward[2] + cummulative_reward[3]) / (2*2*number_of_games)) + " to " +str((cummulative_reward[0] + cummulative_reward[1]) / (2*2*number_of_games)))
      #print("--------Episode: " + str(i_episode) + " game simulation (s) = " + str(t1 - t0))
      #print("--------Cummulative reward: " + str(cummulative_reward))
      #print("--------per game reward: " + str([i /i_episode for i in cummulative_reward] ))
      #print("--------MCTS rewards: " + str(((cummulative_reward[1] + cummulative_reward[3]) / i_episode)/2))
      #print("--------MCTS rewards: " + str(((cummulative_reward[1] + cummulative_reward[3]) / i_episode)/2))


if __name__ == '__main__':
  main()