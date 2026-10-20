import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from schafkopfrl.environment.pettingzoo_env import SchafkopfAECEnv
from schafkopfrl.evaluation import Player, RawStatePlayer, play_game
from schafkopfrl.policy.pimc_policy import PIMCPolicy
from schafkopfrl.policy.rulebased_policy import RuleBasedPolicy


def main() -> None:
  """Run a round-robin tournament between PIMC and rule-based policies."""

  participants: list[Player] = [RawStatePlayer(PIMCPolicy(10, 40)), RawStatePlayer(RuleBasedPolicy())]

  number_of_games: int = 10
  env = SchafkopfAECEnv()

  for i in range(len(participants)):
    for j in range(i+1, len(participants)):
      p1 = participants[i]
      p2 = participants[j]

      cummulative_reward: list[float] = [0, 0, 0, 0]
      for k in range(2): #run the same tournament twice with differen positions of players
        print(' ')
        if k == 0:
          players = [p1, p1, p2, p2]
        else:
          players = [p2, p2, p1, p1]
          cummulative_reward.reverse()

        for game_nr in range(1, number_of_games+1):
          rewards = play_game(env, players, seed=game_nr)
          cummulative_reward = [cummulative_reward[m] + rewards[m] for m in range(4)]

          if game_nr % 100 == 0:
            print('.', end = '')

      print("player "+str(i)+" vs. player "+str(j)+" = " + str((cummulative_reward[2] + cummulative_reward[3]) / (2*2*number_of_games)) + " to " +str((cummulative_reward[0] + cummulative_reward[1]) / (2*2*number_of_games)))


if __name__ == '__main__':
  main()
