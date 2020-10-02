import time

from players.mcts_player import MCTSPlayer
from players.random_player import RandomPlayer
from schafkopf_env import SchafkopfEnv


def main():
  mcts_player = MCTSPlayer(5, 20, RandomPlayer())
  players = [RandomPlayer(),mcts_player, RandomPlayer(),mcts_player]
  #players = [RandomPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer()]

  # create a game simulation
  schafkopf_env = SchafkopfEnv(None)

  i_episode = 0
  cummulative_reward = [0, 0, 0, 0]
  # tournament loop
  for _ in range(0, 500):

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


if __name__ == '__main__':
  main()