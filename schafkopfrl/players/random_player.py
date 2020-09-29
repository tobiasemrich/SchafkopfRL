from schafkopfrl.players.player import Player
import random


class RandomPlayer(Player):
  '''
  Player that chooses a game and cards randomly
  '''

  def __init__(self):
    super().__init__()

  def act(self, state):
    allowed_actions = state["allowed_actions"]
    selected_action = random.choice(allowed_actions)
    return selected_action, 1

