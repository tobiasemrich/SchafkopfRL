from schafkopfrl.players.player import Player
from schafkopfrl.rules import Rules
from schafkopfrl.memory import Memory
import random


class RandomCowardPlayer(Player):
  '''
  Player that chooses actions randomly, except the game that he selects, where he will never select a solo
  '''

  def __init__(self):
    super().__init__()

  def act(self, state):
    allowed_actions, gamestate = state["allowed_actions"],  state["game_state"]

    if gamestate.game_stage == self.rules.BIDDING:
      allowed_actions = [game for game in allowed_actions if game in [[0, 0], [2, 0], [3, 0], [None, None]]]

    selected_action = random.choice(allowed_actions)
    return selected_action, 1


