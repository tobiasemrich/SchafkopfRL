from schafkopfrl.players.player import Player


class MCTSPlayer(Player):

  def __init__(self, id, policy):
    super().__init__(id)


  def call_game_type(self, game_state):
    selected_game = None
    return selected_game, 1



  def select_card(self, game_state):
    selected_card = None
    return selected_card, 1
