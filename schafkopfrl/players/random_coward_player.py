from schafkopfrl.players.player import Player
from schafkopfrl.rules import Rules
from schafkopfrl.memory import Memory
import random


class RandomCowardPlayer(Player):
  '''
  Player that chooses a game (except solo) and cards randomly
  '''

  def call_game_type(self, game_state):
    '''
    Calls a game according to the following rules:
    - randomly chooses between a Sauspiel and weiter!

    :param game_state: the current game state
    :type game_state: game_state
    :return: the game to play
    :rtype: list
    '''
    allowed_games = self.rules.allowed_games(self.cards)

    allowed_games = [game for game in allowed_games if game in [[0, 0], [2, 0], [3, 0], [None, None]]]

    selected_game = random.choice(allowed_games)

    return selected_game, 1

  def contra_retour(self, game_state):
    return False, 1

  def select_card(self, game_state):
    selected_card = random.choice(self.rules.allowed_cards(game_state, self))

    self.cards.remove(selected_card)
    #Davonlaufen needs to be tracked
    if game_state.game_type[1] == 0: # Sauspiel
      first_player_of_trick = game_state.first_player if game_state.trick_number == 0 else game_state.trick_owner[game_state.trick_number - 1]
      rufsau = [game_state.game_type[0],7]
      if game_state.game_type[0] == selected_card[0] and selected_card != rufsau and first_player_of_trick == self.id and selected_card not in self.rules.get_sorted_trumps(game_state.game_type) and rufsau in self.cards:
        self.davongelaufen = True
    return selected_card, 1

  def retrieve_reward(self, reward, game_state):
    pass

