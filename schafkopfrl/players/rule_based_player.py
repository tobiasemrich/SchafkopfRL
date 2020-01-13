from schafkopfrl.players.player import Player
from schafkopfrl.rules import Rules
from schafkopfrl.memory import Memory
import random


class RuleBasedPlayer(Player):

  def __init__(self, id):
    self.id = id
    # state, action, reward, logprob trajectory
    self.memory = Memory()
    self.cards = []
    self.davongelaufen = False
    self.rules = Rules()

  def take_cards(self, cards):
    super().take_cards(cards)
    self.spieler_or_mitspieler = False


  def call_game_type(self, game_state):
    '''
    Calls a game according to the following rules:
    - if in some solo more then 7 trumps then play the solo
    - if at least 2 unter and for all colors in hand the ace is also in hand then play wenz
    - if 4 or more trumps then play a sauspiel on the color with the fewest other cards of that color
    - otherwise weiter

    :param game_state: the current game state
    :type game_state: game_state
    :return: the game to play
    :rtype: list
    '''
    allowed_games = self.rules.allowed_games(self.cards)


    # solo heuristic
    for solo in [[0, 2], [1, 2], [2, 2], [3, 2]]:
      trump_count = 0
      for card in self.rules.get_sorted_trumps(solo):
        if card in self.cards:
          trump_count+= 1
      if trump_count >= 7:
        return [solo]

    # wenz heurisitc
    wenz_count = len([card for card in [[0, 3], [1, 3], [2, 3], [3, 3]] if card in self.cards])
    spazen_count = 0
    if wenz_count >= 2:
      for color in range(4):
        if [color, 7] in self.cards:
          continue
        for number in range(7):
          if number == 3:
            continue
          if [color, number] in self.cards:
            spazen_count += 1
      if spazen_count < 2:
        return [[None, 1]]

    # sauspiel heuristic
    trumps = [card for card in self.rules.get_sorted_trumps([0,0]) if card in self.cards]
    if len(trumps) >= 4:
      non_trump_cards = [card for card in self.cards if card not in trumps]
      allowed_saupiele = [game for game in allowed_games if game in [[0, 0], [2, 0], [3, 0]]]

      best_color = -1
      best_color_count = 10
      if len(allowed_saupiele) > 0:
        for [color, _] in allowed_saupiele:
          color_count = len([[color,_] for [color,_] in  non_trump_cards])
          if color_count < best_color_count:
            best_color = color
        return [best_color, 0]

    return [None, None]



  def select_card(self, game_state):
    allowed_cards = self.rules.allowed_cards(game_state, self)
    selected_card = random.choice(allowed_cards)

    #precompute some interesting features
    if len(self.cards) == 8 and game_state.game_player == self.id or (game_state.game_type in [[0, 0], [2, 0], [3, 0]] and [game_state.game_type[0],7] in self.cards):
      self.spieler_or_mitspieler = True
    played_cards_in_trick = game_state.played_cards % 4
    trump_cards = [trump for trump in self.rules.get_sorted_trumps() if trump in allowed_cards]
    color_aces = [ace for ace in allowed_cards if ace in [[0, 7], [1, 7], [2, 7], [3, 7]]]


    if game_state.game_type in [[0, 0], [2, 0], [3, 0]]: #Sauspiel
      if [1, 7] in color_aces:
        color_aces.remove([1, 7])
      if self.spieler_or_mitspieler:
        if played_cards_in_trick == 0:
          # play highest or lowest trump if possible
          if len(trump_cards) > 0:
            selected_card = random.choice(trump_cards[[0, -1]])
          #otherwise play an ace
          elif len(color_aces) > 0:
            selected_card = random.choice(color_aces)
        else:
          pass

    elif game_state.game_type in [[0, 2], [1, 2], [2, 2], [3, 2]]: #Solo
      if [game_state.game_type[0], 7] in color_aces:
        color_aces.remove([1, 7])
      if played_cards_in_trick == 0:
        if len(trump_cards) > 0:
          selected_card = trump_cards[-1]
        else:
          pass
    else: # Wenz
      if played_cards_in_trick == 0:
        pass
      else:
        pass

    self.cards.remove(selected_card)
    #Davonlaufen needs to be tracked
    if game_state.game_type[1] == 0: # Sauspiel
      first_player_of_trick = game_state.first_player if game_state.trick_number == 0 else game_state.trick_owner[game_state.trick_number - 1]
      rufsau = [game_state.game_type[0],7]
      if game_state.game_type[0] == selected_card[0] and selected_card != rufsau and first_player_of_trick == self.id and selected_card not in self.rules.get_sorted_trumps(game_state.game_type) and rufsau in self.cards:
        self.davongelaufen = True
    return selected_card

  def lowest_card_that_takes_trick(self, allowed_cards, trick):
    pass
  def retrieve_reward(self, reward, game_state):
    pass

