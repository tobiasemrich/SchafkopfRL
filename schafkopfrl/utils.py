from random import random

import numpy as np
from schafkopfrl.rules import Rules


'''
    encodes a card using 8 (for number) + 4 (for color) bits
'''
def two_hot_encode_card(card):
    encoding = np.zeros(12)
    encoding[card[1]] = 1
    encoding[8+card[0]] = 1
    return encoding
'''
    encodes a game using 3 (for gametype) + 4 (for color) bits
'''
def two_hot_encode_game(game):
    encoding = np.zeros(7)
    if game[1] is not None:
        encoding[game[1]] = 1
    if game[0] is not None:
        encoding[3 + game[0]] = 1
    return encoding

def one_hot_games(games):
  one_hot_games = np.zeros(9)
  for game in games:
    one_hot_games[Rules().games.index(game)] = 1
  return one_hot_games

def one_hot_cards(cards):
  one_hot_cards = np.zeros(32)
  for card in cards:
    one_hot_cards[Rules().cards.index(card)] = 1
  return one_hot_cards

def sample_player_hands(game_state, ego_player_hand, ego_player_id):


  played_cards = [card for trick in game_state.couse_of_game for card in trick if card != [None, None]]
  remaining_cards = [card for card in Rules.cards if ((card not in played_cards) and (card not in ego_player_hand))]

  ####################### analyse for possible constraints: player may #################################################
  #1 - need to be trump free, if not played trump while trump was first card
  #2 - need to be color free, if not played color while color was played
  #3 - must not have the rufsau, if gave contra
  #4 - need at least 1,2, 3 of rufcolor (including rufsau) if he ran away
  #5 - If in the current trick it is searched and the ace has not been played, then either first_player must have 3 additional color cards or another player must have the ace.
  #    Additionally, all player who played color card must not have the ace
  # contraints has the form [0, None, 1, 0, None, None] means (0: must not have, None: any number, [1, 2, 3...] at least this amount)
  # [schelle, herz, gras, eichel, trump, rufsau]

  rules = Rules()
  trumps = rules.get_sorted_trumps(game_state.game_type)
  constraints = [[None] * 6]*4

  #constraint 3
  if game_state.game_type[1] == 0: #Sauspiel
    if len(game_state.contra_retour) > 0:
      constraints[game_state.contra_retour[0]][5] = 0

  davongelaufen_player = None
  gesucht = False

  for trick in range(game_state.trick_number + 1):
    first_player = game_state.first_player
    if trick != 0:
      first_player = game_state.trick_owner[trick-1]
    first_card = game_state.couse_of_game[trick]
    trump_played = False
    if first_card in trumps:
      trump_played = True

    # constraint 4
    if game_state.game_type[1] == 0:
      if first_card[0] == game_state.game_type[0] and first_card not in trumps:
        if trick != game_state.trick_number:
          if gesucht == False and [game_state.game_type[0], 7] not in game_state.couse_of_game[trick]:
            davongelaufen_player = first_player
            constraints[player_id][game_state.game_type[0]] = 3
            constraints[player_id][5] = 1
            gesucht = True


    for i, card in enumerate(game_state.couse_of_game[trick]):
      player_id = (first_player+i)%4
      if card == [None, None] or i == 0:
        continue
      else:
        #constraint 1
        if trump_played and game_state.couse_of_game[trick][i] not in trumps:
          constraints[player_id][4] = 0
        elif not trump_played:
          # constraint 2
          if first_card[0] != card[0] and card[0] not in trumps:
            constraints[player_id][first_card[0]] = 0
        #discount davonlaufen necessary cards (constraint 4)
        if game_state.game_type[1] == 0 and davongelaufen_player == player_id and card not in trumps and card[0] == game_state.game_type[0]:
          constraints[game_state.contra_retour[player_id]][game_state.game_type[0]]-=1
          if card == [game_state.game_type[0], 7]:
            constraints[player_id][5] = 0

  ############################################ distribute cards ######################################################
  needed_player_cards = [8, 8, 8, 8]

  for trick in range(game_state.trick_number + 1):
    for i, card in enumerate(game_state.course_of_game_playerwise):
      if card != [None, None]:
        needed_player_cards[i] -= 1

  needed_player_cards[ego_player_id] = 0

  valid_card_distribution = False

  while not valid_card_distribution:
    valid_card_distribution = True
    player_cards = [[], [], [], []]
    player_cards[ego_player_id] = ego_player_hand
    random.shuffle(remaining_cards)

    from_card = 0
    for i, nededed_cards in enumerate(needed_player_cards):
      player_cards[i] = remaining_cards[from_card:from_card+nededed_cards]
      from_card += nededed_cards

    ## check constraints ##
    for player_id in range(4):
      #color constraints
      for color in range(4):
        color_cards = [card for card in player_cards[player_id] if card[0] == color and card not in trumps]
        if (constraints[player_id][color] == 0 and len(color_cards)>0) or (constraints[player_id][color] is not None and constraints[player_id][color] < len(color_cards) ):
          valid_card_distribution == False
      #trump constraint
      if constraints[player_id][color] == 0 and len([card for card in player_cards[player_id] if card in trumps]) != 0:
        valid_card_distribution == False
      # rufsau constraint
      if constraints[player_id][5] == 0 and [game_state.game_type[0], 7] in player_cards[player_id]:
        valid_card_distribution == False
      elif constraints[player_id][5] == 1 and [game_state.game_type[0], 7] not in player_cards[player_id]:
        valid_card_distribution == False

    #constraint 5
    first_card = game_state.couse_of_game[game_state.trick_number][0]
    if game_state.game_type[1] == 0 and first_card[0] == game_state.game_type[0] and first_card not in trumps and not gesucht and [game_state.game_type[0], 7] not in game_state.couse_of_game[game_state.trick_number]:
      first_player = game_state.first_player
      if trick != 0:
        first_player = game_state.trick_owner[trick - 1]
      ruf_color_cards = [card for card in player_cards[first_player] if card[0] == game_state.game_type[0] and card not in trumps]

      if len(ruf_color_cards) < 3 or [game_state.game_type[0], 7] not in player_cards[first_player]:
        valid_card_distribution == False
        # now check if one of the unplayed players has the ace, this may save us
        for i in range(3):
          if game_state.couse_of_game[game_state.trick_number][i+1] == [None, None] and [game_state.game_type[0], 7] in player_cards[(first_player+1+i)%4]:
            valid_card_distribution == True

  return player_cards















