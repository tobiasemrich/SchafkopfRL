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