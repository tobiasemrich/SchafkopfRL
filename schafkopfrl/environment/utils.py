from typing import Optional

import numpy as np
import numpy.typing as npt

from .rules import Rules

Card = tuple[int, int]
GameType = tuple[Optional[int], Optional[int]]

def two_hot_encode_card(card: Card) -> npt.NDArray[np.int32]:
    """Encode a card as a two-hot vector (8 number bits + 4 color bits).

    Parameters
    ----------
    card : Card
        A ``(color, number)`` tuple.

    Returns
    -------
    numpy.ndarray
        Binary array of shape ``(12,)``.
    """
    encoding: npt.NDArray[np.int32] = np.zeros(12, dtype=np.int32)
    encoding[card[1]] = 1
    encoding[8+card[0]] = 1
    return encoding
def two_hot_encode_game(game: GameType) -> npt.NDArray[np.int32]:
    """Encode a game type as a two-hot vector (3 type bits + 4 color bits).

    Parameters
    ----------
    game : GameType
        A ``(color, type)`` tuple; either component may be None.

    Returns
    -------
    numpy.ndarray
        Binary array of shape ``(7,)``.
    """
    encoding: npt.NDArray[np.int32] = np.zeros(7, dtype=np.int32)
    if game[1] is not None:
        encoding[game[1]] = 1
    if game[0] is not None:
        encoding[3 + game[0]] = 1
    return encoding

def one_hot_games(games: list[GameType]) -> npt.NDArray[np.int32]:
  """One-hot encode a list of game types against the canonical game list.

  Parameters
  ----------
  games : list[GameType]
      Game types to encode.

  Returns
  -------
  numpy.ndarray
      Binary array of shape ``(9,)``.
  """
  one_hot_games: npt.NDArray[np.int32] = np.zeros(9, dtype=np.int32)
  for game in games:
    one_hot_games[Rules().games.index(game)] = 1
  return one_hot_games

def one_hot_cards(cards: list[Card]) -> npt.NDArray[np.int32]:
  """One-hot encode a list of cards against the canonical 32-card deck.

  Parameters
  ----------
  cards : list[Card]
      Cards to encode.

  Returns
  -------
  numpy.ndarray
      Binary array of shape ``(32,)``.
  """
  one_hot_cards: npt.NDArray[np.int32] = np.zeros(32, dtype=np.int32)
  for card in cards:
    one_hot_cards[Rules().cards.index(card)] = 1
  return one_hot_cards
















