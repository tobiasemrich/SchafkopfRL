from random import random

from public_gamestate import PublicGameState
from rules import Rules
import numpy as np

class SchafkpfEnv:

  rules = Rules()

  def __init__(self, seed=None):
    self.gamestate = None
    self.player_cards = None
    self.setSeed(seed)

  def reset(self):

    #select dealer
    dealer = random.choice([0, 1, 2, 3])
    self.gamestate = PublicGameState(dealer)

    # deal cards
    cards = self.rules.cards.copy()
    random.shuffle(cards)
    self.player_cards = [cards[8 * p:8 * (p + 1)] for p in range(4)]

    self.gamestate.game_stage = Rules.BIDDING

    state = {}
    state["game_state"] = self.gamestate
    state["curent_player_cards"] = self.player_cards[self.gamestate.current_player]
    state["allowed_actions"] = self.rules.allowed_games(self.player_cards[self.gamestate.current_player])

  def step(self, action, action_prob=1):


    if self.gamestate.game_stage == Rules.BIDDING:
      self.gamestate.bidding_round[self.gamestate.current_player] = action
      self.gamestate.action_probabilities[0][self.gamestate.current_player] = action_prob
      self.gamestate.current_player = (self.gamestate.current_player+1)%4
      if self.gamestate.bidding_round[self.gamestate.current_player] != None:
        self.gamestate.game_player, self.gamestate.game_player = self.rules.highest_game(self.gamestate.bidding_round)
        self.gamestate.game_stage = Rules.CONTRA

    elif self.gamestate.game_stage == Rules.CONTRA:
      self.gamestate.contra[self.gamestate.current_player] = action
      self.gamestate.action_probabilities[1][self.gamestate.current_player] = action_prob
      self.gamestate.current_player = (self.gamestate.current_player + 1) % 4
      if self.gamestate.contra[self.gamestate.current_player] != None:
        self.gamestate.game_stage = Rules.RETOUR
        if all(not x for x in self.gamestate.contra):
          self.gamestate.game_stage = Rules.TRICK

    elif self.gamestate.game_stage == Rules.RETOUR:
      self.gamestate.retour[self.gamestate.current_player] = action
      self.gamestate.action_probabilities[3][self.gamestate.current_player] = action_prob
      self.gamestate.current_player = (self.gamestate.current_player + 1) % 4
      if self.gamestate.retour[self.gamestate.current_player] != None:
        self.gamestate.game_stage = Rules.TRICK

    elif self.gamestate.game_stage == Rules.TRICK:

      self.gamestate.course_of_game_playerwise[self.gamestate.trick_number][self.gamestate.current_player] = action
      self.gamestate.course_of_game[self.gamestate.trick_number][self.gamestate.played_cards % 4] = action
      self.played_cards += 1
      self.gamestate.current_player = (self.gamestate.current_player + 1) % 4

      if self.played_cards % 4 == 0:  # trick complete
        first_player = self.first_player if self.gamestate.trick_nr == 0 else self.self.gamestate.trick_owner[self.gamestate.trick_nr - 1]
        trick_cards = self.gamestate.course_of_game_playerwise[self.gamestate.trick_nr]
        trick_owner = self.highest_card(trick_cards,
                                        first_player,
                                        self.gamestate.game_type)
        self.gamestate.trick_owner[self.gamestate.trick_number] = trick_owner
        self.gamestate.scores[trick_owner] += self.count_points(trick_cards)
        self.trick_number += 1
        self.gamestate.current_player = trick_owner

    state = {}
    state["game_state"] = self.gamestate
    state["curent_player_cards"] = self.player_cards[self.gamestate.current_player]
    state["allowed_actions"] = self.rules.allowed_games(self.player_cards[self.gamestate.current_player])

    #return (public_game_state, player_cards, allowed_actions)

  def setSeed(self, seed):
    if seed != None:
      np.random.seed(seed)
      random.seed(seed)
