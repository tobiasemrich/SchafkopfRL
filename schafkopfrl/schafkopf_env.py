from random import random

from public_gamestate import PublicGameState
from rules import Rules
import numpy as np

class SchafkpfEnv:

  rules = Rules()

  def __init__(self, seed=None):
    self.gamestate = None
    self.player_cards = [None, None, None, None]
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

    return state, 0, False

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

    terminal = False
    if self.trick_number == 8:
      terminal = True

    return state, 0, terminal

  # return the number of points in trick
  #def count_points(self, trick):
  #  return sum([self.rules.card_scores[number] for color, number in self.course_of_game_playerwise[trick]])

  def get_player_team(self):
    player_team = [self.gamestate.game_player]
    if self.gamestate.game_type[1] == 0:  # Sauspiel
      for trick in range(8):
        for player_id in range(4):
          if self.gamestate.course_of_game_playerwise[trick][player_id] == [self.gamestate.game_type[0], 7]:
            player_team.append(player_id)
        # TODO: add davonlaufen (since this function is also used during a game to check if the teams are already known)

    return player_team

  def getRewards(self):
    if self.gamestate.trick_number != 8:
      return None

    rewards = [0, 0, 0, 0]

    if self.gamestate.game_type == [None, None]:
      return rewards

    player_team_points = 0
    for player_id in self.get_player_team():
      player_team_points += self.gamestate.scores[player_id]

    # basic reward
    reward = self.rules.reward_basic[self.gamestate.game_type[1] + 1]

    # add schneider/schwarz bonus
    if player_team_points > self.rules.winning_thresholds[4] or player_team_points <= self.rules.winning_thresholds[
      0]:  # schwarz
      reward += self.rules.reward_schneider[2]
    elif player_team_points > self.rules.winning_thresholds[3] or player_team_points <= \
            self.rules.winning_thresholds[1]:  # schneider
      reward += self.rules.reward_schneider[1]

    # add Laufende
    laufende = 0
    joint_player_team_cards = []
    for p in self.get_player_team():
      joint_player_team_cards += [i[p] for i in self.gamestate.course_of_game_playerwise]
    for trump in reversed(self.rules.get_sorted_trumps(self.gamestate.game_type)):
      if trump in joint_player_team_cards:
        laufende += 1
      else:
        break
    if laufende == 0:  # calculate gegenlaufende
      for trump in reversed(self.rules.get_sorted_trumps(self.gamestate.game_type)):
        if trump not in joint_player_team_cards:
          laufende += 1
        else:
          break
    if laufende >= self.rules.min_laufende[self.gamestate.game_type[1]]:
      reward += laufende * self.rules.reward_laufende

    # contra/retour doubles
    reward *= 2 ** len(self.gamestate.contra_retour)

    # calculate reward distribution
    if player_team_points <= self.rules.winning_thresholds[2]:
      reward *= -1
    if self.gamestate.game_type[1] >= 1:  # Solo or Wenz
      rewards = [-reward] * 4
      rewards[self.gamestate.game_player] = 3 * reward
    else:
      for player_id in range(4):
        if player_id in self.get_player_team():
          rewards[player_id] = reward
        else:
          rewards[player_id] = -reward

    return rewards



  def setSeed(self, seed):
    if seed != None:
      np.random.seed(seed)
      random.seed(seed)
