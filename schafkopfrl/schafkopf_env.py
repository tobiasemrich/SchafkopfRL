import random

from public_gamestate import PublicGameState
from rules import Rules
import numpy as np

class SchafkopfEnv:

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
    state["current_player_cards"] = self.player_cards[self.gamestate.current_player]
    state["allowed_actions"] = self.rules.allowed_games(self.player_cards[self.gamestate.current_player])


    return state, [0, 0, 0, 0], False

  def step(self, action, action_prob=1):

    state = {}

    if self.gamestate.game_stage == Rules.BIDDING:
      self.gamestate.bidding_round[self.gamestate.current_player] = action
      self.gamestate.action_probabilities[0][self.gamestate.current_player] = action_prob
      self.gamestate.current_player = (self.gamestate.current_player+1)%4
      if self.gamestate.bidding_round[self.gamestate.current_player] != None:
        self.gamestate.game_player, self.gamestate.game_type = self.rules.highest_game(self.gamestate.bidding_round, self.gamestate.first_player)
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
      self.gamestate.action_probabilities[2][self.gamestate.current_player] = action_prob
      self.gamestate.current_player = (self.gamestate.current_player + 1) % 4
      if self.gamestate.retour[self.gamestate.current_player] != None:
        self.gamestate.game_stage = Rules.TRICK

    elif self.gamestate.game_stage == Rules.TRICK:

      self.gamestate.course_of_game_playerwise[self.gamestate.trick_number][self.gamestate.current_player] = action
      self.gamestate.course_of_game[self.gamestate.trick_number][self.gamestate.played_cards % 4] = action
      self.gamestate.action_probabilities[3+self.gamestate.trick_number][self.gamestate.current_player] = action_prob
      self.player_cards[self.gamestate.current_player].remove(action)
      self.gamestate.played_cards += 1
      self.gamestate.current_player = (self.gamestate.current_player + 1) % 4

      if self.gamestate.played_cards % 4 == 0:  # trick complete
        first_player = self.gamestate.first_player if self.gamestate.trick_number == 0 else self.gamestate.trick_owner[self.gamestate.trick_number - 1]
        trick_cards = self.gamestate.course_of_game_playerwise[self.gamestate.trick_number]
        trick_owner = self.rules.trick_owner(trick_cards,
                                        first_player,
                                        self.gamestate.game_type)
        self.gamestate.trick_owner[self.gamestate.trick_number] = trick_owner
        self.gamestate.scores[trick_owner] += self.rules.count_points(trick_cards)
        self.gamestate.current_player = trick_owner

        # Davonlaufen needs to be tracked (after trick is complete such that no other player can use this information beforehand)
        if self.gamestate.game_type[1] == 0:  # Sauspiel
          first_player_of_trick = self.gamestate.first_player if self.gamestate.trick_number == 0 else self.gamestate.trick_owner[self.gamestate.trick_number - 1]
          card_played = trick_cards[first_player_of_trick]
          rufsau = [self.gamestate.game_type[0], 7]
          if self.gamestate.game_type[0] == card_played[0] and card_played != rufsau and card_played not in self.rules.get_sorted_trumps(self.gamestate.game_type) and rufsau in self.player_cards[self.gamestate.current_player]:
            self.gamestate.davongelaufen = first_player_of_trick

        self.gamestate.trick_number += 1


    state["game_state"] = self.gamestate
    state["allowed_actions"] = self.rules.allowed_actions(self.gamestate, self.player_cards[self.gamestate.current_player])
    state["current_player_cards"] = self.player_cards[self.gamestate.current_player]

    terminal = False
    rewards = [0, 0, 0, 0]
    if self.gamestate.trick_number == 8:
      terminal = True
      rewards = self.get_rewards()

    return state, rewards, terminal

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
    return player_team

  def get_rewards(self):
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
    if np.any(self.gamestate.contra):
      reward *= 2
    if np.any(self.gamestate.retour):
      reward *= 2

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

  def print_game(self):

    br = ""
    #only print player cards when game is not finished
    if self.gamestate.trick_number != 8:
      for p in range(4):
        br += "Player "+ str(p) +" cards: "+ str(self.player_cards[p]) + " \n"

    br += "Bidding Round: "
    for i in range(4):
      if self.gamestate.first_player == i:
        br += "(" + str(i) + "^)"
      else:
        br += "(" + str(i) + ")"
      if self.gamestate.bidding_round[i] is None:
        br += "None "
      else:
        if self.gamestate.bidding_round[i][1] != None:
          if self.gamestate.bidding_round[i][0] != None:
            br += self.rules.card_color[self.gamestate.bidding_round[i][0]] + " "
          br += self.rules.game_names[self.gamestate.bidding_round[i][1]] + " "
        else:
          br += "weiter! "
        br += "[{:0.3f}]  ".format(self.gamestate.action_probabilities[0][i])
    print(br + "\n")

    played_game_str = "Played Game: "
    if self.gamestate.game_type[1] != None:
      if self.gamestate.game_type[0] != None:
        played_game_str += self.rules.card_color[self.gamestate.game_type[0]] + " "
      played_game_str += self.rules.game_names[self.gamestate.game_type[1]] + " "
    else:
      played_game_str += "no game "
    print(played_game_str + "played by player: " + str(self.gamestate.game_player) + "\n")
    contra_str = "Contra/Retour: "
    for p in range(4):
      if self.gamestate.contra[p]:
        contra_str += "player " + str(p)
        contra_str += "[{:0.3f}]".format(self.gamestate.action_probabilities[1][p])
        contra_str += "  |   "
    for p in range(4):
      if self.gamestate.retour[p]:
        contra_str += "player " + str(p)
        contra_str += "[{:0.3f}]".format(self.gamestate.action_probabilities[2][p])
        contra_str += "  |   "
    print(contra_str + "\n")

    if self.gamestate.game_type[1] != None:
      print("Course of game")
      for trick in range(8):
        trick_str = ""
        for player in range(4):
          trick_str_ = "(" + str(player)
          if (trick == 0 and self.gamestate.first_player == player) or (
                  trick != 0 and self.gamestate.trick_owner[trick - 1] == player):
            trick_str_ += "^"
          if self.gamestate.trick_owner[trick] == player:
            trick_str_ += "*"
          trick_str_ += ")"

          if self.gamestate.course_of_game_playerwise[trick][player] == [None, None]:
            trick_str_ += "None"
          else:
            if self.gamestate.course_of_game_playerwise[trick][player] in self.rules.get_sorted_trumps(
                    self.gamestate.game_type):
              trick_str_ += '\033[91m'

            trick_str_ += self.rules.card_color[self.gamestate.course_of_game_playerwise[trick][player][0]] + " " + \
                          self.rules.card_number[self.gamestate.course_of_game_playerwise[trick][player][1]]

            trick_str_ += "[{:0.3f}]".format(self.gamestate.action_probabilities[trick + 3][player])
            if self.gamestate.course_of_game_playerwise[trick][player] in self.rules.get_sorted_trumps(
                    self.gamestate.game_type):
              trick_str_ += '\033[0m'
              trick_str += trick_str_.ljust(39)
            else:
              trick_str += trick_str_.ljust(30)
        print(trick_str)

      print("\nScores: " + str(self.gamestate.scores) + "\n")
    rewards = self.get_rewards()
    print("Rewards: " + str(rewards))


  def set_state(self, game_state, player_cards):

    self.gamestate = game_state
    self.player_cards = player_cards

    state = {}
    state["game_state"] = self.gamestate
    state["current_player_cards"] = self.player_cards[self.gamestate.current_player]
    state["allowed_actions"] = self.rules.allowed_actions(self.gamestate, self.player_cards[self.gamestate.current_player])
    if self.gamestate.played_cards == 32:
      return state, self.get_rewards(), True
    else:
      return state, [0, 0, 0, 0], False
