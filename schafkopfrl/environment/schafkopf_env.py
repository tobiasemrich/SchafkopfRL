import random
from typing import Dict, Any

import numpy as np

from environment.public_gamestate import PublicGameState
from environment.rules import Rules


class SchafkopfEnv():
    
    def __init__(self, seed=None):
        super().__init__()
        self.public_gamestate = None
        self.player_cards = [None, None, None, None]
        self.last_allowed_actions = []
        self.rules = Rules()


    def _compile_state(self):
        state = {}
        state["game_state"] = self.public_gamestate
        state["allowed_actions"] = self.rules.allowed_actions(self.public_gamestate,
                                                              self.player_cards[self.public_gamestate.current_player])
        state["current_player_cards"] = self.player_cards[self.public_gamestate.current_player]
        self.last_allowed_actions = state["allowed_actions"]
        return state

    def reset(self, seed):
        if seed != None:
            np.random.seed(seed)
            random.seed(seed)
        self.public_gamestate = PublicGameState(3)

        # deal cards
        cards = self.rules.cards.copy()
        random.shuffle(cards)
        self.player_cards = [cards[8 * p:8 * (p + 1)] for p in range(4)]

        self.public_gamestate.game_stage = Rules.BIDDING

        return self._compile_state(), None

    def step(self, action):

        if action not in self.last_allowed_actions:
            raise Exception("Action not allowed!")

        if self.public_gamestate.game_stage == Rules.BIDDING:
            self.public_gamestate.bidding_round[self.public_gamestate.current_player] = action
            self.public_gamestate.current_player = (self.public_gamestate.current_player + 1) % 4
            if self.public_gamestate.bidding_round[self.public_gamestate.current_player] != None:
                self.public_gamestate.game_player, self.public_gamestate.game_type = self.rules.highest_game(
                    self.public_gamestate.bidding_round, self.public_gamestate.first_player)
                self.public_gamestate.game_stage = Rules.CONTRA

        elif self.public_gamestate.game_stage == Rules.CONTRA:
            self.public_gamestate.contra[self.public_gamestate.current_player] = action
            self.public_gamestate.current_player = (self.public_gamestate.current_player + 1) % 4
            if self.public_gamestate.contra[self.public_gamestate.current_player] != None:
                self.public_gamestate.game_stage = Rules.RETOUR
                if all(not x for x in self.public_gamestate.contra):
                    self.public_gamestate.game_stage = Rules.TRICK

        elif self.public_gamestate.game_stage == Rules.RETOUR:
            self.public_gamestate.retour[self.public_gamestate.current_player] = action
            self.public_gamestate.current_player = (self.public_gamestate.current_player + 1) % 4
            if self.public_gamestate.retour[self.public_gamestate.current_player] != None:
                self.public_gamestate.game_stage = Rules.TRICK

        elif self.public_gamestate.game_stage == Rules.TRICK:

            self.public_gamestate.course_of_game_playerwise[self.public_gamestate.trick_number][
                self.public_gamestate.current_player] = action
            self.public_gamestate.course_of_game[self.public_gamestate.trick_number][self.public_gamestate.played_cards % 4] = action
            self.player_cards[self.public_gamestate.current_player].remove(action)
            self.public_gamestate.played_cards += 1
            self.public_gamestate.current_player = (self.public_gamestate.current_player + 1) % 4

            if self.public_gamestate.played_cards % 4 == 0:  # trick complete
                first_player = self.public_gamestate.first_player if self.public_gamestate.trick_number == 0 else \
                self.public_gamestate.trick_owner[self.public_gamestate.trick_number - 1]
                trick_cards = self.public_gamestate.course_of_game_playerwise[self.public_gamestate.trick_number]
                trick_owner = self.rules.trick_owner(trick_cards,
                                                     first_player,
                                                     self.public_gamestate.game_type)
                self.public_gamestate.trick_owner[self.public_gamestate.trick_number] = trick_owner
                self.public_gamestate.scores[trick_owner] += self.rules.count_points(trick_cards)
                self.public_gamestate.current_player = trick_owner

                # Davonlaufen needs to be tracked (after trick is complete such that no other player can use this information beforehand)
                if self.public_gamestate.game_type[1] == 0:  # Sauspiel
                    first_player_of_trick = self.public_gamestate.first_player if self.public_gamestate.trick_number == 0 else \
                    self.public_gamestate.trick_owner[self.public_gamestate.trick_number - 1]
                    card_played = trick_cards[first_player_of_trick]
                    rufsau = [self.public_gamestate.game_type[0], 7]
                    if self.public_gamestate.game_type[0] == card_played[
                        0] and card_played != rufsau and card_played not in self.rules.get_sorted_trumps(
                            self.public_gamestate.game_type) and rufsau in self.player_cards[first_player_of_trick]:
                        self.public_gamestate.davongelaufen = first_player_of_trick

                self.public_gamestate.trick_number += 1


        terminal = False
        rewards = [0, 0, 0, 0]
        if self.public_gamestate.trick_number == 8:
            terminal = True
            rewards = self.get_rewards()

        return self._compile_state(), rewards, terminal

    def render(self):
        # prints the game
        br = ""
        # only print player cards when game is not finished
        if self.public_gamestate.trick_number != 8:
            for p in range(4):
                br += "Player " + str(p) + " cards: " + str(self.player_cards[p]) + " \n"

        br += "Bidding Round: "
        for i in range(4):
            if self.public_gamestate.first_player == i:
                br += "(" + str(i) + "^)"
            else:
                br += "(" + str(i) + ")"
            if self.public_gamestate.bidding_round[i] is None:
                br += "None "
            else:
                if self.public_gamestate.bidding_round[i][1] != None:
                    if self.public_gamestate.bidding_round[i][0] != None:
                        br += self.rules.card_color[self.public_gamestate.bidding_round[i][0]] + " "
                    br += self.rules.game_names[self.public_gamestate.bidding_round[i][1]] + " "
                else:
                    br += "weiter! "
        print(br + "\n")

        played_game_str = "Played Game: "
        if self.public_gamestate.game_type[1] != None:
            if self.public_gamestate.game_type[0] != None:
                played_game_str += self.rules.card_color[self.public_gamestate.game_type[0]] + " "
            played_game_str += self.rules.game_names[self.public_gamestate.game_type[1]] + " "
        else:
            played_game_str += "no game "
        print(played_game_str + "played by player: " + str(self.public_gamestate.game_player) + "\n")
        contra_str = "Contra/Retour: "
        for p in range(4):
            if self.public_gamestate.contra[p]:
                contra_str += "player " + str(p)
                contra_str += "  |   "
        for p in range(4):
            if self.public_gamestate.retour[p]:
                contra_str += "player " + str(p)
                contra_str += "  |   "
        print(contra_str + "\n")

        if self.public_gamestate.game_type[1] != None:
            print("Course of game")
            for trick in range(8):
                trick_str = ""
                for player in range(4):
                    trick_str_ = "(" + str(player)
                    if (trick == 0 and self.public_gamestate.first_player == player) or (
                            trick != 0 and self.public_gamestate.trick_owner[trick - 1] == player):
                        trick_str_ += "^"
                    if self.public_gamestate.trick_owner[trick] == player:
                        trick_str_ += "*"
                    trick_str_ += ")"

                    if self.public_gamestate.course_of_game_playerwise[trick][player] == [None, None]:
                        trick_str_ += "None"
                    else:
                        if self.public_gamestate.course_of_game_playerwise[trick][player] in self.rules.get_sorted_trumps(
                                self.public_gamestate.game_type):
                            trick_str_ += '\033[91m'

                        trick_str_ += self.rules.card_color[
                                          self.public_gamestate.course_of_game_playerwise[trick][player][0]] + " " + \
                                      self.rules.card_number[self.public_gamestate.course_of_game_playerwise[trick][player][1]]

                        if self.public_gamestate.course_of_game_playerwise[trick][player] in self.rules.get_sorted_trumps(
                                self.public_gamestate.game_type):
                            trick_str_ += '\033[0m'
                            trick_str += trick_str_.ljust(39)
                        else:
                            trick_str += trick_str_.ljust(30)
                print(trick_str)

            print("\nScores: " + str(self.public_gamestate.scores) + "\n")
        rewards = self.get_rewards()
        print("Rewards: " + str(rewards))

    def get_player_team(self):
        player_team = [self.public_gamestate.game_player]
        if self.public_gamestate.game_type[1] == 0:  # Sauspiel
            for trick in range(8):
                for player_id in range(4):
                    if self.public_gamestate.course_of_game_playerwise[trick][player_id] == [self.public_gamestate.game_type[0], 7]:
                        player_team.append(player_id)
        return player_team

    def get_rewards(self):
        if self.public_gamestate.trick_number != 8:
            return None

        rewards = [0, 0, 0, 0]

        if self.public_gamestate.game_type == [None, None]:
            return rewards

        player_team_points = 0
        player_team = self.get_player_team()
        for player_id in player_team:
            player_team_points += self.public_gamestate.scores[player_id]

        # basic reward
        reward = self.rules.reward_basic[self.public_gamestate.game_type[1] + 1]

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
        for p in player_team:
            joint_player_team_cards += [i[p] for i in self.public_gamestate.course_of_game_playerwise]
        for trump in reversed(self.rules.get_sorted_trumps(self.public_gamestate.game_type)):
            if trump in joint_player_team_cards:
                laufende += 1
            else:
                break
        if laufende == 0:  # calculate gegenlaufende
            for trump in reversed(self.rules.get_sorted_trumps(self.public_gamestate.game_type)):
                if trump not in joint_player_team_cards:
                    laufende += 1
                else:
                    break
        if laufende >= self.rules.min_laufende[self.public_gamestate.game_type[1]]:
            reward += laufende * self.rules.reward_laufende

        # contra/retour doubles
        if np.any(self.public_gamestate.contra):
            reward *= 2
        if np.any(self.public_gamestate.retour):
            reward *= 2

        # calculate reward distribution
        if player_team_points <= self.rules.winning_thresholds[2]:
            reward *= -1
        if self.public_gamestate.game_type[1] >= 1:  # Solo or Wenz
            rewards = [-reward] * 4
            rewards[self.public_gamestate.game_player] = 3 * reward
        else:
            for player_id in range(4):
                if player_id in player_team:
                    rewards[player_id] = reward
                else:
                    rewards[player_id] = -reward

        return rewards