# contains information about the game that is known by all players
import numpy as np
from schafkopfrl.rules import Rules


class GameState:
    """
    The GameState contains all public visible game information like dealer, game_type, game_player, course_of_game, current_scores ...
    """
    def __init__(self, dealer):

        self.rules = Rules()

        self.dealer = dealer
        self.game_type = [None, None]
        self.game_player = None
        self.first_player = (dealer + 1) % 4
        self.trick_number = 0
        self.played_cards = 0

        # who wants to play what
        self.bidding_round = [[None, None] for x in range(4)]

        # cards ordered by players
        self.course_of_game_playerwise = [[[None, None] for x in range(4)] for y in range(8)]

        # cards ordered by the time they were played
        self.course_of_game = [[[None, None] for x in range(4)] for y in range(8)]

        # which player took the trick
        self.trick_owner = [None] * 8

        self.scores = [0, 0, 0, 0]

        #for debugging purposes remember probs for picking an action
        self.action_probabilities=[[[None, None] for x in range(4)] for y in range(9)]

    def player_plays_card(self, player_id, card, prob):
        self.course_of_game_playerwise[self.trick_number][player_id] = card
        self.course_of_game[self.trick_number][self.played_cards % 4] = card
        self.action_probabilities[self.trick_number+1][player_id] = prob
        self.played_cards += 1

        if self.played_cards % 4 == 0:  # trick complete
            trick_owner = self.highest_card(self.trick_number)
            self.trick_owner[self.trick_number] = trick_owner
            self.scores[trick_owner] += self.count_points(self.trick_number)
            self.trick_number += 1

    # return the player id who played the highest card in the trick
    def highest_card(self, trick_nr):
        first_player = self.first_player if trick_nr == 0 else self.trick_owner[trick_nr - 1]
        cards_list = self.course_of_game_playerwise[trick_nr]

        highest_card_index = first_player
        for i in range(1, 4):
            player_id = (first_player + i) % 4
            if self.rules.higher_card(self.game_type, cards_list[highest_card_index], cards_list[player_id]):
                highest_card_index = player_id
        return highest_card_index

    # return the number of points in trick
    def count_points(self, trick):
        return sum([self.rules.card_scores[number] for color, number in self.course_of_game_playerwise[trick]])

    def get_player_team(self):
        player_team = [self.game_player]
        if self.game_type[1] == 0:  # Sauspiel
            for trick in range(8):
                for player_id in range(4):
                    if self.course_of_game_playerwise[trick][player_id] == [self.game_type[0], 7]:
                        player_team.append(player_id)
        # TODO: add davonlaufen (since this function is also used during a game to check if the teams are already known)

        return player_team

    def get_rewards(self):

        rewards = [0, 0, 0, 0]

        if self.game_type == [None, None]:
            return rewards

        player_team_points = 0
        for player_id in self.get_player_team():
            player_team_points += self.scores[player_id]

        # basic reward
        reward = self.rules.reward_basic[self.game_type[1] + 1]

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
            joint_player_team_cards += [i[p] for i in self.course_of_game_playerwise]
        for trump in reversed(self.rules.get_sorted_trumps(self.game_type)):
            if trump in joint_player_team_cards:
                laufende += 1
            else:
                break
        if laufende == 0:  # calculate gegenlaufende
            for trump in reversed(self.rules.get_sorted_trumps(self.game_type)):
                if trump not in joint_player_team_cards:
                    laufende += 1
                else:
                    break
        if laufende >= self.rules.min_laufende[self.game_type[1]]:
            reward += laufende * self.rules.reward_laufende

        # calculate reward distribution
        if player_team_points <= self.rules.winning_thresholds[2]:
            reward *= -1
        if self.game_type[1] >= 1:  # Solo or Wenz
            rewards = [-reward] * 4
            rewards[self.game_player] = 3 * reward
        else:
            for player_id in range(4):
                if player_id in self.get_player_team():
                    rewards[player_id] = reward
                else:
                    rewards[player_id] = -reward

        # rewards = (np.array(rewards) + np.array(self.scores)/10).tolist()
        return rewards
