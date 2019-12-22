import random
from os import listdir

import torch

from schafkopfrl.gamestate import GameState
from schafkopfrl.memory import Memory
from schafkopfrl.models.actor_critic6_ego import ActorCriticNetwork6_ego
from schafkopfrl.players.rl_player import RlPlayer
from schafkopfrl.rules import Rules
import numpy as np


class SchafkopfGame:
    """
    This class is responsible for managing a round of 4 players to play the game.
    The round is created given the policies of the 4 players.
    The main function is play_one_game.
  """

    def __init__(self, policy1, policy2, policy3, policy4, seed=None):
        policy1.eval()
        policy2.eval()
        policy3.eval()
        policy4.eval()
        self.players = [RlPlayer(0, policy1), RlPlayer(1, policy2), RlPlayer(2, policy3), RlPlayer(3, policy4)]
        self.rules = Rules()
        if seed != None:
            self.setSeed(seed)

        # some logging counts
        self.game_count = [0, 0, 0, 0]  # weiter, sauspiel, wenz, solo
        self.won_game_count = [0, 0, 0, 0]

    def play_one_game(self):
        """
    Simulates one game of Schafkopf between 4 players with given policies:
    1) selects a dealer
    2) asks every player to call a game and selects the highest
    3) performs the trick phase where each player plays one card and the highest card gets the trick
    4) determins winners and rewards

    :return: return the final game_state of the game
    :rtype: game_state
    """
        dealer = random.choice([0, 1, 2, 3])
        game_state = GameState(dealer)

        # deal cards
        random.shuffle(self.rules.cards)
        for p in range(4):
            self.players[p].take_cards(self.rules.cards[8 * p:8 * (p + 1)])

        # every player beginning with the one left of the dealer calls his game
        current_highest_game = [None, None]
        game_player = None
        for p in range(4):
            current_player_id = (game_state.first_player + p) % 4
            current_player = self.players[current_player_id]
            game_type, prob = current_player.call_game_type(game_state)
            game_state.bidding_round[current_player_id] = game_type
            game_state.action_probabilities[0][current_player_id] = prob
            if current_highest_game[1] == None or (not game_type[1] == None and game_type[1] > current_highest_game[1]):
                current_highest_game = game_type
                game_player = current_player_id
        game_state.game_player = game_player
        game_state.game_type = current_highest_game

        if game_state.game_type != [None, None]:
            # then play the game
            first_player_of_trick = game_state.first_player
            for trick_number in range(8):
                game_state.trick_number = trick_number
                for p in range(4):
                    current_player_id = (first_player_of_trick + p) % 4
                    current_player = self.players[current_player_id]
                    card, prob = current_player.select_card(game_state)
                    game_state.player_plays_card(current_player_id, card, prob)
                first_player_of_trick = game_state.trick_owner[trick_number]

        # determine winner(s) and rewards
        player_rewards = game_state.get_rewards()
        for p in range(4):
            self.players[p].retrieve_reward(player_rewards[p], game_state)

        # update statistics just for logging purposes
        self.update_statistics(game_state)

        return game_state

    def setSeed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self.rules = Rules()

    def get_player_memories(self, ids=None):
        """
    summarizes the memories (states, actions, rewards, ...) of selected players
    :param ids: the ids of the player memories
    :type ids: list
    :return: the combined memories of the selected players
    :rtype: memory
    """
        memory = Memory()
        if ids == None:
            ids = range(4)
        for i in ids:
            memory.append_memory(self.players[i].memory)
        return memory

    def print_game(self, game_state):
        br = "Bidding Round: "
        for i in range(4):
            if game_state.first_player == i:
                br += "(" + str(i) + "*)"
            else:
                br += "(" + str(i) + ")"
            if game_state.bidding_round[i][1] != None:
                if game_state.bidding_round[i][0] != None:
                    br += self.rules.card_color[game_state.bidding_round[i][0]] + " "
                br += self.rules.game_names[game_state.bidding_round[i][1]] + " "
            else:
                br += "weiter! "
            br += "[{:0.3f}]  ".format(game_state.action_probabilities[0][i])
        print(br + "\n")

        played_game_str = "Played Game: "
        if game_state.game_type[1] != None:
            if game_state.game_type[0] != None:
                played_game_str += self.rules.card_color[game_state.game_type[0]] + " "
            played_game_str += self.rules.game_names[game_state.game_type[1]] + " "
        else:
            played_game_str += "no game "
        print(played_game_str + "played by player: " + str(game_state.game_player) + "\n")

        if game_state.game_type[1] != None:
            print("Course of game")
            for trick in range(8):
                trick_str = ""
                for player in range(4):
                    trick_str_ = "(" + str(player)
                    if (trick == 0 and game_state.first_player == player) or (
                            trick != 0 and game_state.trick_owner[trick - 1] == player):
                        trick_str_ += "^"
                    if game_state.trick_owner[trick] == player:
                        trick_str_ += "*"
                    trick_str_ += ")"

                    if game_state.course_of_game_playerwise[trick][player] in self.rules.get_sorted_trumps(
                            game_state.game_type):
                        trick_str_ += '\033[91m'

                    trick_str_ += self.rules.card_color[game_state.course_of_game_playerwise[trick][player][0]] + " " + \
                                  self.rules.card_number[game_state.course_of_game_playerwise[trick][player][1]]

                    trick_str_ += "[{:0.3f}]".format(game_state.action_probabilities[trick+1][player])
                    if game_state.course_of_game_playerwise[trick][player] in self.rules.get_sorted_trumps(
                            game_state.game_type):
                        trick_str_ += '\033[0m'
                        trick_str += trick_str_.ljust(39)
                    else:
                        trick_str += trick_str_.ljust(30)
                print(trick_str)

            print("\nScores: " + str(game_state.scores) + "\n")
        rewards = game_state.get_rewards()
        print("Rewards: " + str(rewards))

    def update_statistics(self, game_state):
        if game_state.game_type[1] == None:
            self.game_count[0] += 1
        else:
            self.game_count[game_state.game_type[1] + 1] += 1
            if game_state.get_rewards()[game_state.game_player] > 0:
                self.won_game_count[game_state.game_type[1] + 1] += 1


def main():
    all_rewards = np.array([0, 0, 0, 0])

    policy = ActorCriticNetwork6_ego()
    # take the newest generation available
    # file pattern = policy-000001.pt
    generations = [int(f[:8]) for f in listdir("../policies") if f.endswith(".pt")]
    if len(generations) > 0:
        max_gen = max(generations)
        policy.load_state_dict(torch.load("../policies/" + str(max_gen).zfill(8) + ".pt"))

    # policy.eval()
    policy.to(device='cuda')

    gs = SchafkopfGame(policy, policy, policy, policy, 1)

    for i in range(2):
        print("playing game " + str(i))
        gs = SchafkopfGame(policy, policy, policy, policy, 1) #<--------remove this
        game_state = gs.play_one_game()
        rewards = np.array(game_state.get_rewards())
        all_rewards += rewards
        gs.print_game(game_state)
    print(all_rewards)
    print(sum(all_rewards))


if __name__ == '__main__':
    main()
