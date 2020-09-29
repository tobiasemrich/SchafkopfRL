from schafkopfrl.gamestate import GameState
from schafkopfrl.players.player import Player
from schafkopfrl.rules import Rules
import numpy as np
import torch
from torch.distributions import Categorical
from schafkopfrl.memory import Memory
from schafkopfrl.utils import one_hot_games, one_hot_cards


class RlPlayer(Player):

  def __init__(self, policy):
    super().__init__()
    self.memory = Memory()
    self.policy = policy

  def act(self, state):

    encoded_state = self.policy.preprocess(state)

    action_probs, value = self.policy(encoded_state)

    dist = Categorical(action_probs)
    action = dist.sample()

    self.memory.states.append([s.detach() for s in encoded_state])
    self.memory.actions.append(action)
    self.memory.logprobs.append(dist.log_prob(action).detach())

    action_prob = dist.probs[action].item()  # only for debugging purposes

    if action.item() < 9: # bidding round
      return self.rules.games[action], action_prob
    elif action.item() < 11: # contra or retour
      return action.item() - 9 == 0, action_prob
    else: #play card
      return self.rules.cards[action - 11], action_prob


  def call_game_type(self, game_state):

    allowed_games = self.rules.allowed_games(self.cards)


    # hard coded solo and wenz decisions
    for solo in [[0, 2], [1, 2], [2, 2], [3, 2]]:
      trump_count = 0
      for card in self.rules.get_sorted_trumps(solo):
        if card in self.cards:
          trump_count+= 1
      if trump_count >= 7:
        allowed_games = [solo]
        break;

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
        allowed_games = [[None, 1]]

    action, prob = self.act(
      self.policy.preprocess(game_state, self),
      torch.tensor(np.concatenate((one_hot_games(allowed_games), np.zeros(2), np.zeros(32)))).float())

    selected_game = self.rules.games[action]

    return selected_game, prob

  def retrieve_reward(self, reward):
    steps =  len(self.memory.states) - len(self.memory.rewards)
    rewards = steps * [0.]

    rewards[-1] += reward
    self.memory.rewards += rewards
    is_terminal = steps * [False]
    is_terminal[-1] = True
    self.memory.is_terminals += is_terminal

  def print_action_probs(self, distribution):
    card_names = [self.rules.card_color[color] + " " +self.rules.card_number[number] for [color, number] in self.rules.cards]
    game_names = ["weiter", "schellen Sauspiel", "gras Sauspiel", "eichel Sauspiel", "Wenz", "schellen Solo", "herz Solo", "gras Solo", "eichel Solo"]
    action_list = game_names + card_names
    sorted_by_probability = sorted(zip(action_list, distribution), key=lambda tup: tup[1], reverse=True)
    print(sorted_by_probability)
