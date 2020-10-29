from players.player import Player
from rules import Rules
import numpy as np
import torch
from torch.distributions import Categorical
from memory import Memory
from utils import one_hot_games, one_hot_cards


class RlPlayer(Player):

  def __init__(self, policy, action_shaping=True, eval=False):
    super().__init__()
    self.memory = Memory()
    self.policy = policy
    self.action_shaping = action_shaping
    self.eval = eval

  def act(self, state):

    if self.action_shaping and state["game_state"].game_stage == Rules.BIDDING:
      state["allowed_actions"] = self.game_selection_shaping(state["current_player_cards"], state["allowed_actions"])

    encoded_state = self.policy.preprocess(state)

    action_probs, value = self.policy(encoded_state)
    dist = Categorical(action_probs)
    if self.eval:# during evaluation select best action
      action = torch.argmax(action_probs, 0)
      #action = action.item()
    else: #during training select action according to distribution
      action = dist.sample()
      self.memory.states.append([s.detach() for s in encoded_state])
      self.memory.actions.append(action)
      self.memory.logprobs.append(dist.log_prob(action).detach())

    action_prob = dist.probs[action].item()  # only for debugging purposes

    # translate output from NN to interpretable action
    if state["game_state"].game_stage == Rules.BIDDING:
      return self.rules.games[action], action_prob
    elif state["game_state"].game_stage == Rules.CONTRA or state["game_state"].game_stage == Rules.RETOUR:
      return action.item() - 9 == 0, action_prob
    else: #trick stage
      return self.rules.cards[action - 11], action_prob


  def game_selection_shaping(self, player_cards, allowed_games):

    # hard coded solo and wenz decisions
    for solo in [[0, 2], [1, 2], [2, 2], [3, 2]]:
      trump_count = 0
      for card in self.rules.get_sorted_trumps(solo):
        if card in player_cards:
          trump_count+= 1
      if trump_count >= 7:
        allowed_games = [solo]
        break;

    wenz_count = len([card for card in [[0, 3], [1, 3], [2, 3], [3, 3]] if card in player_cards])
    spazen_count = 0
    if wenz_count >= 2:
      for color in range(4):
        if [color, 7] in player_cards:
          continue
        for number in range(7):
          if number == 3:
            continue
          if [color, number] in player_cards:
            spazen_count += 1
      if spazen_count < 2:
        allowed_games = [[None, 1]]

    return allowed_games

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
