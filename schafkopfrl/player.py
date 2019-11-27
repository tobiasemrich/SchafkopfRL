from schafkopfrl.rules import Rules
import numpy as np
import torch
from torch.distributions import Categorical
from schafkopfrl.memory import Memory
from schafkopfrl.utils import one_hot_games, one_hot_cards


class Player():

  def __init__(self, id, policy):
    self.id = id
    #state, action, reward, logprob trajectory
    self.memory = Memory()
    self.cards = []
    self.davongelaufen = False
    self.rules = Rules()

    self.policy = policy

  def take_cards(self, cards):
    self.cards = cards
    self.davongelaufen = False

  def act(self, state, allowed_actions):
    action_probs, value = self.policy(
      state,
      allowed_actions)

    dist = Categorical(action_probs)
    #self.print_action_probs(action_probs.tolist())
    action = dist.sample()

    self.memory.states.append([s.detach() for s in state])
    self.memory.allowed_actions.append(allowed_actions.detach())
    self.memory.actions.append(action)
    self.memory.logprobs.append(dist.log_prob(action).detach())

    return action.item()

  def call_game_type(self, game_state):

    allowed_games = self.rules.allowed_games(self.cards)

    for solo in [[0, 2], [1, 2], [2, 2], [3, 2]]:
      trump_count = 0
      for card in self.rules.get_sorted_trumps(solo):
        if card in self.cards:
          trump_count+= 1
      if trump_count >= 7:
        allowed_games = [solo]
        break;

    wenz_count = len([card for card in [[0, 3], [1, 3], [2, 3], [3, 3]] if card in self.cards])
    if wenz_count >= 2:
      for color in range(4):
        for number in range(7, -1, -1):
          if [color, number] in self.cards:
            wenz_count += 1
          elif number == 3:
            continue
          else:
            break
      if wenz_count >= 7:
        allowed_games = [[None, 1]]


    action = self.act(
      self.policy.preprocess(game_state, self),
      torch.tensor(np.append(one_hot_games(allowed_games), np.zeros(32))).float())

    selected_game = self.rules.games[action]

    return selected_game

    ##random
    #found_game = False
    #while not found_game:
    #  game_index = np.random.choice(9,1, p=[0.4, 0.15,0.15,0.15,0.03, 0.03,0.03, 0.03,0.03])[0]
    #  game = self.rules.games[game_index]
    #  if game in self.allowed_games():
    #    return game


  def select_card(self, game_state):
    action = self.act(
      self.policy.preprocess(game_state, self),
      torch.tensor(np.append(np.zeros(9), one_hot_cards(self.rules.allowed_cards(game_state, self)))).float())

    selected_card = self.rules.cards[action - 9]

    ##random
    #selected_card = random.choice(self.allowed_cards(game_state))

    self.cards.remove(selected_card)
    #Davonlaufen needs to be tracked
    if game_state.game_type[1] == 0: # Sauspiel
      first_player_of_trick = game_state.first_player if game_state.trick_number == 0 else game_state.trick_owner[game_state.trick_number - 1]
      rufsau = [game_state.game_type[0],7]
      if game_state.game_type[0] == selected_card[0] and selected_card != rufsau and first_player_of_trick == self.id and selected_card not in self.rules.get_sorted_trumps(game_state.game_type) and rufsau in self.cards:
        self.davongelaufen = True
    return selected_card

  def retrieve_reward(self, reward, game_state):
    steps_per_game = 9
    if game_state.game_type == [None, None]:
      steps_per_game=1
    rewards = steps_per_game*[0.]
    for i in range(steps_per_game-1):
      points = game_state.count_points(i)
      if game_state.trick_owner[i] == self.id:
        rewards[i+1] += points/5
      elif (self.id in game_state.get_player_team() and game_state.trick_owner[i] in game_state.get_player_team()) or (self.id not in game_state.get_player_team() and game_state.trick_owner[i] not in game_state.get_player_team()):
        rewards[i + 1] += points/5

    #steps_since_last_reward = len(self.memory.actions) - len(self.memory.rewards)
    #rewards = steps_since_last_reward*[0]
    rewards[-1] += reward
    self.memory.rewards += rewards
    is_terminal = steps_per_game * [False]
    is_terminal[-1] = True
    self.memory.is_terminals += is_terminal

  def print_action_probs(self, distribution):

    card_names = [self.rules.card_color[color] + " " +self.rules.card_number[number] for [color, number] in self.rules.cards]
    game_names = ["weiter", "schellen Sauspiel", "gras Sauspiel", "eichel Sauspiel", "Wenz", "schellen Solo", "herz Solo", "gras Solo", "eichel Solo"]
    action_list = game_names + card_names
    sorted_by_probability = sorted(zip(action_list, distribution), key=lambda tup: tup[1], reverse=True)
    print(sorted_by_probability)
