from schafkopfrl.rules import Rules
from schafkopfrl.memory import Memory
import random


class RandomCowardPlayer():
  '''
  Player that chooses a game (except solo) and cards randomly
  '''

  def __init__(self, id):
    self.id = id
    # state, action, reward, logprob trajectory
    self.memory = Memory()
    self.cards = []
    self.davongelaufen = False
    self.rules = Rules()

  def take_cards(self, cards):
    self.cards = cards


  def call_game_type(self, game_state):
    '''
    Calls a game according to the following rules:
    - randomly chooses between a Sauspiel and weiter!

    :param game_state: the current game state
    :type game_state: game_state
    :return: the game to play
    :rtype: list
    '''
    allowed_games = self.rules.allowed_games(self.cards)

    allowed_games = [game for game in allowed_games if game in [[0, 0], [2, 0], [3, 0], [None, None]]]

    selected_game = random.choice(allowed_games)

    return selected_game



  def select_card(self, game_state):

    selected_card = random.choice(self.rules.allowed_cards(game_state, self))

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

    # REWARD SHAPING: reward for each action = number of points made/lost
    for i in range(steps_per_game-1):
      points = game_state.count_points(i)
      if game_state.trick_owner[i] == self.id:
        rewards[i+1] += points/5
      elif (self.id in game_state.get_player_team() and game_state.trick_owner[i] in game_state.get_player_team()) or (self.id not in game_state.get_player_team() and game_state.trick_owner[i] not in game_state.get_player_team()):
        rewards[i + 1] += points/5
      else:
        rewards[i + 1] -= points/5
    #steps_since_last_reward = len(self.memory.actions) - len(self.memory.rewards)
    #rewards = steps_since_last_reward*[0]
    rewards[-1] += reward
    self.memory.rewards += rewards
    is_terminal = steps_per_game * [False]
    is_terminal[-1] = True
    self.memory.is_terminals += is_terminal

