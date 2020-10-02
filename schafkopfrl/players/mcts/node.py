import numpy as np

class Node:
  '''
  pretty much copy paste from https://github.com/Taschee/schafkopf/blob/master/schafkopf/players/uct_player.py
  '''
  def __init__(self, parent, previous_action, game_state, player_hands, allowed_actions):
    self.parent = parent
    self.previous_action = previous_action
    self.children = []
    self.cumulative_rewards = [0, 0, 0, 0]
    self.visits = 0

    self.game_state = game_state
    self.player_hands = player_hands
    self.allowed_actions = allowed_actions

  def add_child(self, child_node):
    self.children.append(child_node)

  def is_terminal(self):
    if self.game_state.trick_number == 8:
      return True
    else:
      return False

  def get_average_reward(self, player):
    if self.visits > 0:
      return self.cumulative_rewards[player] / self.visits
    else:
      return 0

  def is_leaf(self):
    if len(self.children) == 0:
      return True
    else:
      return False


  def fully_expanded(self):
    if len(self.children) == len(self.allowed_actions):
      return True
    else:
      return False

  def best_child(self, ucb_const):
    if not self.is_leaf():
      return max(self.children, key=lambda child: child.ucb_value(ucb_const))

  def ucb_value(self, ucb_const):
    if self.visits != 0:
      average_reward = self.get_average_reward(player=self.parent.game_state.current_player)
      return average_reward + ucb_const * np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    else:
      return np.infty

  def ucb_values(self, ucb_const):
    return [child.ucb_value(ucb_const) for child in self.children]

  def update_visits(self):
    self.visits += 1

  def update_rewards(self, rewards):
    for i in range(len(self.cumulative_rewards)):
      self.cumulative_rewards[i] += rewards[i]
