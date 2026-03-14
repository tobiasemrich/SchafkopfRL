import math

_INF = float('inf')

class Node:
  __slots__ = ('parent', 'previous_action', 'children', 'cumulative_rewards',
               'visits', 'game_state', 'player_hands', 'allowed_actions')

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
    return self.game_state.trick_number == 8

  def get_average_reward(self, player):
    if self.visits > 0:
      return self.cumulative_rewards[player] / self.visits
    else:
      return 0

  def is_leaf(self):
    return len(self.children) == 0

  def fully_expanded(self):
    return len(self.children) == len(self.allowed_actions)

  def best_child(self, ucb_const):
    if self.children:
      log_2_parent = 2.0 * math.log(self.visits)
      player = self.game_state.current_player
      best = None
      best_val = float('-inf')
      for child in self.children:
        if child.visits == 0:
          return child
        val = child.cumulative_rewards[player] / child.visits + ucb_const * math.sqrt(log_2_parent / child.visits)
        if val > best_val:
          best_val = val
          best = child
      return best

  def update_visits(self):
    self.visits += 1

  def update_rewards(self, rewards):
    cr = self.cumulative_rewards
    cr[0] += rewards[0]
    cr[1] += rewards[1]
    cr[2] += rewards[2]
    cr[3] += rewards[3]
