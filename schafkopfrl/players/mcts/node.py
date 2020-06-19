class Node:

  def __init__(self, parent, action, game_state, player_hands):
    self.parent = parent
    self.action = action
    self.children = []
    self.rewards = [0, 0, 0, 0]
    self.visits = 0

    self.game_state = game_state
    self.player_hands = player_hands

  def add_child(self, child_node):
    self.children.append(child_node)

  def is_terminal(self):
    if len(self.game_state.trick_number) == 7:
      return True
    else:
      return False