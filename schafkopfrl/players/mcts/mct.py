from players.mcts.node import Node


class MonteCarloTree:

  def __init__(self, game_state, player_hands):
    self.root = Node(None, None, game_state, player_hands)

  def uct_search(self, num_playouts):
    for _ in range(num_playouts):
      selected_node = self.selection()
      rewards = self.simulation(selected_node)
      self.backup_rewards(leaf_node=selected_node, rewards=rewards)

  def selection(self):
    current_node = self.root
    while not current_node.is_terminal():
      if not current_node.fully_expanded():
        return self.expand(mc_tree=mc_tree, node=current_node)
      else:
        current_node = current_node.best_child(ucb_const=self.ucb_const)
    return current_node

  def backup_rewards(self):
    pass

  def get_action_count_rewards(self):
    result = {}
    for child in self.root.children:
      result[child.action] = (child.visits, child.rewards)
    return result