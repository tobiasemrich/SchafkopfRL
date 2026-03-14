import random

from policy.mcts.node import Node
from environment.schafkopf_env import SchafkopfEnv
from copy import deepcopy


class MonteCarloTree:
  '''
  Inspired by https://github.com/Taschee/schafkopf/blob/master/schafkopf/players/uct_player.py
  '''
  def __init__(self, game_state, player_hands, allowed_actions, ucb_const=1):
    self.root = Node(None, None, game_state, player_hands, allowed_actions)
    self.ucb_const = ucb_const

  def uct_search(self, num_playouts):
    for _ in range(num_playouts):
      selected_node = self.selection()
      rewards = self.simulation(selected_node)
      self.backup_rewards(leaf_node=selected_node, rewards=rewards)

    results = []
    for child in self.root.children:
      results.append((child.previous_action, child.visits, child.get_average_reward(self.root.game_state.current_player)))

    return results

  def selection(self):
    current_node = self.root
    while not current_node.is_terminal():
      if not current_node.fully_expanded():
        return self.expand(current_node)
      else:
        current_node = current_node.best_child(ucb_const=self.ucb_const)
    return current_node

  def expand(self, node):
    visited = {child.previous_action for child in node.children}
    not_visited_actions = [a for a in node.allowed_actions if a not in visited]

    #TODO: check if this should be random or chosen by player policy
    chosen_action = random.choice(not_visited_actions)

    schafkopf_env = SchafkopfEnv()
    schafkopf_env.set_state(deepcopy(node.game_state), [node.player_hands[i][:] for i in range(4)])
    state, _, terminal = schafkopf_env.step(chosen_action)

    new_node = Node(parent=node, game_state=state["game_state"], previous_action=chosen_action, player_hands=schafkopf_env.player_cards, allowed_actions=state["allowed_actions"])
    node.add_child(child_node=new_node)
    return new_node

  def simulation(self, selected_node):

    schafkopf_env = SchafkopfEnv()

    #state, reward, terminal = schafkopf_env.set_state(deepcopy(selected_node.game_state), deepcopy(selected_node.player_hands))
    state, reward, terminal = schafkopf_env.set_state(deepcopy(selected_node.game_state),
                                                      [selected_node.player_hands[i][:] for i in range(4)])
    while not terminal:
      # choose action at random
      allowed_actions = state["allowed_actions"]
      selected_action = random.choice(allowed_actions)
      state, reward, terminal = schafkopf_env.step(selected_action)

    return reward

  def backup_rewards(self, leaf_node, rewards):
    current_node = leaf_node
    while current_node != self.root:
      current_node.update_rewards(rewards)
      current_node.update_visits()
      current_node = current_node.parent
    self.root.update_visits()

  def get_action_count_rewards(self):
    result = {}
    for child in self.root.children:
      result[child.previous_action] = (child.visits, child.cumulative_rewards)
    return result