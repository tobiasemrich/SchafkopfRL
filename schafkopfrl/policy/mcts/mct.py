import random
from typing import Any

from schafkopfrl.policy.mcts.node import Node
from schafkopfrl.environment.schafkopf_env import SchafkopfEnv
from copy import deepcopy


class MonteCarloTree:
  """Upper Confidence Bound for Trees (UCT) search for Schafkopf.

  Inspired by `Taschee/schafkopf <https://github.com/Taschee/schafkopf>`_.

  Parameters
  ----------
  game_state : Any
      The current public game state.
  player_hands : list[list]
      Card hands for all four players.
  allowed_actions : list
      Actions available at the root node.
  ucb_const : float, optional
      Exploration constant for UCB1, by default 1.
  """
  def __init__(self, game_state: Any, player_hands: list[list], allowed_actions: list, ucb_const: float = 1) -> None:
    self.root: Node = Node(None, None, game_state, player_hands, allowed_actions)
    self.ucb_const: float = ucb_const

  def uct_search(self, num_playouts: int) -> list[tuple[Any, int, float]]:
    """Run UCT search for a given number of playouts.

    Parameters
    ----------
    num_playouts : int
        Number of simulation playouts to perform.

    Returns
    -------
    list[tuple[Any, int, float]]
        List of ``(action, visits, average_reward)`` for each root child.
    """
    for _ in range(num_playouts):
      selected_node = self.selection()
      rewards = self.simulation(selected_node)
      self.backup_rewards(leaf_node=selected_node, rewards=rewards)

    results: list[tuple[Any, int, float]] = []
    for child in self.root.children:
      results.append((child.previous_action, child.visits, child.get_average_reward(self.root.game_state.current_player)))

    return results

  def selection(self) -> Node:
    """Select a leaf node by traversing the tree using UCB1.

    Returns
    -------
    Node
        The selected leaf node to expand or simulate from.
    """
    current_node: Node = self.root
    while not current_node.is_terminal():
      if not current_node.fully_expanded():
        return self.expand(current_node)
      else:
        current_node = current_node.best_child(ucb_const=self.ucb_const)
    return current_node

  def expand(self, node: Node) -> Node:
    """Expand a node by adding one unvisited child.

    Parameters
    ----------
    node : Node
        The node to expand.

    Returns
    -------
    Node
        The newly created child node.
    """
    visited: set = {child.previous_action for child in node.children}
    not_visited_actions: list = [a for a in node.allowed_actions if a not in visited]

    #TODO: check if this should be random or chosen by player policy
    chosen_action: Any = random.choice(not_visited_actions)

    schafkopf_env: SchafkopfEnv = SchafkopfEnv()
    schafkopf_env.set_state(deepcopy(node.game_state), [node.player_hands[i][:] for i in range(4)])
    state, _, terminal = schafkopf_env.step(chosen_action)

    new_node: Node = Node(parent=node, game_state=state["game_state"], previous_action=chosen_action, player_hands=schafkopf_env.player_cards, allowed_actions=state["allowed_actions"])
    node.add_child(child_node=new_node)
    return new_node

  def simulation(self, selected_node: Node) -> list[float]:
    """Simulate a random playout from the given node to a terminal state.

    Parameters
    ----------
    selected_node : Node
        The node to start the simulation from.

    Returns
    -------
    list[float]
        Per-player rewards from the completed game.
    """

    schafkopf_env: SchafkopfEnv = SchafkopfEnv()

    #state, reward, terminal = schafkopf_env.set_state(deepcopy(selected_node.game_state), deepcopy(selected_node.player_hands))
    state, reward, terminal = schafkopf_env.set_state(deepcopy(selected_node.game_state),
                                                      [selected_node.player_hands[i][:] for i in range(4)])
    while not terminal:
      # choose action at random
      allowed_actions: list = state["allowed_actions"]
      selected_action: Any = random.choice(allowed_actions)
      state, reward, terminal = schafkopf_env.step(selected_action)

    return reward

  def backup_rewards(self, leaf_node: Node, rewards: list[float]) -> None:
    """Back-propagate rewards from a leaf node up to the root.

    Parameters
    ----------
    leaf_node : Node
        The leaf node where simulation ended.
    rewards : list[float]
        Per-player rewards to propagate.
    """
    current_node: Node = leaf_node
    while current_node != self.root:
      current_node.update_rewards(rewards)
      current_node.update_visits()
      current_node = current_node.parent
    self.root.update_visits()

  def get_action_count_rewards(self) -> dict[Any, tuple[int, list[float]]]:
    """Return visit counts and cumulative rewards for each root child action.

    Returns
    -------
    dict[Any, tuple[int, list[float]]]
        Mapping from action to ``(visits, cumulative_rewards_per_player)``.
    """
    result: dict[Any, tuple[int, list[float]]] = {}
    for child in self.root.children:
      result[child.previous_action] = (child.visits, child.cumulative_rewards)
    return result