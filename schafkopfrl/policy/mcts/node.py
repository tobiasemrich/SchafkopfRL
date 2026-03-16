import math
from typing import Any, Optional

_INF: float = float('inf')

class Node:
  """A node in the Monte Carlo search tree.

  Parameters
  ----------
  parent : Node or None
      Parent node (None for root).
  previous_action : Any
      Action that led to this node.
  game_state : Any
      The public game state at this node.
  player_hands : list[list]
      Card hands for all four players.
  allowed_actions : list
      Actions available from this node.
  """
  __slots__ = ('parent', 'previous_action', 'children', 'cumulative_rewards',
               'visits', 'game_state', 'player_hands', 'allowed_actions')

  def __init__(self, parent: "Optional[Node]", previous_action: Any, game_state: Any, player_hands: list[list], allowed_actions: list) -> None:
    self.parent: "Optional[Node]" = parent
    self.previous_action: Any = previous_action
    self.children: "list[Node]" = []
    self.cumulative_rewards: list[float] = [0, 0, 0, 0]
    self.visits: int = 0

    self.game_state: Any = game_state
    self.player_hands: list[list] = player_hands
    self.allowed_actions: list = allowed_actions

  def add_child(self, child_node: "Node") -> None:
    self.children.append(child_node)

  def is_terminal(self) -> bool:
    return self.game_state.trick_number == 8

  def get_average_reward(self, player: int) -> float:
    if self.visits > 0:
      return self.cumulative_rewards[player] / self.visits
    else:
      return 0

  def is_leaf(self) -> bool:
    return len(self.children) == 0

  def fully_expanded(self) -> bool:
    return len(self.children) == len(self.allowed_actions)

  def best_child(self, ucb_const: float) -> "Optional[Node]":
    """Select the child with the highest UCB1 value.

    Parameters
    ----------
    ucb_const : float
        Exploration constant for the UCB1 formula.

    Returns
    -------
    Node or None
        The best child node, or None if there are no children.
    """
    if self.children:
      log_2_parent: float = 2.0 * math.log(self.visits)
      player: int = self.game_state.current_player
      best: "Optional[Node]" = None
      best_val: float = float('-inf')
      for child in self.children:
        if child.visits == 0:
          return child
        val: float = child.cumulative_rewards[player] / child.visits + ucb_const * math.sqrt(log_2_parent / child.visits)
        if val > best_val:
          best_val = val
          best = child
      return best

  def update_visits(self) -> None:
    self.visits += 1

  def update_rewards(self, rewards: list[float]) -> None:
    cr: list[float] = self.cumulative_rewards
    cr[0] += rewards[0]
    cr[1] += rewards[1]
    cr[2] += rewards[2]
    cr[3] += rewards[3]
