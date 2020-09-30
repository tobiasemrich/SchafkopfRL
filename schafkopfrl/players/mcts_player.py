import utils
from players.mcts.mct import MonteCarloTree
from schafkopfrl.players.player import Player


class MCTSPlayer(Player):

  def __init__(self, samples, playouts):
    super().__init__()
    self.samples = samples
    self.playouts = playouts


  def act(self, state):
    return self.run_mcts(state("game_state"), state("current_player_cards"), state("allowed_actions")), 1


  def run_mcts(self, game_state, player_cards, allowed_actions):

    cummulative_action_count_rewards = {}

    for i in range (self.samples):
      sampled_player_hands = utils.sample_player_hands(game_state, player_cards)
      mct = MonteCarloTree(game_state,sampled_player_hands)
      mct.uct_search(self.playouts)
      action_count_rewards = mct.get_action_count_rewards()

      for action in action_count_rewards:
        if action in cummulative_action_count_rewards:
          cummulative_action_count_rewards[action] = (cummulative_action_count_rewards[action][0] + action_count_rewards[action][0],
                                                      [cummulative_action_count_rewards[action][1][i] + action_count_rewards[action][1][i] for i in range(4)])
        else:
          cummulative_action_count_rewards[action] = action_count_rewards[action]

    return max(cummulative_action_count_rewards, key = lambda x: x[0])