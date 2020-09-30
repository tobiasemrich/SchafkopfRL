from settings import Settings
import numpy as np

class GameStatistics:
  def __init__(self):
    self.reset()

  def reset(self):
    self.game_count = [0, 0, 0, 0]  # weiter, sauspiel, wenz, solo
    self.won_game_count = [0, 0, 0, 0]
    self.contra_retour = [0, 0]

  def update_statistics(self, game_state, rewards):
      if game_state.game_type[1] == None:
          self.game_count[0] += 1
      else:
          self.game_count[game_state.game_type[1] + 1] += 1
          if rewards[game_state.game_player] > 0:
              self.won_game_count[game_state.game_type[1] + 1] += 1
          if np.any(game_state.contra):
              self.contra_retour[0] += 1
          if np.any(game_state.retour):
              self.contra_retour[1] += 1

  def write_and_reset(self, i_episode):
    Settings.summary_writer.add_scalar('Game_Statistics/fraction_weiter', self.game_count[0] / Settings.update_games, i_episode)
    Settings.summary_writer.add_scalar('Game_Statistics/fraction_sauspiel',
                                       self.game_count[1] / Settings.update_games, i_episode)
    Settings.summary_writer.add_scalar('Game_Statistics/fraction_wenz',
                                       self.game_count[2] / Settings.update_games, i_episode)
    Settings.summary_writer.add_scalar('Game_Statistics/fraction_solo',
                                       self.game_count[3] / Settings.update_games, i_episode)

    Settings.summary_writer.add_scalar('Game_Statistics/winning_prob_sauspiel',
                                       np.divide(self.won_game_count[1], self.game_count[1]),
                                       i_episode)
    Settings.summary_writer.add_scalar('Game_Statistics/winning_prob_wenz',
                                       np.divide(self.won_game_count[2], self.game_count[2]),
                                       i_episode)
    Settings.summary_writer.add_scalar('Game_Statistics/winning_prob_solo',
                                       np.divide(self.won_game_count[3], self.game_count[3]),
                                       i_episode)

    Settings.summary_writer.add_scalar('Game_Statistics/contra_prob',
                                       np.divide(self.contra_retour[0], Settings.update_games), i_episode)
    self.reset()