import logging
from sys import path

import torch

from torch.utils.tensorboard import SummaryWriter
import experience_dataset_lstm
import experience_dataset_linear
from schafkopfrl.experience_dataset_linear import ExperienceDatasetLinear
from schafkopfrl.experience_dataset_lstm import ExperienceDatasetLSTM
from schafkopfrl.models.actor_critic_linear_contra import ActorCriticNetworkLinearContra
from schafkopfrl.models.actor_critic_lstm_contra import ActorCriticNetworkLSTMContra


class Settings:

  #global logger
  logger = logging.getLogger(__name__)
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  logger.setLevel(logging.INFO)

  #tensorboard writer
  runs_folder = path.abspath(path.join(path.dirname(__file__), '..', 'runs'))
  summary_writer = SummaryWriter(log_dir=runs_folder)

  #device used for pytorch
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # what model to use
  model, dataset, collate_fun = ActorCriticNetworkLinearContra,ExperienceDatasetLinear, experience_dataset_linear.custom_collate
  #model, dataset = ActorCriticNetworkLSTMContra,ExperienceDatasetLSTM, experience_dataset_lstm.custom_collate

  ############################# Hyperparameters #############################################
  update_games = 1000  # update policy every n games
  batch_size = update_games * 6
  mini_batch_size = 1000  # make this as large as possible to fit in gpu

  eval_games = 500
  checkpoint_folder = "../policies"

  # lr = 0.0002
  lr = 0.001
  lr_stepsize = 30000000  # 300000
  lr_gamma = 0.3

  betas = (0.9, 0.999)
  gamma = 0.99  # discount factor
  K_epochs = 16  # 8  # update policy for K epochs
  eps_clip = 0.2  # clip parameter for PPO
  c1, c2 = 0.5, 0.005  # 0.001

  optimizer_weight_decay = 5e-5

  random_seed = None