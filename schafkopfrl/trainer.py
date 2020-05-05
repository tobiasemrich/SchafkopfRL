import time
from os import listdir, path

import numpy as np
import torch
from schafkopfrl.models.actor_critic_linear import ActorCriticNetworkLinear
from schafkopfrl.models.actor_critic_lstm import ActorCriticNetworkLSTM
from schafkopfrl.models.actor_critic_lstm_contra import ActorCriticNetworkLSTMContra
from schafkopfrl.players.random_coward_player import RandomCowardPlayer
from schafkopfrl.players.random_player import RandomPlayer
from schafkopfrl.players.rl_player import RlPlayer
from schafkopfrl.players.rule_based_player import RuleBasedPlayer
from schafkopfrl.ppo import PPO
from schafkopfrl.schafkopf_game import SchafkopfGame

from tensorboard import program

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

  ############## Hyperparameters ##############

  update_games = 50000  # update policy every n games
  batch_size = update_games * 6
  mini_batch_size = 20000 # make this as large as possible to fit in gpu

  eval_games = 500
  checkpoint_folder = "../policies"

  #lr = 0.0002
  lr = 0.001
  lr_stepsize = 30000000 #300000
  lr_gamma = 0.3

  betas = (0.9, 0.999)
  gamma = 0.99  # discount factor
  K_epochs = 16 #8  # update policy for K epochs
  eps_clip = 0.2  # clip parameter for PPO
  c1, c2 = 0.5, 0.005#0.001
  random_seed = None #<---------------------------------------------------------------- set to None
  #############################################

  print("Cuda available: "+str(torch.cuda.is_available()))

  #model = ActorCriticNetworkLinear
  model = ActorCriticNetworkLSTMContra

  #start tensorboard
  tb = program.TensorBoard()
  runs_folder = path.abspath(path.join(path.dirname(__file__), '..', 'runs'))
  tb.configure(argv=[None, '--logdir', runs_folder])
  tb.launch()

  # creating environment
  if random_seed:
      torch.manual_seed(random_seed)

  #loading initial policy
  policy = model().to(device)
  # take the newest generation available
  # file pattern = policy-000001.pt
  max_gen = 0
  generations = [int(f[:8]) for f in listdir(checkpoint_folder) if f.endswith(".pt")]
  if len(generations) > 0:
      max_gen = max(generations)
      policy.load_state_dict(torch.load(checkpoint_folder+"/" + str(max_gen).zfill(8) + ".pt"))

  #create ppo
  ppo = PPO(policy, [lr, lr_stepsize, lr_gamma], betas, gamma, K_epochs, eps_clip, batch_size,mini_batch_size, c1=c1, c2=c2, start_episode=max_gen-1  )

  #create a game simulation

  gs = SchafkopfGame(RlPlayer(0, ppo.policy_old), RlPlayer(1, ppo.policy_old), RlPlayer(2, ppo.policy_old), RlPlayer(3, ppo.policy_old), random_seed)

  # training loop
  i_episode = max_gen
  for _ in range(0, 90000000):
    ppo.logger.info("playing " +str(update_games)+ " games")

    # play a bunch of games
    t0 = time.time()
    for _ in range(update_games):
        game_state = gs.play_one_game()
        i_episode += 1
    t1 = time.time()

    #update the policy
    ppo.logger.info("updating policy")
    ppo.update(gs.get_player_memories(), i_episode)
    t2 = time.time()
    ppo.lr_scheduler.step(i_episode)

    # logging
    ppo.logger.info("Episode: "+str(i_episode) + " game simulation (s) = "+str(t1-t0) + " update (s) = "+str(t2-t1))
    gs.print_game(game_state) #<------------------------------------remove
    ppo.writer.add_scalar('Game_Statistics/fraction_weiter', gs.game_count[0]/update_games, i_episode)
    ppo.writer.add_scalar('Game_Statistics/fraction_sauspiel', gs.game_count[1] / update_games, i_episode)
    ppo.writer.add_scalar('Game_Statistics/fraction_wenz', gs.game_count[2] / update_games, i_episode)
    ppo.writer.add_scalar('Game_Statistics/fraction_solo', gs.game_count[3] / update_games, i_episode)

    ppo.writer.add_scalar('Game_Statistics/winning_prob_sauspiel', np.divide(gs.won_game_count[1], gs.game_count[1]), i_episode)
    ppo.writer.add_scalar('Game_Statistics/winning_prob_wenz', np.divide(gs.won_game_count[2],gs.game_count[2]), i_episode)
    ppo.writer.add_scalar('Game_Statistics/winning_prob_solo', np.divide(gs.won_game_count[3],gs.game_count[3]), i_episode)

    ppo.writer.add_scalar('Game_Statistics/contra_prob', np.divide(gs.contra_retour[0], update_games),
                          i_episode)

    # reset memories and replace policy
    gs = SchafkopfGame(RlPlayer(0, ppo.policy_old), RlPlayer(1, ppo.policy_old), RlPlayer(2, ppo.policy_old), RlPlayer(3, ppo.policy_old), random_seed)

    # save the policy
    ppo.logger.info("Saving Checkpoint")
    torch.save(ppo.policy_old.state_dict(), checkpoint_folder + "/" + str(i_episode).zfill(8) + ".pt")

    ppo.logger.info("Evaluation")
    play_against_other_players(checkpoint_folder, model, [RandomPlayer, RandomCowardPlayer, RuleBasedPlayer], eval_games,
                               ppo.writer)

def play_against_other_players(checkpoint_folder, model_class, other_player_classes, runs, summary_writer):

  generations = [int(f[:8]) for f in listdir(checkpoint_folder) if f.endswith(".pt")]
  max_gen = max(generations)
  policy = model_class()
  policy.to(device=device)
  policy.load_state_dict(torch.load(checkpoint_folder + "/" + str(max_gen).zfill(8) + ".pt"))

  for other_player_class in other_player_classes:

    gs = SchafkopfGame(other_player_class(0), RlPlayer(1, policy), other_player_class(2),
                       RlPlayer(3, policy), 1)
    all_rewards = np.array([0., 0., 0., 0.])
    for j in range(runs):
      game_state = gs.play_one_game()
      rewards = np.array(game_state.get_rewards())
      all_rewards += rewards

    gs = SchafkopfGame(RlPlayer(0, policy), other_player_class(1), RlPlayer(2, policy),
                       other_player_class(3), 1)
    all_rewards = all_rewards[[1, 0, 3, 2]]
    for j in range(runs):
      game_state = gs.play_one_game()
      # gs.print_game(game_state)
      rewards = np.array(game_state.get_rewards())
      all_rewards += rewards

    summary_writer.add_scalar('Evaluation/' + str(other_player_class.__name__),
                              (all_rewards[0] + all_rewards[2]) / (4 * runs), max_gen)

'''
def play_against_old_checkpoints(checkpoint_folder, model_class, every_n_checkpoint, runs, summary_writer):
  generations = [int(f[:8]) for f in listdir(checkpoint_folder) if f.endswith(".pt")]
  generations.sort()
  if len(generations) > 1:
    max_gen = max(generations)
    for i in generations:
      if i != max_gen and i % every_n_checkpoint == 0:
        policy_old = model_class()
        policy_old.to(device=device)
        policy_old.load_state_dict(torch.load(checkpoint_folder + "/" + str(i).zfill(8) + ".pt"))

        policy_new = model_class()
        policy_new.to(device=device)
        policy_new.load_state_dict(torch.load(checkpoint_folder + "/" + str(max_gen).zfill(8) + ".pt"))

        gs = SchafkopfGame(RlPlayer(0, policy_old), RlPlayer(1, policy_new), RlPlayer(2, policy_old),
                           RlPlayer(3, policy_new), 1)
        all_rewards = np.array([0., 0., 0., 0.])
        for j in range(runs):
          game_state = gs.play_one_game()
          rewards = np.array(game_state.get_rewards())
          all_rewards += rewards

        gs = SchafkopfGame(RlPlayer(0, policy_new), RlPlayer(1, policy_old), RlPlayer(2, policy_new),
                           RlPlayer(3, policy_old), 1)
        all_rewards = all_rewards[[1, 0, 3, 2]]
        for j in range(runs):
          game_state = gs.play_one_game()
          # gs.print_game(game_state)
          rewards = np.array(game_state.get_rewards())
          all_rewards += rewards

        print(str(max_gen) + " vs " + str(i) + " = " + str(all_rewards[0] + all_rewards[2]) + ":" + str(
          all_rewards[1] + all_rewards[3]) + "\n")
        summary_writer.add_scalar('Evaluation/generation_' + str(max_gen),
                                  (all_rewards[0] + all_rewards[2]) / (4 * runs), i)
'''

if __name__ == '__main__':
  main()