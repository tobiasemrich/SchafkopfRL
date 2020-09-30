import time
from os import listdir

import numpy as np
import torch

from game_statistics import GameStatistics
from memory import Memory
from models.actor_critic_linear_contra import ActorCriticNetworkLinearContra
from schafkopf_env import SchafkopfEnv
from schafkopfrl.players.random_coward_player import RandomCowardPlayer
from schafkopfrl.players.random_player import RandomPlayer
from schafkopfrl.players.rl_player import RlPlayer
from schafkopfrl.players.rule_based_player import RuleBasedPlayer
from schafkopfrl.ppo import PPO
from schafkopfrl.schafkopf_game import SchafkopfGame

from tensorboard import program

from settings import Settings


def main():

  print("Cuda available: "+str(torch.cuda.is_available()))

  #start tensorboard
  tb = program.TensorBoard()
  tb.configure(argv=[None, '--logdir', Settings.runs_folder])
  tb.launch()

  # set seed for debugging
  if Settings.random_seed:
      torch.manual_seed(Settings.random_seed)

  #loading initial policy
  policy = Settings.model().to(Settings.device)
  # take the newest generation available
  i_episode = max_gen = 0
  generations = [int(f[:8]) for f in listdir(Settings.checkpoint_folder) if f.endswith(".pt")]
  if len(generations) > 0:
      max_gen = max(generations)
      policy.load_state_dict(torch.load(Settings.checkpoint_folder+"/" + str(max_gen).zfill(8) + ".pt"))
      i_episode = max_gen
  #create ppo
  ppo = PPO(policy, [Settings.lr, Settings.lr_stepsize, Settings.lr_gamma], Settings.betas, Settings.gamma, Settings.K_epochs, Settings.eps_clip, Settings.batch_size,Settings.mini_batch_size, c1=Settings.c1, c2=Settings.c2, start_episode=max_gen-1  )

  #create four players
  players = [RlPlayer(ppo.policy_old), RlPlayer(ppo.policy_old), RlPlayer(ppo.policy_old), RlPlayer(ppo.policy_old)]
  #create a game simulation
  schafkopf_env = SchafkopfEnv(Settings.random_seed)
  game_statistics = GameStatistics()

  # training loop
  for _ in range(0, 90000000):
    Settings.logger.info("playing " +str(Settings.update_games)+ " games")

    # play a bunch of games
    t0 = time.time()
    for _ in range(Settings.update_games):
      state, reward, terminal = schafkopf_env.reset()
      while not terminal:
        action, prob = players[state["game_state"].current_player].act(state)
        state, reward, terminal = schafkopf_env.step(action, prob)
      for p in range(4):
        players[p].retrieve_reward(reward[p])
      i_episode += 1
      game_statistics.update_statistics(state["game_state"], reward)
    t1 = time.time()

    #update the policy
    Settings.logger.info("updating policy")

    player_memories = Memory()
    for p in players:
      player_memories.append_memory(p.memory)

    ppo.update(player_memories, i_episode)
    t2 = time.time()
    ppo.lr_scheduler.step(i_episode)

    # writing game statistics for tensorboard
    Settings.logger.info("Episode: "+str(i_episode) + " game simulation (s) = "+str(t1-t0) + " update (s) = "+str(t2-t1))
    schafkopf_env.print_game()
    game_statistics.write_and_reset (i_episode)

    # reset memories and replace policy
    players = [RlPlayer(ppo.policy_old), RlPlayer(ppo.policy_old), RlPlayer(ppo.policy_old), RlPlayer(ppo.policy_old)]

    # save and evaluate the policy
    Settings.logger.info("Saving Checkpoint")
    torch.save(ppo.policy_old.state_dict(), Settings.checkpoint_folder + "/" + str(i_episode).zfill(8) + ".pt")
    Settings.logger.info("Evaluation")
    play_against_other_players(Settings.checkpoint_folder, Settings.model, [RandomPlayer, RandomCowardPlayer, RuleBasedPlayer], Settings.eval_games,
                               Settings.summary_writer)


def play_against_other_players(checkpoint_folder, model_class, other_player_classes, runs, summary_writer):

  generations = [int(f[:8]) for f in listdir(checkpoint_folder) if f.endswith(".pt")]
  max_gen = max(generations)
  policy = model_class()
  policy.to(device=Settings.device)
  policy.load_state_dict(torch.load(checkpoint_folder + "/" + str(max_gen).zfill(8) + ".pt"))

  for other_player_class in other_player_classes:

    players = [other_player_class(), RlPlayer(policy), other_player_class(), RlPlayer(policy)]
    schafkopf_env = SchafkopfEnv(1)

    all_rewards = np.array([0., 0., 0., 0.])
    for j in range(runs):
      state, reward, terminal = schafkopf_env.reset()
      while not terminal:
        action, prob = players[state["game_state"].current_player].act(state)
        state, reward, terminal = schafkopf_env.step(action, prob)

      all_rewards += reward

    all_rewards = all_rewards[[1, 0, 3, 2]]

    players = [RlPlayer(policy), other_player_class(), RlPlayer(policy), other_player_class()]
    schafkopf_env = SchafkopfEnv(1)

    for j in range(runs):
      state, reward, terminal = schafkopf_env.reset()
      while not terminal:
        action, prob = players[state["game_state"].current_player].act(state)
        state, reward, terminal = schafkopf_env.step(action, prob)

      all_rewards += reward

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