import time
from os import listdir

import numpy as np
import torch
from torch import nn
from torch.utils import data

from dataset import HandPredictionDatasetLSTM
from game_statistics import GameStatistics
from models.hand_predictor import HandPredictor
from players.smart_mcts_player import SmartMCTSPlayer
from schafkopf_env import SchafkopfEnv
from players.random_player import RandomPlayer
from players.rl_player import RlPlayer

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
  hand_predictor = HandPredictor().to(Settings.device)
  # take the newest generation available
  i_episode = max_gen = 0
  generations = [int(f[:8]) for f in listdir(Settings.checkpoint_folder) if f.endswith(".pt")]
  if len(generations) > 0:
      max_gen = max(generations)
      hand_predictor.load_state_dict(torch.load(Settings.checkpoint_folder+"/" + str(max_gen).zfill(8) + ".pt"))
      i_episode = max_gen

  optimizer = torch.optim.Adam(hand_predictor.parameters(),lr=Settings.lr, betas=Settings.betas, weight_decay=Settings.optimizer_weight_decay)

  # training loop
  for _ in range(0, 90000000):
    Settings.logger.info("playing " +str(Settings.update_games)+ " games")

    smart_mcts_player = SmartMCTSPlayer(10, 40, RandomPlayer(), hand_predictor)
    # create four players
    players = [smart_mcts_player, smart_mcts_player, smart_mcts_player, smart_mcts_player]
    # create a game simulation
    schafkopf_env = SchafkopfEnv(Settings.random_seed)
    game_statistics = GameStatistics()


    memory_states = []
    memory_player_hands = []

    # play a bunch of games
    t0 = time.time()
    for _ in range(Settings.update_games):
      state, reward, terminal = schafkopf_env.reset()

      while not terminal:
        memory_states.append(hand_predictor.preprocess(state)) #TODO: happens twice now and could be optimized
        memory_player_hands.append(hand_predictor.encode_player_hands(schafkopf_env.player_cards, state["game_state"].current_player))

        action, prob = players[state["game_state"].current_player].act(state)
        state, reward, terminal = schafkopf_env.step(action, prob)

      print("game "+str(i_episode))
      i_episode += 1
      game_statistics.update_statistics(state["game_state"], reward)
      #return None
    t1 = time.time()

    #update the policy
    Settings.logger.info("updating policy")
    # Create dataset from collected experiences
    dataset = HandPredictionDatasetLSTM(memory_states, memory_player_hands)
    training_generator = data.DataLoader(dataset, collate_fn=dataset.custom_collate,batch_size=Settings.mini_batch_size, shuffle=True)

    #logging
    avg_loss = 0
    count = 0

    hand_predictor.train()
    for epoch in range(Settings.K_epochs):  # epoch

      mini_batches_in_batch = int(Settings.batch_size / Settings.mini_batch_size)
      optimizer.zero_grad()

      for i, (states, hands) in enumerate(training_generator):  # mini batch
        # Transfer to GPU
        states = [state.to(Settings.device) for state in states]
        hands = hands.to(Settings.device)
        pred = hand_predictor(states)
        #loss = nn.MSELoss()(pred, hands) #TODO: replace by cross entropy
        loss = nn.BCELoss()(pred, hands)

        avg_loss += loss.mean().item()
        count +=1

        loss.mean().backward()

        if (i + 1) % mini_batches_in_batch == 0:
          optimizer.step()
          optimizer.zero_grad()
    t2 = time.time()
    hand_predictor.eval()

    # writing game statistics for tensorboard
    Settings.logger.info("Episode: "+str(i_episode) + " game simulation (s) = "+str(t1-t0) + " update (s) = "+str(t2-t1))
    schafkopf_env.print_game()
    game_statistics.write_and_reset (i_episode)
    Settings.summary_writer.add_scalar('Loss/MSE_Loss', avg_loss / count, i_episode)

    # save and evaluate the policy
    Settings.logger.info("Saving Checkpoint")
    torch.save(hand_predictor.state_dict(), Settings.checkpoint_folder + "/" + str(i_episode).zfill(8) + ".pt")
    Settings.logger.info("Evaluation")
    #play_against_other_players(Settings.checkpoint_folder, Settings.model, [RandomPlayer, RandomCowardPlayer, RuleBasedPlayer], Settings.eval_games,
    #                           Settings.summary_writer)


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


if __name__ == '__main__':
  import cProfile

  pr = cProfile.Profile()
  pr.enable()
  main()
  pr.disable()
  # after your program ends
  pr.print_stats(sort="cumtime")
