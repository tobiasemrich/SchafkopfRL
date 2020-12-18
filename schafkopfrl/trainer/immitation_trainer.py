import pickle
from os import listdir

import torch
from sqlitedict import SqliteDict

from dataset import PredictionDatasetLSTM
from models.actor_critic_lstm import ActorCriticNetworkLSTM
from models.immitation_policy import ImmitationPolicy
from public_gamestate import PublicGameState
from rules import Rules
from schafkopf_env import SchafkopfEnv
from settings import Settings
from tensorboard import program
import numpy as np
from torch.utils import data
from torch import nn

from utils import two_hot_encode_card, one_hot_cards, two_hot_encode_game, one_hot_games

rules = Rules()

def main():

  # loading initial policy
  immitation_policy = ImmitationPolicy().to(Settings.device)

  states = []
  actions = []
  #load and preprocess database
  games = SqliteDict('../sauspiel/games.sqlite')
  count = 0
  for g in games:
    game = games[g]
    if len(game.sonderregeln) == 0:
      count += 1
      game_states, game_actions = get_states_actions(game, immitation_policy)
      states += game_states
      actions += game_actions

      if count % 1000 == 0:
        print("Read " +str(count) + " normal games")

  '''
  with open('dataset.pkl', 'wb') as output:
    pickle.dump(states, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(actions, output, pickle.HIGHEST_PROTOCOL)
  
  with open('dataset.pkl', 'rb') as input:
    states = pickle.load(input)
    actions = pickle.load(input)
  '''
  dataset = PredictionDatasetLSTM(states, actions, 2)

  #split into train/test
  train_size = int(0.9 * len(dataset))
  test_size = len(dataset) - train_size

  torch.manual_seed(0)
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


  training_generator = data.DataLoader(train_dataset, collate_fn=dataset.custom_collate,batch_size=10000, shuffle=True)
  testing_generator = data.DataLoader(test_dataset, collate_fn=dataset.custom_collate, batch_size=10000)

  # start tensorboard
  tb = program.TensorBoard()
  tb.configure(argv=[None, '--logdir', Settings.runs_folder])
  tb.launch()


  # take the newest generation available
  count = max_gen = 0
  generations = [int(f[:8]) for f in listdir(Settings.checkpoint_folder) if f.endswith(".pt")]
  if len(generations) > 0:
    max_gen = max(generations)
    immitation_policy.load_state_dict(torch.load(Settings.checkpoint_folder + "/" + str(max_gen).zfill(8) + ".pt"))
    count = max_gen

  optimizer = torch.optim.Adam(immitation_policy.parameters(), lr=Settings.lr, betas=Settings.betas,
                               weight_decay=Settings.optimizer_weight_decay)

  # training loop
  immitation_policy.train()


  for epoch in range(100000):  # epoch
    # testing
    if epoch % 1 == 0:
      correct = 0
      total = 0
      with torch.no_grad():
        for i, (states, actions) in enumerate(testing_generator):
          pred = immitation_policy(states)
          _, predicted = torch.max(pred.data, 1)
          total += actions.size(0)
          correct += (predicted.cpu() == actions.cpu()).sum().item()
      Settings.summary_writer.add_scalar('Testing/Accuracy', correct / total, count)

    #training
    for i, (states, actions) in enumerate(training_generator):  #batch

      count += 1

      # Transfer to GPU
      states = [state.to(Settings.device) for state in states]
      actions = actions.to(Settings.device)

      optimizer.zero_grad()

      pred = immitation_policy(states)
      #loss = nn.MSELoss()(pred, actions) #TODO: replace by cross entropy
      loss = nn.CrossEntropyLoss()(pred, actions)
      #loss = nn.NLLLoss()(pred, actions)

      l = loss.mean().item()

      loss.mean().backward()
      optimizer.step()

      # writing game statistics for tensorboard
      Settings.logger.info("Iteration: " + str(count))
      Settings.summary_writer.add_scalar('Training/CrossEntropy_Loss', l, count)

      if count == 10000:
        for param_group in optimizer.param_groups:
          param_group['lr'] = 0.0002
    # save the policy
    Settings.logger.info("Saving Checkpoint")
    torch.save(immitation_policy.state_dict(), Settings.checkpoint_folder + "/" + str(count).zfill(8) + ".pt")



def get_states_actions(game_transcript, policy):

  states = []
  actions = []

  schafkopf_env = SchafkopfEnv()

  state, _, _ = schafkopf_env.set_state(PublicGameState(3), [game_transcript.player_hands[i] for i in range(4)])
  states.append(policy.preprocess(state))

  #Bidding stage

  game_player = None
  game_type = None

  if len(game_transcript.bidding_round) != 4:  # not all said weiter
    player_bidding = None
    for i in range(1, 5):
      if "Vortritt" not in game_transcript.bidding_round[-i]:
        player_bidding = game_transcript.bidding_round[-i]
        break

    if player_bidding.startswith("Ex-Sauspieler"):
      game_player = game_transcript.player_dict[player_bidding.split(" ")[0] + " " + player_bidding.split(" ")[1]]
    else:
      game_player = game_transcript.player_dict[player_bidding.split(" ")[0]]

    player_bidding = player_bidding.split(' ', 1)[1] #remove player name in case it contains one of the following words
    if "Hundsgfickte" in player_bidding:
      game_type = [0, 0]
    elif "Blaue" in player_bidding:
      game_type = [2, 0]
    elif "Alte" in player_bidding:
      game_type = [3, 0]
    elif "Schelle" in player_bidding:
      game_type = [0, 2]
    elif "Herz" in player_bidding:
      game_type = [1, 2]
    elif "Gras" in player_bidding:
      game_type = [2, 2]
    elif "Eichel" in player_bidding:
      game_type = [3, 2]
    elif "Wenz" in player_bidding:
      game_type = [None, 1]


  for i in range(4):
    action = [None, None]
    if i == game_player:
      action = game_type
    actions.append(preprocess_action(Rules.BIDDING, action))
    state, _, _ = schafkopf_env.step(action)
    if not (len(game_transcript.bidding_round) == 4 and i == 3): #don't take the last state of the game into the dataset
      states.append(policy.preprocess(state))

  if len(game_transcript.bidding_round) != 4: # if not all said weiter

    con_ret = [game_transcript.player_dict[p] for p in game_transcript.kontra]

    #CONTRA stage
    for i in range(4):
      action = False
      if len(con_ret) > 0 and i == con_ret[0]:
        action = True
      actions.append(preprocess_action(Rules.CONTRA, action))
      state, _, _ = schafkopf_env.step(action)
      states.append(policy.preprocess(state))

    # RETOUR stage
    if len(con_ret) > 0:
      for i in range(4):
        action = False
        if len(con_ret) == 2 and i == con_ret[1]:
          action = True
        actions.append(preprocess_action(Rules.RETOUR, action))
        state, _, _ = schafkopf_env.step(action)
        states.append(policy.preprocess(state))

    # TRICK stage

    for trick in range(8):
      for c in range(4):
        action = game_transcript.course_of_game[trick][c]
        actions.append(preprocess_action(Rules.TRICK, action))
        state, _, _ = schafkopf_env.step(action)
        if not (trick == 7 and c == 3): # all but the last state
          states.append(policy.preprocess(state))

  return states, actions

def preprocess_action(stage, action):
  index = None
  if stage == Rules.BIDDING:
    index = rules.games.index(action)
  elif stage == Rules.CONTRA or stage == Rules.RETOUR:
    if action == True:
      index = 9
    else:
      index = 10
  else:  # trick stage
    index = 11 + rules.cards.index(action)
  action_representation = np.zeros(43)
  action_representation[index] = 1
  #return torch.tensor(action_representation).float()
  return torch.tensor(index, dtype=torch.long)


if __name__ == '__main__':
  main()