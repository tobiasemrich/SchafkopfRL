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


  states = []
  actions = []
  #load and preprocess database
  games = SqliteDict('../sauspiel/games.sqlite')
  count = 0
  for g in games:
    game = games[g]
    if len(game.sonderregeln) == 0:
      print(g)
      count += 1
      game_states, game_actions = get_states_actions(game)
      states += game_states
      actions += game_actions

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
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


  training_generator = data.DataLoader(train_dataset, collate_fn=dataset.custom_collate,batch_size=10000, shuffle=True)
  testing_generator = data.DataLoader(test_dataset, collate_fn=dataset.custom_collate, batch_size=10000)

  # start tensorboard
  tb = program.TensorBoard()
  tb.configure(argv=[None, '--logdir', Settings.runs_folder])
  tb.launch()

  # loading initial policy
  immitation_policy = ImmitationPolicy().to(Settings.device)
  # take the newest generation available
  i_episode = max_gen = 0
  generations = [int(f[:8]) for f in listdir(Settings.checkpoint_folder) if f.endswith(".pt")]
  if len(generations) > 0:
    max_gen = max(generations)
    immitation_policy.load_state_dict(torch.load(Settings.checkpoint_folder + "/" + str(max_gen).zfill(8) + ".pt"))
    i_episode = max_gen

  optimizer = torch.optim.Adam(immitation_policy.parameters(), lr=Settings.lr, betas=Settings.betas,
                               weight_decay=Settings.optimizer_weight_decay)

  # training loop
  immitation_policy.train()

  count = 0

  for epoch in range(100000):  # epoch

    for i, (states, actions) in enumerate(training_generator):  #batch

      count += 1

      # Transfer to GPU
      states = [state.to(Settings.device) for state in states]
      actions = actions.to(Settings.device)

      optimizer.zero_grad()

      pred, val = immitation_policy(states)
      #loss = nn.MSELoss()(pred, actions) #TODO: replace by cross entropy
      loss = nn.CrossEntropyLoss()(pred, actions)
      #loss = nn.NLLLoss()(pred, actions)

      l = loss.mean().item()

      loss.mean().backward()
      optimizer.step()

      # writing game statistics for tensorboard
      Settings.logger.info("Iteration: " + str(count))
      Settings.summary_writer.add_scalar('Training/CrossEntropy_Loss', l, count)
    # save the policy
    Settings.logger.info("Saving Checkpoint")
    torch.save(immitation_policy.state_dict(), Settings.checkpoint_folder + "/" + str(count).zfill(8) + ".pt")

    #testing
    if epoch % 1 == 0:
      correct = 0
      total = 0
      with torch.no_grad():
        for i, (states, actions) in enumerate(testing_generator):
          pred, val = immitation_policy(states)
          _, predicted = torch.max(pred.data, 1)
          total += actions.size(0)
          correct += (predicted.cpu() == actions.cpu()).sum().item()
      Settings.summary_writer.add_scalar('Testing/Accuracy', correct/total, count)





def get_states_actions(game_transcript):

  states = []
  actions = []

  schafkopf_env = SchafkopfEnv()

  state, _, _ = schafkopf_env.set_state(PublicGameState(3), [game_transcript.player_hands[i] for i in range(4)])
  states.append(preprocess(state))

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
      states.append(preprocess(state))

  if len(game_transcript.bidding_round) != 4: # if not all said weiter

    con_ret = [game_transcript.player_dict[p] for p in game_transcript.kontra]

    #CONTRA stage
    for i in range(4):
      action = False
      if len(con_ret) > 0 and i == con_ret[0]:
        action = True
      actions.append(preprocess_action(Rules.CONTRA, action))
      state, _, _ = schafkopf_env.step(action)
      states.append(preprocess(state))

    # RETOUR stage
    if len(con_ret) > 0:
      for i in range(4):
        action = False
        if len(con_ret) == 2 and i == con_ret[1]:
          action = True
        actions.append(preprocess_action(Rules.RETOUR, action))
        state, _, _ = schafkopf_env.step(action)
        states.append(preprocess(state))

    # TRICK stage

    for trick in range(8):
      for c in range(4):
        action = game_transcript.course_of_game[trick][c]
        actions.append(preprocess_action(Rules.TRICK, action))
        state, _, _ = schafkopf_env.step(action)
        if not (trick == 7 and c == 3): # all but the last state
          states.append(preprocess(state))

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

def preprocess(state):
  """
  state_size:
  - info_vector: 70 (74)
    - game_stage: 11
    - game_type: 7 [two bit encoding]
    - game_player: 4
    - contra_retour: 8
    - first_player: 4
    - current_scores: 4 (divided by 120 for normalization purpose)
    - remaining cards: 32
    (- teams: 4 [bits of players are set to 1])
  - game_history: x * 16
      - course_of_game: x * (12 + 4) each played card in order plus the player that played it
  - current_trick: x * 16
      - current_trick: x * (12 + 4) each played card in order plus the player that played it

  action_size (43):
   - games: 9
   - contra/double: 2
   - cards:  32
  """

  game_state = state["game_state"]
  player_cards = state["current_player_cards"]
  allowed_actions = state["allowed_actions"]

  ############### gamestate ##################
  ego_player = game_state.current_player

  # game stage
  game_stage = np.zeros(11)
  if game_state.game_stage == Rules.BIDDING:
    game_stage[0] = 1
  elif game_state.game_stage == Rules.CONTRA:
    game_stage[1] = 1
  elif game_state.game_stage == Rules.RETOUR:
    game_stage[2] = 1
  else:
    game_stage[3 + game_state.trick_number] = 1

  game_enc = two_hot_encode_game(game_state.game_type)

  game_player_enc = np.zeros(4)
  if game_state.game_player != None:
    game_player_enc[(game_state.game_player - ego_player) % 4] = 1

  contra_retour = np.zeros(8)
  for p in range(4):
    if game_state.contra[p]:
      contra_retour[(p - ego_player) % 4] = 1
  for p in range(4):
    if game_state.retour[p]:
      contra_retour[4 + (p - ego_player) % 4] = 1

  first_player_enc = np.zeros(4)
  first_player_enc[(game_state.first_player - ego_player) % 4] = 1
  '''
  team_encoding = np.zeros(4)
  if game_state.get_player_team() != [None]:
      player_team = [(t-ego_player)%4 for t in game_state.get_player_team()]

      if game_state.game_type[1] != 0 and len(player_team) == 1:
          team_encoding[player_team] = 1
      elif game_state.game_type[1] == 0 and len(player_team) == 2:
          team_encoding[player_team] = 1
  '''

  # course of game
  # course_of_game_enc = [torch.zeros(16).float().to(device='cuda')]
  course_of_game_enc = np.zeros((1, 16))
  current_trick_enc = np.zeros((1, 16))
  for trick in range(len(game_state.course_of_game)):
    for card in range(len(game_state.course_of_game[trick])):
      if game_state.course_of_game[trick][card] == [None, None]:
        continue
      else:
        card_player = game_state.first_player
        if trick != 0:
          card_player = game_state.trick_owner[trick - 1]
        card_player = (card_player + card) % 4
        card_player_enc = np.zeros(4)
        card_player_enc[(card_player - ego_player) % 4] = 1
        if trick != game_state.trick_number:
          course_of_game_enc = np.vstack((course_of_game_enc, np.append(
            np.array(two_hot_encode_card(game_state.course_of_game[trick][card])), card_player_enc)))
        else:
          current_trick_enc = np.vstack((current_trick_enc, np.append(
            np.array(two_hot_encode_card(game_state.course_of_game[trick][card])), card_player_enc)))

  info_vector = np.concatenate((game_stage, game_enc, game_player_enc, contra_retour, first_player_enc,
                                np.true_divide(game_state.scores, 120), one_hot_cards(player_cards)))  # , team_encoding

  # return torch.tensor(info_vector).float().to(device='cuda')
  # return [torch.tensor(info_vector).float().to(device='cuda'), course_of_game_enc]
  if course_of_game_enc.shape[0] > 1:
    course_of_game_enc = np.delete(course_of_game_enc, 0, 0)
  course_of_game_enc = torch.tensor(course_of_game_enc).float().to(device=Settings.device)
  course_of_game_enc = course_of_game_enc.view(len(course_of_game_enc), 1, 16)

  if current_trick_enc.shape[0] > 1:
    current_trick_enc = np.delete(current_trick_enc, 0, 0)
  current_trick_enc = torch.tensor(current_trick_enc).float().to(device=Settings.device)
  current_trick_enc = current_trick_enc.view(len(current_trick_enc), 1, 16)

  ############### allowed actions ##################
  allowed_actions_enc = np.zeros(43)
  if game_state.game_stage == Rules.BIDDING:
    allowed_actions_enc[0:9] = one_hot_games(allowed_actions)
  elif game_state.game_stage == Rules.CONTRA or game_state.game_stage == Rules.RETOUR:
    allowed_actions_enc[10] = 1
    if np.any(allowed_actions):
      allowed_actions_enc[9] = 1
  else:
    allowed_actions_enc[11:] = one_hot_cards(allowed_actions)

  return [torch.tensor(info_vector).float().to(device=Settings.device), course_of_game_enc, current_trick_enc,
          torch.tensor(allowed_actions_enc).float().to(device=Settings.device)]


if __name__ == '__main__':
  main()