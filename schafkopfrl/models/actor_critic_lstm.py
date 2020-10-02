import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from rules import Rules
from schafkopfrl.utils import two_hot_encode_game, two_hot_encode_card, one_hot_cards, one_hot_games


'''
The network should have the following form

input: 55 (game info) + 16*x (lstm of game history) + 16*x (lstm of current trick)
linear layer: 256     + 256                         + 256        
relu
linear layer: 256       
relu
linear layer: 256   +  256
relu  + relu
action layer: (9[games]+32[cards])    + value layer: 1
softmax layer

'''
class ActorCriticNetworkLSTM(nn.Module):
    def __init__(self):
        super(ActorCriticNetworkLSTM, self).__init__()

        self.hidden_neurons = 512

        self.lstm_course_of_game = nn.LSTM(16, self.hidden_neurons, num_layers=2)  # Input dim is 16, output dim is hidden_neurons
        self.lstm_current_trick = nn.LSTM(16, self.hidden_neurons, num_layers=2)  # Input dim is 16, output dim is hidden_neurons

        self.fc1 = nn.Linear(70, self.hidden_neurons)
        self.fc2 = nn.Linear(self.hidden_neurons*3, self.hidden_neurons)
        #self.fc2_bn = nn.BatchNorm1d(2048)
        self.fc3a = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        #self.fc3a_bn = nn.BatchNorm1d(1024)
        self.fc3b = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        #self.fc3b_bn = nn.BatchNorm1d(1024)
        self.fc4a = nn.Linear(self.hidden_neurons, 43)
        self.fc4b = nn.Linear(self.hidden_neurons, 1)

        from settings import Settings
        self.device = Settings.device


    def forward(self, state_encoding):
        [info_vector, course_of_game, current_trick, allowed_actions] = state_encoding


        output, ([h1_,h2_], [c1_,c2_]) = self.lstm_course_of_game(course_of_game)

        output, ([h3_, h4_], [c3_, c4_]) = self.lstm_current_trick(current_trick)


        x = F.relu(self.fc1(info_vector))
        x = torch.cat((x, torch.squeeze(h2_), torch.squeeze(h4_)), -1)
        x = F.relu(self.fc2(x))
        ax = F.relu(self.fc3a(x))
        bx = F.relu(self.fc3b(x))
        ax = self.fc4a(ax)
        bx = self.fc4b(bx)

        ax = ax.masked_fill(allowed_actions == 0, -1e9)

        ax = F.softmax(ax, dim=-1)

        return ax, bx

    def evaluate(self, state_vector, action):
        action_probs, state_value = self(state_vector)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def preprocess(self, state):
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

        #game stage
        game_stage = np.zeros(11)
        if game_state.game_stage == Rules.BIDDING:
            game_stage[0] = 1
        elif game_state.game_stage == Rules.CONTRA:
            game_stage[1] = 1
        elif game_state.game_stage == Rules.RETOUR:
            game_stage[2] = 1
        else:
            game_stage[3+game_state.trick_number] = 1


        game_enc = two_hot_encode_game(game_state.game_type)

        game_player_enc = np.zeros(4)
        if game_state.game_player != None:
            game_player_enc[(game_state.game_player-ego_player)%4] = 1

        contra_retour = np.zeros(8)
        for p in range (4):
            if game_state.contra[p]:
                contra_retour[(p-ego_player)%4] = 1
        for p in range (4):
            if game_state.retour[p]:
                contra_retour[4 + (p-ego_player)%4] = 1

        first_player_enc = np.zeros(4)
        first_player_enc[(game_state.first_player-ego_player)%4] = 1
        '''
        team_encoding = np.zeros(4)
        if game_state.get_player_team() != [None]:
            player_team = [(t-ego_player)%4 for t in game_state.get_player_team()]

            if game_state.game_type[1] != 0 and len(player_team) == 1:
                team_encoding[player_team] = 1
            elif game_state.game_type[1] == 0 and len(player_team) == 2:
                team_encoding[player_team] = 1
        '''

        #course of game
        #course_of_game_enc = [torch.zeros(16).float().to(device='cuda')]
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
                    card_player_enc[(card_player-ego_player)%4] = 1
                    if trick != game_state.trick_number:
                        course_of_game_enc = np.vstack((course_of_game_enc, np.append(np.array(two_hot_encode_card(game_state.course_of_game[trick][card])), card_player_enc)))
                    else:
                        current_trick_enc = np.vstack((current_trick_enc, np.append(np.array(two_hot_encode_card(game_state.course_of_game[trick][card])), card_player_enc)))

        info_vector = np.concatenate((game_stage, game_enc, game_player_enc, contra_retour, first_player_enc, np.true_divide(game_state.scores, 120), one_hot_cards(player_cards))) #, team_encoding

        #return torch.tensor(info_vector).float().to(device='cuda')
        #return [torch.tensor(info_vector).float().to(device='cuda'), course_of_game_enc]
        if course_of_game_enc.shape[0] > 1:
            course_of_game_enc = np.delete(course_of_game_enc, 0, 0)
        course_of_game_enc = torch.tensor(course_of_game_enc).float().to(device=self.device)
        course_of_game_enc = course_of_game_enc.view(len(course_of_game_enc),1,  16)

        if current_trick_enc.shape[0] > 1:
            current_trick_enc = np.delete(current_trick_enc, 0, 0)
        current_trick_enc = torch.tensor(current_trick_enc).float().to(device=self.device)
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


        return [torch.tensor(info_vector).float().to(device=self.device), course_of_game_enc, current_trick_enc, torch.tensor(allowed_actions_enc).float().to(device=self.device)]