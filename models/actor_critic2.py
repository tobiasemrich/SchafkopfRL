import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import one_hot_cards

'''
The network should have the following form

input: 1045 (state size) + 32 (remaining cards of player) + 4 player id
hidden layer: 1028
relu
##hidden layer: 1028
##relu
hidden layer: 512   + hidden layer: 512
relu                + relu
action layer: (9[games]+32[cards])    + value layer: 1
softmax layer

'''
class ActorCriticNetwork2(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork2, self).__init__()
        hidden_neurons = 512

        self.conv1 = torch.nn.Conv1d(1, 64, 132, stride=132)

        self.fc1 = nn.Linear(189, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        #self.fc2_bn = nn.BatchNorm1d(2048)
        self.fc3a = nn.Linear(hidden_neurons, hidden_neurons)
        #self.fc3a_bn = nn.BatchNorm1d(1024)
        self.fc3b = nn.Linear(hidden_neurons, hidden_neurons)
        #self.fc3b_bn = nn.BatchNorm1d(1024)
        self.fc4a = nn.Linear(hidden_neurons, 41)
        self.fc4b = nn.Linear(hidden_neurons, 1)

    def forward(self, state_vector, allowed_actions):
        #state_vector = torch.reshape(state_vector, (-1, 1245))
        #info = state_vector[:, :189]
        #course_of_game_playerwise = torch.reshape(state_vector[:, 189:], (-1, 1, 1056))
        #x1 = F.relu(self.conv1(course_of_game_playerwise))
        #x1 = torch.flatten(x1, start_dim=1)
        #x2=F.relu(self.fc1(info))
        #x2 = torch.flatten(x2,start_dim=1)
        #x = torch.cat((x1, x2), 1)
        x = F.relu(self.fc1(state_vector))
        x = F.relu(self.fc2(x))
        ax = F.relu(self.fc3a(x))
        bx = F.relu(self.fc3b(x))
        ax = self.fc4a(ax)
        bx = self.fc4b(bx)
        ax = F.softmax(ax)
        ax = torch.mul(ax, allowed_actions)
        ax /= torch.sum(ax)

        return ax, bx

    def evaluate(self, state_vector, allowed_actions, action):
        self.eval()
        action_probs, state_value = self(state_vector, allowed_actions)
        self.train()
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def preprocess(self, game_state, player):
        """
        state_size:
        - info_vector: 189
          - game_type: 9
          - game_player: 4
          - first_player: 4
          - current_scores: 4 (divided by 120 for normalization purpose)
          - player_id: 4
          - remaining cards: 32
          - current trick (4*32 + 4)
            - played cards
            - first player
        - game_history: 1056 (8*132)
            - course_of_game_playerwise: 8*(4*32+4)=1056 for each trick the played cards plus the first player

        """

        #game state
        game_enc = np.zeros(9)
        if game_state.game_type != [None, None]:
            game_enc[game_state.rules.games.index(game_state.game_type)] = 1

        game_player_enc = np.zeros(4)
        if game_state.game_player != None:
            game_player_enc[game_state.game_player] = 1

        first_player_enc = np.zeros(4)
        first_player_enc[game_state.first_player] = 1

        course_of_game_enc = np.zeros(1056)
        current_trick_enc = np.zeros(132)

        for trick in range(len(game_state.course_of_game_playerwise)):
            trick_index = trick * 132
            for card in range(len(game_state.course_of_game_playerwise[trick])):
                if game_state.course_of_game_playerwise[trick][card] == [None, None]:
                    continue
                index = trick_index + card * 32 + game_state.rules.cards.index(game_state.course_of_game_playerwise[trick][card])
                if trick == game_state.trick_number:
                    current_trick_enc[card * 32 + game_state.rules.cards.index(game_state.course_of_game_playerwise[trick][card])] = 1
                else:
                    course_of_game_enc[index] = 1

            if trick == 0:
                if trick == game_state.trick_number:
                    current_trick_enc[128+game_state.first_player] = 1
                else:
                    course_of_game_enc[trick_index+128+game_state.first_player] = 1
            elif game_state.trick_owner[trick-1] != None:
                if trick == game_state.trick_number:
                    current_trick_enc[128 + game_state.trick_owner[trick-1]] = 1
                else:
                    course_of_game_enc[trick_index + 128 + game_state.trick_owner[trick-1]] = 1

        #player
        ego_player_enc = np.zeros(4)
        ego_player_enc[player.id] = 1
        player_enc = np.append(ego_player_enc, one_hot_cards(player.cards))

        info_vector = np.concatenate((game_enc, game_player_enc, first_player_enc, np.true_divide(game_state.scores, 120), player_enc, current_trick_enc))

        #return torch.tensor(np.concatenate((info_vector, course_of_game_enc))).float().to(device='cuda')
        return torch.tensor(info_vector).float().to(device='cuda')