import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()
        hidden_neurons = 256
        self.fc1 = nn.Linear(1081, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        #self.fc2_bn = nn.BatchNorm1d(2048)
        self.fc3a = nn.Linear(hidden_neurons, hidden_neurons)
        #self.fc3a_bn = nn.BatchNorm1d(1024)
        self.fc3b = nn.Linear(hidden_neurons, hidden_neurons)
        #self.fc3b_bn = nn.BatchNorm1d(1024)
        self.fc4a = nn.Linear(hidden_neurons, 41)
        self.fc4b = nn.Linear(hidden_neurons, 1)

        self.dropout = nn.Dropout(0.0)

    def forward(self, state_vector, allowed_actions):

        x = self.dropout(F.relu(self.fc1(state_vector)))
        x = self.dropout(F.relu(self.fc2(x)))
        ax = self.dropout(F.relu(self.fc3a(x)))
        bx = self.dropout(F.relu(self.fc3b(x)))
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
        state_size: 1045
        - game
          - game_type: 9
          - game_player: 4
          - first_player: 4
          - course_of_game_playerwise: 32*32=1024
          - current_scores: 4 (divided by 120 for normalization purpose)
        - player: 36
          - player_id: 4
          - remaining cards: 32
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

        course_of_game_enc = np.zeros(1024)
        for trick in range(len(game_state.course_of_game_playerwise)):
            for card in range(len(game_state.course_of_game_playerwise[trick])):
                if game_state.course_of_game_playerwise[trick][card] == [None, None]:
                    continue
                index = trick * 128 + card * 32 + game_state.rules.cards.index(game_state.course_of_game_playerwise[trick][card])
                course_of_game_enc[index] = 1

        game_vector = np.concatenate((game_enc, game_player_enc, first_player_enc, course_of_game_enc, np.true_divide(game_state.scores, 120)))

        #player
        ego_player_enc = np.zeros(4)
        ego_player_enc[player.id] = 1
        player_enc = np.append(ego_player_enc, player.one_hot_cards(player.cards))

        return torch.tensor(np.append(game_vector, player_enc)).float().to(device='cuda')