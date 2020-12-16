import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from rules import Rules
from utils import two_hot_encode_game, two_hot_encode_card, one_hot_cards, one_hot_games


'''
The network should have the following form

input: 55 (game info) + 16*x (lstm of game history) + 16*x (lstm of current trick)
linear layer: 256     + 256                         + 256        
relu
linear layer: 256       
relu
linear layer: 256
relu  + relu
action layer: (9[games]+32[cards])
softmax layer

'''
class ImmitationPolicy(nn.Module):
    def __init__(self):
        super(ImmitationPolicy, self).__init__()

        self.hidden_neurons = 512

        self.lstm_course_of_game = nn.LSTM(16, self.hidden_neurons, num_layers=2)  # Input dim is 16, output dim is hidden_neurons
        self.lstm_current_trick = nn.LSTM(16, self.hidden_neurons, num_layers=2)  # Input dim is 16, output dim is hidden_neurons

        self.fc1 = nn.Linear(70, self.hidden_neurons)
        self.fc2 = nn.Linear(self.hidden_neurons*3, self.hidden_neurons)
        self.fc3 = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.fc4 = nn.Linear(self.hidden_neurons, 43)

        from settings import Settings
        self.device = Settings.device


    def forward(self, state_encoding):
        [info_vector, course_of_game, current_trick, allowed_actions] = state_encoding


        output, ([h1_,h2_], [c1_,c2_]) = self.lstm_course_of_game(course_of_game)

        output, ([h3_, h4_], [c3_, c4_]) = self.lstm_current_trick(current_trick)


        x = F.relu(self.fc1(info_vector))
        x = torch.cat((x, torch.squeeze(h2_), torch.squeeze(h4_)), -1)
        x = F.relu(self.fc2(x))
        ax = F.relu(self.fc3(x))
        ax = self.fc4a(ax)

        ax = ax.masked_fill(allowed_actions == 0, -1e9)

        return ax
