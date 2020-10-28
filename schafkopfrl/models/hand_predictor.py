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
class HandPredictor(nn.Module):
    def __init__(self):
        super(HandPredictor, self).__init__()

        self.hidden_neurons = 512

        self.lstm_course_of_game = nn.LSTM(16, self.hidden_neurons, num_layers=2)  # Input dim is 16, output dim is hidden_neurons

        self.fc1 = nn.Linear(70, self.hidden_neurons)
        self.fc2 = nn.Linear(self.hidden_neurons*2, self.hidden_neurons)
        self.fc3 = nn.Linear(self.hidden_neurons, 32*4)

        from settings import Settings
        self.device = Settings.device

        self.rules = Rules()


    def forward(self, state_encoding):
        [info_vector, course_of_game] = state_encoding


        output, ([h1_,h2_], [c1_,c2_]) = self.lstm_course_of_game(course_of_game)

        x = F.relu(self.fc1(info_vector))

        x = torch.cat((torch.squeeze(x), torch.squeeze(h2_)), -1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = torch.reshape(x, (-1, 32, 4))


        x = F.softmax(x, dim=2)

        x = torch.squeeze(x)
        return x

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

        course_of_game_enc = np.zeros((1, 16))

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
                    course_of_game_enc = np.vstack((course_of_game_enc, np.append(np.array(two_hot_encode_card(game_state.course_of_game[trick][card])), card_player_enc)))


        info_vector = np.concatenate((game_stage, game_enc, game_player_enc, contra_retour, first_player_enc, np.true_divide(game_state.scores, 120), one_hot_cards(player_cards))) #, team_encoding


        if course_of_game_enc.shape[0] > 1:
            course_of_game_enc = np.delete(course_of_game_enc, 0, 0)
        course_of_game_enc = torch.tensor(course_of_game_enc).float().to(device=self.device)
        course_of_game_enc = course_of_game_enc.view(len(course_of_game_enc),1,  16)


        return [torch.tensor(info_vector).float().to(device=self.device), course_of_game_enc]

    def encode_player_hands(self, player_hands, current_player):
        card_dist_enc = np.zeros((32, 4))
        card_dist_enc[:, 3] = 1
        for p in range(4):
            if p == current_player:
                continue
            for card in player_hands[p]:
                card_index = self.rules.cards.index(card)
                card_dist_enc[card_index, (p- current_player)%4 -1] = 1
                card_dist_enc[card_index, 3] = 0

        return torch.tensor(card_dist_enc).float().to(device=self.device)



