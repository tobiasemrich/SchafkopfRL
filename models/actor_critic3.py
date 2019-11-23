import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import two_hot_encode_game, one_hot_cards
from utils import two_hot_encode_card

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
class ActorCriticNetwork3(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork3, self).__init__()

        self.hidden_neurons = 512

        self.lstm = nn.LSTM(16, self.hidden_neurons, num_layers=2)  # Input dim is 16, output dim is hidden_neurons

        self.fc1 = nn.Linear(107, self.hidden_neurons)
        self.fc2 = nn.Linear(self.hidden_neurons*2, self.hidden_neurons)
        #self.fc2_bn = nn.BatchNorm1d(2048)
        self.fc3a = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        #self.fc3a_bn = nn.BatchNorm1d(1024)
        self.fc3b = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        #self.fc3b_bn = nn.BatchNorm1d(1024)
        self.fc4a = nn.Linear(self.hidden_neurons, 41)
        self.fc4b = nn.Linear(self.hidden_neurons, 1)

    def forward(self, state_vector, allowed_actions):
        [info_vector, course_of_game] = state_vector

        #seq_lengths = [len(x) for x in course_of_game]
        # pad the seq_batch
        #padded_seq_batch = torch.nn.utils.rnn.pad_sequence(course_of_game)
        # pack the padded_seq_batch
        #packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lengths, enforce_sorted=False)

        #packed_input = torch.nn.utils.rnn.pack_padded_sequence(course_of_game, seq_lengths, enforce_sorted=False)
        #hidden = (torch.zeros(1, 1, self.hidden_neurons).float().to(device='cuda'), torch.zeros(1, 1, self.hidden_neurons).float().to(device='cuda'))  # clean out hidden state
        output, ([h1_,h2_], [c1_,c2_]) = self.lstm(course_of_game)
        #output, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(packed_output)

        #padded_output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(output, total_length=max(seq_lengths))

        x = F.relu(self.fc1(info_vector))
        x = torch.cat((x, torch.squeeze(h2_)), -1)
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
        action_probs, state_value = self(state_vector, allowed_actions)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def preprocess(self, game_state, player):
        """
        state_size:
        - info_vector: 107
          - game_type: 7 [two bit encoding]
          - game_player: 4
          - first_player: 4
          - current_scores: 4 (divided by 120 for normalization purpose)
          - player_id: 4
          - remaining cards: 32
          - current trick (3*(12 + 4)) [2 bit encoded cards + player that played it]
            - played cards
            - card player
          - teams: 4 [bits of players are set to 1]
        - game_history: x * 16
            - course_of_game: x * (12 + 4) each played card in order plus the player that played it

        """

        #game state
        game_enc = two_hot_encode_game(game_state.game_type)

        game_player_enc = np.zeros(4)
        if game_state.game_player != None:
            game_player_enc[game_state.game_player] = 1

        first_player_enc = np.zeros(4)
        first_player_enc[game_state.first_player] = 1

        team_encoding = np.zeros(4)
        player_team = game_state.get_player_team()
        if game_state.game_type[1] != 0 and len(player_team) == 1 and player_team != [None]:
            team_encoding[player_team] = 1
        elif game_state.game_type[1] == 0 and len(player_team) == 2:
            team_encoding[player_team] = 1

        #current trick
        current_trick_enc = np.zeros(48)
        for card in range(len(game_state.course_of_game[game_state.trick_number])):
            if game_state.course_of_game[game_state.trick_number][card] == [None, None]:
                continue
            else:
                card_player = game_state.first_player
                if game_state.trick_number != 0:
                    card_player = game_state.trick_owner[game_state.trick_number-1]
                card_player = (card_player + card)%4
                card_player_enc = np.zeros(4)
                card_player_enc[card_player] = 1

                current_trick_enc[card*16:(card+1)*16] = np.append(two_hot_encode_card(game_state.course_of_game[game_state.trick_number][card]), card_player_enc)

        #             course_of_game_enc = np.append(course_of_game_enc, np.array(two_hot_encode_card(game_state.course_of_game[trick][card])), axis=1)
        #         if game_state.trick_number == trick:
        #             card_enc = two_hot_encode_card(game_state.course_of_game[trick][card])
        #             current_trick_enc[card*16:(card+1)*16] = card_enc

        #course of game
        #course_of_game_enc = [torch.zeros(16).float().to(device='cuda')]
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
                    card_player_enc[card_player] = 1
                    #course_of_game_enc += [torch.tensor(np.append(np.array(two_hot_encode_card(game_state.course_of_game[trick][card])), card_player_enc)).float().to(device='cuda')]
                    course_of_game_enc = np.vstack((course_of_game_enc, np.append(np.array(two_hot_encode_card(game_state.course_of_game[trick][card])), card_player_enc)))


        #player
        ego_player_enc = np.zeros(4)
        ego_player_enc[player.id] = 1
        player_enc = np.append(ego_player_enc, one_hot_cards(player.cards))

        info_vector = np.concatenate((game_enc, game_player_enc, first_player_enc, np.true_divide(game_state.scores, 120), player_enc, team_encoding, current_trick_enc))

        #return torch.tensor(info_vector).float().to(device='cuda')
        #return [torch.tensor(info_vector).float().to(device='cuda'), course_of_game_enc]
        course_of_game_enc = torch.tensor(course_of_game_enc).float().to(device='cuda')
        course_of_game_enc = course_of_game_enc.view(len(course_of_game_enc),1,  16)
        return [torch.tensor(info_vector).float().to(device='cuda'), course_of_game_enc]