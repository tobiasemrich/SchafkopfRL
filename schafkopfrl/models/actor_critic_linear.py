import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rules import Rules
from schafkopfrl.utils import two_hot_encode_game, one_hot_cards, one_hot_games

'''
The network should have the following form

input: 55 (game info)
linear layer: 256     + 256                         + 256        
relu
linear layer: 256       
relu
linear layer: 256   +  256
relu  + relu
action layer: (9[games]+32[cards])    + value layer: 1
softmax layer

'''
class ActorCriticNetworkLinear(nn.Module):
    def __init__(self):
        super(ActorCriticNetworkLinear, self).__init__()

        self.hidden_neurons = 64

        self.fc1 = nn.Linear(342, self.hidden_neurons)
        self.fc2 = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.fc3a = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.fc3b = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.fc4a = nn.Linear(self.hidden_neurons, 43)
        self.fc4b = nn.Linear(self.hidden_neurons, 1)

        from schafkopfrl.settings import Settings
        self.device = Settings.device


    def forward(self, state_encoding):

        [state_vector, allowed_actions] = state_encoding
        allowed_actions = allowed_actions.to(device=self.device).detach()


        x = F.relu(self.fc1(state_vector))
        x = F.relu(self.fc2(x))
        ax = F.relu(self.fc3a(x))
        bx = F.relu(self.fc3b(x))
        ax = self.fc4a(ax)
        bx = self.fc4b(bx)

        ax = ax.masked_fill(allowed_actions == 0, -1e9)
        ax = F.softmax(ax, dim=-1)

        return ax, bx

    def evaluate(self, state_encoding, action):
        action_probs, state_value = self(state_encoding)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def preprocess(self, state):
        """
        state_size:
        - info_vector: 342
          - game_stage: 11
          - game_type: 7
          - game_player: 4
          - contra_retour: 8
          - first_player: 4
          - current_scores: 4 (divided by 120 for normalization purpose)
          - remaining cards: 32
          - played cards by player: 4*32
          - current_trick: 4 * 36

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

        played_cards = np.zeros(32*4)
        for p in range(4):
            cards = [game_state.course_of_game[trick][p] for trick in range(8) if game_state.course_of_game[trick][p] != [None, None]]
            enc_cards = one_hot_cards(cards)
            p_id = (p - ego_player) % 4
            played_cards[p_id*32:(p_id+1)*32] = enc_cards


        current_trick_enc = np.zeros(36*4)

        trick = game_state.trick_number
        for card in range(4):
            if game_state.course_of_game[trick][card] == [None, None]:
                continue
            else:
                card_player = game_state.first_player
                if trick != 0:
                    card_player = game_state.trick_owner[trick - 1]
                card_player = (card_player + card) % 4
                card_player_enc = np.zeros(4)
                card_player_enc[(card_player-ego_player)%4] = 1

                card_enc = one_hot_cards([game_state.course_of_game[trick][card]])

                current_trick_enc[card*36:(card+1)*36] = np.concatenate((card_enc, card_player_enc))

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

        state_vector = np.concatenate((game_stage, game_enc, game_player_enc,contra_retour, first_player_enc, np.true_divide(game_state.scores, 120), one_hot_cards(player_cards), played_cards,current_trick_enc))

        return [torch.tensor(state_vector).float().to(device=self.device), torch.tensor(allowed_actions_enc).float().to(device=self.device)]