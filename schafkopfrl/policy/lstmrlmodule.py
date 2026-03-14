from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.torch_utils import FLOAT_MIN

from schafkopfrl.environment.utils import *
import torch.nn as nn
import torch


class LSTMRLModule(RLModule, ValueFunctionAPI, nn.Module):

    @override(RLModule)
    def setup(self):
        nn.Module.__init__(self)
        self.framework = "torch"

        self.NUM_ACTIONS = 43
        self.EMB_SIZE_ACTIONS = 32
        self.EMB_SIZE_PLAYER = 4
        self.LSTM_HIDDEN = self.model_config.get("lstm_hidden_size", 128)

        hidden_layers = self.model_config.get("fcnet_hiddens", [64, 64])
        output_dim = self.action_space.n

        self.history_encoder = LSTMHistoryEncoder(
            num_actions=self.NUM_ACTIONS,
            emb_size_actions=self.EMB_SIZE_ACTIONS,
            emb_size_player=self.EMB_SIZE_PLAYER,
            lstm_hidden_size=self.LSTM_HIDDEN,
            num_layers=self.model_config.get("lstm_num_layers", 1)
        )

        input_dim = 32 + self.LSTM_HIDDEN  # player_hand + encoded history

        # === Policy network ===
        policy_layers = []
        input_size = input_dim
        for h in hidden_layers:
            policy_layers.append(nn.Linear(input_size, h))
            policy_layers.append(nn.ReLU())
            input_size = h
        policy_layers.append(nn.Linear(input_size, output_dim))
        self._policy_net = nn.Sequential(*policy_layers)

        # === Value function network ===
        value_layers = []
        input_size = input_dim
        for h in hidden_layers:
            value_layers.append(nn.Linear(input_size, h))
            value_layers.append(nn.ReLU())
            input_size = h
        value_layers.append(nn.Linear(input_size, 1))
        self._value_net = nn.Sequential(*value_layers)

        # Init
        self._policy_net.apply(self.init_weights)
        self._value_net.apply(self.init_weights)

    def _get_input_tensor(self, batch):
        player_hand = batch["obs"]["player_hand"]  # shape: (B, 32)
        action_history_flat = batch["obs"]["action_history"]  # shape: (B, 88)
        B = action_history_flat.shape[0]
        action_history = action_history_flat.reshape(B, -1, 2)  # (B, 44, 2)
        lengths = batch["obs"]["action_history_len"].reshape(-1).long()  # always (B,)

        history_encoded = self.history_encoder(action_history, lengths)  # (B, D)
        x = torch.cat([player_hand, history_encoded], dim=-1)  # (B, 32 + H)
        return x

    @override(RLModule)
    def _forward(self, batch, **kwargs):
        x = self._get_input_tensor(batch)
        logits = self._policy_net(x)

        action_mask = batch["obs"]["action_mask"]
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return {Columns.ACTION_DIST_INPUTS: masked_logits}

    @override(ValueFunctionAPI)
    def compute_values(self, batch, **kwargs):
        x = self._get_input_tensor(batch)
        return self._value_net(x).squeeze(1)

    @override(RLModule)
    def get_state(self, **kwargs):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    @override(RLModule)
    def set_state(self, state, **kwargs):
        device = next(self.parameters()).device
        self.load_state_dict({k: v.to(device) for k, v in state.items()})

    @override(RLModule)
    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    @override(RLModule)
    def get_train_action_dist_cls(self):
        return TorchCategorical

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class LSTMHistoryEncoder(nn.Module):
    def __init__(self, num_actions, emb_size_actions, emb_size_player, lstm_hidden_size, num_layers):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions + 1, emb_size_actions, padding_idx=0)
        self.player_embed = nn.Embedding(5, emb_size_player, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=emb_size_actions + emb_size_player,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            num_layers=num_layers
        )

    def forward(self, action_history, lengths):
        # action_history: (B, T, 2) => action_id, player_id
        action_ids = action_history[:, :, 0].long() + 1  # pad -1 → 0
        player_ids = action_history[:, :, 1].long() + 1

        action_emb = self.action_embed(action_ids)
        player_emb = self.player_embed(player_ids)

        lstm_input = torch.cat([action_emb, player_emb], dim=-1)  # (B, T, F)

        # need to handle special case of empty sequences
        encoded = torch.zeros(action_history.shape[0], self.lstm.hidden_size, device=action_history.device)

        if lengths.gt(0).any():
            # Get indices of non-empty sequences
            nonzero_idx = lengths.gt(0).nonzero(as_tuple=True)[0]
            lengths_nz = lengths[nonzero_idx]
            input_nz = lstm_input[nonzero_idx]

            packed = nn.utils.rnn.pack_padded_sequence(
                input_nz, lengths_nz.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
            encoded[nonzero_idx] = h_n[-1]  # (B_nz, hidden_size)

        return encoded  # (B, hidden_size)