from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.torch_utils import FLOAT_MIN

from schafkopfrl.environment.utils import *
import torch.nn as nn
import torch


class LinearRLModule(RLModule, ValueFunctionAPI, nn.Module):

    @override(RLModule)
    def setup(self):
        nn.Module.__init__(self)
        self.framework = "torch"

        self.MAX_ACTIONS = 44 # the maximum length of actions in a game (4+4+4+32)
        self.NUM_ACTIONS = 43 # how many actions there are (9 games, 2 double, 32 cards)
        self.EMB_SIZE_ACTONS = 32
        self.EMB_SIZE_PLAYER = 4

        input_dim = 32 + self.MAX_ACTIONS*(self.EMB_SIZE_ACTONS + self.EMB_SIZE_PLAYER)
        hidden_layers = self.model_config.get("fcnet_hiddens", [64, 64])
        output_dim = self.action_space.n

        self.action_embed = nn.Embedding(self.NUM_ACTIONS+1, self.EMB_SIZE_ACTONS, padding_idx=0)
        self.player_embed = nn.Embedding(4+1, self.EMB_SIZE_PLAYER, padding_idx=0)

        # === Policy network ===
        layers = []
        input_size = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            input_size = h

        # Final layer: output logits for discrete actions
        layers.append(nn.Linear(input_size, output_dim))
        self._policy_net = nn.Sequential(*layers)
        # === Value function network ===
        val_layers = []
        input_size = input_dim
        for h in hidden_layers:
            val_layers.append(nn.Linear(input_size, h))
            val_layers.append(nn.ReLU())
            input_size = h

        val_layers.append(nn.Linear(input_size, 1))
        self._value_net = nn.Sequential(*val_layers)

        # init weghts
        self._value_net.apply(self.init_weights)
        self._policy_net.apply(self.init_weights)
        #self._value = None

        self.dev = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count()== 1) else "cpu")
        self.to(self.dev)


    def _get_input_tensor(self, batch):
        
        player_hand = batch["obs"]["player_hand"].to(self.dev)
        action_history = batch["obs"]["action_history"].to(self.dev)


        # embedding of action history
        action_vec = self.action_embed(action_history[:, :, 0]+1) # need +1 for getting -1 to a valid index
        player_vec = self.player_embed(action_history[:, :, 1]+1) # need +1 for getting -1 to a valid index

        # Shared input to policy and value networks
        x = torch.cat([player_hand, torch.flatten(action_vec, start_dim=1), torch.flatten(player_vec, start_dim=1)], dim=-1)

        return x

    @override(RLModule)
    def _forward(self, batch, **kwargs):
        x = self._get_input_tensor(batch)
        logits = self._policy_net(x)

        # Mask invalid actions by setting logits to a large negative value
        action_mask = batch["obs"]["action_mask"].to(self.dev)
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return {Columns.ACTION_DIST_INPUTS: masked_logits}

    @override(ValueFunctionAPI)
    def compute_values(self, batch, **kwargs):
        x = self._get_input_tensor(batch)
        values = self._value_net(x).squeeze(1)
        return values

    @override(RLModule)
    def get_state(self, **kwargs):
        return self.state_dict()

    @override(RLModule)
    def set_state(self, state, **kwargs):
        self.load_state_dict(state)

    @override(RLModule)
    def get_exploration_action_dist_cls(self):
        # For discrete action spaces with torch, use TorchCategorical
        return TorchCategorical
    @override(RLModule)
    def get_train_action_dist_cls(self):
        # For discrete action spaces with torch, use TorchCategorical
        return TorchCategorical

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Good for ReLU activations
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
