import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.models.catalog import ModelCatalog


class LinearModel(TorchModelV2, nn.Module):
    def __init__(
        self, 
        obs_space, 
        action_space, 
        num_outputs, 
        model_config, 
        name
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.obs_size = obs_space["observation"].shape[0]
        self.num_outputs = num_outputs

        # === Policy network ===
        layers = []
        input_size = self.obs_size
        hidden_layers = model_config.get("fcnet_hiddens", [64, 64])

        for h in hidden_layers:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            input_size = h

        # Final layer: output logits for discrete actions
        layers.append(nn.Linear(input_size, num_outputs))
        self.policy_net = nn.Sequential(*layers)

        # === Value function network ===
        val_layers = []
        input_size = self.obs_size
        for h in hidden_layers:
            val_layers.append(nn.Linear(input_size, h))
            val_layers.append(nn.ReLU())
            input_size = h

        val_layers.append(nn.Linear(input_size, 1))
        self.value_net = nn.Sequential(*val_layers)

        self._value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["observation"]  # shape: [B, obs_dim]
        action_mask = input_dict["obs"]["action_mask"]  # shape: [B, num_actions]

        logits = self.policy_net(obs)

        # Mask invalid actions by setting logits to a large negative value
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Store value for value_function()
        self._value = self.value_net(obs).squeeze(1)

        return masked_logits, state

    def value_function(self):
        return self._value
