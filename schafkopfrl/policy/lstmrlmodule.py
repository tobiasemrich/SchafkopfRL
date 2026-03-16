from typing import Any

from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.torch_utils import FLOAT_MIN

from schafkopfrl.environment.utils import *
from schafkopfrl.environment.multi_agent_env import SchafkopfMultiAgentEnv
import torch.nn as nn
import torch


class LSTMRLModule(RLModule, ValueFunctionAPI, nn.Module):
    """LSTM-based RL module for Schafkopf with action masking.

    Encodes the action history with an LSTM, concatenates it with the
    player hand and info vector, then passes through separate policy
    and value heads.
    """

    @override(RLModule)
    def setup(self) -> None:
        nn.Module.__init__(self)
        self.framework: str = "torch"

        self.NUM_ACTIONS: int = 43
        self.EMB_SIZE_ACTIONS: int = 32
        self.EMB_SIZE_PLAYER: int = 4
        self.LSTM_HIDDEN: int = self.model_config.get("lstm_hidden_size", 128)

        hidden_layers: list[int] = self.model_config.get("fcnet_hiddens", [64, 64])
        output_dim: int = self.action_space.n

        self.history_encoder = LSTMHistoryEncoder(
            num_actions=self.NUM_ACTIONS,
            emb_size_actions=self.EMB_SIZE_ACTIONS,
            emb_size_player=self.EMB_SIZE_PLAYER,
            lstm_hidden_size=self.LSTM_HIDDEN,
            num_layers=self.model_config.get("lstm_num_layers", 1)
        )

        input_dim: int = 32 + SchafkopfMultiAgentEnv.INFO_VECTOR_SIZE + self.LSTM_HIDDEN  # player_hand + info_vector + encoded history

        # === Policy network ===
        policy_layers: list[nn.Module] = []
        input_size: int = input_dim
        for h in hidden_layers:
            policy_layers.append(nn.Linear(input_size, h))
            policy_layers.append(nn.ReLU())
            input_size = h
        policy_layers.append(nn.Linear(input_size, output_dim))
        self._policy_net = nn.Sequential(*policy_layers)

        # === Value function network ===
        value_layers: list[nn.Module] = []
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

    def _get_input_tensor(self, batch: dict[str, Any]) -> torch.Tensor:
        """Build the concatenated input tensor from a batch of observations.

        Parameters
        ----------
        batch : dict[str, Any]
            Batch dict containing ``obs`` with observation components.

        Returns
        -------
        torch.Tensor
            Concatenated tensor of shape ``(B, 32 + 38 + H)``.
        """
        player_hand: torch.Tensor = batch["obs"]["player_hand"]  # shape: (B, 32)
        info_vector: torch.Tensor = batch["obs"]["info_vector"]  # shape: (B, 38)
        action_history_flat: torch.Tensor = batch["obs"]["action_history"]  # shape: (B, 88)
        B: int = action_history_flat.shape[0]
        action_history: torch.Tensor = action_history_flat.reshape(B, -1, 2)  # (B, 44, 2)
        lengths: torch.Tensor = batch["obs"]["action_history_len"].reshape(-1).long()  # always (B,)

        history_encoded: torch.Tensor = self.history_encoder(action_history, lengths)  # (B, D)
        x: torch.Tensor = torch.cat([player_hand, info_vector, history_encoded], dim=-1)  # (B, 32 + 38 + H)
        return x

    @override(RLModule)
    def _forward(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, torch.Tensor]:
        x: torch.Tensor = self._get_input_tensor(batch)
        logits: torch.Tensor = self._policy_net(x)

        action_mask: torch.Tensor = batch["obs"]["action_mask"]
        inf_mask: torch.Tensor = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits: torch.Tensor = logits + inf_mask

        return {Columns.ACTION_DIST_INPUTS: masked_logits}

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        x = self._get_input_tensor(batch)
        return self._value_net(x).squeeze(1)

    @override(RLModule)
    def get_state(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.state_dict().items()}

    @override(RLModule)
    def set_state(self, state: dict[str, torch.Tensor], **kwargs: Any) -> None:
        device: torch.device = next(self.parameters()).device
        self.load_state_dict({k: v.to(device) for k, v in state.items()})

    @override(RLModule)
    def get_exploration_action_dist_cls(self) -> type[TorchCategorical]:
        return TorchCategorical

    @override(RLModule)
    def get_train_action_dist_cls(self) -> type[TorchCategorical]:
        return TorchCategorical

    def init_weights(self, m: nn.Module) -> None:
        """Initialize linear layer weights with Xavier uniform.

        Parameters
        ----------
        m : nn.Module
            Module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class LSTMHistoryEncoder(nn.Module):
    """LSTM encoder for the action history sequence.

    Embeds action and player ids, then processes the sequence with an LSTM.
    Returns the final hidden state as the history encoding.

    Parameters
    ----------
    num_actions : int
        Number of distinct actions (used for embedding table size).
    emb_size_actions : int
        Embedding dimension for actions.
    emb_size_player : int
        Embedding dimension for player ids.
    lstm_hidden_size : int
        Hidden size of the LSTM.
    num_layers : int
        Number of LSTM layers.
    """
    def __init__(self, num_actions: int, emb_size_actions: int, emb_size_player: int, lstm_hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(num_actions + 1, emb_size_actions, padding_idx=0)
        self.player_embed = nn.Embedding(5, emb_size_player, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=emb_size_actions + emb_size_player,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            num_layers=num_layers
        )

    def forward(self, action_history: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode a batch of action history sequences.

        Parameters
        ----------
        action_history : torch.Tensor
            Shape ``(B, T, 2)`` with ``[action_id, player_id]`` per step.
        lengths : torch.Tensor
            Shape ``(B,)`` with sequence lengths per batch element.

        Returns
        -------
        torch.Tensor
            Encoded history of shape ``(B, lstm_hidden_size)``.
        """
        # action_history: (B, T, 2) => action_id, player_id
        action_ids: torch.Tensor = action_history[:, :, 0].long() + 1  # pad -1 → 0
        player_ids: torch.Tensor = action_history[:, :, 1].long() + 1

        action_emb: torch.Tensor = self.action_embed(action_ids)
        player_emb: torch.Tensor = self.player_embed(player_ids)

        lstm_input: torch.Tensor = torch.cat([action_emb, player_emb], dim=-1)  # (B, T, F)

        # need to handle special case of empty sequences
        encoded: torch.Tensor = torch.zeros(action_history.shape[0], self.lstm.hidden_size, device=action_history.device)

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