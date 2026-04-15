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


class TransformerRLModule(RLModule, ValueFunctionAPI, nn.Module):
    """Transformer-based RL module for Schafkopf with action masking.

    Encodes the action history with a Transformer encoder, concatenates it with the
    player hand and info vector, then passes through separate policy and value heads.
    """

    @override(RLModule)
    def setup(self) -> None:
        nn.Module.__init__(self)
        self.framework: str = "torch"

        self.NUM_ACTIONS: int = 43
        self.EMB_SIZE_ACTIONS: int = 32
        self.EMB_SIZE_PLAYER: int = 4
        self.TRANSFORMER_DIM: int = self.model_config.get("transformer_dim", 128)

        hidden_layers: list[int] = self.model_config.get("fcnet_hiddens", [64, 64])
        output_dim: int = self.action_space.n

        self.history_encoder = TransformerHistoryEncoder(
            num_actions=self.NUM_ACTIONS,
            emb_size_actions=self.EMB_SIZE_ACTIONS,
            emb_size_player=self.EMB_SIZE_PLAYER,
            transformer_dim=self.TRANSFORMER_DIM,
            num_heads=self.model_config.get("num_attention_heads", 4),
            num_layers=self.model_config.get("transformer_num_layers", 2),
        )

        input_dim: int = 32 + SchafkopfMultiAgentEnv.INFO_VECTOR_SIZE + self.TRANSFORMER_DIM  # player_hand + info_vector + encoded history

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
            Concatenated tensor of shape ``(B, 32 + 38 + D)``.
        """
        player_hand: torch.Tensor = batch["obs"]["player_hand"]  # shape: (B, 32)
        info_vector: torch.Tensor = batch["obs"]["info_vector"]  # shape: (B, 38)
        action_history_flat: torch.Tensor = batch["obs"]["action_history"]  # shape: (B, 88)
        B: int = action_history_flat.shape[0]
        action_history: torch.Tensor = action_history_flat.reshape(B, -1, 2)  # (B, 44, 2)
        lengths: torch.Tensor = batch["obs"]["action_history_len"].reshape(-1).long()  # always (B,)

        history_encoded: torch.Tensor = self.history_encoder(action_history, lengths)  # (B, D)
        x: torch.Tensor = torch.cat([player_hand, info_vector, history_encoded], dim=-1)  # (B, 32 + 38 + D)
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


class TransformerHistoryEncoder(nn.Module):
    """Transformer encoder for the action history sequence.

    Embeds action and player ids, projects to model dimension, and
    processes with a Transformer encoder. Returns the representation
    at the last valid timestep. Handles empty sequences by returning zeros.

    Parameters
    ----------
    num_actions : int
        Number of distinct actions.
    emb_size_actions : int
        Embedding dimension for actions.
    emb_size_player : int
        Embedding dimension for player ids.
    transformer_dim : int
        Model dimension for the Transformer.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    """
    def __init__(self, num_actions: int, emb_size_actions: int, emb_size_player: int, transformer_dim: int, num_heads: int, num_layers: int) -> None:
        super().__init__()

        self.action_embed = nn.Embedding(num_actions + 1, emb_size_actions, padding_idx=0)
        self.player_embed = nn.Embedding(5, emb_size_player, padding_idx=0)
        self.input_proj = nn.Linear(emb_size_actions + emb_size_player, transformer_dim)

        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_dim: int = transformer_dim

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
            Encoded history of shape ``(B, transformer_dim)``.
        """
        B, T, _ = action_history.shape
        device: torch.device = action_history.device

        # Initialize with zeros - will be updated for non-empty sequences
        encoded: torch.Tensor = torch.zeros(B, self.output_dim, device=device)

        # Only process sequences with length > 0
        nonempty_mask = lengths > 0
        if nonempty_mask.any():
            nonempty_idx = nonempty_mask.nonzero(as_tuple=True)[0]
            nonempty_lengths = lengths[nonempty_idx]
            nonempty_history = action_history[nonempty_idx]  # (B', T, 2)

            action_ids: torch.Tensor = nonempty_history[:, :, 0].long() + 1
            player_ids: torch.Tensor = nonempty_history[:, :, 1].long() + 1

            action_emb: torch.Tensor = self.action_embed(action_ids)
            player_emb: torch.Tensor = self.player_embed(player_ids)
            input_seq: torch.Tensor = torch.cat([action_emb, player_emb], dim=-1)

            x: torch.Tensor = self.input_proj(input_seq)

            # Create padding mask for transformer
            pad_mask: torch.Tensor = torch.arange(T, device=device).expand(len(nonempty_idx), T) >= nonempty_lengths.unsqueeze(1)

            transformer_out: torch.Tensor = self.transformer(x, src_key_padding_mask=pad_mask)

            # Get last valid position for each sequence
            last_idx: torch.Tensor = (nonempty_lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.output_dim).long()
            pooled: torch.Tensor = transformer_out.gather(dim=1, index=last_idx).squeeze(1)

            encoded[nonempty_idx] = pooled

        return encoded
