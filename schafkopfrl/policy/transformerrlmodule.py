from typing import Any

from ray.rllib.core import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.torch_utils import FLOAT_MIN

from schafkopfrl.environment.utils import *
import torch.nn as nn
import torch


class TransformerRLModule(RLModule, ValueFunctionAPI, nn.Module):
    """Transformer-based RL module for Schafkopf with action masking.

    Encodes the action history with a Transformer encoder, concatenates
    it with the player hand, then passes through separate policy and
    value heads.
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


        self.history_encoder = TransformerHistoryEncoder(
            num_actions=self.NUM_ACTIONS,
            emb_size_actions=self.EMB_SIZE_ACTIONS,
            emb_size_player=self.EMB_SIZE_PLAYER,
            transformer_dim=self.model_config.get("transformer_dim", 128),
            num_heads=self.model_config.get("num_attention_heads", 4),
            num_layers=self.model_config.get("transformer_num_layers", 2),
            max_seq_len=self.model_config.get("max_action_history_len", 44),
        )

        input_dim: int = 32 + self.LSTM_HIDDEN  # player_hand + encoded history

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

        self.dev: torch.device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() == 1 else "cpu")
        self.to(self.dev)

    def _get_input_tensor(self, batch: dict[str, Any]) -> torch.Tensor:
        """Build the concatenated input tensor from a batch of observations.

        Parameters
        ----------
        batch : dict[str, Any]
            Batch dict containing ``obs`` with observation components.

        Returns
        -------
        torch.Tensor
            Concatenated tensor of shape ``(B, 32 + H)``.
        """
        player_hand: torch.Tensor = batch["obs"]["player_hand"].to(self.dev)  # shape: (B, 32)
        action_history: torch.Tensor = batch["obs"]["action_history"].to(self.dev)  # shape: (B, T, 2)
        lengths: torch.Tensor = batch["obs"]["action_history_len"].to(self.dev)  # shape: (B,)

        history_encoded: torch.Tensor = self.history_encoder(action_history, lengths)  # (B, D)
        x: torch.Tensor = torch.cat([player_hand, history_encoded], dim=-1)  # (B, 32 + H)
        return x

    @override(RLModule)
    def _forward(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, torch.Tensor]:
        x: torch.Tensor = self._get_input_tensor(batch)
        logits: torch.Tensor = self._policy_net(x)

        action_mask: torch.Tensor = batch["obs"]["action_mask"].to(self.dev)
        inf_mask: torch.Tensor = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits: torch.Tensor = logits + inf_mask

        return {Columns.ACTION_DIST_INPUTS: masked_logits}

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        x = self._get_input_tensor(batch)
        return self._value_net(x).squeeze(1)

    @override(RLModule)
    def get_state(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.state_dict()

    @override(RLModule)
    def set_state(self, state: dict[str, torch.Tensor], **kwargs: Any) -> None:
        self.load_state_dict(state)

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
    at the last valid timestep.

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
    max_seq_len : int
        Maximum sequence length.
    """
    def __init__(self, num_actions: int, emb_size_actions: int, emb_size_player: int, transformer_dim: int, num_heads: int, num_layers: int, max_seq_len: int) -> None:
        super().__init__()

        self.action_embed = nn.Embedding(num_actions + 1, emb_size_actions, padding_idx=0)
        self.player_embed = nn.Embedding(5, emb_size_player, padding_idx=0)
        self.input_proj = nn.Linear(emb_size_actions + emb_size_player, transformer_dim)

        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.max_seq_len: int = max_seq_len
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
        # action_history: (B, T, 2)
        B, T, _ = action_history.shape
        device: torch.device = action_history.device

        action_ids: torch.Tensor = action_history[:, :, 0] + 1  # pad -1 → 0
        player_ids: torch.Tensor = action_history[:, :, 1] + 1

        action_emb: torch.Tensor = self.action_embed(action_ids)  # (B, T, E1)
        player_emb: torch.Tensor = self.player_embed(player_ids)  # (B, T, E2)
        input_seq: torch.Tensor = torch.cat([action_emb, player_emb], dim=-1)  # (B, T, E1+E2)

        x: torch.Tensor = self.input_proj(input_seq)  # (B, T, D)

        # Create attention mask (True = pad, False = valid)
        # Shape must be (B, T)
        pad_mask: torch.Tensor = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)

        transformer_out: torch.Tensor = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, T, D)

        # Option A: Take last valid step
        last_idx: torch.Tensor = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.output_dim).long()  # (B, 1, D)
        pooled: torch.Tensor = transformer_out.gather(dim=1, index=last_idx).squeeze(1)  # (B, D)

        return pooled