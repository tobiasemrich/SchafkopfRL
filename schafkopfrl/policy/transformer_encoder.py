import torch.nn as nn
import torch


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
