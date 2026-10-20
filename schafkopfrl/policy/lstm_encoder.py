import torch.nn as nn
import torch


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