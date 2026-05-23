"""
colab/deepscalper/architecture.py — DeepScalper Dueling Q-Network.

Implements the DuelingQNetwork with:
  • 2-layer LSTM encoder (hidden_size=128) for temporal sequence processing.
    LSTM was chosen over CNN because it naturally captures long-range
    temporal dependencies in 1-minute bar sequences without requiring
    manual kernel tuning for different time scales.
  • Temporal attention layer that learns to weight recent bars more heavily.
  • LayerNorm applied to the LSTM output for training stability.
  • Dueling head: separate Value stream V(s) and Advantage stream A(s,a).
  • Output: Q(s,a) = V(s) + A(s,a) − mean(A(s,a))

Input shape:  (batch_size, LOOKBACK_BARS, INPUT_DIM)  e.g. (64, 60, 11)
Output shape: (batch_size, ACTION_DIM)                 e.g. (64, 3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Learnable additive attention over the LSTM output sequence.

    Produces a context vector by computing a weighted sum of all hidden states,
    where the weights are learned via a single linear projection + softmax.

    Args:
        hidden_size: Dimensionality of each LSTM hidden state.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        # Projects each hidden state to a scalar score
        self.score_proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Compute the attended context vector.

        Args:
            lstm_out: Tensor of shape (batch, seq_len, hidden_size) —
                      output from all LSTM time-steps.

        Returns:
            Context tensor of shape (batch, hidden_size).
        """
        # scores: (batch, seq_len, 1)
        scores = self.score_proj(lstm_out)
        # weights: (batch, seq_len, 1)  — softmax over the time dimension
        weights = F.softmax(scores, dim=1)
        # context: (batch, hidden_size)  — weighted sum over time-steps
        context = (weights * lstm_out).sum(dim=1)
        return context


class DuelingQNetwork(nn.Module):
    """DeepScalper Dueling Q-Network with Temporal Attention.

    Architecture overview:
        Input → 2-layer LSTM → LayerNorm → TemporalAttention
              ├─→ Value stream:     FC(256) → ReLU → Dropout → FC(128) → ReLU → Dropout → FC(1)
              └─→ Advantage stream: FC(256) → ReLU → Dropout → FC(128) → ReLU → Dropout → FC(ACTION_DIM)
        Output: Q(s,a) = V(s) + A(s,a) − mean(A(s,a))

    Args:
        lookback_bars: Sequence length (number of 1-min bars in state).
        input_dim: Number of features per bar (INPUT_DIM).
        action_dim: Number of discrete actions (ACTION_DIM = 3).
        hidden_size: LSTM hidden state size.
        fc_size: Width of the first FC layer in each dueling stream.
        dropout_rate: Dropout probability applied after each FC activation.
    """

    def __init__(
        self,
        lookback_bars: int,
        input_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        fc_size: int = 256,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.lookback_bars = lookback_bars
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # ----------------------------------------------------------------
        # Encoder: 2-layer LSTM
        # dropout inside LSTM only applies between layers (not after last)
        # ----------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,   # Applied between LSTM layers
        )

        # LayerNorm applied over the feature dimension of LSTM outputs
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Temporal attention: collapses (batch, seq_len, hidden) → (batch, hidden)
        self.attention = TemporalAttention(hidden_size)

        # ----------------------------------------------------------------
        # Value stream: V(s) — scalar estimate of state value
        # ----------------------------------------------------------------
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size, fc_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size // 2, 1),
        )

        # ----------------------------------------------------------------
        # Advantage stream: A(s, a) — relative advantage of each action
        # ----------------------------------------------------------------
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size, fc_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size // 2, action_dim),
        )

        # Initialise weights for stable early training
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform init to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for each action given the input state sequence.

        Args:
            x: Input tensor of shape (batch_size, lookback_bars, input_dim).

        Returns:
            Q-value tensor of shape (batch_size, action_dim).
        """
        # LSTM encoding — lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # LayerNorm for training stability
        lstm_out = self.layer_norm(lstm_out)

        # Temporal attention — context: (batch, hidden_size)
        context = self.attention(lstm_out)

        # Dueling streams
        value = self.value_stream(context)           # (batch, 1)
        advantage = self.advantage_stream(context)   # (batch, action_dim)

        # Dueling combination formula: Q = V + (A − mean(A))
        # Subtracting the mean advantage stabilises learning by removing the
        # identifiability problem between V and A.
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
