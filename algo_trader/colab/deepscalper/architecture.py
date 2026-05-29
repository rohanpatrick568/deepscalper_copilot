"""
colab/deepscalper/architecture.py — DeepScalper Network (1:1 paper replica).

Implements the full architecture from:
  "DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture
   Fleeting Intraday Trading Opportunities"  (CIKM '22, Sun et al.)

Four building blocks (Figure 3):

  (a) MicroEncoder    — Two-stream GRU over LOB sequence + private-state sequence.
                        LOB stream  : GRU(lob_dim,  gru_hidden) → last hidden h_lob
                        Priv stream : GRU(priv_dim, gru_hidden) → last hidden h_priv
                        micro_embed = concat(h_lob, h_priv)   shape (B, 2*gru_hidden)

  (b) MacroEncoder    — Single MLP over current-bar OHLCV + technical indicators.
                        Input  : (B, MACRO_DIM)  — z_open…z_d_30 (Table 2)
                        Output : (B, macro_embed)

  Market embedding    = concat(micro_embed, macro_embed)        shape (B, embed_dim)

  (c) BranchingDuelingQNetwork (BDQ) — Figure 3(d)
        Value head    : FC → V(s)                                shape (B, 1)
        Dir advantage : FC → A_dir(s, a_dir)                     shape (B, N_DIR)
        Size advantage: FC → A_size(s, a_size)                   shape (B, N_SIZE)
        Q_dir  = V + A_dir  − mean(A_dir)
        Q_size = V + A_size − mean(A_size)

      During inference : a_dir  = argmax Q_dir,  a_size = argmax Q_size
      BDQ loss         : (1/2) Σ_{d∈{dir,size}} E[(y_d − Q_d)²]   (IS-weighted)

    (d) TradeMaster parity path uses Q-value heads only (no auxiliary vol head).

Observation split (must match environment.py / state_builder.py):
    lob   : (B, seq_len, LOB_DIM=5)   — microstructure sequence
  priv  : (B, seq_len, PRIV_DIM=2)  — private-state sequence (position, P&L)
  macro : (B, MACRO_DIM=11)         — current-bar macro features (no time dim)

Action space (BDQ):
    direction : N_DIR=3    (0=SHORT, 1=FLAT, 2=LONG)
    size      : N_SIZE=4
"""

import torch
import torch.nn as nn

# Default dimension constants (kept in sync with config.py)
_MACRO_DIM  = 11
_LOB_DIM    = 5
_PRIV_DIM   = 2
_N_DIR      = 3
_N_SIZE     = 4
_GRU_HIDDEN = 128
_MACRO_EMB  = 64
_FC_HIDDEN  = 128


# ---------------------------------------------------------------------------
# (a) Micro-level encoder — Figure 3(a)
# ---------------------------------------------------------------------------

class MicroEncoder(nn.Module):
    """Two-stream GRU encoder for micro-level market information.

    Stream 1 (LOB)  : encodes the intrabar microstructure (LOB proxy) sequence.
    Stream 2 (Priv) : encodes the trader's private-state sequence.

    Both streams produce one GRU layer each.  The last hidden states are
    concatenated to form the micro-level embedding e^i_t.

    Args:
        lob_dim    : Feature dim per timestep for the LOB stream (default 5).
        priv_dim   : Feature dim per timestep for the private-state stream (default 2).
        gru_hidden : GRU hidden size for each stream (paper searches [32,64,128]).
    """

    def __init__(
        self,
        lob_dim:    int = _LOB_DIM,
        priv_dim:   int = _PRIV_DIM,
        gru_hidden: int = _GRU_HIDDEN,
    ) -> None:
        super().__init__()
        self.gru_lob  = nn.GRU(lob_dim,  gru_hidden, num_layers=1, batch_first=True)
        self.gru_priv = nn.GRU(priv_dim, gru_hidden, num_layers=1, batch_first=True)

    def forward(self, lob: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lob  : (batch, seq_len, lob_dim)
            priv : (batch, seq_len, priv_dim)
        Returns:
            Micro embedding of shape (batch, 2 * gru_hidden).
        """
        _, h_lob  = self.gru_lob(lob)    # h_lob:  (1, batch, gru_hidden)
        _, h_priv = self.gru_priv(priv)  # h_priv: (1, batch, gru_hidden)
        return torch.cat([h_lob.squeeze(0), h_priv.squeeze(0)], dim=-1)


# ---------------------------------------------------------------------------
# (b) Macro-level encoder — Figure 3(b)
# ---------------------------------------------------------------------------

class MacroEncoder(nn.Module):
    """MLP encoder for macro-level market information.

    Takes the current bar's OHLCV-derived + moving-average feature vector
    (no recurrence needed — temporal history is already encoded via the
    z_d_5…z_d_30 moving-average spread features).

    Architecture: Linear(macro_dim, hidden) → ReLU → Linear(hidden, embed_dim) → ReLU

    Args:
        macro_dim : Number of macro features (MACRO_DIM = 11).
        hidden    : MLP hidden layer width (paper searches [32,64,128]).
        embed_dim : Output embedding dimension.
    """

    def __init__(
        self,
        macro_dim: int = _MACRO_DIM,
        hidden:    int = _FC_HIDDEN,
        embed_dim: int = _MACRO_EMB,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(macro_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, macro: torch.Tensor) -> torch.Tensor:
        """
        Args:
            macro : (batch, macro_dim)
        Returns:
            Macro embedding of shape (batch, embed_dim).
        """
        return self.net(macro)


# ---------------------------------------------------------------------------
# (c) Risk-aware auxiliary task — Section 4.4
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Full DeepScalper network: BDQ
# ---------------------------------------------------------------------------

class DeepScalperNet(nn.Module):
    """DeepScalper complete network (BDQ).

    Combines the macro and micro encoders into a market embedding, then
    branches into:
      - Shared value head  V(s)
      - Direction advantage head  A_dir(s, a_dir)
      - Size advantage head       A_size(s, a_size)

    Q-value computation (standard dueling aggregation per branch):
        Q_dir  = V(s) + A_dir(s,a)  − mean_{a'} A_dir(s,a')
        Q_size = V(s) + A_size(s,a) − mean_{a'} A_size(s,a')

    Args:
        macro_dim   : MACRO_DIM (11).
        lob_dim     : LOB_DIM (5).
        priv_dim    : PRIV_DIM (2).
        gru_hidden  : GRU hidden size per stream.
        macro_embed : MacroEncoder output dimension.
        fc_hidden   : FC layer width in advantage / value heads.
        n_dir       : Number of direction actions (N_DIR = 3).
        n_size      : Number of size actions (N_SIZE = 4).
    """

    def __init__(
        self,
        macro_dim:   int = _MACRO_DIM,
        lob_dim:     int = _LOB_DIM,
        priv_dim:    int = _PRIV_DIM,
        gru_hidden:  int = _GRU_HIDDEN,
        macro_embed: int = _MACRO_EMB,
        fc_hidden:   int = _FC_HIDDEN,
        n_dir:       int = _N_DIR,
        n_size:      int = _N_SIZE,
    ) -> None:
        super().__init__()

        self.n_dir  = n_dir
        self.n_size = n_size

        self.micro_enc = MicroEncoder(lob_dim, priv_dim, gru_hidden)
        self.macro_enc = MacroEncoder(macro_dim, fc_hidden, macro_embed)

        # Total market embedding dimension
        embed_dim = 2 * gru_hidden + macro_embed

        # Shared value stream  V(s)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, 1),
        )

        # Direction advantage stream  A_dir(s, a_dir)
        self.adv_dir = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, n_dir),
        )

        # Size advantage stream  A_size(s, a_size)
        self.adv_size = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, n_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def embed(
        self,
        lob:   torch.Tensor,
        priv:  torch.Tensor,
        macro: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the full market embedding from the three observation streams.

        Args:
            lob   : (batch, seq_len, lob_dim)
            priv  : (batch, seq_len, priv_dim)
            macro : (batch, macro_dim)
        Returns:
            Market embedding of shape (batch, embed_dim).
        """
        e_micro = self.micro_enc(lob, priv)   # (batch, 2*gru_hidden)
        e_macro = self.macro_enc(macro)        # (batch, macro_embed)
        return torch.cat([e_micro, e_macro], dim=-1)

    def forward(
        self,
        lob:   torch.Tensor,
        priv:  torch.Tensor,
        macro: torch.Tensor,
    ):
        """Forward pass.

        Args:
            lob   : (batch, seq_len, lob_dim)
            priv  : (batch, seq_len, priv_dim)
            macro : (batch, macro_dim)

        Returns:
            q_dir  : (batch, n_dir)   — Q-values for each direction action
            q_size : (batch, n_size)  — Q-values for each size action
        """
        e = self.embed(lob, priv, macro)   # (batch, embed_dim)

        value  = self.value_head(e)        # (batch, 1)
        a_dir  = self.adv_dir(e)           # (batch, n_dir)
        a_size = self.adv_size(e)          # (batch, n_size)

        # Dueling combination (per branch)
        q_dir  = value + (a_dir  - a_dir.mean(dim=1,  keepdim=True))
        q_size = value + (a_size - a_size.mean(dim=1, keepdim=True))

        return q_dir, q_size


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

# The old class name is kept so that any existing checkpoints or imports
# don't break immediately; it simply instantiates DeepScalperNet.
class DuelingQNetwork(DeepScalperNet):
    """Deprecated — use DeepScalperNet.  Kept for backward compatibility."""

    def __init__(self, lookback_bars=60, input_dim=11, action_dim=3,
                 hidden_size=128, fc_size=128, dropout_rate=0.0, **kwargs):
        super().__init__(
            macro_dim=input_dim,
            fc_hidden=fc_size,
            gru_hidden=hidden_size,
            n_dir=action_dim,
        )



