"""
execution/state_builder.py — Live Bar Data → DeepScalper Observation Dict.

Bridge between Lumibot's live bar DataFrame and the DeepScalperNet model.

Returns a dict observation matching the format expected by the network:
    'lob'   : torch.FloatTensor  shape (1, seq_len, LOB_DIM=4)   — micro features
    'priv'  : torch.FloatTensor  shape (1, seq_len, PRIV_DIM=2)  — private state
    'macro' : torch.FloatTensor  shape (1, MACRO_DIM=11)          — macro features

Feature engineering is delegated to the shared utils module so that the
training and inference pipelines are identical.

Usage:
    from execution.state_builder import build_observation
    obs = build_observation(bars_df, position=1, unrealized_pnl_pct=0.003)
    # obs['lob']  : (1, LOOKBACK_BARS, 4)
    # obs['priv'] : (1, LOOKBACK_BARS, 2)
    # obs['macro']: (1, 11)
"""

import logging
from typing import Optional

import numpy as np
import torch

from config import LOOKBACK_BARS, MACRO_DIM, LOB_DIM, PRIV_DIM
from colab.deepscalper.utils import compute_macro_features, compute_micro_features

logger = logging.getLogger(__name__)


def build_observation(
    bars,
    position:            int   = 0,
    unrealized_pnl_pct:  float = 0.0,
    lob_override: Optional[np.ndarray] = None,
    device:              str   = "cpu",
) -> dict:
    """Build a DeepScalperNet observation dict from a raw OHLCV bar DataFrame.

    Args:
        bars               : pandas DataFrame with OHLCV columns and DatetimeIndex.
                             Must have at least LOOKBACK_BARS rows.
        position           : Current position flag: 1 long, 0 flat.
        unrealized_pnl_pct : Unrealized P&L as a fraction of notional.
        lob_override       : Optional micro-feature override array with shape
                     (n, LOB_DIM) or (1, LOB_DIM). Used by live strategy
                     to inject real/proxy LOB features directly.
        device             : PyTorch device string.

    Returns:
        Dict with keys 'lob', 'priv', 'macro' containing float32 tensors with a
        batch dimension of 1.
    """
    seq_len = LOOKBACK_BARS

    # ---- Compute feature matrices ----
    macro_arr = compute_macro_features(bars)   # (n, 11)

    if lob_override is None:
        lob_arr = compute_micro_features(bars, use_proxy=True)   # (n, LOB_DIM)
    else:
        lob_arr = np.asarray(lob_override, dtype=np.float32)
        if lob_arr.ndim == 1:
            lob_arr = lob_arr.reshape(1, -1)
        if lob_arr.shape[1] != LOB_DIM:
            raise ValueError(f"lob_override must have shape (*, {LOB_DIM}), got {lob_arr.shape}")

        n_bars = len(bars)
        if lob_arr.shape[0] == 1:
            lob_arr = np.repeat(lob_arr, n_bars, axis=0)
        elif lob_arr.shape[0] > n_bars:
            lob_arr = lob_arr[-n_bars:]
        elif lob_arr.shape[0] < n_bars:
            pad = np.zeros((n_bars - lob_arr.shape[0], LOB_DIM), dtype=np.float32)
            lob_arr = np.vstack([pad, lob_arr])

    # Take the last seq_len rows
    macro_seq = macro_arr[-seq_len:]  # (seq_len, 11)
    lob_seq   = lob_arr[-seq_len:]    # (seq_len, LOB_DIM)

    # Pad to seq_len if bars are insufficient
    if macro_seq.shape[0] < seq_len:
        pad_len  = seq_len - macro_seq.shape[0]
        macro_seq = np.vstack([np.zeros((pad_len, MACRO_DIM), dtype=np.float32), macro_seq])
        lob_seq   = np.vstack([np.zeros((pad_len, LOB_DIM),   dtype=np.float32), lob_seq])

    # ---- Private state sequence ----
    # For live inference we only have the current state; replicate it across the window.
    pos_flag  = 1.0 if position != 0 else 0.0
    pnl_clamp = float(np.clip(unrealized_pnl_pct, -0.5, 0.5))
    priv_vec  = np.array([pos_flag, pnl_clamp], dtype=np.float32)
    priv_seq  = np.tile(priv_vec, (seq_len, 1))   # (seq_len, 2)

    # ---- Macro: use only the current bar's features (no time dimension) ----
    macro_current = macro_seq[-1]  # (11,)

    # ---- Convert to tensors with batch dim ----
    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)

    return {
        'lob':   _t(lob_seq),         # (1, LOOKBACK_BARS, LOB_DIM)
        'priv':  _t(priv_seq),        # (1, LOOKBACK_BARS, PRIV_DIM)
        'macro': _t(macro_current),   # (1, 11)
    }


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

def build_state_tensor(bars, device: str = "cpu") -> torch.Tensor:
    """Deprecated — use build_observation().

    Returns the old flat macro-sequence tensor for any legacy call sites.
    Shape: (1, LOOKBACK_BARS, MACRO_DIM).
    """
    import warnings
    warnings.warn(
        "build_state_tensor() is deprecated; use build_observation().",
        DeprecationWarning,
        stacklevel=2,
    )
    macro_arr = compute_macro_features(bars)
    seq = macro_arr[-LOOKBACK_BARS:]
    if seq.shape[0] < LOOKBACK_BARS:
        pad = np.zeros((LOOKBACK_BARS - seq.shape[0], MACRO_DIM), dtype=np.float32)
        seq = np.vstack([pad, seq])
    return torch.tensor(seq[None], dtype=torch.float32, device=device)
