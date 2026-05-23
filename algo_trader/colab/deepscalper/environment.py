"""
colab/deepscalper/environment.py — DeepScalper Intraday Trading Environment.

Faithful implementation of the trading environment described in:
  "DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture
   Fleeting Intraday Trading Opportunities"  (CIKM '22, Sun et al.)

Key paper-faithful design choices:

Observation Space (Dict):
    'lob'   : Box(seq_len, LOB_DIM=5)    — micro sequence (LOB proxy)
    'priv'  : Box(seq_len, PRIV_DIM=2)   — private state: (position_flag, unrealized_pnl%)
    'macro' : Box(MACRO_DIM=11,)          — current bar macro features (Table 2)

Action Space: MultiDiscrete([N_DIR=3, N_SIZE=4])
    Direction : 0=HOLD  1=BUY  2=SELL
    Size      : 0=25%   1=50%  2=75%   3=100%  (of max_notional)

Reward:
    r_t = log(P_{t+1} / P_t) × position (mark-to-market P&L)
    Hindsight bonus (training only, applied in agent.store()):
        r_H = r_t + w × log(P_{t+h} / P_t) × position   (Section 4.2)

Vol target (info dict):
    info['vol_target'] = std(z_close) over lookback window — for the
    volatility auxiliary task (Section 4.4).

Episode lifecycle:
    • One episode = one full trading day (day-start to day-end).
    • The environment cycles through pre-computed days randomly during training.
    • Private state history is a deque of (position_flag, unrealized_pnl_pct)
      maintained over the lookback window.

Usage:
    env = ScalperEnv(
        lob_features   = lob_arr,     # (n_bars, 5)
        macro_features = macro_arr,   # (n_bars, 11)
        close_prices   = close_arr,   # (n_bars,)
        day_starts     = [0, 390, …], # day boundary indices from compute_day_starts
        lookback_bars  = 60,
        max_notional   = 10_000.0,
        transaction_cost_pct = 0.001,
    )
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([1, 2]))
"""

import logging
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

# Size-branch → fraction of max_notional mapping
_SIZE_FRACTIONS = [0.25, 0.50, 0.75, 1.00]

# Action branch indices
_HOLD = 0
_BUY  = 1
_SELL = 2


class ScalperEnv(gym.Env):
    """DeepScalper intraday trading environment.

    Args:
        lob_features          : Pre-computed LOB/micro features  (n_bars, LOB_DIM).
        macro_features        : Pre-computed macro features       (n_bars, MACRO_DIM).
        close_prices          : Raw close prices array            (n_bars,).
        day_starts            : List of integer indices where each trading day begins.
        lookback_bars         : Number of bars in the observation window (T in the paper).
        max_notional          : Maximum position size in dollars.
        transaction_cost_pct  : One-way transaction cost as fraction of trade value.
        hindsight_horizon     : h — look-ahead bars for hindsight bonus (default 60).
        n_dir                 : Direction branch actions  (default 3).
        n_size                : Size branch actions       (default 4).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        lob_features:         np.ndarray,
        macro_features:       np.ndarray,
        close_prices:         np.ndarray,
        day_starts:           List[int],
        lookback_bars:        int   = 60,
        max_notional:         float = 10_000.0,
        transaction_cost_pct: float = 0.001,
        hindsight_horizon:    int   = 60,
        n_dir:                int   = 3,
        n_size:               int   = 4,
    ) -> None:
        super().__init__()

        self.lob_features   = np.asarray(lob_features,   dtype=np.float32)
        self.macro_features = np.asarray(macro_features, dtype=np.float32)
        self.close_prices   = np.asarray(close_prices,   dtype=np.float64)

        self.day_starts          = list(day_starts)
        self.lookback_bars       = lookback_bars
        self.max_notional        = max_notional
        self.transaction_cost_pct = transaction_cost_pct
        self.hindsight_horizon   = hindsight_horizon

        lob_dim   = self.lob_features.shape[1]
        macro_dim = self.macro_features.shape[1]
        priv_dim  = 2  # (position_flag, unrealized_pnl_pct)

        # ----------------------------------------------------------------
        # Observation space
        # ----------------------------------------------------------------
        self.observation_space = spaces.Dict({
            'lob':   spaces.Box(-np.inf, np.inf, shape=(lookback_bars, lob_dim),   dtype=np.float32),
            'priv':  spaces.Box(-np.inf, np.inf, shape=(lookback_bars, priv_dim),  dtype=np.float32),
            'macro': spaces.Box(-np.inf, np.inf, shape=(macro_dim,),               dtype=np.float32),
        })

        # Action space: MultiDiscrete([n_dir, n_size])
        self.action_space = spaces.MultiDiscrete([n_dir, n_size])

        # Episode state (reset in reset())
        self._day_idx:     int   = 0
        self._t:           int   = 0
        self._day_end:     int   = 0
        self._position:    int   = 0      # +1 long, -1 short, 0 flat
        self._entry_price: float = 0.0
        self._priv_history: deque = deque(maxlen=lookback_bars)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self._day_idx = random.randrange(len(self.day_starts))
        self._t = self.day_starts[self._day_idx]
        if self._day_idx + 1 < len(self.day_starts):
            self._day_end = self.day_starts[self._day_idx + 1] - 1
        else:
            self._day_end = len(self.close_prices) - 1

        # Ensure we have at least lookback_bars + 1 bars for a step
        if self._day_end - self._t < self.lookback_bars + 1:
            return self.reset(seed=seed, options=options)

        self._position    = 0
        self._entry_price = 0.0
        self._priv_history = deque(
            [np.zeros(2, dtype=np.float32)] * self.lookback_bars,
            maxlen=self.lookback_bars,
        )

        # Advance t to lookback_bars so we have a full window
        self._t = self.day_starts[self._day_idx] + self.lookback_bars - 1

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one bar step.

        Args:
            action : np.ndarray of shape (2,) — [dir_action, size_action].

        Returns:
            obs, reward, terminated, truncated, info
        """
        dir_action  = int(action[0])
        size_action = int(action[1])
        fraction    = _SIZE_FRACTIONS[size_action]

        current_price = float(self.close_prices[self._t])
        next_t        = self._t + 1
        next_price    = float(self.close_prices[min(next_t, self._day_end)])

        cost = 0.0
        prev_position = self._position

        # ---- Execute action ----
        if dir_action == _BUY and self._position <= 0:
            if self._position < 0:
                # Close short
                pnl = (self._entry_price - current_price) / (self._entry_price + 1e-10)
                cost += self.transaction_cost_pct
            self._position    = 1
            self._entry_price = current_price
            cost += self.transaction_cost_pct

        elif dir_action == _SELL and self._position >= 0:
            if self._position > 0:
                # Close long
                pnl = (current_price - self._entry_price) / (self._entry_price + 1e-10)
                cost += self.transaction_cost_pct
            self._position    = -1
            self._entry_price = current_price
            cost += self.transaction_cost_pct

        # HOLD: keep current position unchanged

        # ---- Compute step reward (log return × position) ----
        if self._position != 0 and current_price > 0:
            log_ret = float(np.log(next_price / current_price))
            reward  = log_ret * self._position - cost
        else:
            reward = -cost

        # ---- Unrealized P&L for private state ----
        if self._position != 0 and self._entry_price > 0:
            unreal_pnl = (
                (current_price - self._entry_price) / self._entry_price * self._position
            )
        else:
            unreal_pnl = 0.0

        # ---- Update private state history ----
        pos_flag = 1.0 if self._position != 0 else 0.0
        self._priv_history.append(
            np.array([pos_flag, float(np.clip(unreal_pnl, -0.5, 0.5))], dtype=np.float32)
        )

        # ---- Advance time ----
        self._t = next_t
        terminated = self._t >= self._day_end

        # ---- Volatility target for auxiliary task ----
        # std of z_close over lookback window (macro feature index 3)
        win_start  = max(0, self._t - self.lookback_bars)
        win_end    = self._t
        z_close_win = self.macro_features[win_start:win_end, 3]  # z_close
        vol_target  = float(z_close_win.std()) if len(z_close_win) > 1 else 0.0

        # ---- Hindsight future price (stored in info for agent to use) ----
        h_idx = min(self._t + self.hindsight_horizon, self._day_end)
        future_price = float(self.close_prices[h_idx])

        info = {
            'vol_target':    vol_target,
            'current_price': current_price,
            'future_price':  future_price,
            'position':      prev_position,
            'dir_action':    dir_action,
            'size_action':   size_action,
        }

        obs = self._get_obs()
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Observation assembly
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Assemble the current observation dict."""
        t = self._t
        win_start = max(0, t - self.lookback_bars + 1)
        win_end   = t + 1

        lob_seq  = self.lob_features[win_start:win_end]
        # Pad at the front if we don't yet have a full window
        if lob_seq.shape[0] < self.lookback_bars:
            pad = np.zeros(
                (self.lookback_bars - lob_seq.shape[0], lob_seq.shape[1]), dtype=np.float32
            )
            lob_seq = np.vstack([pad, lob_seq])

        priv_arr = np.array(list(self._priv_history), dtype=np.float32)  # (seq, 2)
        macro    = self.macro_features[t]  # (macro_dim,)

        return {
            'lob':   lob_seq.astype(np.float32),
            'priv':  priv_arr.astype(np.float32),
            'macro': macro.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Rendering (not required for training)
    # ------------------------------------------------------------------

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
