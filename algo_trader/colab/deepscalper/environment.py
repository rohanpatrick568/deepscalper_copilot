"""
colab/deepscalper/environment.py — DeepScalper Intraday Trading Environment.

Faithful implementation of the trading environment described in:
  "DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture
   Fleeting Intraday Trading Opportunities"  (CIKM '22, Sun et al.)

Key paper-faithful design choices:

Observation Space (Dict):
    'lob'   : Box(seq_len, LOB_DIM=4)    — micro sequence (dual-mode: proxy or real LOB)
    'priv'  : Box(seq_len, PRIV_DIM=2)   — private state: (position_flag, unrealized_pnl%)
    'macro' : Box(MACRO_DIM=11,)          — current bar macro features (Table 2)

Action Space: Discrete(3)
    0 = SHORT
    1 = FLAT
    2 = LONG

Reward (V2 CHANGE — TradeMaster-aligned):
    r_t = log_return × position                     (immediate mark-to-market P&L)
          + ω × max_future_log_return              (hindsight bonus, training-only)
          - α × rolling_return_std                 (risk-aware auxiliary task)
    Where ω = 0.2 (HINDSIGHT_WEIGHT) and h = 10 (HINDSIGHT_HORIZON)

Transaction cost (V2 CHANGE):
    Volume-tiered taker fee in the range 0.0012–0.0025 (12–25 bps per side).

Episode lifecycle:
    • One episode = one full UTC calendar day (midnight to midnight).
    • The environment cycles through pre-computed UTC days randomly during training.
    • Private state history is a deque of (position_flag, unrealized_pnl_pct)
      maintained over the lookback window.

Usage:
    env = ScalperEnv(
        lob_features   = lob_arr,     # (n_bars, 4)  <- V2: 4 features
        macro_features = macro_arr,   # (n_bars, 11)
        close_prices   = close_arr,   # (n_bars,)
        day_starts     = [0, 1440, …], # UTC day boundary indices
        lookback_bars  = 10,           # V2: 10 bars
        max_notional   = 10_000.0,
        transaction_cost_pct = 0.0018, # Optional override (18 bps example)
    )
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(1)  # V2: int, not array
"""

import logging
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

# 3-action semantics
_SHORT = 0
_FLAT = 1
_LONG = 2

# Alpaca crypto taker fee guidance (volume-tiered): 12–25 bps.
MIN_TAKER_FEE_PCT = 0.0012
MAX_TAKER_FEE_PCT = 0.0025
DEFAULT_TAKER_FEE_PCT = 0.0018

# Approximate 30-day notional USD tiers -> taker fee.
_TAKER_FEE_TIERS = (
    (100_000_000.0, 0.0012),
    (50_000_000.0, 0.0015),
    (10_000_000.0, 0.0018),
    (0.0, 0.0025),
)


class ScalperEnv(gym.Env):
    """DeepScalper trading environment with SHORT/FLAT/LONG actions.

    Args:
        lob_features          : Pre-computed LOB/micro features  (n_bars, LOB_DIM=4).
        macro_features        : Pre-computed macro features       (n_bars, MACRO_DIM=11).
        close_prices          : Raw close prices array            (n_bars,).
        day_starts            : UTC day boundary indices from compute_day_starts().
        random_day_reset      : If True, reset samples random days (training mode).
                    If False, reset iterates days deterministically.
        lookback_bars         : Number of bars in the observation window (10 in V2).
        max_notional          : Maximum position size in dollars.
        transaction_cost_pct  : Optional one-way transaction cost override.
                    If None, a volume-tiered taker fee is inferred.
        thirty_day_volume_usd : Optional 30-day notional used for fee-tier selection.
        hindsight_horizon     : h — look-ahead bars for hindsight bonus (10 in V2).
        hindsight_weight      : ω — hindsight bonus coefficient (0.2 in V2).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        lob_features:         np.ndarray,
        macro_features:       np.ndarray,
        close_prices:         np.ndarray,
        day_starts:           List[int],
        random_day_reset:     bool  = True,
        lookback_bars:        int   = 10,       # V2 CHANGE: was 60
        max_notional:         float = 10_000.0,
        transaction_cost_pct: Optional[float] = None,
        thirty_day_volume_usd: Optional[float] = None,
        hindsight_horizon:    int   = 10,       # V2 CHANGE: was 60 (TradeMaster: 5 bars)
        hindsight_weight:     float = 0.2,      # V2 CHANGE: was 0.01 (TradeMaster default)
    ) -> None:
        super().__init__()

        self.lob_features   = np.asarray(lob_features,   dtype=np.float32)
        self.macro_features = np.asarray(macro_features, dtype=np.float32)
        self.close_prices   = np.asarray(close_prices,   dtype=np.float64)

        self.day_starts          = list(day_starts)
        self.random_day_reset    = bool(random_day_reset)
        self._sequential_day_cursor = 0
        self.lookback_bars       = lookback_bars
        self.max_notional        = max_notional
        self.transaction_cost_pct = self._resolve_transaction_cost_pct(
            transaction_cost_pct,
            thirty_day_volume_usd,
        )
        self.hindsight_horizon   = hindsight_horizon
        self.hindsight_weight    = hindsight_weight

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

        self.action_space = spaces.Discrete(3)

        # Episode state (reset in reset())
        self._day_idx:      int   = 0
        self._t:            int   = 0
        self._day_end:      int   = 0
        self._position:     int   = 0      # -1=short, 0=flat, +1=long
        self._entry_price:  float = 0.0
        self._returns_history: deque = deque(maxlen=100)
        self._priv_history: deque = deque(maxlen=lookback_bars)

    @staticmethod
    def _resolve_transaction_cost_pct(
        transaction_cost_pct: Optional[float],
        thirty_day_volume_usd: Optional[float],
    ) -> float:
        """Resolve one-way taker fee from explicit override or volume tier.

        If `transaction_cost_pct` is provided, use it directly after clamping to
        the expected 12–25 bps envelope. Otherwise infer from 30-day notional.
        """
        if transaction_cost_pct is not None:
            return float(np.clip(transaction_cost_pct, MIN_TAKER_FEE_PCT, MAX_TAKER_FEE_PCT))

        if thirty_day_volume_usd is None:
            return DEFAULT_TAKER_FEE_PCT

        volume = max(0.0, float(thirty_day_volume_usd))
        for threshold, fee in _TAKER_FEE_TIERS:
            if volume >= threshold:
                return fee

        return MAX_TAKER_FEE_PCT

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        options = options or {}

        requested_day_idx = options.get("day_idx")
        random_day_reset = bool(options.get("random_day_reset", self.random_day_reset))

        if requested_day_idx is not None:
            self._day_idx = int(requested_day_idx) % len(self.day_starts)
        elif random_day_reset:
            self._day_idx = random.randrange(len(self.day_starts))
        else:
            self._day_idx = self._sequential_day_cursor % len(self.day_starts)
            self._sequential_day_cursor += 1

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
        self._returns_history.clear()
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
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one bar step.

        Action is a single int (0=SHORT, 1=FLAT, 2=LONG).

        Args:
            action : int — 0=SHORT, 1=FLAT, 2=LONG.

        Returns:
            obs, reward, terminated, truncated, info
        """
        action = int(action)
        current_price = float(self.close_prices[self._t])
        next_t        = self._t + 1
        next_price    = float(self.close_prices[min(next_t, self._day_end)])

        prev_position = self._position

        if action == _SHORT:
            new_position = -1
        elif action == _FLAT:
            new_position = 0
        elif action == _LONG:
            new_position = 1
        else:
            raise ValueError(f"Invalid action {action}")

        turnover = abs(new_position - prev_position)
        trade_occurred = turnover > 0
        transaction_cost = turnover * self.transaction_cost_pct

        if current_price > 0:
            base_log_ret = float(np.log(next_price / current_price))
        else:
            base_log_ret = 0.0

        if prev_position == 1:
            log_ret = base_log_ret
        elif prev_position == -1:
            log_ret = -base_log_ret
        else:
            log_ret = 0.0

        immediate_reward = log_ret - transaction_cost

        # Update position and entry price
        if trade_occurred:
            if new_position in (_LONG, _SHORT):
                self._entry_price = current_price
            else:
                self._entry_price = 0.0
        self._position = new_position

        # Track returns for risk penalty
        self._returns_history.append(log_ret)

        # V2 CHANGE: Full reward with hindsight bonus + risk penalty
        reward = self._compute_reward(immediate_reward, current_price, prev_position)

        # Unrealized P&L for private state
        if self._position == 1 and self._entry_price > 0:
            unreal_pnl = (current_price - self._entry_price) / self._entry_price
        elif self._position == -1 and self._entry_price > 0:
            unreal_pnl = (self._entry_price - current_price) / self._entry_price
        else:
            unreal_pnl = 0.0

        # Update private state history
        self._priv_history.append(
            np.array([float(self._position), float(np.clip(unreal_pnl, -0.5, 0.5))],
                     dtype=np.float32)
        )

        # Advance time
        self._t = next_t
        terminated = self._t >= self._day_end

        # Volatility target for auxiliary task (std of z_close over lookback)
        win_start  = max(0, self._t - self.lookback_bars)
        z_close_win = self.macro_features[win_start:self._t, 3]
        vol_target  = float(z_close_win.std()) if len(z_close_win) > 1 else 0.0

        info = {
            'vol_target':    vol_target,
            'current_price': current_price,
            'position':      self._position,
            'log_return':    log_ret,
        }

        obs = self._get_obs()
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # V2 CHANGE: DeepScalper reward function (TradeMaster-aligned)
    # ------------------------------------------------------------------

    def _compute_reward(self, immediate: float, current_price: float, prev_position: int) -> float:
        """DeepScalper reward with hindsight bonus + risk-aware penalty.

        Formula (Section 3.3 of the paper, TradeMaster-aligned parameters):
            r_t = immediate_log_return
                  + ω × best_future_log_return  (hindsight — training-only oracle)
                  - α × rolling_return_std       (risk-aware auxiliary task)

        Where:
            ω = hindsight_weight = 0.2  (TradeMaster: future_weights=0.2)
            h = hindsight_horizon = 10  (TradeMaster: forward_num_day=5)
            α = 0.01                    (keeps risk signal as regulariser)

        CRITICAL: The hindsight bonus uses future prices accessible only in
        training. It is NEVER used during inference. It acts as a coaching
        signal that accelerates learning without introducing lookahead bias.

        Args:
            immediate:     Already-computed log_return minus transaction cost.
            current_price: Price at the current bar.
            prev_position: Position held before this step (0=flat, 1=long).

        Returns:
            Scalar total reward.
        """
        # Component 1: Immediate return (already computed in step())
        total = immediate

        # Component 2: Hindsight bonus (training oracle)
        if prev_position == 1:
            future_end = min(self._t + self.hindsight_horizon, self._day_end)
            if future_end > self._t:
                future_prices = self.close_prices[self._t:future_end]
                best_future = float(np.max(
                    np.log(future_prices / (current_price + 1e-10) + 1e-10)
                ))
                total += self.hindsight_weight * max(best_future, 0.0)
        elif prev_position == -1:
            future_end = min(self._t + self.hindsight_horizon, self._day_end)
            if future_end > self._t:
                future_prices = self.close_prices[self._t:future_end]
                best_future = float(np.max(
                    np.log((current_price + 1e-10) / (future_prices + 1e-10) + 1e-10)
                ))
                total += self.hindsight_weight * max(best_future, 0.0)

        # Component 3: Risk-aware auxiliary task (penalise return variance)
        if len(self._returns_history) >= 10:
            recent = np.array(list(self._returns_history)[-20:])
            total -= 0.01 * float(np.std(recent))

        return total

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
