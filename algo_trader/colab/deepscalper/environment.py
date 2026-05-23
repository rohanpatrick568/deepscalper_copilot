"""
colab/deepscalper/environment.py — Custom Gym Environment for DeepScalper.

A gymnasium.Env subclass that simulates intraday trading on 1-minute OHLCV data.

Episode semantics:
  • One episode = one full trading day (up to 390 1-minute bars).
  • Each reset() picks a random trading day from the training dataset.
  • Observation: last LOOKBACK_BARS bars of normalised features.
  • Action space: Discrete(3)  — 0=HOLD, 1=BUY, 2=SELL.
  • Position tracking: flat (0) or long (+1).  No shorting supported.
  • No-trade buffer: BUY/SELL actions during the first and last 15 minutes
    are overridden to HOLD to match live circuit-breaker behaviour.

Reward function:
  • On HOLD or same-position actions: r = 0.
  • On closing (SELL from long): r = log(exit_price / entry_price) − λ × 2.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from colab.deepscalper.utils import compute_features

logger = logging.getLogger(__name__)

# Action indices (must match agent.py and strategy.py)
ACTION_HOLD = 0
ACTION_BUY  = 1
ACTION_SELL = 2

_SESSION_MINUTES = 390           # Full regular session length
_NO_TRADE_OPEN   = 15            # First 15 min: no trades
_NO_TRADE_CLOSE  = 15            # Last 15 min: no trades
_TRANSACTION_COST_LAMBDA = 0.0001


class TradingEnv(gym.Env):
    """Intraday trading environment for a single stock.

    Args:
        features: Numpy array of shape (total_bars, INPUT_DIM) containing
                  pre-computed normalised features for every minute bar.
        day_starts: List of indices into `features` where each trading day starts.
        lookback_bars: Number of bars in the observation window (default 60).
        transaction_cost_lambda: λ for the reward cost term.
        seed: Random seed for reproducible episode selection.

    Observation space:
        Box of shape (lookback_bars, INPUT_DIM) — same as the network input.
    Action space:
        Discrete(3) — 0=HOLD, 1=BUY, 2=SELL.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        day_starts: list,
        lookback_bars: int = 60,
        transaction_cost_lambda: float = _TRANSACTION_COST_LAMBDA,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self._features = features.astype(np.float32)
        self._day_starts = list(day_starts)
        self._lookback = lookback_bars
        self._lambda = transaction_cost_lambda
        self._total_bars = len(features)
        self._input_dim = features.shape[1]

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_bars, self._input_dim),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Internal state (reset on each episode)
        self._current_step: int = 0
        self._day_start_idx: int = 0
        self._episode_end_idx: int = 0
        self._position: int = 0        # 0=flat, 1=long
        self._entry_price: float = 0.0
        self._daily_pnl: float = 0.0

        # Price column index in features matrix (feature 0 = return; price is implicit)
        # We need raw close prices for reward computation — store separately
        self._episode_prices: Optional[np.ndarray] = None

        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Start a new episode by selecting a random training day.

        Returns:
            Tuple of (observation, info_dict).
        """
        if seed is not None:
            np.random.seed(seed)

        # Pick a random day that has enough bars for at least one full window
        valid_days = [
            d for d in self._day_starts
            if d + self._lookback < self._total_bars
        ]
        if not valid_days:
            raise RuntimeError("No valid training days found in dataset.")

        self._day_start_idx = np.random.choice(valid_days)

        # Episode covers up to 390 bars from day start (or end of dataset)
        self._episode_end_idx = min(
            self._day_start_idx + _SESSION_MINUTES,
            self._total_bars - 1,
        )
        # Start step at lookback so first obs is fully filled
        self._current_step = self._day_start_idx + self._lookback

        # Reset position and P&L
        self._position = 0
        self._entry_price = 0.0
        self._daily_pnl = 0.0

        obs = self._get_obs()
        info = self._get_info(0.0)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.

        Args:
            action: Action integer — 0=HOLD, 1=BUY, 2=SELL.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Enforce no-trade buffers — override BUY/SELL with HOLD
        minute_in_session = self._current_step - self._day_start_idx
        if minute_in_session < _NO_TRADE_OPEN or \
                minute_in_session >= _SESSION_MINUTES - _NO_TRADE_CLOSE:
            action = ACTION_HOLD

        reward = 0.0

        # Approximate current close price via cumulative return reconstruction
        # Feature index 0 is the bar return; we reconstruct price relative to 100
        current_price = self._reconstruct_price(self._current_step)

        if action == ACTION_BUY and self._position == 0:
            # Open long position
            self._position = 1
            self._entry_price = current_price

        elif action == ACTION_SELL and self._position == 1:
            # Close long position and compute reward
            reward = self._compute_reward(current_price)
            self._daily_pnl += reward
            self._position = 0
            self._entry_price = 0.0

        # Force-close any open position at EOD
        self._current_step += 1
        terminated = self._current_step >= self._episode_end_idx

        if terminated and self._position == 1:
            eod_price = self._reconstruct_price(self._current_step - 1)
            reward += self._compute_reward(eod_price)
            self._position = 0

        obs = self._get_obs()
        info = self._get_info(current_price)

        return obs, float(reward), terminated, False, info

    def render(self) -> None:
        """Rendering is not supported in this environment."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Extract the (lookback_bars, input_dim) observation window."""
        start = max(0, self._current_step - self._lookback)
        end = self._current_step
        window = self._features[start:end]

        # Pad with zeros if we're at the beginning of data
        if len(window) < self._lookback:
            pad = np.zeros((self._lookback - len(window), self._input_dim), dtype=np.float32)
            window = np.vstack([pad, window])

        return window.astype(np.float32)

    def _reconstruct_price(self, step_idx: int) -> float:
        """Approximate price by exponentiating cumulative returns.

        Feature column 0 is the bar return (close_{t}/close_{t-1} - 1).
        We reconstruct price relative to an arbitrary base of 100.

        Args:
            step_idx: Absolute index into self._features.

        Returns:
            Reconstructed price float.
        """
        day_returns = self._features[self._day_start_idx:step_idx + 1, 0]
        price = 100.0 * np.prod(1.0 + day_returns.clip(-0.1, 0.1))
        return float(price)

    def _compute_reward(self, exit_price: float) -> float:
        """Compute log-return reward minus round-trip transaction cost.

        Args:
            exit_price: Price at which the long position is closed.

        Returns:
            Reward scalar.
        """
        if self._entry_price <= 0:
            return 0.0
        log_ret = np.log(exit_price / (self._entry_price + 1e-10))
        cost = self._lambda * 2   # 10 bps per side × 2 (round-trip)
        return float(log_ret - cost)

    def _get_info(self, price: float) -> Dict[str, Any]:
        """Build the info dict returned alongside each step.

        Args:
            price: Approximate current price.

        Returns:
            Dict with keys: price, position, daily_pnl, minute_in_session.
        """
        return {
            "price": price,
            "position": self._position,
            "daily_pnl": self._daily_pnl,
            "minute_in_session": self._current_step - self._day_start_idx,
        }
