"""
tests/test_config.py — Validate every config constant for correct type and value.

These tests catch accidental regressions when parameters are changed (e.g.
someone accidentally sets N_DIR=3 for crypto or forgets CLOSE_ALL_EOD=False).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


# ---------------------------------------------------------------------------
# Trading universe
# ---------------------------------------------------------------------------

class TestTradingUniverse:
    def test_crypto_pairs_is_list(self):
        assert isinstance(config.CRYPTO_PAIRS, list)

    def test_crypto_pairs_nonempty(self):
        assert len(config.CRYPTO_PAIRS) >= 1

    def test_btc_usd_in_crypto_pairs(self):
        assert "BTC/USD" in config.CRYPTO_PAIRS

    def test_crypto_pairs_slash_format(self):
        """All pairs must use the Alpaca data-API slash format."""
        for pair in config.CRYPTO_PAIRS:
            assert "/" in pair, f"{pair!r} is missing '/' separator"

    def test_close_all_eod_disabled(self):
        """Crypto is 24/7; EOD close must be disabled."""
        assert config.CLOSE_ALL_EOD is False

    def test_no_trade_buffers_none(self):
        """Crypto has no market session; buffers must be None."""
        assert config.NO_TRADE_OPEN_BUFFER_MIN is None
        assert config.NO_TRADE_CLOSE_BUFFER_MIN is None


# ---------------------------------------------------------------------------
# Action space (V2: binary FLAT/LONG)
# ---------------------------------------------------------------------------

class TestActionSpace:
    def test_n_dir_is_2(self):
        assert config.N_DIR == 2, "V2 requires N_DIR=2 (FLAT/LONG only)"

    def test_n_size_is_1(self):
        assert config.N_SIZE == 1, "V2: size externally determined by Kelly"

    def test_action_dim_is_2(self):
        assert config.ACTION_DIM == 2

    def test_action_dim_matches_n_dir(self):
        assert config.ACTION_DIM == config.N_DIR


# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------

class TestFeatureDims:
    def test_lob_dim_is_4(self):
        assert config.LOB_DIM == 4, "V2 dual-mode LOB has 4 features"

    def test_macro_dim_is_11(self):
        assert config.MACRO_DIM == 11

    def test_priv_dim_is_2(self):
        assert config.PRIV_DIM == 2

    def test_lookback_bars_positive(self):
        assert config.LOOKBACK_BARS >= 1

    def test_lookback_bars_v2(self):
        assert config.LOOKBACK_BARS == 10, "V2 TradeMaster default is 10"


# ---------------------------------------------------------------------------
# Hindsight / reward parameters
# ---------------------------------------------------------------------------

class TestHindsightParams:
    def test_hindsight_weight_range(self):
        assert 0.0 <= config.HINDSIGHT_WEIGHT <= 1.0

    def test_hindsight_weight_v2(self):
        assert config.HINDSIGHT_WEIGHT == 0.2

    def test_hindsight_horizon_v2(self):
        assert config.HINDSIGHT_HORIZON == 10

    def test_hindsight_horizon_positive(self):
        assert config.HINDSIGHT_HORIZON >= 1


# ---------------------------------------------------------------------------
# Risk / circuit breaker
# ---------------------------------------------------------------------------

class TestRiskParams:
    def test_max_daily_loss_range(self):
        assert 0.0 < config.MAX_DAILY_LOSS_PCT <= 1.0

    def test_kelly_fraction_range(self):
        assert 0.0 < config.KELLY_FRACTION <= 1.0

    def test_max_position_pct_range(self):
        assert 0.0 < config.MAX_POSITION_PCT <= 1.0

    def test_volatility_halt_multiplier_positive(self):
        assert config.VOLATILITY_HALT_MULTIPLIER > 1.0

    def test_consecutive_loss_halt_positive(self):
        assert config.CONSECUTIVE_LOSS_HALT >= 1

    def test_transaction_cost_positive(self):
        assert config.TRANSACTION_COST_LAMBDA > 0.0

    def test_transaction_cost_v2(self):
        """25 bps for Alpaca crypto taker."""
        assert abs(config.TRANSACTION_COST_LAMBDA - 0.0025) < 1e-9

    def test_starting_capital_positive(self):
        assert config.STARTING_CAPITAL > 0.0


# ---------------------------------------------------------------------------
# Architecture dimensions
# ---------------------------------------------------------------------------

class TestArchDims:
    def test_gru_hidden_positive(self):
        assert config.GRU_HIDDEN >= 1

    def test_fc_hidden_positive(self):
        assert config.FC_HIDDEN >= 1

    def test_macro_embed_dim_positive(self):
        assert config.MACRO_EMBED_DIM >= 1

    def test_atr_period_positive(self):
        assert config.ATR_PERIOD >= 2
