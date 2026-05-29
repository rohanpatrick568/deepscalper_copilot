"""tests/test_config.py — Validate current equities DeepScalper config constants."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

import config


# ---------------------------------------------------------------------------
# Trading universe (equities)
# ---------------------------------------------------------------------------

class TestTradingUniverse:
    def test_trading_universe_is_list(self):
        assert isinstance(config.TRADING_UNIVERSE, list)

    def test_trading_universe_nonempty(self):
        assert len(config.TRADING_UNIVERSE) >= 1

    def test_symbols_are_equity_style(self):
        for symbol in config.TRADING_UNIVERSE:
            assert "/" not in symbol

    def test_close_all_eod_enabled(self):
        assert config.CLOSE_ALL_EOD is True

    def test_open_close_buffers_enabled(self):
        assert isinstance(config.NO_TRADE_OPEN_BUFFER_MIN, int)
        assert isinstance(config.NO_TRADE_CLOSE_BUFFER_MIN, int)
        assert config.NO_TRADE_OPEN_BUFFER_MIN >= 0
        assert config.NO_TRADE_CLOSE_BUFFER_MIN >= 0

    def test_crypto_pairs_kept_as_empty_alias(self):
        assert isinstance(config.CRYPTO_PAIRS, list)
        assert config.CRYPTO_PAIRS == []


# ---------------------------------------------------------------------------
# Action space (SHORT/FLAT/LONG)
# ---------------------------------------------------------------------------

class TestActionSpace:
    def test_n_dir_is_3(self):
        assert config.N_DIR == 3

    def test_n_size_is_4(self):
        assert config.N_SIZE == 4

    def test_action_dim_is_3(self):
        assert config.ACTION_DIM == 3

    def test_action_dim_matches_n_dir(self):
        assert config.ACTION_DIM == config.N_DIR


# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------

class TestFeatureDims:
    def test_lob_dim_is_5(self):
        assert config.LOB_DIM == 5

    def test_macro_dim_is_11(self):
        assert config.MACRO_DIM == 11

    def test_priv_dim_is_2(self):
        assert config.PRIV_DIM == 2

    def test_lookback_bars_positive(self):
        assert config.LOOKBACK_BARS >= 1

    def test_lookback_bars_paper_aligned(self):
        assert config.LOOKBACK_BARS == 60


# ---------------------------------------------------------------------------
# Hindsight / reward parameters
# ---------------------------------------------------------------------------

class TestHindsightParams:
    def test_hindsight_weight_range(self):
        assert 0.0 <= config.HINDSIGHT_WEIGHT <= 1.0

    def test_hindsight_weight_v2(self):
        assert config.HINDSIGHT_WEIGHT == 0.2

    def test_hindsight_horizon_paper_aligned(self):
        assert config.HINDSIGHT_HORIZON == 60

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

    def test_transaction_cost_baseline(self):
        assert abs(config.TRANSACTION_COST_LAMBDA - 0.001) < 1e-9

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


# ---------------------------------------------------------------------------
# TradeMaster canonical training block
# ---------------------------------------------------------------------------

class TestTradeMasterTrainingBlock:
    def test_epochs_match_trademaster(self):
        assert config.EPOCHS == 20

    def test_horizon_len_match_trademaster(self):
        assert config.HORIZON_LEN == 128

    def test_buffer_size_match_trademaster(self):
        assert config.BUFFER_SIZE == 1_000_000

    def test_learning_rate_match_trademaster(self):
        assert abs(config.LEARNING_RATE - 1e-3) < 1e-12

    def test_gamma_match_trademaster(self):
        assert abs(config.GAMMA - 0.9) < 1e-12

    def test_repeat_times_match_trademaster(self):
        assert abs(config.REPEAT_TIMES - 1.0) < 1e-12

    def test_clip_grad_norm_match_trademaster(self):
        assert abs(config.CLIP_GRAD_NORM - 3.0) < 1e-12

    def test_soft_update_tau_match_trademaster(self):
        assert abs(config.SOFT_UPDATE_TAU - 0.0) < 1e-12

    def test_state_value_tau_match_trademaster(self):
        assert abs(config.STATE_VALUE_TAU - 0.005) < 1e-12

    def test_explore_rate_match_trademaster(self):
        assert abs(config.EXPLORE_RATE - 0.25) < 1e-12
