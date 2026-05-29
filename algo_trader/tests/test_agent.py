"""
tests/test_agent.py — Unit tests for colab/deepscalper/agent.py.

Covers:
    _SumTree               — add, update, retrieve, total
    PrioritizedReplayBuffer — push, sample, update_priorities, len
    DeepScalperAgent       — constructor, select_action, store, learn, save/load
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "colab"))

from config import LOB_DIM, N_DIR, N_SIZE

from deepscalper.agent import (
    DeepScalperAgent,
    PrioritizedReplayBuffer,
    _SumTree,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(seq: int = 10, lob_dim: int = LOB_DIM, priv_dim: int = 2, macro_dim: int = 11):
    """Create a random obs dict (numpy arrays, no batch dim)."""
    rng = np.random.default_rng(0)
    return {
        "lob":   rng.normal(0, 0.01, (seq, lob_dim)).astype(np.float32),
        "priv":  rng.normal(0, 0.01, (seq, priv_dim)).astype(np.float32),
        "macro": rng.normal(0, 0.01, (macro_dim,)).astype(np.float32),
    }


def _make_agent(n_dir: int = N_DIR, lob_dim: int = LOB_DIM, buffer_capacity: int = 500) -> DeepScalperAgent:
    """Small V2-like agent for fast tests."""
    return DeepScalperAgent(
        macro_dim=11, lob_dim=lob_dim, priv_dim=2,
        n_dir=n_dir, n_size=N_SIZE,
        gru_hidden=16, macro_embed=8, fc_hidden=16,
        lr=1e-3, gamma=0.9, soft_update_tau=0.0,
        repeat_times=1.0, clip_grad_norm=3.0,
        batch_size=8,
        buffer_capacity=buffer_capacity,
        explore_rate=0.25,
        device="cpu",
    )


# ===========================================================================
# _SumTree
# ===========================================================================

class TestSumTree:
    def test_total_starts_at_zero(self):
        tree = _SumTree(16)
        assert tree.total == 0.0

    def test_add_single(self):
        tree = _SumTree(16)
        tree.add(1.0, "data")
        assert tree.total == pytest.approx(1.0)

    def test_add_multiple_sums_correctly(self):
        tree = _SumTree(8)
        for i in range(1, 5):
            tree.add(float(i), f"d{i}")
        assert tree.total == pytest.approx(10.0)  # 1+2+3+4

    def test_update_changes_total(self):
        tree = _SumTree(8)
        tree.add(1.0, "a")
        idx = tree.capacity - 1  # first leaf index
        tree.update(idx, 5.0)
        assert tree.total == pytest.approx(5.0)

    def test_get_returns_data(self):
        tree = _SumTree(8)
        tree.add(1.0, "hello")
        _, _, data = tree.get(0.5)
        assert data == "hello"

    def test_n_entries_bounded_by_capacity(self):
        cap  = 4
        tree = _SumTree(cap)
        for i in range(10):
            tree.add(1.0, i)
        assert tree.n_entries == cap

    def test_retrieve_in_range(self):
        tree = _SumTree(16)
        for i in range(8):
            tree.add(1.0, i)
        for s in np.linspace(0.01, tree.total - 0.01, 20):
            idx, prio, data = tree.get(s)
            assert data is not None


# ===========================================================================
# PrioritizedReplayBuffer
# ===========================================================================

class TestPrioritizedReplayBuffer:
    def _make_buf(self, capacity: int = 100) -> PrioritizedReplayBuffer:
        return PrioritizedReplayBuffer(capacity=capacity, alpha=0.6, beta_start=0.4)

    def _push_n(self, buf: PrioritizedReplayBuffer, n: int = 10):
        obs = _make_obs()
        for i in range(n):
            buf.push(obs, 0, 0, float(i), obs, i == n - 1, 0.01)

    def test_len_zero_initially(self):
        assert len(self._make_buf()) == 0

    def test_len_after_push(self):
        buf = self._make_buf()
        self._push_n(buf, 5)
        assert len(buf) == 5

    def test_len_bounded_by_capacity(self):
        buf = self._make_buf(capacity=4)
        self._push_n(buf, 10)
        assert len(buf) == 4

    def test_sample_returns_batch_keys(self):
        buf = self._make_buf()
        self._push_n(buf, 20)
        batch, indices, weights = buf.sample(8, device="cpu")
        required = {"lob", "priv", "macro", "dir_acts", "size_acts",
                    "rewards", "next_lob", "next_priv", "next_macro",
                    "dones", "vol_tgts", "weights"}
        assert required <= set(batch.keys())

    def test_sample_batch_size(self):
        buf = self._make_buf()
        self._push_n(buf, 20)
        batch, _, _ = buf.sample(8, device="cpu")
        assert batch["lob"].shape[0] == 8

    def test_sample_lob_shape(self):
        buf = self._make_buf()
        self._push_n(buf, 20)
        batch, _, _ = buf.sample(8, device="cpu")
        assert batch["lob"].shape == (8, 10, LOB_DIM)

    def test_sample_rewards_tensor(self):
        buf = self._make_buf()
        self._push_n(buf, 20)
        batch, _, _ = buf.sample(8, device="cpu")
        assert isinstance(batch["rewards"], torch.Tensor)

    def test_is_weights_in_unit_range(self):
        buf = self._make_buf()
        self._push_n(buf, 50)
        _, _, weights = buf.sample(16, device="cpu")
        assert 0.0 < weights.max() <= 1.0 + 1e-6

    def test_update_priorities_does_not_raise(self):
        buf = self._make_buf()
        self._push_n(buf, 20)
        batch, indices, _ = buf.sample(8, device="cpu")
        buf.update_priorities(indices, np.ones(8))

    def test_beta_increases_over_time(self):
        buf = PrioritizedReplayBuffer(beta_frames=10)
        beta0 = buf.beta
        for _ in range(10):
            self._push_n(buf, 20)
            buf.sample(8)   # increments frame
        assert buf.beta >= beta0

    def test_beta_never_exceeds_1(self):
        buf = PrioritizedReplayBuffer(beta_frames=5)
        for _ in range(100):
            self._push_n(buf, 5)
            if len(buf) >= 8:
                buf.sample(8)
        assert buf.beta <= 1.0 + 1e-9


# ===========================================================================
# DeepScalperAgent — constructor
# ===========================================================================

class TestAgentConstructor:
    def test_creates_online_and_target_nets(self):
        agent = _make_agent()
        assert agent.online_net is not None
        assert agent.target_net is not None

    def test_target_equals_online_at_init(self):
        agent = _make_agent()
        for op, tp in zip(agent.online_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(op, tp)

    def test_explore_rate_default(self):
        agent = _make_agent()
        assert agent.explore_rate == pytest.approx(0.25)

    def test_buffer_empty_at_init(self):
        agent = _make_agent()
        assert len(agent.buffer) == 0


# ===========================================================================
# DeepScalperAgent — select_action
# ===========================================================================

class TestSelectAction:
    def test_returns_tuple_of_two_ints(self):
        agent = _make_agent(n_dir=N_DIR)
        obs   = _make_obs()
        result = agent.select_action(obs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_dir_action_in_range(self):
        agent = _make_agent(n_dir=N_DIR)
        obs   = _make_obs()
        for _ in range(50):
            dir_act, _ = agent.select_action(obs)
            assert 0 <= dir_act < N_DIR, f"dir_act={dir_act} out of range"

    def test_size_action_in_range(self):
        agent = _make_agent(n_dir=N_DIR)
        obs   = _make_obs()
        for _ in range(50):
            _, size_act = agent.select_action(obs)
            assert 0 <= size_act < N_SIZE

    def test_high_explore_rate_is_random(self):
        """With explore_rate=1.0, actions should vary over many samples."""
        agent = _make_agent(n_dir=N_DIR)
        agent.explore_rate = 1.0
        agent.epsilon = 1.0
        obs   = _make_obs()
        actions = set()
        for _ in range(100):
            dir_act, _ = agent.select_action(obs)
            actions.add(dir_act)
        # Over 100 tries with N_DIR=3, at least two actions should appear.
        assert len(actions) >= 2

    def test_zero_explore_rate_is_greedy(self):
        """With explore_rate=0, the same obs should always return the same action."""
        agent = _make_agent(n_dir=N_DIR)
        agent.explore_rate = 0.0
        agent.epsilon = 0.0
        obs = _make_obs()
        dir_acts = {agent.select_action(obs)[0] for _ in range(20)}
        # Greedy policy must be deterministic
        assert len(dir_acts) == 1

    def test_explore_rate_remains_static_over_steps(self):
        agent = _make_agent(n_dir=N_DIR)
        rate_before = agent.explore_rate
        obs = _make_obs()
        for _ in range(100):
            agent.select_action(obs)
        assert agent.explore_rate == pytest.approx(rate_before)

    def test_epsilon_alias_tracks_explore_rate(self):
        agent = _make_agent(n_dir=N_DIR)
        agent.explore_rate = 0.17
        obs = _make_obs()
        for _ in range(10):
            agent.select_action(obs)
        assert agent.epsilon == pytest.approx(agent.explore_rate)


# ===========================================================================
# DeepScalperAgent — store + learn
# ===========================================================================

class TestStoreAndLearn:
    def test_learn_returns_none_when_buffer_small(self):
        agent = _make_agent()
        result = agent.learn()
        assert result is None

    def test_learn_returns_float_after_enough_samples(self):
        agent = _make_agent(buffer_capacity=200)
        obs   = _make_obs()
        for _ in range(20):
            agent.store(obs, 0, 0, 0.0, obs, False, 0.01)
        loss = agent.learn()
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_multiple_learns_do_not_crash(self):
        agent = _make_agent()
        obs   = _make_obs()
        for _ in range(30):
            agent.store(obs, 0, 0, 0.01, obs, False, 0.001)
        for _ in range(5):
            agent.learn()

    def test_learn_loss_nonnegative(self):
        agent = _make_agent()
        obs   = _make_obs()
        for _ in range(20):
            agent.store(obs, 0, 0, 0.0, obs, False, 0.0)
        loss = agent.learn()
        if loss is not None:
            assert loss >= 0.0


# ===========================================================================
# DeepScalperAgent — save / load
# ===========================================================================

class TestSaveLoad:
    def test_save_and_load_round_trip(self, tmp_path):
        agent = _make_agent()
        path  = str(tmp_path / "agent.pt")
        agent.save(path)

        agent2 = _make_agent()
        agent2.load(path)

        for p1, p2 in zip(agent.online_net.parameters(), agent2.online_net.parameters()):
            assert torch.allclose(p1, p2)

    def test_explore_rate_restored(self, tmp_path):
        agent = _make_agent()
        agent.explore_rate = 0.42
        agent.epsilon = 0.42
        agent._steps  = 999
        path = str(tmp_path / "agent.pt")
        agent.save(path)

        agent2 = _make_agent()
        agent2.load(path)
        assert agent2.explore_rate == pytest.approx(0.42)

    def test_load_nonexistent_raises(self):
        agent = _make_agent()
        with pytest.raises((FileNotFoundError, RuntimeError)):
            agent.load("/nonexistent/path/agent.pt")


# ===========================================================================
# compute_hindsight_reward
# ===========================================================================

class TestHindsightReward:
    def test_flat_position_returns_base(self):
        agent = _make_agent()
        r = agent.compute_hindsight_reward(0.5, position=0, current_price=100, future_price=110)
        assert r == pytest.approx(0.5)

    def test_long_positive_future_adds_bonus(self):
        agent = _make_agent()
        base  = 0.0
        r = agent.compute_hindsight_reward(base, position=1, current_price=100, future_price=110)
        assert r > base

    def test_long_negative_future_reduces_reward(self):
        agent = _make_agent()
        r = agent.compute_hindsight_reward(0.0, position=1, current_price=100, future_price=90)
        assert r < 0.0

    def test_zero_future_price_returns_base(self):
        agent = _make_agent()
        r = agent.compute_hindsight_reward(1.0, position=1, current_price=100, future_price=0)
        assert r == pytest.approx(1.0)
