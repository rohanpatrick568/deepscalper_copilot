"""
tests/test_architecture.py — Unit tests for colab/deepscalper/architecture.py.

Covers:
    MicroEncoder   — GRU two-stream, output shape
    MacroEncoder   — MLP, output shape
    DeepScalperNet — forward pass: (q_dir, q_size) shapes
    DeepScalperNet — V2 config (n_dir=2, lob_dim=4)
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "colab"))

from config import LOB_DIM, N_DIR, N_SIZE

from deepscalper.architecture import (
    DeepScalperNet,
    MacroEncoder,
    MicroEncoder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand(shape, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape)


# ===========================================================================
# MicroEncoder
# ===========================================================================

class TestMicroEncoder:
    @pytest.mark.parametrize("batch", [1, 4])
    def test_output_shape(self, batch):
        enc = MicroEncoder(lob_dim=LOB_DIM, priv_dim=2, gru_hidden=32)
        lob  = _rand((batch, 10, LOB_DIM))
        priv = _rand((batch, 10, 2))
        out  = enc(lob, priv)
        assert out.shape == (batch, 64)  # 2 × gru_hidden

    def test_no_nan(self):
        enc  = MicroEncoder(lob_dim=LOB_DIM, priv_dim=2, gru_hidden=32)
        lob  = _rand((2, 10, LOB_DIM))
        priv = _rand((2, 10, 2))
        out  = enc(lob, priv)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self):
        enc  = MicroEncoder(lob_dim=LOB_DIM, priv_dim=2, gru_hidden=16)
        lob  = _rand((2, 5, LOB_DIM)).requires_grad_(True)
        priv = _rand((2, 5, 2)).requires_grad_(True)
        enc(lob, priv).sum().backward()
        assert lob.grad is not None
        assert priv.grad is not None


# ===========================================================================
# MacroEncoder
# ===========================================================================

class TestMacroEncoder:
    @pytest.mark.parametrize("batch", [1, 8])
    def test_output_shape(self, batch):
        enc   = MacroEncoder(macro_dim=11, hidden=32, embed_dim=16)
        macro = _rand((batch, 11))
        out   = enc(macro)
        assert out.shape == (batch, 16)

    def test_no_nan(self):
        enc   = MacroEncoder(macro_dim=11, hidden=32, embed_dim=16)
        macro = _rand((4, 11))
        assert not torch.isnan(enc(macro)).any()


# ===========================================================================
# DeepScalperNet — V2 defaults (N_DIR=2, LOB_DIM=4)
# ===========================================================================

class TestDeepScalperNetDefaults:
    @pytest.fixture
    def net(self):
        torch.manual_seed(0)
        return DeepScalperNet()   # defaults: n_dir=2, n_size=1, lob_dim=4

    def _batch(self, net, batch=2, seq=10):
        lob   = _rand((batch, seq, LOB_DIM))
        priv  = _rand((batch, seq, 2))
        macro = _rand((batch, 11))
        return net(lob, priv, macro)

    def test_forward_returns_2_outputs(self, net):
        result = self._batch(net)
        assert len(result) == 2

    def test_q_dir_shape(self, net):
        q_dir, *_ = self._batch(net)
        assert q_dir.shape == (2, N_DIR)

    def test_q_size_shape(self, net):
        _, q_size = self._batch(net)
        assert q_size.shape == (2, N_SIZE)

    def test_no_nan(self, net):
        q_dir, q_size = self._batch(net)
        for t, name in [(q_dir, "q_dir"), (q_size, "q_size")]:
            assert not torch.isnan(t).any(), f"NaN in {name}"

    def test_no_inf(self, net):
        q_dir, q_size = self._batch(net)
        for t in (q_dir, q_size):
            assert torch.isfinite(t).all()


# ===========================================================================
# DeepScalperNet — V2 config (N_DIR=2, LOB_DIM=4)
# ===========================================================================

class TestDeepScalperNetConfigured:
    @pytest.fixture
    def net_cfg(self):
        torch.manual_seed(42)
        return DeepScalperNet(
            macro_dim=11, lob_dim=LOB_DIM, priv_dim=2,
            gru_hidden=32, macro_embed=16, fc_hidden=32,
            n_dir=N_DIR, n_size=N_SIZE,
        )

    def _batch(self, net, batch: int = 2, seq: int = 10):
        lob   = _rand((batch, seq, LOB_DIM))
        priv  = _rand((batch, seq, 2))
        macro = _rand((batch, 11))
        return net(lob, priv, macro)

    def test_q_dir_shape(self, net_cfg):
        q_dir, *_ = self._batch(net_cfg)
        assert q_dir.shape == (2, N_DIR)

    def test_q_size_shape(self, net_cfg):
        _, q_size = self._batch(net_cfg)
        assert q_size.shape == (2, N_SIZE)

    def test_no_nan_cfg(self, net_cfg):
        q_dir, q_size = self._batch(net_cfg)
        for t in (q_dir, q_size):
            assert not torch.isnan(t).any()

    def test_batch_size_1(self, net_cfg):
        q_dir, q_size = self._batch(net_cfg, batch=1)
        assert q_dir.shape == (1, N_DIR)
        assert q_size.shape == (1, N_SIZE)

    def test_batch_size_32(self, net_cfg):
        q_dir, q_size = self._batch(net_cfg, batch=32)
        assert q_dir.shape == (32, N_DIR)


# ===========================================================================
# embed() method
# ===========================================================================

class TestEmbed:
    def test_embed_shape(self):
        """embed() should return (batch, 2*gru_hidden + macro_embed)."""
        net = DeepScalperNet(
            macro_dim=11, lob_dim=LOB_DIM, priv_dim=2,
            gru_hidden=32, macro_embed=16, fc_hidden=32,
            n_dir=N_DIR, n_size=N_SIZE,
        )
        lob   = _rand((3, 10, LOB_DIM))
        priv  = _rand((3, 10, 2))
        macro = _rand((3, 11))
        emb   = net.embed(lob, priv, macro)
        # embed_dim = 2*32 + 16 = 80
        assert emb.shape == (3, 80)

    def test_embed_no_nan(self):
        net = DeepScalperNet(
            macro_dim=11, lob_dim=LOB_DIM, priv_dim=2,
            gru_hidden=32, macro_embed=16, fc_hidden=32,
            n_dir=N_DIR, n_size=N_SIZE,
        )
        emb = net.embed(_rand((2, 10, LOB_DIM)), _rand((2, 10, 2)), _rand((2, 11)))
        assert not torch.isnan(emb).any()


# ===========================================================================
# Gradient flow through full network
# ===========================================================================

class TestGradients:
    def test_backward_ok(self):
        net = DeepScalperNet(
            macro_dim=11, lob_dim=LOB_DIM, priv_dim=2,
            gru_hidden=16, macro_embed=8, fc_hidden=16,
            n_dir=N_DIR, n_size=N_SIZE,
        )
        lob   = _rand((4, 10, LOB_DIM)).requires_grad_(True)
        priv  = _rand((4, 10, 2)).requires_grad_(True)
        macro = _rand((4, 11)).requires_grad_(True)

        q_dir, q_size = net(lob, priv, macro)
        loss = q_dir.sum() + q_size.sum()
        loss.backward()

        assert lob.grad is not None
        assert priv.grad is not None
        assert macro.grad is not None

    def test_parameter_gradients_nonzero(self):
        net = DeepScalperNet(
            macro_dim=11, lob_dim=LOB_DIM, priv_dim=2,
            gru_hidden=16, macro_embed=8, fc_hidden=16,
            n_dir=N_DIR, n_size=N_SIZE,
        )
        lob   = _rand((2, 10, LOB_DIM))
        priv  = _rand((2, 10, 2))
        macro = _rand((2, 11))
        q_dir, q_size = net(lob, priv, macro)
        (q_dir.sum() + q_size.sum()).backward()

        param_grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert len(param_grads) > 0
        # At least one parameter has a nonzero gradient
        has_nonzero = any(g.abs().sum().item() > 0 for g in param_grads)
        assert has_nonzero
