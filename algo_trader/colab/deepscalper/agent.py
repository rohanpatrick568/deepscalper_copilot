"""
colab/deepscalper/agent.py — DeepScalper RL Agent (1:1 paper replica).

Implements the four core components of DeepScalper (CIKM '22, Sun et al.):

  1. Branching DQN (BDQ) with per-branch Double-DQN targets
       L_q = (1/|D|) Σ_d Σ_b w_b (y_d - Q_d(s_b, a_d_b))²
       where d ∈ {direction, size}, w_b = IS weights from PER

  2. Hindsight bonus reward (Section 4.2)
       r_H = r_t + w × log(P_{t+h} / P_t) × position
       Applied at store() time; w=0.01, h=60.

  3. Prioritized Experience Replay (PER) with SumTree (O(log N) ops)
       Stores dict observations + separate vol_target for the auxiliary task.

  4. Volatility auxiliary task (Section 4.4)
       L_vol = MSE(vol_head(e_t), realized_vol_target)
       Total: L = L_q + η × L_vol  (η = 1.0)

Observation format (dict):
    obs['lob']   → np.ndarray (seq_len, LOB_DIM=5)   micro features
    obs['priv']  → np.ndarray (seq_len, PRIV_DIM=2)  private state (pos, pnl)
    obs['macro'] → np.ndarray (MACRO_DIM=11,)         current-bar macro features

Actions:
    dir_action  ∈ {0=HOLD, 1=BUY, 2=SELL}
    size_action ∈ {0=25%, 1=50%, 2=75%, 3=100%}
"""

import logging
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from colab.deepscalper.architecture import DeepScalperNet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prioritized Experience Replay — SumTree
# ---------------------------------------------------------------------------

class _SumTree:
    """Binary SumTree for O(log N) priority sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data     = np.empty(capacity, dtype=object)
        self.write    = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx  = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """PER buffer that stores dict observations and the vol_target.

    Each transition stores:
        (obs_dict, dir_action, size_action, reward, next_obs_dict, done, vol_target)

    Args:
        capacity    : Maximum number of transitions.
        alpha       : Priority exponent (0=uniform, 1=full PER).
        beta_start  : IS-weight correction exponent start value.
        beta_frames : Number of frames over which β is annealed to 1.0.
        epsilon     : Small constant to ensure non-zero priority.
    """

    def __init__(
        self,
        capacity:    int   = 100_000,
        alpha:       float = 0.6,
        beta_start:  float = 0.4,
        beta_frames: int   = 100_000,
        epsilon:     float = 1e-6,
    ) -> None:
        self.tree        = _SumTree(capacity)
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.epsilon     = epsilon
        self.frame       = 1
        self.max_prio    = 1.0

    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(
        self,
        obs:        Dict[str, np.ndarray],
        dir_action: int,
        size_action: int,
        reward:     float,
        next_obs:   Dict[str, np.ndarray],
        done:       bool,
        vol_target: float,
    ) -> None:
        """Store a transition with maximum current priority."""
        data = (obs, dir_action, size_action, reward, next_obs, done, vol_target)
        self.tree.add(self.max_prio ** self.alpha, data)

    def sample(self, batch_size: int, device: str = "cpu"):
        """Sample a batch of transitions proportional to priority.

        Returns:
            Tuple of (batch tensors, indices, IS weights).
        """
        n = self.tree.n_entries
        indices   = np.zeros(batch_size, dtype=np.int64)
        weights   = np.zeros(batch_size, dtype=np.float32)
        segment   = self.tree.total / batch_size
        min_prob  = (self.tree.tree[self.tree.capacity - 1] / self.tree.total + 1e-10)

        obs_lob   = []
        obs_priv  = []
        obs_macro = []
        dir_acts  = []
        size_acts = []
        rewards   = []
        nob_lob   = []
        nob_priv  = []
        nob_macro = []
        dones     = []
        vol_tgts  = []

        beta = self.beta
        self.frame += 1

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, prio, data = self.tree.get(s)
            prob = prio / self.tree.total
            weights[i] = ((1.0 / (n * max(prob, 1e-10))) ** beta)
            indices[i] = idx

            obs, d_act, s_act, rew, nobs, dn, v_tgt = data
            obs_lob.append(obs['lob'])
            obs_priv.append(obs['priv'])
            obs_macro.append(obs['macro'])
            dir_acts.append(d_act)
            size_acts.append(s_act)
            rewards.append(rew)
            nob_lob.append(nobs['lob'])
            nob_priv.append(nobs['priv'])
            nob_macro.append(nobs['macro'])
            dones.append(float(dn))
            vol_tgts.append(v_tgt)

        weights /= weights.max()

        def _t(arr, dtype=torch.float32):
            return torch.tensor(np.array(arr), dtype=dtype, device=device)

        batch = {
            'lob':        _t(obs_lob),
            'priv':       _t(obs_priv),
            'macro':      _t(obs_macro),
            'dir_acts':   _t(dir_acts, torch.long),
            'size_acts':  _t(size_acts, torch.long),
            'rewards':    _t(rewards),
            'next_lob':   _t(nob_lob),
            'next_priv':  _t(nob_priv),
            'next_macro': _t(nob_macro),
            'dones':      _t(dones),
            'vol_tgts':   _t(vol_tgts),
            'weights':    _t(weights),
        }
        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, prio in zip(indices, priorities):
            p = (float(prio) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), p)
            self.max_prio = max(self.max_prio, p)

    def __len__(self) -> int:
        return self.tree.n_entries


# ---------------------------------------------------------------------------
# DeepScalper Agent
# ---------------------------------------------------------------------------

class DeepScalperAgent:
    """DeepScalper RL agent — BDQ + PER + hindsight bonus + vol auxiliary task.

    Args:
        macro_dim      : MACRO_DIM (11).
        lob_dim        : LOB_DIM (5).
        priv_dim       : PRIV_DIM (2).
        n_dir          : N_DIR (3) — direction branch action count.
        n_size         : N_SIZE (4) — size branch action count.
        gru_hidden     : GRU hidden units per stream.
        macro_embed    : MacroEncoder output dim.
        fc_hidden      : Width of FC layers in BDQ heads.
        lr             : Adam learning rate.
        gamma          : Reward discount factor.
        tau            : Soft target-network update rate.
        batch_size     : Training mini-batch size.
        buffer_capacity: PER buffer capacity.
        epsilon_start  : Initial exploration rate.
        epsilon_end    : Minimum exploration rate.
        epsilon_decay  : Steps over which ε is linearly decayed.
        aux_eta        : Weight for the volatility auxiliary loss (η).
        hindsight_w    : Hindsight bonus weight (w).
        device         : 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        macro_dim:       int   = 11,
        lob_dim:         int   = 5,
        priv_dim:        int   = 2,
        n_dir:           int   = 3,
        n_size:          int   = 4,
        gru_hidden:      int   = 128,
        macro_embed:     int   = 64,
        fc_hidden:       int   = 128,
        lr:              float = 1e-4,
        gamma:           float = 0.99,
        tau:             float = 0.01,
        batch_size:      int   = 64,
        buffer_capacity: int   = 100_000,
        epsilon_start:   float = 1.0,
        epsilon_end:     float = 0.05,
        epsilon_decay:   int   = 50_000,
        aux_eta:         float = 1.0,
        hindsight_w:     float = 0.01,
        device:          str   = "cpu",
    ) -> None:
        self.n_dir         = n_dir
        self.n_size        = n_size
        self.gamma         = gamma
        self.tau           = tau
        self.batch_size    = batch_size
        self.aux_eta       = aux_eta
        self.hindsight_w   = hindsight_w
        self.device        = device
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._steps        = 0

        net_kwargs = dict(
            macro_dim=macro_dim,
            lob_dim=lob_dim,
            priv_dim=priv_dim,
            gru_hidden=gru_hidden,
            macro_embed=macro_embed,
            fc_hidden=fc_hidden,
            n_dir=n_dir,
            n_size=n_size,
        )
        self.online_net = DeepScalperNet(**net_kwargs).to(device)
        self.target_net = DeepScalperNet(**net_kwargs).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

    # ------------------------------------------------------------------
    # Hindsight bonus reward — Section 4.2
    # ------------------------------------------------------------------

    def compute_hindsight_reward(
        self,
        base_reward:   float,
        position:      int,    # +1 = long, -1 = short, 0 = flat
        current_price: float,
        future_price:  float,
    ) -> float:
        """Add hindsight bonus: r_H = r_t + w × log(P_{t+h}/P_t) × position.

        Args:
            base_reward   : The environment's step reward r_t.
            position      : Current position flag (+1/0/-1).
            current_price : P_t — price at current bar.
            future_price  : P_{t+h} — price h bars ahead (from environment).

        Returns:
            Augmented reward with hindsight bonus.
        """
        if position == 0 or future_price <= 0 or current_price <= 0:
            return base_reward
        bonus = self.hindsight_w * float(np.log(future_price / current_price)) * position
        return base_reward + bonus

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store(
        self,
        obs:        Dict[str, np.ndarray],
        dir_action: int,
        size_action: int,
        reward:     float,
        next_obs:   Dict[str, np.ndarray],
        done:       bool,
        vol_target: float,
    ) -> None:
        """Push one transition into the replay buffer."""
        self.buffer.push(obs, dir_action, size_action, reward, next_obs, done, vol_target)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: Dict[str, np.ndarray],
    ) -> Tuple[int, int]:
        """ε-greedy action selection.

        Args:
            obs : Dict with keys 'lob' (seq,5), 'priv' (seq,2), 'macro' (11,).

        Returns:
            (dir_action, size_action) — integer indices for each branch.
        """
        # Linear ε decay
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
        )
        self._steps += 1

        if random.random() < self.epsilon:
            return random.randrange(self.n_dir), random.randrange(self.n_size)

        self.online_net.eval()
        with torch.no_grad():
            lob   = torch.tensor(obs['lob'][None],   dtype=torch.float32, device=self.device)
            priv  = torch.tensor(obs['priv'][None],  dtype=torch.float32, device=self.device)
            macro = torch.tensor(obs['macro'][None],  dtype=torch.float32, device=self.device)
            q_dir, q_size, _ = self.online_net(lob, priv, macro)
        self.online_net.train()
        return int(q_dir.argmax(1).item()), int(q_size.argmax(1).item())

    # ------------------------------------------------------------------
    # Training step — BDQ Double-DQN + PER + aux loss
    # ------------------------------------------------------------------

    def learn(self) -> Optional[float]:
        """Sample from buffer, compute BDQ + vol loss, and update online net.

        Returns:
            Total loss as a float, or None if buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        batch, indices, _ = self.buffer.sample(self.batch_size, device=self.device)

        lob   = batch['lob']
        priv  = batch['priv']
        macro = batch['macro']
        nlob  = batch['next_lob']
        npriv = batch['next_priv']
        nmacro = batch['next_macro']

        dir_acts  = batch['dir_acts']    # (B,)
        size_acts = batch['size_acts']   # (B,)
        rewards   = batch['rewards']     # (B,)
        dones     = batch['dones']       # (B,)
        vol_tgts  = batch['vol_tgts']    # (B,)
        is_weights = batch['weights']    # (B,) IS correction

        # -- Online network forward --
        q_dir, q_size, vol_pred = self.online_net(lob, priv, macro)

        # -- Double-DQN targets: select action with online net, eval with target --
        with torch.no_grad():
            # online net picks best next-action for each branch
            nq_dir_online, nq_size_online, _ = self.online_net(nlob, npriv, nmacro)
            next_dir_acts  = nq_dir_online.argmax(1)
            next_size_acts = nq_size_online.argmax(1)

            # target net evaluates those actions
            nq_dir_target, nq_size_target, _ = self.target_net(nlob, npriv, nmacro)
            next_q_dir  = nq_dir_target.gather(1,  next_dir_acts.unsqueeze(1)).squeeze(1)
            next_q_size = nq_size_target.gather(1, next_size_acts.unsqueeze(1)).squeeze(1)

            y_dir  = rewards + self.gamma * next_q_dir  * (1.0 - dones)
            y_size = rewards + self.gamma * next_q_size * (1.0 - dones)

        # -- BDQ Q-loss (both branches, IS-weighted) --
        q_dir_a  = q_dir.gather(1,  dir_acts.unsqueeze(1)).squeeze(1)
        q_size_a = q_size.gather(1, size_acts.unsqueeze(1)).squeeze(1)

        td_dir  = (y_dir  - q_dir_a).pow(2)
        td_size = (y_size - q_size_a).pow(2)
        l_q = (is_weights * (td_dir + td_size) / 2.0).mean()

        # -- Volatility auxiliary loss (Section 4.4) --
        vol_pred_flat = vol_pred.squeeze(1)
        l_vol = F.mse_loss(vol_pred_flat, vol_tgts)

        loss = l_q + self.aux_eta * l_vol

        # -- Backprop --
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # -- Update PER priorities --
        td_errors = ((td_dir + td_size) / 2.0).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors + 1e-6)

        # -- Soft target update --
        for online_p, target_p in zip(
            self.online_net.parameters(), self.target_net.parameters()
        ):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)

        return float(loss.item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights + agent meta to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "steps":      self._steps,
        }, str(p))
        logger.info("Saved agent to %s", path)

    def load(self, path: str) -> None:
        """Load model weights + agent meta from disk."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self._steps  = ckpt.get("steps",   0)
        logger.info("Loaded agent from %s", path)
