"""
colab/deepscalper/agent.py — DeepScalper RL Agent.

Implements:
  • Double DQN update rule with soft-updating target network (τ = 0.01).
  • Prioritized Experience Replay (PER) with SumTree — O(log N) sampling.
  • ε-greedy exploration with linear decay over EPSILON_DECAY_STEPS steps.
  • Adam optimiser with cosine annealing learning rate schedule.
  • Gradient clipping (max_norm = 1.0).
  • Reward function: r = log(P_close / P_entry) − λ × transaction_cost

Usage (inside training notebook):
    agent = DeepScalperAgent(lookback_bars=60, input_dim=11, action_dim=3)
    action = agent.select_action(state)
    agent.store(state, action, reward, next_state, done)
    loss = agent.learn()
    agent.save("AAPL.pth")
"""

import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from colab.deepscalper.architecture import DuelingQNetwork

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prioritized Experience Replay — SumTree
# ---------------------------------------------------------------------------

class _SumTree:
    """Binary SumTree for O(log N) prioritized sampling.

    Args:
        capacity: Maximum number of transitions stored.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write_ptr: int = 0
        self.n_entries: int = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
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
        """Insert a new experience with the given priority."""
        leaf_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(leaf_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        """Update the priority of an existing leaf node."""
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def sample(self, s: float) -> Tuple[int, float, object]:
        """Sample a leaf by traversing the tree for cumulative sum s.

        Returns:
            Tuple of (leaf_idx, priority, data).
        """
        leaf_idx = self._retrieve(0, s)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer backed by a SumTree.

    Args:
        capacity: Maximum number of stored transitions.
        alpha: Priority exponent — how strongly priorities affect sampling.
        beta_start: Initial IS weight exponent (annealed → 1.0 during training).
    """

    _EPSILON: float = 1e-6    # Minimum priority to avoid zero-probability sampling

    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4) -> None:
        self._tree = _SumTree(capacity)
        self._alpha = alpha
        self._beta = beta_start
        self._max_priority: float = 1.0

    def push(self, state, action, reward, next_state, done) -> None:
        """Add a new transition with maximum priority (ensures it's sampled at least once).

        Args:
            state: Current state tensor/array.
            action: Action taken (integer).
            reward: Reward received (float).
            next_state: Next state tensor/array.
            done: Episode termination flag (bool).
        """
        priority = self._max_priority ** self._alpha
        self._tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions using priority-weighted probabilities.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones,
                      importance_weights, leaf_indices)
        """
        total = self._tree.total
        segment = total / batch_size

        leaf_indices, priorities, samples = [], [], []
        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            s = random.uniform(lo, hi)
            s = min(s, total - self._EPSILON)
            idx, priority, data = self._tree.sample(s)
            leaf_indices.append(idx)
            priorities.append(priority)
            samples.append(data)

        # Compute importance-sampling weights
        probs = np.array(priorities) / (total + self._EPSILON)
        is_weights = (self._tree.n_entries * probs) ** (-self._beta)
        is_weights /= is_weights.max()   # Normalise so max weight = 1

        states      = np.array([s[0] for s in samples], dtype=np.float32)
        actions     = np.array([s[1] for s in samples], dtype=np.int64)
        rewards     = np.array([s[2] for s in samples], dtype=np.float32)
        next_states = np.array([s[3] for s in samples], dtype=np.float32)
        dones       = np.array([s[4] for s in samples], dtype=np.float32)

        return states, actions, rewards, next_states, dones, is_weights.astype(np.float32), leaf_indices

    def update_priorities(self, leaf_indices: list, td_errors: np.ndarray) -> None:
        """Update priorities for sampled transitions after learning.

        Args:
            leaf_indices: Leaf indices returned by sample().
            td_errors: Absolute TD errors for the corresponding transitions.
        """
        for idx, err in zip(leaf_indices, td_errors):
            priority = (abs(err) + self._EPSILON) ** self._alpha
            self._tree.update(idx, priority)
            self._max_priority = max(self._max_priority, priority)

    def anneal_beta(self, step: int, total_steps: int) -> None:
        """Linearly anneal beta from beta_start → 1.0.

        Args:
            step: Current training step.
            total_steps: Total expected training steps.
        """
        fraction = min(1.0, step / total_steps)
        self._beta = 0.4 + fraction * 0.6   # 0.4 → 1.0

    def __len__(self) -> int:
        return self._tree.n_entries


# ---------------------------------------------------------------------------
# DeepScalper Agent
# ---------------------------------------------------------------------------

class DeepScalperAgent:
    """Full DeepScalper RL Agent with Double DQN + PER.

    Args:
        lookback_bars: Sequence length fed to the network.
        input_dim: Feature count per bar.
        action_dim: Number of discrete actions (3).
        hidden_size: LSTM hidden state size.
        fc_size: First FC width in dueling head.
        dropout_rate: Dropout probability.
        lr: Adam learning rate.
        gamma: Discount factor.
        batch_size: Minibatch size for each learning step.
        buffer_capacity: PER buffer capacity.
        target_update_freq: Steps between soft-target updates.
        tau: Soft-update coefficient (target ← τ·online + (1−τ)·target).
        epsilon_start: Initial exploration rate.
        epsilon_end: Final (minimum) exploration rate.
        epsilon_decay_steps: Steps over which ε decays linearly.
        per_alpha: PER priority exponent.
        per_beta_start: PER IS-weight annealing start.
        transaction_cost_lambda: Cost coefficient in reward function.
        device: PyTorch device string.
    """

    def __init__(
        self,
        lookback_bars: int,
        input_dim: int,
        action_dim: int = 3,
        hidden_size: int = 128,
        fc_size: int = 256,
        dropout_rate: float = 0.2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_update_freq: int = 100,
        tau: float = 0.01,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10_000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        transaction_cost_lambda: float = 0.0001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:

        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.transaction_cost_lambda = transaction_cost_lambda
        self.device = torch.device(device)

        # Online network (trained with gradient descent)
        self.online_net = DuelingQNetwork(
            lookback_bars, input_dim, action_dim, hidden_size, fc_size, dropout_rate
        ).to(self.device)

        # Target network (periodically synced via soft-update)
        self.target_net = DuelingQNetwork(
            lookback_bars, input_dim, action_dim, hidden_size, fc_size, dropout_rate
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimiser + LR scheduler
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10_000, eta_min=lr * 0.1
        )

        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(buffer_capacity, per_alpha, per_beta_start)

        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """ε-greedy action selection.

        Args:
            state: Numpy array of shape (lookback_bars, input_dim).
            eval_mode: If True, always returns the greedy action (ε = 0).

        Returns:
            Action index in {0, 1, 2}.
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer.

        Args:
            state: Current state array (lookback_bars, input_dim).
            action: Action taken.
            reward: Reward received.
            next_state: Next state array.
            done: True if the episode ended on this step.
        """
        self.memory.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def learn(self) -> Optional[float]:
        """Sample a batch from PER and perform one gradient update.

        Returns:
            Mean batch loss as a Python float, or None if the buffer is
            too small to form a full batch.
        """
        if len(self.memory) < self.batch_size:
            return None

        self._step_count += 1
        self.memory.anneal_beta(self._step_count, total_steps=50_000)

        # Sample batch
        states, actions, rewards, next_states, dones, is_weights, leaf_idxs = \
            self.memory.sample(self.batch_size)

        # Convert to tensors
        s  = torch.from_numpy(states).float().to(self.device)
        a  = torch.from_numpy(actions).long().to(self.device)
        r  = torch.from_numpy(rewards).float().to(self.device)
        ns = torch.from_numpy(next_states).float().to(self.device)
        d  = torch.from_numpy(dones).float().to(self.device)
        w  = torch.from_numpy(is_weights).float().to(self.device)

        # Current Q-values from online network
        q_current = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN: online selects action, target evaluates it
        with torch.no_grad():
            next_actions = self.online_net(ns).argmax(dim=1, keepdim=True)
            q_next = self.target_net(ns).gather(1, next_actions).squeeze(1)
            q_target = r + self.gamma * q_next * (1.0 - d)

        # TD errors for PER priority updates
        td_errors = (q_current.detach() - q_target.detach()).abs().cpu().numpy()
        self.memory.update_priorities(leaf_idxs, td_errors)

        # Weighted Huber loss
        loss_elementwise = F.smooth_l1_loss(q_current, q_target, reduction="none")
        loss = (loss_elementwise * w).mean()

        # Gradient update
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # ε decay
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        # Soft-update target network
        if self._step_count % self.target_update_freq == 0:
            self._soft_update()

        return float(loss.item())

    def _soft_update(self) -> None:
        """Soft-update target network: θ_target ← τ·θ_online + (1−τ)·θ_target."""
        for online_p, target_p in zip(
            self.online_net.parameters(), self.target_net.parameters()
        ):
            target_p.data.copy_(
                self.tau * online_p.data + (1.0 - self.tau) * target_p.data
            )

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_reward(
        entry_price: float,
        exit_price: float,
        held: bool,
        transaction_cost_lambda: float = 0.0001,
    ) -> float:
        """Compute the DeepScalper reward for a completed trade.

        Reward = log(P_exit / P_entry) − λ × transaction_cost
        where transaction_cost = 2 (one round-trip = entry + exit).

        Args:
            entry_price: Price at position open.
            exit_price: Price at position close.
            held: True if the agent held a position (incurs transaction cost).
            transaction_cost_lambda: λ coefficient for transaction cost.

        Returns:
            Scalar reward value.
        """
        if not held or entry_price <= 0:
            return 0.0
        log_return = np.log(exit_price / (entry_price + 1e-10))
        cost = transaction_cost_lambda * 2   # Round-trip cost
        return float(log_return - cost)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the online network's state_dict to disk.

        Args:
            path: File path for the .pth weight file.
        """
        torch.save(self.online_net.state_dict(), path)
        logger.info("Saved model weights → %s", path)

    def load(self, path: str) -> None:
        """Load weights into the online network from a .pth file.

        Args:
            path: File path to the .pth weight file.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.online_net.eval()
        self.target_net.eval()
        logger.info("Loaded model weights ← %s", path)
