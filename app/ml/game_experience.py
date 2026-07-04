"""
Prioritized Experience Replay (PER) buffer for Double DQN training.

Full pipeline:
  1. Transitions are stored in a SumTree keyed by |TD-error|^alpha priority.
  2. sample() does stratified sampling across batch_size equal-width segments
     of the total priority mass — O(log n) per draw via the sum-tree.
  3. compute_targets() accepts pre-computed Q-value arrays from the caller
     (keeping this class free of neural-network calls) and returns DDQN
     targets plus per-sample TD errors for priority updates.
  4. IS weights (beta annealed from beta_start toward 1.0) correct for the
     non-uniform sampling bias introduced by PER.
"""
import numpy as np


class SumTree:
    """Array-backed binary tree where leaf values are priorities and internal nodes store partial sums.
    Enables O(log n) priority updates and O(log n) stratified sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data = [None] * capacity
        self._write = 0
        self._n_entries = 0

    def _propagate(self, idx, delta):
        while idx != 0:
            parent = (idx - 1) // 2
            self._tree[parent] += delta
            idx = parent

    def update(self, idx, priority):
        """Update leaf at data index idx and propagate the change up to the root."""
        tree_idx = idx + self.capacity - 1
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def add(self, priority, data):
        """Insert data with given priority, overwriting the oldest entry when full."""
        self._data[self._write] = data
        self.update(self._write, priority)
        self._write = (self._write + 1) % self.capacity
        self._n_entries = min(self._n_entries + 1, self.capacity)

    def sample(self, value):
        """Traverse from root to find the leaf whose cumulative range contains value.
        Returns (data_idx, priority, data)."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = left + 1
        data_idx = idx - (self.capacity - 1)
        return data_idx, self._tree[idx], self._data[data_idx]

    @property
    def total(self):
        return float(self._tree[0])

    def __len__(self):
        return self._n_entries


class GameExperience:
    """
    Prioritized Experience Replay (PER) buffer using a sum-tree.

    This class is a pure data structure: it stores transitions, samples them
    with priority weighting, and computes DDQN targets from caller-supplied
    Q-value arrays.  All neural-network inference is the caller's responsibility,
    keeping this class decoupled from the model implementation.
    """

    # Added to |TD error| before raising to alpha so no transition ever gets
    # zero priority (which would make it permanently invisible to the sampler).
    _PER_EPSILON = 1e-6

    def __init__(self, max_memory=100, discount=0.95,
                 alpha=0.6, beta_start=0.4, beta_end=1.0, beta_steps=None):
        self.max_memory = max_memory
        self.discount = discount
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        # Default: anneal beta over ~20 full-buffer sweeps so IS weights are
        # near-unbiased by the end of training.  The multiplier 20 is a
        # heuristic; pass beta_steps explicitly to override it.
        self.beta_increment = (beta_end - beta_start) / max(beta_steps or max_memory * 20, 1)
        self._tree = SumTree(max_memory)
        self._max_priority = 1.0  # tracks max p^alpha seen; new transitions receive this priority

    def remember(self, episode):
        """Store a transition with maximum current priority so it is sampled at least once."""
        self._tree.add(self._max_priority, episode)

    def sample(self, batch_size=32):
        """
        Stratified sampling: divide [0, total_priority] into batch_size equal segments
        and draw one value uniformly from each.

        Returns (indices, experiences, is_weights) where indices are data-array positions
        needed to update priorities after the training step.
        """
        n = len(self._tree)
        if n == 0:
            raise RuntimeError("sample() called on empty replay buffer")

        total = self._tree.total
        segment = total / batch_size
        indices, experiences, priorities = [], [], []

        for i in range(batch_size):
            value = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self._tree.sample(min(value, total - 1e-10))
            indices.append(idx)
            experiences.append(data)
            # 1e-10 here is a numerical safety floor for IS weight computation
            # (prevents division by zero); it differs from _PER_EPSILON, which
            # prevents zero-priority storage.  Duplicate indices are possible
            # when the buffer is nearly empty — an accepted PER approximation.
            priorities.append(max(float(priority), 1e-10))

        priorities = np.array(priorities, dtype=np.float64)
        # P(i) = stored_priority[i] / total; w_i = (1 / N*P(i))^beta, normalised by max weight
        is_weights = (1.0 / (n * priorities / total)) ** self.beta
        is_weights = (is_weights / is_weights.max()).astype(np.float32)

        return indices, experiences, is_weights

    def compute_targets(self, batch, q_values, q_next_main, q_next_target, is_weights):
        """
        Compute DDQN targets and TD errors from caller-supplied Q-value arrays (vectorized).

        DDQN rule: main network selects best next action; target network evaluates it.
        IS weights scale each sample's update to correct the prioritized-sampling bias.

        Returns (targets, td_errors) where targets is (batch_size, num_actions) float32
        and td_errors is (batch_size,) for priority updates via update_priorities().
        """
        batch_size = len(batch)
        actions_arr = np.array([b[1] for b in batch], dtype=int)
        rewards_arr = np.array([b[2] for b in batch], dtype=np.float32)
        dones_arr = np.array([b[4] for b in batch], dtype=bool)

        best_next = np.argmax(q_next_main, axis=1)
        next_q = q_next_target[np.arange(batch_size), best_next]
        td_target = rewards_arr + self.discount * next_q * (~dones_arr)

        raw_targets = q_values.copy()
        raw_targets[np.arange(batch_size), actions_arr] = td_target

        td_errors = np.abs(
            raw_targets[np.arange(batch_size), actions_arr]
            - q_values[np.arange(batch_size), actions_arr]
        )

        # IS correction: scale the (target − q) delta for each sample.
        # Non-taken action slots have zero delta and are unaffected.
        targets = q_values + is_weights[:, np.newaxis] * (raw_targets - q_values)
        return targets.astype(np.float32), td_errors

    def update_priorities(self, indices, td_errors):
        """Recompute priorities as (|td_error| + ε)^alpha and update the sum-tree."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self._PER_EPSILON) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            self._tree.update(idx, priority)
