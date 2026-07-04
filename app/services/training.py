"""
Training service layer — DQN agent orchestration without DB or Socket.IO concerns.

Keeping agent training, weight persistence, and completion checks here makes
the API layer thin (HTTP in/out, DB writes, event emissions) and this logic
independently testable.
"""
import os
import numpy as np

from app.ml.agent import DQNAgent
from app.config import TrainingConfig


class TrainingService:
    """Orchestrates one DQN training run: build model, train, save weights, check completion."""

    def run(self, maze_array: np.ndarray, config: TrainingConfig,
            models_dir: str, run_id: str, maze_target=None,
            epoch_callback=None, stop_fn=None):
        """
        Train a DQNAgent on maze_array with the given config.

        epoch_callback and stop_fn are passed through to DQNAgent.train() so
        the caller (API layer) can emit live Socket.IO updates and honour cancel
        requests without this service knowing about either concern.

        Returns (result, weights_path, passed) where:
          result       — dict from DQNAgent.train() with stop_reason, epochs_run, etc.
          weights_path — filesystem path where model weights were saved
          passed       — bool from completion_check(), or None if training was cancelled
        """
        agent = DQNAgent(maze_size=maze_array.size)
        result = agent.train(
            maze_array, config,
            epoch_callback=epoch_callback,
            stop_fn=stop_fn,
            maze_target=maze_target,
        )

        weights_path = os.path.join(models_dir, f"{run_id}.weights.h5")
        agent.save_weights(weights_path)

        passed = None
        if result['stop_reason'] != 'cancelled':
            passed = agent.completion_check(maze_array, maze_target=maze_target)

        return result, weights_path, passed
