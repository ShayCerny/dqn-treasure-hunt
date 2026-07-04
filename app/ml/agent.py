import random
import datetime
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import PReLU

from .treasure_maze import TreasureMaze, NUM_ACTIONS
from .game_experience import GameExperience
from app.config import TrainingConfig


class DQNAgent:
    """
    Deep Q-Network agent for maze navigation.

    Maintains a main network (updated every step) and a target network
    (updated periodically) to stabilize Q-value estimates during training.
    """

    def __init__(self, maze_size: int):
        self.maze_size = maze_size
        self.model = None
        self.target_model = None

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self, learning_rate: float = 0.001, hidden_layer_size: int = None):
        """
        Build the Q-network and its target copy.

        hidden_layer_size defaults to maze_size when not specified, which gives
        the network enough capacity to represent every cell's Q-values directly.
        """
        size = hidden_layer_size if hidden_layer_size is not None else self.maze_size
        model = Sequential([
            Dense(size, input_shape=(self.maze_size,)),
            PReLU(),
            Dense(size),
            PReLU(),
            Dense(NUM_ACTIONS),
        ])
        self.model = model
        self.target_model = clone_model(model)
        self.target_model.set_weights(model.get_weights())

        # Create a traced train_step scoped to this model instance so TF
        # doesn't retrace across multiple DQNAgent instances.
        # model.compile() is intentionally omitted: training uses _train_step
        # exclusively, so the compile-time optimizer would be dead code.
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model_ref = self.model

        @tf.function
        def _train_step(x, y):
            with tf.GradientTape() as tape:
                q_values = model_ref(x, training=True)
                loss = loss_fn(y, q_values)
            grads = tape.gradient(loss, model_ref.trainable_variables)
            optimizer.apply_gradients(zip(grads, model_ref.trainable_variables))
            return loss

        self._train_step = _train_step

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, maze_array: np.ndarray, config: TrainingConfig,
              epoch_callback=None, stop_fn=None, maze_target=None):
        """
        Run the full training loop using Double DQN and Prioritized Experience Replay.

        Each epoch the agent plays one game from a random starting cell,
        storing every transition in replay memory and updating the network
        after every step. The DDQN target decouples action selection (main net)
        from evaluation (target net) to reduce Q-value overestimation. PER
        samples high-error transitions more often via a sum-tree; IS weights
        (annealed from beta_start toward 1.0) correct the sampling bias.

        epoch_callback(epoch, loss, win_rate, epsilon, n_episodes) is called
        after each epoch so callers can stream live updates via Socket.IO.
        stop_fn() is polled each epoch; returning True cancels training early.

        Early stopping requires two conditions to both be true for `patience`
        consecutive epochs:
          1. win_rate == 1.0 (agent wins every sampled game)
          2. per-step loss has plateaued — the relative range of losses over
             the patience window is below loss_min_delta, meaning the Q-values
             have converged and the policy is no longer improving.

        Returns a dict with stop_reason ('complete' | 'early_stop' | 'cancelled'),
        epochs_run, final win_rate, and elapsed_seconds.
        """
        self.build_model(
            learning_rate=config.learning_rate,
            hidden_layer_size=config.hidden_layer_size,
        )

        epsilon = config.epsilon
        n_epoch = config.n_epoch
        max_memory = config.max_memory
        batch_size = config.batch_size
        target_update_freq = config.target_update_freq

        qmaze = TreasureMaze(maze_array, target=maze_target)
        experience = GameExperience(max_memory=max_memory,
                                    discount=config.discount,
                                    alpha=config.alpha,
                                    beta_start=config.beta_start)

        win_history = []
        # Win rate is computed over a sliding window of half the maze size
        # so it reflects recent performance rather than the full training history.
        hsize = qmaze.maze.size // 2
        win_rate = 0.0
        consecutive_perfect = 0
        # deque with maxlen automatically drops the oldest entry — O(1) vs
        # O(n) for list.pop(0) — and removes the need for a manual length guard.
        loss_window = collections.deque(maxlen=config.patience)
        stop_reason = 'complete'

        start_time = datetime.datetime.now()

        for epoch in range(n_epoch):
            loss = 0.0

            agent_cell = random.choice(qmaze.free_cells)
            qmaze.reset(agent_cell)
            env_state = qmaze.observe()
            n_episodes = 0

            while qmaze.game_status() == 'not_over':
                previous_envstate = env_state
                valid_actions_list = qmaze.valid_actions()

                # Epsilon-greedy: call model directly (GameExperience is a pure buffer)
                if np.random.rand() < epsilon:
                    action = random.choice(valid_actions_list)
                else:
                    q_vals = self.model(
                        np.asarray(env_state, dtype=np.float32), training=False
                    ).numpy()[0]
                    action = int(max(valid_actions_list, key=lambda a: q_vals[a]))

                env_state, reward, game_status = qmaze.act(action)

                episode = [previous_envstate, action, reward, env_state, game_status != 'not_over']
                experience.remember(episode)
                n_episodes += 1

                # Sample a batch and compute all Q-values in the agent layer.
                # Combining states and next_states into one forward pass halves
                # the number of main-net kernel dispatches per training step.
                indices, batch, is_weights = experience.sample(batch_size)
                states = np.vstack([b[0] for b in batch]).astype(np.float32)
                next_states = np.vstack([b[3] for b in batch]).astype(np.float32)

                q_both = self.model(
                    np.vstack([states, next_states]), training=False
                ).numpy()
                q_values_batch = q_both[:batch_size]
                q_next_main = q_both[batch_size:]
                q_next_target = self.target_model(next_states, training=False).numpy()

                targets, td_errors = experience.compute_targets(
                    batch, q_values_batch, q_next_main, q_next_target, is_weights
                )
                batch_loss = self._train_step(states, targets)
                loss += float(batch_loss)
                experience.update_priorities(indices, td_errors)
                experience.beta = min(experience.beta_end,
                                      experience.beta + experience.beta_increment)

            if qmaze.game_status() == 'win':
                win_history.append(1)
            else:
                win_history.append(0)

            # Periodically sync the target network to the main network.
            # The lag between the two networks stabilizes Q-target computation.
            if epoch % target_update_freq == 0:
                self.target_model.set_weights(self.model.get_weights())

            win_rate = (sum(win_history[-hsize:]) / hsize
                        if len(win_history) >= hsize else 0.0)

            # Once the agent is winning reliably, collapse epsilon to its floor
            # to maximize exploitation of learned policy.
            if win_rate > 0.9:
                epsilon = config.epsilon_min
            else:
                epsilon = max(epsilon * config.epsilon_decay, config.epsilon_min)

            if epoch_callback:
                epoch_callback(epoch, loss, win_rate, epsilon, n_episodes)

            if stop_fn and stop_fn():
                stop_reason = 'cancelled'
                break

            # --- early-stop bookkeeping ---

            if win_rate == 1.0:
                consecutive_perfect += 1
            else:
                consecutive_perfect = 0

            # Normalize loss by training steps so it's comparable across epochs
            # regardless of how long the episode ran.
            avg_loss = loss / max(n_episodes, 1)
            loss_window.append(avg_loss)

            # Loss has plateaued when the relative range (max-min)/max within
            # the patience window falls below loss_min_delta.  This ensures the
            # network's Q-values have converged, not just that it is winning.
            peak = max(loss_window)
            loss_plateaued = (
                len(loss_window) >= config.patience
                and peak > 1e-8
                and (peak - min(loss_window)) / peak < config.loss_min_delta
            )

            if consecutive_perfect >= config.patience and loss_plateaued:
                stop_reason = 'early_stop'
                break

        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        return {'epochs_run': epoch + 1, 'win_rate': win_rate,
                'elapsed_seconds': elapsed, 'stop_reason': stop_reason}

    # ------------------------------------------------------------------
    # Game evaluation helpers
    # ------------------------------------------------------------------

    def _greedy_loop(self, qmaze: TreasureMaze, max_steps: int):
        """
        Shared greedy action generator used by play_game and trace_game.

        Yields (row, col, game_status) for each step.  The caller decides
        when to stop based on game_status so the two callers can apply
        different termination conditions without duplicating the inner loop.
        """
        envstate = qmaze.observe()
        for _ in range(max_steps):
            state = np.asarray(envstate, dtype=np.float32)
            q_values = self.model(state, training=False).numpy()[0]
            valid = qmaze.valid_actions()
            action = (int(max(valid, key=lambda a: q_values[a]))
                      if valid else int(np.argmax(q_values)))
            envstate, _, game_status = qmaze.act(action)
            row, col, _ = qmaze.state
            yield row, col, game_status

    def play_game(self, qmaze: TreasureMaze, pirate_cell: tuple,
                  max_steps: int = None) -> bool:
        """
        Run one greedy game from pirate_cell and return True if the agent wins.
        max_steps defaults to 4× the maze size to bound the loop.

        Pure greedy: the network's Q-values drive every action choice with no
        heuristic overrides.  'lose' (reward-budget exhausted) is intentionally
        ignored so the min_reward training shortcut does not cut evaluation short
        on harder starting cells — only reaching the target or hitting max_steps
        ends the loop.
        """
        qmaze.reset(pirate_cell)
        if max_steps is None:
            max_steps = qmaze.maze.size * 4
        for _, _, game_status in self._greedy_loop(qmaze, max_steps):
            if game_status == 'win':
                return True
            # 'lose' ignored — evaluation is not about reward efficiency
        return False

    def completion_check(self, maze_array: np.ndarray, maze_target=None) -> bool:
        """
        Return True only if the agent wins from every valid starting cell.
        This is a stricter evaluation than average win rate.
        """
        qmaze = TreasureMaze(maze_array, target=maze_target)
        for cell in qmaze.free_cells:
            # Skip cells with no valid actions: they are geometrically isolated
            # (surrounded on all sides by walls) and unsolvable by any policy.
            if not qmaze.valid_actions(cell):
                continue
            if not self.play_game(qmaze, cell):
                return False
        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def trace_game(self, qmaze: TreasureMaze, pirate_cell: tuple,
                   max_steps: int = None) -> tuple:
        """
        Run one greedy game and return (steps, outcome).
        steps is a list of [row, col] positions visited.
        outcome is 'win' or 'lose'.
        """
        qmaze.reset(pirate_cell)
        steps = [list(pirate_cell)]
        if max_steps is None:
            max_steps = qmaze.maze.size * 4
        for row, col, game_status in self._greedy_loop(qmaze, max_steps):
            steps.append([row, col])
            if game_status in ('win', 'lose'):
                break
        return steps, qmaze.game_status()

    def save_weights(self, path: str):
        if self.model:
            self.model.save_weights(path)

    def load_weights(self, path: str):
        if self.model:
            self.model.load_weights(path)
