"""
Maze environment for DQN training.

State encoding convention: 0.0 = wall, 1.0 = free cell.  Two sentinel
values fall strictly between those extremes so the model can distinguish
cell types:
  visited_mark (0.8) — cell the agent has already stepped on
  pirate_mark  (0.5) — current agent position

Reward shaping is designed to encourage short, loop-free paths to the
target. See get_reward() for the full schedule.
"""
import numpy as np

# Sentinel float values that must sit strictly between 0.0 (wall) and
# 1.0 (free) so the model can distinguish cell types; their exact values
# within that range are arbitrary.
visited_mark = 0.8
pirate_mark = 0.5

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3
# Number of discrete actions; defined here since the action space belongs to
# the environment, not the agent.
NUM_ACTIONS = 4


class TreasureMaze:
    """
    Maze environment for the DQN agent.

    The maze is a 2-D NumPy array where 1.0 = free cell and 0.0 = wall.
    The agent (pirate) navigates to the target cell to win. A per-game reward
    budget (min_reward) terminates the episode early when exhausted, preventing
    infinite loops on unsolvable starting positions.
    """

    def __init__(self, maze, pirate=(0, 0), target=None):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = tuple(target) if target is not None else (nrows - 1, ncols - 1)
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        # Exclude the target from valid starting positions: starting there
        # would trivially "win" without any navigation and skew training.
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if pirate not in self.free_cells:
            raise Exception("Invalid pirate location: must sit on a free cell")
        self.reset(pirate)

    def reset(self, pirate):
        self.pirate = pirate
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = pirate
        self.maze[row, col] = pirate_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        """Apply action and advance the pirate's position and mode.

        Mode is 'valid' on a legal move, 'invalid' when the action would
        cross a wall or boundary (position unchanged), and 'blocked' when no
        valid actions exist from the current cell.  Visited cells are stamped
        with visited_mark (0.8) so the model can observe its own path history.
        """
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = pirate_row, pirate_col, mode = self.state

        if self.maze[pirate_row, pirate_col] > 0.0:
            self.visited.add((pirate_row, pirate_col))
            self.maze[pirate_row, pirate_col] = visited_mark

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            elif action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:
            nmode = 'invalid'

        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        """Return the shaped reward for the current state.

        +1.0  — reached the target (win)
        −0.04 — valid step (small step penalty encourages shorter paths)
        −0.5  — revisiting a cell (discourages loops)
        −0.75 — invalid action, i.e. walked into a wall (harder penalty than revisit)
        min_reward−1 — all actions blocked (effectively terminal)
        """
        pirate_row, pirate_col, mode = self.state
        if (pirate_row, pirate_col) == self.target:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (pirate_row, pirate_col) in self.visited:
            return -0.5
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        """Return the current state as a (1, N) array for direct model input."""
        canvas = self._build_observation_grid()
        return canvas.reshape((1, -1))

    def _build_observation_grid(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        for r in range(nrows):
            for c in range(ncols):
                # Reset only the pirate_mark from reset() back to a free cell;
                # visited_mark (0.8) is preserved so the model can see visit history.
                if canvas[r, c] == pirate_mark:
                    canvas[r, c] = 1.0
        row, col, valid = self.state
        canvas[row, col] = pirate_mark
        return canvas

    def game_status(self):
        pirate_row, pirate_col, mode = self.state
        if (pirate_row, pirate_col) == self.target:
            return 'win'
        if self.total_reward < self.min_reward:
            return 'lose'
        return 'not_over'

    def valid_actions(self, cell=None):
        """Return the list of legal actions from cell (or the current pirate position).

        An action is removed if it would move outside the grid boundary or into
        a wall cell (value 0.0).  visited_mark (0.8) cells are still passable —
        the penalty is applied via get_reward, not by removing the action.
        """
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows - 1:
            actions.remove(3)
        if col == 0:
            actions.remove(0)
        elif col == ncols - 1:
            actions.remove(2)
        if row > 0 and self.maze[row - 1, col] == 0.0:
            actions.remove(1)
        if row < nrows - 1 and self.maze[row + 1, col] == 0.0:
            actions.remove(3)
        if col > 0 and self.maze[row, col - 1] == 0.0:
            actions.remove(0)
        if col < ncols - 1 and self.maze[row, col + 1] == 0.0:
            actions.remove(2)
        return actions
