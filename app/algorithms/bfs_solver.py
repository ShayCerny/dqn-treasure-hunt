from collections import deque

import numpy as np


class BFSSolver:
    """
    BFS-based shortest-path solver for grid mazes.

    Maze cells with value 1.0 are passable; 0.0 are walls.
    Movements are 4-directional (up, down, left, right).
    """

    _DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, maze: np.ndarray, start=None, target=None):
        rows, cols = maze.shape
        self.maze = np.copy(maze)  # copy so caller mutations don't affect solve()
        self.start = tuple(start) if start is not None else (0, 0)
        self.target = tuple(target) if target is not None else (rows - 1, cols - 1)

    def is_solvable(self) -> bool:
        """Return True if a path from start to target exists.

        Delegates to solve() for simplicity.  For small mazes (≤20×20) the
        overhead of building the full path list is negligible.  A visited-only
        BFS (no path list) would use less memory for pure reachability checks,
        but the added complexity is not warranted at this scale.
        """
        return self.solve() is not None

    def solve(self) -> list | None:
        """
        BFS shortest path from start to target.

        Returns a list of (row, col) tuples from start to target (inclusive),
        or None if no path exists. Time and space complexity: O(rows * cols).
        """
        rows, cols = self.maze.shape
        visited = {self.start}
        queue = deque([(self.start, [self.start])])

        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == self.target:
                return path
            for dr, dc in self._DIRS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and self.maze[nr, nc] == 1.0
                        and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        return None
