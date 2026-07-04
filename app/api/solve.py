import numpy as np
from flask import Blueprint, request, jsonify

from app.algorithms.bfs_solver import BFSSolver

api_solve_bp = Blueprint('api_solve', __name__, url_prefix='/api/solve')


@api_solve_bp.post('/')
def solve():
    """Return the BFS shortest path through the given grid, or null if none exists.

    Expects JSON: {grid, start, target}.  Returns {path, length} on success.
    """
    body = request.get_json(silent=True) or {}
    grid = body.get('grid')
    if not grid or not isinstance(grid, list):
        return jsonify({'error': 'grid is required'}), 400

    arr = np.array(grid, dtype=float)
    nrows, ncols = arr.shape
    start = body.get('start', [0, 0])
    target = body.get('target', [nrows - 1, ncols - 1])

    sr, sc = start
    tr, tc = target
    if not (0 <= sr < nrows and 0 <= sc < ncols):
        return jsonify({'error': 'start is outside grid bounds'}), 400
    if not (0 <= tr < nrows and 0 <= tc < ncols):
        return jsonify({'error': 'target is outside grid bounds'}), 400

    path = BFSSolver(arr, start=start, target=target).solve()
    if path is None:
        return jsonify({'path': None})
    return jsonify({'path': [list(p) for p in path], 'length': len(path)})
