import uuid
import numpy as np
from flask import Blueprint, request, jsonify

from app.extensions import db
from app.models import Maze
from app.algorithms.bfs_solver import BFSSolver

api_mazes_bp = Blueprint('api_mazes', __name__, url_prefix='/api/mazes')

_MAX_MAZE_ROWS = 20
_MAX_MAZE_COLS = 20


@api_mazes_bp.get('/')
def get_mazes():
    """Return summary info for all saved mazes, ordered by creation time."""
    mazes = db.session.execute(db.select(Maze).order_by(Maze.created_at)).scalars().all()
    return jsonify([m.to_summary_dict() for m in mazes])


@api_mazes_bp.post('/')
def create_maze():
    """Create and save a new maze. Expects JSON: {grid, name, start, target}.

    grid must be a rectangular 2-D list of 0/1 values with at most
    20 rows and 20 columns.  Returns {id, name} on success.
    """
    body = request.get_json(silent=True) or {}
    grid = body.get('grid')
    name = body.get('name', 'Untitled Maze')

    if not grid or not isinstance(grid, list):
        return jsonify({'error': 'grid is required'}), 400

    # Validate rectangular grid before converting to NumPy
    row_lengths = {len(row) for row in grid}
    if len(row_lengths) != 1:
        return jsonify({'error': 'grid must be rectangular (all rows same length)'}), 400

    nrows = len(grid)
    ncols = row_lengths.pop()
    if nrows > _MAX_MAZE_ROWS or ncols > _MAX_MAZE_COLS:
        return jsonify({
            'error': f'Maze too large; maximum size is {_MAX_MAZE_ROWS}×{_MAX_MAZE_COLS}'
        }), 400

    arr = np.array(grid, dtype=float)

    start = body.get('start', [0, 0])
    target = body.get('target', [nrows - 1, ncols - 1])
    sr, sc = start
    tr, tc = target

    if not (0 <= sr < nrows and 0 <= sc < ncols):
        return jsonify({'error': 'start is outside grid bounds'}), 400
    if not (0 <= tr < nrows and 0 <= tc < ncols):
        return jsonify({'error': 'target is outside grid bounds'}), 400
    if arr[sr, sc] != 1.0:
        return jsonify({'error': 'Start cell must be free'}), 400
    if arr[tr, tc] != 1.0:
        return jsonify({'error': 'Target cell must be free'}), 400
    if start == target:
        return jsonify({'error': 'Start and target cannot be the same cell'}), 400

    if not BFSSolver(arr, start=start, target=target).is_solvable():
        return jsonify({'error': 'Maze has no path from start to target'}), 422

    maze = Maze(
        id=str(uuid.uuid4())[:8],
        name=name,
        rows=nrows,
        cols=ncols,
        grid=grid,
        start=start,
        target=target,
    )
    db.session.add(maze)
    db.session.commit()

    return jsonify({'id': maze.id, 'name': maze.name}), 201


@api_mazes_bp.get('/<maze_id>')
def get_maze(maze_id):
    """Return the full maze definition including grid data."""
    maze = db.get_or_404(Maze, maze_id)
    return jsonify(maze.to_full_dict())


@api_mazes_bp.delete('/<maze_id>')
def delete_maze(maze_id):
    """Delete a maze and all its associated training runs (cascade)."""
    maze = db.get_or_404(Maze, maze_id)
    db.session.delete(maze)
    db.session.commit()
    return jsonify({'deleted': maze_id})
