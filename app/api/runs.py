import uuid
from datetime import datetime, timezone
import numpy as np
from flask import Blueprint, request, jsonify, current_app

from app import socketio
from app.extensions import db
from app.models import Maze, TrainingRun, TrainingEpoch
from app.config import TrainingConfig
from app.services.training import TrainingService
from app.ml.treasure_maze import TreasureMaze

api_runs_bp = Blueprint('api_runs', __name__, url_prefix='/api/runs')

# In-memory cancel flags keyed by run_id.  The cancel endpoint sets the flag;
# stop_fn() reads it without a DB round-trip.  The flag is cleaned up when
# training finishes.
_cancel_flags: dict = {}


def _run_training(app, run_id: str, maze_array: np.ndarray,
                  config: TrainingConfig, models_dir: str,
                  maze_target=None):
    """
    Background task that orchestrates one DQN training run.

    DB writes and Socket.IO emissions stay here (API concerns); the pure
    training logic lives in TrainingService so it can be tested independently.
    Commits TrainingEpoch rows every 25 epochs (and on the final epoch) to
    reduce write pressure while still emitting epoch_update every epoch so
    the live chart stays responsive.
    """
    with app.app_context():
        run_obj = db.session.get(TrainingRun, run_id)
        run_obj.status = 'running'
        db.session.commit()

        n_epoch = config.n_epoch
        service = TrainingService()

        def on_epoch(epoch, loss, win_rate, epsilon, n_episodes):
            payload = {
                'run_id': run_id,
                'epoch': epoch,
                'loss': round(loss, 4),
                'win_rate': round(win_rate, 4),
                'epsilon': round(epsilon, 4),
                'n_episodes': n_episodes,
                'progress': round((epoch + 1) / n_epoch * 100, 1),
            }
            db.session.add(TrainingEpoch(
                run_id=run_id,
                epoch=epoch,
                loss=round(loss, 4),
                win_rate=round(win_rate, 4),
                epsilon=round(epsilon, 4),
                n_episodes=n_episodes,
                progress=round((epoch + 1) / n_epoch * 100, 1),
            ))
            # Batch DB commits to reduce write pressure: commit every 25 epochs
            # and always on the last epoch.  Socket.IO still emits every epoch
            # so the live chart stays responsive.
            if epoch % 25 == 0 or epoch == n_epoch - 1:
                db.session.commit()
            socketio.emit('epoch_update', payload)

        def stop_fn():
            # Read the in-memory flag rather than hitting the DB on every epoch.
            return _cancel_flags.get(run_id, False)

        try:
            result, weights_path, passed = service.run(
                maze_array, config, models_dir, run_id, maze_target,
                epoch_callback=on_epoch, stop_fn=stop_fn,
            )

            stop_reason = result['stop_reason']
            run_obj.weights_path = weights_path
            run_obj.epochs_run = result['epochs_run']
            run_obj.final_win_rate = result['win_rate']
            run_obj.elapsed_seconds = result['elapsed_seconds']
            run_obj.stop_reason = stop_reason
            run_obj.completed_at = datetime.now(timezone.utc)

            if stop_reason == 'cancelled':
                run_obj.status = 'cancelled'
                db.session.commit()
                socketio.emit('training_complete', {
                    'run_id': run_id,
                    'stop_reason': 'cancelled',
                    'epochs_run': result['epochs_run'],
                    'final_win_rate': result['win_rate'],
                })
            else:
                run_obj.status = 'complete'
                run_obj.completion_check = passed
                db.session.commit()
                socketio.emit('training_complete', {
                    'run_id': run_id,
                    'stop_reason': stop_reason,
                    'completion_check': passed,
                    'epochs_run': result['epochs_run'],
                    'final_win_rate': result['win_rate'],
                })

        except Exception as e:
            run_obj.status = 'error'
            run_obj.error = str(e)
            db.session.commit()
            socketio.emit('training_error', {'run_id': run_id, 'error': str(e)})
        finally:
            _cancel_flags.pop(run_id, None)


@api_runs_bp.post('/')
def start_run():
    """
    Start a new training run. Expects JSON: { maze_id, config }.
    Creates a TrainingRun row, launches _run_training as a background
    thread, and returns the new run_id immediately (202 Accepted).
    """
    body = request.get_json(silent=True) or {}
    maze_id = body.get('maze_id')
    config_data = body.get('config', {})

    if not maze_id:
        return jsonify({'error': 'maze_id is required'}), 400

    maze = db.session.get(Maze, maze_id)
    if maze is None:
        return jsonify({'error': 'Maze not found'}), 404

    try:
        config = TrainingConfig.from_dict(config_data)
    except (ValueError, TypeError) as e:
        return jsonify({'error': str(e)}), 400

    maze_array = np.array(maze.grid, dtype=float)
    maze_target = tuple(maze.target)

    run_id = str(uuid.uuid4())[:8]
    run_obj = TrainingRun(
        id=run_id,
        maze_id=maze_id,
        maze_name=maze.name,
        config=config.to_dict(),
        status='pending',
        cancel_requested=False,
    )
    db.session.add(run_obj)
    db.session.commit()

    models_dir = current_app.config['MODELS_DIR']
    app = current_app._get_current_object()
    socketio.start_background_task(
        _run_training, app, run_id, maze_array, config, models_dir, maze_target
    )

    return jsonify({'run_id': run_id}), 202


@api_runs_bp.get('/')
def get_runs():
    """Return summary info for training runs, newest first.

    Query params:
      ?limit (int, default 50, max 200) — number of runs to return
      ?page  (int, default 1)           — 1-based page number
    """
    try:
        limit = min(int(request.args.get('limit', 50)), 200)
        page = max(int(request.args.get('page', 1)), 1)
    except (ValueError, TypeError):
        return jsonify({'error': 'limit and page must be integers'}), 400
    offset = (page - 1) * limit
    runs = db.session.execute(
        db.select(TrainingRun).order_by(TrainingRun.created_at.desc())
        .limit(limit).offset(offset)
    ).scalars().all()
    return jsonify([r.to_summary_dict() for r in runs])


@api_runs_bp.post('/<run_id>/cancel')
def cancel_run(run_id):
    """Request cancellation of a running training job."""
    run_obj = db.session.get(TrainingRun, run_id)
    if run_obj is None:
        return jsonify({'error': 'Not found'}), 404
    if run_obj.status != 'running':
        return jsonify({'error': 'Run is not active'}), 400
    run_obj.cancel_requested = True
    _cancel_flags[run_id] = True
    db.session.commit()
    return jsonify({'ok': True})


@api_runs_bp.get('/<run_id>/play')
def play_run(run_id):
    """
    Load saved weights for a completed run and trace one greedy game,
    returning both the DQN path and the BFS optimal path for comparison.

    Optional query param ?start=row,col to override the default start position.
    """
    from app.ml.agent import DQNAgent
    from app.algorithms.bfs_solver import BFSSolver

    run_obj = db.session.get(TrainingRun, run_id)
    if run_obj is None:
        return jsonify({'error': 'Not found'}), 404

    weights_path = run_obj.weights_path
    if not weights_path or not __import__('os').path.exists(weights_path):
        return jsonify({'error': 'No saved weights for this run'}), 404

    maze = db.session.get(Maze, run_obj.maze_id)
    if maze is None:
        return jsonify({'error': 'Maze not found'}), 404

    maze_array = np.array(maze.grid, dtype=float)
    target = tuple(maze.target)
    default_start = tuple(maze.start)

    raw_start = request.args.get('start')
    if raw_start:
        try:
            r, c = (int(x) for x in raw_start.split(','))
            start = (r, c)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid start parameter'}), 400
    else:
        start = default_start

    stored_config = run_obj.config or {}
    agent = DQNAgent(maze_size=maze_array.size)
    agent.build_model(
        learning_rate=stored_config.get('learning_rate', 0.001),
        hidden_layer_size=stored_config.get('hidden_layer_size', None),
    )
    agent.load_weights(weights_path)

    qmaze = TreasureMaze(maze_array, pirate=start, target=target)
    steps, outcome = agent.trace_game(qmaze, start)

    pruned = [steps[0]]
    for pos in steps[1:]:
        if pos != pruned[-1]:
            pruned.append(pos)

    bfs_result = BFSSolver(maze_array, start=start, target=target).solve()
    bfs_path = [list(p) for p in bfs_result] if bfs_result else None

    return jsonify({
        'grid': maze.grid,
        'start': list(start),
        'target': list(target),
        'steps': pruned,
        'outcome': outcome,
        'bfs_path': bfs_path,
    })


@api_runs_bp.get('/<run_id>/epochs')
def get_epochs(run_id):
    """Return per-epoch metric rows for a training run, ordered by epoch.

    Query param ?limit (int, default 10000, max 50000) caps the result set.
    """
    run_obj = db.session.get(TrainingRun, run_id)
    if run_obj is None:
        return jsonify({'error': 'Not found'}), 404
    try:
        limit = min(int(request.args.get('limit', 10000)), 50000)
    except (ValueError, TypeError):
        return jsonify({'error': 'limit must be an integer'}), 400
    epochs = db.session.execute(
        db.select(TrainingEpoch)
        .where(TrainingEpoch.run_id == run_id)
        .order_by(TrainingEpoch.epoch)
        .limit(limit)
    ).scalars().all()
    return jsonify([{
        'run_id': run_id,
        'epoch': e.epoch,
        'loss': e.loss,
        'win_rate': e.win_rate,
        'epsilon': e.epsilon,
        'n_episodes': e.n_episodes,
        'progress': e.progress,
    } for e in epochs])
