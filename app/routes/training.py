from flask import Blueprint, render_template, abort

from app.extensions import db
from app.models import Maze, TrainingRun
from app.config import TrainingConfig

training_bp = Blueprint('training', __name__)

_TRAINING_DEFAULTS = TrainingConfig().to_dict()


def _maze_list():
    mazes = db.session.execute(db.select(Maze).order_by(Maze.created_at)).scalars().all()
    return [m.to_summary_dict() for m in mazes]


@training_bp.route('/train')
def train():
    return render_template('train.html', mazes=_maze_list(),
                           defaults=_TRAINING_DEFAULTS, watch_run=None)


@training_bp.route('/train/<run_id>')
def train_run(run_id):
    run = db.session.get(TrainingRun, run_id)
    if run is None:
        abort(404)
    return render_template('train.html', mazes=_maze_list(),
                           defaults=_TRAINING_DEFAULTS, watch_run=run.to_dict())


@training_bp.route('/runs')
def runs_page():
    # Default to the 100 most recent runs to keep the page snappy.
    # Deep history is available via the API with ?page= pagination.
    all_runs = db.session.execute(
        db.select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(100)
    ).scalars().all()
    return render_template('runs.html', runs=[r.to_dict() for r in all_runs])
