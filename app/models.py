from datetime import datetime, timezone
from app.extensions import db


class Maze(db.Model):
    """A saved maze definition with its grid, start, and target cells."""

    __tablename__ = 'maze'

    # 8-char truncated UUID4.  Collision probability is negligible for capstone
    # scale; full UUIDs or auto-incrementing integers would be safer at volume.
    id = db.Column(db.String(8), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    rows = db.Column(db.Integer, nullable=False)
    cols = db.Column(db.Integer, nullable=False)
    grid = db.Column(db.JSON, nullable=False)
    start = db.Column(db.JSON, nullable=False)
    target = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    runs = db.relationship('TrainingRun', backref='maze_obj', lazy=True,
                           cascade='all, delete-orphan')

    def __repr__(self):
        return f"<Maze {self.id} {self.name!r}>"

    def to_summary_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'rows': self.rows,
            'cols': self.cols,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    def to_full_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'rows': self.rows,
            'cols': self.cols,
            'grid': self.grid,
            'start': self.start,
            'target': self.target,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class TrainingRun(db.Model):
    """A single DQN training job, including its hyperparameter config and results.

    maze_name is denormalised from the Maze row intentionally: it preserves the
    name as it was at run-time so historical records remain accurate even if the
    maze is later renamed.

    Deleting a TrainingRun cascades to all associated TrainingEpoch rows.
    """

    __tablename__ = 'training_run'

    # 8-char truncated UUID4 — same collision caveat as Maze.id above.
    id = db.Column(db.String(8), primary_key=True)
    maze_id = db.Column(db.String(8), db.ForeignKey('maze.id'), nullable=False)
    maze_name = db.Column(db.String(100), nullable=False)
    config = db.Column(db.JSON, nullable=False)
    status = db.Column(db.String(20), nullable=False, default='pending')
    stop_reason = db.Column(db.String(50))
    completion_check = db.Column(db.Boolean)
    final_win_rate = db.Column(db.Float)
    epochs_run = db.Column(db.Integer, default=0)
    weights_path = db.Column(db.String(256))
    elapsed_seconds = db.Column(db.Float)
    error = db.Column(db.Text)
    cancel_requested = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime)

    epochs = db.relationship('TrainingEpoch', backref='run', lazy=True,
                             order_by='TrainingEpoch.epoch',
                             cascade='all, delete-orphan')

    def __repr__(self):
        return f"<TrainingRun {self.id} status={self.status!r}>"

    def to_summary_dict(self):
        return {
            'id': self.id,
            'maze_id': self.maze_id,
            'maze_name': self.maze_name,
            'status': self.status,
            'epochs_run': self.epochs_run,
            'final_win_rate': self.final_win_rate,
            'has_weights': bool(self.weights_path),
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    def to_dict(self):
        return {
            'id': self.id,
            'maze_id': self.maze_id,
            'maze_name': self.maze_name,
            'status': self.status,
            'config': self.config,
            'epochs_run': self.epochs_run,
            'final_win_rate': self.final_win_rate,
            'completion_check': self.completion_check,
            'stop_reason': self.stop_reason,
            'elapsed_seconds': self.elapsed_seconds,
            'has_weights': bool(self.weights_path),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }


class TrainingEpoch(db.Model):
    """Per-epoch metrics snapshot for a TrainingRun."""

    __tablename__ = 'training_epoch'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    run_id = db.Column(db.String(8), db.ForeignKey('training_run.id'), nullable=False)
    epoch = db.Column(db.Integer, nullable=False)
    loss = db.Column(db.Float)
    win_rate = db.Column(db.Float)
    epsilon = db.Column(db.Float)
    n_episodes = db.Column(db.Integer)
    progress = db.Column(db.Float)

    def __repr__(self):
        return f"<TrainingEpoch run={self.run_id} epoch={self.epoch}>"
