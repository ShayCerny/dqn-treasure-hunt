from flask import Blueprint, render_template

from app.extensions import db
from app.models import TrainingRun

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    recent = db.session.execute(
        db.select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(10)
    ).scalars().all()
    return render_template('index.html', runs=[r.to_summary_dict() for r in recent])


@main_bp.route('/help')
def help_page():
    return render_template('help.html')
