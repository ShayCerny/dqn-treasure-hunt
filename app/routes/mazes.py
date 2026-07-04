"""Route for the maze builder UI."""
from flask import Blueprint, render_template

from app.config import DEFAULT_MAZE

mazes_bp = Blueprint('mazes', __name__)


@mazes_bp.route('/mazes')
def maze_builder():
    return render_template('maze_builder.html', default_maze=DEFAULT_MAZE)
