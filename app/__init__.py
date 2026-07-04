"""Flask application factory.

Initialisation order matters: load_dotenv() in run.py must execute before
this module so environment variables are available when config is evaluated.
Extension objects (socketio) are created at module level so they can be
imported by other modules without triggering create_app(); db and migrate
live in extensions.py for the same reason.
"""
import os
from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO()


def create_app():
    app = Flask(__name__)
    config_name = os.environ.get('APP_ENV', 'app.config.DevelopmentConfig')
    app.config.from_object(config_name)

    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config['MODELS_DIR'], exist_ok=True)

    from app.extensions import db, migrate
    db.init_app(app)
    migrate.init_app(app, db)
    # Defaults to '*' for local development convenience. Set CORS_ALLOWED_ORIGINS
    # (comma-separated) in the environment to restrict this for public deployments.
    cors_origins = os.environ.get('CORS_ALLOWED_ORIGINS', '*')
    if cors_origins != '*':
        cors_origins = [o.strip() for o in cors_origins.split(',')]
    socketio.init_app(app, cors_allowed_origins=cors_origins, async_mode='threading')

    # Import models here (not at module level) to avoid circular imports while
    # keeping the side-effect registration explicit and outside the app context.
    from app import models as _models  # noqa: F401

    with app.app_context():
        # create_all() provides first-run convenience in development.
        # Use 'flask db upgrade' for incremental migrations on existing databases.
        db.create_all()

    from app.routes.main import main_bp
    from app.routes.mazes import mazes_bp
    from app.routes.training import training_bp
    from app.api.mazes import api_mazes_bp
    from app.api.runs import api_runs_bp
    from app.api.solve import api_solve_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(mazes_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(api_mazes_bp)
    app.register_blueprint(api_runs_bp)
    app.register_blueprint(api_solve_bp)

    return app
