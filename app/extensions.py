# Extension objects live here rather than in __init__.py to break the
# circular-import cycle: models.py needs db, and __init__.py needs models
# loaded before create_app() calls db.create_all().
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()
