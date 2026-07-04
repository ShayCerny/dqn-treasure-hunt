"""Entry point for the Flask development server.

load_dotenv() must run before the app import so that environment variables
(SECRET_KEY, DATABASE_URL, FLASK_DEBUG, etc.) are available when the
application config is evaluated.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() != 'false'
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    # threading async_mode has no production-grade server (no eventlet/gevent
    # dependency); allow_unsafe_werkzeug accepts Werkzeug for this low-traffic
    # single-container deployment instead of adding a separate WSGI server.
    socketio.run(app, debug=debug, host=host, port=port, allow_unsafe_werkzeug=not debug)
