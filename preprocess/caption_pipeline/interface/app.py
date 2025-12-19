"""
Flask application module for the video caption dataset.
This module sets up the Flask web interface for caption verification.
"""

import os
from flask import Flask
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

# Import project configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, SECRET_KEY, SESSION_TYPE
from caption_pipeline.database.db import init_app
from caption_pipeline.database.models import User


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure the app
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['SESSION_TYPE'] = SESSION_TYPE
    app.config['WTF_CSRF_CHECK_DEFAULT'] = False  # Disable CSRF by default
    app.config['WTF_CSRF_ENABLED'] = True  # But enable it when explicitly requested

    # Initialize CSRF protection
    csrf = CSRFProtect(app)
    
    # Add datetime.utcnow to Jinja environment
    from datetime import datetime
    app.jinja_env.globals.update(now=datetime.utcnow)
    
    # Initialize the database
    db = init_app(app)
    
    # Set up Flask-Login
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Register blueprints
    from caption_pipeline.interface.routes.auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)
    
    from caption_pipeline.interface.routes.main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    from caption_pipeline.interface.routes.api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')
    
    return app


def run_app():
    """Run the Flask application."""
    app = create_app()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)


if __name__ == '__main__':
    run_app()
