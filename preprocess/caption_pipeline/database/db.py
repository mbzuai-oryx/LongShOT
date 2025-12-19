"""
Database module for the Arabic video dataset.
This module handles database connection and setup.
"""

import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Create SQLAlchemy instance
db = SQLAlchemy()


def init_app(app):
    """Initialize the database with the Flask app."""
    # Use Flask's instance folder for the database
    app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///caption_pipeline.db"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Ensure instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)
    
    db.init_app(app)
    Migrate(app, db)
    
    # Import models to ensure they are registered with SQLAlchemy
    from caption_pipeline.database.models import User, Video, Caption, Verification, CaptionSegment
    
    # Create tables
    with app.app_context():
        db.create_all()
        print(f"Database created at: {os.path.join(app.instance_path, 'caption_pipeline.db')}")
    
    return db
