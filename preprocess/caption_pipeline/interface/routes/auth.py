"""
Authentication routes for the Flask web interface.
"""

from flask import Blueprint, render_template, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, current_user
from datetime import datetime

from caption_pipeline.database.db import db
from caption_pipeline.database.models import User

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from config import ADMIN_USERNAME, ADMIN_PASSWORD, ALLOW_REGISTRATION

auth = Blueprint('auth', __name__)


@auth.route('/login')
def login():
    """Route for the login page."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    return render_template('login.html')


@auth.route('/login', methods=['POST'])
def login_post():
    """Handle login form submission."""
    username = request.form.get('username')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False
    
    user = User.query.filter_by(username=username).first()
    
    # Check if user exists and password is correct
    if not user or not user.check_password(password):
        flash('Please check your login details and try again.', 'danger')
        return redirect(url_for('auth.login'))
    
    # Update last login time
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    # Log in the user
    login_user(user, remember=remember)
    return redirect(url_for('main.index'))


@auth.route('/signup')
def signup():
    """Route for the signup page."""
    if not ALLOW_REGISTRATION:
        flash('Registration is currently disabled.', 'warning')
        return redirect(url_for('auth.login'))
    
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    return render_template('signup.html')


@auth.route('/signup', methods=['POST'])
def signup_post():
    """Handle signup form submission."""
    if not ALLOW_REGISTRATION:
        flash('Registration is currently disabled.', 'warning')
        return redirect(url_for('auth.login'))
    
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    
    # Check if user already exists
    user = User.query.filter_by(username=username).first()
    if user:
        flash('Username already exists.', 'danger')
        return redirect(url_for('auth.signup'))
    
    # Check if email already exists
    user = User.query.filter_by(email=email).first()
    if user:
        flash('Email address already in use.', 'danger')
        return redirect(url_for('auth.signup'))
    
    # Create new user
    new_user = User(
        username=username,
        email=email,
        is_admin=False
    )
    new_user.set_password(password)
    
    # Add user to database
    db.session.add(new_user)
    db.session.commit()
    
    flash('Account created successfully! You can now log in.', 'success')
    return redirect(url_for('auth.login'))


@auth.route('/logout')
@login_required
def logout():
    """Log out the current user."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


def create_admin_user(app):
    """Create admin user if it doesn't exist."""
    with app.app_context():
        admin = User.query.filter_by(username=ADMIN_USERNAME).first()
        if not admin:
            admin = User(
                username=ADMIN_USERNAME,
                email='admin@example.com',
                is_admin=True
            )
            admin.set_password(ADMIN_PASSWORD)
            db.session.add(admin)
            db.session.commit()
            print(f"Admin user '{ADMIN_USERNAME}' created.")
