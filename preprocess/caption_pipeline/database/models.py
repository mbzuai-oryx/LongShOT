"""
Database models for the video caption dataset.
This module defines the database models for the web interface.
"""

from datetime import datetime
from sqlalchemy import and_
from caption_pipeline.database.db import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin


class User(db.Model, UserMixin):
    """User model for authentication and tracking verifications."""
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    assigned_videos = db.relationship('VideoAssignment', 
                                    primaryjoin='User.id==VideoAssignment.user_id',
                                    back_populates='user',
                                    cascade='all, delete-orphan')
    assignments_given = db.relationship('VideoAssignment',
                                      primaryjoin='User.id==VideoAssignment.assigned_by',
                                      back_populates='assigner')
    verifications = db.relationship('Verification',
                                  back_populates='user',
                                  cascade='all, delete-orphan')
    verified_segments = db.relationship('CaptionSegment',
                                      primaryjoin='User.id==CaptionSegment.verified_by',
                                      back_populates='verifier')
    
    def set_password(self, password):
        """Set the password hash for the user."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if the provided password matches."""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class Video(db.Model):
    """Video model for storing video metadata."""
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(20), unique=True, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    channel = db.Column(db.String(100))
    duration = db.Column(db.Integer)  # Duration in seconds
    view_count = db.Column(db.Integer)
    publish_date = db.Column(db.Date)
    description = db.Column(db.Text)
    tags = db.Column(db.Text)
    download_date = db.Column(db.Date)
    file_path = db.Column(db.String(256))
    thumbnail_path = db.Column(db.String(256))  # Path to the video thumbnail
    audio_path = db.Column(db.String(256))
    status = db.Column(db.String(50))  # downloaded, audio_extracted, captioned, verified
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    captions = db.relationship('Caption', back_populates='video', lazy=True,
                             cascade='all, delete-orphan')
    assignments = db.relationship('VideoAssignment', back_populates='video',
                                cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Video {self.title}>'


class Caption(db.Model):
    """Caption model for storing auto-generated captions."""
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
    caption_path = db.Column(db.String(256))  # Path to the caption file
    total_segments = db.Column(db.Integer)  # Total number of segments in caption
    verified_segments = db.Column(db.Integer, default=0)  # Number of verified segments
    verification_progress = db.Column(db.Float, default=0)  # Progress percentage
    verification_count = db.Column(db.Integer, default=0)  # Number of times verified
    is_verified = db.Column(db.Boolean, default=False)  # Whether fully verified
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    video = db.relationship('Video', back_populates='captions')
    segments = db.relationship('CaptionSegment', back_populates='caption',
                             order_by='CaptionSegment.segment_index',
                             cascade='all, delete-orphan')
    verifications = db.relationship('Verification', back_populates='caption',
                                  cascade='all, delete-orphan')
    
    def update_verification_status(self):
        """Update verification status based on segment verification status."""
        # Count verified segments (approved or edited)
        verified_segments = CaptionSegment.query.filter(
            CaptionSegment.caption_id == self.id,
            CaptionSegment.verification_status.in_(['approved', 'edited'])
        ).count()
        
        # Update caption verification stats
        self.verified_segments = verified_segments
        self.total_segments = self.total_segments or CaptionSegment.query.filter_by(caption_id=self.id).count()
        
        # Calculate verification progress percentage
        if self.total_segments > 0:
            self.verification_progress = (verified_segments / self.total_segments) * 100
        else:
            self.verification_progress = 0
            
        # Update fully verified status
        self.is_verified = (self.verification_progress == 100)
        
        # Commit changes
        db.session.commit()
    
    def __repr__(self):
        return f'<Caption for Video {self.video_id}>'


class CaptionSegment(db.Model):
    """Model for individual caption segments that can be verified independently."""
    
    id = db.Column(db.Integer, primary_key=True)
    caption_id = db.Column(db.Integer, db.ForeignKey('caption.id'), nullable=False)
    segment_index = db.Column(db.Integer, nullable=False)  # Order of segments in the caption
    start_time = db.Column(db.Float, nullable=False)  # Start time in seconds
    end_time = db.Column(db.Float, nullable=False)  # End time in seconds
    original_text = db.Column(db.Text, nullable=False)  # Original transcribed text
    verified_text = db.Column(db.Text)  # Verified/corrected text
    verification_status = db.Column(db.String(20), default='pending')  # pending, approved, edited
    verification_time = db.Column(db.Float, default=0.0)  # Time taken to verify this segment in seconds
    verified_by = db.Column(db.Integer, db.ForeignKey('user.id'))  # User who verified this segment
    verified_at = db.Column(db.DateTime)  # When this segment was verified
    verification_notes = db.Column(db.Text)  # Any notes from verification
    
    # Relationships
    caption = db.relationship('Caption', back_populates='segments')
    verifier = db.relationship('User', back_populates='verified_segments')
    
    def __init__(self, caption_id=None, segment_index=None, start_time=None, end_time=None, 
                 original_text=None, verified_text=None, verification_status='pending',
                 verification_time=0.0, verified_by=None, verification_notes=None):
        self.caption_id = caption_id
        self.segment_index = segment_index
        self.start_time = start_time
        self.end_time = end_time
        self.original_text = original_text
        self.verified_text = verified_text
        self.verification_status = verification_status
        self.verification_time = verification_time
        self.verified_by = verified_by
        self.verification_notes = verification_notes
    
    def __repr__(self):
        return f'<CaptionSegment {self.id} for Caption {self.caption_id}>'


class VideoAssignment(db.Model):
    """Model for tracking which videos are assigned to which users."""
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assigned_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assigned_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='assigned')  # assigned, completed
    
    # Relationships
    video = db.relationship('Video', back_populates='assignments')
    user = db.relationship('User', 
                         primaryjoin='VideoAssignment.user_id==User.id',
                         back_populates='assigned_videos')
    assigner = db.relationship('User',
                             primaryjoin='VideoAssignment.assigned_by==User.id',
                             back_populates='assignments_given')
    
    def __init__(self, video_id=None, user_id=None, assigned_by=None, status='assigned'):
        self.video_id = video_id
        self.user_id = user_id
        self.assigned_by = assigned_by
        self.status = status
    
    def __repr__(self):
        return f'<VideoAssignment {self.id} - {self.video_id} to {self.user_id}>'


class Verification(db.Model):
    """Verification model for tracking caption verifications."""
    
    id = db.Column(db.Integer, primary_key=True)
    caption_id = db.Column(db.Integer, db.ForeignKey('caption.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    verified_caption_path = db.Column(db.String(256))
    verification_notes = db.Column(db.Text)
    verification_time = db.Column(db.Float)  # Time taken to verify in seconds
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    caption = db.relationship('Caption', back_populates='verifications')
    user = db.relationship('User', back_populates='verifications')
    
    def __init__(self, caption_id=None, user_id=None, verified_caption_path=None,
                 verification_notes=None, verification_time=None):
        self.caption_id = caption_id
        self.user_id = user_id
        self.verified_caption_path = verified_caption_path
        self.verification_notes = verification_notes
        self.verification_time = verification_time
    
    def __repr__(self):
        return f'<Verification {self.id} by User {self.user_id}>'
