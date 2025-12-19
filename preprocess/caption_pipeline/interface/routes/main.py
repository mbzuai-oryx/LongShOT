"""
Main routes for the Flask web interface.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, send_file, abort, send_from_directory, jsonify
from flask_login import login_required, current_user
from caption_pipeline.database.models import Video, Caption, CaptionSegment, User, VideoAssignment
from caption_pipeline.database.db import db
import os
import json
from datetime import datetime
import pandas as pd
from functools import wraps

from caption_pipeline.database.db import db
from caption_pipeline.database.models import Video, Caption, Verification, CaptionSegment, User, VideoAssignment

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from config import VIDEO_DIR, CAPTIONS_DIR

main = Blueprint('main', __name__)

def admin_required(f):
    """Decorator to require admin access for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user or not current_user.is_admin:
            flash('Admin access required.', 'error')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

def can_access_video(user, video):
    """Check if a user can access a video."""
    # Admins can access any video
    if user.is_admin:
        return True
        
    # Check if video is assigned to user
    assignment = VideoAssignment.query.filter_by(
        video_id=video.id,
        user_id=user.id
    ).first()
    return assignment is not None
    
@main.route('/')
@login_required
def index():
    """Route for the home page."""
    # Get statistics
    total_videos = Video.query.count()
    if current_user.is_admin:
        captioned_videos = Video.query.filter_by(status='captioned').count()
        verified_videos = Video.query.filter_by(status='verified').count()
        
        # Get list of users for assignment
        users = User.query.filter(User.is_admin == False).all()

        # Get unassigned videos
        assigned_video_ids = db.session.query(VideoAssignment.video_id).distinct().all()
        assigned_video_ids = [v[0] for v in assigned_video_ids]
        query = Video.query.filter(Video.status == 'captioned')
        if assigned_video_ids:
            query = query.filter(~Video.id.in_(assigned_video_ids))
        unassigned_videos = query.limit(5).all()
        
        # Get recent assignments with progress
        recent_assignments = (
            VideoAssignment.query
            .join(Video)
            .join(User, VideoAssignment.user_id == User.id)
            .join(Caption, Caption.video_id == Video.id)
            .filter(VideoAssignment.status == 'assigned')
            .all()
        )
        
        # Get recent activities for admin (all users)
        recent_segment_verifications = (
            CaptionSegment.query
            .join(Caption)
            .join(Video)
            .join(User, CaptionSegment.verified_by == User.id)
            .filter(CaptionSegment.verified_by.isnot(None))
            .order_by(CaptionSegment.verified_at.desc())
            .limit(20)
            .all()
        )
        
        recent_assignments_activities = (
            VideoAssignment.query
            .join(Video)
            .join(User, VideoAssignment.user_id == User.id)
            .order_by(VideoAssignment.assigned_at.desc())
            .limit(20)
            .all()
        )
        
        # Combine and sort activities
        recent_activities = []
        for verification in recent_segment_verifications:
            caption = verification.caption
            progress = caption.verification_progress if caption else 0
            recent_activities.append({
                'type': 'verification',
                'timestamp': verification.verified_at,
                'user': User.query.get(verification.verified_by),
                'video': verification.caption.video,
                'segment': verification,
                'progress': progress
            })
            
        for assignment in recent_assignments_activities:
            recent_activities.append({
                'type': 'assignment',
                'timestamp': assignment.assigned_at,
                'user': assignment.user,
                'video': assignment.video,
                'assigner': User.query.get(assignment.assigned_by) if assignment.assigned_by else None,
                'progress': 0  # New assignments start at 0%
            })
            
        # Sort combined activities by timestamp
        recent_activities = sorted(
            recent_activities, 
            key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min, 
            reverse=True
        )[:20]
    else:
        # Only show stats for assigned videos for regular users
        assigned_video_ids = db.session.query(VideoAssignment.video_id).filter_by(user_id=current_user.id).all()
        assigned_video_ids = [v[0] for v in assigned_video_ids]
        captioned_videos = Video.query.filter(Video.id.in_(assigned_video_ids), Video.status == 'captioned').count()
        verified_videos = Video.query.filter(Video.id.in_(assigned_video_ids), Video.status == 'verified').count()
        unassigned_videos = None
        recent_assignments = None
        
        # Get recent activities only for current user
        recent_segment_verifications = (
            CaptionSegment.query
            .join(Caption)
            .join(Video)
            .filter(CaptionSegment.verified_by == current_user.id)
            .order_by(CaptionSegment.verified_at.desc())
            .limit(20)
            .all()
        )
        
        # Regular users only see their own activities
        recent_activities = [{
            'type': 'verification',
            'timestamp': verification.verified_at,
            'user': current_user,
            'video': verification.caption.video,
            'segment': verification,
            'progress': verification.caption.verification_progress if verification.caption else 0
        } for verification in recent_segment_verifications]
        
        # Calculate user completion percentage
        total_segments_in_assigned = 0
        verified_segments_count = 0
        
        # Get total segments in assigned videos
        for video_id in assigned_video_ids:
            caption = Caption.query.filter_by(video_id=video_id).first()
            if caption:
                total_segments_in_assigned += caption.total_segments or 0
        
        # Get verified segments by this user
        verified_segments_count = CaptionSegment.query.filter_by(verified_by=current_user.id).count()
        
        # Calculate completion percentage
        user_completion_percentage = 0
        if total_segments_in_assigned > 0:
            user_completion_percentage = (verified_segments_count / total_segments_in_assigned) * 100
        
        # Add these values to template context
        user_total_segments = total_segments_in_assigned
        user_verified_segments = verified_segments_count
    
    # Get user's assigned videos
    assigned_videos = (
        Video.query
        .join(VideoAssignment)
        .join(Caption)
        .filter(VideoAssignment.user_id == current_user.id)
        .order_by(VideoAssignment.assigned_at.desc())
        .all()
    )

    # Get videos where user has recently verified segments
    from sqlalchemy.orm import joinedload
    from sqlalchemy import desc
    
    # Get videos where user has verified segments recently
    recent_segment_videos = db.session.query(Video).join(Caption).join(CaptionSegment).filter(
        CaptionSegment.verified_by == current_user.id
    ).options(joinedload(Video.captions)).distinct().order_by(desc(CaptionSegment.verified_at)).limit(10).all()
    
    # Combine and deduplicate recent videos
    verified_by_user = []
    seen_video_ids = set()
    
    # Add videos from segment verification first (more recent system)
    for video in recent_segment_videos:
        if video.id not in seen_video_ids and can_access_video(current_user, video):
            verified_by_user.append(video)
            seen_video_ids.add(video.id)
    
    # Limit to 10 most recent
    verified_by_user = verified_by_user[:10]
    
    return render_template(
        'index.html',
        total_videos=total_videos,
        captioned_videos=captioned_videos,
        verified_videos=verified_videos,
        assigned_videos=assigned_videos,
        verified_by_user=verified_by_user,
        unassigned_videos=unassigned_videos,
        recent_assignments=recent_assignments,
        recent_activities=recent_activities,
        users=users if current_user.is_admin else None,
        # Add these variables for non-admin users
        user_completion_percentage=user_completion_percentage if not current_user.is_admin else 0,
        user_total_segments=user_total_segments if not current_user.is_admin else 0,
        user_verified_segments=user_verified_segments if not current_user.is_admin else 0
    )


@main.route('/videos')
@login_required
def videos():
    """Route for the videos list page."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    status = request.args.get('status', 'all')
    search_query = request.args.get('q', '').strip()
    
    # Start with base query
    query = Video.query
    
    # Get assignments
    assignments_by_video = {}
    if current_user.is_admin:
        # Admin can see all videos and assignments
        assignments = VideoAssignment.query.all()
        for assignment in assignments:
            if assignment.video_id not in assignments_by_video:
                assignments_by_video[assignment.video_id] = []
            assignments_by_video[assignment.video_id].append({
                'user': assignment.user,
                'status': assignment.status,
                'assigned_at': assignment.assigned_at
            })
    else:
        # Non-admin users can only see assigned videos
        assigned_video_ids = db.session.query(VideoAssignment.video_id).filter_by(user_id=current_user.id).all()
        assigned_video_ids = [v[0] for v in assigned_video_ids]
        query = query.filter(Video.id.in_(assigned_video_ids))
    
    # Apply status filter if specified
    if status != 'all':
        query = query.filter_by(status=status)
    
    # Apply search filter if provided
    if search_query:
        search_term = f"%{search_query}%"
        query = query.filter(
            db.or_(
                Video.title.ilike(search_term),
                Video.channel.ilike(search_term),
                Video.description.ilike(search_term),
                Video.video_id.ilike(search_term)
            )
        )
    
    # Order by most recently added
    query = query.order_by(Video.id.desc())
    
    # Paginate results
    pagination = query.paginate(page=page, per_page=per_page)
    videos = pagination.items
    
    # Get admin user list for assignment
    users = None
    if current_user.is_admin:
        users = User.query.filter(User.is_admin == False).all()
    
    return render_template(
        'videos.html',
        videos=videos,
        pagination=pagination,
        status=status,
        search_query=search_query,
        assignments_by_video=assignments_by_video,
        users=users
    )


@main.route('/video/<string:video_id>')
@login_required
def video_detail(video_id):
    """Route for the video detail page."""
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    
    # Check access
    if not can_access_video(current_user, video):
        flash('You do not have access to this video.', 'error')
        return redirect(url_for('main.videos'))
    
    # Get all users for admin assignment
    users = None
    assignments = None
    if current_user.is_admin:
        users = User.query.filter(User.is_admin == False).all()
        assignments = VideoAssignment.query.filter_by(video_id=video.id).all()
    
    # Check if user wants to view the original captions (for comparison)
    show_original = request.args.get('show_original', 'false') == 'true'
    
    # Get caption data
    caption = Caption.query.filter_by(video_id=video.id).first()
    caption_data = None
    caption_source = "original"
    
    # Check for verified captions first
    verified_caption_path = None
    original_caption_available = False
    
    if caption:
        # Get the most recent verification
        latest_verification = Verification.query.filter_by(caption_id=caption.id).order_by(
            Verification.created_at.desc()
        ).first()
        
        if latest_verification and latest_verification.verified_caption_path and os.path.exists(latest_verification.verified_caption_path):
            verified_caption_path = latest_verification.verified_caption_path
            caption_source = "verified"
            
            try:
                with open(verified_caption_path, 'r', encoding='utf-8') as f:
                    caption_data = json.load(f)
                print(f"Using verified caption at: {verified_caption_path}")
                
            except Exception as e:
                print(f"Error loading verified caption: {e}")
                verified_caption_path = None
            
        # Check if original caption is available
        if caption.caption_path and os.path.exists(caption.caption_path):
            original_caption_available = True
    
    # Decide which caption to load - verified or original based on user preference
    if show_original and original_caption_available:
        try:
            with open(caption.caption_path, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
            caption_source = "original"
        except Exception as e:
            print(f"Error loading original caption: {e}")
    
    # Get previous verifications
    verifications = []
    if caption:
        verifications = Verification.query.filter_by(caption_id=caption.id).order_by(
            Verification.created_at.desc()
        ).all()
    
    # Pass a flag indicating if we're displaying verified captions
    is_verified = caption_source == "verified"
    
    return render_template(
        'video_detail.html',
        video=video,
        caption=caption,
        caption_data=caption_data,
        verifications=verifications,
        is_verified=is_verified,
        caption_source=caption_source,
        show_original=show_original,
        original_caption_available=original_caption_available,
        has_verified_caption=verified_caption_path is not None,
        users=users,
        assignments=assignments
    )


@main.route('/verify/<string:video_id>')
@login_required
def verify_caption(video_id):
    """Route for the caption verification page."""
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    
    # Check access
    if not can_access_video(current_user, video):
        flash('You do not have access to this video.', 'error')
        return redirect(url_for('main.videos'))
    
    # Get or create caption
    caption = Caption.query.filter_by(video_id=video.id).first()
    if not caption:
        caption = Caption()
        caption.video_id = video.id
        db.session.add(caption)
        db.session.commit()
    
    # Ensure segments exist
    if not ensure_caption_segments(caption, video_id):
        flash("No caption data available for this video.", "error")
        return redirect(url_for('main.video_detail', video_id=video_id))
    
    # Get the requested segment or first unverified segment
    segment_index = request.args.get('segment', type=int)
    if segment_index is None:
        segment = CaptionSegment.query.filter_by(
            caption_id=caption.id,
            verification_status='pending'
        ).order_by(CaptionSegment.segment_index).first()
        segment_index = segment.segment_index if segment else 0
    else:
        segment = CaptionSegment.query.filter_by(
            caption_id=caption.id,
            segment_index=segment_index
        ).first()
    
    # If still no segment found, get the first segment
    if not segment:
        segment = CaptionSegment.query.filter_by(
            caption_id=caption.id
        ).order_by(CaptionSegment.segment_index).first()
        
        if not segment:
            flash("No caption segments found for verification.", "error")
            return redirect(url_for('main.video_detail', video_id=video_id))
        
        segment_index = segment.segment_index
    
    # Get verification stats
    total_segments = caption.total_segments or 0
    verified_segments = CaptionSegment.query.filter(
        CaptionSegment.caption_id == caption.id,
        CaptionSegment.verification_status.in_(['approved', 'edited'])
    ).count()
    verification_progress = (verified_segments / total_segments * 100) if total_segments > 0 else 0
    
    # Get status for each segment for the timeline
    segments_status = []
    if total_segments > 0:
        all_segments = CaptionSegment.query.filter_by(caption_id=caption.id).order_by(CaptionSegment.segment_index).all()
        segment_status_dict = {seg.segment_index: seg.verification_status for seg in all_segments}
        segments_status = [segment_status_dict.get(i, 'pending') for i in range(total_segments)]
    
    # Update caption progress and video status
    caption.verified_segments = verified_segments
    caption.verification_progress = verification_progress
    caption.is_verified = verification_progress == 100
    if caption.is_verified:
        video.status = 'verified'
        # Update assignment status to completed
        assignment = VideoAssignment.query.filter_by(
            video_id=video.id,
            user_id=current_user.id,
            status='assigned'
        ).first()
        if assignment:
            assignment.status = 'completed'
    db.session.commit()
    
    return render_template(
        'verify_caption.html',
        video=video,
        caption=caption,
        segment=segment,
        segment_index=segment_index,
        total_segments=total_segments,
        verification_progress=verification_progress,
        segments_status=segments_status
    )


def ensure_caption_segments(caption, video_id):
    """Ensure caption segments exist in database, create from JSON if needed."""
    # Check if segments already exist
    existing_segments = CaptionSegment.query.filter_by(caption_id=caption.id).count()
    if existing_segments > 0:
        return True
    
    # Try to load from caption file
    caption_file = None
    if caption.caption_path and os.path.exists(caption.caption_path):
        caption_file = caption.caption_path
    else:
        # Try common locations
        possible_paths = [
            os.path.join(CAPTIONS_DIR, f"{video_id}.json"),
            os.path.join(CAPTIONS_DIR, f"{video_id}_captions.json"),
            os.path.join(CAPTIONS_DIR, f"{video_id}_transcript.json")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                caption_file = path
                # Update caption with correct path
                caption.caption_path = path
                break
    
    if not caption_file:
        print(f"No caption file found for video {video_id}")
        return False
    
    try:
        with open(caption_file, 'r', encoding='utf-8') as f:
            caption_data = json.load(f)
        
        # Handle different JSON structures
        segments = []
        if 'transcript' in caption_data and 'segments' in caption_data['transcript']:
            segments = caption_data['transcript']['segments']
        elif 'segments' in caption_data:
            segments = caption_data['segments']
        elif isinstance(caption_data, list):
            segments = caption_data
        
        if not segments:
            print(f"No segments found in caption file {caption_file}")
            return False
        
        # Create CaptionSegment objects
        for i, seg_data in enumerate(segments):
            segment = CaptionSegment()
            segment.caption_id = caption.id
            segment.segment_index = i
            segment.start_time = float(seg_data.get('start', 0))
            segment.end_time = float(seg_data.get('end', segment.start_time + 5))
            segment.original_text = seg_data.get('text', '').strip()
            segment.verification_status = 'pending'
            db.session.add(segment)
        
        # Update caption metadata
        caption.total_segments = len(segments)
        db.session.commit()
        
        print(f"Created {len(segments)} segments for video {video_id}")
        return True
        
    except Exception as e:
        print(f"Error creating segments from caption file {caption_file}: {e}")
        db.session.rollback()
        return False


@main.route('/api/segments/<string:video_id>', methods=['GET'])
@login_required
def get_segments(video_id):
    """API endpoint to get caption segments."""
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    caption = Caption.query.filter_by(video_id=video.id).first_or_404()
    
    segments = CaptionSegment.query.filter_by(caption_id=caption.id)\
        .order_by(CaptionSegment.segment_index).all()
        
    return jsonify([{
        'id': seg.id,
        'index': seg.segment_index,
        'start_time': seg.start_time,
        'end_time': seg.end_time,
        'text': seg.verified_text or seg.original_text,
        'status': seg.verification_status
    } for seg in segments])


@main.route('/api/segments/<string:video_id>/<int:segment_index>', methods=['PUT'])
@login_required
def update_segment(video_id, segment_index):
    """API endpoint to update a caption segment."""
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    caption = Caption.query.filter_by(video_id=video.id).first_or_404()
    segment = CaptionSegment.query.filter_by(
        caption_id=caption.id,
        segment_index=segment_index
    ).first_or_404()
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Update segment
    if 'text' in data:
        segment.verified_text = data['text']
        segment.verification_status = 'edited'
    elif 'status' in data and data['status'] == 'approved':
        segment.verification_status = 'approved'
    
    segment.verified_by = current_user.id
    segment.verified_at = datetime.utcnow()
    segment.verification_time = data.get('verification_time', 0.0)
    
    # Update caption progress
    if hasattr(caption, 'update_verification_status'):
        caption.update_verification_status()
    else:
        # Fallback manual update if method doesn't exist
        verified_segments = CaptionSegment.query.filter(
            CaptionSegment.caption_id == caption.id,
            CaptionSegment.verification_status.in_(['approved', 'edited'])
        ).count()
        
        total_segments = caption.total_segments or CaptionSegment.query.filter_by(caption_id=caption.id).count()
        caption.verified_segments = verified_segments
        caption.verification_progress = (verified_segments / total_segments * 100) if total_segments > 0 else 0
        caption.is_verified = caption.verification_progress == 100
        db.session.commit()
    
    # Check if video is now fully verified and create a Verification record
    if caption.is_verified and video.status != 'verified':
        video.status = 'verified'
        
        # Create a Verification record for tracking (maintains compatibility)
        verification = Verification()
        verification.caption_id = caption.id
        verification.user_id = current_user.id
        verification.verification_notes = f"Completed segment-by-segment verification"
        # Sum only non-null verification times
        total_time = db.session.query(db.func.sum(CaptionSegment.verification_time))\
            .filter(CaptionSegment.caption_id == caption.id,
                   CaptionSegment.verified_by == current_user.id,
                   CaptionSegment.verification_time != None).scalar()
        verification.verification_time = float(total_time if total_time else 0)
        verification.created_at = datetime.utcnow()
        db.session.add(verification)
    
    db.session.commit()
    
    # Return next unverified segment index
    next_segment = CaptionSegment.query.filter_by(
        caption_id=caption.id,
        verification_status='pending'
    ).order_by(CaptionSegment.segment_index).first()
    
    return jsonify({
        'success': True,
        'next_segment': next_segment.segment_index if next_segment else None,
        'verification_progress': caption.verification_progress
    })


@main.route('/video-file/<string:video_id>')
@login_required
def video_file(video_id):
    """Route to serve video files."""
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    
    if not video.file_path or not os.path.exists(video.file_path):
        flash('Video file not found.', 'danger')
        return redirect(url_for('main.video_detail', video_id=video_id))
    
    # Get directory and filename from the file path
    directory = os.path.dirname(video.file_path)
    filename = os.path.basename(video.file_path)
    
    return send_from_directory(directory, filename)


@main.route('/thumbnail/<video_id>')
@login_required
def thumbnail(video_id):
    """Route to serve a video thumbnail."""
    from caption_pipeline.utils.video_utils import generate_thumbnail
    
    # Get requested size
    size = request.args.get('size', 'medium')
    size_mapping = {
        'small': 320,
        'medium': 640,
        'large': 1280
    }
    width = size_mapping.get(size, 640)
    
    # Get video from database
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    
    # Check if thumbnail already exists
    thumbnail_name = f"{video_id}_{size}.jpg"
    thumbnail_dir = os.path.join(os.path.dirname(VIDEO_DIR), 'thumbnails')
    thumbnail_path = os.path.join(thumbnail_dir, thumbnail_name)
    
    # If thumbnail doesn't exist, generate it
    if not os.path.exists(thumbnail_path):
        # Create thumbnail directory if it doesn't exist
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        # Find the video file
        video_path = None
        for ext in ['.mp4', '.avi', '.mkv', '.mov']:
            potential_path = os.path.join(VIDEO_DIR, f"{video_id}{ext}")
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        
        if video_path:
            # Generate thumbnail
            success = generate_thumbnail(video_path, thumbnail_path, width=width)
            if not success:
                # If generation fails, return default thumbnail
                return send_from_directory(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../static/img'),
                    'default-thumbnail.jpg'
                )
        else:
            # If video not found, return a default thumbnail
            return send_from_directory(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '../static/img'),
                'default-thumbnail.jpg'
            )
    
    # Update database with thumbnail path if not already set
    if not video.thumbnail_path:
        standard_thumbnail = os.path.join(thumbnail_dir, f"{video_id}_medium.jpg")
        if os.path.exists(standard_thumbnail):
            video.thumbnail_path = standard_thumbnail
            db.session.commit()
    
    # Serve the thumbnail with appropriate cache headers
    response = send_from_directory(thumbnail_dir, thumbnail_name)
    response.headers['Cache-Control'] = 'public, max-age=86400'  # Cache for 24 hours
    return response


@main.route('/api/user-stats')
@login_required
def user_stats():
    """API endpoint to get user statistics."""
    from sqlalchemy import func
    
    # Get user's verification count
    user_verifications = CaptionSegment.query.filter_by(verified_by=current_user.id).count()
    
    # Get user's total verification time
    total_time_result = db.session.query(func.sum(CaptionSegment.verification_time)).filter_by(
        verified_by=current_user.id
    ).scalar()
    total_time = float(total_time_result) if total_time_result else 0.0
    
    return jsonify({
        'total_verifications': user_verifications,
        'total_verification_time': total_time
    })


@main.route('/admin/assign-video/<string:video_id>', methods=['POST'])
@login_required
@admin_required
def assign_video(video_id):
    """Admin route to assign a video to a user."""
    user_id = request.form.get('user_id')
    if not user_id:
        flash('No user selected.', 'error')
        return redirect(url_for('main.video_detail', video_id=video_id))
    
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    user = User.query.get_or_404(user_id)
    
    # Check if video is already assigned to this user
    existing = VideoAssignment.query.filter_by(
        video_id=video.id,
        user_id=user.id,
    ).first()
    
    if existing:
        flash(f'Video already assigned to {user.username}.', 'warning')
    else:
        assignment = VideoAssignment(
            video_id=video.id,
            user_id=user.id,
            assigned_by=current_user.id
        )
        db.session.add(assignment)
        db.session.commit()
        flash(f'Video assigned to {user.username}.', 'success')
    
    return redirect(url_for('main.video_detail', video_id=video_id))

@main.route('/admin/unassign-video/<string:video_id>', methods=['POST'])
@login_required
@admin_required 
def unassign_video(video_id):
    """Admin route to unassign a video from a user."""
    user_id = request.form.get('user_id')
    if not user_id:
        flash('No user selected.', 'error')
        return redirect(url_for('main.video_detail', video_id=video_id))
    
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    user = User.query.get_or_404(user_id)
    assignment = VideoAssignment.query.filter_by(
        video_id=video.id,
        user_id=user.id
    ).first()
    
    if assignment:
        db.session.delete(assignment)
        db.session.commit()
        flash(f'Video unassigned from {user.username}.', 'success')
    else:
        flash(f'Video was not assigned to {user.username}.', 'warning')
    
    return redirect(url_for('main.video_detail', video_id=video_id))

@main.route('/admin/bulk-unassign-video/<string:video_id>', methods=['POST'])
@login_required
@admin_required
def bulk_unassign_video(video_id):
    """Admin route to unassign multiple users from a video."""
    selected_users = request.form.get('selected_users', '')
    
    if not selected_users:
        flash('No users selected.', 'error')
        return redirect(url_for('main.video_detail', video_id=video_id))
    
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    
    try:
        user_ids = [int(uid) for uid in selected_users.split(',')]
        
        # Find assignments to remove
        assignments = VideoAssignment.query.filter(
            VideoAssignment.video_id == video.id,
            VideoAssignment.user_id.in_(user_ids)
        ).all()
        
        if not assignments:
            flash('No matching assignments found.', 'warning')
            return redirect(url_for('main.video_detail', video_id=video_id))
        
        # Delete assignments
        count = 0
        for assignment in assignments:
            db.session.delete(assignment)
            count += 1
        
        db.session.commit()
        
        flash(f'Successfully unassigned {count} users from this video.', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error unassigning users: {str(e)}', 'error')
    
    return redirect(url_for('main.video_detail', video_id=video_id))

@main.route('/users')
@login_required
@admin_required
def users():
    """Route for the user management page."""
    # Get all users
    users = User.query.filter(User.id != current_user.id).order_by(User.username).all()
    
    # Prepare user data with assignment statistics
    user_data = []
    for user in users:
        # Get assignments
        assignments = VideoAssignment.query.filter_by(user_id=user.id).all()
        
        # Get verified segments
        verified_segments = CaptionSegment.query.filter_by(verified_by=user.id).count()
        
        # Calculate verification time
        verification_time = db.session.query(db.func.sum(CaptionSegment.verification_time))\
            .filter_by(verified_by=user.id).scalar()
        verification_time = float(verification_time) if verification_time else 0.0
        
        # Calculate completion percentage
        total_assigned = len(assignments)
        completed = VideoAssignment.query.filter_by(user_id=user.id, status='completed').count()
        completion_percentage = (completed / total_assigned * 100) if total_assigned > 0 else 0
        
        user_data.append({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'assigned_videos': total_assigned,
            'completed_videos': completed,
            'verified_segments': verified_segments,
            'completion_percentage': completion_percentage,
            'verification_time': verification_time,
            'created_at': user.created_at,
        })
    
    return render_template('users.html', users=user_data)
