"""

API routes for the web interface.
These routes provide JSON endpoints for the frontend JavaScript to interact with.
"""

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
import os
import json
from datetime import datetime
import pytz
from functools import wraps

from caption_pipeline.database.db import db
from caption_pipeline.database.models import Video, Caption, Verification, CaptionSegment, User, VideoAssignment

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from config import CAPTIONS_DIR

api = Blueprint('api', __name__)


def get_client_timezone():
    """Get timezone (default to 'Asia/Dubai')."""
    # try:
    return pytz.timezone('Asia/Dubai')
    # except pytz.exceptions.UnknownTimeZoneError:
    #     return pytz.UTC


def convert_to_client_timezone(utc_dt):
    """Convert UTC datetime to client timezone string."""
    if not utc_dt:
        return None
    
    # Make sure datetime is UTC aware
    if utc_dt.tzinfo is None:
        utc_dt = pytz.UTC.localize(utc_dt)
    
    # Convert to client timezone
    local_dt = utc_dt.astimezone(get_client_timezone())
    return local_dt.strftime('%Y %b %d, %I:%M %p')  # Updated format to include year


def admin_required(f):
    """Decorator to require admin access for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user or not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function


@api.route('/videos')
@login_required
def get_videos():
    """API endpoint to get videos."""
    status = request.args.get('status', 'all')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Start with base query
    query = Video.query
    
    # Non-admin users can only see assigned videos
    if not current_user.is_admin:
        assigned_video_ids = db.session.query(VideoAssignment.video_id).filter_by(user_id=current_user.id).all()
        assigned_video_ids = [v[0] for v in assigned_video_ids]
        query = query.filter(Video.id.in_(assigned_video_ids))
    
    # Query videos with optional status filter
    if status != 'all':
        query = query.filter_by(status=status)
    
    # Paginate results
    pagination = query.paginate(page=page, per_page=per_page)
    videos = pagination.items
    
    # Get assignments for admin
    assignments_by_video = {}
    if current_user.is_admin:
        assignments = VideoAssignment.query.filter(
            VideoAssignment.video_id.in_([v.id for v in videos])
        ).all()
        for assignment in assignments:
            if assignment.video_id not in assignments_by_video:
                assignments_by_video[assignment.video_id] = []
            assignments_by_video[assignment.video_id].append({
                'id': assignment.user.id,
                'username': assignment.user.username,
                'status': assignment.status
            })
    
    # Format response
    result = {
        'videos': [{
            'id': video.id,
            'video_id': video.video_id,
            'title': video.title,
            'channel': video.channel,
            'duration': video.duration,
            'status': video.status,
            'assignments': assignments_by_video.get(video.id, []) if current_user.is_admin else None
        } for video in videos],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': pagination.page
    }
    
    return jsonify(result)


@api.route('/captions/<string:video_id>')
@login_required
def get_captions(video_id):
    """API endpoint to get captions for a video."""
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    caption = Caption.query.filter_by(video_id=video.id).first()
    
    if not caption or not caption.caption_path or not os.path.exists(caption.caption_path):
        return jsonify({'error': 'Caption not found'}), 404
    
    with open(caption.caption_path, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)
    
    return jsonify(caption_data)


@api.route('/save-verification/<string:video_id>', methods=['POST'])
@login_required
def save_verification(video_id):
    """API endpoint to save caption verification."""
    print(f"Saving verification for video {video_id} by user {current_user.username}")
    
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    caption = Caption.query.filter_by(video_id=video.id).first_or_404()
    
    # Get verification data from request
    data = request.json
    if not data or 'segments' not in data:
        error_msg = f"Invalid verification data: {data}"
        print(error_msg)
        return jsonify({'error': error_msg}), 400
    
    print(f"Received {len(data['segments'])} segments for verification")
    
    # Create directory for verified captions if it doesn't exist
    verified_captions_dir = os.path.join(CAPTIONS_DIR, 'verified')
    os.makedirs(verified_captions_dir, exist_ok=True)
    
    # Save verified caption to file
    verified_caption_path = os.path.join(verified_captions_dir, f"{video_id}_verified_{current_user.id}.json")
    print(f"Saving verified caption to: {verified_caption_path}")
    
    # Check if there's an existing verified caption to merge metadata
    existing_data = {}
    if os.path.exists(verified_caption_path):
        try:
            with open(verified_caption_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                print(f"Loaded existing verified caption: {verified_caption_path}")
        except Exception as e:
            print(f"Error loading existing verified caption: {e}")
    
    # Prepare caption data with verification metadata
    verified_data = {
        'video_id': video_id,
        'title': video.title,
        'duration': video.duration,
        'transcript': {
            'segments': data['segments']
        },
        'verification': {
            'verified_by': current_user.username,
            'user_id': current_user.id,
            'verification_date': datetime.utcnow().isoformat(),
            'notes': data.get('notes', ''),
            'previous_verification_date': existing_data.get('verification', {}).get('verification_date') if existing_data else None
        },
        'verification_status': 'verified'
    }
    
    # If there's metadata from the original caption that we want to preserve, add it
    if 'metadata' in existing_data:
        verified_data['metadata'] = existing_data['metadata']
    
    # Write the verified caption to file
    try:
        with open(verified_caption_path, 'w', encoding='utf-8') as f:
            json.dump(verified_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved verified caption to {verified_caption_path}")
    except Exception as e:
        error_msg = f"Error saving verified caption: {e}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500
    
    # Record verification in database
    try:
        verification = Verification(
            caption_id=caption.id,
            user_id=current_user.id,
            verified_caption_path=verified_caption_path,
            verification_notes=data.get('notes', ''),
            verification_time=data.get('verification_time', 0)
        )
        
        db.session.add(verification)
        
        # Update caption verification status
        caption.is_verified = True
        caption.verification_count += 1
        
        # Update video status
        video.status = 'verified'
        
        db.session.commit()
        print(f"Successfully updated database records for verification")
    except Exception as e:
        error_msg = f"Error saving verification to database: {e}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500
    
    return jsonify({
        'success': True, 
        'message': 'Verification saved successfully',
        'verified_caption_path': verified_caption_path
    })


@api.route('/user-stats')
@login_required
def user_stats():
    """API endpoint to get statistics for the current user."""
    # Count verifications by status
    approved_segments = CaptionSegment.query.filter_by(
        verified_by=current_user.id, 
        verification_status='approved'
    ).count()
    
    edited_segments = CaptionSegment.query.filter_by(
        verified_by=current_user.id, 
        verification_status='edited'
    ).count()
    
    total_user_verifications = approved_segments + edited_segments
    
    # Get assigned video IDs
    assigned_video_ids = db.session.query(VideoAssignment.video_id).filter_by(
        user_id=current_user.id
    ).all()
    assigned_video_ids = [v_id[0] for v_id in assigned_video_ids]
    
    # Count total segments in all assigned videos
    total_assigned_segments = 0
    for video_id in assigned_video_ids:
        caption = Caption.query.filter_by(video_id=video_id).first()
        if caption:
            total_assigned_segments += caption.total_segments or 0
    
    # Calculate completion percentage
    completion_percentage = 0
    if total_assigned_segments > 0:
        completion_percentage = (total_user_verifications / total_assigned_segments) * 100
    
    # Get recent activity from segment verification system
    recent_segments = CaptionSegment.query.filter_by(verified_by=current_user.id).order_by(
        CaptionSegment.verified_at.desc()
    ).limit(10).all()
    
    # Build recent activity list from segments
    recent_activity = []
    seen_video_ids = set()
    
    for segment in recent_segments:
        caption = Caption.query.get(segment.caption_id)
        if caption:
            video = Video.query.get(caption.video_id)
            if video and video.id not in seen_video_ids:
                recent_activity.append({
                    'video_id': video.video_id,
                    'title': video.title,
                    'date': convert_to_client_timezone(segment.verified_at)
                })
                seen_video_ids.add(video.id)
                
                # Limit to 5 recent videos
                if len(recent_activity) >= 5:
                    break
    
    # Get recent activity from legacy verification system if needed
    if len(recent_activity) < 5:
        legacy_verifications = Verification.query.filter_by(user_id=current_user.id).order_by(
            Verification.created_at.desc()
        ).limit(5 - len(recent_activity)).all()
        
        for verification in legacy_verifications:
            caption = Caption.query.get(verification.caption_id)
            if caption:
                video = Video.query.get(caption.video_id)
                if video and video.id not in seen_video_ids:
                    recent_activity.append({
                        'video_id': video.video_id,
                        'title': video.title,
                        'date': convert_to_client_timezone(verification.created_at)
                    })
                    seen_video_ids.add(video.id)
    
    # Calculate total verification time from segments
    segment_times = db.session.query(db.func.sum(CaptionSegment.verification_time)).filter_by(
        verified_by=current_user.id
    ).scalar()
    total_time = float(segment_times) if segment_times else 0.0
    
    return jsonify({
        'total_verifications': total_user_verifications,
        'approved_segments': approved_segments,
        'edited_segments': edited_segments,
        'total_assigned_segments': total_assigned_segments,
        'completion_percentage': completion_percentage,
        'recent_activity': recent_activity,
        'total_verification_time': total_time
    })


# Segment management endpoints
@api.route('/segments/<string:video_id>/<int:segment_index>', methods=['PUT'])
@login_required
def update_segment(video_id, segment_index):
    """API endpoint to update a caption segment."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
        
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    video = Video.query.filter_by(video_id=video_id).first_or_404()
    caption = Caption.query.filter_by(video_id=video.id).first_or_404()
    segment = CaptionSegment.query.filter_by(
        caption_id=caption.id,
        segment_index=segment_index
    ).first_or_404()

    verification_time = data.get('verification_time', 0)
    
    try:
        if data.get('status') == 'approved':
            # Just approve the segment as is
            segment.verification_status = 'approved'
            segment.verified_text = segment.original_text
        elif data.get('status') == 'edited':
            # Update with edited text
            segment.verification_status = 'edited'
            segment.verified_text = data.get('text', segment.original_text)
        
        # Ensure verification_time is in seconds
        verification_time = float(data.get('verification_time', 0))
        if verification_time > 86400:  # If greater than 24 hours, assume it's in milliseconds
            verification_time = verification_time / 1000
        segment.verification_time = verification_time
        segment.verified_by = current_user.id
        segment.verified_at = datetime.utcnow()
        
        # Update caption progress
        verified_segments = CaptionSegment.query.filter(
            CaptionSegment.caption_id == caption.id,
            CaptionSegment.verification_status.in_(['approved', 'edited'])
        ).count()
        
        caption.verified_segments = verified_segments
        caption.verification_progress = (verified_segments / caption.total_segments * 100) if caption.total_segments > 0 else 0
        caption.is_verified = caption.verification_progress == 100
        
        if caption.is_verified:
            video.status = 'verified'
        
        db.session.commit()
        
        # Determine next unverified segment
        next_segment = CaptionSegment.query.filter(
            CaptionSegment.caption_id == caption.id,
            CaptionSegment.verification_status == 'pending',
            CaptionSegment.segment_index > segment_index
        ).order_by(CaptionSegment.segment_index).first()
        
        return jsonify({
            'success': True,
            'next_segment': next_segment.segment_index if next_segment else None,
            'verification_progress': caption.verification_progress
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error updating segment: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/segments/<string:video_id>/<int:segment_index>/timing', methods=['PUT'])
@login_required
def update_segment_timing(video_id, segment_index):
    """API endpoint to update a segment's timing."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
        
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    video = Video.query.filter_by(video_id=video_id).first_or_404()
    caption = Caption.query.filter_by(video_id=video.id).first_or_404()
    segment = CaptionSegment.query.filter_by(
        caption_id=caption.id,
        segment_index=segment_index
    ).first_or_404()
    
    try:
        start_time = float(data.get('start_time', segment.start_time))
        end_time = float(data.get('end_time', segment.end_time))
        
        if start_time >= end_time:
            return jsonify({'error': 'End time must be after start time'}), 400
            
        segment.start_time = start_time
        segment.end_time = end_time
        db.session.commit()
        
        return jsonify({
            'success': True,
            'start_time': start_time,
            'end_time': end_time
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error updating segment timing: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/segments/<string:video_id>/<int:segment_index>/report-issue', methods=['POST'])
@login_required
def report_segment_issue(video_id, segment_index):
    """API endpoint to report an issue with a segment."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
        
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    video = Video.query.filter_by(video_id=video_id).first_or_404()
    caption = Caption.query.filter_by(video_id=video.id).first_or_404()
    segment = CaptionSegment.query.filter_by(
        caption_id=caption.id,
        segment_index=segment_index
    ).first_or_404()
    
    try:
        issue_type = data.get('issueType')
        notes = data.get('notes', '')
        
        # Store issue information in segment notes
        segment.verification_notes = json.dumps({
            'type': issue_type,
            'notes': notes,
            'reported_by': current_user.id,
            'reported_at': datetime.utcnow().isoformat()
        })
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        print(f"Error reporting segment issue: {e}")
        return jsonify({'error': str(e)}), 500


@api.route('/v1/unassigned_videos')
@login_required
@admin_required
def get_unassigned_videos():
    """API endpoint to get unassigned videos."""
    # Get assigned video IDs from video_id field, not id
    assigned_video_ids = db.session.query(Video.video_id).join(VideoAssignment).distinct()
    
    # Get unassigned videos that are ready for verification
    videos = Video.query.filter(
        Video.video_id.notin_(assigned_video_ids),
        Video.status == 'captioned'
    ).all()
    
    return jsonify({
        'videos': [{
            'video_id': video.video_id,
            'title': video.title,
            'channel': video.channel,
            'duration': video.duration,
            'status': video.status,
            'thumbnail_url': f'/thumbnail/{video.video_id}?size=small'
        } for video in videos]
    })


@api.route('/v1/assign_videos', methods=['POST'])
@login_required
@admin_required
def assign_videos():
    """API endpoint to assign videos to users."""
    data = request.get_json()
    if not data or 'videos' not in data or 'users' not in data:
        return jsonify({'success': False, 'message': 'Missing required data'}), 400
        
    video_ids = data['videos']  # These are video_id strings
    user_ids = data['users']    # These are user id integers
    
    try:
        # Check if videos exist and are available
        videos = Video.query.filter(Video.video_id.in_(video_ids)).all()
        if len(videos) != len(video_ids):
            return jsonify({'success': False, 'message': 'One or more videos not found'}), 404
            
        # Check if users exist
        users = User.query.filter(User.id.in_(user_ids)).all()
        if len(users) != len(user_ids):
            return jsonify({'success': False, 'message': 'One or more users not found'}), 404
            
        # Create assignments and track activities
        assignments_created = 0
        assignments_skipped = 0
        activities = []
        now = datetime.utcnow()

        for video in videos:
            for user in users:
                # Check if assignment already exists
                existing = VideoAssignment.query.filter_by(
                    video_id=video.id,
                    user_id=user.id,
                ).first()
                
                if not existing:
                    assignment = VideoAssignment(
                        video_id=video.id,
                        user_id=user.id,
                        assigned_by=current_user.id,
                        status='assigned'
                    )
                    db.session.add(assignment)
                    assignments_created += 1
                    
                    # Add to activities list
                    activities.append({
                        'username': user.username,
                        'video_title': video.title,
                        'timestamp': convert_to_client_timezone(now)  # Use timezone conversion
                    })
                else:
                    assignments_skipped += 1
        
        db.session.commit()
        
        message = f'Created {assignments_created} video assignments'
        if assignments_skipped > 0:
            message += f' (skipped {assignments_skipped} existing assignments)'
            
        return jsonify({
            'success': True,
            'message': message,
            'activities': activities
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error assigning videos: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@api.route('/v1/reassign-videos/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def reassign_videos(user_id):
    """API endpoint to reassign a user's videos."""
    user = User.query.get_or_404(user_id)
    
    try:
        # Get incomplete assignments
        assignments = VideoAssignment.query.filter_by(
            user_id=user.id,
            status='assigned'
        ).all()
        
        if not assignments:
            return jsonify({'success': True, 'message': 'No videos to reassign'})
            
        # Remove assignments
        for assignment in assignments:
            db.session.delete(assignment)
            
        db.session.commit()
        return jsonify({
            'success': True,
            'message': f'Reassigned {len(assignments)} videos'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


@api.route('/user-progress/<int:user_id>')
@login_required
@admin_required
def get_user_progress(user_id):
    """Get detailed progress information for a user."""
    user = User.query.get_or_404(user_id)
    
    assignments = VideoAssignment.query.filter_by(user_id=user_id).all()
    
    progress = []
    for assignment in assignments:
        video = assignment.video
        caption = Caption.query.filter_by(video_id=video.id).first()
        if caption:
            verified_segments = CaptionSegment.query.filter(
                CaptionSegment.caption_id == caption.id,
                CaptionSegment.verified_by == user_id
            ).count()
            total_segments = caption.total_segments or 0
            progress.append({
                'video_id': video.video_id,
                'title': video.title,
                'verified_segments': verified_segments,
                'total_segments': total_segments,
                'progress': (verified_segments / total_segments * 100) if total_segments > 0 else 0
            })
    
    return jsonify({
        'user': {
            'username': user.username,
            'total_assignments': len(assignments),
            'completed_assignments': len([a for a in assignments if a.status == 'completed'])
        },
        'progress': progress
    })


@api.route('/v1/bulk_unassign/<string:video_id>', methods=['POST'])
@login_required
@admin_required
def bulk_unassign(video_id):
    """API endpoint to unassign multiple users from a video."""
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    selected_users = request.form.get('selected_users', '')
    
    if not selected_users:
        return jsonify({'success': False, 'message': 'No users selected'}), 400
    
    try:
        user_ids = [int(uid) for uid in selected_users.split(',')]
        
        # Find assignments to remove
        assignments = VideoAssignment.query.filter(
            VideoAssignment.video_id == video.id,
            VideoAssignment.user_id.in_(user_ids)
        ).all()
        
        if not assignments:
            return jsonify({'success': False, 'message': 'No matching assignments found'}), 404
        
        # Get usernames for response
        usernames = []
        for assignment in assignments:
            user = User.query.get(assignment.user_id)
            if user:
                usernames.append(user.username)
            db.session.delete(assignment)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Unassigned {len(assignments)} users from this video',
            'removed_users': usernames,
            'removed_count': len(assignments)
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


@api.route('/admin/assign-video/<string:video_id>', methods=['POST'])
@login_required
@admin_required
def assign_video(video_id):
    """Admin route to assign a video to a user."""
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'No user selected.'}), 400
    
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    user = User.query.get_or_404(user_id)
    
    # Check if video is already assigned to this user
    existing = VideoAssignment.query.filter_by(
        video_id=video.id,
        user_id=user.id,
    ).first()
    
    if existing:
        return jsonify({
            'success': False,
            'message': f'Video already assigned to {user.username}.'
        })
    else:
        assignment = VideoAssignment(
            video_id=video.id,
            user_id=user.id,
            assigned_by=current_user.id
        )
        db.session.add(assignment)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Video assigned to {user.username}.',
            'assignment': {
                'user_id': user.id,
                'username': user.username,
                'assigned_at': assignment.assigned_at.strftime('%Y-%m-%d')
            }
        })


@api.route('/user/<int:user_id>')
@login_required
@admin_required
def get_user_details(user_id):
    """API endpoint to get detailed information about a user."""
    user = User.query.get_or_404(user_id)
    
    # Get assignments with video details
    assignments_query = db.session.query(
        VideoAssignment, Video
    ).join(
        Video, VideoAssignment.video_id == Video.id
    ).filter(
        VideoAssignment.user_id == user_id
    ).order_by(
        VideoAssignment.assigned_at.desc()
    ).all()
    
    assignments_data = []
    for assignment, video in assignments_query:
        # Get caption progress if available
        caption = Caption.query.filter_by(video_id=video.id).first()
        progress = 0
        verified_segments = 0
        total_segments = 0
        
        if caption:
            verified_segments = CaptionSegment.query.filter(
                CaptionSegment.caption_id == caption.id,
                CaptionSegment.verified_by == user_id,
                CaptionSegment.verification_status.in_(['approved', 'edited'])
            ).count()
            total_segments = caption.total_segments or 0
            progress = (verified_segments / total_segments * 100) if total_segments > 0 else 0
            
        assignments_data.append({
            'id': assignment.id,
            'video_id': video.video_id,
            'title': video.title,
            'duration': video.duration,
            'status': assignment.status,
            'assigned_at': convert_to_client_timezone(assignment.assigned_at),
            'progress': progress,
            'verified_segments': verified_segments,
            'total_segments': total_segments
        })
    
    # Get user statistics
    total_segments_verified = CaptionSegment.query.filter_by(verified_by=user_id).count()
    total_verification_time = db.session.query(db.func.sum(CaptionSegment.verification_time))\
        .filter_by(verified_by=user_id).scalar()
    total_verification_time = float(total_verification_time) if total_verification_time else 0.0
    
    # Get recent verifications
    recent_verifications = CaptionSegment.query.filter_by(verified_by=user_id)\
        .order_by(CaptionSegment.verified_at.desc()).limit(10).all()
    recent_activity = []
    for verification in recent_verifications:
        caption = Caption.query.get(verification.caption_id)
        if caption:
            video = Video.query.get(caption.video_id)
            if video:
                recent_activity.append({
                    'segment_id': verification.id,
                    'video_id': video.video_id,
                    'title': video.title,
                    'verified_at': convert_to_client_timezone(verification.verified_at),
                    'status': verification.verification_status
                })
    
    return jsonify({
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'is_admin': user.is_admin,
            'created_at': convert_to_client_timezone(user.created_at),
            'total_segments_verified': total_segments_verified,
            'total_verification_time': total_verification_time,
            'hours_worked': total_verification_time / 3600 if total_verification_time else 0
        },
        'assignments': assignments_data,
        'recent_activity': recent_activity
    })

@api.route('/user/<int:user_id>/unassign/<string:video_id>', methods=['DELETE'])
@login_required
@admin_required
def unassign_user_video(user_id, video_id):
    """API endpoint to unassign a video from a user."""
    video = Video.query.filter_by(video_id=video_id).first_or_404()
    assignment = VideoAssignment.query.filter_by(
        video_id=video.id,
        user_id=user_id
    ).first_or_404()
    
    try:
        db.session.delete(assignment)
        db.session.commit()
        return jsonify({
            'success': True,
            'message': f'Successfully unassigned video from user'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@api.route('/user/<int:user_id>/unassign-all', methods=['DELETE'])
@login_required
@admin_required
def unassign_all_user_videos(user_id):
    """API endpoint to unassign all videos from a user."""
    user = User.query.get_or_404(user_id)
    assignments = VideoAssignment.query.filter_by(user_id=user_id).all()
    
    if not assignments:
        return jsonify({
            'success': False,
            'message': f'No assignments found for user {user.username}'
        })
    
    try:
        count = 0
        for assignment in assignments:
            db.session.delete(assignment)
            count += 1
        
        db.session.commit()
        return jsonify({
            'success': True,
            'message': f'Successfully unassigned {count} videos from {user.username}'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500
