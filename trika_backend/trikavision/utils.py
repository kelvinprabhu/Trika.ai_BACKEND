# WebSocket utilities and enhanced video processing
# utils.py for trikavision with WebSocket support

import cv2
import json
import base64
import numpy as np
import logging
from datetime import datetime
from .workoutAnalyser import WorkoutAnalyzer

# Set up logging
logger = logging.getLogger(__name__)

class WebSocketFrameProcessor:
    """Helper class for processing frames in WebSocket context"""
    
    def __init__(self, model_path=None):
        self.analyzer = WorkoutAnalyzer(model_path=model_path)
        self.processing_stats = {
            'frames_processed': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
    def process_base64_frame(self, base64_frame_data):
        """Process base64 encoded frame and return JSON-serializable result"""
        try:
            # Decode base64 frame
            if "," in base64_frame_data:
                base64_frame = base64_frame_data.split(",")[1]
            else:
                base64_frame = base64_frame_data
            
            frame_bytes = base64.b64decode(base64_frame)
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode frame")
            
            # Process frame
            result = self.analyzer.process_frame_for_websocket(frame)
            
            # Convert to JSON-serializable format
            json_result = self.format_result_for_json(result)
            
            self.processing_stats['frames_processed'] += 1
            
            return json_result
            
        except Exception as e:
            self.processing_stats['errors'] += 1
            logger.error(f"Error processing base64 frame: {e}")
            return self.get_error_result(str(e))
    
    def format_result_for_json(self, result):
        """Convert result to JSON-serializable format"""
        try:
            json_result = {
                'timestamp': datetime.now().isoformat(),
                'landmarks_detected': result.get('landmarks_detected', False),
                'exercise': result.get('predicted_exercise'),
                'confidence': float(result.get('confidence', 0)),
                'feedback': result.get('form_feedback', 'No feedback'),
                'angle': float(result.get('angle', 0)),
                'reps': int(result.get('repetition_count', 0)),
                'form_color': {
                    'r': int(result.get('form_color', (255, 255, 255))[0]),
                    'g': int(result.get('form_color', (255, 255, 255))[1]),
                    'b': int(result.get('form_color', (255, 255, 255))[2])
                },
                'processing_stats': self.get_processing_stats()
            }
            
            # Add joint position if available
            if 'joint_position' in result and result['joint_position']:
                json_result['joint_position'] = {
                    'x': float(result['joint_position'][0]),
                    'y': float(result['joint_position'][1])
                }
            
            return json_result
            
        except Exception as e:
            logger.error(f"Error formatting result for JSON: {e}")
            return self.get_error_result("Result formatting error")
    
    def get_error_result(self, error_message):
        """Get standardized error result"""
        return {
            'timestamp': datetime.now().isoformat(),
            'landmarks_detected': False,
            'exercise': None,
            'confidence': 0.0,
            'feedback': f'Error: {error_message}',
            'angle': 0,
            'reps': 0,
            'form_color': {'r': 255, 'g': 0, 'b': 0},
            'error': True,
            'processing_stats': self.get_processing_stats()
        }
    
    def get_processing_stats(self):
        """Get current processing statistics"""
        elapsed_time = (datetime.now() - self.processing_stats['start_time']).total_seconds()
        fps = self.processing_stats['frames_processed'] / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'frames_processed': self.processing_stats['frames_processed'],
            'errors': self.processing_stats['errors'],
            'fps': round(fps, 2),
            'elapsed_time': round(elapsed_time, 2),
            'analyzer_stats': self.analyzer.get_session_stats()
        }
    
    def reset_session(self):
        """Reset processing session"""
        self.analyzer.reset_session()
        self.processing_stats = {
            'frames_processed': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        logger.info("WebSocket frame processor session reset")


def encode_frame_to_base64(frame, format='.jpg', quality=85):
    """Encode OpenCV frame to base64 string"""
    try:
        if format == '.jpg':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        else:
            encode_param = []
        
        _, buffer = cv2.imencode(format, frame, encode_param)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{frame_base64}"
        
    except Exception as e:
        logger.error(f"Error encoding frame to base64: {e}")
        return None


def decode_base64_frame(base64_string):
    """Decode base64 string to OpenCV frame"""
    try:
        if "," in base64_string:
            base64_data = base64_string.split(",")[1]
        else:
            base64_data = base64_string
        
        frame_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        return frame
        
    except Exception as e:
        logger.error(f"Error decoding base64 frame: {e}")
        return None


class WorkoutSessionManager:
    """Manages workout sessions with detailed tracking"""
    
    def __init__(self):
        self.sessions = {}
        self.current_session_id = None
    
    def start_session(self, session_id=None):
        """Start a new workout session"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session_id = session_id
        self.sessions[session_id] = {
            'start_time': datetime.now(),
            'exercises': {},
            'total_frames': 0,
            'total_reps': 0,
            'active': True
        }
        
        logger.info(f"Started workout session: {session_id}")
        return session_id
    
    def update_session(self, exercise, reps, feedback_score=0):
        """Update current session with exercise data"""
        if not self.current_session_id:
            self.start_session()
        
        session = self.sessions[self.current_session_id]
        session['total_frames'] += 1
        
        if exercise:
            if exercise not in session['exercises']:
                session['exercises'][exercise] = {
                    'first_detected': datetime.now(),
                    'total_reps': 0,
                    'best_form_score': 0,
                    'total_feedback_scores': 0,
                    'feedback_count': 0
                }
            
            exercise_data = session['exercises'][exercise]
            exercise_data['total_reps'] = max(exercise_data['total_reps'], reps)
            session['total_reps'] = sum(ex['total_reps'] for ex in session['exercises'].values())
            
            # Update feedback scoring
            if feedback_score > 0:
                exercise_data['total_feedback_scores'] += feedback_score
                exercise_data['feedback_count'] += 1
                exercise_data['best_form_score'] = max(exercise_data['best_form_score'], feedback_score)
    
    def get_session_summary(self, session_id=None):
        """Get summary of workout session"""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        current_time = datetime.now()
        duration = (current_time - session['start_time']).total_seconds()
        
        summary = {
            'session_id': session_id,
            'start_time': session['start_time'].isoformat(),
            'duration': duration,
            'total_exercises': len(session['exercises']),
            'total_reps': session['total_reps'],
            'total_frames': session['total_frames'],
            'exercises': {}
        }
        
        for exercise, data in session['exercises'].items():
            avg_form_score = 0
            if data['feedback_count'] > 0:
                avg_form_score = data['total_feedback_scores'] / data['feedback_count']
            
            summary['exercises'][exercise] = {
                'total_reps': data['total_reps'],
                'best_form_score': data['best_form_score'],
                'average_form_score': round(avg_form_score, 2),
                'duration': (current_time - data['first_detected']).total_seconds()
            }
        
        return summary
    
    def end_session(self, session_id=None):
        """End current workout session"""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id in self.sessions:
            self.sessions[session_id]['active'] = False
            self.sessions[session_id]['end_time'] = datetime.now()
            
            if session_id == self.current_session_id:
                self.current_session_id = None
            
            logger.info(f"Ended workout session: {session_id}")
            return self.get_session_summary(session_id)
        
        return None


# Enhanced video processing function (original + WebSocket compatibility)
def process_video(video_path, model_path="/mnt/d/Projects/Trika AI Fitness Platform/referenceGit/workout_recognization/22_class_model_30fps.h5", 
                 websocket_compatible=False, frame_callback=None):
    """Process video with enhanced display and feedback, optionally WebSocket compatible"""
    
    # Initialize analyzer
    if websocket_compatible:
        processor = WebSocketFrameProcessor(model_path=model_path)
        analyzer = processor.analyzer
    else:
        analyzer = WorkoutAnalyzer(model_path=model_path)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
    
    if not websocket_compatible:
        print(f"Video loaded: {video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Total frames: {total_frames}")
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 'p' to pause/resume")
        print("- Press 'r' to reset counters")
        print("- Press SPACE to pause and analyze current frame")
        
        # Create display window
        cv2.namedWindow('Workout Analysis - ' + video_path, cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Workout Analysis - ' + video_path, 1200, 800)
    
    frame_count = 0
    paused = False
    exercise_summary = {}
    session_manager = WorkoutSessionManager()
    session_id = session_manager.start_session()
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
        
        # Process frame
        if websocket_compatible:
            # Use WebSocket-compatible processing
            result = processor.process_base64_frame(encode_frame_to_base64(frame))
            
            # Call frame callback if provided
            if frame_callback:
                frame_callback(result, frame_count, total_frames)
            
            # Update session
            session_manager.update_session(
                result.get('exercise'),
                result.get('reps', 0),
                calculate_feedback_score(result.get('feedback', ''))
            )
            
        else:
            # Use original processing with display
            processed_frame = analyzer.process_frame(frame.copy())
            
            # Add enhanced overlay
            overlay_frame = add_detailed_overlay(
                processed_frame, analyzer, frame_count, total_frames, fps, exercise_summary
            )
            
            # Track exercise statistics
            if analyzer.current_exercise:
                if analyzer.current_exercise not in exercise_summary:
                    exercise_summary[analyzer.current_exercise] = {
                        'total_reps': 0,
                        'total_time': 0,
                        'first_seen': frame_count
                    }
                exercise_summary[analyzer.current_exercise]['total_reps'] = analyzer.form_checker.repetition_count
                exercise_summary[analyzer.current_exercise]['total_time'] = (frame_count - exercise_summary[analyzer.current_exercise]['first_seen']) / fps
            
            # Display frame
            cv2.imshow('Workout Analysis - ' + video_path, overlay_frame)
            
            # Control playback
            wait_time = 1 if paused else int(1000/fps)
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('p') or key == ord(' '):  # Pause/Resume
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord('r'):  # Reset counters
                analyzer.reset_session()
                exercise_summary.clear()
                session_manager = WorkoutSessionManager()
                session_id = session_manager.start_session()
                print("Counters reset!")
    
    # Cleanup
    cap.release()
    if not websocket_compatible:
        cv2.destroyAllWindows()
    
    # Get final session summary
    final_summary = session_manager.end_session(session_id)
    
    logger.info(f"Video processing complete. Processed {frame_count} frames")
    
    if websocket_compatible:
        return final_summary
    else:
        # Print summary (original behavior)
        print(f"\n=== WORKOUT ANALYSIS COMPLETE ===")
        print(f"Processed {frame_count} frames")
        print(f"Video duration processed: {frame_count/fps:.2f} seconds")
        
        if exercise_summary:
            print("\n=== EXERCISE SUMMARY ===")
            total_workout_time = 0
            for exercise, stats in exercise_summary.items():
                print(f"{exercise.upper()}:")
                print(f"  - Repetitions: {int(stats['total_reps'])}")
                print(f"  - Time spent: {stats['total_time']:.1f} seconds")
                total_workout_time += stats['total_time']
            print(f"\nTotal active workout time: {total_workout_time:.1f} seconds")
        else:
            print("No exercises detected in the video")


def calculate_feedback_score(feedback):
    """Calculate numerical score from feedback text"""
    if not feedback:
        return 0
    
    feedback = feedback.lower()
    if "perfect" in feedback or "excellent" in feedback:
        return 100
    elif "great" in feedback or "good" in feedback:
        return 80
    elif "adjust" in feedback or "improve" in feedback:
        return 60
    elif "try" in feedback or "focus" in feedback:
        return 40
    else:
        return 20


def add_detailed_overlay(frame, analyzer, frame_count, total_frames, fps, exercise_summary):
    """Add comprehensive overlay with workout information (unchanged from original)"""
    # Keep the original overlay function implementation
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    
    # Define regions
    top_bar_height = 40
    main_info_height = 150
    summary_width = 300 if exercise_summary else 0
    bottom_bar_height = 30
    
    # Create semi-transparent backgrounds for each section
    # Top progress bar
    cv2.rectangle(overlay, (0, 0), (w, top_bar_height), (0, 0, 0), -1)
    
    # Main info panel
    cv2.rectangle(overlay, (10, top_bar_height + 10), (w - summary_width - 20, top_bar_height + main_info_height), (0, 0, 0), -1)
    
    # Summary sidebar (if exists)
    if exercise_summary:
        cv2.rectangle(overlay, (w - summary_width, top_bar_height + 10), (w - 10, h - bottom_bar_height - 10), (0, 0, 0), -1)
    
    # Bottom controls bar
    cv2.rectangle(overlay, (0, h - bottom_bar_height), (w, h), (0, 0, 0), -1)
    
    # Apply transparency
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Progress bar (top)
    progress = frame_count / total_frames if total_frames > 0 else 0
    bar_width = w - 40
    cv2.rectangle(frame, (20, 15), (20 + int(bar_width * progress), 25), (0, 255, 0), -1)
    cv2.rectangle(frame, (20, 15), (20 + bar_width, 25), (255, 255, 255), 2)
    
    # Video info (top right)
    current_time = frame_count / fps
    total_time = total_frames / fps
    time_text = f"Time: {current_time:.1f}s / {total_time:.1f}s"
    frame_text = f"Frame: {frame_count} / {total_frames}"
    
    cv2.putText(frame, time_text, (w - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, frame_text, (w - 250, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Current exercise info (main panel)
    y_offset = top_bar_height + 30
    
    if analyzer.current_exercise:
        exercise_color = (0, 255, 0)  # Green for detected
        
        # Get current prediction confidence
        if len(analyzer.predictions_history) > 0:
            cv2.putText(frame, f"Detection: ACTIVE", 
                       (500, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, exercise_color, 2)
        
        # Form feedback
        if hasattr(analyzer, 'last_feedback'):
            feedback_color = (0, 255, 0) if 'Perfect' in analyzer.last_feedback or 'Great' in analyzer.last_feedback else (0, 165, 255)
            # Wrap feedback text if needed
            feedback = analyzer.last_feedback
            if len(feedback) > 40:  # Arbitrary length limit
                # Simple text wrapping
                parts = [feedback[i:i+40] for i in range(0, len(feedback), 40)]
                for i, part in enumerate(parts):
                    cv2.putText(frame, part, (20, y_offset + 90 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback_color, 1)
            else:
                cv2.putText(frame, feedback, (20, y_offset + 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback_color, 1)
    else:
        cv2.putText(frame, "EXERCISE: Detecting...", 
                   (20, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, "Stand in view of camera", 
                   (20, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Exercise summary sidebar (if any exercises detected)
    if exercise_summary:
        sidebar_x = w - summary_width + 10
        cv2.putText(frame, "WORKOUT SUMMARY", 
                   (sidebar_x, top_bar_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset = top_bar_height + 60
        for exercise, stats in exercise_summary.items():
            color = (0, 255, 0) if exercise == analyzer.current_exercise else (255, 255, 255)
            cv2.putText(frame, f"{exercise[:20]}", 
                       (sidebar_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"Reps: {int(stats['total_reps'])}", 
                       (sidebar_x, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(frame, f"Time: {stats['total_time']:.1f}s", 
                       (sidebar_x, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 55
            
            # Don't go beyond the bottom of the summary area
            if y_offset > h - bottom_bar_height - 20:
                cv2.putText(frame, "...", (sidebar_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                break
    
    # Instructions (bottom)
    cv2.putText(frame, "Controls: Q=Quit, P=Pause, R=Reset, SPACE=Pause", 
               (20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame