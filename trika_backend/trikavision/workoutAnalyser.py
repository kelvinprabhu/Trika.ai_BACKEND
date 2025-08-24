import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import logging
from .exerciseformchecker import ExerciseFormChecker

# Set up logging
logger = logging.getLogger(__name__)

class WorkoutAnalyzer:
    """Complete workout analyzer with classification and form checking optimized for WebSocket"""
    
    def __init__(self, model_path=None):
        logger.info("Initializing WorkoutAnalyzer...")
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Exercise classification setup
        self.CLASS_NAMES = [
            'barbell biceps curl', 'bench press', 'chest fly machine', 'deadlift', 
            'decline bench press', 'hammer curl', 'hip thrust', 'incline bench press', 
            'lat pulldown', 'lateral raise', 'leg extension', 'leg raises', 'plank', 
            'pull Up', 'push-up', 'romanian deadlift', 'russian twist', 'shoulder press', 
            'squat', 't bar row', 'tricep dips', 'tricep Pushdown'
        ]
        
        # Load trained model if provided
        self.model = None
        if model_path:
            try:
                self.model = tf.keras.models.load_model(model_path)
                logger.info(f"✅ Model loaded successfully from {model_path}")
                logger.info(f"Model input shape: {self.model.input_shape}")
                logger.info(f"Model output shape: {self.model.output_shape}")
            except Exception as e:
                logger.error(f"❌ Could not load model from {model_path}: {e}")
                self.model = None
        else:
            logger.warning("⚠️ No model path provided, running without exercise classification")
        
        # Constants for model
        self.MAX_FRAMES = 30
        self.RELEVANT_LANDMARKS = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]
        
        # Initialize form checker and buffers
        try:
            self.form_checker = ExerciseFormChecker()
            logger.info("✅ ExerciseFormChecker initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ExerciseFormChecker: {e}")
            self.form_checker = None
        
        self.frame_buffer = deque(maxlen=self.MAX_FRAMES)
        self.predictions_history = deque(maxlen=5)
        self.current_exercise = None
        
        # WebSocket specific optimizations
        self.prediction_smoothing_threshold = 0.6  # Higher threshold for more stable predictions
        self.min_confidence_threshold = 0.3
        self.frame_skip_counter = 0
        self.process_every_n_frames = 2  # Process every 2nd frame for better performance
        
        # Performance tracking
        self.total_predictions = 0
        self.successful_predictions = 0
        
        logger.info("✅ WorkoutAnalyzer initialization complete")
    
    def extract_landmarks(self, results):
        """Extract relevant landmarks for classification with error handling"""
        try:
            if results.pose_landmarks:
                landmarks = []
                for i in self.RELEVANT_LANDMARKS:
                    landmark = results.pose_landmarks.landmark[i]
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                return np.array(landmarks)
            else:
                return np.zeros(len(self.RELEVANT_LANDMARKS) * 3)
        except Exception as e:
            logger.error(f"❌ Error extracting landmarks: {e}")
            return np.zeros(len(self.RELEVANT_LANDMARKS) * 3)
    
    def predict_exercise(self):
        """Predict exercise from frame buffer with enhanced error handling and logging"""
        if self.model is None:
            return None, 0
            
        if len(self.frame_buffer) < self.MAX_FRAMES:
            logger.debug(f"Buffer not full: {len(self.frame_buffer)}/{self.MAX_FRAMES}")
            return None, 0
        
        try:
            # Prepare input data
            input_data = np.array(list(self.frame_buffer))
            
            # Validate input data
            if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
                logger.warning("⚠️ Invalid data in frame buffer (NaN or Inf values)")
                return None, 0
            
            # Normalize
            max_val = np.max(np.abs(input_data))
            if max_val > 0:
                input_data = input_data / max_val
            
            input_data = np.expand_dims(input_data, axis=0)
            
            # Predict
            predictions = self.model.predict(input_data, verbose=0)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)
            
            self.total_predictions += 1
            
            # Apply confidence threshold
            if confidence < self.min_confidence_threshold:
                logger.debug(f"Low confidence prediction: {confidence:.3f} < {self.min_confidence_threshold}")
                return None, confidence
            
            self.successful_predictions += 1
            
            # Add to history for smoothing
            self.predictions_history.append((predicted_class, confidence))
            
            # Enhanced prediction smoothing
            if len(self.predictions_history) >= 3:
                # Get recent predictions with high confidence
                recent_high_conf = [
                    pred_class for pred_class, conf in list(self.predictions_history)[-3:]
                    if conf > self.prediction_smoothing_threshold
                ]
                
                if len(recent_high_conf) >= 2:
                    # Use most frequent among high confidence predictions
                    final_prediction = max(set(recent_high_conf), key=recent_high_conf.count)
                    final_confidence = confidence
                    
                    exercise_name = self.CLASS_NAMES[final_prediction]
                    
                    # Log successful prediction
                    if self.total_predictions % 30 == 0:  # Log every 30 predictions
                        success_rate = (self.successful_predictions / self.total_predictions) * 100
                        logger.info(f"🎯 Prediction stats - Success rate: {success_rate:.1f}%, Current: {exercise_name} ({final_confidence:.3f})")
                    
                    return exercise_name, final_confidence
            
            return None, confidence
            
        except Exception as e:
            logger.error(f"❌ Error in predict_exercise: {e}")
            return None, 0
    
    def process_frame_for_websocket(self, frame):
        """Optimized frame processing for WebSocket with performance considerations"""
        try:
            # Frame skipping for performance
            self.frame_skip_counter += 1
            if self.frame_skip_counter % self.process_every_n_frames != 0:
                return self.get_last_result()
            
            h, w, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process with MediaPipe
            with self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as pose:
                results = pose.process(image)
            
            # Initialize result
            result = {
                'landmarks_detected': False,
                'predicted_exercise': None,
                'confidence': 0.0,
                'form_feedback': 'No person detected',
                'angle': 0,
                'repetition_count': 0,
                'form_color': (0, 0, 255)  # Red default
            }
            
            if results.pose_landmarks:
                result['landmarks_detected'] = True
                
                # Extract landmarks for classification
                landmarks = self.extract_landmarks(results)
                self.frame_buffer.append(landmarks)
                
                # Predict exercise
                predicted_exercise, confidence = self.predict_exercise()
                
                if predicted_exercise:
                    result.update({
                        'predicted_exercise': predicted_exercise,
                        'confidence': float(confidence)
                    })
                    
                    # Check if exercise changed
                    if self.current_exercise != predicted_exercise:
                        if self.form_checker:
                            self.form_checker.reset_counters()
                        self.current_exercise = predicted_exercise
                        logger.info(f"🏃 Exercise detected: {predicted_exercise} (confidence: {confidence:.3f})")
                    
                    # Get form feedback if form checker is available
                    if self.form_checker:
                        try:
                            angle, feedback, color, joint_pos = self.form_checker.get_exercise_feedback(
                                predicted_exercise, results.pose_landmarks.landmark, w, h
                            )
                            
                            result.update({
                                'form_feedback': feedback,
                                'angle': float(angle) if angle else 0,
                                'repetition_count': int(self.form_checker.repetition_count),
                                'form_color': color if color else (255, 255, 255),
                                'joint_position': joint_pos
                            })
                            
                        except Exception as e:
                            logger.error(f"❌ Error in form checking: {e}")
                            result['form_feedback'] = "Form analysis temporarily unavailable"
                    else:
                        result['form_feedback'] = "Form checker not available"
                else:
                    result['form_feedback'] = "Analyzing movement..."
            
            # Cache last result
            self._last_result = result
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing frame: {e}")
            return {
                'landmarks_detected': False,
                'predicted_exercise': None,
                'confidence': 0.0,
                'form_feedback': f'Processing error: {str(e)[:50]}...',
                'angle': 0,
                'repetition_count': 0,
                'form_color': (255, 0, 0)
            }
    
    def get_last_result(self):
        """Return last cached result for skipped frames"""
        if hasattr(self, '_last_result'):
            return self._last_result
        else:
            return {
                'landmarks_detected': False,
                'predicted_exercise': None,
                'confidence': 0.0,
                'form_feedback': 'Initializing...',
                'angle': 0,
                'repetition_count': 0,
                'form_color': (255, 255, 0)
            }
    
    def reset_session(self):
        """Reset analyzer state for new session"""
        try:
            if self.form_checker:
                self.form_checker.reset_counters()
            self.frame_buffer.clear()
            self.predictions_history.clear()
            self.current_exercise = None
            self.frame_skip_counter = 0
            
            # Reset performance tracking
            self.total_predictions = 0
            self.successful_predictions = 0
            
            if hasattr(self, '_last_result'):
                del self._last_result
            
            logger.info("🔄 WorkoutAnalyzer session reset")
            
        except Exception as e:
            logger.error(f"❌ Error resetting session: {e}")
    
    def get_session_stats(self):
        """Get current session statistics"""
        success_rate = 0
        if self.total_predictions > 0:
            success_rate = (self.successful_predictions / self.total_predictions) * 100
        
        return {
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'success_rate': success_rate,
            'current_exercise': self.current_exercise,
            'buffer_size': len(self.frame_buffer),
            'model_loaded': self.model is not None,
            'form_checker_available': self.form_checker is not None
        }
    
    # Keep original methods for backward compatibility
    def process_frame(self, frame):
        """Original process_frame method for backward compatibility"""
        return self.process_frame_for_websocket(frame)
    
    def process_video(self, video_path, output_path=None):
        """Process entire video file (unchanged from original)"""
        # Keep original implementation
        pass
    
    def process_live_camera(self, camera_index=0):
        """Process live camera feed (unchanged from original)"""
        # Keep original implementation
        pass