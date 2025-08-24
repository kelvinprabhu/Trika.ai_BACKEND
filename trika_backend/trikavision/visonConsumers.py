from channels.generic.websocket import AsyncWebsocketConsumer
import json
import base64
import cv2
import numpy as np
import asyncio
import logging
from asgiref.sync import sync_to_async
from .workoutAnalyser import WorkoutAnalyzer
import traceback
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class PostureConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        print("✅ WebSocket Connected")
        logger.info("WebSocket connection established")
        
        # Initialize workout analyzer with model
        try:
            model_path = "/mnt/d/Projects/Trika AI Fitness Platform/referenceGit/workout_recognization/22_class_model_30fps.h5"  # Update this path
            self.workout_analyzer = WorkoutAnalyzer(model_path=model_path)
            logger.info(f"✅ WorkoutAnalyzer initialized successfully with model: {model_path}")
            
            # Initialize session tracking
            self.session_start_time = datetime.now()
            self.frame_count = 0
            self.last_exercise = None
            self.exercise_sessions = {}
            self.is_paused = False
            self.pause_start_time = None
            self.total_pause_duration = 0
            
            # Send initialization success message
            await self.send(text_data=json.dumps({
                "type": "system",
                "payload": {
                    "status": "initialized",
                    "message": "Workout analyzer ready",
                    "model_loaded": self.workout_analyzer.model is not None,
                    "supported_exercises": len(self.workout_analyzer.CLASS_NAMES)
                }
            }))
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WorkoutAnalyzer: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Send error message to client
            await self.send(text_data=json.dumps({
                "type": "error",
                "payload": {
                    "message": f"Failed to initialize workout analyzer: {str(e)}",
                    "model_loaded": False
                }
            }))
    
    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            msg_type = data.get("type")
            
            if msg_type == "frame":
                await self.process_frame(data)
            elif msg_type == "reset":
                await self.reset_session()
            elif msg_type == "get_summary":
                await self.send_session_summary()
            elif msg_type == "camera_ready":
                await self.handle_camera_ready(data.get("payload", {}))
            elif msg_type == "start_workout":
                await self.handle_start_workout(data.get("payload", {}))
            elif msg_type == "pause_workout":
                await self.handle_pause_workout(data.get("payload", {}))
            elif msg_type == "resume_workout":
                await self.handle_resume_workout(data.get("payload", {}))
            elif msg_type == "end_workout":
                await self.handle_end_workout(data.get("payload", {}))
            elif msg_type == "camera_error":
                await self.handle_camera_error(data.get("payload", {}))
            else:
                logger.warning(f"Unknown message type received: {msg_type}")
                
        except Exception as e:
            logger.error(f"❌ Error processing message: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send_error_message(str(e))
    
    async def handle_camera_ready(self, payload):
        """Handle camera ready notification from frontend"""
        try:
            width = payload.get("width", 1280)
            height = payload.get("height", 720)
            frame_rate = payload.get("frameRate", 30)
            
            logger.info(f"📷 Camera ready: {width}x{height} @ {frame_rate}fps")
            
            await self.send(text_data=json.dumps({
                "type": "system",
                "payload": {
                    "status": "camera_ready",
                    "message": f"Camera initialized: {width}x{height}",
                    "camera_specs": {
                        "width": width,
                        "height": height,
                        "frameRate": frame_rate
                    }
                }
            }))
            
        except Exception as e:
            logger.error(f"❌ Error handling camera ready: {str(e)}")
            await self.send_error_message(f"Camera ready error: {str(e)}")
    
    async def handle_start_workout(self, payload):
        """Handle start workout command from frontend"""
        try:
            workout_type = payload.get("workoutType", "General Workout")
            timestamp = payload.get("timestamp", datetime.now().isoformat())
            
            logger.info(f"🏃 Starting workout: {workout_type}")
            
            # Reset session data
            self.session_start_time = datetime.now()
            self.frame_count = 0
            self.last_exercise = None
            self.exercise_sessions = {}
            self.is_paused = False
            self.pause_start_time = None
            self.total_pause_duration = 0
            
            # Reset workout analyzer
            if hasattr(self, 'workout_analyzer'):
                self.workout_analyzer.form_checker.reset_counters()
                self.workout_analyzer.frame_buffer.clear()
                self.workout_analyzer.predictions_history.clear()
            
            await self.send(text_data=json.dumps({
                "type": "system",
                "payload": {
                    "status": "workout_started",
                    "message": f"Workout session started: {workout_type}",
                    "workout_type": workout_type,
                    "timestamp": timestamp
                }
            }))
            
        except Exception as e:
            logger.error(f"❌ Error starting workout: {str(e)}")
            await self.send_error_message(f"Start workout error: {str(e)}")
    
    async def handle_pause_workout(self, payload):
        """Handle pause workout command from frontend"""
        try:
            timestamp = payload.get("timestamp", datetime.now().isoformat())
            session_duration = payload.get("sessionDuration", 0)
            
            logger.info(f"⏸️ Pausing workout at {session_duration}s")
            
            self.is_paused = True
            self.pause_start_time = datetime.now()
            
            await self.send(text_data=json.dumps({
                "type": "system",
                "payload": {
                    "status": "workout_paused",
                    "message": "Workout session paused",
                    "session_duration": session_duration,
                    "timestamp": timestamp
                }
            }))
            
        except Exception as e:
            logger.error(f"❌ Error pausing workout: {str(e)}")
            await self.send_error_message(f"Pause workout error: {str(e)}")
    
    async def handle_resume_workout(self, payload):
        """Handle resume workout command from frontend"""
        try:
            timestamp = payload.get("timestamp", datetime.now().isoformat())
            
            logger.info("▶️ Resuming workout")
            
            # Calculate pause duration
            if self.pause_start_time:
                pause_duration = (datetime.now() - self.pause_start_time).total_seconds()
                self.total_pause_duration += pause_duration
            
            self.is_paused = False
            self.pause_start_time = None
            
            await self.send(text_data=json.dumps({
                "type": "system",
                "payload": {
                    "status": "workout_resumed",
                    "message": "Workout session resumed",
                    "total_pause_duration": self.total_pause_duration,
                    "timestamp": timestamp
                }
            }))
            
        except Exception as e:
            logger.error(f"❌ Error resuming workout: {str(e)}")
            await self.send_error_message(f"Resume workout error: {str(e)}")
    
    async def handle_end_workout(self, payload):
        """Handle end workout command from frontend"""
        try:
            workout_type = payload.get("workoutType", "Unknown")
            duration = payload.get("duration", 0)
            reps = payload.get("reps", 0)
            accuracy = payload.get("accuracy", 0)
            
            logger.info(f"🏁 Ending workout: {workout_type}, Duration: {duration}s, Reps: {reps}")
            
            # Generate final session summary
            await self.send_session_summary()
            
            await self.send(text_data=json.dumps({
                "type": "system",
                "payload": {
                    "status": "workout_ended",
                    "message": f"Workout completed: {workout_type}",
                    "final_stats": {
                        "workout_type": workout_type,
                        "duration": duration,
                        "reps": reps,
                        "accuracy": accuracy,
                        "total_pause_duration": self.total_pause_duration
                    }
                }
            }))
            
        except Exception as e:
            logger.error(f"❌ Error ending workout: {str(e)}")
            await self.send_error_message(f"End workout error: {str(e)}")
    
    async def handle_camera_error(self, payload):
        """Handle camera error notification from frontend"""
        try:
            error_message = payload.get("error", "Unknown camera error")
            logger.error(f"📷 Camera error reported by frontend: {error_message}")
            
            await self.send(text_data=json.dumps({
                "type": "system",
                "payload": {
                    "status": "camera_error_acknowledged",
                    "message": "Camera error received, please check camera permissions",
                    "error_details": error_message
                }
            }))
            
        except Exception as e:
            logger.error(f"❌ Error handling camera error: {str(e)}")
            await self.send_error_message(f"Camera error handling failed: {str(e)}")
    
    async def process_frame(self, data):
        """Process incoming video frame"""
        try:
            # Skip processing if paused
            if self.is_paused:
                return
            
            # Decode base64 frame
            b64_frame = data["payload"].split(",")[1]  # remove data:image/jpeg;base64,
            frame_bytes = base64.b64decode(b64_frame)
            
            # Convert to OpenCV image
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("❌ Failed to decode frame")
                return
            
            self.frame_count += 1
            
            # Process frame with workout analyzer (run in thread to avoid blocking)
            result = await sync_to_async(self.analyze_frame)(frame)
            
            # Log frame processing
            if self.frame_count % 30 == 0:  # Log every 30 frames (roughly every second)
                logger.info(f"📊 Processed {self.frame_count} frames. Current exercise: {result.get('exercise', 'None')}")
            
            # Send results to client
            await self.send(text_data=json.dumps({
                "type": "workout",
                "payload": result
            }))
            
        except Exception as e:
            logger.error(f"❌ Error processing frame: {str(e)}")
            logger.error(traceback.format_exc())
            await self.send_error_message(f"Frame processing error: {str(e)}")
    
    def analyze_frame(self, frame):
        """Analyze frame using workout analyzer (sync function)"""
        try:
            # Get frame dimensions
            h, w, _ = frame.shape
            
            # Convert frame for MediaPipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process with MediaPipe
            with self.workout_analyzer.mp_pose.Pose(
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5
            ) as pose:
                results = pose.process(image)
            
            # Calculate effective session time (excluding pauses)
            current_time = datetime.now()
            if self.is_paused and self.pause_start_time:
                # Currently paused, calculate time up to pause start
                effective_session_time = (self.pause_start_time - self.session_start_time).total_seconds() - self.total_pause_duration
            else:
                # Not paused, calculate total time minus previous pauses
                effective_session_time = (current_time - self.session_start_time).total_seconds() - self.total_pause_duration
            
            # Initialize result data
            result_data = {
                "exercise": None,
                "confidence": 0.0,
                "feedback": "No person detected",
                "angle": 0,
                "reps": 0,
                "landmarks_detected": False,
                "form_color": [255, 0, 0],  # Red default
                "session_time": max(0, effective_session_time),
                "frame_count": self.frame_count,
                "is_paused": self.is_paused
            }
            
            if results.pose_landmarks:
                result_data["landmarks_detected"] = True
                
                # Extract landmarks for classification
                landmarks = self.workout_analyzer.extract_landmarks(results)
                self.workout_analyzer.frame_buffer.append(landmarks)
                
                # Predict exercise
                predicted_exercise, confidence = self.workout_analyzer.predict_exercise()
                
                if predicted_exercise and confidence > 0.3:  # Confidence threshold
                    result_data["exercise"] = predicted_exercise
                    result_data["confidence"] = float(confidence)
                    
                    # Check if exercise changed
                    if self.last_exercise != predicted_exercise:
                        logger.info(f"🏃 Exercise changed: {self.last_exercise} → {predicted_exercise}")
                        self.workout_analyzer.form_checker.reset_counters()
                        self.last_exercise = predicted_exercise
                        
                        # Track exercise session
                        if predicted_exercise not in self.exercise_sessions:
                            self.exercise_sessions[predicted_exercise] = {
                                "start_time": datetime.now(),
                                "total_reps": 0,
                                "best_form_score": 0
                            }
                    
                    # Get form feedback
                    angle, feedback, color, joint_pos = self.workout_analyzer.form_checker.get_exercise_feedback(
                        predicted_exercise, results.pose_landmarks.landmark, w, h
                    )
                    
                    result_data.update({
                        "feedback": feedback,
                        "angle": float(angle) if angle else 0,
                        "reps": int(self.workout_analyzer.form_checker.repetition_count),
                        "form_color": [int(c) for c in color] if color else [255, 255, 255]
                    })
                    
                    # Update exercise session data
                    if predicted_exercise in self.exercise_sessions:
                        self.exercise_sessions[predicted_exercise]["total_reps"] = int(
                            self.workout_analyzer.form_checker.repetition_count
                        )
                        
                        # Simple form scoring (you can enhance this)
                        form_score = self.calculate_form_score(feedback, angle)
                        if form_score > self.exercise_sessions[predicted_exercise]["best_form_score"]:
                            self.exercise_sessions[predicted_exercise]["best_form_score"] = form_score
                
                else:
                    result_data["feedback"] = "Analyzing movement..."
                    logger.debug(f"Low confidence prediction: {predicted_exercise} ({confidence:.2f})")
            
            return result_data
            
        except Exception as e:
            logger.error(f"❌ Error in analyze_frame: {str(e)}")
            return {
                "exercise": None,
                "confidence": 0.0,
                "feedback": f"Analysis error: {str(e)}",
                "angle": 0,
                "reps": 0,
                "landmarks_detected": False,
                "form_color": [255, 0, 0],
                "session_time": 0,
                "frame_count": self.frame_count,
                "is_paused": self.is_paused
            }
    
    def calculate_form_score(self, feedback, angle):
        """Simple form scoring based on feedback"""
        if "Perfect" in feedback or "Great" in feedback:
            return 100
        elif "Good" in feedback:
            return 80
        elif "Adjust" in feedback or "Improve" in feedback:
            return 60
        else:
            return 40
    
    async def reset_session(self):
        """Reset workout session"""
        try:
            self.workout_analyzer.form_checker.reset_counters()
            self.workout_analyzer.frame_buffer.clear()
            self.workout_analyzer.predictions_history.clear()
            self.last_exercise = None
            self.exercise_sessions.clear()
            self.session_start_time = datetime.now()
            self.frame_count = 0
            self.is_paused = False
            self.pause_start_time = None
            self.total_pause_duration = 0
            
            logger.info("🔄 Workout session reset")
            
            await self.send(text_data=json.dumps({
                "type": "system",
                "payload": {
                    "status": "reset",
                    "message": "Session reset successfully"
                }
            }))
            
        except Exception as e:
            logger.error(f"❌ Error resetting session: {str(e)}")
            await self.send_error_message(f"Reset error: {str(e)}")
    
    async def send_session_summary(self):
        """Send workout session summary"""
        try:
            session_duration = (datetime.now() - self.session_start_time).total_seconds() - self.total_pause_duration
            
            summary = {
                "session_duration": max(0, session_duration),
                "total_frames": self.frame_count,
                "exercises_performed": len(self.exercise_sessions),
                "total_pause_duration": self.total_pause_duration,
                "exercise_details": {}
            }
            
            for exercise, data in self.exercise_sessions.items():
                exercise_duration = (datetime.now() - data["start_time"]).total_seconds()
                summary["exercise_details"][exercise] = {
                    "total_reps": data["total_reps"],
                    "duration": exercise_duration,
                    "best_form_score": data["best_form_score"]
                }
            
            await self.send(text_data=json.dumps({
                "type": "summary",
                "payload": summary
            }))
            
            logger.info(f"📈 Session summary sent: {len(self.exercise_sessions)} exercises performed")
            
        except Exception as e:
            logger.error(f"❌ Error sending summary: {str(e)}")
            await self.send_error_message(f"Summary error: {str(e)}")
    
    async def send_error_message(self, error_msg):
        """Send error message to client"""
        try:
            await self.send(text_data=json.dumps({
                "type": "error",
                "payload": {
                    "message": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
            }))
        except Exception as e:
            logger.error(f"❌ Failed to send error message: {str(e)}")
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        session_duration = (datetime.now() - self.session_start_time).total_seconds() - self.total_pause_duration
        logger.info(f"❌ WebSocket Disconnected: {close_code}")
        logger.info(f"📊 Session stats - Duration: {session_duration:.2f}s, Frames: {self.frame_count}, Exercises: {len(self.exercise_sessions)}")
        
        # Clean up resources
        if hasattr(self, 'workout_analyzer'):
            del self.workout_analyzer