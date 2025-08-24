import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# --- Utility Functions ---
def calculate_angle(a, b, c):
    """Calculate angle between three points (b is the vertex)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle
    
class ExerciseFormChecker:
    """Modular exercise form checker for 22 different workout types"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.prev_angle = None
        self.direction = None
        self.repetition_count = 0
        self.direction_state = 0  # For push-up logic
        
    def reset_counters(self):
        """Reset rep counters for new exercise"""
        self.prev_angle = None
        self.direction = None
        self.repetition_count = 0
        self.direction_state = 0

    def check_squat(self, landmarks, w, h):
        """Squat form checking logic"""
        hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
               landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
        knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]
        ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
        
        angle = calculate_angle(hip, knee, ankle)
        color = (0, 255, 0) if 70 <= angle <= 160 else (0, 0, 255)
        
        if 70 <= angle <= 160:
            feedback = 'Perfect squat depth!'
        elif angle < 70:
            feedback = 'Great depth! Now stand up!'
        else:
            feedback = 'Go deeper - parallel thighs!'
            
        # Rep counting
        if self.prev_angle is not None:
            if angle > 150 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 90 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, knee

    def check_barbell_biceps_curl(self, landmarks, w, h):
        """Barbell biceps curl form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        color = (0, 255, 0) if 30 <= angle <= 160 else (0, 0, 255)
        
        if angle < 40:
            feedback = 'Great contraction!'
        elif angle < 60:
            feedback = 'Keep curling up!'
        elif angle > 150:
            feedback = 'Perfect extension!'
        elif angle > 130:
            feedback = 'Control the weight down!'
        else:
            feedback = 'Good form!'
            
        # Rep counting
        if self.prev_angle is not None:
            if angle > 150 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 50 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, elbow

    def check_bench_press(self, landmarks, w, h):
        """Bench press form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        color = (0, 255, 0) if 70 <= angle <= 180 else (0, 0, 255)
        
        if angle < 80:
            feedback = 'Touch chest, then press!'
        elif angle > 170:
            feedback = 'Perfect lockout!'
        else:
            feedback = 'Press up strong!'
            
        if self.prev_angle is not None:
            if angle > 160 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 90 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, elbow

    def check_push_up(self, landmarks, w, h):
        """Push-up form checking with enhanced logic"""
        elbow_angle = calculate_angle(
            [landmarks[11].x * w, landmarks[11].y * h],
            [landmarks[13].x * w, landmarks[13].y * h],
            [landmarks[15].x * w, landmarks[15].y * h]
        )
        
        shoulder_angle = calculate_angle(
            [landmarks[13].x * w, landmarks[13].y * h],
            [landmarks[11].x * w, landmarks[11].y * h],
            [landmarks[23].x * w, landmarks[23].y * h]
        )
        
        hip_angle = calculate_angle(
            [landmarks[11].x * w, landmarks[11].y * h],
            [landmarks[23].x * w, landmarks[23].y * h],
            [landmarks[25].x * w, landmarks[25].y * h]
        )
        
        color = (0, 255, 0) if hip_angle > 160 else (0, 0, 255)
        
        if elbow_angle <= 90 and hip_angle > 160:
            feedback = "Push UP!"
            if self.direction_state == 0:
                self.repetition_count += 0.5
                self.direction_state = 1
        elif elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160:
            feedback = "Go DOWN!"
            if self.direction_state == 1:
                self.repetition_count += 0.5
                self.direction_state = 0
        else:
            feedback = "Keep body straight!"
            
        elbow_pos = [landmarks[13].x * w, landmarks[13].y * h]
        return elbow_angle, feedback, color, elbow_pos

    def check_plank(self, landmarks, w, h):
        """Plank form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        
        angle = calculate_angle(shoulder, hip, ankle)
        color = (0, 255, 0) if 160 <= angle <= 180 else (0, 0, 255)
        
        if 165 <= angle <= 180:
            feedback = 'Perfect plank position!'
        elif angle < 165:
            feedback = 'Lift your hips up!'
        else:
            feedback = 'Lower hips slightly!'
            
        self.prev_angle = angle
        return angle, feedback, color, hip

    def check_chest_fly_machine(self, landmarks, w, h):
        """Chest fly machine form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        color = (0, 255, 0) if 40 <= angle <= 120 else (0, 0, 255)
        
        if angle < 50:
            feedback = 'Great chest squeeze!'
        elif angle > 110:
            feedback = 'Feel the stretch!'
        else:
            feedback = 'Perfect fly motion!'
            
        if self.prev_angle is not None:
            if angle > 110 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 60 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, elbow

    def check_deadlift(self, landmarks, w, h):
        """Deadlift form checking"""
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        
        angle = calculate_angle(hip, knee, ankle)
        color = (0, 255, 0) if 150 <= angle <= 180 else (0, 0, 255)
        
        if angle < 160:
            feedback = 'Drive through heels!'
        elif angle > 175:
            feedback = 'Perfect lockout!'
        else:
            feedback = 'Keep lifting!'
            
        if self.prev_angle is not None:
            if angle > 170 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 160 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, knee

    def check_decline_bench_press(self, landmarks, w, h):
        """Decline bench press form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        color = (0, 255, 0) if 60 <= angle <= 170 else (0, 0, 255)
        
        if angle < 70:
            feedback = 'Lower to chest level!'
        elif angle > 160:
            feedback = 'Perfect decline press!'
        else:
            feedback = 'Press up and back!'
            
        if self.prev_angle is not None:
            if angle > 150 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 80 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, elbow

    def check_hammer_curl(self, landmarks, w, h):
        """Hammer curl form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        color = (0, 255, 0) if 35 <= angle <= 155 else (0, 0, 255)
        
        if angle < 45:
            feedback = 'Perfect hammer contraction!'
        elif angle < 70:
            feedback = 'Keep curling up!'
        elif angle > 145:
            feedback = 'Control the descent!'
        else:
            feedback = 'Solid hammer form!'
            
        if self.prev_angle is not None:
            if angle > 140 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 55 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, elbow

    def check_hip_thrust(self, landmarks, w, h):
        """Hip thrust form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        
        angle = calculate_angle(shoulder, hip, knee)
        color = (0, 255, 0) if 150 <= angle <= 180 else (0, 0, 255)
        
        if angle > 170:
            feedback = 'Perfect hip extension!'
        elif angle < 160:
            feedback = 'Thrust hips higher!'
        else:
            feedback = 'Great glute squeeze!'
            
        if self.prev_angle is not None:
            if angle > 170 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 160 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, hip

    def check_incline_bench_press(self, landmarks, w, h):
        """Incline bench press form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        color = (0, 255, 0) if 75 <= angle <= 175 else (0, 0, 255)
        
        if angle < 85:
            feedback = 'Lower to upper chest!'
        elif angle > 165:
            feedback = 'Perfect incline press!'
        else:
            feedback = 'Press up and forward!'
            
        if self.prev_angle is not None:
            if angle > 155 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 95 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, elbow

    def check_lat_pulldown(self, landmarks, w, h):
        """Lat pulldown form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        color = (0, 255, 0) if 60 <= angle <= 150 else (0, 0, 255)
        
        if angle < 70:
            feedback = 'Pull to upper chest!'
        elif angle > 140:
            feedback = 'Control the stretch!'
        else:
            feedback = 'Squeeze those lats!'
            
        if self.prev_angle is not None:
            if angle > 140 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 80 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, elbow

    def check_lateral_raise(self, landmarks, w, h):
        """Lateral raise form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        # For lateral raise, we check arm elevation from side
        arm_elevation = abs(shoulder[1] - elbow[1])  # Vertical difference
        shoulder_width = abs(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x - 
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) * w
        
        # Check if arms are raised to shoulder level
        is_raised = arm_elevation < shoulder_width * 0.3
        color = (0, 255, 0) if is_raised else (0, 0, 255)
        
        if is_raised:
            feedback = 'Perfect shoulder level!'
        else:
            feedback = 'Raise to shoulder height!'
            
        # Simple rep counting based on arm position
        current_state = 1 if is_raised else 0
        if hasattr(self, 'lateral_prev_state'):
            if self.lateral_prev_state == 0 and current_state == 1:
                self.repetition_count += 1
        self.lateral_prev_state = current_state
        
        angle = arm_elevation  # Use elevation as angle measure
        return angle, feedback, color, elbow

    def check_leg_extension(self, landmarks, w, h):
        """Leg extension form checking"""
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        
        angle = calculate_angle(hip, knee, ankle)
        color = (0, 255, 0) if 70 <= angle <= 170 else (0, 0, 255)
        
        if angle > 160:
            feedback = 'Perfect quad extension!'
        elif angle < 90:
            feedback = 'Extend those quads!'
        else:
            feedback = 'Keep extending!'
            
        if self.prev_angle is not None:
            if angle > 160 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 100 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, knee

    def check_leg_raises(self, landmarks, w, h):
        """Leg raises form checking"""
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        
        # For leg raises, check hip flexion angle
        torso_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        
        angle = calculate_angle(torso_hip, hip, knee)
        color = (0, 255, 0) if 60 <= angle <= 150 else (0, 0, 255)
        
        if angle < 80:
            feedback = 'Great leg lift!'
        elif angle > 140:
            feedback = 'Control the descent!'
        else:
            feedback = 'Lift legs to 90 degrees!'
            
        if self.prev_angle is not None:
            if angle < 90 and self.direction != 'up':
                self.direction = 'up'
                self.repetition_count += 1
            elif angle > 130 and self.direction != 'down':
                self.direction = 'down'
                
        self.prev_angle = angle
        return angle, feedback, color, knee

    def check_pull_up(self, landmarks, w, h):
        """Pull-up form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        color = (0, 255, 0) if 60 <= angle <= 160 else (0, 0, 255)
        
        if angle < 80:
            feedback = 'Pull chin over bar!'
        elif angle > 150:
            feedback = 'Control the descent!'
        else:
            feedback = 'Strong pull-up!'
            
        if self.prev_angle is not None:
            if angle > 150 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 90 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, elbow

    def check_romanian_deadlift(self, landmarks, w, h):
        """Romanian deadlift form checking"""
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        
        angle = calculate_angle(hip, knee, ankle)
        color = (0, 255, 0) if 160 <= angle <= 180 else (0, 0, 255)
        
        if angle < 165:
            feedback = 'Hinge at hips!'
        elif angle > 175:
            feedback = 'Perfect RDL form!'
        else:
            feedback = 'Drive hips forward!'
            
        if self.prev_angle is not None:
            if angle > 170 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 165 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, knee

    def check_russian_twist(self, landmarks, w, h):
        """Russian twist form checking"""
        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        
        # Calculate torso rotation based on shoulder alignment
        shoulder_angle = np.arctan2(right_shoulder[1] - left_shoulder[1], 
                                   right_shoulder[0] - left_shoulder[0]) * 180 / np.pi
        
        angle = abs(shoulder_angle)
        color = (0, 255, 0) if 5 <= angle <= 30 else (0, 0, 255)
        
        if 10 <= angle <= 25:
            feedback = 'Perfect twist!'
        elif angle < 10:
            feedback = 'Twist more!'
        else:
            feedback = 'Control the rotation!'
            
        # Count twists based on direction changes
        if self.prev_angle is not None:
            if abs(angle - self.prev_angle) > 15:
                self.repetition_count += 0.5
                
        self.prev_angle = angle
        return angle, feedback, color, hip

    def check_shoulder_press(self, landmarks, w, h):
        """Shoulder press form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        color = (0, 255, 0) if 70 <= angle <= 160 else (0, 0, 255)
        
        if angle < 90:
            feedback = 'Lower to shoulder level!'
        elif angle > 150:
            feedback = 'Perfect overhead press!'
        else:
            feedback = 'Press overhead!'
            
        if self.prev_angle is not None:
            if angle > 150 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 100 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, elbow

    def check_t_bar_row(self, landmarks, w, h):
        """T-bar row form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        color = (0, 255, 0) if 70 <= angle <= 150 else (0, 0, 255)
        
        if angle < 80:
            feedback = 'Pull to lower chest!'
        elif angle > 140:
            feedback = 'Control the stretch!'
        else:
            feedback = 'Squeeze shoulder blades!'
            
        if self.prev_angle is not None:
            if angle > 140 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 90 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, elbow

    def check_tricep_dips(self, landmarks, w, h):
        """Tricep dips form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        color = (0, 255, 0) if 70 <= angle <= 160 else (0, 0, 255)
        
        if angle < 90:
            feedback = 'Deep dip, now push up!'
        elif angle > 150:
            feedback = 'Perfect tricep extension!'
        else:
            feedback = 'Keep dipping down!'
            
        if self.prev_angle is not None:
            if angle > 150 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 100 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, elbow

    def check_tricep_pushdown(self, landmarks, w, h):
        """Tricep pushdown form checking"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        color = (0, 255, 0) if 20 <= angle <= 120 else (0, 0, 255)
        
        if angle < 30:
            feedback = 'Perfect tricep squeeze!'
        elif angle > 110:
            feedback = 'Control the return!'
        else:
            feedback = 'Push down fully!'
            
        if self.prev_angle is not None:
            if angle > 100 and self.direction != 'up':
                self.direction = 'up'
            elif angle < 40 and self.direction != 'down':
                self.direction = 'down'
                self.repetition_count += 1
                
        self.prev_angle = angle
        return angle, feedback, color, elbow

    def get_exercise_feedback(self, exercise_name, landmarks, w, h):
        """Main method to get feedback for any exercise"""
        exercise_methods = {
            'barbell biceps curl': self.check_barbell_biceps_curl,
            'bench press': self.check_bench_press,
            'chest fly machine': self.check_chest_fly_machine,
            'deadlift': self.check_deadlift,
            'decline bench press': self.check_decline_bench_press,
            'hammer curl': self.check_hammer_curl,
            'hip thrust': self.check_hip_thrust,
            'incline bench press': self.check_incline_bench_press,
            'lat pulldown': self.check_lat_pulldown,
            'lateral raise': self.check_lateral_raise,
            'leg extension': self.check_leg_extension,
            'leg raises': self.check_leg_raises,
            'plank': self.check_plank,
            'push-up': self.check_push_up,
            'pull Up': self.check_pull_up,
            'romanian deadlift': self.check_romanian_deadlift,
            'russian twist': self.check_russian_twist,
            'shoulder press': self.check_shoulder_press,
            'squat': self.check_squat,
            't bar row': self.check_t_bar_row,
            'tricep dips': self.check_tricep_dips,
            'tricep Pushdown': self.check_tricep_pushdown
        }
        if exercise_name in exercise_methods:
            return exercise_methods[exercise_name](landmarks, w, h)
        else:
            return 0, "Unknown exercise", (0, 0, 255), None