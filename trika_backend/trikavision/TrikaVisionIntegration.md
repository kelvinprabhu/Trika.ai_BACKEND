# WebSocket Workout Analyzer Integration Guide

## Overview
This integration combines your existing MediaPipe pose detection with the advanced workout analyzer model for real-time exercise classification, form checking, and rep counting through WebSocket connections.

## Key Features Added
- ✅ Real-time exercise classification (22 different exercises)
- ✅ Form analysis and feedback
- ✅ Repetition counting
- ✅ Performance analytics
- ✅ Session management
- ✅ Comprehensive logging
- ✅ Error handling and recovery

## Setup Instructions

### 1. Update Your Project Structure
```
your_project/
├── workoutAnalyser.py          # Updated analyzer
├── utils.py                    # WebSocket utilities  
├── visionConsumers.py         # Updated WebSocket consumer
├── exerciseformchecker.py     # Your existing form checker
└── requirements.txt           # Updated dependencies
```

### 2. Update Dependencies
Add these to your `requirements.txt`:
```txt
tensorflow>=2.8.0
mediapipe>=0.8.0
opencv-python>=4.5.0
numpy>=1.21.0
channels>=3.0.0
asgiref>=3.4.0
```

### 3. Model Setup
```python
# Update the model path in visionConsumers.py (line 20)
model_path = "/path/to/your/22_class_model_30fps.h5"

# Or set it as an environment variable
import os
model_path = os.getenv('WORKOUT_MODEL_PATH', '/default/path/model.h5')
```

### 4. Logging Configuration
Add to your Django `settings.py`:
```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'workout_analyzer.log',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'your_app_name': {  # Replace with your app name
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
```

## WebSocket Message Protocol

### Client to Server Messages

#### 1. Video Frame
```json
{
    "type": "frame",
    "payload": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
}
```

#### 2. Reset Session
```json
{
    "type": "reset"
}
```

#### 3. Get Session Summary
```json
{
    "type": "get_summary"
}
```

### Server to Client Messages

#### 1. Workout Analysis Result
```json
{
    "type": "workout",
    "payload": {
        "exercise": "push-up",
        "confidence": 0.89,
        "feedback": "Great form! Keep your back straight",
        "angle": 85,
        "reps": 12,
        "landmarks_detected": true,
        "form_color": [0, 255, 0],
        "session_time": 45.2,
        "frame_count": 1354
    }
}
```

#### 2. System Messages
```json
{
    "type": "system",
    "payload": {
        "status": "initialized",
        "message": "Workout analyzer ready",
        "model_loaded": true,
        "supported_exercises": 22
    }
}
```

#### 3. Error Messages
```json
{
    "type": "error",
    "payload": {
        "message": "Frame processing error: Invalid frame format",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

#### 4. Session Summary
```json
{
    "type": "summary",
    "payload": {
        "session_duration": 180.5,
        "total_frames": 5415,
        "exercises_performed": 3,
        "exercise_details": {
            "push-up": {
                "total_reps": 20,
                "duration": 60.2,
                "best_form_score": 95
            },
            "squat": {
                "total_reps": 15,
                "duration": 45.8,
                "best_form_score": 88
            }
        }
    }
}
```

## Frontend JavaScript Integration Example

```javascript
class WorkoutAnalyzer {
    constructor(websocketUrl) {
        this.ws = new WebSocket(websocketUrl);
        this.setupEventHandlers();
    }
    
    setupEventHandlers() {
        this.ws.onopen = () => {
            console.log('✅ Connected to workout analyzer');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
        };
        
        this.ws.onclose = () => {
            console.log('❌ WebSocket connection closed');
        };
    }
    
    sendFrame(canvas) {
        const frameData = canvas.toDataURL('image/jpeg', 0.8);
        this.ws.send(JSON.stringify({
            type: 'frame',
            payload: frameData
        }));
    }
    
    handleMessage(data) {
        switch(data.type) {
            case 'workout':
                this.updateWorkoutUI(data.payload);
                break;
            case 'system':
                this.handleSystemMessage(data.payload);
                break;
            case 'error':
                this.handleError(data.payload);
                break;
            case 'summary':
                this.showSessionSummary(data.payload);
                break;
        }
    }
    
    updateWorkoutUI(workout) {
        document.getElementById('exercise').textContent = workout.exercise || 'Detecting...';
        document.getElementById('reps').textContent = workout.reps;
        document.getElementById('feedback').textContent = workout.feedback;
        document.getElementById('confidence').textContent = (workout.confidence * 100).toFixed(1) + '%';
        
        // Update form indicator color
        const indicator = document.getElementById('form-indicator');
        const color = workout.form_color;
        indicator.style.backgroundColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    }
    
    resetSession() {
        this.ws.send(JSON.stringify({type: 'reset'}));
    }
    
    getSummary() {
        this.ws.send(JSON.stringify({type: 'get_summary'}));
    }
}

// Usage
const analyzer = new WorkoutAnalyzer('ws://localhost:8000/ws/workout/');

// In your camera capture loop
function processVideoFrame() {
    // Draw camera feed to canvas
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    // Send frame to analyzer (throttle to ~10 FPS for better performance)
    if (frameCount % 3 === 0) {
        analyzer.sendFrame(canvas);
    }
    frameCount++;
}
```

## Performance Optimization Tips

### 1. Frame Processing
- The system processes every 2nd frame by default for better performance
- Adjust `process_every_n_frames` in WorkoutAnalyzer if needed
- Consider reducing frame quality for faster processing

### 2. Model Optimization
- Ensure your model is optimized for inference
- Consider using TensorFlow Lite for mobile deployment
- GPU acceleration recommended for heavy loads

### 3. WebSocket Optimization
- Implement frame buffering on client side
- Add compression for base64 frames
- Consider using WebRTC for lower latency

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```
   ❌ Could not load model from /path/to/model.h5
   ```
   - Check model file path and permissions
   - Verify TensorFlow version compatibility
   - Ensure model file is not corrupted

2. **MediaPipe Initialization Error**
   - Update MediaPipe to latest version
   - Check camera permissions
   - Verify OpenCV installation

3. **High Memory Usage**
   - Reduce frame buffer size (MAX_FRAMES)
   - Clear buffers more frequently
   - Monitor frame processing rate

4. **Low Prediction Accuracy**
   - Ensure good lighting conditions
   - Check camera angle and distance
   - Verify model is trained for your use case

### Logging Analysis
Monitor these log patterns:
- `✅ WorkoutAnalyzer initialized successfully` - System ready
- `🏃 Exercise detected: push-up (confidence: 0.89)` - Exercise changes
- `📊 Processed 1800 frames. Current exercise: squat` - Progress updates
- `❌ Error processing frame` - Processing errors

## Advanced Features

### Custom Exercise Addition
To add new exercises:
1. Retrain model with new exercise data
2. Update `CLASS_NAMES` in WorkoutAnalyzer
3. Add form checking rules in ExerciseFormChecker
4. Update frontend exercise list

### Form Scoring Enhancement
```python
def enhanced_form_scoring(self, landmarks, exercise):
    """Enhanced form scoring with multiple criteria"""
    scores = {
        'alignment': self.check_body_alignment(landmarks),
        'range_of_motion': self.check_rom(landmarks, exercise),
        'tempo': self.check_movement_tempo(),
        'stability': self.check_stability(landmarks)
    }
    return sum(scores.values()) / len(scores)
```

### Real-time Coaching
Implement progressive coaching:
```python
def get_coaching_tips(self, exercise, performance_history):
    """Provide personalized coaching based on performance"""
    if len(performance_history) < 5:
        return "Focus on form over speed"
    
    recent_scores = performance_history[-5:]
    avg_score = sum(recent_scores) / len(recent_scores)
    
    if avg_score < 60:
        return f"Work on {exercise} fundamentals"
    elif avg_score < 80:
        return "Good progress! Focus on consistency"
    else:
        return "Excellent form! Try increasing reps"
```

This integration provides a robust foundation for real-time workout analysis through WebSockets. The system is designed to be scalable, maintainable, and extensible for future enhancements.