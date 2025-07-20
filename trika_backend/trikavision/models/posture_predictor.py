import numpy as np
import tensorflow as tf
from trikavision.models.class_names import CLASS_NAMES
from trikavision.utils.preprocess import flatten_landmarks, normalize_landmarks, pad_to_shape

# Load DL model
model = tf.keras.models.load_model("22_class_model.h5")

EXPECTED_SHAPE = (1, 100, 36)  # 100 frames, 12 landmarks * 3 (x,y,z)

# Buffer for 100-frame input
frame_buffer = []


def predict_from_landmark_frame(landmarks):
    global frame_buffer

    flat = flatten_landmarks(landmarks)
    norm = normalize_landmarks(flat)
    frame_buffer.append(norm)

    if len(frame_buffer) < 100:
        return "Buffering...", 0.0

    input_tensor = np.array(frame_buffer[-100:]).reshape(EXPECTED_SHAPE)
    predictions = model.predict(input_tensor)
    idx = np.argmax(predictions)
    confidence = float(np.max(predictions))

    return CLASS_NAMES[idx], confidence
