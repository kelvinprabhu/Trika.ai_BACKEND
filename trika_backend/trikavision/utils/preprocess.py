import numpy as np

def flatten_landmarks(landmarks):
    return np.array([v for lm in landmarks for v in [lm["x"], lm["y"], lm["z"]]])

def normalize_landmarks(flat):
    return flat / np.max(flat) if np.max(flat) > 0 else flat


def pad_to_shape(arr, target_shape):
    padded = np.zeros(target_shape, dtype=np.float32)
    padded[0, :len(arr)] = arr[:target_shape[1]]
    return padded