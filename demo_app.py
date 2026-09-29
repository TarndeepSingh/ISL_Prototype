import os
import json
import collections
import cv2
import numpy as np
import mediapipe as mp

# Try LiteRT first, fallback to standard TFLite
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# =======================================================
# CONFIGURATION & CALIBRATION TOGGLES
# =======================================================
MODEL_PATH = "isl_production_model.tflite"
LABELS_PATH = "labels.json"
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.50  # Lowered slightly for live testing
REQUIRED_CONSECUTIVE = 2

# TOGGLES: Change these if predictions are erratic
NORMALIZE_RELATIVE_TO_NOSE = False  # Set to True if model was trained with nose-centering
MIRROR_CAMERA = False               # False keeps natural orientation for MediaPipe hand tracking

# =======================================================
# ASSET VERIFICATION & LOADING
# =======================================================
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("Ensure isl_production_model.tflite and labels.json are in this folder.")

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

expected_shape = input_details[0]['shape']
feature_dim = expected_shape[2]  # Should be 258

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Real-time state buffers
frame_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
prediction_history = collections.deque(maxlen=REQUIRED_CONSECUTIVE)
current_word = "Listening..."
current_conf = 0.0
top_3_predictions = []

def extract_features(results, normalize_to_nose=False):
    """
    Extracts 258 features:
    - 33 Pose landmarks (x, y, z, visibility) = 132 features
    - 21 Left Hand landmarks (x, y, z) = 63 features
    - 21 Right Hand landmarks (x, y, z) = 63 features
    """
    if normalize_to_nose and results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        nx, ny, nz = nose.x, nose.y, nose.z
    else:
        nx, ny, nz = 0.0, 0.0, 0.0

    # 1. Pose landmarks (132 features)
    pose = []
    if results.pose_landmarks:
        for r in results.pose_landmarks.landmark:
            pose.extend([r.x - nx, r.y - ny, r.z - nz, r.visibility])
    else:
        pose = [0.0] * 132

    # 2. Left Hand landmarks (63 features)
    lh = []
    if results.left_hand_landmarks:
        for r in results.left_hand_landmarks.landmark:
            lh.extend([r.x - nx, r.y - ny, r.z - nz])
    else:
        lh = [0.0] * 63

    # 3. Right Hand landmarks (63 features)
    rh = []
    if results.right_hand_landmarks:
        for r in results.right_hand_landmarks.landmark:
            rh.extend([r.x - nx, r.y - ny, r.z - nz])
    else:
        rh = [0.0] * 63

    return np.concatenate([pose, lh, rh])

# =======================================================
# CAMERA LOOP
# =======================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n" + "=" * 60)
print("  ISL TRANSLATOR RUNNING (DEBUG MODE ACTIVE)")
print(f"  Nose Centering: {NORMALIZE_RELATIVE_TO_NOSE} | Mirror Camera: {MIRROR_CAMERA}")
print("  Watch the terminal output for real-time probabilities.")
print("  Press 'q' in the camera window to exit.")
print("=" * 60 + "\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if MIRROR_CAMERA:
        frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    # Render skeletal connections
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Extract 258 features
    keypoints = extract_features(results, normalize_to_nose=NORMALIZE_RELATIVE_TO_NOSE)
    frame_buffer.append(keypoints)

    # Run inference when buffer fills to 30 frames
    if len(frame_buffer) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(frame_buffer, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]['index'])[0]

        top_indices = np.argsort(probs)[::-1][:3]
        top_3_predictions = [
            (label_map.get(str(i), f"Class {i}"), float(probs[i]))
            for i in top_indices
        ]

        best_word, best_conf = top_3_predictions[0]

        # Terminal debug printout
        print(f"[Inference] Top-1: {best_word} ({best_conf:.1%}) | Top-2: {top_3_predictions[1][0]} ({top_3_predictions[1][1]:.1%})")

        if best_conf >= CONFIDENCE_THRESHOLD:
            prediction_history.append(best_word)
            if len(prediction_history) == REQUIRED_CONSECUTIVE and len(set(prediction_history)) == 1:
                current_word = best_word
                current_conf = best_conf
                frame_buffer.clear()
                prediction_history.clear()
        else:
            prediction_history.append("Idle")

    # =======================================================
    # HEADS-UP DISPLAY (HUD)
    # =======================================================
    cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)
    cv2.putText(frame, "INDIAN SIGN LANGUAGE TRANSLATION SYSTEM", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(frame, f"Detected: {current_word} ({current_conf:.1%})", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 127), 2, cv2.LINE_AA)

    # Top-3 Probabilities card
    card_x1, card_y1, card_x2, card_y2 = w - 350, 100, w - 20, 255
    cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (30, 30, 30), -1)
    cv2.putText(frame, "Top-3 Probabilities:", (card_x1 + 15, card_y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    for idx, (lbl, conf) in enumerate(top_3_predictions):
        y_pos = card_y1 + 60 + (idx * 38)
        clean_lbl = (lbl[:17] + '..') if len(lbl) > 17 else lbl
        cv2.putText(frame, f"{idx+1}. {clean_lbl}", (card_x1 + 15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        bar_len = int(100 * conf)
        cv2.rectangle(frame, (card_x2 - 120, y_pos - 12), (card_x2 - 20, y_pos - 2), (60, 60, 60), 1)
        cv2.rectangle(frame, (card_x2 - 120, y_pos - 12), (card_x2 - 120 + bar_len, y_pos - 2), (0, 255, 127), -1)

    # Temporal buffer fill bar
    buf_percent = len(frame_buffer) / SEQUENCE_LENGTH
    cv2.rectangle(frame, (30, h - 35), (230, h - 20), (50, 50, 50), -1)
    cv2.rectangle(frame, (30, h - 35), (30 + int(200 * buf_percent), h - 20), (255, 165, 0), -1)
    cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH}", (30, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    cv2.imshow("ISL Real-Time Translation Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()