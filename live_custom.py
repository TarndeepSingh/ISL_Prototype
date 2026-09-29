import os
import json
import collections
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

MODEL_PATH = "custom_model.h5"
LABELS_PATH = "custom_labels.json"
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.70
REQUIRED_CONSECUTIVE = 2

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

frame_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
history = collections.deque(maxlen=REQUIRED_CONSECUTIVE)
current_word = "Listening for signs..."
current_conf = 0.0
top_3_predictions = []

def extract_features(results):
    pose = []
    if results.pose_landmarks:
        for r in results.pose_landmarks.landmark:
            pose.extend([r.x, r.y, r.z, r.visibility])
    else:
        pose = [0.0] * 132

    lh = []
    if results.left_hand_landmarks:
        for r in results.left_hand_landmarks.landmark:
            lh.extend([r.x, r.y, r.z])
    else:
        lh = [0.0] * 63

    rh = []
    if results.right_hand_landmarks:
        for r in results.right_hand_landmarks.landmark:
            rh.extend([r.x, r.y, r.z])
    else:
        rh = [0.0] * 63

    return np.concatenate([pose, lh, rh])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n" + "=" * 60)
print("  LIVE TRANSLATION RUNNING (CALIBRATED SYSTEM)")
print("  Press 'q' to exit.")
print("=" * 60 + "\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    keypoints = extract_features(results)
    frame_buffer.append(keypoints)

    if len(frame_buffer) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(frame_buffer, axis=0)
        probs = model.predict(input_data, verbose=0)[0]

        top_indices = np.argsort(probs)[::-1][:3]
        top_3_predictions = [(label_map[str(i)], float(probs[i])) for i in top_indices]

        best_word, best_conf = top_3_predictions[0]

        if best_conf >= CONFIDENCE_THRESHOLD:
            history.append(best_word)
            if len(history) == REQUIRED_CONSECUTIVE and len(set(history)) == 1:
                current_word = best_word
                current_conf = best_conf
                frame_buffer.clear()
                history.clear()
        else:
            history.append("Idle")

    # HUD Overlay
    cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)
    cv2.putText(frame, "INDIAN SIGN LANGUAGE TRANSLATION ENGINE", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(frame, f"Sign: {current_word} ({current_conf:.1%})", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 127), 2, cv2.LINE_AA)

    # Top-3 Predictions Sidebar Card
    card_x1, card_y1, card_x2, card_y2 = w - 340, 100, w - 20, 255
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

    # Buffer indicator
    buf_percent = len(frame_buffer) / SEQUENCE_LENGTH
    cv2.rectangle(frame, (30, h - 35), (230, h - 20), (50, 50, 50), -1)
    cv2.rectangle(frame, (30, h - 35), (30 + int(200 * buf_percent), h - 20), (255, 165, 0), -1)
    cv2.putText(frame, f"Temporal Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH}", (30, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    cv2.imshow("ISL Real-Time Translation System - Live Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()