import os
import time
import cv2
import numpy as np
import mediapipe as mp

# Define 6 high-impact demonstration signs
ACTIONS = ["Hello", "Thank You", "Yes", "No", "Help", "Stop"]
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
DATA_PATH = "custom_data"

os.makedirs(DATA_PATH, exist_ok=True)
for action in ACTIONS:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def extract_features(results):
    # Pose landmarks: 33 * 4 = 132
    pose = []
    if results.pose_landmarks:
        for r in results.pose_landmarks.landmark:
            pose.extend([r.x, r.y, r.z, r.visibility])
    else:
        pose = [0.0] * 132

    # Left Hand landmarks: 21 * 3 = 63
    lh = []
    if results.left_hand_landmarks:
        for r in results.left_hand_landmarks.landmark:
            lh.extend([r.x, r.y, r.z])
    else:
        lh = [0.0] * 63

    # Right Hand landmarks: 21 * 3 = 63
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
print("  DATA COLLECTION MODULE")
print(f"  Actions: {ACTIONS}")
print("  Position yourself comfortably in front of the camera.")
print("=" * 60 + "\n")

for action in ACTIONS:
    # Wait for user confirmation before each sign class
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"READY TO RECORD: '{action.upper()}'", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, "Press SPACEBAR when ready to start recording this sign", (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow("Data Collection Window", frame)
        if cv2.waitKey(1) & 0xFF == 32:  # Spacebar
            break

    # Record 30 distinct samples of the gesture
    for sequence in range(NO_SEQUENCES):
        frames_list = []
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            keypoints = extract_features(results)
            frames_list.append(keypoints)

            # Visual cues during collection
            cv2.putText(frame, f"Recording '{action}' | Sample {sequence + 1}/{NO_SEQUENCES}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Frame {frame_num + 1}/{SEQUENCE_LENGTH}", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Data Collection Window", frame)
            cv2.waitKey(1)

        # Save the 30x258 sequence
        save_path = os.path.join(DATA_PATH, action, f"{sequence}.npy")
        np.save(save_path, np.array(frames_list))
        time.sleep(0.3)  # Brief pause between samples

cap.release()
cv2.destroyAllWindows()
holistic.close()
print("\nData collection complete for all actions!")