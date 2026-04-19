import cv2
import mediapipe as mp
import socket
import json
import numpy as np
from collections import deque

# -------------------------
# UDP Setup
# -------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
SERVER_ADDRESS = ("127.0.0.1", 5052)

# -------------------------
# Smoothing buffer
# -------------------------
SMOOTH_FRAMES = 3
landmark_buffer = deque(maxlen=SMOOTH_FRAMES)

# -------------------------
# MediaPipe Setup
# -------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # slightly faster, still stable
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -------------------------
# Camera Setup
# -------------------------
cap = cv2.VideoCapture(0)

# -------------------------
# Smoothing function
# -------------------------
def smooth_landmarks(raw):
    landmark_buffer.append(raw)

    if len(landmark_buffer) < 2:
        return raw

    smoothed = []
    for i in range(len(raw)):
        avg_x = np.mean([frame[i]["x"] for frame in landmark_buffer])
        avg_y = np.mean([frame[i]["y"] for frame in landmark_buffer])
        avg_z = np.mean([frame[i]["z"] for frame in landmark_buffer])
        avg_v = np.mean([frame[i]["v"] for frame in landmark_buffer])

        smoothed.append({
            "x": float(avg_x),
            "y": float(avg_y),
            "z": float(avg_z),
            "v": float(avg_v)
        })

    return smoothed

print("=== MediaPipe → Unity Pose Sender ===")
print("Press ESC to exit")

# -------------------------
# Main loop
# -------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # -------------------------
        # Extract landmarks (NO flipping here)
        # -------------------------
        raw_landmarks = []
        for lm in results.pose_landmarks.landmark:
            raw_landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "v": lm.visibility
            })

        # Smooth
        smoothed = smooth_landmarks(raw_landmarks)

        # Send
        sock.sendto(
            json.dumps(smoothed).encode(),
            SERVER_ADDRESS
        )

    else:
        cv2.putText(frame, "No pose detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Pose Sender", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
sock.close()
cv2.destroyAllWindows()