import cv2
import math
import os
import time
import ssl
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. Model Setup ---
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading hand landmarker model (~8MB)...")
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Model ready.")

# Use VIDEO mode for temporal continuity & lower detection threshold
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.45,
    min_hand_presence_confidence=0.45,
    min_tracking_confidence=0.45,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# --- 2. Camera Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- 3. Tuning & State ---
SMOOTHING_ALPHA = 0.65
TRIGGER_RATIO_FIRE = 0.45
TRIGGER_RATIO_ARM  = 0.65

smooth_x, smooth_y = None, None
is_armed = False
last_fire_time = 0.0
last_timestamp_ms = 0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

def dist_3d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Monotonic millisecond timestamp required for RunningMode.VIDEO
    current_timestamp_ms = int(time.time() * 1000)
    if current_timestamp_ms <= last_timestamp_ms:
        current_timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = current_timestamp_ms

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect_for_video(mp_image, current_timestamp_ms)

    crosshair_pos = None
    action_text = "SEARCHING HAND"

    if detection_result.hand_landmarks:
        lm = detection_result.hand_landmarks[0]

        # Draw skeleton
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = (int(lm[start_idx].x * w), int(lm[start_idx].y * h))
            pt2 = (int(lm[end_idx].x * w), int(lm[end_idx].y * h))
            cv2.line(frame, pt1, pt2, (0, 220, 0), 2)
        for pt in lm:
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, (0, 0, 255), -1)

        hand_scale = max(dist_3d(lm[0], lm[9]), 1e-4)

        # Simplified 3-Finger Rule:
        # 1. Index extended forward or straight out
        index_pointing = (dist_3d(lm[8], lm[5]) / hand_scale > 0.60) or ((lm[5].z - lm[8].z) > 0.05)
        # 2. Middle finger curled (differentiates gun from open hand/peace sign)
        middle_curled = (dist_3d(lm[12], lm[9]) / hand_scale) < 0.78

        is_gun_pose = index_pointing and middle_curled

        if is_gun_pose:
            raw_x, raw_y = lm[8].x * w, lm[8].y * h

            if smooth_x is None:
                smooth_x, smooth_y = raw_x, raw_y
            else:
                smooth_x = SMOOTHING_ALPHA * raw_x + (1 - SMOOTHING_ALPHA) * smooth_x
                smooth_y = SMOOTHING_ALPHA * raw_y + (1 - SMOOTHING_ALPHA) * smooth_y

            crosshair_pos = (int(smooth_x), int(smooth_y))

            # 3. Thumb hammer trigger check
            thumb_ratio = dist_3d(lm[4], lm[5]) / hand_scale

            if thumb_ratio > TRIGGER_RATIO_ARM:
                is_armed = True
                action_text = "ARMED (READY TO FIRE)"
            elif is_armed and thumb_ratio < TRIGGER_RATIO_FIRE:
                action_text = "BANG! FIRED!"
                is_armed = False
                last_fire_time = time.time()
                print(f"[SHOT REGISTERED] Target: {crosshair_pos}")
            elif not is_armed:
                action_text = "COCK THE HAMMER (RAISE THUMB)"
        else:
            action_text = "FORM GUN GESTURE"
            smooth_x, smooth_y = None, None

    if crosshair_pos:
        cx, cy = crosshair_pos
        is_recoil = (time.time() - last_fire_time < 0.15)
        color = (0, 0, 255) if is_recoil else (0, 255, 0)
        radius = 26 if is_recoil else 18

        cv2.circle(frame, (cx, cy), radius, color, 2)
        cv2.line(frame, (cx - radius - 8, cy), (cx + radius + 8, cy), color, 2)
        cv2.line(frame, (cx, cy - radius - 8), (cx, cy + radius + 8), color, 2)

    cv2.putText(frame, action_text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow("Hand Light-Gun Prototype", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()