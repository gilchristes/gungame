import cv2
import math
import os
import time
import json
import ssl
import urllib.request
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model setup
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.40,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gun Calibration Wizard")
clock = pygame.time.Clock()

title_font = pygame.font.SysFont("Trebuchet MS", 34, bold=True)
body_font  = pygame.font.SysFont("Trebuchet MS", 22)
val_font   = pygame.font.SysFont("Consolas", 28, bold=True)

def dist_3d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# Step 0: Ready, Step 1: Up, Step 2: Down, Step 3: Finished
step = 0
up_samples = []
down_samples = []
step_start_time = 0
last_timestamp_ms = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if step == 0:
                step = 1
                step_start_time = time.time()
                up_samples.clear()
            elif step == 3:
                running = False

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    current_timestamp_ms = int(time.time() * 1000)
    if current_timestamp_ms <= last_timestamp_ms:
        current_timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = current_timestamp_ms

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect_for_video(mp_image, current_timestamp_ms)

    current_ratio = 0.0
    hand_detected = False

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        hand_scale = max(dist_3d(lm[0], lm[9]), 1e-4)
        current_ratio = dist_3d(lm[4], lm[5]) / hand_scale
        hand_detected = True

        # Collecting Phase 1: Thumb Up (3 seconds)
        if step == 1:
            up_samples.append(current_ratio)
            if time.time() - step_start_time > 3.0:
                step = 2
                step_start_time = time.time()
                down_samples.clear()

        # Collecting Phase 2: Thumb Down (3 seconds)
        elif step == 2:
            down_samples.append(current_ratio)
            if time.time() - step_start_time > 3.0:
                step = 3
                # Compute averages and dynamic margins
                avg_up = sum(up_samples) / max(len(up_samples), 1)
                avg_down = sum(down_samples) / max(len(down_samples), 1)
                
                # Arm threshold: 65% toward the open pose
                # Fire threshold: 35% toward the open pose
                delta = avg_up - avg_down
                calc_arm = avg_down + (delta * 0.65)
                calc_fire = avg_down + (delta * 0.35)

                # Save configuration
                config = {
                    "avg_cocked": round(avg_up, 2),
                    "avg_fired": round(avg_down, 2),
                    "arm_threshold": round(calc_arm, 2),
                    "fire_threshold": round(calc_fire, 2)
                }
                with open("calibration.json", "w") as f:
                    json.dump(config, f, indent=4)
                print(f"[SUCCESS] Saved to calibration.json: {config}")

    # UI Rendering
    screen.fill((25, 27, 34))

    if step == 0:
        t_surf = title_font.render("Hand Calibration Setup", True, (240, 240, 240))
        d_surf = body_font.render("Form a gun gesture with your hand. Press [SPACE] to start.", True, (180, 180, 190))
        screen.blit(t_surf, (WIDTH//2 - t_surf.get_width()//2, 140))
        screen.blit(d_surf, (WIDTH//2 - d_surf.get_width()//2, 220))

    elif step == 1:
        time_left = max(0.0, 3.0 - (time.time() - step_start_time))
        t_surf = title_font.render("STEP 1: RAISE THUMB (COCKED)", True, (255, 215, 0))
        d_surf = body_font.render(f"Hold thumb UP in gun position... {time_left:.1f}s", True, (220, 220, 220))
        screen.blit(t_surf, (WIDTH//2 - t_surf.get_width()//2, 140))
        screen.blit(d_surf, (WIDTH//2 - d_surf.get_width()//2, 220))

    elif step == 2:
        time_left = max(0.0, 3.0 - (time.time() - step_start_time))
        t_surf = title_font.render("STEP 2: SNAP THUMB DOWN (FIRED)", True, (255, 80, 60))
        d_surf = body_font.render(f"Hold thumb DOWN against knuckles... {time_left:.1f}s", True, (220, 220, 220))
        screen.blit(t_surf, (WIDTH//2 - t_surf.get_width()//2, 140))
        screen.blit(d_surf, (WIDTH//2 - d_surf.get_width()//2, 220))

    elif step == 3:
        t_surf = title_font.render("Calibration Complete!", True, (50, 255, 120))
        with open("calibration.json", "r") as f:
            data = json.load(f)
        c_surf = body_font.render(f"Cocked: {data['avg_cocked']} | Fired: {data['avg_fired']}", True, (220, 220, 220))
        res_surf = val_font.render(f"ARM: >{data['arm_threshold']}  |  FIRE: <{data['fire_threshold']}", True, (255, 215, 0))
        d_surf = body_font.render("Saved to calibration.json. Press [SPACE] to exit.", True, (180, 180, 180))
        screen.blit(t_surf, (WIDTH//2 - t_surf.get_width()//2, 100))
        screen.blit(c_surf, (WIDTH//2 - c_surf.get_width()//2, 180))
        screen.blit(res_surf, (WIDTH//2 - res_surf.get_width()//2, 240))
        screen.blit(d_surf, (WIDTH//2 - d_surf.get_width()//2, 330))

    # Live ratio bar
    val_color = (100, 200, 255) if hand_detected else (100, 100, 100)
    rt_surf = val_font.render(f"LIVE THUMB RATIO: {current_ratio:.2f}", True, val_color)
    screen.blit(rt_surf, (WIDTH//2 - rt_surf.get_width()//2, 450))

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()