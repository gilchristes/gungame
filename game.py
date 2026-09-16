import cv2
import math
import os
import random
import time
import ssl
import json
import urllib.request
from collections import deque
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------------------------------------------------
# 1. Vision Initialization & Profile Loading
# -------------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading hand landmarker model...")
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

# Default ratios derived directly from your camera telemetry
TRIGGER_RATIO_ARM = 1.45
TRIGGER_RATIO_FIRE = 1.15

# Automatically load calibration if present
if os.path.exists("calibration.json"):
    try:
        with open("calibration.json", "r") as f:
            cfg = json.load(f)
            TRIGGER_RATIO_ARM = cfg.get("arm_threshold", TRIGGER_RATIO_ARM)
            TRIGGER_RATIO_FIRE = cfg.get("fire_threshold", TRIGGER_RATIO_FIRE)
            print(f"[CALIBRATION LOADED] ARM: {TRIGGER_RATIO_ARM:.2f} | FIRE: {TRIGGER_RATIO_FIRE:.2f}")
    except Exception as e:
        print(f"[WARN] Failed loading calibration.json: {e}")

SMOOTHING_ALPHA = 0.65
smooth_x, smooth_y = None, None
is_armed = False
last_timestamp_ms = 0
hand_lost_frames = 0
current_thumb_ratio = 0.0

aim_history = deque(maxlen=6)

def dist_3d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# -------------------------------------------------------------------------
# 2. Pygame Setup & Entities
# -------------------------------------------------------------------------
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Bottle Range - Hand Light Gun")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Trebuchet MS", 24, bold=True)
hud_font = pygame.font.SysFont("Trebuchet MS", 36, bold=True)
debug_font = pygame.font.SysFont("Consolas", 18, bold=True)

COLOR_BG = (28, 30, 38)
COLOR_SHELF = (60, 42, 33)
COLOR_SHELF_TOP = (110, 75, 55)
COLOR_CROSSHAIR = (50, 255, 120)
COLOR_RECOIL = (255, 60, 60)

class GlassShard:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-6, 6)
        self.vy = random.uniform(-10, -2)
        self.gravity = 0.45
        self.size = random.randint(4, 9)
        self.color = color
        self.alpha = 255

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.alpha = max(0, self.alpha - 6)

    def draw(self, surface):
        if self.alpha > 0:
            shard_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.polygon(shard_surf, (*self.color, self.alpha), [
                (self.size, 0),
                (self.size * 2, self.size),
                (0, self.size * 2)
            ])
            surface.blit(shard_surf, (self.x, self.y))

class Bottle:
    def __init__(self, x, y, color=(40, 160, 90)):
        self.width = 46
        self.height = 130
        self.x = x
        self.y = y
        self.color = color
        self.rect = pygame.Rect(x - self.width // 2, y - self.height, self.width, self.height)
        self.is_alive = True
        self.respawn_timer = 0.0

    def hit(self, particles_list):
        if not self.is_alive:
            return False
        self.is_alive = False
        self.respawn_timer = time.time() + 2.5
        for _ in range(25):
            px = random.randint(self.rect.left, self.rect.right)
            py = random.randint(self.rect.top, self.rect.bottom)
            particles_list.append(GlassShard(px, py, self.color))
        return True

    def update(self):
        if not self.is_alive and time.time() >= self.respawn_timer:
            self.is_alive = True

    def draw(self, surface):
        if not self.is_alive:
            return

        body_rect = pygame.Rect(self.x - self.width // 2, self.y - 85, self.width, 85)
        pygame.draw.rect(surface, self.color, body_rect, border_radius=6)
        neck_rect = pygame.Rect(self.x - 8, self.y - 130, 16, 45)
        pygame.draw.rect(surface, self.color, neck_rect, border_radius=3)
        cork_rect = pygame.Rect(self.x - 10, self.y - 136, 20, 8)
        pygame.draw.rect(surface, (180, 140, 90), cork_rect, border_radius=2)
        pygame.draw.line(surface, (255, 255, 255, 120), (self.x - 14, self.y - 80), (self.x - 14, self.y - 10), 3)

SHELF_Y = 560
bottles = [
    Bottle(x=220, y=SHELF_Y, color=(35, 145, 90)),
    Bottle(x=430, y=SHELF_Y, color=(160, 60, 45)),
    Bottle(x=640, y=SHELF_Y, color=(45, 100, 180)),
    Bottle(x=850, y=SHELF_Y, color=(180, 130, 35)),
    Bottle(x=1060, y=SHELF_Y, color=(120, 60, 150))
]
particles = []
score = 0
total_shots = 0
screen_shake = 0
last_shot_time = 0.0

# -------------------------------------------------------------------------
# 3. Main Loop
# -------------------------------------------------------------------------
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
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

    crosshair_pos = None
    trigger_pulled = False
    gesture_status = "SEARCHING HAND"

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        hand_scale = max(dist_3d(lm[0], lm[9]), 1e-4)

        # Tolerant checks to prevent dropouts during thumb flick
        index_pointing = (dist_3d(lm[8], lm[5]) / hand_scale > 0.50) or ((lm[5].z - lm[8].z) > 0.03)
        middle_curled  = (dist_3d(lm[12], lm[9]) / hand_scale) < 0.92
        is_gun_pose = index_pointing and middle_curled

        if is_gun_pose:
            hand_lost_frames = 0
            target_x = lm[8].x * SCREEN_WIDTH
            target_y = lm[8].y * SCREEN_HEIGHT

            if smooth_x is None:
                smooth_x, smooth_y = target_x, target_y
            else:
                smooth_x = SMOOTHING_ALPHA * target_x + (1 - SMOOTHING_ALPHA) * smooth_x
                smooth_y = SMOOTHING_ALPHA * target_y + (1 - SMOOTHING_ALPHA) * smooth_y

            crosshair_pos = (int(smooth_x), int(smooth_y))
            aim_history.append(crosshair_pos)

            current_thumb_ratio = dist_3d(lm[4], lm[5]) / hand_scale

            # Trigger FSM with updated boundaries
            if current_thumb_ratio >= TRIGGER_RATIO_ARM:
                is_armed = True
                gesture_status = "ARMED (READY TO FIRE)"
            elif is_armed and current_thumb_ratio <= TRIGGER_RATIO_FIRE:
                trigger_pulled = True
                is_armed = False
                last_shot_time = time.time()
                gesture_status = "BANG! FIRED!"
            elif is_armed:
                gesture_status = "ARMED (SNAP THUMB DOWN)"
            else:
                gesture_status = "COCK HAMMER (RAISE THUMB)"
        else:
            gesture_status = "GUN GESTURE REQUIRED"
            hand_lost_frames += 1
            if hand_lost_frames > 8:
                smooth_x, smooth_y = None, None
                aim_history.clear()
            elif smooth_x is not None:
                crosshair_pos = (int(smooth_x), int(smooth_y))
    else:
        hand_lost_frames += 1
        if hand_lost_frames > 8:
            smooth_x, smooth_y = None, None
            aim_history.clear()
        elif smooth_x is not None:
            crosshair_pos = (int(smooth_x), int(smooth_y))

    # Collision Check
    if trigger_pulled and (crosshair_pos or len(aim_history) > 0):
        total_shots += 1
        screen_shake = 12
        hit_any = False
        
        shot_target = aim_history[0] if len(aim_history) >= 4 else (crosshair_pos or aim_history[-1])

        for bottle in bottles:
            expanded_rect = bottle.rect.inflate(44, 20)
            if bottle.is_alive and expanded_rect.collidepoint(shot_target):
                if bottle.hit(particles):
                    score += 100
                    hit_any = True
                    break

    # Entity Updates
    for bottle in bottles:
        bottle.update()

    for p in particles[:]:
        p.update()
        if p.alpha <= 0:
            particles.remove(p)

    # Render
    shake_x = random.randint(-screen_shake, screen_shake) if screen_shake > 0 else 0
    shake_y = random.randint(-screen_shake, screen_shake) if screen_shake > 0 else 0
    screen_shake = max(0, screen_shake - 2)

    screen.fill(COLOR_BG)

    shelf_rect = pygame.Rect(0 + shake_x, SHELF_Y + shake_y, SCREEN_WIDTH, 40)
    shelf_lip = pygame.Rect(0 + shake_x, SHELF_Y - 6 + shake_y, SCREEN_WIDTH, 10)
    pygame.draw.rect(screen, COLOR_SHELF, shelf_rect)
    pygame.draw.rect(screen, COLOR_SHELF_TOP, shelf_lip)

    for bottle in bottles:
        bottle.draw(screen)

    for p in particles:
        p.draw(screen)

    if crosshair_pos:
        cx = crosshair_pos[0] + shake_x
        cy = crosshair_pos[1] + shake_y
        is_firing = (time.time() - last_shot_time < 0.12)
        reticle_color = COLOR_RECOIL if is_firing else COLOR_CROSSHAIR
        r = 28 if is_firing else 18

        pygame.draw.circle(screen, reticle_color, (cx, cy), r, 2)
        pygame.draw.circle(screen, reticle_color, (cx, cy), 3)
        pygame.draw.line(screen, reticle_color, (cx - r - 8, cy), (cx + r + 8, cy), 2)
        pygame.draw.line(screen, reticle_color, (cx, cy - r - 8), (cx, cy + r + 8), 2)

    accuracy = int((score / (total_shots * 100)) * 100) if total_shots > 0 else 100
    score_surf = hud_font.render(f"SCORE: {score}", True, (240, 240, 240))
    shots_surf = font.render(f"SHOTS: {total_shots}  |  ACCURACY: {accuracy}%", True, (170, 180, 190))
    
    status_color = (255, 60, 60) if "BANG" in gesture_status else ((255, 215, 0) if is_armed else (180, 180, 180))
    status_surf = font.render(f"STATUS: {gesture_status}", True, status_color)

    screen.blit(score_surf, (30, 20))
    screen.blit(shots_surf, (30, 65))
    screen.blit(status_surf, (SCREEN_WIDTH - status_surf.get_width() - 30, 20))

    # Live Meter
    meter_x, meter_y, meter_w, meter_h = SCREEN_WIDTH - 260, 65, 230, 14
    pygame.draw.rect(screen, (50, 50, 60), (meter_x, meter_y, meter_w, meter_h), border_radius=3)
    fill_ratio = max(0.0, min(current_thumb_ratio / 2.50, 1.0))
    bar_color = (255, 215, 0) if is_armed else (100, 200, 255)
    pygame.draw.rect(screen, bar_color, (meter_x, meter_y, int(meter_w * fill_ratio), meter_h), border_radius=3)
    
    debug_text = f"THUMB: {current_thumb_ratio:.2f} (ARM >{TRIGGER_RATIO_ARM:.2f} | FIRE <{TRIGGER_RATIO_FIRE:.2f})"
    debug_surf = debug_font.render(debug_text, True, (200, 200, 200))
    screen.blit(debug_surf, (SCREEN_WIDTH - debug_surf.get_width() - 30, 88))

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()