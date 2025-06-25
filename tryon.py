import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
from datetime import datetime

OVERLAY_FOLDER = "overlays/"
CAPTURE_FOLDER = "captures/"
MAX_FRAMES = 5
SCALE_FACTOR = 1.0
offset_x = 0
offset_y = 0
rotation_angle = 0

if not os.path.exists(CAPTURE_FOLDER):
    os.makedirs(CAPTURE_FOLDER)

overlay_files = sorted([f for f in os.listdir(OVERLAY_FOLDER) if f.endswith(".png")])
overlay_images = [cv2.imread(os.path.join(OVERLAY_FOLDER, f), cv2.IMREAD_UNCHANGED) for f in overlay_files]
overlay_names = [os.path.splitext(f)[0] for f in overlay_files]
current_index = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
eye_buffer = deque(maxlen=MAX_FRAMES)

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def apply_overlay(frame, glasses_img, eye_coords, scale=1.0, offset_x=0, offset_y=0, rotation_angle=0):
    if glasses_img is None or not eye_coords:
        return frame

    (x1, y1), (x2, y2) = eye_coords
    eye_distance = np.linalg.norm(np.array((x2, y2)) - np.array((x1, y1)))
    glasses_width = int(eye_distance * 2.2 * scale)
    orig_h, orig_w = glasses_img.shape[:2]
    aspect_ratio = orig_h / orig_w
    glasses_height = int(glasses_width * aspect_ratio)

    glasses_resized = cv2.resize(glasses_img, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
    glasses_rotated = rotate_image(glasses_resized, -rotation_angle)

    center_x = (x1 + x2) // 2 + offset_x
    center_y = (y1 + y2) // 2 + offset_y
    top_left_x = center_x - glasses_width // 2
    top_left_y = center_y - glasses_height // 2

    for i in range(glasses_rotated.shape[0]):
        for j in range(glasses_rotated.shape[1]):
            y = top_left_y + i
            x = top_left_x + j
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                if glasses_rotated.shape[2] == 4:
                    alpha = glasses_rotated[i, j, 3] / 255.0
                    if alpha > 0:
                        frame[y, x] = (
                            alpha * glasses_rotated[i, j, :3] +
                            (1 - alpha) * frame[y, x]
                        )
    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == ord('n'):
        current_index = (current_index + 1) % len(overlay_images)
    elif key == ord('+') or key == ord('='):
        SCALE_FACTOR += 0.05
    elif key == ord('-'):
        SCALE_FACTOR = max(0.1, SCALE_FACTOR - 0.05)
    elif key == ord('w'):
        offset_y -= 5
    elif key == ord('s'):
        offset_y += 5
    elif key == ord('a'):
        offset_x -= 5
    elif key == ord('d'):
        offset_x += 5
    elif key == ord('r'):
        SCALE_FACTOR = 1.0
        offset_x = 0
        offset_y = 0
    elif key == ord('s'):
        filename = f"{CAPTURE_FOLDER}/tryon_{overlay_names[current_index]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, frame)
        print(f"[✅] Saved: {filename}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    eye_coords = None

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            l = landmarks.landmark[33]
            r = landmarks.landmark[263]
            lx, ly = int(l.x * w), int(l.y * h)
            rx, ry = int(r.x * w), int(r.y * h)
            eye_coords = ((lx, ly), (rx, ry))
            eye_buffer.append(eye_coords)
            rotation_angle = np.degrees(np.arctan2(ry - ly, rx - lx))

    if eye_buffer:
        avg_l = tuple(np.mean([pt[0] for pt in eye_buffer], axis=0).astype(int))
        avg_r = tuple(np.mean([pt[1] for pt in eye_buffer], axis=0).astype(int))
        frame = apply_overlay(
            frame,
            overlay_images[current_index],
            (avg_l, avg_r),
            scale=SCALE_FACTOR,
            offset_x=offset_x,
            offset_y=offset_y,
            rotation_angle=rotation_angle
        )

    cv2.putText(frame, f"Style: {overlay_names[current_index]} ({current_index + 1}/{len(overlay_images)})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Zoom: {int(SCALE_FACTOR * 100)}%", (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 200, 200), 1)
    cv2.putText(frame, "N: Next | +/-: Resize | WASD: Move | R: Reset | S: Save | ESC: Exit", (20, 100),
                cv2.FONT_HERSHEY_PLAIN, 1.2, (180, 180, 180), 1)

    cv2.imshow("Lenskart Virtual Try-On", frame)

cap.release()
cv2.destroyAllWindows()
