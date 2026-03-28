"""
Synthetic Test Video Generator
Creates a realistic-looking dashcam test video for pipeline validation.
The video simulates:
 - A road scene with moving objects
 - A sudden motion spike (simulated collision at frame 150)
 - Pre and post-incident frames

Run: python3 data/samples/create_test_video.py
"""

import cv2
import numpy as np
import os
import subprocess

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "test_dashcam_h264.mp4")
RAW_PATH = os.path.join(os.path.dirname(__file__), "test_dashcam_raw.mp4")
FPS = 30
DURATION_S = 15   # 15 seconds total
WIDTH, HEIGHT = 1280, 720


def draw_road(frame, t):
    """Draw a simple road with lane markings."""
    # Sky gradient
    for y in range(HEIGHT // 2):
        alpha = y / (HEIGHT // 2)
        c = int(100 + alpha * 50)
        frame[y, :] = [c + 20, c + 30, c + 50]  # sky blue-ish

    # Ground
    frame[HEIGHT // 2:, :] = [60, 65, 60]  # asphalt grey

    # Road polygon
    pts = np.array([
        [WIDTH // 2 - 80, HEIGHT // 2],
        [WIDTH // 2 + 80, HEIGHT // 2],
        [WIDTH, HEIGHT],
        [0, HEIGHT]
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (70, 70, 70))

    # Lane markings (dashes scrolling)
    speed = 15
    offset = int(t * speed * FPS) % 60
    for y in range(HEIGHT // 2, HEIGHT, 60):
        y_draw = y + offset - 60
        if HEIGHT // 2 < y_draw < HEIGHT:
            x = int(WIDTH // 2 - (y_draw - HEIGHT // 2) * 0.05)
            cv2.rectangle(frame, (x - 4, y_draw), (x + 4, y_draw + 35), (220, 220, 100), -1)


def draw_car(frame, x, y, w, h, color, label=""):
    """Draw a simplified car rectangle."""
    # Body
    cv2.rectangle(frame, (x, y + h // 3), (x + w, y + h), color, -1)
    # Roof
    roof_pts = np.array([
        [x + w // 4, y + h // 3],
        [x + 3 * w // 4, y + h // 3],
        [x + 2 * w // 3, y],
        [x + w // 3, y],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [roof_pts], tuple(max(0, c - 40) for c in color))
    # Windows
    cv2.rectangle(frame, (x + w // 3 + 2, y + 2), (x + 2 * w // 3 - 2, y + h // 3 - 2), (160, 200, 220), -1)
    # Wheels
    for wx in [x + w // 5, x + 4 * w // 5]:
        cv2.ellipse(frame, (wx, y + h), (w // 8, h // 6), 0, 0, 360, (30, 30, 30), -1)
    if label:
        cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def add_hud(frame, t, frame_idx, incident=False):
    """Add dashcam HUD overlay."""
    # Timestamp
    mins = int(t) // 60
    secs = int(t) % 60
    ms = int((t % 1) * 100)
    ts = f"REC  {mins:02d}:{secs:02d}.{ms:02d}"
    cv2.putText(frame, ts, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, "CAM: FRONT  GPS: 18.52N, 73.85E", (20, HEIGHT - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    if incident:
        cv2.putText(frame, "!! IMPACT DETECTED !!", (WIDTH // 2 - 180, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    # Frame counter
    cv2.putText(frame, f"F:{frame_idx:04d}", (WIDTH - 130, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main():
    total_frames = FPS * DURATION_S
    # Incident at frame 150 (5 seconds in)
    INCIDENT_FRAME = 150

    # Step 1: write raw frames with mp4v
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(RAW_PATH, fourcc, float(FPS), (WIDTH, HEIGHT))

    print(f"[Generator] Creating {DURATION_S}s synthetic dashcam video ({total_frames} frames)...")

    for fi in range(total_frames):
        t = fi / FPS
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_road(frame, t)

        # Car 1: red sedan
        car1_y = HEIGHT // 2 + 40 + int(50 * (t / DURATION_S))
        car1_x = WIDTH // 2 - 120 - int(t * 10)
        draw_car(frame, car1_x, car1_y, 110, 60, (40, 40, 180), "CAR")

        # Car 2: white SUV approaching to collide
        approach = min(1.0, max(0.0, (t - 3.0) / 2.0))
        car2_x = int(WIDTH * 0.9 - approach * 500)
        car2_y = HEIGHT // 2 + 30 + int(approach * 80)
        draw_car(frame, car2_x, car2_y, 130, 70, (220, 220, 220), "SUV")

        # Pedestrian
        ped_x = WIDTH // 2 + 200 - int(t * 5)
        cv2.rectangle(frame, (ped_x, HEIGHT // 2 + 20), (ped_x + 20, HEIGHT // 2 + 75), (100, 150, 220), -1)
        cv2.circle(frame, (ped_x + 10, HEIGHT // 2 + 10), 10, (210, 180, 140), -1)

        # Impact shake + flash
        incident_active = abs(fi - INCIDENT_FRAME) < 10
        if incident_active:
            frame = np.roll(frame, int(8 * np.random.randn()), axis=1)
            alpha = 1.0 - abs(fi - INCIDENT_FRAME) / 10.0
            frame = np.clip(frame * (1 - alpha * 0.5) + 255 * alpha * 0.5, 0, 255).astype(np.uint8)

        noise = np.random.randint(-8, 8, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        add_hud(frame, t, fi, incident=incident_active)
        out.write(frame)

    out.release()

    # Step 2: re-encode to H.264 for browser compatibility
    print("[Generator] Re-encoding to H.264 (browser-compatible)...")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", RAW_PATH,
        "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        OUTPUT_PATH
    ], capture_output=True, text=True)
    if result.returncode == 0:
        os.remove(RAW_PATH)
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        print(f"[Generator] ✓ H.264 video saved → {OUTPUT_PATH} ({size_mb:.1f}MB)")
    else:
        print(f"[Generator] FFmpeg failed; raw saved → {RAW_PATH}")
        print(result.stderr[:400])


if __name__ == "__main__":
    main()
