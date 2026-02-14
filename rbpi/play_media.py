"""
Play a GIF or video file on the ST7735 TFT display (160x128).

Usage:
    python play_media.py /path/to/animation.gif
    python play_media.py /path/to/video.mp4 --fps 15
    python play_media.py /path/to/clip.avi --loop --fps 24

Supported formats:
    GIF  — via PIL (no extra deps)
    Video (mp4, avi, mkv, mov, webm) — requires opencv-python
"""

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from run_on_tft import initialize_display, show_text

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
GIF_EXTENSIONS = {".gif"}
DISPLAY_WIDTH = 160
DISPLAY_HEIGHT = 128


def play_gif(device, path, fps, loop):
    """Extract frames from a GIF and display them on the TFT."""
    img = Image.open(path)
    n_frames = getattr(img, "n_frames", 1)
    delay = 1.0 / fps if fps else None

    print(f"Playing GIF: {path} ({n_frames} frames)")
    try:
        while True:
            for i in range(n_frames):
                img.seek(i)
                frame = img.convert("RGB").resize(
                    (DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.LANCZOS
                )
                device.display(frame)
                if delay is not None:
                    time.sleep(delay)
                else:
                    # Use GIF's own frame duration (in ms), default 33ms
                    duration_ms = img.info.get("duration", 33)
                    time.sleep(duration_ms / 1000.0)
            if not loop:
                break
    except KeyboardInterrupt:
        pass


def play_video(device, path, fps, loop):
    """Decode a video file and display frames on the TFT."""
    try:
        import cv2
    except ImportError:
        print(
            "opencv-python is required for video playback.\n"
            "Install it with:  pip install opencv-python"
        )
        sys.exit(1)

    print(f"Playing video: {path}")
    try:
        while True:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                print(f"Error: cannot open {path}")
                sys.exit(1)

            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            delay = 1.0 / (fps if fps else video_fps)

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame_rgb).resize(
                    (DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.LANCZOS
                )
                device.display(frame)
                time.sleep(delay)

            cap.release()
            if not loop:
                break
    except KeyboardInterrupt:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Play GIF/video on ST7735 TFT")
    p.add_argument("file", type=str, help="Path to GIF or video file")
    p.add_argument("--fps", type=float, default=None, help="Override playback FPS")
    p.add_argument("--loop", action="store_true", help="Loop playback continuously")
    return p.parse_args()


def main():
    args = parse_args()
    filepath = Path(args.file)

    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    ext = filepath.suffix.lower()
    if ext not in GIF_EXTENSIONS | VIDEO_EXTENSIONS:
        print(f"Unsupported format: {ext}")
        print(f"Supported: {', '.join(sorted(GIF_EXTENSIONS | VIDEO_EXTENSIONS))}")
        sys.exit(1)

    device, gpio_backend = initialize_display()
    show_text(device, "Loading...", filepath.name[:20])
    time.sleep(0.5)

    try:
        if ext in GIF_EXTENSIONS:
            play_gif(device, str(filepath), args.fps, args.loop)
        else:
            play_video(device, str(filepath), args.fps, args.loop)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            show_text(device, "Stopped")
            time.sleep(0.5)
            device.clear()
            device.show()
        except Exception:
            pass
        if gpio_backend is not None:
            try:
                gpio_backend.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    main()
