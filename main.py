import cv2
import os
import queue
import threading
import time
import numpy as np
from discord_webhook import DiscordWebhook
from dotenv import load_dotenv
from facial_detection import detect_faces
from facial_recognition import build_deepface, verify_face_not_in_excludes

load_dotenv()

FRAME_SKIP = 3
FACE_IOU_THRESHOLD = 0.3
FACE_TRACK_TTL = 1.0  # seconds until a face track expires after last seen


def _iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = max(0, min(ax + aw, bx + bw) - ix)
    ih = max(0, min(ay + ah, by + bh) - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def expose(frame: cv2.typing.MatLike, now: float) -> None:
    frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
    webhook = DiscordWebhook(url=os.getenv("DISCORD_WEBHOOK_URL"), username="Goofy Ahh Camera", content="Look at this mf")
    webhook.add_file(file=frame_bytes, filename=f"img-{str(int(now))}.jpg")
    webhook.execute()


class FrameGrabber(threading.Thread):
    """Continuously drains the camera buffer so cap.read() always returns the latest frame."""

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self.cap = cap
        self.frame = None
        self.lock = threading.Lock()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def latest(self):
        with self.lock:
            return self.frame


class RecognitionWorker(threading.Thread):
    """Runs DeepFace verification off the main thread. Always processes the latest queued frame."""

    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue(maxsize=1)
        self._last_expose = 0
        self.ready = threading.Event()

    def submit(self, frame: np.ndarray, crop: np.ndarray):
        try:
            self.queue.put_nowait((frame, crop))
        except queue.Full:
            pass

    def run(self):
        build_deepface()
        self.ready.set()
        while True:
            frame, crop = self.queue.get()
            if verify_face_not_in_excludes(crop, os.getenv("EXCLUDES_PATH", "faces_to_ignore")):
                now = time.monotonic()
                if now - self._last_expose >= int(os.getenv("EXPOSE_COOLDOWN_SEC", 1)):
                    expose(frame, now)
                    self._last_expose = now


def main():
    cap = cv2.VideoCapture(os.getenv("CAMERA_URL"))

    grabber = FrameGrabber(cap)
    recognizer = RecognitionWorker()
    grabber.start()
    recognizer.start()

    print("Waiting for model to load...")
    recognizer.ready.wait()
    print("Camera stream opened, starting detection loop...")

    frame_count = 0
    # tracked_faces: list of {"bbox": (x,y,w,h), "last_seen": float}
    tracked_faces = []

    try:
        while True:
            frame = grabber.latest()
            if frame is None:
                continue

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            now = time.monotonic()

            # Expire stale tracks
            tracked_faces = [t for t in tracked_faces if now - t["last_seen"] < FACE_TRACK_TTL]

            for bbox in detect_faces(frame):
                x, y, w, h = bbox

                # Find matching track by IoU
                match = next((t for t in tracked_faces if _iou(bbox, t["bbox"]) >= FACE_IOU_THRESHOLD), None)
                if match:
                    match["bbox"] = bbox
                    match["last_seen"] = now
                else:
                    # New face — add track and submit crop for recognition
                    tracked_faces.append({"bbox": bbox, "last_seen": now})
                    crop = frame[y:y + h, x:x + w]
                    recognizer.submit(frame, crop)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
