import cv2
import os
import queue
import tempfile
import threading
import time
from discord_webhook import DiscordWebhook
from dotenv import load_dotenv
from facial_detection import detect_face
from facial_recognition import build_deepface, verify_face_not_in_excludes  # build_deepface called inside RecognitionWorker

load_dotenv()

def expose(frame: cv2.typing.MatLike, now: float) -> None:
    frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
    webhook = DiscordWebhook(url=os.getenv("DISCORD_WEBHOOK_URL"), username="Goofy Ahh Camera", content="Look at this mf")
    webhook.add_file(file=frame_bytes, filename=f"img-{str(int(now))}.jpg")
    response = webhook.execute()


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
        # maxsize=1: drop stale frames, only keep the most recent
        self.queue = queue.Queue(maxsize=1)
        self._last_expose = 0
        self.ready = threading.Event()

    def submit(self, frame):
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass  # recognition is busy; drop the old frame, caller will resubmit next detection hit

    def run(self):
        build_deepface()
        self.ready.set()
        while True:
            frame = self.queue.get()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                cv2.imwrite(tmp_path, frame)
                if verify_face_not_in_excludes(tmp_path, os.getenv("EXCLUDES_PATH", "faces_to_ignore")):
                    now = time.monotonic()
                    if now - self._last_expose >= int(os.getenv("EXPOSE_COOLDOWN_SEC", 1)):
                        expose(frame, now)
                        self._last_expose = now
            finally:
                os.unlink(tmp_path)


def main():
    cap = cv2.VideoCapture(os.getenv("CAMERA_URL"))

    grabber = FrameGrabber(cap)
    recognizer = RecognitionWorker()
    grabber.start()
    recognizer.start()

    print("Waiting for model to load...")
    recognizer.ready.wait()
    print("Camera stream opened, starting detection loop...")
    try:
        while True:
            frame = grabber.latest()
            if frame is None:
                continue

            if detect_face(frame):
                recognizer.submit(frame)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
