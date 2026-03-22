import sys
import cv2
import numpy as np

_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_SCALE = 0.5


def detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Returns list of (x, y, w, h) bounding boxes in original frame coordinates."""
    small = cv2.resize(frame, (0, 0), fx=_SCALE, fy=_SCALE)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = _detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return []
    return [
        (int(x / _SCALE), int(y / _SCALE), int(w / _SCALE), int(h / _SCALE))
        for x, y, w, h in faces
    ]


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.png"
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")
    print(detect_faces(frame))
