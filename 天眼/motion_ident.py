import cv2
import numpy as np
import requests
import time
import face_recognition
import pickle
import os
from dotenv import load_dotenv

load_dotenv()
FEED = os.getenv("FEED_URL")
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
MOTION_THRESHOLD = 25000 # of pixels that need to change for it to trigger
LOCATION = "Rice 442" # friendly location name of the camera
CACHE_FILE = "face_encodings.pkl"
print("[*] Loading cached face encodings...")

if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError(f"Cache not found. Run 'python3 learn_faces.py' first.")

with open(CACHE_FILE, "rb") as f:
    data = pickle.load(f)
    known_face_encodings = data["encodings"]
    subject_names = data["names"]

print(f"[*] Loaded {len(known_face_encodings)} known faces from cache")
print("[*] Opening video stream...")
cap = cv2.VideoCapture(FEED)
prev_gray = None


def send_to_discord(frame):
    _, img_encoded = cv2.imencode(".jpg", frame)
    files = {"file": ("capture.jpg", img_encoded.tobytes(), "image/jpeg")}

    try:
        requests.post(WEBHOOK, files=files, timeout=3)
        print("[*] Sent image to Discord webhook")
    except Exception as e:
        print(f"[!] Failed to send image to Discord: {e}")


def detect_and_draw_faces(frame):
    recognized_names = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    face_locs = face_recognition.face_locations(rgb, model="cnn")

    if len(face_locs) == 0:
        face_locs = face_recognition.face_locations(rgb, model="cnn")

    print(f"[*] Detected {len(face_locs)} face(s)")

    try:
        encodings = face_recognition.face_encodings(rgb, face_locs)
    except Exception as e:
        print(f"[!] Failed to compute face encodings: {e}")
        encodings = []

    print(f"[*] Computed {len(encodings)} face encoding(s)")

    for (top, right, bottom, left), enc in zip(face_locs, encodings):
        matches = face_recognition.compare_faces(known_face_encodings, enc)
        name = "Unknown"

        if len(known_face_encodings) > 0:
            distances = face_recognition.face_distance(known_face_encodings, enc)
            best = np.argmin(distances)

            if matches[best]:
                name = subject_names[best]

        recognized_names.append(name)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, recognized_names


while True:
    ret, frame = cap.read()
    if not ret:
        print("[!] Failed to read frame from stream")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    if prev_gray is None:
        prev_gray = gray
        continue

    # Motion detection
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = np.count_nonzero(thresh)

    if MOTION_THRESHOLD * 4 > motion_score > MOTION_THRESHOLD:
        print(f"[!!] MOTION DETECTED | Score: {motion_score}")
        annotated, recognized_names = detect_and_draw_faces(frame.copy())
        send_to_discord(annotated)

        # Send known-face alert
        for name in recognized_names:
            payload = {"content": f"{name}已被AI监控系统于{LOCATION}侦测到，{time.strftime('时间为%Y年%m月%d日%H点%M分美国东部夏令时间')}"}
            try:
                requests.post(WEBHOOK, json=payload, timeout=3)
                print(f"[*] Sent known face alert: {payload['content']}")
            except Exception as e:
                print(f"[!] Failed to send known face alert: {e}")

        # flush buffer
        cap.release()
        cap = cv2.VideoCapture(FEED)
        prev_gray = None
        continue

    prev_gray = gray
