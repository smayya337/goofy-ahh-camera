import face_recognition
import pickle
import os

OUTPUT_PICKLE = "face_encodings.pkl"

# Subjects and paths: create a directory called "faces/" and put "faces/Display Name 1.jpg", "face/Display Name 2.png", ...
subjects = {
    os.path.splitext(filename)[0]: os.path.join("faces", filename)
    for filename in os.listdir("faces")
}

known_face_encodings = []
subject_names = []
print("[*] Computing face encodings...")

for name, path in subjects.items():
    if not os.path.exists(path):
        print(f"[!] File not found: {path}")
        continue

    img = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(img)

    if len(encodings) == 0:
        print(f"[!] No faces found in {path}")
        continue

    known_face_encodings.append(encodings[0])
    subject_names.append(name)
    print(f"[*] Encoded {name}")

# Cache encodings to prevent recomputing
with open(OUTPUT_PICKLE, "wb") as file:
    pickle.dump({
        "encodings": known_face_encodings,
        "names": subject_names
    }, file)

print(f"[*] Saved cached encodings to {OUTPUT_PICKLE}")
