from deepface import DeepFace
import json
import os
import sys

def verify_face_not_in_excludes(face_path: str = sys.argv[1], exclude_path: str = "faces_to_ignore") -> bool:
    for exclude_face in os.listdir(exclude_path):
        exclude_face_path = os.path.join(exclude_path, exclude_face)
        # try:
        if DeepFace.verify(face_path, exclude_face_path, model_name='ArcFace', detector_backend='opencv')["verified"]:
            return False
        # except Exception:
        #     return True  # If there's an error (e.g., no face detected), we treat it as a suspicious case and return True
    return True

image_to_check = sys.argv[1]

if verify_face_not_in_excludes(image_to_check, "faces_to_ignore"):
    print("This face is suspicious.")
else:
    print("We know this person.")