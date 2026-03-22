from deepface import DeepFace
import os
import sys

def build_deepface():
    DeepFace.build_model('ArcFace')

def verify_face_not_in_excludes(face_path: str, exclude_path: str) -> bool:
    for exclude_face in os.listdir(exclude_path):
        exclude_face_path = os.path.join(exclude_path, exclude_face)
        try:
            if DeepFace.verify(face_path, exclude_face_path, model_name='ArcFace', detector_backend='opencv')["verified"]:
                return False
        except Exception:
            return False  # If there's an error (e.g., no face detected), we just assume it's not that deep and return False
    return True

if __name__ == "__main__":
    image_to_check = sys.argv[1]
    if verify_face_not_in_excludes(image_to_check, "faces_to_ignore"):
        print("This face is suspicious.")
    else:
        print("We know this person.")