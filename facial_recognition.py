from deepface import DeepFace
import numpy as np
import os
import sys


def build_deepface():
    DeepFace.build_model('ArcFace')


def verify_face_not_in_excludes(face_crop: np.ndarray, exclude_path: str) -> bool:
    for exclude_face in os.listdir(exclude_path):
        exclude_face_path = os.path.join(exclude_path, exclude_face)
        try:
            if DeepFace.verify(face_crop, exclude_face_path, model_name='ArcFace', detector_backend='opencv')["verified"]:
                return False
        except Exception:
            return False
    return True


if __name__ == "__main__":
    import cv2
    image_to_check = sys.argv[1]
    frame = cv2.imread(image_to_check)
    if verify_face_not_in_excludes(frame, "faces_to_ignore"):
        print("This face is suspicious.")
    else:
        print("We know this person.")
