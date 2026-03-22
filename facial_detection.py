import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from retinaface import RetinaFace


def detect_and_display_faces(image_path: str, output_path: str = "faces_output.png"):
    results = RetinaFace.detect_faces(image_path)

    if not results:
        print("No faces detected.")
        return

    img = plt.imread(image_path)
    num_faces = len(results)
    print(f"Detected {num_faces} face(s).")

    fig, axes = plt.subplots(1, num_faces + 1, figsize=(5 * (num_faces + 1), 5))
    if num_faces == 0:
        axes = [axes]

    # Original image with bounding boxes
    ax_orig = axes[0]
    ax_orig.imshow(img)
    ax_orig.set_title("Detected Faces")
    ax_orig.axis("off")

    for i, (face_id, face_data) in enumerate(results.items()):
        x1, y1, x2, y2 = face_data["facial_area"]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax_orig.add_patch(rect)
        ax_orig.text(x1, y1 - 5, f"Face {i + 1}", color="red", fontsize=9, fontweight="bold")

        # Cropped face
        face_crop = img[y1:y2, x1:x2]
        axes[i + 1].imshow(face_crop)
        axes[i + 1].set_title(f"Face {i + 1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved output to {output_path}")
    plt.show()


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.png"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "faces_output.png"
    detect_and_display_faces(image_path, output_path)
