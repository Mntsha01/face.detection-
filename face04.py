import face_recognition
import cv2
from pathlib import Path

# ---------------- Configuration ----------------
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.5       # 0.4 = strict, 0.6 = loose
MODEL = "hog"         # "hog" for CPU, "cnn" for GPU
FRAME_THICKNESS = 2
FONT_THICKNESS = 1
# ------------------------------------------------

print("Loading known faces...")
known_faces = []
known_names = []

for person_dir in Path(KNOWN_FACES_DIR).iterdir():
    if not person_dir.is_dir():
        continue
    name = person_dir.name
    for image_path in person_dir.glob("*"):
        image = face_recognition.load_image_file(str(image_path))
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_faces.append(encodings[0])
            known_names.append(name)
            print(f"Loaded {image_path} for {name}")

if not known_faces:
    raise SystemExit("No known faces found. Add images to ./known_faces/<name>/")

# ---------------- Start Video ----------------
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise SystemExit
