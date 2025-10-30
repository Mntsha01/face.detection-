#!/usr/bin/env python3
"""
Simple real-time face recognition using face_recognition + OpenCV.

Usage:
  1) Prepare known faces in ./known_faces/<name>/*.jpg
  2) Run: python recognize.py
"""

'''import os
import time
import face_recognition
import cv2
from pathlib import Path

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.5       # lower = stricter (0.4-0.6 typical)
FRAME_THICKNESS = 2
FONT_THICKNESS = 1
MODEL = "hog"         # "hog" (CPU) or "cnn" (GPU, slower to load; requires dlib compiled with CUDA)

print("Loading known faces...")

known_faces = []
known_names = []

# Walk through known faces directory
for person_dir in Path(KNOWN_FACES_DIR).iterdir():
    if not person_dir.is_dir():
        continue
    name = person_dir.name
    for img_path in person_dir.glob("*"):
        try:
            image = face_recognition.load_image_file(str(img_path))
            encodings = face_recognition.face_encodings(image)
            if len(encodings) == 0:
                print(f"Warning: no faces found in {img_path}; skipping")
                continue
            known_faces.append(encodings[0])
            known_names.append(name)
            print(f"Loaded {img_path} for {name}")
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")

if len(known_faces) == 0:
    raise SystemExit("No known faces found. Put images in ./known_faces/<name>/")

# Open webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise SystemExit("Unable to open webcam. Check camera permission in macOS System Settings.")

process_this_frame = True

print("Starting video. Press 'q' to quit.")
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to read from camera")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Find all faces and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small, model=MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_names = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, encoding, tolerance=TOLERANCE)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            distances = face_recognition.face_distance(known_faces, encoding)
            if len(distances) > 0:
                best_match_index = distances.argmin()
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)
        # Draw label
        text_width, text_height = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.5, FONT_THICKNESS)[0]
        cv2.rectangle(frame, (left, bottom - text_height - 10), (left + text_width + 10, bottom), (0,255,0), cv2.FILLED)
        cv2.putText(frame, name, (left + 5, bottom - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), FONT_THICKNESS)

    cv2.imshow("Face Recognition", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()'''

import cv2 as cv
image = cv.imread('family2.jpeg')
cv.imshow('Found Face',image)
cv.waitkey(0)
cv.destroyAllWindows()
