
import cv2
import os
import dlib

cap = cv2.VideoCapture(0)
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    detections = cascade_classifier.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles on detected faces
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import os

# Ask for person name
person_name = input("Enter the name of the person: ").strip()

# Create a folder for that person inside 'faces'
base_dir = 'faces'
person_dir = os.path.join(base_dir, person_name)
os.makedirs(person_dir, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

count = 0
print("Capturing images... Press 'q' to stop early.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the live frame
    cv2.imshow("Capture Faces", frame)

    # Save every few frames (you can adjust this)
    img_name = os.path.join(person_dir, f"{count}.jpg")
    cv2.imwrite(img_name, frame)
    count += 1

    # Stop if enough images are captured
    if count >= 30:  # Capture 30 images per person
        print("✅ Face data collection complete.")
        break

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped manually.")
        break

cap.release()
cv2.destroyAllWindows()


print(f"Dataset created for {user_name} with {count} images.")


