import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

input_dir = "faces"
output_dir = "faces_cropped"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for person_name in os.listdir(input_dir):
    person_path = os.path.join(input_dir, person_name)
    output_person_path = os.path.join(output_dir, person_name)
    os.makedirs(output_person_path, exist_ok=True)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))
            save_path = os.path.join(output_person_path, f"{i}_{img_name}")
            cv2.imwrite(save_path, face_resized)

print("All faces cropped and saved in:", output_dir)

