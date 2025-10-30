import cv2
import numpy as np
import os

# Path to cropped faces
data_dir = "faces_cropped"

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_dict = {}
current_label = 0

# Loop through each folder (each person)
for person_name in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    label_dict[current_label] = person_name
    print(f"Processing: {person_name}")

    # Go through all images in the person's folder
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue  # skip unreadable files

        faces.append(img)
        labels.append(current_label)

    current_label += 1

# Train the recognizer
print("\nTraining the model, please wait...")
recognizer.train(faces, np.array(labels))

# Save the trained model
recognizer.save("face_model.yml")

# Save label dictionary
with open("labels.txt", "w") as f:
    for label, name in label_dict.items():
        f.write(f"{label}:{name}\n")

print("\n✅ Training complete!")
print("Model saved as 'face_model.yml'")
print("Labels saved as 'labels.txt'")


