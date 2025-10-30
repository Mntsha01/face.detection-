import cv2
import os
import time

# Ask for person name
person_name = input("Enter the name of the person: ").strip()

# Create a folder for that person inside 'faces'
base_dir = 'faces'
person_dir = os.path.join(base_dir, person_name)
os.makedirs(person_dir, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

count = 0
print("Capturing images... Move your head slightly (left, right, up, down). Press 'q' to stop early.")

start_time = time.time()
capture_interval = 0.5  # seconds between captures

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Faces", frame)

    # Capture a frame every 0.5 seconds
    current_time = time.time()
    if current_time - start_time >= capture_interval:
        img_name = os.path.join(person_dir, f"{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        count += 1
        start_time = current_time  # reset timer

    # Stop after 100 images OR after pressing 'q'
    if count >= 100:
        print("✅ Face data collection complete.")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped manually.")
        break

cap.release()
cv2.destroyAllWindows()
