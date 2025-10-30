"""print('''Twinkle, twinkle, little star

How I wonder what you are

Up above the world so high

Like a diamond in the sky

Twinkle, twinkle little star

How I wonder what you are

When the blazing sun is gone

When he nothing shines upon

Then you show your little light

Twinkle, twinkle, all the night
      ''')
# triple single quotes used to print"""
"""encodings = face_recognition.face_encodings(img)
if len(encodings) > 0:
    encode = encodings[0]
    encodeList.append(encode)
else:
    print(f"No face found in {cl}")"""



    import face_recognition, os

path = "/Users/mntshakhan/Downloads/chapter 1/faces_cropped/mntsha"

for f in os.listdir(path):
    img_path = os.path.join(path, f)
    img = face_recognition.load_image_file(img_path)
    enc = face_recognition.face_encodings(img)
    if len(enc) == 0:
        print("Deleting:", f)
        os.remove(img_path)




