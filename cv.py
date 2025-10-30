import cv2
video_cap=cv2.VideoCapture(0)
while True :
    ret , video = video_cap.read()
    cv2.imshow("video_live",video_data)
    if {
        cv2.waitkey(10)==ord("a")
        }:
    break;
video_cap.release()
Input in [3]
if cv2.waitkey(10)=ord("a"):