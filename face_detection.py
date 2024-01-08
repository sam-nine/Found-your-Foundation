import cv2
import os

# Set the directory to save the extracted faces
output_directory = "extracted_faces"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

while True:
    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        face_roi = frames[y:y+h, x:x+w]
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video', frames)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save each detected face as a separate image when 's' is pressed
        face_filename = os.path.join(output_directory, f"face_{len(os.listdir(output_directory))}.png")
        cv2.imwrite(face_filename, face_roi)
        print(f"Face saved as {face_filename}")

video_capture.release()
cv2.destroyAllWindows()
