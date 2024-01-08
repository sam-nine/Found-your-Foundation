import cv2
import os

output_directory = "extracted_faces"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Full path to the Haar Cascade XML file
cascPath = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
frames = cv2.imread(r"C:\Users\Mahita\Desktop\Sem 5 pdfs\img2.jpg") 
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
    face_filename = os.path.join(output_directory, f"face_{len(os.listdir(output_directory))}.png")
    cv2.imwrite(face_filename, face_roi)
    cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Image with Faces', frames)
cv2.waitKey(0)
cv2.destroyAllWindows()
