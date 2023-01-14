import cv2

# Load the video
cap = cv2.VideoCapture("C:\\Users\\enoobis\\Desktop\\body\\vid.mp4")

# Create the face classifier
face_classifier = cv2.CascadeClassifier('C:\\Users\\enoobis\\Desktop\\body\\haarcascade_frontalface_default.xml')

# Get the video frames per second (fps) and frame size
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Create the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size, isColor=True)

while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray, scaleFactor=3, minNeighbors=3)

    # Draw rectangles around the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    # Show the frame
    cv2.imshow("Face Recognition", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()