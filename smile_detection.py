import cv2

# Load pre-trained Haar Cascade classifiers for face and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (Haar Cascades work on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) for the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect smiles within the face ROI
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

        # Iterate over each detected smile
        for (sx, sy, sw, sh) in smiles:
            # Draw a rectangle around the smile
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.putText(roi_color, 'Smile', (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with detected faces and smiles
    cv2.imshow('Real-Time Smile Detection', frame)

    # Exit the loop if 'e' is pressed
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()