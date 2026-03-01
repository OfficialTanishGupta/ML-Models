import cv2
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("../model/svm_mask_model.pkl", "rb"))
scaler = pickle.load(open("../model/scaler.pkl", "rb"))

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

img_size = 100

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (img_size, img_size))

        sample = face.flatten().reshape(1, -1)
        sample_scaled = scaler.transform(sample)

        prediction = model.predict(sample_scaled)

        label = "Mask" if prediction[0] == 0 else "No Mask"
        color = (0, 255, 0) if prediction[0] == 0 else (0, 0, 255)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Put label
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    cv2.imshow("Face Mask Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()