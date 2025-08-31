import cv2
import face_recognition
import pickle

model = pickle.load(open("model.pkl", "rb"))
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()

while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        print("Frame could not be captured.")
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    faces = face_recognition.face_encodings(rgb_frame, face_locations)

    if not faces:
        cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for (top, right, bottom, left), enc in zip(face_locations, faces):
            try:
                prediction = model.predict([enc])[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, prediction, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            except Exception as e:
                print(f"Prediction error: {e}")
                cv2.putText(frame, "Unidentified", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()