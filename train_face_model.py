import os
import face_recognition
from sklearn import neighbors
import pickle

data_dir = "data"
X = []
y = []

for person in os.listdir(data_dir):
    person_folder = os.path.join(data_dir, person)
    for file in os.listdir(person_folder):
        path = os.path.join(person_folder, file)
        image = face_recognition.load_image_file(path)
        faces = face_recognition.face_encodings(image)
        if faces:
            X.append(faces[0])
            y.append(person)

model = neighbors.KNeighborsClassifier(n_neighbors=2)
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")