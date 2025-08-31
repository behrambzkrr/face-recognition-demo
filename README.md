# Face Recognition Demo

This repository contains two main scripts:

- `train_face_model.py`: Trains a face recognition model using images and saves it as `model.pkl`.
- `live_face_recognition.py`: Uses the trained model to recognize faces from webcam in real-time.

## Usage

### Training the Model

1. Create a `data/` directory. Inside, create subfolders for each person, using their name as the folder name. Place face images of each person in their respective folders:
    ```
    data/
      Alice/
        alice1.jpg
        alice2.jpg
      Bob/
        bob1.jpg
        bob2.jpg
    ```
2. Train the model:
    ```
    python train_face_model.py
    ```

### Real-Time Face Recognition

After training the model:
```
python live_face_recognition.py
```
The script will start your webcam and recognize faces in real-time. Press `q` to exit.

## Requirements

- Python 3.x
- face_recognition
- opencv-python
- scikit-learn

Install dependencies using pip:
```
pip install face_recognition opencv-python scikit-learn
```

## Notes

- Make sure your images are clear and each subfolder contains only images of the correct person.
- The trained model will be saved as `model.pkl` in the current directory.
