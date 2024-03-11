from mtcnn import MTCNN
from deepface import DeepFace
import cv2
import imutils
from tqdm import tqdm

import numpy as np



detector = MTCNN()

def process_video(path, prediction_model):


    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    i = 0

    # Get total frames

    all_embeddings = []
    predictions = []
    history = {}
    face_counts = []

    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret or i > 10:
                break

            frame = imutils.resize(frame, width=1024)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detections = detector.detect_faces(frame)
            face_counts.append(0)
            history[i] = []
            for detection in detections:
                confidence = detection["confidence"]
                if confidence > 0.8:
                    x, y, w, h = detection["box"]
                    detected_face = frame[int(y):int(y+h), int(x):int(x+w)]

                    embedding = np.array(DeepFace.represent(detected_face, model_name='Facenet512', enforce_detection=False)[0]['embedding'])
                    all_embeddings.append(embedding)
                    predictions.append(prediction_model.predict(detected_face))
                    # all_embeddings.append(np.append(embedding, i/5))
                    face_counts[-1] += 1
                    history[i].append((len(all_embeddings)-1, (x, y, w, h)))

            pbar.update(1)
            i += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    all_embeddings = np.array(all_embeddings)
    predictions = np.array(predictions)

    return history, all_embeddings, predictions, face_counts




