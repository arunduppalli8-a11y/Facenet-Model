import os
import numpy as np
import cv2
from deepface import DeepFace

# CREATE DATASET
dir = "Dataset"
os.makedirs(dir, exist_ok=True)
def create_dataset(name):
    person = os.path.join(dir, name)
    os.makedirs(person, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    count = 0
    print("📸 Starting camera... Look straight. Press Q to stop.")

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera not detected")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            count += 1
            face_path = os.path.join(person, f"{name}_{count}.jpg")
            cv2.imwrite(face_path, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Capturing {count}/20", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Capture Face - Press Q to Quit", frame)

        # Slow down capture (200ms per frame)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            print("🛑 Stopped by User")
            break


    cap.release()
    cv2.destroyAllWindows()


# TRAIN DATASET
def train_dataset():
    embeddings = {}
    for person_name in os.listdir(dir):
        person_path = os.path.join(dir, person_name)
        if os.path.isdir(person_path):
            embeddings[person_name] = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    rep = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)
                    emb = rep[0]["embedding"]
                    embeddings[person_name].append(emb)
                except Exception as e:
                    print("Failed to process image:", e)

    return embeddings


# RECOGNIZE FACE
def recognize_face(embeddings):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

            try:
                analysis = DeepFace.analyze(face_img, actions=["age","gender","emotion"], enforce_detection=False)
                if isinstance(analysis, list): analysis = analysis[0]

                age = analysis["age"]
                gender = analysis["gender"]
                if not isinstance(gender, str):
                    gender = max(gender, key=gender.get)

                emotion = max(analysis["emotion"], key=analysis["emotion"].get)

                face_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]

                match = "Unknown"
                max_similarity = -1

                for person, embeds in embeddings.items():
                    for emb in embeds:
                        sim = np.dot(face_embedding, emb) / (np.linalg.norm(face_embedding) * np.linalg.norm(emb))
                        if sim > max_similarity:
                            max_similarity = sim
                            match = person

                if max_similarity < 0.7:
                    match = "Unknown"

                text = f"{match} | Age:{age} | {gender} | {emotion}"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            except Exception as e:
                pass

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# MAIN
if __name__ == '__main__':
    while True:
        print("1. Create Dataset")
        print("2. Train Dataset")
        print("3. Recognize Face")

        choice = int(input("Enter your choice: "))

        if choice == 1:
            name = input("Enter your name: ")
            create_dataset(name)

        elif choice == 2:
            embeddings = train_dataset()
            np.save("embeddings.npy", embeddings)
            print("Training complete")

        elif choice == 3:
            if os.path.exists("embeddings.npy"):
                embeddings = np.load("embeddings.npy", allow_pickle=True).item()
                recognize_face(embeddings)
            else:
                print("Train dataset first")

        else:
            print("Invalid choice")
