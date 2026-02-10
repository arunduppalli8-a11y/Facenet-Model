🎯 Face Recognition System using DeepFace & OpenCV

A real-time Face Recognition System built using Python, OpenCV, and DeepFace that can:
📸 Capture face images and create a dataset
🧠 Train facial embeddings using FaceNet
🧑‍💻 Recognize faces in real time via webcam
📊 Predict Age, Gender, and Emotion
❌ Identify Unknown faces

🚀 Features
Real-time face detection using Haar Cascade
Face embedding generation using FaceNet (DeepFace)
Cosine similarity–based face matching
Emotion, age, and gender analysis
Modular code structure (Dataset → Train → Recognize)
Works with live webcam feed

🛠️ Technologies Used
Python	      - Core programming language
OpenCV        -	Camera access & face detection
DeepFace      -	Face recognition & analysis
NumPy         -	Numerical computations
Haar Cascade	- Face detection

📂 Project Structure
Face-Recognition-System/
│
├── Dataset/
│   ├── Person1/
│   │   ├── Person1_1.jpg
│   │   ├── Person1_2.jpg
│   └── Person2/
│
├── embeddings.npy
├── main.py
└── README.md

⚙️ Installation & Setup
1️⃣ Clone the Repository
2️⃣ Install Required Libraries
pip install opencv-python numpy deepface tensorflow keras
⚠️ Make sure Python 3.8+ is installed
Webcam access is required
▶️ How to Run the Project
Run the main Python file: python main.py

You will see the following menu:
1. Create Dataset
2. Train Dataset
3. Recognize Face

📸 Step 1: Create Dataset
Choose option 1
Enter your name,The camera will open,Face images will be captured automatically,Press Q to stop
📁 Images are stored in: Dataset/YourName/

🧠 Step 2: Train Dataset
Choose option 2
Facial embeddings are generated using FaceNet
Embeddings are saved as: embeddings.npy

🧑‍💻 Step 3: Recognize Face
Choose option 3
Camera opens for real-time recognition
Displays:Name (or Unknown), Age, Gender, Emotion

Press Q to exit.

🔐 Face Matching Logic : Uses Cosine Similarity
Threshold:
similarity < 0.7 → Unknown
similarity ≥ 0.7 → Recognized

📊 Output Example
Arun | Age:21 | Man | Happy
Unknown | Age:25 | Woman | Neutral

🎯 Use Cases
Secure login systems
Attendance systems
Surveillance & monitoring
Smart authentication
Academic & research projects

🧠 Future Enhancements
🔐 Add liveness detection (anti-spoofing)
💾 Store embeddings in database
🌐 Web-based interface (Flask / Django)
📱 Mobile camera support
🔑 Multi-factor authentication integration

📜 License
This project is open-source and free to use for educational and research purposes.

🙌 Author
Arun Duppally
B.Tech CSE | Cybersecurity & AI Enthusiast
📍 India
