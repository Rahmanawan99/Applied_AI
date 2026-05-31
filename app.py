import cv2
import torch
import joblib
import numpy as np
from datetime import datetime
from PIL import Image
from flask import Flask, render_template, Response
from facenet_pytorch import InceptionResnetV1, MTCNN
from liveness import BlinkDetector
from attendance import mark_attendance

# =========================
# INIT MODELS
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Server starting on: {device}")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

try:
    model = joblib.load("models/face_classifier.pkl")
    print("Classifier loaded successfully.")
except Exception as e:
    print(f"Model error: {e}")

blink_detector = BlinkDetector()
THRESHOLD = 0.7

app = Flask(__name__)
cap = cv2.VideoCapture(0)

# =========================
# PROCESSING LOGIC
# =========================
def process_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    face = mtcnn(img)

    # Liveness check via MediaPipe
    blink_triggered, ear = blink_detector.detect_blink(frame)

    if face is not None:
        face = face.unsqueeze(0).to(device)
        embedding = resnet(face).detach().cpu().numpy()
        probs = model.predict_proba(embedding)[0]
        confidence = np.max(probs)
        classes = model.named_steps['kneighborsclassifier'].classes_
        name = classes[np.argmax(probs)]

        if confidence > 0.6:
            color = (0, 255, 0)
            cv2.putText(frame, f"{name.upper()} ({confidence:.2f})", (20, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            
            if blink_triggered:
                mark_attendance(name)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 255), 10)
                cv2.putText(frame, "ATTENDANCE MARKED", (100, 250), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
        else:
            cv2.putText(frame, f"UNKNOWN ({confidence:.2f})", (20, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    
    # Simple EAR HUD on Web
    cv2.rectangle(frame, (20, 70), (220, 90), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 70), (20 + int(min(ear*400, 200)), 90), (255, 255, 0), -1)
    cv2.putText(frame, f"Liveness: {ear:.2f}", (20, 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    return frame

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
