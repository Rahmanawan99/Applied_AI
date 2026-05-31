import cv2
import mediapipe as mp
import numpy as np
import time

class BlinkDetector:
    def __init__(self):
        # MediaPipe initialization fix
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmarks for eyes (standard indices for MediaPipe Face Mesh)
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Vertical and horizontal indices to calculate EAR (Eye Aspect Ratio)
        self.LEFT_VERT = [386, 374]
        self.LEFT_HORIZ = [263, 362]
        self.RIGHT_VERT = [159, 145]
        self.RIGHT_HORIZ = [133, 33]

        self.eye_closed_frames = 0
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.ear_threshold = 0.22  # Adjusted for better sensitivity
        self.consecutive_frames = 2

    def calculate_ear(self, landmarks, vert_idx, horiz_idx, img_w, img_h):
        # Vertical distance
        p1 = np.array([landmarks[vert_idx[0]].x * img_w, landmarks[vert_idx[0]].y * img_h])
        p2 = np.array([landmarks[vert_idx[1]].x * img_w, landmarks[vert_idx[1]].y * img_h])
        dist_v = np.linalg.norm(p1 - p2)

        # Horizontal distance
        p3 = np.array([landmarks[horiz_idx[0]].x * img_w, landmarks[horiz_idx[0]].y * img_h])
        p4 = np.array([landmarks[horiz_idx[1]].x * img_w, landmarks[horiz_idx[1]].y * img_h])
        dist_h = np.linalg.norm(p3 - p4)

        return dist_v / (dist_h + 1e-6)

    def detect_blink(self, frame):
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_ear = self.calculate_ear(landmarks, self.LEFT_VERT, self.LEFT_HORIZ, img_w, img_h)
            right_ear = self.calculate_ear(landmarks, self.RIGHT_VERT, self.RIGHT_HORIZ, img_w, img_h)
            
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < self.ear_threshold:
                self.eye_closed_frames += 1
            else:
                if self.eye_closed_frames >= self.consecutive_frames:
                    current_time = time.time()
                    if current_time - self.last_blink_time > 1.2:
                        self.blink_count += 1
                        self.last_blink_time = current_time
                        self.eye_closed_frames = 0
                        return True, avg_ear
                self.eye_closed_frames = 0
            
            return False, avg_ear
        
        return False, 0.0
