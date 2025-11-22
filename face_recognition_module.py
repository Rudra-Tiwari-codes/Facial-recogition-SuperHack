import cv2
import numpy as np
import pickle
import os

class FaceRecognizer:
    def __init__(self, model_path='models'):
        self.model_path = model_path
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Load trained model if exists
        model_file = os.path.join(model_path, 'face_model.yml')
        if os.path.exists(model_file):
            self.recognizer.read(model_file)
            self.trained = True
        else:
            self.trained = False
            print("Warning: No trained model found. Please train the system first.")
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame
        Returns list of face coordinates (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def recognize(self, face_img):
        """
        Recognize a face
        Returns (person_id, confidence)
        """
        if not self.trained:
            return "Unknown", 0.0
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))
        
        person_id, confidence = self.recognizer.predict(gray)
        
        # Convert confidence to a 0-1 scale (lower is better in LBPH)
        confidence_score = max(0, 100 - confidence) / 100
        
        return person_id, confidence_score
    
    def train(self, data_dir):
        """
        Train the recognizer with images from data_dir
        Expected structure: data_dir/person_id/image.jpg
        """
        faces = []
        labels = []
        
        for person_id in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person_id)
            if not os.path.isdir(person_path):
                continue
            
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                img = cv2.resize(img, (200, 200))
                faces.append(img)
                labels.append(int(person_id))
        
        if len(faces) > 0:
            self.recognizer.train(faces, np.array(labels))
            
            os.makedirs(self.model_path, exist_ok=True)
            model_file = os.path.join(self.model_path, 'face_model.yml')
            self.recognizer.save(model_file)
            
            self.trained = True
            print(f"Model trained successfully with {len(faces)} images")
        else:
            print("No training data found")
