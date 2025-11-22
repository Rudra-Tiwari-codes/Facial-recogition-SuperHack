import cv2
import numpy as np
import json
import os
from face_recognition_module import FaceRecognizer
from age_detection import AgeDetector

class SmartHomeFacialRecognition:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.face_recognizer = FaceRecognizer(self.config['model_path'])
        self.age_detector = AgeDetector(self.config['age_model_path'])
        self.camera = cv2.VideoCapture(self.config['camera_id'])
        
    def run(self):
        print("Starting Facial Recognition System...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.face_recognizer.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                
                # Recognize person
                person_id, confidence = self.face_recognizer.recognize(face_img)
                
                # Estimate age
                age = self.age_detector.estimate_age(face_img)
                
                # Draw rectangle and labels
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                label = f"ID: {person_id} (Conf: {confidence:.2f})"
                age_label = f"Age: {age}"
                
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, age_label, (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Trigger smart home actions
                if confidence > self.config['recognition_threshold']:
                    self.trigger_smart_home_action(person_id, age)
            
            cv2.imshow('Smart Home Facial Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.camera.release()
        cv2.destroyAllWindows()
    
    def trigger_smart_home_action(self, person_id, age):
        """
        Trigger smart home automations based on recognized person
        """
        # This method would integrate with smart home APIs
        # Example: Adjust lighting, temperature, access control
        print(f"User {person_id} detected (Age: {age}). Applying personalized settings...")

if __name__ == "__main__":
    app = SmartHomeFacialRecognition()
    app.run()
