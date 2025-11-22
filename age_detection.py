import cv2
import numpy as np
import os

class AgeDetector:
    def __init__(self, model_path='models'):
        self.model_path = model_path
        
        # Age ranges for classification
        self.age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                          '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        # Load pre-trained age detection model
        # Using OpenCV's pre-trained DNN models
        age_proto = os.path.join(model_path, 'age_deploy.prototxt')
        age_model = os.path.join(model_path, 'age_net.caffemodel')
        
        if os.path.exists(age_proto) and os.path.exists(age_model):
            self.age_net = cv2.dnn.readNet(age_model, age_proto)
            self.model_loaded = True
        else:
            self.model_loaded = False
            print("Warning: Age detection model not found.")
            print("Download models from: https://github.com/GilLevi/AgeGenderDeepLearning")
    
    def estimate_age(self, face_img):
        """
        Estimate age from a face image
        Returns estimated age range
        """
        if not self.model_loaded:
            return "Unknown"
        
        blob = cv2.dnn.blobFromImage(
            face_img, 
            1.0, 
            (227, 227), 
            (78.4263377603, 87.7689143744, 114.895847746), 
            swapRB=False
        )
        
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age_idx = age_preds[0].argmax()
        
        return self.age_ranges[age_idx]
    
    def estimate_age_simple(self, face_img):
        """
        Simple age estimation based on face dimensions (fallback method)
        This is a basic heuristic when DNN model is not available
        """
        # This is a simplified placeholder implementation
        # In practice, would use more sophisticated features
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Use facial feature ratios as a basic heuristic
        # This is not accurate but serves as a fallback
        h, w = gray.shape
        ratio = h / w
        
        if ratio < 1.2:
            return "(4-12)"
        elif ratio < 1.35:
            return "(15-32)"
        else:
            return "(38-100)"
