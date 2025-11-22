import argparse
import os
import cv2
from face_recognition_module import FaceRecognizer

def collect_training_data(person_id, output_dir, num_samples=50):
    """
    Collect face images from webcam for training
    """
    os.makedirs(os.path.join(output_dir, str(person_id)), exist_ok=True)
    
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    count = 0
    print(f"Collecting {num_samples} samples for person {person_id}")
    print("Press SPACE to capture, ESC to cancel")
    
    while count < num_samples:
        ret, frame = camera.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Collecting Training Data', frame)
        
        key = cv2.waitKey(1)
        if key == 32:  # SPACE
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = gray[y:y+h, x:x+w]
                
                img_path = os.path.join(output_dir, str(person_id), f"{count}.jpg")
                cv2.imwrite(img_path, face_img)
                count += 1
                print(f"Captured image {count}/{num_samples}")
        elif key == 27:  # ESC
            break
    
    camera.release()
    cv2.destroyAllWindows()
    print(f"Collection complete. {count} images saved.")

def train_model(data_dir):
    """
    Train the face recognition model
    """
    print(f"Training model with data from {data_dir}")
    recognizer = FaceRecognizer()
    recognizer.train(data_dir)
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train facial recognition system')
    parser.add_argument('--data-dir', type=str, default='face_data',
                       help='Directory containing training images')
    parser.add_argument('--collect', type=int, default=None,
                       help='Collect training data for person ID')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of samples to collect')
    
    args = parser.parse_args()
    
    if args.collect is not None:
        collect_training_data(args.collect, args.data_dir, args.samples)
    
    train_model(args.data_dir)
