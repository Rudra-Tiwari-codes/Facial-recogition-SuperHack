# Facial Recognition System for Smart Homes

A Python-based facial recognition system designed for smart home applications with age detection capabilities.

## Features

- Real-time face detection and recognition
- Age estimation
- Person identification for access control
- Personalized smart home automation based on user recognition
- Support for multiple user profiles

## Requirements

- Python 3.8+
- Webcam or IP camera
- See `requirements.txt` for Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the System

```bash
python train_faces.py --data-dir ./face_data
```

### Running Recognition

```bash
python main.py
```

### Configuration

Edit `config.json` to customize:
- Camera settings
- Recognition threshold
- Age detection sensitivity
- Smart home integration endpoints

## Smart Home Integration

The system can integrate with popular smart home platforms to:
- Control access to specific rooms or devices
- Adjust lighting, temperature, and music based on recognized user preferences
- Track household member presence
- Provide age-appropriate content restrictions

## Project Structure

```
├── main.py                 # Main application entry point
├── face_recognition.py     # Face detection and recognition logic
├── age_detection.py        # Age estimation model
├── train_faces.py          # Training script for new faces
├── config.json             # Configuration file
├── requirements.txt        # Python dependencies
└── face_data/              # Directory for training images
```

## License

MIT License
