# Crime and Violence Detection using YOLO

This project implements a real-time crime and violence detection system using the YOLO (You Only Look Once) object detection model. The model is trained to differentiate between violent and non-violent activities in video footage.

## Features
- Uses YOLOv8 for video detection
- Detects violent and non-violent activities
- Trained on a custom dataset with two classes: `Violence` and `NonViolence`
- Can process video streams or real-time footage

## Dataset
The dataset consists of two categories:
1. **Violence** - Videos/images depicting violent activities.
2. **NonViolence** - Videos/images with normal activities.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/crime-violence-detection.git
   cd crime-violence-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the YOLOv6 model weights and place them in the `weights/` directory.

## Usage
### Training the Model
```bash
python train.py --data data.yaml --cfg yolov6.yaml --weights yolov6s.pt --epochs 50
```

### Running Detection
```bash
python detect.py --source video.mp4 --weights weights/best.pt --conf 0.5
```

### Real-time Detection
```bash
python detect.py --source 0 --weights weights/best.pt --conf 0.5
```

## Results
- The model achieves high accuracy in differentiating between violent and non-violent activities.
- Can be integrated into security surveillance systems for real-time monitoring.

## Future Improvements
- Improve dataset size and diversity.
- Fine-tune model for better accuracy.
- Deploy as a web-based application.

## Contributing
Feel free to contribute to this project by submitting issues and pull requests.

## License
This project is licensed under the MIT License.

---

### Contact
For any queries, contact **your-email@example.com** or open an issue on GitHub.
