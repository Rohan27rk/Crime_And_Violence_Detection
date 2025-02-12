from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import requests

# Initialize Flask app
app = Flask(__name__)

# Initialize YOLO models
violence_model = YOLO("runs/detect/my_yolo_model/weights/best.pt")
weapon_model = YOLO("weap/runs/detect/Normal_Compressed/weights/best.pt")

# Pushbullet Access Token and URL
PUSHBULLET_ACCESS_TOKEN = "o.7pC5XHC9RfhGQ8Zp0vdRoCRCrmlxmglp"
PUSHBULLET_URL = "https://api.pushbullet.com/v2/pushes"

def send_push_notification(title, body):
    """Send a push notification using the Pushbullet API."""
    headers = {
        "Access-Token": PUSHBULLET_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }
    data = {
        "type": "note",
        "title": title,
        "body": body
    }
    try:
        response = requests.post(PUSHBULLET_URL, json=data, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(f"Notification sent: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send notification: {e}")

# Routes
@app.route('/')
def index():
    # Display the main page for streaming
    return render_template('index.html')

@app.route('/stream')
def stream():
    # Start webcam streaming
    return Response(process_video_realtime(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def process_video_realtime():
    # Open the default webcam
    cap = cv2.VideoCapture(0)  # 0 refers to the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame using both YOLO models
        violence_results = violence_model.predict(frame, conf=0.5)  # Ensure `predict` is used correctly
        weapon_results = weapon_model.predict(frame, conf=0.5)

        detected_classes_violence = []
        detected_classes_weapon = []

        # Extract detected classes for violence
        if violence_results and len(violence_results[0].boxes) > 0:
            detected_classes_violence = [
                violence_model.names[int(box.cls)] for box in violence_results[0].boxes
            ]

        # Extract detected classes for weapons
        if weapon_results and len(weapon_results[0].boxes) > 0:
            detected_classes_weapon = [
                weapon_model.names[int(box.cls)] for box in weapon_results[0].boxes
            ]

        # Send notifications based on detected classes
        if 'violence' in detected_classes_violence:
            send_push_notification(
                title="Alert: Violence Detected",
                body="Possible violent activity detected on the webcam feed!"
            )

        if detected_classes_weapon:  # Check if there are any detected weapons
            weapon_names = ", ".join(detected_classes_weapon)  # Combine weapon names if multiple
            send_push_notification(
                title="Alert: Weapon Detected",
                body=f"Weapon(s) detected: {weapon_names} on the webcam feed!"
            )

        # Annotate the frame with detection results
        for result in [violence_results, weapon_results]:
            if result and len(result[0].boxes) > 0:  # Check for detections
                frame = result[0].plot()

        # Encode the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()

        # Yield the frame to the stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    cap.release()


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
