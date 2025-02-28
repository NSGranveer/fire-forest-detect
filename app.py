from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLO model
model = YOLO("best.pt")

@app.route('/')
def home():
    return "Welcome to the Fire Detection API! Use the /track endpoint to run the model."

@app.route('/track', methods=['POST'])  # Corrected: Route starts with a slash
def track_fire():
    video_file = request.form.get('video_file', 'fire1.mp4')
    confidence = float(request.form.get('conf', 0.01))
    iou_threshold = float(request.form.get('iou', 0.3))

    try:
        results = model.track(source=video_file, save=True, conf=confidence, iou=iou_threshold)
        result_image_path = str(results[0].save_dir)
        return jsonify({"status": "success", "result_image_path": result_image_path})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)





# from flask import Flask, request, jsonify, Response
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os

# app = Flask(__name__)

# # Load YOLOv8 model
# MODEL_PATH = "best.pt"
# model = YOLO(MODEL_PATH)

# # Ensure upload directory exists
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # ---------------- FILE UPLOAD & PROCESSING ---------------- #
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"}), 400

#     file = request.files['file']
#     filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(filepath)

#     # Run YOLO on uploaded file
#     results = model(filepath)
#     detections = []

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
#             confidence = float(box.conf[0])  # Confidence score
#             detections.append({"bbox": [x1, y1, x2, y2], "confidence": confidence})

#     return jsonify({"status": "success", "detections": detections, "file_saved": filepath})


# # ---------------- FIXED LIVE WEBCAM STREAMING ---------------- #
# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Open webcam only when requested

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         # Perform inference
#         results = model(frame)
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 confidence = float(box.conf[0])

#                 # Draw bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(frame, f"Fire {confidence:.2f}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Encode frame
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()  # Release webcam when the loop ends

# @app.route('/video_feed')
# def video_feed():
#     """ Start webcam streaming only when requested """
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/')
# def index():
#     return "Welcome to Fire Detection API! Use /upload for file processing and /video_feed for live detection."

# if __name__ == '__main__':
#     app.run(debug=True)
