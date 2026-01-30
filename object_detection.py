import cv2
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load YOLO model with absolute paths
weights_path = os.path.join(script_dir, "yolov3.weights")
config_path = os.path.join(script_dir, "yolov3.cfg")
names_path = os.path.join(script_dir, "coco.names")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Load class labels (COCO dataset)
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture device (camera not found)")
    cap.release()
    exit()

# Variable to toggle camera flip
flip_enabled = True

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera")
        break
    
    # Flip camera horizontally (mirror effect) if enabled
    if flip_enabled:
        frame = cv2.flip(frame, 1)
    
    height, width, channels = frame.shape

    # Convert frame to input blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Lists for detected objects
    class_ids = []
    confidences = []
    boxes = []
    detected_objects = []  # For debugging

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.10:  # threshold (lowered from 0.15)
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_objects.append((classes[class_id], confidence))

    # Non-max suppression to remove duplicates
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    indexes = indexes.flatten() if len(indexes) > 0 else []
    
    # Debug: Print detected objects and their confidence scores
    if detected_objects:
        print("Detected objects:", [(obj, f"{conf:.4f}") for obj, conf in detected_objects])

    # Face detection using cascade classifier (always run for better detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    person_detected = any(classes[class_ids[i]] == 'person' for i in indexes)
    
    # Detect faces with optimized parameters (runs regardless of YOLO person detection)
    # scaleFactor=1.03 (very sensitive), minNeighbors=4 (balanced), minSize=(20, 20)
    faces = face_cascade.detectMultiScale(gray, 1.03, 4, minSize=(20, 20))
    
    # Debug: Print face detection info
    if len(faces) > 0:
        print(f"Faces detected: {len(faces)}")

    # Draw YOLO bounding boxes
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), font, 0.5, color, 2)
    
    # Draw face detection boxes (blue color to distinguish from YOLO)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), font, 0.5, (255, 0, 0), 2)

    cv2.imshow("YOLO Object Detection", frame)

    # Handle keyboard input
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == ord('f') or key == ord('F'):  # F to toggle flip
        flip_enabled = not flip_enabled
        print(f"Flip: {'ON' if flip_enabled else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
