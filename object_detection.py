import cv2
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load YOLO model with absolute paths
weights_path = os.path.join(script_dir, "yolov3.weights")
config_path = os.path.join(script_dir, "yolov3.cfg")
names_path = os.path.join(script_dir, "coco.names")


def run_object_detection(stop_event=None):
    # Ensure model files exist and are readable
    if not os.path.isfile(config_path):
        print(f"Error: config file not found at {config_path}")
        return
    if not os.path.isfile(weights_path):
        print(f"Error: weights file not found at {weights_path}. Please download yolov3.weights into the project folder")
        return
    if not os.path.isfile(names_path):
        print(f"Error: names file not found at {names_path}")
        return

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # Load class labels (COCO dataset)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video capture device (camera not found)")
        cap.release()
        return

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            break

        height, width, channels = frame.shape

        # Convert frame to input blob
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Lists for detected objects
        class_ids = []
        confidences = []
        boxes = []

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # threshold
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

        # Non-max suppression to remove duplicates
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        indexes = indexes.flatten() if len(indexes) > 0 else []

        # Draw bounding boxes
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), font, 0.5, color, 2)

        cv2.imshow("YOLO Object Detection", frame)

        # Exit when ESC key is pressed
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_object_detection()
