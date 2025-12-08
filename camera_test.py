import cv2
import time

# --- Set up camera ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access camera.")
    exit()

# Load face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: Could not load face cascade.")
    cap.release()
    exit()

prev_time = time.time()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Mirror the frame for a more natural feel
    frame = cv2.flip(frame, 1)

    # --- Face detection ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),  # green box
            2
        )

    # --- FPS calculation ---
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # --- On-screen text (instructions + FPS) ---
    cv2.putText(
        frame,
        "Press 'q' to quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Show the frame
    cv2.imshow("Camera Test - TSA Project", frame)

    # Wait for key press; quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Clean up ---
cap.release()
cv2.destroyAllWindows()
