import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not access camera.")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize for speed
    small = cv2.resize(gray, None, fx=0.5, fy=0.5)

    faces = face_cascade.detectMultiScale(
        small,
        scaleFactor=1.3,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        x, y, w, h = x*2, y*2, w*2, h*2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    current_time = time.time()
    delta = current_time - prev_time
    fps = 0.9 * fps + 0.1 * (1 / delta if delta > 0 else 0)
    prev_time = current_time

    cv2.putText(frame, "Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Camera Test - TSA Project", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
