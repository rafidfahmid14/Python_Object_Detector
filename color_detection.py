import cv2
import numpy as np
from sklearn.cluster import KMeans
import time
import ctypes


def get_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = image.reshape(-1, 3)

    # Remove extreme values
    pixels = pixels[(pixels[:, 0] > 20) & (pixels[:, 0] < 240)]

    if len(pixels) > 1000:
        pixels = pixels[np.random.choice(len(pixels), 1000, replace=False)]

    kmeans = KMeans(n_clusters=2, n_init=5)
    kmeans.fit(pixels)

    dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    # Convert back to RGB
    dominant_rgb = cv2.cvtColor(
        np.uint8([[dominant]]),
        cv2.COLOR_LAB2RGB
    )[0][0]

    return tuple(int(c) for c in dominant_rgb)


def run_color_detection(stop_event=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found or not accessible")
        return
    window_name = "Webcam Dominant Color"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    display_w, display_h = 640, 480
    cv2.resizeWindow(window_name, display_w, display_h)
    # center window where possible
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        move_x = max((screen_w - display_w) // 2, 0)
        move_y = max((screen_h - display_h) // 2, 0)
        cv2.moveWindow(window_name, move_x, move_y)
    except Exception:
        cv2.moveWindow(window_name, 120, 120)

    cv2.imshow(window_name, np.zeros((200, 400, 3), dtype=np.uint8))
    cv2.waitKey(1)

    prev_time = time.time()
    status_text = "Color mode"
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break

        color = get_dominant_color(frame)
        print("Dominant color (RGB):", color)

        # Convert RGB -> BGR for OpenCV drawing
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))

        # Show frame with a rectangle of dominant color and label
        cv2.rectangle(frame, (10, 10), (110, 110), color_bgr, -1)
        label = f"RGB: {color[0]}, {color[1]}, {color[2]}"
        cv2.putText(frame, label, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Compute and display FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # status overlay
        try:
            overlay = frame.copy()
            cv2.rectangle(overlay, (8, 8), (180, 36), (0, 0, 0), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, status_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) == 27:  # ESC key to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_color_detection()
