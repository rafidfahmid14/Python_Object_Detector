import threading
import time
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
#import pyttsx3
import queue
from common import speak

class CombinedManager:
    def __init__(self, stop_event=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(self.script_dir, "yolov3.weights")
        self.config_path = os.path.join(self.script_dir, "yolov3.cfg")
        self.names_path = os.path.join(self.script_dir, "coco.names")

        self.cap = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.object_enabled = False
        self.color_enabled = False
        self.stopped = False
        self.stop_event = stop_event

        self.net = None
        self.output_layers = None
        self.classes = []
        self.detections = []
        self.detections_lock = threading.Lock()

        self.dominant_color = (0, 0, 0)
        self.color_lock = threading.Lock()

        self.threads = []
        # TTS
        self._tts_queue = queue.Queue()
        self._tts_engine = None
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

        # track last-announcement times to avoid repeats: label -> timestamp
        self._last_announced = {}
        self._announce_cooldown = 5.0  # seconds (reduced for testing)

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not available")

        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start(); self.threads.append(t)
        t = threading.Thread(target=self._object_loop, daemon=True)
        t.start(); self.threads.append(t)
        t = threading.Thread(target=self._color_loop, daemon=True)
        t.start(); self.threads.append(t)
        t = threading.Thread(target=self._display_loop, daemon=True)
        t.start(); self.threads.append(t)

    def stop(self):
        self.stopped = True
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()

    def enable_object(self, enable=True):
        if enable and not self.net:
            # lazy-load YOLO
            if not os.path.isfile(self.config_path) or not os.path.isfile(self.weights_path) or not os.path.isfile(self.names_path):
                print("Missing YOLO files; cannot enable object detection")
                return
            print("[DEBUG] Loading YOLO model (this may take 30-60 seconds)...")
            try:
                self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
                with open(self.names_path, "r") as f:
                    self.classes = [l.strip() for l in f.readlines()]
                print(f"[DEBUG] Loaded {len(self.classes)} object classes")
                layer_names = self.net.getLayerNames()
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
                print("[DEBUG] YOLO model ready")
            except Exception as e:
                print(f"[DEBUG] Error loading YOLO: {e}")
                self.net = None
                return
        self.object_enabled = enable
        print(f"[DEBUG] Object detection enabled: {enable}")

    def enable_color(self, enable=True):
        self.color_enabled = enable

    def _capture_loop(self):
        while not self.stopped and (self.stop_event is None or not self.stop_event.is_set()):
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.frame_lock:
                self.latest_frame = frame.copy()
            time.sleep(0.01)

    def _object_loop(self):
        while not self.stopped and (self.stop_event is None or not self.stop_event.is_set()):
            if not self.object_enabled or self.net is None:
                time.sleep(0.05)
                continue
            # get latest frame
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                time.sleep(0.01); continue

            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0,0,0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = int(np.argmax(scores))
                    confidence = float(scores[class_id])
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x,y,w,h])
                        confidences.append(confidence)
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            indexes = indexes.flatten() if len(indexes) > 0 else []


            dets = []
            labels_present = set()
            for i in indexes:
                dets.append((boxes[i], class_ids[i], confidences[i]))
                if class_ids[i] < len(self.classes):
                    labels_present.add(self.classes[class_ids[i]])

            with self.detections_lock:
                self.detections = dets

            # Announce newly detected labels (with cooldown)
            if labels_present:
                now = time.time()
                print(f"[DEBUG] Detected labels: {labels_present}")
                for label in labels_present:
                    last = self._last_announced.get(label, 0)
                    time_since = now - last
                    if time_since >= self._announce_cooldown:
                        # enqueue text-to-speech
                        print(f"[DEBUG] Announcing: '{label}' (last {time_since:.1f}s ago)")
                        self._tts_queue.put(label)
                        self._last_announced[label] = now
                    else:
                        print(f"[DEBUG] Skipping '{label}' (announced {time_since:.1f}s ago, cooldown {self._announce_cooldown}s)")

            # throttle
            time.sleep(0.2)

    def _get_dominant_color(self, image):
        try:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        except Exception:
            return (0,0,0)
        pixels = img.reshape(-1,3)
        pixels = pixels[(pixels[:,0] > 20) & (pixels[:,0] < 240)]
        if len(pixels) > 1000:
            pixels = pixels[np.random.choice(len(pixels), 1000, replace=False)]
        kmeans = KMeans(n_clusters=2, n_init=3)
        kmeans.fit(pixels)
        dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        dominant_rgb = cv2.cvtColor(np.uint8([[dominant]]), cv2.COLOR_LAB2RGB)[0][0]
        return tuple(int(c) for c in dominant_rgb)

    def _color_loop(self):
        while not self.stopped and (self.stop_event is None or not self.stop_event.is_set()):
            if not self.color_enabled:
                time.sleep(0.05); continue
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                time.sleep(0.01); continue
            color = self._get_dominant_color(frame)
            with self.color_lock:
                self.dominant_color = color
            time.sleep(0.2)

    def _display_loop(self):
        window_name = "Combined View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        while not self.stopped and (self.stop_event is None or not self.stop_event.is_set()):
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            if frame is None:
                time.sleep(0.01); continue

            # draw object detections
            with self.detections_lock:
                dets = list(self.detections)
            for (box, cid, conf) in dets:
                x,y,w,h = box
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                label = self.classes[cid] if cid < len(self.classes) else str(cid)
                cv2.putText(frame, f"{label} {conf:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # draw color box
            with self.color_lock:
                color = self.dominant_color
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.rectangle(frame, (10,10), (110,110), color_bgr, -1)
            cv2.putText(frame, f"RGB: {color[0]}, {color[1]}, {color[2]}", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # status
            status = f"Obj: {'ON' if self.object_enabled else 'OFF'}  Col: {'ON' if self.color_enabled else 'OFF'}"
            cv2.rectangle(frame, (8,8), (260,36), (0,0,0), -1)
            cv2.addWeighted(frame, 0.9, frame, 0.1, 0, frame)
            cv2.putText(frame, status, (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) == 27:
                self.stop()
                break
            time.sleep(0.01)

    

    def _tts_worker(self):
        # cleaned up speak to move to function to solve issue with pyttsx not working for multiple calls in a row        
        while not self.stopped and (self.stop_event is None or not self.stop_event.is_set()):
            try:
                text = self._tts_queue.get(timeout=0.5)
                if text == "person":
                    continue
                print(f"[DEBUG] Speaking: {text}")
                speak(text)
                print(f"[DEBUG] Speaking (after runAndWait): {text}")
            except queue.Empty:
                continue
           

    