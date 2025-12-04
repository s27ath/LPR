# Web_cam.py
import cv2
from ultralytics import YOLO

MODEL_PATH = "best.pt"  # if not in this folder, put the full path here
CONF = 0.25
IMGSZ = 640

model = YOLO(MODEL_PATH)

# On macOS, AVFoundation backend is often the most reliable
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    # fallback to default backend if needed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions in System Settings > Privacy & Security > Camera.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.predict(source=frame, imgsz=IMGSZ, conf=CONF, iou=0.6, max_det=1, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("License-plate detector", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):  # ESC or q to quit
        break

cap.release()
cv2.destroyAllWindows()

