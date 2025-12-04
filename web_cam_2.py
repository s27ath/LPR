# web_cam_2.py
# Real-time LPR: YOLOv8 + EasyOCR (lightweight)
# - AVFoundation + MJPG @ 640x480 (macOS-friendly)
# - Run YOLO on downscaled frame; rescale boxes back
# - Single best detection per frame
# - OCR every N frames (cached text between)
# - Uses MPS (Apple Silicon) if available
# - Overlay only (no saving)

import time
import cv2
import numpy as np
from ultralytics import YOLO
from ocr_easy import PlateOCR

# -------- Tunables --------
CONF_TH       = 0.25     # YOLO confidence
IOU_TH        = 0.60     # YOLO NMS IoU
FRAME_W       = 640
FRAME_H       = 480
DET_SCALE     = 0.4      # YOLO on 50% size
OCR_EVERY_N   = 12        # OCR once per N frames (raise if still laggy)
FONT_SCALE    = 0.9
TEXT_THICK    = 2
BOX_THICK     = 2
# --------------------------

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2

def main():
    # Load YOLO
    det = YOLO("best.pt")

    # Try Apple Silicon GPU (MPS) if available
    try:
        import torch
        det.to("mps" if torch.backends.mps.is_available() else "cpu")
    except Exception:
        pass

    # Init EasyOCR wrapper
    ocr = PlateOCR(lang='en')

    # Open camera (AVFoundation on macOS)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found. On macOS, enable Camera access for your terminal app.")

    # Camera config
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    window = "LPR: YOLOv8 + EasyOCR   (press 'q' to quit)"
    last_text, last_conf, last_box = "", 0.0, None
    frame_id = 0
    prev_t, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]

        # Run YOLO on a smaller image
        small = cv2.resize(frame, (0, 0), fx=DET_SCALE, fy=DET_SCALE, interpolation=cv2.INTER_AREA)
        r0 = det(source=small, conf=CONF_TH, iou=IOU_TH, verbose=False,
                 imgsz=int(FRAME_W * DET_SCALE))[0]

        annotated = frame.copy()
        boxes = getattr(r0, "boxes", None)

        if boxes is not None and len(boxes) > 0:
            # Most confident detection
            confs = boxes.conf
            idx = int(confs.argmax().item()) if hasattr(confs, "argmax") else 0
            x1s, y1s, x2s, y2s = boxes.xyxy[idx].cpu().numpy().tolist()

            # Rescale to full-res coords
            x1 = int(x1s / DET_SCALE); y1 = int(y1s / DET_SCALE)
            x2 = int(x2s / DET_SCALE); y2 = int(y2s / DET_SCALE)
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICK)

            # OCR every N frames only
            if (frame_id % OCR_EVERY_N == 0) and (x2 - x1 >= 14 and y2 - y1 >= 14):
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    try:
                        text, conf = ocr.read(crop)
                        if text:
                            last_text, last_conf, last_box = text, conf, (x1, y1, x2, y2)
                    except Exception:
                        pass

            # Draw cached/fresh OCR
            if last_text and last_box:
                bx1, by1, bx2, by2 = last_box
                label = f"{last_text} ({last_conf:.2f})"
                cv2.putText(annotated, label, (bx1, max(by1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), TEXT_THICK)

        # FPS overlay (optional)
        now = time.time()
        dt = now - prev_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        prev_t = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(window, annotated)
        frame_id += 1
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

