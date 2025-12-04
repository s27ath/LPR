import argparse, os
import cv2
from ultralytics import YOLO

def main(args):
    # Load model (fine-tuned on CCPD)
    model = YOLO(args.weights)

    # Run inference
    results = model.predict(
        source=args.image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        verbose=False
    )
    r = results[0]

    # Create output paths
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_img = os.path.join(args.out_dir, f"{stem}_annotated.jpg")
    crops_dir = os.path.join(args.out_dir, f"{stem}_crops")
    os.makedirs(crops_dir, exist_ok=True)

    # Save annotated image
    annotated = r.plot()  # BGR
    cv2.imwrite(out_img, annotated)

    # Save plate crops (sorted by confidence)
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes
        confs = boxes.conf.cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        img = cv2.imread(args.image)

        # sort by confidence (desc) and keep max_det crops
        order = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)[:args.max_det]
        for rank, i in enumerate(order, start=1):
            x1, y1, x2, y2 = xyxy[i]
            crop = img[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
            cv2.imwrite(os.path.join(crops_dir, f"{rank:02d}_{confs[i]:.3f}.jpg"), crop)

        print(f"[OK] Saved annotated: {out_img}")
        print(f"[OK] Saved {len(order)} crop(s) to: {crops_dir}")
    else:
        print("[WARN] No plate detected. Try lowering --conf or increasing --imgsz.")
        print(f"[INFO] Annotated image (without boxes): {out_img}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="best.pt", help="path to fine-tuned weights")
    ap.add_argument("--image",   type=str, required=True,      help="path to input image (jpg/png)")
    ap.add_argument("--out_dir", type=str, default="outputs",  help="where to save results")
    ap.add_argument("--imgsz",   type=int, default=640,        help="inference size (try 960/1024 if far plates)")
    ap.add_argument("--conf",    type=float, default=0.25,     help="confidence threshold")
    ap.add_argument("--iou",     type=float, default=0.6,      help="NMS IoU threshold")
    ap.add_argument("--max_det", type=int, default=2,          help="max plates to keep")
    args = ap.parse_args()
    main(args)
