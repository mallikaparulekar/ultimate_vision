import os
import cv2
import csv
import argparse
import numpy as np
from tqdm import tqdm


def manual_annotate(frames_dir, output_dir, csv_path):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for fname in sorted(os.listdir(frames_dir)):
        img_path = os.path.join(frames_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        clone = img.copy()

        coords = []
        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                coords.append((x, y))
                cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Frame", clone)

        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", on_click)
        cv2.imshow("Frame", clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if not coords:
            raise RuntimeError(f"No click recorded for frame {fname}")
        x, y = coords[0]
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, img)
        rows.append([fname, x, y])

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'x_center', 'y_center'])
        writer.writerows(rows)
    print(f"Manual annotation complete. Results saved to {csv_path}")


def detect_center(frame, prev_center=None, dp=1.2, minDist=30, param1=50, param2=30, minR=5, maxR=30):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
        param1=param1, param2=param2, minRadius=minR, maxRadius=maxR
    )
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype(int)
    if prev_center is not None:
        # pick closest to previous
        dists = [np.hypot(c[0]-prev_center[0], c[1]-prev_center[1]) for c in circles]
        idx = int(np.argmin(dists))
    else:
        idx = 0
    cx, cy, _ = circles[idx]
    return (cx, cy)


def auto_annotate(frames_dir, output_dir, csv_path, tracker_type='csrt', method='tracker'):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    if not frame_files:
        raise RuntimeError("No image files found in frames_dir")

    # Prepare first frame
    first = frame_files[0]
    first_path = os.path.join(frames_dir, first)
    frame0 = cv2.imread(first_path)
    if frame0 is None:
        raise RuntimeError("Failed to read first frame")

    prev_center = None

    if method == 'tracker':
        # User selects initial bounding box
        bbox = cv2.selectROI("Init - select frisbee region", frame0, fromCenter=False)
        cv2.destroyAllWindows()
        tracker = {
            'csrt': cv2.TrackerCSRT_create,
            'kcf': cv2.TrackerKCF_create,
            'mosse': cv2.TrackerMOSSE_create
        }[tracker_type]()
        ok = tracker.init(frame0, bbox)
        if not ok:
            raise RuntimeError("Tracker initialization failed")
        x, y, w, h = bbox
        prev_center = (int(x + w/2), int(y + h/2))
        cv2.circle(frame0, prev_center, 5, (0,255,0), -1)
    else:
        # detector-based: detect on first frame
        c = detect_center(frame0)
        if c is None:
            raise RuntimeError("Detector failed to find frisbee in first frame")
        prev_center = c
        cv2.circle(frame0, prev_center, 5, (255,0,0), -1)

    # save first
    out0 = os.path.join(output_dir, first)
    cv2.imwrite(out0, frame0)
    rows.append([first, prev_center[0], prev_center[1]])

    # Process remaining frames
    for fname in tqdm(frame_files[1:], desc="Annotating frames"):
        path = os.path.join(frames_dir, fname)
        frame = cv2.imread(path)
        if frame is None:
            continue

        if method == 'tracker':
            ok, bbox = tracker.update(frame)
            if not ok:
                print(f"Tracker failed on frame {fname}, stopping.")
                break
            x, y, w, h = bbox
            cx, cy = int(x + w/2), int(y + h/2)
        else:
            det = detect_center(frame, prev_center)
            if det is None:
                print(f"Detector failed on {fname}, skipping.")
                continue
            cx, cy = det

        prev_center = (cx, cy)
        color = (0,255,0) if method=='tracker' else (255,0,0)
        cv2.circle(frame, (cx, cy), 5, color, -1)

        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, frame)
        rows.append([fname, cx, cy])

    # save CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'x_center', 'y_center'])
        writer.writerows(rows)
    print(f"Automatic annotation ({method}) complete. Results saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate frisbee centers in frames: manual or automatic (tracker/detector)."
    )
    parser.add_argument("--frames_dir", required=True,
                        help="Directory containing extracted video frames.")
    parser.add_argument("--mode", choices=["manual","auto"], default="auto",
                        help="Annotation mode: manual clicking or automatic.")
    parser.add_argument("--tracker", choices=["csrt","kcf","mosse"], default="csrt",
                        help="Tracker type for automatic mode.")
    parser.add_argument("--method", choices=["tracker","detector"], default="tracker",
                        help="Auto annotation method: tracker or detector.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save annotated frames.")
    parser.add_argument("--csv", required=True,
                        help="Path to save CSV of centers.")
    args = parser.parse_args()

    if args.mode == "manual":
        manual_annotate(args.frames_dir, args.output_dir, args.csv)
    else:
        auto_annotate(
            args.frames_dir,
            args.output_dir,
            args.csv,
            tracker_type=args.tracker,
            method=args.method
        )  