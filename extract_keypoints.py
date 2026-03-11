"""
Keypoint Extraction Pipeline
APS360 Project — Pose Data Preparation

Reads labels.csv produced by labelling_tool.py, runs YOLO pose estimation on
the FULL video frame (not the crop) for best detection quality, identifies the
setter using the labelled position, and tracks them via bounding-box IoU.

Keypoints are normalised to [0,1] relative to the crop region so the model
sees position-invariant setter motion regardless of where the setter appeared
in the original video.

Usage:
    python extract_keypoints.py
    python extract_keypoints.py --videos-dir /path/to/videos
    python extract_keypoints.py --labels clips/labels.csv --output dataset.npz

Output (.npz contents):
    keypoints   — (N, T, 17, 3)  float32   x,y normalised [0,1], confidence
    labels      — (N,)           int64      encoded direction index
    label_names — (C,)           str        class names in index order
    clip_ids    — (N,)           str        clip IDs matching labels.csv
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


# ─── Constants ────────────────────────────────────────────────────────────────

COCO_KEYPOINT_NAMES = [
    'nose',
    'left_eye',       'right_eye',
    'left_ear',        'right_ear',
    'left_shoulder',   'right_shoulder',
    'left_elbow',      'right_elbow',
    'left_wrist',      'right_wrist',
    'left_hip',        'right_hip',
    'left_knee',       'right_knee',
    'left_ankle',      'right_ankle',
]

VISIBILITY_THRESHOLD = 0.5   # keypoint must exceed this to count as "visible"


# ─── Matching helpers ────────────────────────────────────────────────────────

def find_closest_person(kpts_data, target_point):
    """Return (index, center) of the detected person closest to *target_point*.

    Args:
        kpts_data : YOLO keypoints tensor  (num_people, 17, 3)
        target_point : np.ndarray [x, y]

    Returns:
        (person_idx, person_center_np)  or  (None, None)
    """
    best_idx = None
    best_center = None
    min_dist = float('inf')

    for idx, person_kpts in enumerate(kpts_data):
        visible = person_kpts[person_kpts[:, 2] > VISIBILITY_THRESHOLD]
        if len(visible) == 0:
            continue
        center = visible[:, :2].mean(axis=0).cpu().numpy()
        dist = np.linalg.norm(center - target_point)
        if dist < min_dist:
            min_dist = dist
            best_idx = idx
            best_center = center

    return best_idx, best_center


def box_iou(a, b):
    """IoU between two [x1, y1, x2, y2] boxes (numpy arrays)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_by_iou(boxes, prev_box):
    """Return the index of the box with the highest IoU to *prev_box*.

    Returns (best_idx, best_iou) or (None, 0) if no overlap.
    """
    best_idx = None
    best_iou = 0.0
    for idx, box in enumerate(boxes):
        iou = box_iou(box, prev_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    return best_idx, best_iou


IOU_THRESHOLD = 0.15   # minimum overlap to accept a match


# ─── Per-clip extraction ─────────────────────────────────────────────────────

def extract_clip_keypoints(model, video_path,
                           start_frame, end_frame,
                           crop_x, crop_y, crop_w, crop_h,
                           setter_x, setter_y):
    """Run YOLO pose on the FULL frame and return the setter's keypoints.

    YOLO runs on the full-resolution frame (not the crop) so it gets the best
    possible image quality for detection.  The setter is identified on the
    first frame using (setter_x, setter_y) converted to full-frame coordinates.
    On subsequent frames the setter is tracked by bounding-box IoU.

    Keypoints are then normalised to [0, 1] relative to the crop region so the
    output is position-invariant.

    Returns:
        np.ndarray of shape (num_frames, 17, 3)  — x,y normalised to [0,1]
        relative to crop dimensions, confidence preserved.
        Returns None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ERROR: cannot open video {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    num_frames = end_frame - start_frame + 1
    keypoints = np.zeros((num_frames, 17, 3), dtype=np.float32)

    # Convert setter position from crop coords → full-frame coords
    target_point = np.array([crop_x + setter_x,
                             crop_y + setter_y], dtype=np.float32)
    prev_box = None             # [x1, y1, x2, y2] in full-frame coords
    frames_with_detection = 0

    for offset in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"    WARNING: failed to read frame {start_frame + offset}")
            break

        # Run YOLO pose on the FULL frame — much better detection quality
        results = model(frame, verbose=False)
        result = results[0]
        kpts_data = result.keypoints.data   # (num_people, 17, 3)  full-frame coords
        boxes = result.boxes.xyxy.cpu().numpy()  # (num_people, 4)  full-frame coords

        if len(kpts_data) == 0:
            continue  # leave zeros for this frame

        if prev_box is None:
            # First detection: find person closest to the labelled setter
            # position (in full-frame coords), lock onto their bounding box.
            person_idx, _ = find_closest_person(kpts_data, target_point)
            if person_idx is None:
                continue
        else:
            # Subsequent frames: match by bounding-box overlap
            person_idx, iou = match_by_iou(boxes, prev_box)
            if person_idx is None or iou < IOU_THRESHOLD:
                # No good overlap — skip this frame, keep prev_box unchanged
                # so the next frame still searches near the setter's last
                # known position.
                continue

        # Update tracked bounding box
        prev_box = boxes[person_idx]

        person_kpts = kpts_data[person_idx].cpu().numpy()   # (17, 3)

        # Shift keypoints to crop-relative coords, then normalise to [0, 1]
        person_kpts[:, 0] = (person_kpts[:, 0] - crop_x) / crop_w
        person_kpts[:, 1] = (person_kpts[:, 1] - crop_y) / crop_h

        keypoints[offset] = person_kpts
        frames_with_detection += 1

    cap.release()

    if frames_with_detection == 0:
        print("    WARNING: no person detected in any frame")
        return None

    pct = frames_with_detection / num_frames * 100
    print(f"    tracked setter in "
          f"{frames_with_detection}/{num_frames} frames ({pct:.0f}%)")

    return keypoints


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract setter pose keypoints from labelled volleyball clips.")
    parser.add_argument(
        "--labels", default=None,
        help="Path to labels.csv  (default: clips/labels.csv in script dir)")
    parser.add_argument(
        "--videos-dir", default=None,
        help="Directory containing source videos  (default: script dir)")
    parser.add_argument(
        "--model", default="yolo26n-pose.pt",
        help="YOLO pose model path  (default: yolo26n-pose.pt in script dir)")
    parser.add_argument(
        "--output", default=None,
        help="Output .npz file  (default: dataset.npz in script dir)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    labels_path = args.labels or os.path.join(script_dir, "clips", "labels.csv")
    videos_dir  = args.videos_dir or script_dir
    model_path  = (args.model if os.path.isabs(args.model)
                   else os.path.join(script_dir, args.model))
    output_path = args.output or os.path.join(script_dir, "dataset.npz")

    # ── Load YOLO model ──
    print(f"Loading model : {model_path}")
    model = YOLO(model_path)

    # ── Read labels ──
    print(f"Reading labels: {labels_path}\n")

    if not os.path.exists(labels_path):
        print(f"ERROR: labels file not found — {labels_path}")
        sys.exit(1)

    clips = []
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("setter_x") or not row.get("setter_y"):
                print(f"  SKIP {row['clip_id']}: no setter position (old format?)")
                continue
            clips.append(row)

    print(f"Found {len(clips)} clip(s) with setter positions.\n")
    if not clips:
        print("Nothing to process — label some clips first.")
        sys.exit(0)

    # ── Process each clip ──
    all_keypoints = []
    all_labels    = []
    all_clip_ids  = []
    skipped = 0

    for i, clip in enumerate(clips):
        clip_id     = clip["clip_id"]
        source      = clip["source_video"]
        direction   = clip["set_direction"]
        start_frame = int(clip["start_frame"])
        end_frame   = int(clip["end_frame"])
        crop_x      = int(clip["crop_x"])
        crop_y      = int(clip["crop_y"])
        crop_w      = int(clip["crop_w"])
        crop_h      = int(clip["crop_h"])
        setter_x    = float(clip["setter_x"])
        setter_y    = float(clip["setter_y"])

        # Locate source video
        video_path = os.path.join(videos_dir, source)
        if not os.path.exists(video_path):
            # Fall back: maybe the file is beside the labels CSV
            video_path = os.path.join(os.path.dirname(labels_path), source)
        if not os.path.exists(video_path):
            print(f"  [{i+1}/{len(clips)}] SKIP {clip_id}: "
                  f"video not found — {source}")
            skipped += 1
            continue

        num_frames = end_frame - start_frame + 1
        print(f"  [{i+1}/{len(clips)}] {clip_id}  "
              f"{direction:>12s}  "
              f"frames {start_frame}–{end_frame} ({num_frames}f)  "
              f"crop {crop_w}×{crop_h}  "
              f"setter ({setter_x:.0f}, {setter_y:.0f})")

        kp = extract_clip_keypoints(
            model, video_path,
            start_frame, end_frame,
            crop_x, crop_y, crop_w, crop_h,
            setter_x, setter_y,
        )

        if kp is not None:
            all_keypoints.append(kp)
            all_labels.append(direction)
            all_clip_ids.append(clip_id)
        else:
            skipped += 1

    # ── Summarise & save ──
    if not all_keypoints:
        print("\nNo keypoints extracted — check video paths and labels.")
        sys.exit(1)

    # Encode labels
    unique_labels = sorted(set(all_labels))
    label_to_idx  = {name: idx for idx, name in enumerate(unique_labels)}
    label_indices = np.array([label_to_idx[l] for l in all_labels], dtype=np.int64)

    # Stack keypoints — all same length → (N, T, 17, 3), else object array
    lengths = [kp.shape[0] for kp in all_keypoints]
    if len(set(lengths)) == 1:
        keypoints_array = np.stack(all_keypoints)          # (N, T, 17, 3)
    else:
        keypoints_array = np.array(all_keypoints, dtype=object)

    np.savez(
        output_path,
        keypoints=keypoints_array,
        labels=label_indices,
        label_names=np.array(unique_labels),
        clip_ids=np.array(all_clip_ids),
    )

    # ── Report ──
    print(f"\n{'═' * 60}")
    print(f"  Processed : {len(all_keypoints)} clips")
    print(f"  Skipped   : {skipped}")
    print(f"  Classes   : {label_to_idx}")
    counts = {name: all_labels.count(name) for name in unique_labels}
    print(f"  Counts    : {counts}")
    if isinstance(keypoints_array, np.ndarray) and keypoints_array.dtype != object:
        print(f"  Tensor    : {keypoints_array.shape}  "
              f"(N, frames, 17 keypoints, [x_norm, y_norm, conf])")
    else:
        print(f"  Lengths   : {min(lengths)}–{max(lengths)} frames (variable)")
    print(f"\n  Saved → {output_path}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
