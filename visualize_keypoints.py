"""
Keypoint Visualization
APS360 Project — Visual QA for extracted pose data

Plays each labelled clip (cropped, looping) with the extracted keypoints and
skeleton drawn on top, so you can verify the pose estimation tracked the
setter correctly.

Usage:
    python visualize_keypoints.py
    python visualize_keypoints.py --dataset dataset.npz --clip 0
    python visualize_keypoints.py --clip 2

Controls:
    N / Right  — next clip
    P / Left   — previous clip
    Q / Esc    — quit
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np


# ─── COCO-17 skeleton definition (same pairs YOLO draws) ─────────────────────

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6),                                    # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12),                          # torso
    (11, 12),                                  # hips
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]

# Limb colours (BGR) — left side blue-ish, right side orange-ish, centre green
LIMB_COLORS = [
    (255, 128, 0),   # nose→left_eye
    (0, 128, 255),   # nose→right_eye
    (255, 128, 0),   # left_eye→left_ear
    (0, 128, 255),   # right_eye→right_ear
    (0, 255, 0),     # left_shoulder→right_shoulder
    (255, 128, 0),   # left_shoulder→left_elbow
    (255, 128, 0),   # left_elbow→left_wrist
    (0, 128, 255),   # right_shoulder→right_elbow
    (0, 128, 255),   # right_elbow→right_wrist
    (255, 128, 0),   # left_shoulder→left_hip
    (0, 128, 255),   # right_shoulder→right_hip
    (0, 255, 0),     # left_hip→right_hip
    (255, 128, 0),   # left_hip→left_knee
    (255, 128, 0),   # left_knee→left_ankle
    (0, 128, 255),   # right_hip→right_knee
    (0, 128, 255),   # right_knee→right_ankle
]

KEYPOINT_COLOR = (0, 255, 255)   # cyan dots
CONF_THRESHOLD = 0.3             # don't draw low-confidence points


# ─── Drawing helpers ─────────────────────────────────────────────────────────

def draw_skeleton(frame, kpts, crop_w, crop_h):
    """Draw COCO skeleton and keypoints on a frame.

    Args:
        frame  : BGR image (will be modified in-place)
        kpts   : (17, 3) array — normalised x, y, confidence
        crop_w : original crop width  (for denormalisation)
        crop_h : original crop height (for denormalisation)
    """
    # Denormalise to pixel coordinates
    points = []
    for kx, ky, conf in kpts:
        px = int(kx * crop_w)
        py = int(ky * crop_h)
        points.append((px, py, conf))

    # Draw limbs
    for i, (a, b) in enumerate(COCO_SKELETON):
        ax, ay, ac = points[a]
        bx, by, bc = points[b]
        if ac < CONF_THRESHOLD or bc < CONF_THRESHOLD:
            continue
        color = LIMB_COLORS[i % len(LIMB_COLORS)]
        cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)

    # Draw keypoints on top
    for px, py, conf in points:
        if conf < CONF_THRESHOLD:
            continue
        cv2.circle(frame, (px, py), 4, KEYPOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, (0, 0, 0), 1, cv2.LINE_AA)  # outline


# ─── Load clip frames from source video ──────────────────────────────────────

def load_clip_frames(video_path, start_frame, end_frame,
                     crop_x, crop_y, crop_w, crop_h):
    """Read and crop frames for a clip from the source video.

    Returns:
        list of BGR numpy frames, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open video {video_path}")
        return None, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        frames.append(cropped)

    cap.release()
    return frames, fps


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize extracted setter keypoints on clip video.")
    parser.add_argument(
        "--dataset", default=None,
        help="Path to dataset.npz  (default: dataset.npz in script dir)")
    parser.add_argument(
        "--labels", default=None,
        help="Path to labels.csv  (default: clips/labels.csv in script dir)")
    parser.add_argument(
        "--videos-dir", default=None,
        help="Directory containing source videos  (default: script dir)")
    parser.add_argument(
        "--clip", type=int, default=None,
        help="Index of clip to start at (0-based, default: 0)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_path = args.dataset or os.path.join(script_dir, "dataset.npz")
    labels_path  = args.labels or os.path.join(script_dir, "clips", "labels.csv")
    videos_dir   = args.videos_dir or script_dir

    # ── Load dataset ──
    if not os.path.exists(dataset_path):
        print(f"ERROR: dataset not found — {dataset_path}")
        print("Run  python extract_keypoints.py  first.")
        sys.exit(1)

    data = np.load(dataset_path, allow_pickle=True)
    all_keypoints = data["keypoints"]     # (N, T, 17, 3) or object array
    all_labels    = data["labels"]         # (N,) int
    label_names   = data["label_names"]    # (C,) str
    clip_ids      = data["clip_ids"]       # (N,) str

    n_clips = len(clip_ids)
    print(f"Loaded {n_clips} clips from {dataset_path}")
    print(f"Classes: {list(label_names)}\n")

    # ── Load labels CSV for video/crop metadata ──
    csv_rows = {}
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows[row["clip_id"]] = row

    # ── Display loop ──
    clip_idx = args.clip if args.clip is not None else 0
    clip_idx = max(0, min(clip_idx, n_clips - 1))

    window_name = "Keypoint Visualization (N=next, P=prev, Q=quit)"

    while True:
        cid = str(clip_ids[clip_idx])
        kp = all_keypoints[clip_idx]          # (T, 17, 3)
        label_idx = int(all_labels[clip_idx])
        direction = str(label_names[label_idx])

        if cid not in csv_rows:
            print(f"  SKIP clip {clip_idx} ({cid}): not found in labels.csv")
            clip_idx = (clip_idx + 1) % n_clips
            continue

        row = csv_rows[cid]
        source      = row["source_video"]
        start_frame = int(row["start_frame"])
        end_frame   = int(row["end_frame"])
        crop_x      = int(row["crop_x"])
        crop_y      = int(row["crop_y"])
        crop_w      = int(row["crop_w"])
        crop_h      = int(row["crop_h"])

        # Find video
        video_path = os.path.join(videos_dir, source)
        if not os.path.exists(video_path):
            video_path = os.path.join(os.path.dirname(labels_path), source)
        if not os.path.exists(video_path):
            print(f"  SKIP clip {clip_idx} ({cid}): video not found — {source}")
            clip_idx = (clip_idx + 1) % n_clips
            continue

        print(f"  Clip {clip_idx}/{n_clips - 1}:  {cid}  |  {direction}  |  "
              f"frames {start_frame}–{end_frame}  |  crop {crop_w}×{crop_h}")

        frames, fps = load_clip_frames(
            video_path, start_frame, end_frame,
            crop_x, crop_y, crop_w, crop_h)

        if not frames:
            print("    No frames loaded, skipping.")
            clip_idx = (clip_idx + 1) % n_clips
            continue

        # Scale up small crops for better visibility
        min_display = 400
        scale = 1.0
        if crop_w < min_display and crop_h < min_display:
            scale = min_display / min(crop_w, crop_h)
        disp_w = int(crop_w * scale)
        disp_h = int(crop_h * scale)

        delay = max(1, int(1000 / fps))
        frame_i = 0
        advance = False

        while not advance:
            # Get frame and keypoints for this timestep
            f_idx = frame_i % len(frames)
            display = frames[f_idx].copy()

            # Draw keypoints if we have them for this frame
            if f_idx < len(kp):
                draw_skeleton(display, kp[f_idx], crop_w, crop_h)

            # Scale up for display
            if scale != 1.0:
                display = cv2.resize(display, (disp_w, disp_h),
                                     interpolation=cv2.INTER_LINEAR)

            # Text overlay: clip info
            text = f"[{clip_idx}/{n_clips-1}] {cid}  |  {direction}"
            cv2.putText(display, text, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, text, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

            frame_text = f"frame {f_idx+1}/{len(frames)}"
            cv2.putText(display, frame_text, (8, disp_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, frame_text, (8, disp_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(delay) & 0xFF
            if key in (ord('q'), 27):       # Q or Esc
                cv2.destroyAllWindows()
                print("\nDone.")
                return
            elif key in (ord('n'), 83, 3):  # N or Right arrow
                clip_idx = (clip_idx + 1) % n_clips
                advance = True
            elif key in (ord('p'), 81, 2):  # P or Left arrow
                clip_idx = (clip_idx - 1) % n_clips
                advance = True

            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return

            frame_i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
