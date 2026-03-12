import cv2
import numpy as np
import csv
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")

# COCO pose keypoint names (in order)
COCO_KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

# Video file path
video_path = '/Users/yiyixu/Documents/School/Third Year/Winter/APS360/project/clips/76f4c1961f.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Tracking method: 'position' or 'tracker'
# 'position': Track person closest to a specific position (e.g., center of frame)
# 'tracker': Use YOLO's built-in tracker (assigns IDs, tracks same person)
tracking_method = 'position'  # Change to 'tracker' to use YOLO tracking

# For position-based tracking: track person closest to this point
# Options: 'center' (center of frame), 'left', 'right', or [x, y] coordinates
target_position = 'right'  # Change this to select which person to track

# For tracker-based: the person ID to track (will be set after first frame)
target_person_id = None

# Store previous frame's person position for continuity (position-based)
prev_person_center = None
prev_person_idx = None

# List to store all keypoint data for CSV export
keypoint_data = []

# Frame counter for CSV export (doesn't reset on video loop)
csv_frame_counter = 0

# Cache to store processed frames (key: video frame position, value: annotated frame)
frame_cache = {}

# Track if we've completed the first pass through the video
first_pass_complete = False

# Loop through the video frames (will loop continuously)
frame_count = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        
        # Get current video frame position
        video_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Check if we've already processed this frame (video replay)
        # Only use cache if we've completed first pass and frame is cached
        if first_pass_complete and video_frame_pos in frame_cache:
            # Use cached frame, skip inference
            annotated_frame = frame_cache[video_frame_pos]
        else:
            # New frame - run inference
            # Run YOLO pose estimation with tracking if using tracker method
            if tracking_method == 'tracker':
                results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            else:
                results = model(frame)
            
            result = results[0]
            kpts_data = result.keypoints.data  # (num_people, num_keypoints, 3) -> (x, y, v)
            
            if len(kpts_data) > 0:
                target_idx = None
                
                if tracking_method == 'tracker':
                    # Use YOLO's tracking IDs
                    if hasattr(result, 'boxes') and result.boxes.id is not None:
                        person_ids = result.boxes.id.cpu().numpy().astype(int)
                        
                        # On first frame, select the person closest to target position
                        if target_person_id is None:
                            if target_position == 'center':
                                target_point = np.array([frame_width / 2, frame_height / 2])
                            elif target_position == 'left':
                                target_point = np.array([frame_width * 0.25, frame_height / 2])
                            elif target_position == 'right':
                                target_point = np.array([frame_width * 0.75, frame_height / 2])
                            else:
                                target_point = np.array(target_position)
                            
                            # Find person closest to target point
                            min_dist = float('inf')
                            for idx, person_kpts in enumerate(kpts_data):
                                # Use center of person (average of visible keypoints)
                                visible_kpts = person_kpts[person_kpts[:, 2] > 0.5]
                                if len(visible_kpts) > 0:
                                    person_center = visible_kpts[:, :2].mean(axis=0).cpu().numpy()
                                    dist = np.linalg.norm(person_center - target_point)
                                    if dist < min_dist:
                                        min_dist = dist
                                        target_person_id = person_ids[idx]
                        
                        # Find the person with the target ID
                        if target_person_id in person_ids:
                            target_idx = np.where(person_ids == target_person_id)[0][0]
                        else:
                            # Person lost, try to reacquire by position
                            target_person_id = None
                    else:
                        # No tracking IDs available, fall back to position-based
                        tracking_method = 'position'
                
                if tracking_method == 'position' or target_idx is None:
                    # Position-based tracking: find person closest to target position
                    if target_position == 'center':
                        target_point = np.array([frame_width / 2, frame_height / 2])
                    elif target_position == 'left':
                        target_point = np.array([frame_width * 0.25, frame_height / 2])
                    elif target_position == 'right':
                        target_point = np.array([frame_width * 0.75, frame_height / 2])
                    else:
                        target_point = np.array(target_position)
                    
                    # If we have previous position, use it for continuity
                    if prev_person_center is not None:
                        target_point = prev_person_center
                    
                    # Find person closest to target point
                    min_dist = float('inf')
                    for idx, person_kpts in enumerate(kpts_data):
                        # Use center of person (average of visible keypoints)
                        visible_kpts = person_kpts[person_kpts[:, 2] > 0.5]
                        if len(visible_kpts) > 0:
                            person_center = visible_kpts[:, :2].mean(axis=0).cpu().numpy()
                            dist = np.linalg.norm(person_center - target_point)
                            if dist < min_dist:
                                min_dist = dist
                                target_idx = idx
                                prev_person_center = person_center
                                prev_person_idx = idx
                
                # Filter keypoints to only include the target person
                if target_idx is not None:
                    target_kpts = kpts_data[target_idx:target_idx+1]  # Keep shape (1, num_keypoints, 3)
                    
                    # Store keypoint data for CSV export (only on first pass)
                    if not first_pass_complete:
                        person_kpts = target_kpts[0].cpu().numpy()  # Shape: (17, 3) -> (x, y, visibility)
                        for kpt_idx, (x, y, visibility) in enumerate(person_kpts):
                            keypoint_data.append({
                                'frame': csv_frame_counter,
                                'keypoint_index': kpt_idx,
                                'keypoint_name': COCO_KEYPOINT_NAMES[kpt_idx],
                                'x': float(x),
                                'y': float(y),
                                'visibility': float(visibility)
                            })
                        csv_frame_counter += 1  # Increment once per frame
                    
                    # Temporarily replace keypoints with only the target person
                    original_kpts = result.keypoints.data
                    result.keypoints.data = target_kpts
                    
                    # Visualize the results on the frame (overlays COCO keypoints and skeleton)
                    annotated_frame = result.plot(conf=False, boxes=False)
                    
                    # Restore original keypoints
                    result.keypoints.data = original_kpts
                else:
                    # No target person found, show frame without keypoints
                    annotated_frame = frame
            else:
                # No people detected, just show the frame
                annotated_frame = frame
                prev_person_center = None
            
            # Cache the processed frame
            frame_cache[video_frame_pos] = annotated_frame.copy()

        # Display the annotated frame
        cv2.imshow("Pose Estimation - Tracked Person (Press 'q' to quit)", annotated_frame)

        # Break the loop if 'q' is pressed or window is closed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty("Pose Estimation - Tracked Person (Press 'q' to quit)", cv2.WND_PROP_VISIBLE) < 1:
            break
    else:
        # End of video reached - mark first pass as complete
        if not first_pass_complete:
            first_pass_complete = True
            print(f"\nFirst pass complete. Processed {len(frame_cache)} frames. Video will loop using cached frames.")
        
        # Reset to beginning to loop
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        target_person_id = None  # Reset tracking
        prev_person_center = None

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# Export keypoint data to CSV
if keypoint_data:
    csv_filename = '/Users/yiyixu/Documents/School/Third Year/Winter/APS360/project/tracked_person_keypoints.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'keypoint_index', 'keypoint_name', 'x', 'y', 'visibility']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in keypoint_data:
            writer.writerow(row)
    
    print(f"\nExported {len(keypoint_data)} keypoint records to {csv_filename}")
    print(f"Total frames with tracked person: {max([row['frame'] for row in keypoint_data]) + 1}")
else:
    print("\nNo keypoint data collected.")