import mediapipe as mp
import cv2
import pandas as pd
import numpy as np

def extract_poses(video_path, rep_threshold=0.6, min_rep_duration=15, smoothing_window=5):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    
    cap = cv2.VideoCapture(video_path)
    data = []
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(cv2.resize(frame, (1000, 1000)), cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = []
            for idx in [11, 13, 15, 12, 14, 16]:  # 6 key points
                l = results.pose_landmarks.landmark[idx]
                landmarks.extend([l.x, l.y, l.z, l.visibility])
            landmarks.append(frame_count)
            data.append(landmarks)

        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames.")

    if not data:
        print("No pose data detected in video")
        return []

    cols = ["x11", "y11", "z11", "v11",
            "x13", "y13", "z13", "v13",
            "x15", "y15", "z15", "v15",
            "x12", "y12", "z12", "v12",
            "x14", "y14", "z14", "v14",
            "x16", "y16", "z16", "v16",
            "frame"]
    
    df = pd.DataFrame(data, columns=cols)

    # Focus on wrist movement for shoulder press
    wrist_cols = ['y15', 'y16']  # Left and right wrist y-coordinates
    wrist_data = df[wrist_cols]
    
    # Smooth the data
    smoothed_data = wrist_data.rolling(window=smoothing_window, center=True).mean()
    
    # Calculate vertical movement
    vertical_movement = smoothed_data.diff().abs().mean(axis=1)
    
    # Normalize the movement
    max_movement = vertical_movement.max()
    normalized_movement = vertical_movement / max_movement if max_movement > 0 else vertical_movement
    
    # Detect reps using peak detection
    rep_sequences = []
    in_rep = False
    rep_start = 0
    last_peak = 0
    
    for i in range(1, len(normalized_movement) - 1):
        current = normalized_movement.iloc[i]
        
        # Start of rep (moving up)
        if not in_rep and current > rep_threshold:
            in_rep = True
            rep_start = i
        
        # End of rep (completed movement)
        elif in_rep and i - rep_start >= min_rep_duration:
            # Check if we've moved back down
            if current < rep_threshold:
                rep_sequences.append(df.iloc[rep_start:i].reset_index(drop=True))
                in_rep = False
                last_peak = i
    
    print(f"Detected {len(rep_sequences)} reps")
    return rep_sequences
