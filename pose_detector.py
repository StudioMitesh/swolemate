import mediapipe as mp
import cv2
import pandas as pd
import numpy as np

def extract_poses(video_path, rep_threshold=0.05, min_rep_duration=10, smoothing_window=5):
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

    cols = ["x11", "y11", "z11", "v11",
            "x13", "y13", "z13", "v13",
            "x15", "y15", "z15", "v15",
            "x12", "y12", "z12", "v12",
            "x14", "y14", "z14", "v14",
            "x16", "y16", "z16", "v16",
            "frame"]
    
    df = pd.DataFrame(data, columns=cols)

    smoothed_data = df.drop(columns=['frame'])
    smoothed_data = smoothed_data.rolling(window=smoothing_window).mean().dropna()
    
    displacement = []
    for i in range(1, len(smoothed_data)):
        prev_row = smoothed_data.iloc[i - 1]
        current_row = smoothed_data.iloc[i]
        dist = np.linalg.norm(current_row.values - prev_row.values)
        displacement.append(dist)
    
    displacement = np.array(displacement)
    min_displacement, max_displacement = np.min(displacement), np.max(displacement)
    threshold = (max_displacement - min_displacement) * rep_threshold
    
    rep_sequences = []
    rep_start = None
    rep_frame_count = 0

    for i in range(len(displacement) - 1):
        if displacement[i] > threshold:
            if rep_start is None:
                rep_start = i
                rep_frame_count = 1
        elif displacement[i] < threshold and rep_start is not None:
            rep_frame_count += 1
            if rep_frame_count >= min_rep_duration:  
                rep_sequences.append(df.iloc[rep_start:i + 1].reset_index(drop=True))
                rep_start = None 
                rep_frame_count = 0  

    print(f"Detected {len(rep_sequences)} reps")
    return rep_sequences
