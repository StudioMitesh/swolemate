import mediapipe as mp
import cv2
import pandas as pd
import numpy as np

def extract_poses(video_path, rep_threshold=0.05):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    
    cap = cv2.VideoCapture(video_path)
    data = []
    wrist_y_values = []
    
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
            for idx in [11, 13, 15, 12, 14, 16]:
                l = results.pose_landmarks.landmark[idx]
                landmarks.extend([l.x, l.y, l.z, l.visibility])
            landmarks.append(frame_count)
            data.append(landmarks)

            wrist_y_values.append(landmarks[2 * 4 + 1])

        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames.")

    # Convert data to DataFrame
    cols = ["x11", "y11", "z11", "v11",
            "x13", "y13", "z13", "v13",
            "x15", "y15", "z15", "v15",
            "x12", "y12", "z12", "v12",
            "x14", "y14", "z14", "v14",
            "x16", "y16", "z16", "v16",
            "frame"]
    
    df = pd.DataFrame(data, columns=cols)

    wrist_y_values = np.array(wrist_y_values)
    min_y, max_y = np.min(wrist_y_values), np.max(wrist_y_values)
    threshold = (max_y - min_y) * rep_threshold

    rep_sequences = []
    rep_start = None

    for i in range(len(wrist_y_values) - 1):
        if wrist_y_values[i] - wrist_y_values[i + 1] > threshold:
            if rep_start is None:
                rep_start = i  # start rep
        elif wrist_y_values[i + 1] - wrist_y_values[i] > threshold and rep_start is not None:
            rep_sequences.append(df.iloc[rep_start:i].reset_index(drop=True))  # finish rep
            rep_start = None

    print(f"Detected {len(rep_sequences)} reps")
    return rep_sequences