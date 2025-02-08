import mediapipe as mp
import cv2
import pandas as pd

def extract_poses(video, label):
    mp_poses = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_poses.Pose(static_image_mode=False)
    cap = cv2.VideoCapture(video)
    data = []
    frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (1000,1000))
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = []
            for l in results.pose_landmarks.landmark:
                landmarks.extend([l.x, l.y, l.z, l.visibility])
            landmarks.append(frame_count)
            landmarks.append(label)
            
            data.append(landmarks)

        frame_count += 1

    cap.release()

    print(f"Processed {frame_count} frames.")


    cols = [f"x{i}" for i in range(33)] + [f"y{i}" for i in range(33)] + [f"z{i}" for i in range(33)] + [f"v{i}" for i in range(33)]
    cols.append("frame")
    cols.append("label")
    
    df = pd.DataFrame(data, columns=cols)
    return df
