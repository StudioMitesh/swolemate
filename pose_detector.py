import mediapipe as mp
import cv2
import pandas as pd
import os

mp_poses = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_poses.Pose(static_image_mode=False)

def extract_poses(video, label):
    cap = cv2.VideoCapture(video)
    data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        if results.pose_landmarks:
            landmarks = []
            for l in results.pose_landmarks.landmark:
                landmarks.extend([l.x, l.y, l.z, l.visibility])
            landmarks.append(frame_count)
            landmarks.append(label)
            
            data.append(landmarks)

        frame_count += 1

    cap.release()

    cols = [f"x{i}" for i in range(33)] + [f"y{i}" for i in range(33)] + [f"z{i}" for i in range(33)] + [f"v{i}" for i in range(33)]
    cols.append("frame")
    cols.append("label")
    
    df = pd.DataFrame(data, columns=cols)
    return df

def process_videos_in_folder(folder, label, output_csv):
    all_data = []

    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".mp4") or filename.endswith(".mov"):
            video_path = os.path.join(folder, filename)
            print(f"Processing: {video_path}")
            df = extract_poses(video_path, label)
            all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Saved dataset to {output_csv}")

    return final_df


good_data = process_videos_in_folder("pose_data/good_shoulder_press", "good_shoulder_press", "good_shoulder_press.csv")
bad_data = process_videos_in_folder("pose_data/bad_shoulder_press", "bad_shoulder_press", "bad_shoulder_press.csv")

print(good_data.head())
print(bad_data.head())
