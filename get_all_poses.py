# NOT BEING USED

import os
from pose_detector import extract_poses
import pandas as pd

def process_videos_in_folder(folder, label, output_csv):
    all_data = []

    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".mp4") or filename.endswith(".mov"):
            video_path = os.path.join(folder, filename)
            print(f"Processing: {video_path}")
            df = extract_poses(video_path)
            all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Saved dataset to {output_csv}")

    return final_df


good_data = process_videos_in_folder("pose_data/good_shoulder_press", "good_shoulder_press", "good_shoulder_press.csv")
bad_data = process_videos_in_folder("pose_data/bad_shoulder_press", "bad_shoulder_press", "bad_shoulder_press.csv")

print(good_data.head())
print(bad_data.head())