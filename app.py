from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import io
import tempfile
import os
from tensorflow.keras.models import load_model
from pose_detector import extract_poses

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model('shoulder_press_model.keras')

def process_video(video_path):
    df = extract_poses(video_path, label=None)
    if 'frame' in df.columns:
        df = df.drop(columns=["frame"])
    X = df.drop(columns=["label"]).values
    sequence_length = 60
    num_features = X.shape[1]
    num_samples = X.shape[0] // sequence_length
    if num_samples < 66:
        print(f"Not enough frames. Only {num_samples} sequences available.")
        return np.zeros((66, sequence_length, num_features)) 
    X = X[:num_samples * sequence_length]
    X = X.reshape(num_samples, sequence_length, num_features)
    print("reshaped:", X.shape)
    return X

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try: 
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        video_file.save(f'./uploads/{video_file.filename}')
        video_path = f'./uploads/{video_file.filename}'
        print(f"Video saved at: {video_path}")

        frames = process_video(video_path)
        print("Frames shape:", frames.shape)
        
        predictions = model.predict(frames)
        predicted_class = np.argmax(predictions, axis=1)
        
        os.remove(video_path)


        return jsonify({
            'prediction': int(predicted_class[0]),
            'message': 'Video processed successfully'
        })
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
