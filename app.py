from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
import cv2
import io
import tempfile
import os
from tensorflow.keras.models import load_model
from pose_detector import extract_poses

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model('new_shoulder_press_model.keras')

import mediapipe as mp
import cv2
import pandas as pd
import numpy as np

def save_temp_video(video_file):
    if not video_file:
        raise ValueError("No video file provided")
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_video.mp4')
    video_file.save(video_path)
    return video_path

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the video temporarily
        video_path = save_temp_video(video_file)
        print(f"Video saved to: {video_path}")

        # Analyze the video
        result = analyze_video(video_path)
        return jsonify(result)
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_video(video_path):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'good_sp3.mov')
    rep_sequences = extract_poses(video_path)
    
    X = []
    scaler = StandardScaler()
    for seq in rep_sequences:
        scaled = scaler.fit_transform(seq.drop(columns=['frame']))
        padded = scaled[:60] if len(scaled) >=60 else np.pad(scaled, ((0,60-len(scaled)),(0,0)), mode='edge')
        X.append(padded)
    
    predictions = model.predict(np.array(X))
    error_classes = np.argmax(predictions, axis=1)
    
    feedback = []
    class_mapping = {
        0: "Good form!",
        1: "Focus on left shoulder stability",
        2: "Adjust left elbow angle (aim for 90°)",
        3: "Keep left wrist straight",
        4: "Fix right shoulder positioning",
        5: "Right elbow flaring out",
        6: "Right wrist bending excessively"
    }
    
    for class_idx in error_classes:
        feedback.append({
            'class': int(class_idx),
            'message': class_mapping[class_idx],
            'confidence': float(predictions[0][class_idx])
        })
    
    return jsonify({'results': feedback})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='video/mp4')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
