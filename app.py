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
from llm_chat import initial_call, chat_call
from shoulder_press.sp_metrics import compute_angle, compute_displacement, compute_depth
import mediapipe as mp
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

rep_count = 0
model = load_model('new_shoulder_press_model.keras')


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

        video_path = save_temp_video(video_file)
        print(f"Video saved to: {video_path}")

        result = analyze_video(video_path)
        return jsonify(result)
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

def analyze_video(video_path):
    rep_sequences = extract_poses(video_path)
    all_rep_metrics = []
    for rep in rep_sequences:
        print(f"Processing rep: {rep}")
        left_shoulder = (rep['x11'].iloc[0], rep['y11'].iloc[0])
        right_shoulder = (rep['x12'].iloc[0], rep['y12'].iloc[0])
        left_elbow = (rep['x13'].iloc[0], rep['y13'].iloc[0])
        right_elbow = (rep['x14'].iloc[0], rep['y14'].iloc[0])
        left_wrist = (rep['x15'].iloc[0], rep['y15'].iloc[0])
        right_wrist = (rep['x16'].iloc[0], rep['y16'].iloc[0])

        left_elbow_angle = compute_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = compute_angle(right_shoulder, right_elbow, right_wrist)

        left_shoulder_displacement = compute_displacement(left_shoulder, left_elbow)
        right_shoulder_displacement = compute_displacement(right_shoulder, right_elbow)

        left_forearm_angle = compute_angle(left_elbow, left_wrist, (0, 0))
        right_forearm_angle = compute_angle(right_elbow, right_wrist, (0, 0))

        left_depth = compute_depth(left_shoulder, left_wrist)
        right_depth = compute_depth(right_shoulder, right_wrist)

        rep_metrics = {
            'left_elbow_angle': left_elbow_angle,
            'right_elbow_angle': right_elbow_angle,
            'left_shoulder_displacement': left_shoulder_displacement,
            'right_shoulder_displacement': right_shoulder_displacement,
            'left_forearm_angle': left_forearm_angle,
            'right_forearm_angle': right_forearm_angle,
            'left_depth': left_depth,
            'right_depth': right_depth
        }
        all_rep_metrics.append(rep_metrics)

    if len(rep_sequences) == 0:
        return {'error': 'No pose data detected'}
    
    X = []
    scaler = StandardScaler()
    for seq in rep_sequences:
        scaled = scaler.fit_transform(seq.drop(columns=['frame']))
        padded = scaled[:60] if len(scaled) >=60 else np.pad(scaled, ((0,60-len(scaled)),(0,0)), mode='edge')
        X.append(padded)
    
    X = np.array(X)
    print(f"Processed input shape: {X.shape}")  
    print(f"First row of input data:\n{X[0]}")

    predictions = model.predict(np.array(X))
    print(f"Predictions:\n{predictions}")
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
            'confidence': float(predictions[0][class_idx]),
            "metrics": all_rep_metrics
        })
    
    return {'results': feedback, "rep_count": rep_count}


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='video/mp4')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/initial_chat', methods=['POST'])
def initial_chat():
    try:
        data = request.get_json()
        codes = data.get('codes', '')
        response = initial_call(codes)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in initial chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        history = data.get('history', [])
        response = chat_call(message, history)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
