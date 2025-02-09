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

def analyze_video(video_path):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'bad_sp3.mov')
    rep_sequences = extract_poses(video_path)
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
    
    test_input = np.random.randn(1, 60, X.shape[-1])  # Random data
    pred_test = model.predict(test_input)
    print(f"Prediction on random data: {pred_test}")

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
            'confidence': float(predictions[0][class_idx])
        })
    
    return {'results': feedback}


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
