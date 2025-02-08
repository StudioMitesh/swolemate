from flask import Flask, request, jsonify, render_template, send_from_directory
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
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model('shoulder_press_model.keras')

def process_video(video_path):
    df = extract_poses(video_path, label=None)
    print("df shape:", df.shape)
    print(df.head())
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
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        video_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'recorded_video.mp4')
        print("Saving video to:", video_filename)
        video_file.save(video_filename)
        print("video file saved")

        frames = process_video(video_filename)
        print("Frames shape:", frames.shape)
        
        try:
            print("Predicting...")
            predictions = model.predict(frames)
            predicted_class = np.argmax(predictions, axis=1)
            print("Predicted class:", predicted_class)
            return jsonify({
                'prediction': int(predicted_class[0]),
                'message': 'Video processed successfully'
            })
        except Exception as e:
            print(f"Error occurred in model prediction: {e}")
            return jsonify({'error': 'Internal server error', 'message': str(e)}), 500
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='video/mp4')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
