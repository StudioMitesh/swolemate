import sys
import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Flatten, Dropout, BatchNormalization, Bidirectional, Attention, concatenate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pose_detector import extract_poses
from sp_error import classify_errors, assign_error_type
import glob
import matplotlib.pyplot as plt


# getting the data

def prepare_dataset(good_dir, bad_dir):
    X_good, y_good = process_folder(good_dir, is_good=True)
    X_bad, y_bad = process_folder(bad_dir, is_good=False)
    
    X = np.concatenate([X_good, X_bad])
    y = np.concatenate([y_good, y_bad])
    
    return X, y

def process_folder(folder_path, is_good):
    sequences = []
    labels = []
    
    reference_angles = {'elbow': 90, 'shoulder': 80}
    thresholds = {
        'left_shoulder': 0.15, 'right_shoulder': 0.15,
        'left_elbow': 20, 'right_elbow': 20,
        'left_wrist': 0.1, 'right_wrist': 0.1,
        'left_wrist_orientation': 0.1, 'right_wrist_orientation': 0.1,
        'left_shoulder_angle': 0.1, 'right_shoulder_angle': 0.1,
        'left_depth': 0.1, 'right_depth': 0.1
    }
    allowed_extensions = {'.mov', '.mp4'}

    for video_file in os.listdir(folder_path):
        file_extension = os.path.splitext(video_file)[1].lower()
        if file_extension not in allowed_extensions:
            continue
        video_path = os.path.join(folder_path, video_file)
        rep_sequences = extract_poses(video_path)
        
        for seq in rep_sequences:
            seq_data = seq.drop(columns=['frame']).values
            
            if len(seq_data) < 60:
                padded = np.pad(seq_data, ((0, 60-len(seq_data)), (0,0)), mode='edge')
            else:
                padded = seq_data[:60]
            
            if is_good:
                labels.append(0)
            else:
                errors = classify_errors(padded, reference_angles, thresholds)
                error_class = assign_error_type(errors, thresholds)
                labels.append(error_class)
            
            sequences.append(padded)
    
    return np.array(sequences), np.array(labels)





X, y = prepare_dataset(
    "pose_data/good_shoulder_press",
    "pose_data/bad_shoulder_press"
)

"""good_df = pd.read_csv("good_shoulder_press.csv")
bad_df = pd.read_csv("bad_shoulder_press.csv")

good_df["label"] = 1 
bad_df["label"] = 0

bad_df["error_type"] = np.random.randint(0, 3, size=len(bad_df))
good_df["error_type"] = -1

df = pd.concat([good_df, bad_df], ignore_index=True)

df = df.drop(columns=["frame"])
print(df.head())

X = df.drop(columns=["label", "error_type"]).values
print("Original shape:", X.shape)
y_class = df["label"].values"""

scaler = StandardScaler()
X = np.array([scaler.fit_transform(seq) for seq in X])


print("X shape:", X.shape)
num_samples, sequence_length, num_features = X.shape

"""# the error threshold values stuff

thresholds = {
    'angle': 30.0,             # elbow angle in degrees
    'distance': 0.05,          # shoulder pos in meters
    'wrist_orientation': 15.0, # max forearm deviation in degrees
    'shoulder_angle': 15.0,    # max shoulder deviation in degrees
    'depth': 0.05              # max depth in meters
}

weights = {
    'left_elbow': 0.15,
    'right_elbow': 0.15,
    'left_shoulder': 0.1,
    'right_shoulder': 0.1,
    'left_wrist_orientation': 0.15,
    'right_wrist_orientation': 0.15,
    'left_shoulder_angle': 0.1,
    'right_shoulder_angle': 0.1,
    'left_depth': 0.05,
    'right_depth': 0.05
}


reference_pose = {
    # left is 11, 13, 15
    11: (0.5, 0.5, 0.0),   # left shoulder
    13: (0.6, 0.45, 0.0),  # left elbow
    15: (0.7, 0.35, 0.0),  # left wrist
    # right is 12, 14, 16
    12: (0.5, 0.5, 0.0),   # right shoulder
    14: (0.4, 0.45, 0.0),  # right elbow
    16: (0.3, 0.35, 0.0)   # right wrist
}


error_labels = []
aggregated_errors_list = []
for i in range(num_samples):
    seq = X[i]
    composite_error, aggregated_errors = compute_rep_error(seq, reference_pose, thresholds, weights)
    error_type = assign_error_type(aggregated_errors,
                                   left_threshold=0.5,
                                   right_threshold=0.5)
    error_labels.append(error_type)
    aggregated_errors_list.append(aggregated_errors)
error_labels = np.array(error_labels)"""
num_classes = 7
y = to_categorical(y, num_classes=num_classes)

# train test split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# lstm model

model = Sequential([
    LSTM(128, input_shape=(sequence_length, num_features), return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')  # num_classes = good form + error types
])

# compile the model w adam optimizer, binary crossentropy loss for class output, and categorical crossentropy loss for error output
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
    #class_weight=compute_class_weight('balanced', classes=np.unique(np.argmax(y, axis=1)), y=np.argmax(y, axis=1))
)

# save the model
model.save("new_shoulder_press_model.keras")

# evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)


# metrics for the model eval
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=[
    "Good form", "Left shoulder", "Left elbow", "Left wrist",
    "Right shoulder", "Right elbow", "Right wrist"
]))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)

roc_auc = roc_auc_score(y_test, model.predict(X_test), multi_class='ovr')
print(f"ROC-AUC Score: {roc_auc:.4f}")


# visualization if we want
"""
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
"""