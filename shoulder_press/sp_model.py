import sys
import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Flatten, Dropout, BatchNormalization, Bidirectional, Attention, concatenate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pose_detector import extract_poses
from sp_error import compute_rep_error, assign_error_type
import glob
import matplotlib.pyplot as plt


NUM_CLASSES = 7
# getting the data

def prepare_dataset(folders):
    X = []
    y = []
    
    for label, folder in enumerate(folders):
        sequences, labels = process_folder(folder, label)
        X.extend(sequences)
        y.extend(labels)
    
    return np.array(X), np.array(y)


def process_folder(folder_path, label):
    sequences = []
    labels = []
    
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
                padded = np.pad(seq_data, ((0, 60-len(seq_data)), (0, 0)), mode='edge')
            else:
                padded = seq_data[:60]
            
            labels.append(label)
            sequences.append(padded)
    
    return sequences, labels

folders = [
    "pose_data/good_shoulder_press",  # Label 0
    "pose_data/left_shoulder_error",  # Label 1
    "pose_data/left_elbow_error",     # Label 2
    "pose_data/left_wrist_error",     # Label 3
    "pose_data/right_shoulder_error", # Label 4
    "pose_data/right_elbow_error",    # Label 5
    "pose_data/right_wrist_error"     # Label 6
]

def add_noise(X, y, noise_level=0.01):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
    X_noisy = X + noise * (y.argmax(axis=1) != 0).reshape(-1, 1, 1)
    return np.clip(X_noisy, -1, 1)

X, y = prepare_dataset(folders)

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
y = to_categorical(y, num_classes=NUM_CLASSES)

# train test split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

y_labels = np.argmax(y, axis=1)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_labels), y=y_labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class weights: {class_weights_dict}")

#X_train = add_noise(X_train, y_train, noise_level=0.005)

unique, counts = np.unique(y.argmax(axis=1), return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))


# lstm model

model = Sequential([
    LSTM(128, input_shape=(sequence_length, num_features), return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='sigmoid')  # should be 7 different binary classes
])

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# compile the model w adam optimizer, binary crossentropy loss for class output, and categorical crossentropy loss for error output
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=['accuracy']
)


model.summary()


# train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict
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


# reinforcement learning if we want to include
def choose_action(state):
    state = np.expand_dims(state, axis=0)
    action_probs = model.predict(state)[0]
    action = np.argmax(action_probs)
    return action, action_probs

def train_with_reinforcement_learning(X_train, y_train, epochs=10, batch_size=32):
    for epoch in range(epochs):
        for step in range(len(X_train) // batch_size):
            batch_x = X_train[step * batch_size: (step + 1) * batch_size]
            batch_y = y_train[step * batch_size: (step + 1) * batch_size]
            
            actions, action_probs = choose_action(batch_x)
            
            reward = np.array([1 if actions[i] == np.argmax(batch_y[i]) else -1 for i in range(batch_size)])
            
            with tf.GradientTape() as tape:
                log_prob = tf.reduce_sum(tf.math.log(action_probs) * batch_y, axis=1)
                loss = -tf.reduce_mean(log_prob * reward)
                
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            print(f"Epoch: {epoch+1}/{epochs}, Step: {step+1}/{len(X_train)//batch_size}, Loss: {loss.numpy()}")

#train_with_reinforcement_learning(X_train, y_train, epochs=10, batch_size=32)



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