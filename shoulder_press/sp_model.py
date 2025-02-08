import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Flatten, Dropout, BatchNormalization, concatenate
from sp_error import compute_rep_error, assign_error_type

# getting the data
good_df = pd.read_csv("good_shoulder_press.csv")
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
y_class = df["label"].values

# features for input data
sequence_length = 60 # sequence length we are using is 2 times our FPS frame rate rn (2 * 30 fps)
num_features = X.shape[1]
num_samples = X.shape[0] // sequence_length
X = X[:num_samples * sequence_length]
X = X.reshape(num_samples, sequence_length, num_features)
print("Reshaped shape:", X.shape)

y_class = df["label"].values[:num_samples * sequence_length]
y_class_seq = y_class.reshape(num_samples, sequence_length)
y_class_seq = np.array([np.bincount(seq).argmax() for seq in y_class_seq])
y_class_seq = y_class_seq.reshape(-1, 1)

# the error threshold values stuff

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
error_labels = np.array(error_labels)
num_error_classes = 4
y_error = to_categorical(error_labels, num_classes=num_error_classes)


# train test split data
X_train, X_test, y_class_train, y_class_test, y_error_train, y_error_test = train_test_split(
    X, y_class_seq, y_error, test_size=0.3, random_state=42
)

# the model

input_layer = Input(shape=(sequence_length, num_features))


# cnn model
'''
first a 1 dimensional convolutional layer
second a batch normalization layer
third another 1 dimensional with double dimension convolutional layer
fourth another batch normalization layer
fifth flatten the output
'''

cnn_layer = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
cnn_layer = BatchNormalization()(cnn_layer)
cnn_layer = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn_layer)
cnn_layer = BatchNormalization()(cnn_layer)
cnn_layer = Flatten()(cnn_layer)

# lstm model
'''
first a 128 unit lstm layer
then another 64 unit lstm layer
'''

lstm_layer = LSTM(128, return_sequences=True)(input_layer)
lstm_layer = LSTM(64)(lstm_layer)

# merge the cnn and lstm models together
merged = concatenate([cnn_layer, lstm_layer])
# add dropout layer w 0.5 rate so that we remove overfitting
merged = Dropout(0.5)(merged)

# rn its a sigmoid activation function for the classes output layer
class_output = Dense(1, activation="sigmoid", name="class_output")(merged)

# rn its a softmax activation for the error output
error_output = Dense(num_error_classes, activation="softmax", name="error_output")(merged)

# creation of the model
model = Model(inputs=input_layer, outputs=[class_output, error_output])

# compile the model w adam optimizer, binary crossentropy loss for class output, and categorical crossentropy loss for error output
model.compile(
    optimizer="adam",
    loss={"class_output": "binary_crossentropy", "error_output": "categorical_crossentropy"},
    metrics={"class_output": "accuracy", "error_output": "accuracy"}
)
model.summary()


# train the model
model.fit(
    X_train,
    {"class_output": y_class_train, "error_output": y_error_train},
    epochs=10,
    batch_size=32
)

# evaluate the model
results = model.evaluate(
    X_test,
    {"class_output": y_class_test, "error_output": y_error_test},
    batch_size=32
)
print("Test results:", results)