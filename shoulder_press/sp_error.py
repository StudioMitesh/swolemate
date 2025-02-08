import numpy as np

# helper functions for the error calculations (angle calc, euclidean, normalizer)
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def normalize_error(error_value, max_error):
    return min(error_value / max_error, 1.0)

def angle_with_vertical(vector):
    vertical = np.array([0, 1, 0])
    vector = np.array(vector)
    cosine_angle = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))



# parse a single frame of the pose data
def parse_frame(frame_row, num_features=132):
    landmarks = {}
    required_indices = [11, 13, 15, 12, 14, 16] # indices of the landmarks we are interested in
    for i in required_indices:
        start = i * 4  # each landmark has 4 consec values
        if start + 3 < len(frame_row):
            x = frame_row[start]
            y = frame_row[start+1]
            z = frame_row[start+2]
            landmarks[i] = (x, y, z)
    return landmarks


# frame error calculations
def classify_errors(frame_pose, reference_pose, thresholds):
    errors = {}
    ideal_elbow_angle = 55.0

    # LEFT
    # left elbow angle error
    if 11 in frame_pose and 13 in frame_pose and 15 in frame_pose and 11 in reference_pose and 13 in reference_pose and 15 in reference_pose:
        user_left_elbow_angle = calculate_angle(frame_pose[11], frame_pose[13], frame_pose[15])
        left_elbow_error = abs(user_left_elbow_angle - ideal_elbow_angle)
        errors['left_elbow'] = normalize_error(left_elbow_error, thresholds['angle'])
    else:
        errors['left_elbow'] = 0.0

    # left shoulder distance error
    if 11 in frame_pose and 11 in reference_pose:
        shoulder_distance_error = euclidean_distance(frame_pose[11], reference_pose[11])
        errors['left_shoulder'] = normalize_error(shoulder_distance_error, thresholds['distance'])
    else:
        errors['left_shoulder'] = 0.0

    # left wrist orientation error
    if 13 in frame_pose and 15 in frame_pose and 13 in reference_pose and 15 in reference_pose:
        left_forearm_vector = np.array(frame_pose[15]) - np.array(frame_pose[13])
        left_ref_forearm_vector = np.array(reference_pose[15]) - np.array(reference_pose[13])
        left_current_orientation = angle_with_vertical(left_forearm_vector)
        left_ref_orientation = angle_with_vertical(left_ref_forearm_vector)
        wrist_orientation_error = abs(left_current_orientation - left_ref_orientation)
        errors['left_wrist_orientation'] = normalize_error(wrist_orientation_error, thresholds['wrist_orientation'])
    else:
        errors['left_wrist_orientation'] = 0.0

    # left shoulder angle error
    if 11 in frame_pose and 13 in frame_pose and 11 in reference_pose and 13 in reference_pose:
        left_upper_arm_vector = np.array(frame_pose[13]) - np.array(frame_pose[11])
        left_ref_upper_arm_vector = np.array(reference_pose[13]) - np.array(reference_pose[11])
        left_current_shoulder_angle = angle_with_vertical(left_upper_arm_vector)
        left_ref_shoulder_angle = angle_with_vertical(left_ref_upper_arm_vector)
        shoulder_angle_error = abs(left_current_shoulder_angle - left_ref_shoulder_angle)
        errors['left_shoulder_angle'] = normalize_error(shoulder_angle_error, thresholds['shoulder_angle'])
    else:
        errors['left_shoulder_angle'] = 0.0

    # left depth error
    if 11 in frame_pose and 15 in frame_pose and 11 in reference_pose and 15 in reference_pose:
        left_depth_current = frame_pose[15][2] - frame_pose[11][2]
        left_depth_ref = reference_pose[15][2] - reference_pose[11][2]
        left_depth_error = abs(left_depth_current - left_depth_ref)
        errors['left_depth'] = normalize_error(left_depth_error, thresholds['depth'])
    else:
        errors['left_depth'] = 0.0

    # RIGHT
    # right elbow angle error
    if 12 in frame_pose and 14 in frame_pose and 16 in frame_pose and 12 in reference_pose and 14 in reference_pose and 16 in reference_pose:
        user_right_elbow_angle = calculate_angle(frame_pose[12], frame_pose[14], frame_pose[16])
        right_elbow_error = abs(user_right_elbow_angle - ideal_elbow_angle)
        errors['right_elbow'] = normalize_error(right_elbow_error, thresholds['angle'])
    else:
        errors['right_elbow'] = 0.0

    # right shoulder distance error
    if 12 in frame_pose and 12 in reference_pose:
        right_shoulder_distance_error = euclidean_distance(frame_pose[12], reference_pose[12])
        errors['right_shoulder'] = normalize_error(right_shoulder_distance_error, thresholds['distance'])
    else:
        errors['right_shoulder'] = 0.0

    # right wrist orientation error
    if 14 in frame_pose and 16 in frame_pose and 14 in reference_pose and 16 in reference_pose:
        right_forearm_vector = np.array(frame_pose[16]) - np.array(frame_pose[14])
        right_ref_forearm_vector = np.array(reference_pose[16]) - np.array(reference_pose[14])
        right_current_orientation = angle_with_vertical(right_forearm_vector)
        right_ref_orientation = angle_with_vertical(right_ref_forearm_vector)
        wrist_orientation_error = abs(right_current_orientation - right_ref_orientation)
        errors['right_wrist_orientation'] = normalize_error(wrist_orientation_error, thresholds['wrist_orientation'])
    else:
        errors['right_wrist_orientation'] = 0.0

    # right shoulder angle error
    if 12 in frame_pose and 14 in frame_pose and 12 in reference_pose and 14 in reference_pose:
        right_upper_arm_vector = np.array(frame_pose[14]) - np.array(frame_pose[12])
        right_ref_upper_arm_vector = np.array(reference_pose[14]) - np.array(reference_pose[12])
        right_current_shoulder_angle = angle_with_vertical(right_upper_arm_vector)
        right_ref_shoulder_angle = angle_with_vertical(right_ref_upper_arm_vector)
        shoulder_angle_error = abs(right_current_shoulder_angle - right_ref_shoulder_angle)
        errors['right_shoulder_angle'] = normalize_error(shoulder_angle_error, thresholds['shoulder_angle'])
    else:
        errors['right_shoulder_angle'] = 0.0

    # right depth error
    if 12 in frame_pose and 16 in frame_pose and 12 in reference_pose and 16 in reference_pose:
        right_depth_current = frame_pose[16][2] - frame_pose[12][2]
        right_depth_ref = reference_pose[16][2] - reference_pose[12][2]
        right_depth_error = abs(right_depth_current - right_depth_ref)
        errors['right_depth'] = normalize_error(right_depth_error, thresholds['depth'])
    else:
        errors['right_depth'] = 0.0
    return errors

# aggregate the errors on a full sequence of a rep
def compute_rep_error(seq, reference_pose, thresholds, weights):
    num_frames = seq.shape[0]
    metric_keys = [
        'left_elbow', 'right_elbow',
        'left_shoulder', 'right_shoulder',
        'left_wrist_orientation', 'right_wrist_orientation',
        'left_shoulder_angle', 'right_shoulder_angle',
        'left_depth', 'right_depth'
    ]
    aggregated_errors = {key: 0.0 for key in metric_keys}
    
    for frame in seq:
        frame_pose = parse_frame(frame)
        errors = classify_errors(frame_pose, reference_pose, thresholds)
        for key in metric_keys:
            aggregated_errors[key] += errors.get(key, 0.0)
    for key in aggregated_errors:
        aggregated_errors[key] /= num_frames

    composite_error = 0.0
    for key, err in aggregated_errors.items():
        composite_error += weights.get(key, 0.0) * err
    return composite_error, aggregated_errors

# assignment of a discrete label of a rep based on the error
def assign_error_type(aggregated_errors, left_threshold=0.5, right_threshold=0.5):
    '''
    0 = good form, the errors don't exceed the thresholds
    '''
    left_keys = ['left_elbow', 'left_shoulder', 'left_wrist_orientation', 'left_shoulder_angle', 'left_depth']
    right_keys = ['right_elbow', 'right_shoulder', 'right_wrist_orientation', 'right_shoulder_angle', 'right_depth']

    left_error = sum(aggregated_errors.get(k, 0.0) for k in left_keys)
    right_error = sum(aggregated_errors.get(k, 0.0) for k in right_keys)

    if left_error < left_threshold and right_error < right_threshold:
        return 0
    elif left_error >= left_threshold and right_error < right_threshold:
        return 1
    elif left_error < left_threshold and right_error >= right_threshold:
        return 2
    elif left_error >= left_threshold and right_error >= right_threshold:
        return 3
    return 0


