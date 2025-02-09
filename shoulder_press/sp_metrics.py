import numpy as np

def compute_angle(p1, p2, p3):
    """
    Compute the angle between three points p1, p2, p3.
    p1, p2, and p3 are the 2D coordinates (x, y).
    The angle is calculated between the vectors p1 -> p2 and p2 -> p3.
    """
    # Vector from p1 to p2
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    
    # Vector from p2 to p3
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # Compute the cosine of the angle between the vectors
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    cos_angle = dot_product / (norm_v1 * norm_v2)
    
    # Clip value to avoid rounding errors outside the range [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Compute the angle in radians and convert it to degrees
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle


def compute_displacement(p1, p2):
    """
    Compute the displacement between two points p1 and p2.
    p1 and p2 are 2D coordinates (x, y).
    """
    return np.linalg.norm(np.array([p1[0] - p2[0], p1[1] - p2[1]]))


def compute_depth(shoulder_point, hip_point):
    """
    Compute the depth of a person by measuring the y-axis distance
    between the shoulder and hip points.
    shoulder_point and hip_point are 2D coordinates (x, y).
    """
    return np.abs(shoulder_point[1] - hip_point[1])
