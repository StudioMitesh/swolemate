import cv2

def standardize_frame(frame, target_height=1920, target_width=1080):
    """
    standardize the frame to a target height and width
    """
    h, w = frame.shape[:2]
    
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_frame = cv2.resize(frame, (new_w, new_h))
    """
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left
    
    padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))"""
    
    return resized_frame