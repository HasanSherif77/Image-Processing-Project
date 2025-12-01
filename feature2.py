# feature2.py
# Temporary placeholder for Feature 2

import cv2
import numpy as np

def apply(img):
    """Fake feature 2 — sharpening."""
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)
