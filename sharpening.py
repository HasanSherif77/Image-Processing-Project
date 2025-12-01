# feature3.py
# Temporary placeholder for Feature 3

import cv2

def apply(img):
    """Fake feature 3 — brightening."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.add(hsv[:,:,2], 50)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
