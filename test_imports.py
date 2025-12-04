#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

print("Testing imports for Jigsaw Puzzle Project...")

try:
    import cv2
    print("✅ OpenCV (cv2) imported successfully")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import matplotlib
    print("✅ Matplotlib imported successfully")
except ImportError as e:
    print(f"❌ Matplotlib import failed: {e}")

try:
    import scipy
    print("✅ SciPy imported successfully")
except ImportError as e:
    print(f"❌ SciPy import failed: {e}")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    print("✅ Scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")

try:
    from PIL import Image, ImageTk
    print("✅ Pillow (PIL) imported successfully")
except ImportError as e:
    print(f"❌ Pillow import failed: {e}")

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    print("✅ Tkinter imported successfully")
except ImportError as e:
    print(f"❌ Tkinter import failed: {e}")

print("\n" + "="*50)
print("Testing project module imports...")
print("="*50)

try:
    from jigsaw_pipeline import run_pipeline
    print("✅ jigsaw_pipeline imported successfully")
except ImportError as e:
    print(f"❌ jigsaw_pipeline import failed: {e}")

try:
    from jigsaw_matcher import JigsawMatcher
    print("✅ jigsaw_matcher imported successfully")
except ImportError as e:
    print(f"❌ jigsaw_matcher import failed: {e}")

print("\n🎉 Import testing complete!")
print("If all modules show ✅, your project should work correctly.")
