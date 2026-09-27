# -*- coding: utf-8 -*-
"""
test.py - Basic project functionality tests
Real-Time Face Verification System

Tests performed:
  1. Required files exist
  2. OpenCV can be imported and initialized
  3. DeepFace can be imported
  4. NumPy can be imported
  5. threading module available
  6. Haar Cascade XML can be loaded
  7. reference.jpg can be read as an image
  8. A sample in-memory image can be processed through the Haar Cascade
  9. DeepFace.verify can be called on two small test images

Run with:
    python test.py
"""

import os
import sys

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = []


def report(status, description, detail=""):
    line = "  {}  {}".format(status, description)
    if detail:
        line += "\n         >> {}".format(detail)
    RESULTS.append((status, description))
    print(line)


def test_files_exist():
    print("\n-- 1. Required Files ----------------------------------------")
    required = ["face_detection.py", "main.py", "requirements.txt", "README.md"]
    for fname in required:
        path = os.path.join(BASE_DIR, fname)
        if os.path.isfile(path):
            report(PASS, "File exists: {}".format(fname))
        else:
            report(FAIL, "File missing: {}".format(fname))


def test_reference_image():
    print("\n-- 2. Reference Image ---------------------------------------")
    ref_path = os.path.join(BASE_DIR, "reference.jpg")
    if not os.path.isfile(ref_path):
        report(SKIP, "reference.jpg not found - press 's' in the app to create one")
        return

    try:
        import cv2
        img = cv2.imread(ref_path)
        if img is None:
            report(FAIL, "reference.jpg exists but could not be read as an image")
        else:
            h, w, c = img.shape
            report(PASS, "reference.jpg loaded successfully ({}x{}, {} channels)".format(w, h, c))
    except Exception as e:
        report(FAIL, "Error reading reference.jpg", str(e))


def test_imports():
    print("\n-- 3. Module Imports ----------------------------------------")

    # OpenCV
    try:
        import cv2
        report(PASS, "opencv-python imported (version {})".format(cv2.__version__))
    except ImportError as e:
        report(FAIL, "opencv-python import failed", str(e))
        return

    # NumPy
    try:
        import numpy as np
        report(PASS, "numpy imported (version {})".format(np.__version__))
    except ImportError as e:
        report(FAIL, "numpy import failed", str(e))

    # threading (stdlib)
    try:
        import threading  # noqa: F401
        report(PASS, "threading (stdlib) available")
    except ImportError as e:
        report(FAIL, "threading import failed", str(e))

    # DeepFace
    try:
        from deepface import DeepFace  # noqa: F401
        report(PASS, "deepface imported successfully")
    except ImportError as e:
        report(FAIL, "deepface import failed - run: pip install deepface", str(e))


def test_haar_cascade():
    print("\n-- 4. Haar Cascade Classifier --------------------------------")
    try:
        import cv2
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.isfile(haar_path):
            report(FAIL, "haarcascade_frontalface_default.xml not found", haar_path)
            return
        cascade = cv2.CascadeClassifier(haar_path)
        if cascade.empty():
            report(FAIL, "Haar Cascade loaded but is empty/corrupt")
        else:
            report(PASS, "haarcascade_frontalface_default.xml loaded OK")
    except Exception as e:
        report(FAIL, "Haar Cascade test error", str(e))


def test_image_processing():
    print("\n-- 5. Basic Image Processing ---------------------------------")
    try:
        import cv2
        import numpy as np

        # Create a synthetic 480x640 BGR image (blank grey)
        dummy = np.full((480, 640, 3), 128, dtype=np.uint8)

        # Grayscale conversion
        gray = cv2.cvtColor(dummy, cv2.COLOR_BGR2GRAY)
        report(PASS, "BGR->Grayscale conversion OK (shape: {})".format(gray.shape))

        # Resize (FaceNet input size)
        resized = cv2.resize(dummy, (224, 224))
        report(PASS, "Image resize to 224x224 OK (shape: {})".format(resized.shape))

        # Haar Cascade on blank image (should find 0 faces - that is fine)
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(haar_path)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        report(PASS, "Haar Cascade ran on test image - faces detected: {}".format(len(faces)))

    except Exception as e:
        report(FAIL, "Image processing test error", str(e))


def test_deepface_verify():
    print("\n-- 6. DeepFace Verification (synthetic images) --------------")
    try:
        import numpy as np
        from deepface import DeepFace

        # Two identical small images - should always verify as True
        img_a = np.full((224, 224, 3), 120, dtype=np.uint8)
        img_b = img_a.copy()

        result = DeepFace.verify(
            img1_path=img_a,
            img2_path=img_b,
            model_name="Facenet",
            enforce_detection=False
        )
        verified = result.get("verified")
        distance = round(result.get("distance", -1), 4)
        report(PASS, "DeepFace.verify executed - verified={}, distance={}".format(verified, distance))

    except Exception as e:
        report(FAIL, "DeepFace.verify test failed", str(e))


def main():
    print("=" * 55)
    print("  Real-Time Face Verification System - Test Suite")
    print("=" * 55)

    test_files_exist()
    test_reference_image()
    test_imports()
    test_haar_cascade()
    test_image_processing()
    test_deepface_verify()

    # Summary
    passed = sum(1 for s, _ in RESULTS if s == PASS)
    failed = sum(1 for s, _ in RESULTS if s == FAIL)
    skipped = sum(1 for s, _ in RESULTS if s == SKIP)
    total = len(RESULTS)

    print("\n" + "=" * 55)
    print("  Results: {} passed  |  {} failed  |  {} skipped  |  {} total".format(
        passed, failed, skipped, total))
    print("=" * 55)

    if failed > 0:
        print("\n  [!] Fix the failures above before submitting.")
        sys.exit(1)
    else:
        print("\n  All checks passed. Project is ready to run.")
        sys.exit(0)


if __name__ == "__main__":
    main()
