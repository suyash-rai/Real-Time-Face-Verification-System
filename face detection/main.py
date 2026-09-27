"""
Real-Time Face Verification System
Entry Point: main.py

Run with:
    python main.py

Controls:
    q  - Quit the application
    s  - Save the current frame as the new reference image
"""

import sys
import os

# ── Dependency check ──────────────────────────────────────────────────────────
MISSING = []
try:
    import cv2
except ImportError:
    MISSING.append("opencv-python")

try:
    from deepface import DeepFace  # noqa: F401
except ImportError:
    MISSING.append("deepface")

if MISSING:
    print("[ERROR] Missing required packages:", ", ".join(MISSING))
    print("        Install them with:  python -m pip install -r requirements.txt")
    sys.exit(1)

# ── Reference image check ─────────────────────────────────────────────────────
REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "reference.jpg")
if not os.path.isfile(REFERENCE_PATH):
    print("[WARNING] reference.jpg not found in the project folder.")
    print("          The system will open the webcam anyway.")
    print("          Press 's' while the window is open to capture and save a reference frame.")
    print()

# ── Launch the application ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Real-Time Face Verification System")
    print("=" * 55)
    print("  Controls:")
    print("    q  → Quit")
    print("    s  → Save current frame as reference image")
    print("=" * 55)
    print()

    # Import and run the core module
    try:
        import face_detection  # noqa: F401  (executes on import)
    except KeyboardInterrupt:
        print("\n[INFO] Application stopped by user.")
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")
        sys.exit(1)
