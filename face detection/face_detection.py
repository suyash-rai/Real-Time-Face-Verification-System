"""
face_detection.py — Real-Time Face Verification System (core module)

Pipeline:
    Webcam frame
    → Haar Cascade face detection
    → Resize frame
    → DeepFace.verify (FaceNet model)
    → MATCH / NO MATCH overlay
    → Display

Controls:
    q  - Quit
    s  - Save current frame as reference.jpg
"""

import cv2
import threading
import sys
import os
from deepface import DeepFace

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATH = os.path.join(BASE_DIR, "reference.jpg")
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── Load Haar Cascade face detector ──────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
if face_cascade.empty():
    print("[ERROR] Failed to load Haar Cascade classifier.")
    print(f"        Expected at: {HAAR_PATH}")
    sys.exit(1)

# ── Load reference image ──────────────────────────────────────────────────────
reference_img = cv2.imread(REFERENCE_PATH)
if reference_img is None:
    print("[WARNING] reference.jpg not found. Press 's' to save a reference frame.")

# ── Shared state ──────────────────────────────────────────────────────────────
face_match = False
counter = 0
lock = threading.Lock()


def check_face(frame, reference):
    """
    Run DeepFace verification in a background thread.

    Steps:
    1. Run Haar Cascade on the frame to check if a face is visible.
    2. If a face is found, resize both images to (224, 224) for FaceNet.
    3. Call DeepFace.verify with the FaceNet model.
    4. Update the global `face_match` flag under the thread lock.
    """
    global face_match

    try:
        # Step 1 — Haar Cascade face presence check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            with lock:
                face_match = False
            return

        # Step 2 — Resize for FaceNet input
        frame_resized = cv2.resize(frame, (224, 224))
        reference_resized = cv2.resize(reference, (224, 224))

        # Step 3 — DeepFace verification (FaceNet model)
        result = DeepFace.verify(
            img1_path=frame_resized,
            img2_path=reference_resized,
            model_name="Facenet",
            enforce_detection=False
        )

        # Step 4 — Update shared flag
        with lock:
            face_match = result["verified"]

    except Exception as e:
        print(f"[Verification error] {e}")
        with lock:
            face_match = False


def draw_face_boxes(frame):
    """Detect and draw bounding boxes around faces in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    box_color = (0, 255, 0) if face_match else (0, 0, 255)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
    return frame


def run():
    """Main loop: capture frames, spawn verification threads, display results."""
    global reference_img, face_match, counter

    # ── Open webcam ───────────────────────────────────────────────────────────
    # Try CAP_DSHOW first (Windows DirectShow, lower latency)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback: let OpenCV pick the backend automatically
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        print("        Make sure a camera is connected and not used by another application.")
        print("        On Windows, also check camera permissions in Settings → Privacy → Camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Webcam opened. Starting face verification loop…")
    if reference_img is None:
        print("[INFO] No reference image loaded. Press 's' to capture one.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame from webcam. Retrying…")
            continue

        # Spawn a verification thread every 30 frames if one is not already running
        if reference_img is not None and counter % 30 == 0 and threading.active_count() < 3:
            threading.Thread(
                target=check_face,
                args=(frame.copy(), reference_img.copy()),
                daemon=True
            ).start()

        counter += 1

        # Draw face bounding boxes
        frame = draw_face_boxes(frame)

        # Overlay MATCH / NO MATCH label
        label = "MATCH!" if face_match else "NO MATCH"
        color = (0, 255, 0) if face_match else (0, 0, 255)

        # Semi-transparent banner at the bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 420), (640, 480), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, label, (20, 465),
                    cv2.FONT_HERSHEY_COMPLEX, 1.4, color, 2, cv2.LINE_AA)

        # Status hint
        if reference_img is None:
            cv2.putText(frame, "Press 's' to set reference", (340, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Face Verification — Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quitting…")
            break
        elif key == ord("s"):
            cv2.imwrite(REFERENCE_PATH, frame)
            reference_img = frame.copy()
            print(f"[INFO] Saved current frame as reference image → {REFERENCE_PATH}")

    cap.release()
    cv2.destroyAllWindows()


# Allow direct execution: python face_detection.py
if __name__ == "__main__":
    run()
else:
    # Called via `import face_detection` from main.py
    run()
