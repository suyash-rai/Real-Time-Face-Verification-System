# Real-Time Face Verification System

A Computer Vision project that uses OpenCV and DeepFace to verify whether the person in front of a webcam matches a stored reference image, in real time.

---

## 1. Overview

This system continuously captures frames from a webcam, detects faces using a Haar Cascade classifier, and verifies each detected face against a pre-stored reference image using the FaceNet deep learning model (via the DeepFace library). The result — **MATCH** or **NO MATCH** — is overlaid on the live video feed.

---

## 2. Problem Statement

Traditional PIN-based or password-based authentication systems have well-known weaknesses: passwords can be forgotten, stolen, or shared. Face-based authentication provides a biometric alternative that is harder to spoof and does not require the user to remember anything.

This project demonstrates how face verification — confirming whether two face images belong to the same person — can be implemented using a standard webcam and open-source computer vision tools.

---

## 3. Objectives

- Capture live video from a webcam
- Detect faces in each frame using Haar Cascade detection
- Compare each detected face to a stored reference image using a deep learning model
- Display a real-time MATCH / NO MATCH result on screen
- Allow the user to capture a new reference image at any time
- Handle missing reference images and webcam failures gracefully

---

## 4. Features

- Real-time webcam capture at 640×480
- Face detection using Haar Cascade (OpenCV)
- Face verification using FaceNet via DeepFace
- Bounding box drawn around detected faces
- MATCH / NO MATCH label overlaid on the video feed
- Multithreaded verification (does not block the display thread)
- Press `s` to capture a new reference image on the fly
- Press `q` to quit

---

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| OpenCV (`opencv-python`) | Webcam capture, image processing, display |
| DeepFace | High-level face verification API |
| FaceNet | Deep learning model for facial embeddings |
| NumPy | Array operations |
| threading | Background verification without blocking the UI |

---

## 6. Computer Vision Concepts

### Haar Cascade Face Detection

Haar Cascade is a machine-learning-based approach to object detection proposed by Viola and Jones (2001). It uses a cascade of simple rectangular features (Haar features) trained using AdaBoost to distinguish faces from non-faces very quickly.

OpenCV ships with pre-trained Haar Cascade XML files. This project uses `haarcascade_frontalface_default.xml`, which detects frontal faces.

Key parameters used:
- `scaleFactor=1.1` — the image is reduced by 10% at each scale level
- `minNeighbors=5` — how many overlapping detections a region needs before it is accepted as a face
- `minSize=(50, 50)` — minimum face size in pixels

### DeepFace

DeepFace is an open-source Python library that wraps several state-of-the-art face recognition models. It provides a simple `verify()` API that:
1. Detects faces in the input images (or accepts pre-cropped inputs)
2. Passes each face through the selected model to generate a 128-dimensional embedding vector
3. Computes the distance (Euclidean or cosine) between the two embedding vectors
4. Compares the distance to a threshold to decide `verified = True / False`

### FaceNet

FaceNet (Schroff et al., 2015) is a deep convolutional neural network that maps face images to a compact 128-dimensional embedding space, such that faces of the same person are close together and faces of different people are far apart. This project uses `model_name="Facenet"` in DeepFace.

---

## 7. How the System Works

```
Webcam Frame
     │
     ▼
Haar Cascade Face Detection
(detects faces; if none found → NO MATCH)
     │
     ▼
Frame Resize to 224×224
(FaceNet input size)
     │
     ▼
DeepFace.verify(frame, reference_image, model_name="Facenet")
     │
     ▼
Euclidean distance between FaceNet embeddings
     │
     ▼
Distance < threshold → MATCH
Distance ≥ threshold → NO MATCH
     │
     ▼
Visual Overlay on Live Feed
(green bounding box + "MATCH!" or red + "NO MATCH")
```

The verification runs in a background thread every 30 frames so the display loop is never blocked. The `face_match` flag is protected by a `threading.Lock` to prevent race conditions.

---

## 8. Project Structure

```
face detection/
├── face_detection.py      # Core logic: capture, detect, verify, display
├── main.py                # Entry point with startup checks
├── test.py                # Basic functionality test suite
├── requirements.txt       # Python dependencies
├── reference.jpg          # Reference face image (replace with your own)
├── README.md              # This file
├── .gitignore
├── screenshots/
│   └── README.md          # Instructions for capturing screenshots
└── report/
    └── project_report.md  # Detailed project report
```

---

## 9. Installation

### Step 1 — Clone or download the project

```bash
git clone <repository-url>
cd "face detection"
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
python -m pip install -r requirements.txt
```

> **Note:** DeepFace will download the FaceNet model weights (~90 MB) on the first run. An internet connection is required for this initial download.

---

## 10. Running the Project

```bash
python main.py
```

You can also run the core module directly:

```bash
python face_detection.py
```

---

## 11. How to Use

### Reference Image

Place a clear, well-lit photo of your face named `reference.jpg` in the project root folder. The image should show a single face looking straight at the camera.

If `reference.jpg` does not exist, the system will still open the webcam and display NO MATCH until a reference image is set.

### Capturing a Reference Image Inside the App

1. Run `python main.py`
2. Face the webcam
3. Press **`s`** — the current frame is immediately saved as `reference.jpg`
4. Verification will begin automatically

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Save the current frame as the new reference image |

### Understanding the Output

- **Green bounding box + "MATCH!"** — The detected face matches the reference image
- **Red bounding box + "NO MATCH"** — No match was found

---

## 12. Example Output

When running, the application shows a 640×480 webcam window.

- If your face matches `reference.jpg`: a green rectangle appears around your face and **"MATCH!"** is shown at the bottom.
- If there is no match or no face detected: a red rectangle (if a face is detected) and **"NO MATCH"** is shown.
- The verification updates approximately every second (every 30 frames at ~30 fps).

See `screenshots/README.md` for instructions on capturing screenshots.

---

## 13. Troubleshooting

### Webcam does not open
- Make sure no other application (e.g. Zoom, Teams) is using the camera
- On Windows, check Settings → Privacy → Camera — ensure camera access is allowed for Desktop apps
- Try connecting an external webcam if the built-in camera fails

### reference.jpg not found
- Place a photo named `reference.jpg` in the same folder as `main.py`
- Or press `s` inside the application to capture a new reference

### Dependency installation fails
- Make sure you are using Python 3.10, 3.11, or 3.12 (Python 3.14+ is not yet supported by TensorFlow/DeepFace)
- Run: `python --version` to verify
- Try: `python -m pip install --upgrade pip` before installing dependencies

### DeepFace model download fails
- Check your internet connection — FaceNet weights (~90 MB) are downloaded on first run
- Models are cached in `~/.deepface/weights/`
- If a proxy is required, set the `HTTP_PROXY` / `HTTPS_PROXY` environment variables

### TensorFlow warning messages
- Messages like `oneDNN custom operations are on` and deprecation warnings are normal and do not affect functionality

### Verification is slow
- The first call to `DeepFace.verify` takes a few seconds because the model is loaded into memory. Subsequent calls are faster.

---

## 14. Limitations

- Works best with frontal, well-lit faces. Performance degrades significantly in poor lighting or with extreme head poses.
- Haar Cascade detection can produce false positives (detecting non-faces) and false negatives (missing faces at angles). It is less accurate than modern CNN-based detectors (e.g., MTCNN, RetinaFace).
- Verification runs every 30 frames, so there is a short delay between the actual face change and the label update.
- The system does not defend against spoofing using printed photos or video playback.
- FaceNet weights must be downloaded from the internet on first run.
- Not tested for performance on low-end hardware; processing may be slower on CPUs without hardware acceleration.

---

## 15. Future Improvements

- Replace Haar Cascade with a more accurate face detector (e.g., MTCNN or RetinaFace) for better detection under varied conditions
- Add liveness detection to prevent photo/video spoofing
- Support multiple reference images for the same person
- Add a simple logging system to record match/no-match events with timestamps
- Explore lighter models (e.g., MobileNet-based) for faster inference on embedded hardware
- Add a configuration file for adjusting parameters (threshold, frame skip, model choice) without editing code

---

## 16. Conclusion

This project demonstrates a complete, real-time face verification pipeline using standard open-source computer vision tools. It combines classical feature-based detection (Haar Cascade) with modern deep learning-based recognition (FaceNet via DeepFace) to provide a functional identity verification system running on a standard laptop webcam. The multithreaded design keeps the display smooth while verification runs in the background.