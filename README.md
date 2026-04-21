# Semantic Video Understanding & Object-Level Event Detection System

A high-performance pipeline architecture combining real-time object detection, open-vocabulary semantic attributes matching, visual tracking, and event compliance monitoring. 

It provides both a legacy CLI module and a highly dynamic, glassmorphic Web Application dashboard for live streaming.

## Features & Architecture

This system uses a **Tiered Hybrid Pipeline** to guarantee fast, state-of-the-art accuracy leveraging multiple foundational models:

1. **Tier-1 YOLOv8 + CLIP Attribute Matching**: Uses YOLOv8 for rapid object bounding box extraction. If a text query is provided, crops are projected into the CLIP embedding space to determine cosine similarity (e.g., query "red backpack").
2. **Tier-2 Batched ONNX CLIP Scanning**: Serves as a dynamic fallback if Tier-1 detects nothing. Performs sliding-window frame analysis strictly on regions indicated by a persistent Spatial-Temporal Heatmap, bypassing standard PyTorch overhead.
3. **ByteTrack Multi-Object Tracking**: Wraps the `supervision` ByteTrack API to maintain stable tracker IDs across sequential frames even under transient occlusion.
4. **DeepLabV3 Context Filter**: Eliminates absurd or physically impossible detections by running semantic segmentation on the entire scene (e.g., rejecting a "car" predicted floating in the "sky").
5. **Neural Compliance Checker**: Upgrades from hard-coded heuristics to an iterative, dynamic Multi-Layer Perceptron (MLP) capable of learning custom compliance violations on-the-fly iteratively after 50 tracked data samples.

---

## 💻 The Web Dashboard

Alongside the CLI, this project runs an interactive, dual-server Web Application featuring real-time stream displays and tracking logs:

- **Backend (FastAPI)**: Found in `src/server/app.py`. Streams an MJPEG stream locally, and transmits pipeline Object IDs securely via **WebSockets**.
- **Frontend (Vite + React)**: Found in `ui/`. Programmed entirely via Vanilla CSS ensuring a premium, glassmorphic dark-mode appearance and instantaneous event logging.

---

## 🛠️ Installation & Setup

We recommend utilizing an active virtual environment.

```bash
# 1. Clone the repository and initialize virtualenv
cd Semantic-Video-Understanding-and-Object-Level-Event-Detection-System
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Python Pipeline Dependencies
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart websockets

# 3. Setup the UI Frontend module
cd ui
npm install
```

---

## 🚀 Usage

You can interface with the model using either the CLI or Web Dashboard.

### Method A: Web Interface (Recommended)

Requires two active terminals initializing both the pipeline streaming endpoint and React application.

**Terminal 1 (Backend API):**
```bash
source .venv/bin/activate
uvicorn src.server.app:app --host 0.0.0.0 --port 8000
```
**Terminal 2 (Frontend UI):**
```bash
cd ui
export PATH="/opt/homebrew/bin:$PATH"
npm run dev
```

Browse to `http://localhost:5173/`. Utilize the dashboard to connect seamlessly to your Mac's camera stream (`--video 0`) or seamlessly upload an offline video. 

### Method B: Native CLI Command

```bash
source .venv/bin/activate
python scripts/run_pipeline.py \
  --video path_to_video.MOV \
  --fps 30 \
  --query "person with a helmet" \
  --compliance \
  --show
```
*Appending `--show` projects the native OpenCV local window popup.*

---

## 📁 Directory Structure

```text
├── src/
│   ├── pipeline/
│   │   ├── detector.py          # YOLO & CLIP Tier Models
│   │   ├── byte_tracker.py      # Supervision multi-tracking wrapper
│   │   ├── context_filter.py    # Torchvision semantic layer
│   │   ├── neural_compliance.py # Dynamic Rules Engine
│   │   └── matcher.py           # Embeddings and query logic
│   └── server/
│       └── app.py               # FastAPI MJPEG wrapper
├── scripts/
│   └── run_pipeline.py          # CLI orchestrator script
├── ui/
│   ├── src/                     # React application logic and sockets
│   └── package.json    
└── output/
    ├── frames/                  # Default CLI debug export target
    └── pipeline_results.txt
```
