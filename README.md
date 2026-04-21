# Semantic Video Understanding and Object-Level Event Detection System

This project combines classical video processing with modern vision-language models for two main workflows:

1. Offline video search and compliance checks from the CLI.
2. A live web dashboard for webcam or uploaded-video monitoring with natural-language rules.

At a high level the system uses YOLO for object detection, CLIP for semantic verification, optional ByteTrack IDs for object continuity, an optional semantic context filter, and a rule engine that can evaluate plain-English conditions such as "person on a bike without a helmet".

## Current Project Structure

```text
src/
  models/
    clip_model.py
    onnx_clip_encoder.py
    yolo_model.py
  pipeline/
    aggregator.py
    byte_tracker.py
    compliance.py
    context_filter.py
    detector.py
    matcher.py
    neural_compliance.py
    reasoning.py
    rule_engine.py
    rule_parser.py
    tracker.py
    video_stream.py
  server/
    app.py
scripts/
  run_pipeline.py
  export_clip_onnx.py
  debug_detections.py
tests/
ui/
start_server.bat
```

## Main Components

### CLI pipeline

The CLI entrypoint is [scripts/run_pipeline.py](scripts/run_pipeline.py). It builds a `HybridPipeline` that:

- samples frames from a video with `stream_frames()`
- runs YOLO detection
- optionally matches detections against a natural-language query with CLIP
- optionally attaches ByteTrack IDs
- optionally tags detections with semantic scene validity
- records detection history for reasoning and compliance summaries

This path is useful for offline experiments, debugging, and batch-style video analysis.

### Web backend

The FastAPI server lives in [src/server/app.py](src/server/app.py). It provides:

- `POST /upload` for local video uploads
- `GET /video_feed/{stream_id}` for an MJPEG stream
- `GET /rules`, `POST /rules`, and `DELETE /rules/{rule_id}` for rule management
- `WS /ws/stats/{stream_id}` for live per-frame detection and rule results

The backend reuses `HybridPipeline` for base detections, then augments detections for rule evaluation with CLIP-based verification and rule-specific matching.

### Natural-language rules

The rule flow is:

1. A user enters plain English in the dashboard.
2. [src/pipeline/rule_parser.py](src/pipeline/rule_parser.py) sends that text to OpenAI and converts it into a structured `ParsedRule`.
3. [src/pipeline/rule_engine.py](src/pipeline/rule_engine.py) evaluates the parsed rule against each frame's detections.
4. Matching or violating rules are drawn on the stream and pushed to the UI over WebSockets.

Rule parser logs are written to `output/gpt_rule_log.jsonl` during local runs.

### Frontend

The React dashboard is in [ui/](ui/). The main screen is [ui/src/App.jsx](ui/src/App.jsx) and it:

- uploads a file or uses webcam input
- creates and deletes live rules
- opens the backend MJPEG stream
- listens to WebSocket updates
- shows active rules and an event log in real time

## Environment Setup

### Python

Python 3.10+ is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the main Python dependencies:

```bash
pip install ultralytics opencv-python torch torchvision numpy pillow
pip install fastapi uvicorn python-multipart openai python-dotenv
pip install clip-by-openai supervision onnxruntime
```

Notes:

- `supervision` is needed for ByteTrack support.
- `onnxruntime` is used by the ONNX CLIP image encoder path.
- `clip-by-openai` provides the `clip` Python module imported by the project.

### Frontend

Use Node.js 18+.

```bash
cd ui
npm install
```

### Environment variables

Create a local `.env` file from `.env.example` and provide your key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The server loads `.env` automatically on startup. Do not commit your local `.env`.

## Running the Project

### Option 1: Web dashboard

Start the backend:

Windows:

```bat
start_server.bat
```

Or directly:

```bash
python -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload
```

Start the frontend in a second terminal:

```bash
cd ui
npm run dev
```

Then open:

```text
http://localhost:5173
```

### Option 2: CLI pipeline

Example:

```bash
python scripts/run_pipeline.py ^
  --video path_to_video.mp4 ^
  --query "person with a helmet" ^
  --fps 2 ^
  --compliance ^
  --show
```

Useful flags:

- `--no-tracker`
- `--no-context`
- `--neural-compliance`
- `--attribute-matching`
- `--save-frames`

## Tests

Run the Python test suite with:

```bash
python -m unittest discover -s tests -v
```

## Local Artifacts

These are intentionally local-only and ignored by Git:

- `.env`
- `.claude/`
- `output/`
- `models/`

That keeps secrets, generated checkpoints, logs, and temporary outputs out of GitHub.
