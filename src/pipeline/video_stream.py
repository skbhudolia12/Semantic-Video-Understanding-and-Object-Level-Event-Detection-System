import cv2


def stream_frames(video_path, sample_fps=1):
    """
    Read a video file and yield (frame, timestamp) at the given sample rate.

    Args:
        video_path: Path to the video file.
        sample_fps: Frames to sample per second (default: 1).

    Yields:
        (frame, timestamp) where frame is a BGR numpy array and
        timestamp is the time in seconds.
    """
    if isinstance(video_path, str) and video_path.isdigit():
        video_path = int(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # Default to 30 FPS for webcams if not reported by cap

    frame_interval = max(1, int(video_fps / sample_fps))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            yield frame, timestamp

        frame_idx += 1

    cap.release()
