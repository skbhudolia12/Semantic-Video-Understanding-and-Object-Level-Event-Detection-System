from src.pipeline.aggregator import aggregate_segments


def find_all_occurrences(tracker, label):
    """
    Find all timestamps where a label appears.

    Args:
        tracker: DetectionTracker instance.
        label: Object label to search for.

    Returns:
        Sorted list of timestamps.
    """
    return tracker.get_timestamps_with_label(label)


def find_last_seen(tracker, label):
    """
    Find the last timestamp where a label was detected.

    Args:
        tracker: DetectionTracker instance.
        label: Object label to search for.

    Returns:
        Timestamp (float) or None if never seen.
    """
    timestamps = tracker.get_timestamps_with_label(label)
    return timestamps[-1] if timestamps else None


def check_presence(tracker, label):
    """
    Check if a label was ever detected.

    Args:
        tracker: DetectionTracker instance.
        label: Object label to search for.

    Returns:
        True if the label appears at least once.
    """
    return len(tracker.get_timestamps_with_label(label)) > 0


def explain(query, timestamps, segments):
    """
    Generate a human-readable explanation of results.

    Args:
        query: The original search query.
        timestamps: List of matching timestamps.
        segments: List of (start, end) segment tuples.

    Returns:
        Explanation string.
    """
    if not timestamps:
        return f"'{query}' was not found in the video."

    lines = [f"'{query}' was detected at {len(timestamps)} timestamp(s)."]
    lines.append(f"Grouped into {len(segments)} segment(s):")
    for i, (start, end) in enumerate(segments, 1):
        if start == end:
            lines.append(f"  Segment {i}: {start:.1f}s")
        else:
            lines.append(f"  Segment {i}: {start:.1f}s → {end:.1f}s")
    return "\n".join(lines)
