def aggregate_segments(timestamps, gap_threshold=2.0):
    """
    Group a sorted list of timestamps into contiguous segments.

    A new segment starts when the gap between consecutive timestamps
    exceeds gap_threshold seconds.

    Args:
        timestamps: Sorted list of timestamps (seconds).
        gap_threshold: Max gap (in seconds) to consider part of the same segment.

    Returns:
        List of (start, end) tuples.
    """
    if not timestamps:
        return []

    timestamps = sorted(timestamps)
    segments = []
    seg_start = timestamps[0]
    seg_end = timestamps[0]

    for t in timestamps[1:]:
        if t - seg_end <= gap_threshold:
            seg_end = t
        else:
            segments.append((seg_start, seg_end))
            seg_start = t
            seg_end = t

    segments.append((seg_start, seg_end))
    return segments
