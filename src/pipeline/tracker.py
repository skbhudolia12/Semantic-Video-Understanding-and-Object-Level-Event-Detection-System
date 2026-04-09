class DetectionTracker:
    """
    Stores detections over time, keyed by timestamp.
    """

    def __init__(self):
        self.history = []  # list of (timestamp, detections)

    def add(self, timestamp, detections):
        """
        Record detections for a given timestamp.

        Args:
            timestamp: Time in seconds.
            detections: List of detection dicts.
        """
        self.history.append((timestamp, detections))

    def get_all(self):
        """Return full detection history."""
        return self.history

    def get_timestamps_with_label(self, label):
        """
        Get all timestamps where a given label was detected.

        Args:
            label: Object class label string.

        Returns:
            List of timestamps.
        """
        timestamps = []
        for ts, dets in self.history:
            for d in dets:
                if d.get("label") == label:
                    timestamps.append(ts)
                    break
        return timestamps
