def detect_helmet_violations(tracker):
    """
    Check for frames where a 'person' is detected but no 'helmet' is present.

    Simple rule-based logic:
    - If a person is present in a frame and no helmet is detected
      in the same frame, flag it as a violation.

    Args:
        tracker: DetectionTracker instance.

    Returns:
        List of dicts with keys: timestamp, violation.
    """
    violations = []

    for timestamp, detections in tracker.get_all():
        labels = [d.get("label", "").lower() for d in detections]
        has_person = any(l == "person" for l in labels)
        has_helmet = any("helmet" in l for l in labels)

        if has_person and not has_helmet:
            violations.append({
                "timestamp": timestamp,
                "violation": "Person detected without helmet",
            })

    return violations


def check_compliance(tracker, rules=None):
    """
    Run all compliance checks.

    Args:
        tracker: DetectionTracker instance.
        rules: Optional list of rule names to run.
                Defaults to ["helmet"].

    Returns:
        Dict mapping rule name to list of violations.
    """
    if rules is None:
        rules = ["helmet"]

    results = {}

    if "helmet" in rules:
        results["helmet"] = detect_helmet_violations(tracker)

    return results
