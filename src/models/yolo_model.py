from ultralytics import YOLO


def load_yolo(model_name="yolov8n.pt"):
    """
    Load a YOLOv8 model.

    Args:
        model_name: Model weight file (default: yolov8n.pt).

    Returns:
        A YOLO model instance.
    """
    model = YOLO(model_name)
    return model
