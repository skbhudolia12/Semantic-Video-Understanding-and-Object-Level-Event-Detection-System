import torch
import clip
from PIL import Image


_model = None
_preprocess = None
_device = None


def load_clip(model_name="ViT-B/32"):
    """
    Load CLIP model and preprocessing function.

    Args:
        model_name: CLIP architecture to load.

    Returns:
        (model, preprocess, device)
    """
    global _model, _preprocess, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _preprocess = clip.load(model_name, device=_device)
    return _model, _preprocess, _device


def encode_text(text):
    """
    Encode a text string into a CLIP embedding.

    Args:
        text: A natural language string.

    Returns:
        Normalized text embedding tensor.
    """
    model, _, device = load_clip()
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding


def encode_image(bgr_crop):
    """
    Encode a BGR numpy crop into a CLIP embedding.

    Args:
        bgr_crop: BGR numpy array (from OpenCV).

    Returns:
        Normalized image embedding tensor.
    """
    import cv2

    model, preprocess, device = load_clip()
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding
