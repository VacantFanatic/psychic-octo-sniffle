"""Local image metrics. Nothing here sends image data off the machine."""

import io

import numpy as np
from PIL import Image


def technical_metrics(image_bytes):
    """Sharpness (Laplacian variance), brightness and contrast on a 0-1 scale."""
    gray = np.asarray(
        Image.open(io.BytesIO(image_bytes)).convert("L"), dtype=np.float64
    )
    gray /= 255.0
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return {
        "sharpness": float(lap.var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
    }


def load_clip_scorer(model_id="openai/clip-vit-large-patch14"):
    """Return a callable (image_bytes, prompt) -> cosine similarity."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    def score(image_bytes, prompt):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(
            text=[prompt], images=image, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            out = model(**inputs)
        return float(
            torch.nn.functional.cosine_similarity(
                out.image_embeds, out.text_embeds
            ).item()
        )

    return score


def score_image(image_bytes, prompt=None, clip_scorer=None):
    metrics = technical_metrics(image_bytes)
    if clip_scorer is not None and prompt:
        metrics["clip_similarity"] = clip_scorer(image_bytes, prompt)
    return metrics
