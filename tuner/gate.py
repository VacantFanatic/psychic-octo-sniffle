"""Local NSFW gate. Fails closed: anything uncertain is treated as NSFW."""

import io

NSFW_THRESHOLD = 0.2
MODEL_ID = "Falconsai/nsfw_image_detection"


def load_classifier(model_id=MODEL_ID):
    """Return a callable mapping PNG bytes to {label: score}."""
    from PIL import Image
    from transformers import pipeline

    pipe = pipeline("image-classification", model=model_id)

    def classify(image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return {r["label"].lower(): r["score"] for r in pipe(image, top_k=None)}

    return classify


class NsfwGate:
    """Decides whether an image may ever leave the machine."""

    def __init__(self, classifier, threshold=NSFW_THRESHOLD):
        self.classifier = classifier
        self.threshold = threshold

    def is_safe(self, image_bytes):
        try:
            scores = self.classifier(image_bytes)
            nsfw = scores["nsfw"]
        except Exception:
            return False
        return nsfw < self.threshold

    def label(self, image_bytes):
        return "sfw" if self.is_safe(image_bytes) else "nsfw"
