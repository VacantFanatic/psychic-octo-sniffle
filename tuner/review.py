"""Human review: blurred thumbnails, rank them yourself, save to CSV."""

import csv
import io

from PIL import Image, ImageFilter


def thumbnail(image_bytes, blur=True, size=256):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((size, size))
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(12))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def save_results(path, history):
    """Write one row per round, with params and metrics flattened."""
    rows = []
    for entry in history:
        row = {"round": entry["round"], "gate": entry["gate"]}
        row.update({f"param_{k}": v for k, v in entry["params"].items()})
        row.update({f"metric_{k}": v for k, v in entry["metrics"].items()})
        row["human_rank"] = entry.get("human_rank", "")
        rows.append(row)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "round", k))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
