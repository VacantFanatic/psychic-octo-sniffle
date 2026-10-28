import io
import json
import types

import pytest
from PIL import Image

from tuner import optimizer, review, scoring
from tuner.comfy_client import apply_params
from tuner.gate import NsfwGate


def png(color=(120, 120, 120)):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


class FakeClient:
    def __init__(self, replies):
        self.requests = []
        self.replies = list(replies)
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        text = self.replies.pop(0)
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])


def test_gate_fails_closed_on_error():
    def boom(_):
        raise RuntimeError("model failed")

    assert NsfwGate(boom).is_safe(png()) is False


def test_gate_fails_closed_on_missing_label():
    assert NsfwGate(lambda _: {"normal": 1.0}).is_safe(png()) is False


def test_gate_labels():
    assert NsfwGate(lambda _: {"nsfw": 0.01}).label(png()) == "sfw"
    assert NsfwGate(lambda _: {"nsfw": 0.9}).label(png()) == "nsfw"


def test_clamp_params():
    bounds = {"steps": (10, 40), "cfg": (1.0, 12.0)}
    current = {"steps": 20, "cfg": 7.0}
    out = optimizer.clamp_params({"steps": 999, "cfg": -3, "evil": 1}, bounds, current)
    assert out == {"steps": 40, "cfg": 1.0}
    assert optimizer.clamp_params({"steps": "x"}, bounds, current) == current


def test_assert_text_only_rejects_images():
    bad = [{"role": "user", "content": [{"type": "image", "source": {}}]}]
    with pytest.raises(ValueError):
        optimizer.assert_text_only(bad)


def test_apply_params_does_not_mutate_original():
    workflow = {"3": {"inputs": {"steps": 20}}}
    patched = apply_params(workflow, {"steps": ("3", "steps")}, {"steps": 30})
    assert workflow["3"]["inputs"]["steps"] == 20
    assert patched["3"]["inputs"]["steps"] == 30


def test_loop_sends_no_image_data_and_clamps():
    nsfw_image = png((255, 0, 0))
    client = FakeClient(
        [
            '{"reasoning": "more steps", "next_params": {"steps": 500}}',
            '{"reasoning": "ok", "next_params": {"steps": 25}}',
        ]
    )
    saved = []
    history, params = optimizer.run_loop(
        client=client,
        model="m",
        goal="g",
        params={"steps": 20},
        bounds={"steps": (10, 40)},
        rounds=2,
        render_fn=lambda p: nsfw_image,
        score_fn=scoring.technical_metrics,
        gate=NsfwGate(lambda _: {"nsfw": 0.99}),
        save_fn=lambda i, img, rec: saved.append(i),
    )
    assert [h["gate"] for h in history] == ["nsfw", "nsfw"]
    assert history[1]["params"] == {"steps": 40}
    assert params == {"steps": 25}
    assert saved == [0, 1]
    for request in client.requests:
        optimizer.assert_text_only(request["messages"])
        payload = json.dumps(request)
        assert "base64" not in payload
        assert '"type": "image"' not in payload


def test_technical_metrics_keys():
    assert set(scoring.technical_metrics(png())) == {
        "sharpness",
        "brightness",
        "contrast",
    }


def test_save_results(tmp_path):
    history = [
        {
            "round": 0,
            "params": {"steps": 20},
            "metrics": {"sharpness": 0.1},
            "gate": "sfw",
            "human_rank": 1,
        }
    ]
    path = tmp_path / "results.csv"
    review.save_results(path, history)
    header = path.read_text().splitlines()[0].split(",")
    assert header[0] == "round" and "param_steps" in header


def test_thumbnail_blur_is_png():
    assert review.thumbnail(png()).startswith(b"\x89PNG")
