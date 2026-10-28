"""Minimal ComfyUI HTTP API client (runs against the local server in Colab)."""

import copy
import json
import time
import uuid

import requests

DEFAULT_URL = "http://127.0.0.1:8188"


def load_workflow(path):
    """Load a workflow exported with "Save (API Format)"."""
    with open(path) as f:
        return json.load(f)


def apply_params(workflow, tunable, params):
    """Return a copy of `workflow` with `params` written into its node inputs.

    `tunable` maps a parameter name to a (node_id, input_name) pair.
    """
    patched = copy.deepcopy(workflow)
    for name, value in params.items():
        node_id, field = tunable[name]
        patched[str(node_id)]["inputs"][field] = value
    return patched


def render(workflow, base_url=DEFAULT_URL, timeout=600, poll=1.0):
    """Queue `workflow` and return the PNG bytes of the first output image."""
    resp = requests.post(
        f"{base_url}/prompt",
        json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        timeout=30,
    )
    resp.raise_for_status()
    prompt_id = resp.json()["prompt_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        history = requests.get(f"{base_url}/history/{prompt_id}", timeout=30).json()
        if prompt_id in history:
            for output in history[prompt_id]["outputs"].values():
                if output.get("images"):
                    image = requests.get(
                        f"{base_url}/view", params=output["images"][0], timeout=60
                    )
                    image.raise_for_status()
                    return image.content
            raise RuntimeError("Workflow finished without producing an image")
        time.sleep(poll)
    raise TimeoutError(f"ComfyUI did not finish prompt {prompt_id} in {timeout}s")
