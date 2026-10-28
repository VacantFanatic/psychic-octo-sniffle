"""Claude-driven tuning loop. Requests contain text only, never images."""

import json
import re

SYSTEM_PROMPT = (
    "You tune ComfyUI generation parameters. You are given numeric metrics and "
    "optional human rankings for previous attempts, never images. Goal: {goal}\n"
    "Reply with JSON only: "
    '{{"reasoning": str, "next_params": {{<param>: number}}}}. '
    "Stay inside the stated bounds."
)


def clamp_params(params, bounds, current):
    """Keep only known params, clamped to bounds; fall back to `current`."""
    result = dict(current)
    for name, (low, high) in bounds.items():
        value = params.get(name, current[name])
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        value = min(max(value, low), high)
        if isinstance(current[name], int):
            value = int(round(value))
        result[name] = value
    return result


def build_messages(history, bounds):
    text = json.dumps({"bounds": bounds, "history": history}, indent=2)
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def assert_text_only(messages):
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            continue
        for block in content:
            if block.get("type") != "text":
                raise ValueError(f"Non-text block in Claude request: {block['type']}")


def parse_reply(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object in Claude reply")
    return json.loads(match.group(0))


def ask_claude(client, model, goal, history, bounds):
    messages = build_messages(history, bounds)
    assert_text_only(messages)
    reply = client.messages.create(
        model=model,
        max_tokens=800,
        system=SYSTEM_PROMPT.format(goal=goal),
        messages=messages,
    )
    return parse_reply(reply.content[0].text)


def run_loop(
    *,
    client,
    model,
    goal,
    params,
    bounds,
    rounds,
    render_fn,
    score_fn,
    gate,
    save_fn=None,
    human_rank_fn=None,
):
    """Render, score, and ask Claude for the next parameters.

    `render_fn(params) -> bytes`, `score_fn(bytes) -> dict`,
    `save_fn(round, bytes, record)` persists the image locally, and
    `human_rank_fn(history) -> {round: rank}` can add your rankings.
    """
    history = []
    for i in range(rounds):
        image = render_fn(params)
        record = {
            "round": i,
            "params": params,
            "metrics": score_fn(image),
            "gate": gate.label(image),
        }
        if save_fn:
            save_fn(i, image, record)
        history.append(record)
        if human_rank_fn:
            ranks = human_rank_fn(history) or {}
            for entry in history:
                if entry["round"] in ranks:
                    entry["human_rank"] = ranks[entry["round"]]
        proposal = ask_claude(client, model, goal, history, bounds)
        params = clamp_params(proposal.get("next_params", {}), bounds, params)
    return history, params
