from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "request.default.json"
OUTPUT_PATH = ROOT / "request.json"


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.S)
    candidate = fenced.group(1) if fenced else text
    first, last = candidate.find("{"), candidate.rfind("}")
    if first < 0 or last <= first:
        return {}
    value = json.loads(candidate[first:last + 1])
    if not isinstance(value, dict):
        raise ValueError("Backtest request must be a JSON object")
    return value


def main() -> None:
    request = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    issue_number = ""
    raw_text = os.environ.get("REQUEST_JSON", "")
    if event_path and Path(event_path).exists():
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
        issue = event.get("issue") or {}
        comment = event.get("comment") or {}
        inputs = event.get("inputs") or {}
        issue_number = str(issue.get("number") or "")
        raw_text = str(inputs.get("request_json") or comment.get("body") or issue.get("body") or raw_text)
    override = extract_json(raw_text)
    request.update(override)
    OUTPUT_PATH.write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / ".issue_number").write_text(issue_number, encoding="utf-8")
    print(json.dumps(request, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
