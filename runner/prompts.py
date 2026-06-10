from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List

REQUIRED_FIELDS = ("id", "category", "input", "expected_output", "scoring_method")


class PromptError(ValueError):
    """Raised when a prompt file is malformed or prompt ids collide."""


def _validate_prompt(obj: Any, source: str, stem: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise PromptError(
            f"{source}: each prompt must be a JSON object, got {type(obj).__name__}"
        )
    for field in REQUIRED_FIELDS:
        if field not in obj or obj[field] in (None, ""):
            raise PromptError(
                f"{source}: prompt missing required field {field!r}: {obj!r}"
            )
    if obj["category"] != stem:
        raise PromptError(
            f"{source}: prompt {obj['id']!r} has category {obj['category']!r} "
            f"but filename stem is {stem!r} (they must match)"
        )
    scoring_args = obj.get("scoring_args", {})
    if scoring_args is None:
        scoring_args = {}
    if not isinstance(scoring_args, dict):
        raise PromptError(
            f"{source}: prompt {obj['id']!r} has non-object scoring_args: {scoring_args!r}"
        )
    return {
        "id": str(obj["id"]),
        "category": str(obj["category"]),
        "input": str(obj["input"]),
        "expected_output": str(obj["expected_output"]),
        "scoring_method": str(obj["scoring_method"]),
        "scoring_args": scoring_args,
    }


def load_prompts(dir: str = "prompts") -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(dir, "*.json")))
    prompts: List[Dict[str, Any]] = []
    seen_ids: Dict[str, str] = {}

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise PromptError(f"{path}: invalid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise PromptError(f"{path}: top level must be a JSON array of prompts")

        for obj in data:
            prompt = _validate_prompt(obj, path, stem)
            pid = prompt["id"]
            if pid in seen_ids:
                raise PromptError(
                    f"duplicate prompt id {pid!r} in {path} "
                    f"(first seen in {seen_ids[pid]})"
                )
            seen_ids[pid] = path
            prompts.append(prompt)

    prompts.sort(key=lambda p: p["id"])
    return prompts
