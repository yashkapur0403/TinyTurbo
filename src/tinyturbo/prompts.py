"""Loading and validating the fixed prompt set.

Prompt texts live in the repository as JSON Lines rather than being pulled from a
dataset at run time. Two reasons: a run then needs no network access, and the exact
inputs can never change underneath a recorded result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ("prompt_id", "stratum", "text")


class PromptError(ValueError):
    """Raised when the prompt file is missing, malformed or ambiguous."""


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    stratum: str
    text: str
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "stratum": self.stratum,
            "text": self.text,
            "notes": self.notes,
        }


def load_prompts(path: str | Path) -> list[Prompt]:
    """Read a JSONL prompt file and return the prompts in file order.

    Every line must be a JSON object with a unique `prompt_id`, a `stratum` and a
    non-empty `text`. Failing loudly here is important: a silently skipped prompt
    would quietly shrink the experiment.
    """
    path = Path(path)
    if not path.is_file():
        raise PromptError(f"prompt file not found: {path}")

    prompts: list[Prompt] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromptError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise PromptError(f"{path}:{line_number} must be a JSON object")

            missing = [field for field in REQUIRED_FIELDS if not payload.get(field)]
            if missing:
                raise PromptError(f"{path}:{line_number} is missing: {', '.join(missing)}")

            prompt_id = str(payload["prompt_id"])
            if prompt_id in seen:
                raise PromptError(f"{path}:{line_number} duplicates prompt_id '{prompt_id}'")
            seen.add(prompt_id)

            prompts.append(
                Prompt(
                    prompt_id=prompt_id,
                    stratum=str(payload["stratum"]),
                    text=str(payload["text"]),
                    notes=str(payload.get("notes", "")),
                )
            )

    if not prompts:
        raise PromptError(f"{path} contains no prompts")
    return prompts


def get_prompt(prompts: list[Prompt], prompt_id: str) -> Prompt:
    """Look up one prompt by id, listing the valid ids if the id is unknown."""
    for prompt in prompts:
        if prompt.prompt_id == prompt_id:
            return prompt
    available = ", ".join(prompt.prompt_id for prompt in prompts)
    raise PromptError(f"unknown prompt_id '{prompt_id}'. Available: {available}")


def prompt_set_hash(prompts: list[Prompt]) -> str:
    """A stable hash of the prompt set, recorded so two runs can be proven comparable."""
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.prompt_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(prompt.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()
