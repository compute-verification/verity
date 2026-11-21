"""What this adapter serves (`SERVES`): the one model at its revision and precision, its vocabulary, the window class it
serves it at and the kind of input dataset it plays; and the door (`door`), which reads a dataset through against it
before anything is sent. The door reads token ids and never text, so it needs no tokenizer and no engine. The launchers
read it through `adapters/vllm/tools/served_run/inputs.py`.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.errors import Refused


def window_class_parts(text: str | None) -> tuple[int, ...]:
    """A window class's ``S,P,O[,LANES,EVERY]`` as its integers; `ValueError` naming it where it is not three to five
    positive integers."""
    parts = str(text or "").split(",")
    if not 3 <= len(parts) <= 5 or not all(p.strip().isdecimal() and int(p) >= 1 for p in parts):
        raise ValueError(f"{text!r} is not S,P,O[,LANES,EVERY] of positive integers")
    return tuple(int(p) for p in parts)


@dataclass(frozen=True)
class Serves:
    """The served model and the window class ``window_class`` (``S,P,O``, serve-openai's ``--window-class``): a window of
    at most S requests, each of at most P prompt ids and at most O outputs, greedy."""

    model: str
    revision: str
    precision: str
    vocab: int
    window_class: str
    kind: str = "completion-requests"

    @property
    def slots(self) -> int:
        return window_class_parts(self.window_class)[0]

    @property
    def prompt(self) -> int:
        return window_class_parts(self.window_class)[1]

    @property
    def outputs(self) -> int:
        return window_class_parts(self.window_class)[2]


SERVES = Serves(**json.loads((Path(__file__).parent / "serves.json").read_text()))
#: the timings the sender plays a dataset at (`input_datasets.Dataset.timed`): every request at 0, or ``rate:R`` a second
TIMINGS = ("at-once", "rate:R")


def refusal(e: Exception) -> tuple[str, str]:
    """The name and text of a refusal: the door's (`Refused`: ``kind``, ``not-served``, ``model``, ``prompt``,
    ``positions``, ``steps``, ``sampling``, ``slots``, ``timing``, ``application``) or the input dataset's read's."""
    return (e.what, "; ".join(e.reasons)) if isinstance(e, Refused) else (getattr(e, "name", type(e).__name__), str(e))


def served(model: str, revision: str, precision: str, slots: int, steps: int) -> Serves:
    """`SERVES`, where an application's workload is it: the model at its revision and precision, the class's slots and
    its outputs as the window's steps. Refused (``not-served``) otherwise."""
    said, want = (model, revision, precision, slots, steps), (SERVES.model, SERVES.revision, SERVES.precision,
                                                               SERVES.slots, SERVES.outputs)
    if said != want:
        raise Refused(what="not-served", reasons=f"the vLLM adapter serves {want[0]} at {want[1]} in {want[2]} with {want[3]} slots and "
                                    f"{want[4]} window steps (verity_vllm.inputs.serves.SERVES), not {said[0]} at {said[1]} in "
                                    f"{said[2]} with {said[3]} slots and {said[4]} window steps")
    return SERVES


def timing(said: str) -> tuple[str, dict[str, float]]:
    """``said`` (one of `TIMINGS`) as `Dataset.timed`'s rule and parameters; refused (``timing``) otherwise."""
    if said == "at-once":
        return "at-once", {}
    rule, _, rate = said.partition(":")
    try:
        if rule == "rate" and float(rate) > 0:
            return "rate", {"rate": float(rate)}
    except ValueError:
        pass
    raise Refused(what="timing", reasons=f"a timing is at-once or rate:R with R > 0 requests a second, not {said!r}")


def door(kind: str, records: Iterable[Any], s: Serves | None = None) -> list[dict[str, Any]]:
    """Every record of a dataset of ``kind``, read through once, each a `/v1/completions` body the class of ``s`` (default
    `SERVES`) takes; the first that it can't take refused by name, with its index and user."""
    s = s or SERVES
    if kind != s.kind:
        raise Refused(what="kind", reasons=f"the vLLM adapter plays input datasets of kind {s.kind}, not {kind}")
    out = list(records)
    for j, b in enumerate(out):
        at = f"record {j} ({b.get('user') if isinstance(b, dict) else None})"
        if not isinstance(b, dict) or b.get("model") != s.model:
            raise Refused(what="model", reasons=f"{at} asks for model {b.get('model') if isinstance(b, dict) else b!r}, and the "
                                   f"adapter serves {s.model}")
        p = b.get("prompt")
        if not isinstance(p, list) or not p or not all(type(i) is int and 0 <= i < s.vocab for i in p):
            raise Refused(what="prompt", reasons=f"{at}: its prompt is not a non-empty list of token ids below the vocabulary's "
                                    f"{s.vocab}")
        if len(p) > s.prompt:
            raise Refused(what="positions", reasons=f"{at}: its prompt is {len(p)} ids, past the class {s.window_class}'s {s.prompt}")
        if type(b.get("max_tokens")) is not int or not 1 <= b["max_tokens"] <= s.outputs:
            raise Refused(what="steps", reasons=f"{at}: max_tokens {b.get('max_tokens')!r}, not 1 to the class {s.window_class}'s "
                                   f"{s.outputs} outputs")
        if b.get("temperature") != 0 or b.get("n", 1) != 1:
            raise Refused(what="sampling", reasons=f"{at}: temperature {b.get('temperature')!r} and n {b.get('n', 1)!r}, and the "
                                      "class serves one greedy completion a request")
    if len(out) > s.slots:
        raise Refused(what="slots", reasons=f"the dataset has {len(out)} records, past the class {s.window_class}'s {s.slots} slots")
    return out


def application(path: str) -> tuple[str, str]:
    """The input dataset and timing an application (``applications/NAME.toml``) names in its ``[workload.params]``, once
    `served` takes its model and window; refused (``application``) when it names no dataset and timing."""
    import tomllib
    try:
        params = tomllib.loads(open(path, encoding="utf-8").read())["workload"]["params"]
        served(params["model"], params["revision"], params["precision"], params["slots"], params["window-steps"])
        return params["dataset"], params["timing"]
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError) as e:
        raise Refused(what="application", reasons=f"{path} names no model, window, dataset and timing in [workload.params] "
                                     f"({type(e).__name__}: {e})") from e


def door_line(name: str, said: str) -> str:
    """The door's answer over the input dataset ``name`` at the timing ``said``: its records and identity; refused by name."""
    import input_datasets
    ds = input_datasets.open(name)
    rule, parameters = timing(said)
    timed = ds.timed(rule, **parameters)
    n = len(door(ds.kind, (b for _, b in timed)))
    return f"DOOR {name}: {n} records the class {SERVES.window_class} takes; identity {timed.identity}"


__all__ = ["SERVES", "TIMINGS", "Refused", "Serves", "application", "door", "door_line", "refusal", "served", "timing", "window_class_parts"]
