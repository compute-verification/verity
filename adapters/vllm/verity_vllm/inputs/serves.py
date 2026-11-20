"""What this adapter serves (`SERVES`): the one model at its revision and precision, its vocabulary, the window class it
serves it at and the kind of input dataset it plays; and the door (`door`), which reads a dataset through against it
before anything is sent. The door reads token ids and never text, so it needs no tokenizer and no engine.

    python -m verity_vllm.inputs.serves                      # MODEL REVISION PRECISION CLASS, one line, as the launchers read it
    python -m verity_vllm.inputs.serves door NAME [TIMING]   # the door over the input dataset NAME, refused by name or its identity
    python -m verity_vllm.inputs.serves application FILE     # MODEL REVISION PRECISION CLASS DATASET TIMING, then the door's line
"""
from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


class Refused(Exception):
    """A dataset or an application the statement can't take, by ``name``: ``kind``, ``not-served``, ``model``,
    ``prompt``, ``positions``, ``steps``, ``sampling``, ``slots``, ``timing``, ``application``."""

    def __init__(self, name: str, text: str) -> None:
        super().__init__(text)
        self.name = name


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
        return int(self.window_class.split(",")[0])

    @property
    def prompt(self) -> int:
        return int(self.window_class.split(",")[1])

    @property
    def outputs(self) -> int:
        return int(self.window_class.split(",")[2])


SERVES = Serves("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da", "bfloat16", 128256, "4,27,2")
#: the timings the sender plays a dataset at (`input_datasets.Dataset.timed`): every request at 0, or ``rate:R`` a second
TIMINGS = ("at-once", "rate:R")


def served(model: str, revision: str, precision: str, slots: int, steps: int) -> Serves:
    """`SERVES`, where an application's workload is it: the model at its revision and precision, the class's slots and
    its outputs as the window's steps. Refused (``not-served``) otherwise."""
    said, want = (model, revision, precision, slots, steps), (SERVES.model, SERVES.revision, SERVES.precision,
                                                               SERVES.slots, SERVES.outputs)
    if said != want:
        raise Refused("not-served", f"the vLLM adapter serves {want[0]} at {want[1]} in {want[2]} with {want[3]} slots and "
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
    raise Refused("timing", f"a timing is at-once or rate:R with R > 0 requests a second, not {said!r}")


def door(kind: str, records: Iterable[Any], s: Serves | None = None) -> list[dict[str, Any]]:
    """Every record of a dataset of ``kind``, read through once, each a `/v1/completions` body the class of ``s`` (default
    `SERVES`) takes; the first that it can't take refused by name, with its index and user."""
    s = s or SERVES
    if kind != s.kind:
        raise Refused("kind", f"the vLLM adapter plays input datasets of kind {s.kind}, not {kind}")
    out = list(records)
    for j, b in enumerate(out):
        at = f"record {j} ({b.get('user') if isinstance(b, dict) else None})"
        if not isinstance(b, dict) or b.get("model") != s.model:
            raise Refused("model", f"{at} asks for model {b.get('model') if isinstance(b, dict) else b!r}, and the "
                                   f"adapter serves {s.model}")
        p = b.get("prompt")
        if not isinstance(p, list) or not p or not all(type(i) is int and 0 <= i < s.vocab for i in p):
            raise Refused("prompt", f"{at}: its prompt is not a non-empty list of token ids below the vocabulary's "
                                    f"{s.vocab}")
        if len(p) > s.prompt:
            raise Refused("positions", f"{at}: its prompt is {len(p)} ids, past the class {s.window_class}'s {s.prompt}")
        if type(b.get("max_tokens")) is not int or not 1 <= b["max_tokens"] <= s.outputs:
            raise Refused("steps", f"{at}: max_tokens {b.get('max_tokens')!r}, not 1 to the class {s.window_class}'s "
                                   f"{s.outputs} outputs")
        if b.get("temperature") != 0 or b.get("n", 1) != 1:
            raise Refused("sampling", f"{at}: temperature {b.get('temperature')!r} and n {b.get('n', 1)!r}, and the "
                                      "class serves one greedy completion a request")
    if len(out) > s.slots:
        raise Refused("slots", f"the dataset has {len(out)} records, past the class {s.window_class}'s {s.slots} slots")
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
        raise Refused("application", f"{path} names no model, window, dataset and timing in [workload.params] "
                                     f"({type(e).__name__}: {e})") from e


def _door_line(name: str, said: str) -> str:
    import input_datasets
    ds = input_datasets.open(name)
    rule, parameters = timing(said)
    timed = ds.timed(rule, **parameters)
    n = len(door(ds.kind, (b for _, b in timed)))
    return f"DOOR {name}: {n} records the class {SERVES.window_class} takes; identity {timed.identity}"


def main(argv: list[str]) -> int:
    if not argv:
        print(SERVES.model, SERVES.revision, SERVES.precision, SERVES.window_class)
        return 0
    if not (argv[0] == "door" and len(argv) in (2, 3) or argv[0] == "application" and len(argv) == 2):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    import input_datasets
    try:
        if argv[0] == "application":
            name, said = application(argv[1])
            line = _door_line(name, said)
            print(SERVES.model, SERVES.revision, SERVES.precision, SERVES.window_class, name, said)
        else:
            line = _door_line(argv[1], argv[2] if len(argv) == 3 else "at-once")
    except (Refused, input_datasets.Refused) as e:
        print(f"REFUSED {e.name}: {e}")
        return 3
    print(line)
    return 0


__all__ = ["SERVES", "TIMINGS", "Refused", "Serves", "application", "door", "served", "timing"]

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
