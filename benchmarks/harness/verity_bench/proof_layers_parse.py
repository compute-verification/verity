"""What the readers of `verity bench prove layers` (`proof_layers`) share: an output's tagged lines, its text and
numbers, its device, sample, ε at δ, Lean verdicts and commit, and the challenger its coins came from."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from verity_bench.proof_layers_result import (
    NOT_STATED,
)


def _challenger(source: Any, challenger: Any) -> str | None:
    """`local challenger` where the output names a challenger and says it was the local one; else None."""

    return "local challenger" if source == "challenger" and challenger == "local" else None


def _tagged(text: str, tag: str, bad: list[str]) -> list[dict[str, Any]]:
    """Each line of `text` that opens with `tag` and a space, its JSON; a line that isn't JSON goes to `bad`."""

    out = []
    for line in text.splitlines():
        if line.startswith(tag + " "):
            try:
                doc = json.loads(line[len(tag) + 1:])
            except ValueError as e:
                bad.append(f"a {tag} line isn't JSON ({e}): it may be cut short")
                continue
            if isinstance(doc, dict):
                out.append(doc)
    return out


def _text(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def _number(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _device(timings: Iterable[Mapping[str, Any]]) -> str:
    """The prover's device as a5's summary ``timing`` states it for every member (``gpus``, ``device``: a5.proving_of):
    the CPU, the cards named, else "not stated"."""

    said = [((t or {}).get("gpus"), (t or {}).get("device")) for t in timings]
    if said and all(g == 0 for g, _ in said):
        return "CPU (timing gpus 0)"
    if said and all(isinstance(g, int) and g >= 1 for g, _ in said):
        return "GPU " + ", ".join(sorted({str(d) for _, d in said})) + " (timing device)"
    return NOT_STATED


def _eps_delta(delta: Any) -> str:
    return f"ε {NOT_STATED} at δ {delta}" if _number(delta) is not None else NOT_STATED


def _sampled(counts: Mapping[str, Any], population: Any, ranks: int = 1) -> str:
    sampled = ", ".join(f"{t} x {n}" for t, n in sorted(counts.items())) or "no proof block"
    return f"{sampled} sampled of {population if population is not None else '?'} proof blocks" + (f" over {ranks} ranks" if ranks > 1 else "")


def _lean_verdicts(a5_lines: Sequence[Mapping[str, Any]]) -> list[tuple[str, Any]]:
    return [(k, v) for x in a5_lines for k, v in ((x.get("audit") or {}).get("verdicts") or {}).items()
            if k.startswith("lean-verify")]


def _commit(*paths: Path, after: str = "") -> str | None:
    """The commit the first of `paths` that exists states: its first whole git sha that follows `after` (prove_path's
    run.sh's ``git rev-parse HEAD`` in machine.txt; layers.sh's job.log, after ``source ``, so no sha printed before it
    stands in), never a part of a longer digest."""

    for path in paths:
        if path.is_file():
            found = re.search(re.escape(after) + r"(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])", _text(path))
            return found.group(1) if found else None
    return None


def _delta(x: float) -> str:
    e = math.log2(x) if x > 0 else 0.0
    return f"2^{e:.0f}" if e == round(e) and e < -3 else re.sub(r"e-0*(\d)", r"e-\1", f"{x:g}")
