"""An entry's cost to its target from its run of record, the reference cost at its start and the cost over the reference's: parts the
library prints apart (`library --records`), beside its products-term factor and the paused search's score. None of
them is the Glossary's slowdown, and nothing here combines any of them into one figure.

- **The cost, C_e** (`cost_to_target`): the gates the record counts (`grade.checkpoints`, in `gates.UNIT`) to the
  entry's target, `grade.target` at the entry's gain on the fit through the starts on its start's data
  (`grade.start_fit`), bracketed between its last scored checkpoint above the target and its first at or below; the
  bracket opens at the start, 0 gates, where the first scored checkpoint is already at or below. A record that never
  reaches the target has no cost, and says how far it got.
- **The reference, T_ref** (`reference`): per start, `REFERENCE`. `reference` is the one place it is chosen.
- **The cost to the target over the reference's, C_e / T_ref** (`ratio`, `OVER_REFERENCE`): each end of the bracket
  over the reference; a ratio of two gate counts to a target, not a slowdown.
- **The paused search's score** (`search_score`): S as the grade computes it from the record (`grade.grade`), or
  the grade's reason it has none.

Each record is read as one seed's job on the search split, the split its checkpoints are scored on. Which run is an
entry's run of record, and the record's file, is `of_record.toml`'s (`held`, `read`).
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from . import gates, grade

HELD = Path(__file__).with_name("of_record.toml")
UNIT = (f"the unit is gates ({gates.UNIT}), the count the record's checkpoints carry; the Glossary's cost, "
        "GPU-hours, is not computed")
REFERENCE = "least recorded cost at this start; moves as entries are added; no named reference has a record"
OVER_REFERENCE = "cost to the target over the reference's"
SPLIT = "search"


def held(path: Path = HELD) -> dict:
    """Each held entry's run of record: `{entry: {"run", "art", "path", "sha256"}}`; where its record holds fused
    attention backward rows and no trace, `"partial"`, the gradients those calls asked for (`as_math.parse`); where
    its record's program computes lines the route rightly doesn't, `"record_only"`, those lines (`named`); and where
    its config's trained weights leave θ after the second step, `"step"`, the step compared and why (`steps`)."""
    return tomllib.loads(Path(path).read_text())["held"]


def read(name: str, h: dict, root: Path) -> dict:
    """`name`'s record under `root`, refused unless it is the file `of_record.toml` names."""
    path = Path(root) / h["run"] / h["path"]
    fetch = f"`research data fetch {h['art']} --to {Path(root) / h['run']} --path {h['path']}`"
    if not path.is_file():
        raise SystemExit(f"of_record: {name}'s record is not at {path} ({fetch})")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != h["sha256"]:
        raise SystemExit(f"of_record: {path} is not the record {HELD.name} names for {name}: its sha256 is "
                         f"{hashlib.sha256(raw).hexdigest()} ({fetch})")
    return json.loads(raw)


def job(record: dict) -> dict:
    """A record as the grade reads one seed's job: on the search split, at the seed it states (0 where it states
    none)."""
    return {"seed": record.get("seed", 0), "split": SPLIT, "record": record}


def cost_to_target(record: dict, start: str, gain: float, starts: dict) -> dict:
    """C_e of a record from `start` at `gain` (`starts` the registry's `starts` table): `reached`, the target, and
    where reached the bracket `gates` [last scored checkpoint above the target, first at or below] with its `steps`
    (None at the bracket's low end where it opens at the start); else `why`. Raises the grade's `ValueError` where it
    refuses the record."""
    f, _ = grade.start_fit(starts, start)
    cps = grade.checkpoints(job(record), start)
    at = grade.seed_at(cps, f, gain)
    if not at["reached"]:
        return {"reached": False, "why": at.get("why") or (
            f"misses the target: its last scored checkpoint (step {at['step']}) is at gain {at['gain']:.4g} of "
            f"{gain:g}")}
    above = [c for c in cps if c["gates"] is not None and c["step"] < at["step"]]
    return {"reached": True, "target_bpb": at["target"], "fit": f,
            "gates": [above[-1]["gates"] if above else 0.0, at["C_gates"]],
            "steps": [above[-1]["step"] if above else None, at["step"]]}


def reference(costs: dict[str, tuple[str, dict]]) -> dict[str, dict]:
    """T_ref by start, from each entry's (start, `cost_to_target`): the least high end of a bracket over the entries
    that reach, with that entry's name and `REFERENCE`. A start no entry reaches at has none."""
    out: dict[str, dict] = {}
    for name, (start, c) in sorted(costs.items()):
        if c.get("reached") and (start not in out or c["gates"][1] < out[start]["gates"]):
            out[start] = {"gates": c["gates"][1], "entry": name, "label": REFERENCE}
    return out


def ratio(cost: dict, ref: dict | None) -> dict:
    """C_e / T_ref, `OVER_REFERENCE`: each end of the cost's bracket over the reference's gates, or why there is none."""
    if not cost.get("reached"):
        return {"ratio": None, "why": cost.get("why")}
    if ref is None:
        return {"ratio": None, "why": "no entry at this start reaches its target"}
    return {"ratio": [g / ref["gates"] for g in cost["gates"]]}


def search_score(config: dict, record: dict, starts: dict, honest: dict | None = None,
                 others: dict[str, dict] | None = None) -> dict:
    """The paused search's score of an entry (`config` as `grade.grade` reads one) from its record: the grade's S
    against the least C of the configs given (`S["any"]`), its honest config's record and the other entries' at the
    same start and gain, `others` by name; or the grade's reason it has none. Only a record the grade would score
    needs the others, so they are read only then."""
    def scored(given: dict[str, dict]) -> dict:
        return grade.grade(config, starts, [job(record)], [job(honest)] if honest else [],
                           {n: [job(r)] for n, r in given.items()})

    try:
        got = scored({})
        if got["S"]["why"] is None and others:
            got = scored(others)
    except ValueError as e:
        return {"score": None, "why": str(e)}
    return {"score": got["S"]["any"], "why": got["S"]["why"]}
