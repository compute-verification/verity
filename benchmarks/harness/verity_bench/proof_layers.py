"""`verity bench prove layers`: the proof system by layer, one table of four rows (Flock, Flock with zero knowledge, with
sampling, with recursion), each read from what its own command wrote (benchmarks/prove/README.md, "The proof system by
layer").

    verity bench prove layers [--flock DIR|RUN] [--zk DIR|RUN] [--sampling DIR|RUN] [--recursion DIR|RUN]

DIR is a run's ``out/`` (or the run directory that holds it), read with no store; RUN is a run id, whose records are
fetched through the evidence store where `research` is installed. A layer not given is a row that says "not run". Which
reader a directory takes is decided by what it holds (`detect`):

- ``recursion_e2e.json``: `benchmarks/recursion_e2e`'s job (`read_recursion`);
- an arm's ``audit.log`` with its ``PROVE-PATH`` line, or its ``RECORDS`` line under ``PP_RECORDS=1``:
  `benchmarks/prove_path/run.sh` (`read_prove_path`);
- ``summary.txt``, ``a5.log`` or the audit's ``summary.json``: `benchmarks/one_stage/a5_llama1b_h200.sh`, or `a5.sh`'s
  out (`read_a5`).

A row prints its numbers only when the Lean verifier's verdict, as its output states it, is an accept; any other row
prints none, with why. Its ``ran with: coins COINS; parameters PARAMETERS (DETAIL)`` line names only what its output
says outright (`COINS`, `PARAMETERS`, "not stated" for a half it doesn't say, ``ran with: not stated`` for both), and
the JSON keeps the output's own words as ``detail``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from verity_bench import kit

SCHEMA = "verity-bench/proof-layers/v1"
#: each layer's flag and its row's title, in the table's order
LAYERS = {"flock": "Flock", "zk": "Flock with zero knowledge", "sampling": "with sampling", "recursion": "with recursion"}
RECURSION, PROVE_PATH, A5, LAYERS_SH = "recursion_e2e", "prove_path", "a5", "layers.sh"
#: the outputs a layer's row may be read from
ACCEPTS = {"flock": (PROVE_PATH, A5, LAYERS_SH), "zk": (PROVE_PATH, A5, LAYERS_SH), "sampling": (A5, PROVE_PATH, LAYERS_SH),
           "recursion": (RECURSION,)}
#: the directory of a run each layer's output is in, where the run's root holds none itself: a layers.sh run's
#: (`benchmarks/one_stage/layers.sh all`), or a prove_path served-arm run's (``flock/out``, ``zk/out``)
LAYERS_SH_DIRS = {"flock": "flock", "zk": "zk", "sampling": "sampled"}
#: the words a layers.sh output states its values are stand-ins in, which its row's workload opens with
STAND_IN = "stand-in values, never served"
#: the proof a row of the layer must state it ran, whichever command wrote it: the sampling layer is Flock with zero
#: knowledge at the sampling plan
PROOF = {"flock": "plain", "zk": "zk", "sampling": "zk"}
#: the name the report writes its JSON under
REPORT = "proof-layers"
#: what a runner's metric id may name (`benchmarks/prove/run.py`'s ``LAYER_METRICS``), each with the unit it is written in
FIGURES = {"prover_s": "s a proof block", "lean_s": "s a proof block", "proof_bytes": "bytes a proof block"}
#: the layers each metric id holds, in the order they are written, each with its member: the sampling layer, the system
#: as served, is the id's own reading, written last with no member
MEMBERS = {"flock": "flock", "zk": "zero knowledge", "sampling": None}
#: a research run's id, which the store fetches
RUN_ID = re.compile(r"^r\d{8}-\d{6}-[0-9a-f]{4}$")
#: the files of a run's record the readers read, fetched from the store by these globs
FETCH = ("out/recursion_e2e.json", "out/*audit.log", "out/*summary.json", "out/summary.txt", "out/verdicts.txt",
         "out/machine.txt", "out/*a5.log", "*/out/layers.txt", "*/out/job.log", "*/out/a6/summary.json",
         "*/out/a6/session-*/lean-verify.log", "*/gpu-lease-usage.jsonl", "*/out/machine.txt", "*/out/ran-with.txt",
         "*/out/*audit.log", "*/out/*summary.json", "*/out/*-forged/commit.log")
NOT_STATED = "not stated"
#: what a row's workload opens with where its output states neither stand-in nor served values
VALUES_NOT_STATED = "values not stated"
#: what a run id that can't be fetched here takes instead
FIX = "give the run's out/ as a directory instead (where the store's read key is: `research fetch RUN`, then ~/.research/runs/RUN)"
#: the line a table prints unless every row ran on a deployed challenger's coins at deployed parameters, which this
#: command never takes as an audit: it checks no challenger's signature
NOT_AN_AUDIT = "a measurement, not an audit"
#: what a ``ran with:`` line may say a row's coins and parameters were, each only where its output says so outright
COINS = ("deployed challenger", "local challenger", "in-process")
PARAMETERS = ("deployed", "development")
#: a comma that ends a clause of recursion_e2e's ``inputs``, not one inside a Definition's braces
CLAUSE = re.compile(r",(?![^{]*\})")


class Unreadable(Exception):
    """A row's output can't be found or read."""


@dataclass
class Row:
    """One layer's row: what was given, what its output states, and why it has no number where it has none."""

    layer: str
    status: str = "not run"
    given: str | None = None
    format: str | None = None
    workload: str = "not run"
    lean: str = "not run"
    accepted: bool = False
    prover_s: float | None = None
    prover_unit: str = "s"
    per: str = ""
    device: str = NOT_STATED
    eps_delta: str = NOT_STATED
    prover_what: str = ""
    lean_s: float | None = None
    lean_what: str = ""
    proof_bytes: float | None = None
    bytes_what: str = ""
    per_block: bool = False
    ran_with: str = "ran with: not stated"
    detail: str | None = None
    proof: str | None = None
    why: str | None = None
    notes: list[str] = field(default_factory=list)
    #: the commit its output states it ran at, and which of its files states it
    commit: str | None = None
    commit_from: str | None = None
    #: a layers.sh row's figures, one a statement (its name, units, Flock's and Lean's seconds, its proof's bytes)
    statements: list[dict[str, Any]] = field(default_factory=list)

    def withhold(self, why: str) -> None:
        self.prover_s = self.lean_s = self.proof_bytes = None
        self.statements = []
        self.why = self.why or why

    def said(self, coins: str | None, parameters: str | None, detail: str | None) -> None:
        """The row's ``ran with:`` line from what its output states (`COINS`, `PARAMETERS`, None where it doesn't say),
        and `detail`, the output's own words, beside it."""

        assert coins in (None, *COINS) and parameters in (None, *PARAMETERS)
        self.detail = detail or None
        self.ran_with = ("ran with: not stated" if coins is None and parameters is None else
                         f"ran with: coins {coins or NOT_STATED}; parameters {parameters or NOT_STATED}"
                         + (f" ({detail})" if detail else ""))


def _challenger(source: Any, challenger: Any) -> str | None:
    """`local challenger` where the output names a challenger and says it was the local one; else None."""

    return "local challenger" if source == "challenger" and challenger == "local" else None


# ---- finding a row's output ----------------------------------------------------------------------------------------------


def fetch(run: str) -> Path:
    """The root of `run`'s record as the evidence store holds it, with `FETCH`'s files fetched."""

    try:
        from research.store import labels_sync
        from research.store.local import LocalStore, StoreError
    except ImportError as e:
        raise Unreadable(f"{run}: the evidence store's client isn't installed here ({e}): {FIX}") from e
    store = LocalStore()
    try:
        attempt = store.get_attempt(run)
        if attempt is None:
            raise Unreadable(f"{run} is not in the evidence store this machine reads: {FIX}")
        outputs = attempt.get("outputs") or {}
        art = outputs.get("run_record") or outputs.get("run_files")
        if not art:
            raise Unreadable(f"{run} kept no record")
        labels_sync.read_through(store, [run, art], lambda m: print(f"prove layers: {m}", file=sys.stderr))
        if why := store.withdrawn(art) or store.withdrawn(run):
            raise Unreadable(f"{run} was withdrawn ({why}): nothing of it is read")
        return Path(store.fetch(art, paths=list(FETCH)))
    except StoreError as e:
        raise Unreadable(f"{run}'s record can't be fetched: {e}: {FIX}") from e


def detect(d: Path) -> str | None:
    """Which command's output `d` holds, by its files; None where it holds none of them."""

    if (d / "recursion_e2e.json").is_file():
        return RECURSION
    if (d / "audit.log").is_file() or any(d.glob("*/audit.log")):
        return PROVE_PATH
    if (d / "layers.txt").is_file() and (d / "a6" / "summary.json").is_file():
        return LAYERS_SH
    if any((d / f).is_file() for f in ("summary.txt", "a5.log", "a5/a5.log", "a5/a5/summary.json", "a5/summary.json")):
        return A5
    return None


def locate(given: str, fetcher: Callable[[str], Path]) -> Path:
    """The directory `given` names: a directory (its ``out/`` where it holds no output itself), or a run id's fetched
    ``out/``."""

    p = Path(given)
    if p.is_dir():
        return p / "out" if detect(p) is None and (p / "out").is_dir() else p
    if RUN_ID.fullmatch(given):
        root = fetcher(given)
        return root / "out" if (root / "out").is_dir() else root
    raise Unreadable(f"{given} is neither a directory nor a run id")


# ---- reading -------------------------------------------------------------------------------------------------------------


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


def per_unit(timing: Mapping[str, Any], counts: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """An a5 summary's ``timing`` per sampled proof block of each type, in the form prove_path's ``per_unit`` prints: the
    blocks drawn (``draw_counts``), Flock's ``prove_total_s`` and Lean's verify seconds over them, and one rep's proof's
    bytes where every rep's are the same (a timed repeat of the one proof), else none."""

    out = {}
    for key, t in timing.items():
        tm, n = (t or {}).get("timing") or {}, counts.get(key) or 0
        pb = tm.get("proof_bytes")
        one = pb[0] if isinstance(pb, list) and pb and len(set(pb)) == 1 else pb if isinstance(pb, int) else None
        out[key] = {"units": n, "proof_bytes": one,
                    **{f"{x}_per_unit": v / n if n and (v := _number(raw)) is not None else None
                       for x, raw in (("prove_s", tm.get("prove_total_s")), ("lean_verify_s", (t or {}).get("lean_verify_s")))}}
    return out


def _per_block(row: Row, units: Iterable[Mapping[str, Mapping[str, Any]]],
               bytes_what: str = "one rep's proof over its drawn blocks") -> None:
    """The row's prover and Lean seconds and proof bytes a block, over every drawn block of `units` (prove_path's
    ``per_unit`` maps, one a rank): the seconds' means weighted by each type's blocks, and the bytes' sum over the blocks."""

    n, figures = 0, {"prove": 0.0, "lean": 0.0, "bytes": 0.0}
    missing: set[str] = set()
    for per in units:
        for key, x in per.items():
            k = int(x.get("units") or 0)
            if not k:
                row.notes.append(f"no per-block figure: member {key}'s timing has no drawn-block count (draw_counts)")
            n += k
            for name, v in (("prove", _number(x.get("prove_s_per_unit"))), ("lean", _number(x.get("lean_verify_s_per_unit")))):
                missing |= {name} if v is None else set()
                figures[name] += (v or 0.0) * k
            b = _number(x.get("proof_bytes"))
            missing |= {"bytes"} if b is None else set()
            figures["bytes"] += b or 0.0
    if not n:
        row.notes.append("no drawn block holds a timing")
        return
    row.notes += [f"no per-block {name} figure: a drawn member's timing doesn't give one" for name in sorted(missing)]
    row.prover_unit, row.per, row.per_block = "s a block", "a drawn block", True
    row.prover_s = None if "prove" in missing else figures["prove"] / n
    row.prover_what = f"Flock's prove_total_s, the mean over {n}"
    row.lean_s = None if "lean" in missing else figures["lean"] / n
    row.lean_what = f"Lean's verify, the mean over {n}"
    row.proof_bytes = None if "bytes" in missing else figures["bytes"] / n
    row.bytes_what = bytes_what


def _device(timings: Iterable[Mapping[str, Any]]) -> str:
    """The prover's device as a5's summary ``timing`` states it for every member (``gpus``, ``device``: a5.proving_of):
    the CPU, the cards named, else "not stated"."""

    said = [((t or {}).get("gpus"), (t or {}).get("device")) for t in timings]
    if said and all(g == 0 for g, _ in said):
        return "CPU (timing gpus 0)"
    if said and all(isinstance(g, int) and g >= 1 for g, _ in said):
        return "GPU " + ", ".join(sorted({str(d) for _, d in said})) + " (timing device)"
    return NOT_STATED


def _wrong_units(w: Any) -> str | None:
    """The audit's ``wrong_units`` bound as the row's ε at δ, where it states its bound, population and δ."""

    w = w if isinstance(w, dict) else {}
    if not all(_number(w.get(x)) is not None for x in ("bound", "of", "delta")):
        return None
    return f"at most {w['bound']:,} of {w['of']:,} units wrong except with probability {_delta(w['delta'])}"


def _epsilon(e: Any) -> str | None:
    """An a5 summary's ``epsilon`` table as the row's ε at δ: its first pair (the audit's δ) and its last (the smallest
    δ), each the wrong units it allows of the population."""

    e = e if isinstance(e, dict) else {}
    pairs = [x for x in e.get("table") or [] if isinstance(x, dict) and _number(x.get("wrong_units")) is not None
             and _number(x.get("delta")) is not None]
    if not pairs or _number(e.get("population")) is None:
        return None
    first, last = pairs[0], pairs[-1]
    out = f"at most {first['wrong_units']:,} of {e['population']:,} units wrong except with probability {_delta(first['delta'])}"
    if len(pairs) == 1:
        return out + " (summary.json epsilon)"
    return (out + f"; at most {last['wrong_units']:,} except with probability {_delta(last['delta'])}"
            f" (summary.json epsilon, the first and last of its {len(pairs)} pairs)")


def _eps_delta(delta: Any) -> str:
    return f"ε {NOT_STATED} at δ {delta}" if _number(delta) is not None else NOT_STATED


def _drawn(counts: Mapping[str, Any], population: Any, ranks: int = 1) -> str:
    drawn = ", ".join(f"{t} x {n}" for t, n in sorted(counts.items())) or "no block"
    return f"{drawn} drawn of {population if population is not None else '?'} units" + (f" over {ranks} ranks" if ranks > 1 else "")


def _lean_verdicts(a5_lines: Sequence[Mapping[str, Any]]) -> list[tuple[str, Any]]:
    return [(k, v) for x in a5_lines for k, v in ((x.get("audit") or {}).get("verdicts") or {}).items()
            if k.startswith("lean-verify")]


def read_recursion(d: Path, row: Row) -> None:
    """recursion_e2e.json: the headline's verdict (an accept only where Lean composed V*'s parts into the recursive
    statement and accepted it: ``ok`` and ``composed``), one recursive run's path without Lean's step as the prover's
    seconds, the lean step's wall as Lean's, and V*'s releases' bytes."""

    doc = json.loads((d / "recursion_e2e.json").read_text())
    h, p, inputs = doc.get("headline") or {}, doc.get("params") or {}, doc.get("inputs") or {}
    st = doc.get("staged")
    shape = f"{st['subcircuit']} x {st['n']}" if st else f"K = {p.get('K', '?')}, N = {p.get('N', '?')}"
    inner = [c.strip() for c in CLAUSE.split(str(inputs.get("inner") or "")) if c.strip()]
    row.workload = shape + (f": {inner[0]}" if inner else "")
    row.accepted = h.get("ok") is True and h.get("composed") is True
    if row.accepted:
        row.lean = "accepted: Lean composed V*'s parts into the recursive statement and accepted it"
    elif h.get("refused"):
        row.lean = f"refused: Lean refused the composition ({h['refused']})"
    elif not h.get("complete"):
        row.lean = "not a whole recursive run: " + ", ".join(sorted(doc.get("steps") or {})) + " ran"
    elif h.get("parts_ok", h.get("ok")) and h.get("composed") is not True:
        how = "not composed" if h.get("composed") is False else "not recorded as composed"
        row.lean = f"its parts passed and its recursive statement is {how}, so no accepted recursive run" + (
            f": {h['not_composed']}" if h.get("not_composed") else "")
    else:
        row.lean = "not accepted: a part or a negative did not hold"
    row.per = "a run"
    if p.get("GPU") == "0" and p.get("OPROVE_GPU", "0") == "0":
        row.device = "CPU (params GPU=0, OPROVE_GPU=0)"
    pipeline = h.get("pipeline") or {}
    prover = {k: v for k, v in pipeline.items() if k != "Lean step_s" and _number(v) is not None}
    if prover:
        row.prover_s = sum(float(v) for v in prover.values())
        row.prover_unit = "s a run"
        row.prover_what = "one recursive run's path without Lean's step: " + " + ".join(prover)
    row.lean_s = _number(pipeline.get("Lean step_s"))
    row.lean_what = "the lean step whole: every V* release and the inner statement"
    if isinstance(p.get("commit"), str) and re.fullmatch(r"[0-9a-f]{40}", p["commit"]):
        row.commit, row.commit_from = p["commit"], "params.commit"
    found = re.search(r"\(([^)]*)\)", str(h.get("outer_mode") or ""))
    outer = (h.get("outer") or {}).get(found.group(1) if found else "") or {}
    row.proof_bytes = _number(outer.get("proof_bytes"))
    row.bytes_what = f"V*'s {outer.get('statements', '?')} releases summed, mode {found.group(1) if found else '?'}"
    coins = next((c for c in inner if "coin" in c), None)
    said = ([f"inputs.inner: {coins}"] if coins else []) + ([f"params.tree_from {p['tree_from']}"] if p.get("tree_from") else [])
    row.said(None, None, "; ".join(said))


def read_prove_path(d: Path, row: Row) -> None:
    """An arm's audit.log: its honest PROVE-PATH line (``forged`` false), or under ``PP_RECORDS=1`` its RECORDS line (an
    arm not named ``*-forged``), and each rank's A5 line's Lean verify verdicts; the arm's rank or window summary.json for
    its timing (one rep's proof bytes, where prove_path's ``per_unit`` sums the reps); ``ran-with.txt`` beside the
    output for the ran-with line and whether the values were served; and a ``*-forged`` arm with no audit.log as no
    forged run."""

    arms = [d] if (d / "audit.log").is_file() else sorted(p.parent for p in d.glob("*/audit.log"))
    honest: list[tuple[Path, dict[str, Any] | None, dict[str, Any] | None]] = []
    for arm in arms:
        text = _text(arm / "audit.log")
        if lines := _tagged(text, "PROVE-PATH", row.notes):
            if lines[-1].get("forged") is False:
                honest.append((arm, lines[-1], None))
        elif not arm.name.endswith("-forged") and (records := _tagged(text, "RECORDS", row.notes)):
            honest.append((arm, None, records[-1]))
    if not honest:
        raise Unreadable(f"{d} holds no arm's audit.log with an honest PROVE-PATH line or a RECORDS line (prove_path.py "
                         "audit's)")
    if len(honest) > 1:
        raise Unreadable(f"{d} holds {len(honest)} honest arms ({', '.join(a.name for a, _, _ in honest)}): give one arm's "
                         "directory")
    arm, pp, rec = honest[0]
    audits = [x for x in _tagged(_text(arm / "audit.log"), "A5", row.notes) if "audit" in x]
    per_rank = [_lean_verdicts([x]) for x in audits]
    lean = [v for vs in per_rank for v in vs]
    summary = json.loads(_text(arm / "summary.json") or "{}")
    files = sorted(arm.glob("rank*/summary.json")) or [f for f in [arm / "window" / "summary.json"] if f.is_file()]
    summaries = [json.loads(f.read_text()) for f in files]
    counts: dict[str, int] = {}
    if pp is not None:
        ranks = pp.get("ranks") or {}
        each = len(audits) == len(ranks) and all(per_rank)
        if not each:
            row.notes.append(f"{len(audits)} A5 result line(s) for {len(ranks)} rank(s), "
                             f"{sum(1 for vs in per_rank if not vs)} without a Lean verify verdict: not one verdict a rank")
        row.accepted = (pp.get("ok") is True and bool(ranks) and all(r.get("accepted") is True for r in ranks.values())
                        and each and all(v is True for _, v in lean))
        if row.accepted:
            row.lean = f"accepted: Lean's {', '.join(sorted({k for k, _ in lean}))} true on every rank ({len(ranks)})"
        else:
            held = [f"PROVE-PATH ok {pp.get('ok')}"] + [f"rank {r} accepted {x.get('accepted')}"
                                                       for r, x in sorted(ranks.items()) if x.get("accepted") is not True]
            held += [f"{k} {v}" for k, v in lean if v is not True] or ([] if each else ["not every rank has a Lean verify verdict"])
            row.lean = "not accepted: " + "; ".join(held)
        for r in ranks.values():
            for t, n in (r.get("draw_counts") or {}).items():
                counts[t] = counts.get(t, 0) + int(n)
        population, n_ranks = sum(int(r.get("population") or 0) for r in ranks.values()) or None, len(ranks)
        row.proof = pp.get("proof")
    else:
        rec = rec or {}
        each = len(audits) == 1 and bool(lean)
        held = [f"RECORDS {k} {rec.get(k)}" for k in ("ok", "accepted", "negatives_refused") if rec.get(k) is not True]
        held += [f"{k} {v}" for k, v in lean if v is not True]
        held += [] if each else [f"{len(audits)} A5 result lines with {len(lean)} Lean verify verdicts"]
        held += [] if not audits or (audits[-1].get("audit") or {}).get("accepted") is True else ["a5's checks not accepted"]
        row.accepted = not held
        row.lean = (f"accepted: Lean's {', '.join(sorted({k for k, _ in lean}))} true, with RECORDS ok, accepted and its "
                    "negatives refused" if row.accepted else "not accepted: " + "; ".join(held))
        counts = {t: int(n) for t, n in (rec.get("draw_counts") or {}).items()}
        population, n_ranks = rec.get("population"), 1
        row.proof = "plain" if summary.get("plain") else "zk" if summary else None
    row.commit, row.commit_from = _commit(arm / "machine.txt", arm.parent / "machine.txt"), "machine.txt's git HEAD"
    ran_with = _text(arm.parent / "ran-with.txt").splitlines()
    served = next((x for x in ran_with if x.startswith("served")), None)
    source = {x.get("source") for x in summaries} - {None}
    place = served.split("not main: ", 1)[-1] if served else ""
    row.workload = ((f"the output disagrees on its values: ran-with.txt says served ({place}), the audit's source field "
                     f"says {', '.join(sorted(source))}: " if served and source - {"served"} else
                     f"served per ran-with.txt ({place}): " if served else "")
                    + f"arm {arm.name}: " + _drawn(counts, population, n_ranks))
    for forged in sorted(arm.parent.glob(f"{arm.name}-forged")):
        if not (forged / "audit.log").is_file():
            err = next((x for x in reversed(_text(forged / "commit.log").splitlines()) if re.match(r"\w+(Error|Exception): ", x)),
                       "it wrote no audit.log")
            row.workload += "; no forged run"
            row.notes.append(f"no forged run: {forged.name}/commit.log: {err.split('; ')[0][:240]}")
    timed = [x for x in summaries if x.get("timing")]
    if timed and len(timed) == len(summaries):
        reps = sorted({len(pb) for x in timed for t in x["timing"].values()
                       if isinstance(pb := ((t or {}).get("timing") or {}).get("proof_bytes"), list)})
        _per_block(row, [per_unit(x["timing"], x.get("draw_counts") or {}) for x in timed],
                   f"one rep's proof over its drawn blocks, of {' or '.join(map(str, reps)) or '?'} reps in summary.json's "
                   "timing; prove_path's per_unit sums them")
    elif pp is not None:
        _per_block(row, [r.get("per_unit") or {} for r in (pp.get("ranks") or {}).values()],
                   "every rep's proof files over its drawn blocks, as prove_path's per_unit sums them: no summary.json "
                   "states its reps")
    m = re.fullmatch(r"ran with: coins (.+?); parameters (\S+) \((.*)\)", ran_with[0] if ran_with else "")
    if m and m.group(1) in COINS and m.group(2) in PARAMETERS:
        row.said(m.group(1), m.group(2), "ran-with.txt: " + m.group(3))
    else:
        challenger = (summary.get("coins") or {}).get("challenger")
        coins = (pp or {}).get("coins")
        said = ([f"PROVE-PATH coins {coins}"] if coins else []) + (
            [f"summary.json coins.challenger {challenger}"] if challenger else [])
        row.said(_challenger(coins, challenger), None, "; ".join(said))
    row.device = _device(t for x in summaries for t in (x.get("timing") or {}).values())
    row.eps_delta = (_wrong_units((audits[0].get("audit") or {}).get("wrong_units")) if len(audits) == 1 else None) or (
        _eps_delta((pp or {}).get("delta")))


def _commit(*paths: Path, after: str = "") -> str | None:
    """The commit the first of `paths` that exists states: its first whole git sha that follows `after` (prove_path's
    run.sh's ``git rev-parse HEAD`` in machine.txt; layers.sh's job.log, after ``source ``, so no sha printed before it
    stands in), never a part of a longer digest."""

    for path in paths:
        if path.is_file():
            found = re.search(re.escape(after) + r"(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])", _text(path))
            return found.group(1) if found else None
    return None


def read_a5(d: Path, row: Row) -> None:
    """a5_llama1b_h200.sh's out (or a5.sh's): the audit's last A5 line (its Lean verify verdicts), the audit's summary.json
    for its timing, its coins and its arm, summary.txt's STOPPED, REGISTERED and rc lines, machine.txt's row dir and
    verdicts.txt's serving stages."""

    summary_txt = _text(d / "summary.txt")
    log = next((_text(d / f) for f in ("a5/a5.log", "a5.log") if (d / f).is_file()), summary_txt)
    lines = _tagged(log, "A5", row.notes)
    a5 = next((x for x in reversed(lines) if "audit" in x), lines[-1] if lines else None)
    sj = next((d / f for f in ("a5/a5/summary.json", "a5/summary.json") if (d / f).is_file()), None)
    summary = json.loads(sj.read_text()) if sj else {}
    text = summary_txt.splitlines()
    row.notes += [x for x in text if x.startswith(("replay rc=", "REGISTERED "))]
    row.notes += [f"serving stage {x}" for x in _text(d / "verdicts.txt").splitlines() if x.strip()]
    stopped = next((x for x in text if x.startswith("STOPPED at ")), None)
    if a5 is None:
        row.lean = "not reported: no A5 line in a5.log or summary.txt"
        row.why = stopped or row.lean
    else:
        lean = _lean_verdicts([a5])
        row.accepted = (a5.get("audit") or {}).get("accepted") is True and bool(lean) and all(v is True for _, v in lean)
        row.lean = (f"accepted: Lean's {', '.join(sorted({k for k, _ in lean}))} true, with a5's checks" if row.accepted else
                    "not accepted: " + ("; ".join(f"{k} {v}" for k, v in lean if v is not True) or
                                        ("a5's checks accepted " + str((a5.get("audit") or {}).get("accepted")) if "audit" in a5
                                         else json.dumps(a5)[:300])))
    machine = next((x for x in _text(d / "machine.txt").splitlines() if x.startswith("row dir ")), None)
    registered = next((x for x in text if x.startswith("REGISTERED ")), "")
    window = " ".join(w for w in registered.split() if w.startswith(("kind=", "commitment=")))
    counts = summary.get("draw_counts") or (a5 or {}).get("draw_counts") or {}
    served = summary.get("source") == "served" and "kind=served" in window
    stand_in = summary.get("source") == "stand-in" and "kind=served" not in window
    row.workload = (("served values: " if served else "stand-in values (summary.json source stand-in): " if stand_in else
                     f"{VALUES_NOT_STATED}: ")
                    + (f"row {Path(machine[len('row dir '):]).name}: " if machine else
                       f"source {summary['source']}: " if summary.get("source") else "")
                    + _drawn(counts, summary.get("population")) + (f" ({window})" if window else ""))
    if summary.get("timing"):
        _per_block(row, [per_unit(summary["timing"], counts)])
    verdicts = ((a5 or {}).get("audit") or {}).get("verdicts") or {}
    challengers = sorted(k for k in verdicts if k.startswith("challenger/"))
    coins = summary.get("coins") or {}
    said = (f"summary.json coins.challenger {coins['challenger']}" if coins.get("challenger") else
            f"A5 line's verdicts {', '.join(challengers)}, a challenger per drawn member" if challengers else "")
    drew = next((x[len("COINS "):] for x in text if x.startswith("COINS ")), None)
    if drew and drew.startswith("draw: one local challenger session"):
        clauses = drew.split("; ")
        row.said("local challenger", None,
                 "summary.txt COINS " + "; ".join(clauses[:1] + [c for c in clauses[1:2] if c.startswith("sessions: ")]))
    else:
        row.said(_challenger(coins.get("source"), coins.get("challenger")), None, said)
    row.commit, row.commit_from = _commit(d / "machine.txt"), "machine.txt's git HEAD"
    row.proof = "plain" if summary.get("plain") else "zk" if summary else None
    row.device = _device((summary.get("timing") or {}).values())
    row.eps_delta = (_epsilon(summary.get("epsilon")) or _wrong_units((summary.get("audit") or {}).get("wrong_units"))
                     or _wrong_units(((a5 or {}).get("audit") or {}).get("wrong_units")) or _eps_delta(summary.get("delta")))
    if summary.get("arm"):
        row.notes.append(f"arm {summary['arm']}")


def _delta(x: float) -> str:
    e = math.log2(x) if x > 0 else 0.0
    return f"2^{e:.0f}" if e == round(e) and e < -3 else re.sub(r"e-0*(\d)", r"e-\1", f"{x:g}")


def read_layers_sh(d: Path, row: Row) -> None:
    """A layers.sh layer's out (`benchmarks/one_stage/layers.sh`): layers.txt's ``ran with:`` line and whether its
    values are stand-ins (`STAND_IN`); a6/summary.json's statements, each with Flock's ``prove_total_s``, Lean's verify
    seconds and its proof's bytes, a statement's and not a proof block's; its verdicts (the plain run's every statement's,
    or the audit's, every Lean verdict and every negative case); the sample's ``wrong_units`` bound as its ε at δ; each
    session's lean-verify.log for the proof (``--zk``); and gpu-lease's usage record for the cards Flock proved on."""

    said = _text(d / "layers.txt").splitlines()
    m = re.fullmatch(r"ran with: coins (.+?); parameters (\S+) \((.*)\)", said[0] if said else "")
    if m and m.group(1) in COINS and m.group(2) in PARAMETERS:
        row.said(m.group(1), m.group(2), m.group(3))
    s = json.loads(_text(d / "a6" / "summary.json"))
    if s.get("windows"):
        stmts = [{"name": w.get("name"), "units": w.get("units"),
                  "template": next((x.get("template") for x in ((w.get("lean_verify") or {}).get("unit_draw") or {}).get("strata") or []), None),
                  "prove_s": _number((w.get("live") or {}).get("prove_total_s")), "lean_s": _number((w.get("lean_verify") or {}).get("verify_s")),
                  "proof_bytes": (w.get("live") or {}).get("proof_bytes"),
                  "accepted": (w.get("live") or {}).get("accepted") is True and (w.get("lean_verify") or {}).get("accepted") is True}
                 for w in s["windows"]]
        row.accepted = s.get("accepted") is True and all(x["accepted"] for x in stmts)
        n = sum(x["accepted"] for x in stmts)
        row.lean = (f"accepted: Lean's verify true on every statement ({n} of {len(stmts)})" if row.accepted else
                    f"not accepted: {n} of {len(stmts)} statements accepted, summary accepted {s.get('accepted')}")
        drawn = ""
    else:
        audit, negs = s.get("audit") or {}, s.get("negatives") or {}
        stmts = [{"name": x.get("name"), "template": x.get("template"), "units": (s.get("draw_counts") or {}).get(x.get("name")),
                  "prove_s": _number(((s.get("timing") or {}).get(x.get("name")) or {}).get("timing", {}).get("prove_total_s")),
                  "lean_s": _number(((s.get("timing") or {}).get(x.get("name")) or {}).get("lean_verify_s")),
                  "proof_bytes": (((s.get("timing") or {}).get(x.get("name")) or {}).get("timing") or {}).get("proof_bytes")}
                 for x in s.get("members") or []]
        verdicts, before = audit.get("verdicts") or {}, s.get("checks_before_draw") or {}
        row.accepted = (audit.get("accepted") is True and audit.get("complete") is True and bool(verdicts)
                        and all(v is True for v in (*verdicts.values(), *before.values(), *negs.values())) and bool(negs))
        held = [f"{k} {v}" for k, v in (*verdicts.items(), *before.items()) if v is not True]
        held += [f"negative {k} not refused" for k, v in negs.items() if v is not True]
        row.lean = (f"accepted: every Lean verdict true ({', '.join(sorted({k.split('/')[0] for k in verdicts}))} on each "
                    f"of {len(stmts)} statements), {len(negs)} of {len(negs)} negative cases refused" if row.accepted else
                    "not accepted: " + ("; ".join(held) or f"audit accepted {audit.get('accepted')}"))
        drawn = f", {s.get('k')} of them drawn" if s.get("k") != s.get("population") else ", every unit drawn"
        row.eps_delta = _wrong_units(audit.get("wrong_units")) or NOT_STATED
    population = s.get("population") or sum(int(x.get("units") or 0) for x in stmts)
    row.workload = ((f"{STAND_IN}: " if any(STAND_IN in x for x in said) else f"{VALUES_NOT_STATED}: ")
                    + f"{len(stmts)} statements of {population:,} units{drawn}: "
                    + "; ".join(f"{x['name']}, {x['units']:,} units" + (f" of {x['template']}" if x.get("template") else "")
                                for x in stmts))
    row.commit, row.commit_from = _commit(d / "job.log", after="source "), "job.log's source"
    for x in stmts:
        pb = x.pop("proof_bytes")
        x["proof_bytes"] = pb[0] if isinstance(pb, list) and pb and len(set(pb)) == 1 else pb if isinstance(pb, int) else None
        if isinstance(pb, list) and len(set(pb)) > 1:
            row.notes.append(f"{x['name']}'s reps' proofs differ in bytes ({pb}): no bytes figure")
        x.pop("accepted", None)
    row.statements = stmts
    zk = [" --zk" in _text(d / "a6" / f"session-{x['name']}" / "lean-verify.log") for x in stmts]
    row.proof = "zk" if zk and all(zk) else "plain" if s.get("plain") is True and not any(zk) else None
    leases = [json.loads(x) for x in _text(d.parent / "gpu-lease-usage.jsonl").splitlines() if x.strip()]
    cards = sorted({g.get("index") for x in leases if "flock-circuit prove" in str(x.get("cmd")) for g in x.get("gpus") or []},
                   key=str)
    if cards:
        row.device = "GPU " + ", ".join(map(str, cards)) + " (gpu-lease's record of each flock-circuit prove)"


READERS: dict[str, Callable[[Path, Row], None]] = {RECURSION: read_recursion, PROVE_PATH: read_prove_path, A5: read_a5,
                                                   LAYERS_SH: read_layers_sh}


def read(layer: str, given: str | None, fetcher: Callable[[str], Path] = fetch) -> Row:
    """`layer`'s row from the output `given` names (None: not run)."""

    row = Row(layer)
    if given is None:
        row.why = f"not run: no output given (--{layer} DIR|RUN)"
        return row
    row.given, row.status = given, "read"
    try:
        d = locate(given, fetcher)
        if detect(d) is None and layer in LAYERS_SH_DIRS and detect(own := d / LAYERS_SH_DIRS[layer] / "out"):
            d = own
        row.format = detect(d)
        if row.format is None:
            raise Unreadable(f"{d} holds none of the outputs this reads (recursion_e2e.json, an arm's audit.log, "
                             "summary.txt or a5.log, layers.sh's layers.txt)")
        if row.format not in ACCEPTS[layer]:
            raise Unreadable(f"--{layer} takes {' or '.join(ACCEPTS[layer])}'s output, and {d} holds {row.format}'s")
        READERS[row.format](d, row)
    except (Unreadable, OSError, ValueError, KeyError, TypeError) as e:
        row.status, row.lean, row.workload = "unreadable", "not read", "not read"
        row.withhold(f"unreadable: {e}")
        return row
    if layer in PROOF and row.proof != PROOF[layer]:
        row.withhold(f"--{layer} is the {PROOF[layer]} proof's row, and this output "
                     + (f"says its proof is {row.proof}" if row.proof else "states no proof"))
    if not row.accepted:
        row.withhold(f"the Lean verdict is not an accept ({row.lean})")
    return row


# ---- printing and writing ------------------------------------------------------------------------------------------------


def _cell(x: float | None, unit: str, per: str, what: str) -> str:
    """A figure with its unit as its first words (seconds a drawn block, or a whole run's wall seconds), then how it was
    read."""

    if x is None:
        return "—"
    figure = f"{x:,.0f} bytes" if unit == "bytes" else f"{x:.4g} s"
    wall = ", wall" if unit == "s" and per == "a run" else ""
    return f"{figure} {per}{wall}" + (f" ({what})" if what else "")


def _statements(r: Row, key: str, what: str) -> str:
    """A layers.sh row's cell: each statement's figure, its unit first and a statement's, never a proof block's."""

    cells = [(f"{x[key]:,.0f} bytes" if key == "proof_bytes" else f"{x[key]:.3g} s") + f" a statement of {x['units']:,} units"
             for x in r.statements if x.get(key) is not None]
    return "; ".join(cells) + f" ({what})" if cells else "—"


def has_number(rows: Sequence[Row]) -> bool:
    return any(x is not None for r in rows for x in (r.prover_s, r.lean_s, r.proof_bytes)) or any(
        x.get(k) is not None for r in rows for x in r.statements for k in ("prove_s", "lean_s", "proof_bytes"))


def render(rows: Sequence[Row]) -> str:
    """The table as text: its title; `NOT_AN_AUDIT` unless every row ran on a deployed challenger's coins at deployed
    parameters (and the JSON's ``audit`` is false regardless); each row's ``ran with:`` line; the table; then why each
    row without a number has none. With no number in any row its first line says so (`run` exits 1)."""

    lines = [] if has_number(rows) else ["prove layers: no number: " + "; ".join(f"{r.layer}: {r.why}" for r in rows)]
    lines += ["The proof system by layer, each row read from its own run's output (`verity bench prove layers`)."]
    if not all(r.ran_with.startswith("ran with: coins deployed challenger; parameters deployed") for r in rows):
        lines.append(NOT_AN_AUDIT)
    lines += [f"- {LAYERS[r.layer]}: {r.ran_with}" for r in rows]
    lines.append("- commits: " + "; ".join(f"{LAYERS[r.layer]} " + (f"{r.commit[:11]} ({r.commit_from})" if r.commit else
                                                                     NOT_STATED if r.status == "read" else "—")
                                           for r in rows)
                 + "; a row reruns from main only where its commit is on main (`git merge-base --is-ancestor COMMIT origin/main`)")
    lines += ["", "| layer | workload | proof | device | ε at δ | prover | Lean verifier | Lean | proof bytes |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        read = r.status == "read"
        proof = r.proof or (NOT_STATED if read else "—")
        cells = ((_statements(r, "prove_s", "Flock's prove_total_s"), _statements(r, "lean_s", "Lean's verify"),
                  _statements(r, "proof_bytes", "one rep's proof")) if r.statements else
                 (_cell(r.prover_s, 's', r.per, r.prover_what), _cell(r.lean_s, 's', r.per, r.lean_what),
                  _cell(r.proof_bytes, 'bytes', r.per, r.bytes_what)))
        lines.append(f"| {LAYERS[r.layer]} | {r.workload} | {proof} | {r.device if read else '—'} | "
                     f"{r.eps_delta if read else '—'} | {cells[0]} | {r.lean} | {cells[1]} | {cells[2]} |")
    lines += ["", *(f"- {LAYERS[r.layer]}: no number: {r.why}" for r in rows if r.why)]
    lines += [f"- {LAYERS[r.layer]}: {n}" for r in rows for n in r.notes]
    return "\n".join(lines).rstrip()


def document(rows: Sequence[Row]) -> dict[str, Any]:
    """The table as JSON. ``audit`` is always false: the command checks no challenger's signature."""

    return {"schema": SCHEMA, "audit": False, "rows": [{"title": LAYERS[r.layer], **asdict(r)} for r in rows]}


def measured(rows: Sequence[Row], metrics: Mapping[str, str]) -> list[dict[str, Any]]:
    """The writer's measurements: under each of `metrics`' ids (each naming one of `FIGURES`), one a layer of `MEMBERS`,
    in its order. A layer's value is its figure a proof block, only where its Lean verdict is an accept; else the value is
    None, with why. Recursion's row is no member: its seconds are a whole recursive run's."""

    if bad := sorted(set(metrics.values()) - set(FIGURES)):
        raise ValueError(f"metric ids name {bad}, not one of {sorted(FIGURES)}")
    by_layer = {r.layer: r for r in rows}
    out = []
    for name, figure in metrics.items():
        for layer, member in MEMBERS.items():
            r = by_layer[layer]
            value = getattr(r, figure) if r.accepted and r.per_block else None
            why = None if value is not None else r.why or (
                f"{r.workload.split(':')[0]}: its figures are a statement's, not a proof block's" if r.statements else
                f"its output gives no {figure} a proof block" if r.per_block else "its output gives no per-block figure")
            out.append({"name": name, "value": value, "unit": FIGURES[figure], "member": member, "why": why,
                        "inputs": [r.given] if r.given else [],
                        "extra": {"layer": LAYERS[layer], "status": r.status, "format": r.format, "workload": r.workload,
                                  "lean": r.lean, "ran_with": r.ran_with, "detail": r.detail, "proof": r.proof}})
    return out


def add_arguments(parser: argparse.ArgumentParser) -> None:
    for layer, title in LAYERS.items():
        parser.add_argument(f"--{layer}", metavar="DIR|RUN",
                            help=f"the {title} row's output: a run's out/ directory, or a run id fetched through the store")


def run(args: argparse.Namespace) -> int:
    rows = [read(layer, getattr(args, layer)) for layer in LAYERS]
    kit.report(REPORT, render(rows), document(rows), measured(rows, getattr(args, "metrics", None) or {}))
    return 0 if has_number(rows) else 1
