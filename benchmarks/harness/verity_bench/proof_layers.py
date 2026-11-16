"""`verity bench prove layers`: the proof benchmark by layer, one table of its four benchmark results (Flock, Flock
with zero knowledge, with sampling, with recursion), each read from what its own command wrote
(benchmarks/prove/README.md, "The proof benchmark by layer").

    verity bench prove layers [--flock DIR|RUN] [--zk DIR|RUN] [--sampling DIR|RUN] [--recursion DIR|RUN]

DIR is a run's ``out/`` (or the run directory that holds it), read with no store; RUN is a run id, whose records are
fetched through the evidence store where `research` is installed. A layer not given is a benchmark result that says
"not run". Which reader a directory takes is decided by what it holds (`detect`):

- ``recursion_e2e.json``: `benchmarks/recursion_e2e`'s job (`read_recursion`);
- an arm's ``audit.log`` with its ``PROVE-PATH`` line, or its ``RECORDS`` line under ``PP_RECORDS=1``:
  `benchmarks/prove_path/run.sh` (`read_prove_path`);
- ``summary.txt``, ``a5.log`` or the audit's ``summary.json``: `benchmarks/one_stage/a5_llama1b_h200.sh`, or `a5.sh`'s
  out (`read_a5`).

A benchmark result prints its numbers only when the Lean verifier's verdict, as its output states it, is an accept;
any other prints none, with why. Its ``ran with: coins COINS; parameters PARAMETERS (DETAIL)`` line names only what its output
says outright (`COINS`, `PARAMETERS`, "not stated" for a half it doesn't say, ``ran with: not stated`` for both), and
the JSON keeps the output's own words as ``detail``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from verity_bench import kit
from verity_bench.proof_layers_find import (
    detect,
    fetch,
    locate,
)
from verity_bench.proof_layers_parse import (
    _challenger,
    _commit,
    _delta,
    _device,
    _eps_delta,
    _lean_verdicts,
    _number,
    _sampled,
    _tagged,
    _text,
)
from verity_bench.proof_layers_print import (
    document,
    has_number,
    measured,
    render,
)
from verity_bench.proof_layers_result import (
    A5,
    ACCEPTS,
    CLAUSE,
    COINS,
    FIGURES,
    FIX,
    LAYERS,
    LAYERS_SH,
    LAYERS_SH_DIRS,
    MEMBERS,
    NOT_AN_AUDIT,
    NOT_STATED,
    PARAMETERS,
    PROOF,
    PROVE_PATH,
    RECURSION,
    REPORT,
    SCHEMA,
    STAND_IN,
    VALUES_NOT_STATED,
    Result,
    Unreadable,
)

__all__ = [
    "A5",
    "FIGURES",
    "FIX",
    "LAYERS",
    "LAYERS_SH",
    "MEMBERS",
    "NOT_AN_AUDIT",
    "NOT_STATED",
    "PROVE_PATH",
    "READERS",
    "RECURSION",
    "REPORT",
    "SCHEMA",
    "Result",
    "Unreadable",
    "add_arguments",
    "detect",
    "document",
    "fetch",
    "has_number",
    "locate",
    "measured",
    "per_unit",
    "read",
    "read_a5",
    "read_layers_sh",
    "read_prove_path",
    "read_recursion",
    "render",
    "run",
]


# ---- reading -------------------------------------------------------------------------------------------------------------


def per_unit(timing: Mapping[str, Any], counts: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """An a5 summary's ``timing`` per sampled proof block of each type, in the form prove_path's ``per_unit`` prints:
    the proof blocks sampled (``draw_counts``), Flock's ``prove_total_s`` and Lean's verify seconds over them, and one
    rep's proof's bytes where every rep's are the same (a timed repeat of the one proof), else none."""

    out = {}
    for key, t in timing.items():
        tm, n = (t or {}).get("timing") or {}, counts.get(key) or 0
        pb = tm.get("proof_bytes")
        one = pb[0] if isinstance(pb, list) and pb and len(set(pb)) == 1 else pb if isinstance(pb, int) else None
        out[key] = {"units": n, "proof_bytes": one,
                    **{f"{x}_per_unit": v / n if n and (v := _number(raw)) is not None else None
                       for x, raw in (("prove_s", tm.get("prove_total_s")), ("lean_verify_s", (t or {}).get("lean_verify_s")))}}
    return out


def _per_block(result: Result, units: Iterable[Mapping[str, Mapping[str, Any]]],
               bytes_what: str = "one rep's proof over its sampled proof blocks") -> None:
    """The benchmark result's prover and Lean seconds and proof bytes a proof block, over every sampled proof block of `units` (prove_path's
    ``per_unit`` maps, one a rank): the seconds' means weighted by each type's blocks, and the bytes' sum over the blocks."""

    n, figures = 0, {"prove": 0.0, "lean": 0.0, "bytes": 0.0}
    missing: set[str] = set()
    for per in units:
        for key, x in per.items():
            k = int(x.get("units") or 0)
            if not k:
                result.notes.append(f"no figure a proof block: member {key}'s timing has no count of sampled proof blocks (draw_counts)")
            n += k
            for name, v in (("prove", _number(x.get("prove_s_per_unit"))), ("lean", _number(x.get("lean_verify_s_per_unit")))):
                missing |= {name} if v is None else set()
                figures[name] += (v or 0.0) * k
            b = _number(x.get("proof_bytes"))
            missing |= {"bytes"} if b is None else set()
            figures["bytes"] += b or 0.0
    if not n:
        result.notes.append("no sampled proof block holds a timing")
        return
    result.notes += [f"no {name} figure a proof block: a sampled member's timing doesn't give one" for name in sorted(missing)]
    result.prover_unit, result.per, result.per_block = "s a proof block", "a sampled proof block", True
    result.prover_s = None if "prove" in missing else figures["prove"] / n
    result.prover_what = f"Flock's prove_total_s, the mean over {n}"
    result.lean_s = None if "lean" in missing else figures["lean"] / n
    result.lean_what = f"Lean's verify, the mean over {n}"
    result.proof_bytes = None if "bytes" in missing else figures["bytes"] / n
    result.bytes_what = bytes_what


def _incorrect(w: Any) -> str | None:
    """The audit's ``wrong_units`` bound as the benchmark result's ε at δ, where it states its bound, population and δ."""

    w = w if isinstance(w, dict) else {}
    if not all(_number(w.get(x)) is not None for x in ("bound", "of", "delta")):
        return None
    return f"at most {w['bound']:,} of {w['of']:,} incorrect proof blocks except with probability {_delta(w['delta'])}"


def _epsilon(e: Any) -> str | None:
    """An a5 summary's ``epsilon`` table as the benchmark result's ε at δ: its first pair (the audit's δ) and its last (the smallest
    δ), each the incorrect proof blocks it allows of the population."""

    e = e if isinstance(e, dict) else {}
    pairs = [x for x in e.get("table") or [] if isinstance(x, dict) and _number(x.get("wrong_units")) is not None
             and _number(x.get("delta")) is not None]
    if not pairs or _number(e.get("population")) is None:
        return None
    first, last = pairs[0], pairs[-1]
    out = f"at most {first['wrong_units']:,} of {e['population']:,} incorrect proof blocks except with probability {_delta(first['delta'])}"
    if len(pairs) == 1:
        return out + " (summary.json epsilon)"
    return (out + f"; at most {last['wrong_units']:,} except with probability {_delta(last['delta'])}"
            f" (summary.json epsilon, the first and last of its {len(pairs)} pairs)")


def read_recursion(d: Path, result: Result) -> None:
    """recursion_e2e.json: the headline's verdict (an accept only where Lean composed V*'s parts into the recursive
    statement and accepted it: ``ok`` and ``composed``), one recursive run's path without Lean's step as the prover's
    seconds, the lean step's wall as Lean's, and V*'s releases' bytes."""

    doc = json.loads((d / "recursion_e2e.json").read_text())
    h, p, inputs = doc.get("headline") or {}, doc.get("params") or {}, doc.get("inputs") or {}
    st = doc.get("staged")
    shape = f"{st['subcircuit']} x {st['n']}" if st else f"K = {p.get('K', '?')}, N = {p.get('N', '?')}"
    inner = [c.strip() for c in CLAUSE.split(str(inputs.get("inner") or "")) if c.strip()]
    result.workload = shape + (f": {inner[0]}" if inner else "")
    result.accepted = h.get("ok") is True and h.get("composed") is True
    if result.accepted:
        result.lean = "accepted: Lean composed V*'s parts into the recursive statement and accepted it"
    elif h.get("refused"):
        result.lean = f"refused: Lean refused the composition ({h['refused']})"
    elif not h.get("complete"):
        result.lean = "not a whole recursive run: " + ", ".join(sorted(doc.get("steps") or {})) + " ran"
    elif h.get("parts_ok", h.get("ok")) and h.get("composed") is not True:
        how = "not composed" if h.get("composed") is False else "not recorded as composed"
        result.lean = f"its parts passed and its recursive statement is {how}, so no accepted recursive run" + (
            f": {h['not_composed']}" if h.get("not_composed") else "")
    else:
        result.lean = "not accepted: a part or a negative did not hold"
    result.per = "a run"
    if p.get("GPU") == "0" and p.get("OPROVE_GPU", "0") == "0":
        result.device = "CPU (params GPU=0, OPROVE_GPU=0)"
    pipeline = h.get("pipeline") or {}
    prover = {k: v for k, v in pipeline.items() if k != "Lean step_s" and _number(v) is not None}
    if prover:
        result.prover_s = sum(float(v) for v in prover.values())
        result.prover_unit = "s a run"
        result.prover_what = "one recursive run's path without Lean's step: " + " + ".join(prover)
    result.lean_s = _number(pipeline.get("Lean step_s"))
    result.lean_what = "the lean step whole: every V* release and the inner statement"
    if isinstance(p.get("commit"), str) and re.fullmatch(r"[0-9a-f]{40}", p["commit"]):
        result.commit, result.commit_from = p["commit"], "params.commit"
    found = re.search(r"\(([^)]*)\)", str(h.get("outer_mode") or ""))
    outer = (h.get("outer") or {}).get(found.group(1) if found else "") or {}
    result.proof_bytes = _number(outer.get("proof_bytes"))
    result.bytes_what = f"V*'s {outer.get('statements', '?')} releases summed, mode {found.group(1) if found else '?'}"
    coins = next((c for c in inner if "coin" in c), None)
    said = ([f"inputs.inner: {coins}"] if coins else []) + ([f"params.tree_from {p['tree_from']}"] if p.get("tree_from") else [])
    result.said(None, None, "; ".join(said))


def read_prove_path(d: Path, result: Result) -> None:
    """An arm's audit.log: its honest PROVE-PATH line (``forged`` false), or under ``PP_RECORDS=1`` its RECORDS line (an
    arm not named ``*-forged``), and each rank's A5 line's Lean verify verdicts; the arm's rank or window summary.json for
    its timing (one rep's proof bytes, where prove_path's ``per_unit`` sums the reps); ``ran-with.txt`` beside the
    output for the ran-with line and whether the values were served; and a ``*-forged`` arm with no audit.log as no
    forged run."""

    arms = [d] if (d / "audit.log").is_file() else sorted(p.parent for p in d.glob("*/audit.log"))
    honest: list[tuple[Path, dict[str, Any] | None, dict[str, Any] | None]] = []
    for arm in arms:
        text = _text(arm / "audit.log")
        if lines := _tagged(text, "PROVE-PATH", result.notes):
            if lines[-1].get("forged") is False:
                honest.append((arm, lines[-1], None))
        elif not arm.name.endswith("-forged") and (records := _tagged(text, "RECORDS", result.notes)):
            honest.append((arm, None, records[-1]))
    if not honest:
        raise Unreadable(f"{d} holds no arm's audit.log with an honest PROVE-PATH line or a RECORDS line (prove_path.py "
                         "audit's)")
    if len(honest) > 1:
        raise Unreadable(f"{d} holds {len(honest)} honest arms ({', '.join(a.name for a, _, _ in honest)}): give one arm's "
                         "directory")
    arm, pp, rec = honest[0]
    audits = [x for x in _tagged(_text(arm / "audit.log"), "A5", result.notes) if "audit" in x]
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
            result.notes.append(f"{len(audits)} A5 result line(s) for {len(ranks)} rank(s), "
                             f"{sum(1 for vs in per_rank if not vs)} without a Lean verify verdict: not one verdict a rank")
        result.accepted = (pp.get("ok") is True and bool(ranks) and all(r.get("accepted") is True for r in ranks.values())
                        and each and all(v is True for _, v in lean))
        if result.accepted:
            result.lean = f"accepted: Lean's {', '.join(sorted({k for k, _ in lean}))} true on every rank ({len(ranks)})"
        else:
            held = [f"PROVE-PATH ok {pp.get('ok')}"] + [f"rank {r} accepted {x.get('accepted')}"
                                                       for r, x in sorted(ranks.items()) if x.get("accepted") is not True]
            held += [f"{k} {v}" for k, v in lean if v is not True] or ([] if each else ["not every rank has a Lean verify verdict"])
            result.lean = "not accepted: " + "; ".join(held)
        for r in ranks.values():
            for t, n in (r.get("draw_counts") or {}).items():
                counts[t] = counts.get(t, 0) + int(n)
        population, n_ranks = sum(int(r.get("population") or 0) for r in ranks.values()) or None, len(ranks)
        result.proof = pp.get("proof")
    else:
        rec = rec or {}
        each = len(audits) == 1 and bool(lean)
        held = [f"RECORDS {k} {rec.get(k)}" for k in ("ok", "accepted", "negatives_refused") if rec.get(k) is not True]
        held += [f"{k} {v}" for k, v in lean if v is not True]
        held += [] if each else [f"{len(audits)} A5 result lines with {len(lean)} Lean verify verdicts"]
        held += [] if not audits or (audits[-1].get("audit") or {}).get("accepted") is True else ["a5's checks not accepted"]
        result.accepted = not held
        result.lean = (f"accepted: Lean's {', '.join(sorted({k for k, _ in lean}))} true, with RECORDS ok, accepted and its "
                    "negatives refused" if result.accepted else "not accepted: " + "; ".join(held))
        counts = {t: int(n) for t, n in (rec.get("draw_counts") or {}).items()}
        population, n_ranks = rec.get("population"), 1
        result.proof = "plain" if summary.get("plain") else "zk" if summary else None
    result.commit, result.commit_from = _commit(arm / "machine.txt", arm.parent / "machine.txt"), "machine.txt's git HEAD"
    ran_with = _text(arm.parent / "ran-with.txt").splitlines()
    served = next((x for x in ran_with if x.startswith("served")), None)
    source = {x.get("source") for x in summaries} - {None}
    place = served.split("not main: ", 1)[-1] if served else ""
    result.workload = ((f"the output disagrees on its values: ran-with.txt says served ({place}), the audit's source field "
                     f"says {', '.join(sorted(source))}: " if served and source - {"served"} else
                     f"served per ran-with.txt ({place}): " if served else "")
                    + f"arm {arm.name}: " + _sampled(counts, population, n_ranks))
    for forged in sorted(arm.parent.glob(f"{arm.name}-forged")):
        if not (forged / "audit.log").is_file():
            err = next((x for x in reversed(_text(forged / "commit.log").splitlines()) if re.match(r"\w+(Error|Exception): ", x)),
                       "it wrote no audit.log")
            result.workload += "; no forged run"
            result.notes.append(f"no forged run: {forged.name}/commit.log: {err.split('; ')[0][:240]}")
    timed = [x for x in summaries if x.get("timing")]
    if timed and len(timed) == len(summaries):
        reps = sorted({len(pb) for x in timed for t in x["timing"].values()
                       if isinstance(pb := ((t or {}).get("timing") or {}).get("proof_bytes"), list)})
        _per_block(result, [per_unit(x["timing"], x.get("draw_counts") or {}) for x in timed],
                   f"one rep's proof over its sampled proof blocks, of {' or '.join(map(str, reps)) or '?'} reps in summary.json's "
                   "timing; prove_path's per_unit sums them")
    elif pp is not None:
        _per_block(result, [r.get("per_unit") or {} for r in (pp.get("ranks") or {}).values()],
                   "every rep's proof files over its sampled proof blocks, as prove_path's per_unit sums them: no summary.json "
                   "states its reps")
    m = re.fullmatch(r"ran with: coins (.+?); parameters (\S+) \((.*)\)", ran_with[0] if ran_with else "")
    if m and m.group(1) in COINS and m.group(2) in PARAMETERS:
        result.said(m.group(1), m.group(2), "ran-with.txt: " + m.group(3))
    else:
        challenger = (summary.get("coins") or {}).get("challenger")
        coins = (pp or {}).get("coins")
        said = ([f"PROVE-PATH coins {coins}"] if coins else []) + (
            [f"summary.json coins.challenger {challenger}"] if challenger else [])
        result.said(_challenger(coins, challenger), None, "; ".join(said))
    result.device = _device(t for x in summaries for t in (x.get("timing") or {}).values())
    result.eps_delta = (_incorrect((audits[0].get("audit") or {}).get("wrong_units")) if len(audits) == 1 else None) or (
        _eps_delta((pp or {}).get("delta")))


def read_a5(d: Path, result: Result) -> None:
    """a5_llama1b_h200.sh's out (or a5.sh's): the audit's last A5 line (its Lean verify verdicts), the audit's summary.json
    for its timing, its coins and its arm, summary.txt's STOPPED, REGISTERED and rc lines, machine.txt's row dir and
    verdicts.txt's serving stages."""

    summary_txt = _text(d / "summary.txt")
    log = next((_text(d / f) for f in ("a5/a5.log", "a5.log") if (d / f).is_file()), summary_txt)
    lines = _tagged(log, "A5", result.notes)
    a5 = next((x for x in reversed(lines) if "audit" in x), lines[-1] if lines else None)
    sj = next((d / f for f in ("a5/a5/summary.json", "a5/summary.json") if (d / f).is_file()), None)
    summary = json.loads(sj.read_text()) if sj else {}
    text = summary_txt.splitlines()
    result.notes += [x for x in text if x.startswith(("replay rc=", "REGISTERED "))]
    result.notes += [f"serving stage {x}" for x in _text(d / "verdicts.txt").splitlines() if x.strip()]
    stopped = next((x for x in text if x.startswith("STOPPED at ")), None)
    if a5 is None:
        result.lean = "not reported: no A5 line in a5.log or summary.txt"
        result.why = stopped or result.lean
    else:
        lean = _lean_verdicts([a5])
        result.accepted = (a5.get("audit") or {}).get("accepted") is True and bool(lean) and all(v is True for _, v in lean)
        result.lean = (f"accepted: Lean's {', '.join(sorted({k for k, _ in lean}))} true, with a5's checks" if result.accepted else
                    "not accepted: " + ("; ".join(f"{k} {v}" for k, v in lean if v is not True) or
                                        ("a5's checks accepted " + str((a5.get("audit") or {}).get("accepted")) if "audit" in a5
                                         else json.dumps(a5)[:300])))
    machine = next((x for x in _text(d / "machine.txt").splitlines() if x.startswith("row dir ")), None)
    registered = next((x for x in text if x.startswith("REGISTERED ")), "")
    window = " ".join(w for w in registered.split() if w.startswith(("kind=", "commitment=")))
    counts = summary.get("draw_counts") or (a5 or {}).get("draw_counts") or {}
    served = summary.get("source") == "served" and "kind=served" in window
    stand_in = summary.get("source") == "stand-in" and "kind=served" not in window
    result.workload = (("served values: " if served else "stand-in values (summary.json source stand-in): " if stand_in else
                     f"{VALUES_NOT_STATED}: ")
                    + (f"workload {Path(machine[len('row dir '):]).name}: " if machine else
                       f"source {summary['source']}: " if summary.get("source") else "")
                    + _sampled(counts, summary.get("population")) + (f" ({window})" if window else ""))
    if summary.get("timing"):
        _per_block(result, [per_unit(summary["timing"], counts)])
    verdicts = ((a5 or {}).get("audit") or {}).get("verdicts") or {}
    challengers = sorted(k for k in verdicts if k.startswith("challenger/"))
    coins = summary.get("coins") or {}
    said = (f"summary.json coins.challenger {coins['challenger']}" if coins.get("challenger") else
            f"A5 line's verdicts {', '.join(challengers)}, a challenger per sampled member" if challengers else "")
    line = next((x[len("COINS "):] for x in text if x.startswith("COINS ")), None)
    if line and line.startswith("draw: one local challenger session"):
        clauses = line.split("; ")
        result.said("local challenger", None,
                 "summary.txt COINS " + "; ".join(clauses[:1] + [c for c in clauses[1:2] if c.startswith("sessions: ")]))
    else:
        result.said(_challenger(coins.get("source"), coins.get("challenger")), None, said)
    result.commit, result.commit_from = _commit(d / "machine.txt"), "machine.txt's git HEAD"
    result.proof = "plain" if summary.get("plain") else "zk" if summary else None
    result.device = _device((summary.get("timing") or {}).values())
    result.eps_delta = (_epsilon(summary.get("epsilon")) or _incorrect((summary.get("audit") or {}).get("wrong_units"))
                     or _incorrect(((a5 or {}).get("audit") or {}).get("wrong_units")) or _eps_delta(summary.get("delta")))
    if summary.get("arm"):
        result.notes.append(f"arm {summary['arm']}")


def read_layers_sh(d: Path, result: Result) -> None:
    """A layers.sh layer's out (`benchmarks/one_stage/layers.sh`): layers.txt's ``ran with:`` line and whether its
    values are stand-ins (`STAND_IN`); a6/summary.json's statements, each with Flock's ``prove_total_s``, Lean's verify
    seconds and its proof's bytes, a statement's and not a proof block's; its verdicts (the plain run's every statement's,
    or the audit's, every Lean verdict and every negative case); the sample's ``wrong_units`` bound as its ε at δ; each
    session's lean-verify.log for the proof (``--zk``); and gpu-lease's usage record for the cards Flock proved on."""

    said = _text(d / "layers.txt").splitlines()
    m = re.fullmatch(r"ran with: coins (.+?); parameters (\S+) \((.*)\)", said[0] if said else "")
    if m and m.group(1) in COINS and m.group(2) in PARAMETERS:
        result.said(m.group(1), m.group(2), m.group(3))
    s = json.loads(_text(d / "a6" / "summary.json"))
    if s.get("windows"):
        stmts = [{"name": w.get("name"), "units": w.get("units"),
                  "template": next((x.get("template") for x in ((w.get("lean_verify") or {}).get("unit_draw") or {}).get("strata") or []), None),
                  "prove_s": _number((w.get("live") or {}).get("prove_total_s")), "lean_s": _number((w.get("lean_verify") or {}).get("verify_s")),
                  "proof_bytes": (w.get("live") or {}).get("proof_bytes"),
                  "accepted": (w.get("live") or {}).get("accepted") is True and (w.get("lean_verify") or {}).get("accepted") is True}
                 for w in s["windows"]]
        result.accepted = s.get("accepted") is True and all(x["accepted"] for x in stmts)
        n = sum(x["accepted"] for x in stmts)
        result.lean = (f"accepted: Lean's verify true on every statement ({n} of {len(stmts)})" if result.accepted else
                    f"not accepted: {n} of {len(stmts)} statements accepted, summary accepted {s.get('accepted')}")
        sampled = ""
    else:
        audit, negs = s.get("audit") or {}, s.get("negatives") or {}
        stmts = [{"name": x.get("name"), "template": x.get("template"), "units": (s.get("draw_counts") or {}).get(x.get("name")),
                  "prove_s": _number(((s.get("timing") or {}).get(x.get("name")) or {}).get("timing", {}).get("prove_total_s")),
                  "lean_s": _number(((s.get("timing") or {}).get(x.get("name")) or {}).get("lean_verify_s")),
                  "proof_bytes": (((s.get("timing") or {}).get(x.get("name")) or {}).get("timing") or {}).get("proof_bytes")}
                 for x in s.get("members") or []]
        verdicts, before = audit.get("verdicts") or {}, s.get("checks_before_draw") or {}
        result.accepted = (audit.get("accepted") is True and audit.get("complete") is True and bool(verdicts)
                        and all(v is True for v in (*verdicts.values(), *before.values(), *negs.values())) and bool(negs))
        held = [f"{k} {v}" for k, v in (*verdicts.items(), *before.items()) if v is not True]
        held += [f"negative {k} not refused" for k, v in negs.items() if v is not True]
        result.lean = (f"accepted: every Lean verdict true ({', '.join(sorted({k.split('/')[0] for k in verdicts}))} on each "
                    f"of {len(stmts)} statements), {len(negs)} of {len(negs)} negative cases refused" if result.accepted else
                    "not accepted: " + ("; ".join(held) or f"audit accepted {audit.get('accepted')}"))
        sampled = f", {s.get('k')} of them sampled" if s.get("k") != s.get("population") else ", every proof block sampled"
        result.eps_delta = _incorrect(audit.get("wrong_units")) or NOT_STATED
    population = s.get("population") or sum(int(x.get("units") or 0) for x in stmts)
    result.workload = ((f"{STAND_IN}: " if any(STAND_IN in x for x in said) else f"{VALUES_NOT_STATED}: ")
                    + f"{len(stmts)} statements of {population:,} proof blocks{sampled}: "
                    + "; ".join(f"{x['name']}, {x['units']:,} units" + (f" of {x['template']}" if x.get("template") else "")
                                for x in stmts))
    result.commit, result.commit_from = _commit(d / "job.log", after="source "), "job.log's source"
    for x in stmts:
        pb = x.pop("proof_bytes")
        x["proof_bytes"] = pb[0] if isinstance(pb, list) and pb and len(set(pb)) == 1 else pb if isinstance(pb, int) else None
        if isinstance(pb, list) and len(set(pb)) > 1:
            result.notes.append(f"{x['name']}'s reps' proofs differ in bytes ({pb}): no bytes figure")
        x.pop("accepted", None)
    result.statements = stmts
    zk = [" --zk" in _text(d / "a6" / f"session-{x['name']}" / "lean-verify.log") for x in stmts]
    result.proof = "zk" if zk and all(zk) else "plain" if s.get("plain") is True and not any(zk) else None
    leases = [json.loads(x) for x in _text(d.parent / "gpu-lease-usage.jsonl").splitlines() if x.strip()]
    cards = sorted({g.get("index") for x in leases if "flock-circuit prove" in str(x.get("cmd")) for g in x.get("gpus") or []},
                   key=str)
    if cards:
        result.device = "GPU " + ", ".join(map(str, cards)) + " (gpu-lease's record of each flock-circuit prove)"


READERS: dict[str, Callable[[Path, Result], None]] = {RECURSION: read_recursion, PROVE_PATH: read_prove_path, A5: read_a5,
                                                   LAYERS_SH: read_layers_sh}


def read(layer: str, given: str | None, fetcher: Callable[[str], Path] = fetch) -> Result:
    """The benchmark result for `layer` from the output `given` names (None: not run)."""

    result = Result(layer)
    if given is None:
        result.why = f"not run: no output given (--{layer} DIR|RUN)"
        return result
    result.given, result.status = given, "read"
    try:
        d = locate(given, fetcher)
        if detect(d) is None and layer in LAYERS_SH_DIRS and detect(own := d / LAYERS_SH_DIRS[layer] / "out"):
            d = own
        result.format = detect(d)
        if result.format is None:
            raise Unreadable(f"{d} holds none of the outputs this reads (recursion_e2e.json, an arm's audit.log, "
                             "summary.txt or a5.log, layers.sh's layers.txt)")
        if result.format not in ACCEPTS[layer]:
            raise Unreadable(f"--{layer} takes {' or '.join(ACCEPTS[layer])}'s output, and {d} holds {result.format}'s")
        READERS[result.format](d, result)
    except (Unreadable, OSError, ValueError, KeyError, TypeError) as e:
        result.status, result.lean, result.workload = "unreadable", "not read", "not read"
        result.withhold(f"unreadable: {e}")
        return result
    if layer in PROOF and result.proof != PROOF[layer]:
        result.withhold(f"--{layer} is the {PROOF[layer]} proof's benchmark result, and this output "
                     + (f"says its proof is {result.proof}" if result.proof else "states no proof"))
    if not result.accepted:
        result.withhold(f"the Lean verdict is not an accept ({result.lean})")
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    named = {"flock": "the Flock layer", "zk": "the zero-knowledge layer", "sampling": "the sampling layer",
             "recursion": "the recursion layer"}
    for layer in LAYERS:
        parser.add_argument(f"--{layer}", metavar="DIR|RUN",
                            help=f"the output the benchmark result for {named[layer]} is read from: a run's out/ directory, "
                                 "or a run id fetched through the store")


def run(args: argparse.Namespace) -> int:
    results = [read(layer, getattr(args, layer)) for layer in LAYERS]
    kit.report(REPORT, render(results), document(results), measured(results, getattr(args, "metrics", None) or {}))
    return 0 if has_number(results) else 1
