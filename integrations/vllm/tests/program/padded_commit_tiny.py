#!/usr/bin/env python3
"""pad -- PADDED-TABLE candidate #1 `pad-stage1-tiny`, END TO END on the CPU (R17-46 (5) / SATURATION-ORDER B.4; SCOPE-NOTE §7;
PADDED-TABLE criteria build / commit / counts / materialization / inactive_sampling / overhead / scope).

The tiny LServe fixture (H 32, NL 2, LP 4, D_MAX 8; the same TINY / prompt / weights as `test_serve3` / `test_lifted_tiny`) is built as
the FULL padded Program `LServe_v2` under LIFTING-SPEC v0.7 (explicit enablement B7, ordinary class G1 / A11, Continuation_v1 run/emit as
the one stopping mechanism, the declared LAG as a static), executed by the reference evaluator (there is no served run for the fixture --
the evaluator IS the execution; the Match column is n/a by construction), and then:

  1. the query of record `default(unlifted_width_max = 64)` (R17-43 addendum (1)) is evaluated on it -- the VU population Q(Lift(P));
  2. R(P, Q) = U_{V in Q(P)} in(V) u out(V) + the prescribed inputs + the designated outputs (SCOPE-NOTE §5), as ROOT GATE INDICES from the
     one boundary module (`boundary.in_gates` / `out_gates`, exact) -- distinct identities (a gate is one identity however many sets read it;
     shared references once; a prescribed weight never repeated);
  3. EVERY required word -- inactive included -- is materialized from the complete assignment in its TRUE lifted width (one machine word per
     lifted word: u32 for w+1 <= 32, u64 above; the tag is never split from the payload; an ORDINARY word (G1) is its unwidened width) and
     committed through the production host core: `pos_leaf` per leaf (ONE LEAF PER REQUIRED IDENTITY -- no range / compressed / implicit /
     derived entry; |leaves| == |R(P,Q)|) + `fold_levels` to ONE root, with ONE binding map (leaf -> identity) whose digest is on the record;
  4. the s6 five quantities as FIELDS: total_population / active_population / inactive_population (a VU is inactive iff every word of its
     out(V) is bot by enablement) / total_samples / samples_on_active / samples_on_inactive;
  5. sampled picks (a verifier challenge -> seed = sha256(challenge || root); the SAME draw count on the padded Program and on its
     active-reference twin) with ORIGINAL-ROOT OPENINGS of every out word of the picked VU verified against the one root (`fold_path`), and
     REPLAY BY RELATION: the VU's gates re-evaluated from its committed in(V) words by the registered lifted relations and compared bit for
     bit with the committed out(V) words -- for an inactive pick that is `enable = 0 -> bot`, checked, never assumed;
  6. the active-reference TWIN of the fixture -- `Serve_v1{TINY, LP, STEPS}` unlifted, committed the SAME way (query `default(32)`) -- as
     the denominator of the overhead (bytes / leaves / hash time / fold time / elapsed / peak RSS), at the served length (what a real
     active-reference row commits) and at the cap (the same population modulo the wrapper, A14.2);
  7. scope: Program digests, lifted families present in R(P,Q), the query id, the widened widths (A4.4 width summary at w_max_lifted).

Labels: `lifted-padded`, rules `padded-v1`.  Nothing here is an active-reference cell; nothing is copied from the first table.

    PYTHONPATH=<tree>:<tree>/vllm-poc python3 -m tests.program.padded_commit_tiny --case eos-at-event-1 --lag 0 --draws 64 --out DIR
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.circuits import Program, bind
from verity.ml.circuits import boundary as BD
from verity.ml.circuits import query_ast as GS
from verity.ml.circuits import partition as PT
from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.evaluate import evaluate_region
from verity_vllm.program.registry import b1, lifted as L
from verity_vllm.program.kernels import serve3_reference as ref
from tests.program.test_serve3 import D_MAX, LP, TINY, Y_RAW, _inputs

try:  # the production host core (the one revb accepted on F-r17b-44 / moe M-1023): pos_leaf + fold_levels + fold_path
    from verity_vllm.commit.merkle import fold_path
    from verity_vllm.commit.scheme import fold_levels, pos_leaf
    CORE = "verity_vllm.commit.scheme (production host core: pos_leaf + fold_levels; verity_vllm.commit.merkle.fold_path for openings)"
except Exception as ex:  # noqa: BLE001  -- named, never silent: without the core the record is not the production hashing
    CORE = f"UNAVAILABLE: {type(ex).__name__}: {ex}"
    pos_leaf = fold_levels = fold_path = None  # type: ignore[assignment]

LABEL = "lifted-padded"
SCHEMA = "padded-commit-tiny/1"
CASES = {"cap-run": 16777215, "eos-at-event-1": Y_RAW[1]}  # EOS never emitted (runs to the cap) / EOS = the second decode's raw token
Q_LIFTED, Q_TWIN = f"default({L.W_MAX_LIFTED})", f"default({L.W_MAX})"


def _rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0


def _tree() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _word_bytes(word: int, width: int) -> bytes:
    """One machine word per (lifted or ordinary) word in the smallest power-of-two container that holds its width: u16 for width <= 16
    (a bf16 word, lifted or not -- what a real bf16 committer writes), u32 for <= 32 (a Value<17> lifted bf16, a 32-bit word), u64 for
    <= 64 (a Value<33>).  SCOPE-NOTE §4: the tag is never split from the payload.  [F-r17b-56 revb: writing every <= 32-bit word as
    4 B under-stated the representation cost of the padded row by 25-37 %: the twin's and the ordinary (G1) bf16 words are 2 B.]"""
    if width <= 8:
        return int(word).to_bytes(1, "little")
    if width <= 16:
        return int(word).to_bytes(2, "little")
    if width <= 32:
        return int(word).to_bytes(4, "little")
    if width <= 64:
        return int(word).to_bytes(8, "little")
    raise ValueError(f"width {width} > 64")


def _reading(prim: Any, word: int, width: int) -> str:
    if getattr(prim, "family", None) == "Input":
        return "input"
    if not L.is_lifted_prim(prim):
        return "ordinary"
    w = width - 1
    if word == L.enc_bot(w):
        return "bot"
    if word >> w:
        return "inv"
    return "active"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _owner_name(prog: Program, g: int) -> str:
    """The root-level call owning gate `g` (`slot=t/...` for a step's body; the guard/controller nodes by name), for the per-step table."""
    body = prog.root.body
    k = body.node_at(g)
    n = body.nodes[k]
    return n.name or n.fn.id.split("{", 1)[0]


# ---------------------------------------------------------------------------------------------------------
# the Program, its complete assignment, R(P, Q)
# ---------------------------------------------------------------------------------------------------------


def build(kind: str, case: str, lag: int, steps: int | None = None) -> dict[str, Any]:
    prompt, w = _inputs()
    t0 = time.perf_counter()
    if kind == "padded":
        sp = L.lserve(TINY, LP, D_MAX, (CASES[case],), lag=lag)
        prog = Program(sp)
        inputs = [list(prompt), list(w)]  # G1: prompt and weights are ORDINARY (Value<32> / Value<16>) -- unwidened
        q = Q_LIFTED
    else:
        assert steps is not None
        prog = Program(bind(b1.Serve, C=TINY, LP=LP, STEPS=steps))
        inputs = [list(prompt), list(w)]
        q = Q_TWIN
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    t = ref.materialize(prog, inputs)
    t_eval = time.perf_counter() - t0
    dg = program_digest(encode_program(prog))
    return {"kind": kind, "prog": prog, "t": t, "digest": dg, "query": q, "t_build": t_build, "t_eval": t_eval, "inputs": inputs}


def required(b: dict[str, Any]) -> dict[str, Any]:
    """R(P, Q) as root gate indices + the VU sets with their in/out gates (exact boundary module); Unsupported sets counted apart."""
    prog, q = b["prog"], b["query"]
    t0 = time.perf_counter()
    fam = GS.evaluate(prog, q)
    sets: list[dict[str, Any]] = []
    unsupported = multipart = 0
    req: set[int] = set()
    for gs in fam:
        parts = getattr(gs, "parts", None)
        if parts is None or len(parts) != 1:
            multipart += 1
            continue
        og, ig = BD.out_gates(parts[0]), BD.in_gates(parts[0])
        if not isinstance(og, frozenset) or not isinstance(ig, frozenset):
            unsupported += 1
            continue
        sets.append({"ivs": gs.intervals(), "in": sorted(ig), "out": sorted(og)})
        req.update(og)
        req.update(ig)
    circ = prog.circuit
    prescribed = {gv.index for gv in circ[0:prog.gates] if getattr(gv.prim, "family", None) == "Input"}  # every prescribed-input gate
    designated = set(circ.root_outputs())
    req_all = req | prescribed | designated
    t_req = time.perf_counter() - t0
    verdict = PT.validate_vu_partition(prog, q, L.W_MAX_LIFTED if b["kind"] == "padded" else L.W_MAX)
    return {"fam": fam, "sets": sets, "unsupported": unsupported, "multipart": multipart, "boundary_gates": len(req),
            "prescribed": sorted(prescribed), "designated": sorted(designated), "required": sorted(req_all), "t_required": t_req,
            "verdict": {"status": verdict.status, "count": int(verdict.count)}}


# ---------------------------------------------------------------------------------------------------------
# the explicit one-stage Commit: every required word, one leaf each, one root, one binding map
# ---------------------------------------------------------------------------------------------------------


def commit(b: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    prog, t = b["prog"], b["t"]
    ids = r["required"]
    t0 = time.perf_counter()
    entries: list[dict[str, Any]] = []
    blobs: list[bytes] = []
    prims = {}
    for leaf, g in enumerate(ids):
        p = BD.gate_prim(prog, g)
        prims[g] = p
        width = p.ret.w
        word = t[g]
        blob = _word_bytes(word, width)
        blobs.append(blob)
        entries.append({"leaf": leaf, "gate": g, "spec": p.id, "width": width, "nbytes": len(blob), "class": _reading(p, word, width),
                        "owner": _owner_name(prog, g)})
    t_write = time.perf_counter() - t0
    t0 = time.perf_counter()
    leaves = [pos_leaf(bl) for bl in blobs]
    t_hash = time.perf_counter() - t0
    t0 = time.perf_counter()
    levels = fold_levels(list(leaves))
    t_fold = time.perf_counter() - t0
    root = levels[-1][0]
    bmap = {"schema": "padded-binding-map/1", "label": LABEL, "rules_version": L.PADDED_RULES_VERSION if b["kind"] == "padded" else "r14-v1",
            "program_digest": b["digest"], "query": b["query"], "root": root.hex(), "n_leaves": len(leaves),
            "leaf_rule": "ONE leaf per required identity (root gate): pos_leaf(word bytes); u32 for width <= 32, u64 above; tag never split",
            "core": CORE, "entries": entries}
    bmap["map_digest"] = _sha(entries)
    by_class: dict[str, int] = {}
    for e in entries:
        by_class[e["class"]] = by_class.get(e["class"], 0) + 1
    return {"root": root, "levels": levels, "leaves": leaves, "blobs": blobs, "bmap": bmap, "prims": prims,
            "bytes": sum(len(x) for x in blobs), "n_leaves": len(leaves), "by_class": by_class,
            "t_write": t_write, "t_hash": t_hash, "t_fold": t_fold}


def opening(levels: list[list[bytes]], index: int) -> list[str | None]:
    """The sibling path of leaf `index` (None = the odd trailing node lifted), verifiable by `verity_vllm.commit.merkle.fold_path`."""
    path: list[str | None] = []
    cursor = index
    for lvl in levels[:-1]:
        if cursor == len(lvl) - 1 and len(lvl) % 2 == 1:
            path.append(None)
        else:
            sib = cursor ^ 1
            path.append(lvl[sib].hex())
        cursor //= 2
    return path


# ---------------------------------------------------------------------------------------------------------
# counts, sampling, openings, replay by relation
# ---------------------------------------------------------------------------------------------------------


def classify_sets(b: dict[str, Any], r: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    t = b["t"]
    leaf_of = {e["gate"]: e["leaf"] for e in c["bmap"]["entries"]}
    for s in r["sets"]:
        kinds = {c["bmap"]["entries"][leaf_of[g]]["class"] for g in s["out"]}
        if not kinds:
            s["kind"] = "empty"  # a VU read by nobody (a dead gate of the body, present in the twin too): no boundary identity to bind
        elif kinds <= {"bot"}:
            s["kind"] = "inactive"
        elif "bot" in kinds:
            s["kind"] = "mixed"
        elif "inv" in kinds:
            s["kind"] = "inv"
        else:
            s["kind"] = "active"
    n = {k: sum(1 for s in r["sets"] if s["kind"] == k) for k in ("active", "inactive", "mixed", "inv", "empty")}
    words = {k: 0 for k in ("active", "inactive")}
    for s in r["sets"]:
        if s["kind"] in words:
            words[s["kind"]] += len(s["out"])
    # [n-pad1-2 revb] the prescribed-input singletons (a VU that IS one prescribed Input gate: a weight / prompt word, ORDINARY under G1,
    # never inactive) stated apart, so the inactive share reads over the COMPUTED population
    prescribed = set(r["prescribed"])
    n_presc = sum(1 for s in r["sets"] if s["out"] and all(g in prescribed for g in s["out"]))
    computed = len(r["sets"]) - n_presc - n["empty"]
    return {"total_population": len(r["sets"]), "active_population": n["active"], "inactive_population": n["inactive"],
            "prescribed_input_sets": n_presc, "computed_population": computed,
            "inactive_share_of_computed": (n["inactive"] / computed if computed else None),
            "sets_with_empty_out": sum(1 for s in r["sets"] if not s["out"]),  # a VU read by nobody (a dead gate of the body): no identity to bind
            "mixed_population": n["mixed"], "inv_population": n["inv"], "empty_population": n["empty"], "sets_unsupported": r["unsupported"], "sets_multipart": r["multipart"],
            "required_identities": len(r["required"]), "boundary_identities": r["boundary_gates"], "prescribed_identities": len(r["prescribed"]),
            "designated_identities": len(r["designated"]), "words_by_class": c["by_class"], "out_words_active_sets": words["active"],
            "out_words_inactive_sets": words["inactive"], "_t": t}


def sample_and_check(b: dict[str, Any], r: dict[str, Any], c: dict[str, Any], draws: int, challenge: str) -> dict[str, Any]:
    prog, t = b["prog"], b["t"]
    circ = prog.circuit
    root, levels = c["root"], c["levels"]
    leaf_of = {e["gate"]: e["leaf"] for e in c["bmap"]["entries"]}
    seed = int.from_bytes(hashlib.sha256(challenge.encode() + root).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    n = len(r["sets"])
    picks = sorted(int(x) for x in rng.choice(n, size=min(draws, n), replace=False))
    prescribed = set(r["prescribed"])
    out_rows: list[dict[str, Any]] = []
    on = {"active": 0, "inactive": 0, "mixed": 0, "inv": 0, "empty": 0}
    openings_ok = replay_ok = 0
    n_open = 0
    t0 = time.perf_counter()
    for pi in picks:
        s = r["sets"][pi]
        on[s["kind"]] += 1
        # (a) original-root openings of every out word of the picked VU
        ops = []
        for g in s["out"]:
            li = leaf_of[g]
            path = opening(levels, li)
            ok = fold_path(c["leaves"][li], li, [bytes.fromhex(p) if p is not None else None for p in path]) == root
            n_open += 1
            openings_ok += int(ok)
            ops.append({"gate": g, "leaf": li, "word": c["blobs"][li].hex(), "path": path, "ok": ok})
        # (b) replay by relation: the VU's gates from its committed in(V) words (+ its own prescribed Input gates), compared to out(V)
        t2: dict[int, int] = {g: t[g] for g in s["in"]}
        for lo, hi in s["ivs"]:
            for g in range(lo, hi):
                if g in prescribed:
                    t2[g] = t[g]
        for lo, hi in s["ivs"]:
            evaluate_region(circ, circ[lo:hi], t2)
        equal = all(t2[g] == t[g] for g in s["out"])
        bot_check = None
        if s["kind"] == "inactive":  # enable = 0 -> bot, CHECKED on the replayed words
            bot_check = all(t2[g] == L.enc_bot(c["prims"][g].ret.w - 1) for g in s["out"])
            equal = equal and bot_check
        replay_ok += int(equal)
        out_rows.append({"set": pi, "kind": s["kind"], "ivs": s["ivs"], "n_in": len(s["in"]), "n_out": len(s["out"]),
                         "owner": c["bmap"]["entries"][leaf_of[s["out"][0]]]["owner"] if s["out"] else None,
                         "spec_out": sorted({c["prims"][g].id for g in s["out"]})[:4], "openings": ops, "replay_equal": equal, "bot_checked": bot_check})
    t_check = time.perf_counter() - t0
    inactive_opened = [row for row in out_rows if row["kind"] == "inactive"]
    return {"challenge": challenge, "seed_rule": "sha256(challenge || root)[:8] -> numpy default_rng; VU sets drawn uniformly without replacement",
            "total_samples": len(picks), "samples_on_active": on["active"], "samples_on_inactive": on["inactive"], "samples_on_mixed": on["mixed"],
            "samples_on_inv": on["inv"], "samples_on_empty": on["empty"], "openings": n_open, "openings_ok": openings_ok, "replays_ok": replay_ok, "replays": len(picks),
            "inactive_pick_openings": sum(len(x["openings"]) for x in inactive_opened), "t_check": t_check, "picks": out_rows}


# ---------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------


def run_one(kind: str, case: str, lag: int, draws: int, challenge: str, out: Path, steps: int | None = None) -> dict[str, Any]:
    t_all = time.perf_counter()
    b = build(kind, case, lag, steps)
    r = required(b)
    c = commit(b, r)
    counts = classify_sets(b, r, c)
    counts.pop("_t")
    smp = sample_and_check(b, r, c, draws, challenge)
    prog, t = b["prog"], b["t"]
    outs = [t[g] for g in prog.circuit.root_outputs()]
    # per root-level owner: how many required words, how many bot -- the inactive suffix, step by step
    per_owner: dict[str, dict[str, int]] = {}
    for e in c["bmap"]["entries"]:
        d = per_owner.setdefault(e["owner"], {"words": 0, "bot": 0, "active": 0, "ordinary": 0, "input": 0, "inv": 0})
        d["words"] += 1
        d[e["class"]] += 1
    elapsed = time.perf_counter() - t_all
    name = f"{kind}_{case}_lag{lag}" + (f"_steps{steps}" if steps is not None else "")
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "binding_map.json.gz", "wt") as f:
        json.dump(c["bmap"], f, separators=(",", ":"))
    (d / "openings.json").write_text(json.dumps({"root": c["root"].hex(), "n_leaves": c["n_leaves"], "picks": smp["picks"]}, indent=0))
    if kind == "padded":
        wa = L.lifted_width_admissibility(prog, r["fam"])
        ws = wa.get("width_summary", {})
        width = {k: ws.get(k) for k in ("w_max_lifted", "max_seen", "w_out_lifted_max", "n_out_max", "growth_max", "w_out_payload_max", "complete")}
        width["status_at_w_max_lifted"] = (wa.get("verdict") or {}).get("status")
        width["active_reference_rule_status"] = (wa.get("active_reference_rule_verdict") or {}).get("status")
        width["active_reference_rule_note"] = ("the 32-bit ACTIVE-REFERENCE VU rule (W_MAX 32) evaluated on the padded Program: it fails BY DESIGN on "
                                               "Value<33> lifted words (A11: a lifted row is judged at w_max_lifted, status_at_w_max_lifted) -- "
                                               "not a verdict of this row, kept so the two rules are never confused")
    else:
        width = {"w_max": L.W_MAX, "max_seen": getattr(PT.validate_vu_partition(prog, b["query"], L.W_MAX).width, "max_seen", None)}
    n_active_events = sum(1 for x in outs if not (kind == "padded" and x == L.enc_bot(32)))
    rec = {"schema": SCHEMA, "label": LABEL if kind == "padded" else "active-reference twin of the tiny fixture (committed the same way; a denominator, never a cell of the first table)",
           "kind": kind, "case": case, "lag": lag, "steps": steps, "program": {"id": prog.root.id[:160], "digest": b["digest"], "gates": prog.gates,
           "rules_version": L.PADDED_RULES_VERSION if kind == "padded" else None, "query": b["query"], "query_id": GS.query_id(GS.parse(b["query"])),
           "verdict": r["verdict"]}, "events": {"n": len(outs), "active": n_active_events, "outputs": [int(x) for x in outs]},
           "counts": counts, "inactive_sampling": {k: v for k, v in smp.items() if k != "picks"},
           "commit": {"root": c["root"].hex(), "map_digest": c["bmap"]["map_digest"], "n_leaves": c["n_leaves"], "bytes": c["bytes"],
                      "leaf_rule": c["bmap"]["leaf_rule"], "core": CORE, "leaves_equal_required": c["n_leaves"] == len(r["required"]),
                      "inactive_leaves_canonical": all(int.from_bytes(c["blobs"][e["leaf"]], "little") == L.enc_bot(e["width"] - 1)
                                                       for e in c["bmap"]["entries"] if e["class"] == "bot"),
                      "no_inv": c["by_class"].get("inv", 0) == 0},
           "timing_s": {"build": b["t_build"], "evaluate": b["t_eval"], "required": r["t_required"], "write": c["t_write"], "hash": c["t_hash"],
                        "fold": c["t_fold"], "check": smp["t_check"], "elapsed": elapsed}, "rss_peak_mb": _rss_mb(),
           "per_owner": per_owner, "lifted_families": sorted({e["spec"].split("{", 1)[0] for e in c["bmap"]["entries"] if e["class"] in ("bot", "active", "inv")})[:400],
           "width": width, "files": {"binding_map": str(d / "binding_map.json.gz"), "openings": str(d / "openings.json")}}
    (d / "padded_commit.json").write_text(json.dumps(rec, indent=1))
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", choices=sorted(CASES), default="eos-at-event-1")
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--draws", type=int, default=64, help="C2 draw count -- the SAME on the padded Program and on its twin (R17-45 (3))")
    ap.add_argument("--challenge", default="pad-stage1-tiny/verifier-challenge/1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if pos_leaf is None:
        print(f"host core unavailable: {CORE}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    padded = run_one("padded", a.case, a.lag, a.draws, a.challenge, out)
    # the EXECUTED prefix (R17-1: the post-EOS lag step executes; served length != executed-prefix length): decode steps whose body words are
    # active -- what the active-reference row of record commits (its `build_request_LP<LP>_T<T>` specialization)
    served_steps = max(sum(1 for k, v in padded["per_owner"].items() if k.startswith("s=") and k.endswith("/step") and v["active"] > 0), 1)
    twin_served = run_one("twin", a.case, a.lag, a.draws, a.challenge, out, steps=served_steps)
    twin_cap = twin_served if served_steps == D_MAX else run_one("twin", a.case, a.lag, a.draws, a.challenge, out, steps=D_MAX)

    def ratio(x, y):
        return (x / y) if y else None

    def overhead(tw):
        return {"twin": {"steps": tw["steps"], "digest": tw["program"]["digest"], "root": tw["commit"]["root"], "n_leaves": tw["commit"]["n_leaves"],
                         "bytes": tw["commit"]["bytes"], "hash_s": tw["timing_s"]["hash"], "fold_s": tw["timing_s"]["fold"],
                         "elapsed_s": tw["timing_s"]["elapsed"], "rss_peak_mb": tw["rss_peak_mb"], "population": tw["counts"]["total_population"]},
                "extra_bytes": padded["commit"]["bytes"] - tw["commit"]["bytes"], "bytes_ratio": ratio(padded["commit"]["bytes"], tw["commit"]["bytes"]),
                "extra_leaves": padded["commit"]["n_leaves"] - tw["commit"]["n_leaves"], "leaves_ratio": ratio(padded["commit"]["n_leaves"], tw["commit"]["n_leaves"]),
                "hash_ratio": ratio(padded["timing_s"]["hash"], tw["timing_s"]["hash"]), "fold_ratio": ratio(padded["timing_s"]["fold"], tw["timing_s"]["fold"]),
                "elapsed_ratio": ratio(padded["timing_s"]["elapsed"], tw["timing_s"]["elapsed"]),
                "population_ratio": ratio(padded["counts"]["total_population"], tw["counts"]["total_population"])}

    row = {"schema": SCHEMA, "label": LABEL, "rules_version": L.PADDED_RULES_VERSION, "candidate": "pad-stage1-tiny", "case": a.case, "lag": a.lag,
           "tree": _tree(), "host": platform.node(), "python": sys.version.split()[0], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "core": CORE, "fixture": {"C": TINY, "LP": LP, "D_MAX": D_MAX, "eos": [CASES[a.case]], "events": D_MAX + 1},
           "padded": {k: padded[k] for k in ("program", "events", "counts", "inactive_sampling", "commit", "timing_s", "rss_peak_mb", "per_owner", "lifted_families", "width", "files")},
           "overhead_vs_twin_executed_prefix": overhead(twin_served), "overhead_vs_twin_cap": overhead(twin_cap),
           "inactivity_tests": ["early termination (Continuation_v1 run/emit)" if padded["counts"]["inactive_population"] else "none present (cap run: no inactive VU)",
                                (f"declared async lag (LAG {a.lag} static): no post-EOS step executes -- the EOS step is the last active step, inactivity from the next"
                                 if a.lag == 0 else
                                 f"declared async lag (LAG {a.lag} static): the post-EOS step(s) = the lag step(s) are ACTIVE (sampled token not served), inactivity after them")],
           "criteria": {"build": "full padded Program LServe_v2 under LIFTING-SPEC v0.7 (B7 explicit enablement, G1 ordinary class, Continuation_v1), fixed before execution; digest on the record",
                        "match": "n/a -- no served run for the tiny fixture; the reference evaluator is the execution",
                        "commit": f"every required word incl. inactive committed: {padded['commit']['n_leaves']} leaves = |R(P,Q)| {padded['counts']['required_identities']} -> one root {padded['commit']['root'][:16]}..., one binding map {padded['commit']['map_digest'][:16]}...",
                        "counts": "the s6 five quantities as fields (counts / inactive_sampling)",
                        "materialization": f"leaves_equal_required={padded['commit']['leaves_equal_required']}, inactive_leaves_canonical={padded['commit']['inactive_leaves_canonical']}, no_inv={padded['commit']['no_inv']}; one leaf per identity, no range/compressed/implicit entry",
                        "inactive_sampling": f"{padded['inactive_sampling']['samples_on_inactive']} of {padded['inactive_sampling']['total_samples']} picks on inactive VUs; {padded['inactive_sampling']['inactive_pick_openings']} inactive-pick original-root openings verified; replay by relation {padded['inactive_sampling']['replays_ok']}/{padded['inactive_sampling']['replays']} equal",
                        "overhead": "measured on this row vs the fixture's unlifted twin committed the same way (served length and cap) -- a fixture denominator, stated as such",
                        "scope": "program digest, lifted families, query id, widths (width)"},
           "gaps": ["no served run exists for the tiny fixture: Match is n/a and the commit path is the host core (pos_leaf + fold_levels), not a GPU committer run",
                    "the twin here is the fixture's own unlifted Program committed the same way -- not a row of the first table"]}
    (out / f"row_{a.case}_lag{a.lag}.json").write_text(json.dumps(row, indent=1))
    p = padded
    print(f"[pad-stage1-tiny] {a.case} LAG {a.lag}: population {p['counts']['total_population']} (active {p['counts']['active_population']} / inactive "
          f"{p['counts']['inactive_population']} / mixed {p['counts']['mixed_population']}); required {p['counts']['required_identities']} = leaves "
          f"{p['commit']['n_leaves']} ({p['commit']['bytes']} B); bot words {p['commit']['n_leaves'] and p['counts']['words_by_class'].get('bot', 0)}; "
          f"samples {p['inactive_sampling']['total_samples']} (active {p['inactive_sampling']['samples_on_active']} / inactive {p['inactive_sampling']['samples_on_inactive']}); "
          f"openings {p['inactive_sampling']['openings_ok']}/{p['inactive_sampling']['openings']} replays {p['inactive_sampling']['replays_ok']}/{p['inactive_sampling']['replays']}; "
          f"root {p['commit']['root'][:16]}; vs twin(executed prefix T={twin_served['steps']}) bytes x{row['overhead_vs_twin_executed_prefix']['bytes_ratio']:.3f} "
          f"leaves x{row['overhead_vs_twin_executed_prefix']['leaves_ratio']:.3f}; elapsed {p['timing_s']['elapsed']:.1f}s rss {p['rss_peak_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
