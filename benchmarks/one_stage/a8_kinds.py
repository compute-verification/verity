"""Run A8's census and lowering probe: a served Program's root Calls by Definition kind (family), and for each kind one unit
lowered through C-Flock's circuit prover (M0's `UnitStager`, the path `a8.py` stages on), its outcome one of

* ``lowers``: the unit's circuit composed; its ANDs, k_log, and how many instances a statement holds (`a6.chunk_cap`);
* ``no-piece``: M0's lowering has no piece for a primitive the unit reads (`ir_lower.NoPiece`, the piece named), with the
  largest sub-Definition of the unit that has every piece, lowered the same way;
* ``too-big``: an instance needs a block past `circuit.K_MAX` (or a statement past the spec's m <= 35).

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/a8_kinds.py --descriptor descriptor.json.gz --out kinds.json [--jobs 8] [--mem-gb 96]

The unit of a kind is the Call's coordinate: from the Call, the callee carrying at least half its gates while the
Definition has more than `--unit-gates` word gates. A unit M0 cannot compose for its shape (no input row past one SHA-512
block, or a row that is not whole u16 words) is replaced by the Definition above it, up to the Call. Each kind runs in a
forked child of the process that decoded the Program, so lowerings registered for one kind never meet another's.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.setrecursionlimit(100000)

from verity.ml.circuits.defs import PrimitiveDefinition  # noqa: E402
from verity.ml.circuits.program import input_id, is_input  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity_vllm.program.boolean import load_program  # noqa: E402

#: flock-circuit's spec shape: a statement's m = k_log + nbl, nbl >= 3, is at most 35 (`a6.M_MAX`)
M_MAX = 35


def family(fid: str) -> str:
    return fid.split("{")[0]


def is_boolean(fn) -> bool:
    return callable(getattr(getattr(fn, "defn", fn), "build", None))


def mult(nd) -> int:
    return nd.n if nd.form in ("batch", "scan") else 1


def callees(fn) -> tuple[Counter, dict]:
    """Each composite callee of fn's body: how many times fn calls it, by id."""
    m, defs = Counter(), {}
    for nd in fn.body.nodes:
        if isinstance(nd.fn, PrimitiveDefinition) or is_boolean(nd.fn):
            continue
        m[nd.fn.id] += mult(nd)
        defs[nd.fn.id] = nd.fn
    return m, defs


def chain(fn, unit_gates: int) -> list[tuple[object, int]]:
    """The Call and its dominant callees, each with how many times its parent calls it: down while a Definition has more
    than unit_gates gates and one callee carries at least half of them."""
    out = [(fn, 1)]
    while not isinstance(fn, PrimitiveDefinition) and not is_boolean(fn) and fn.gates > unit_gates:
        m, defs = callees(fn)
        if not m:
            break
        best = max(m, key=lambda k: m[k] * defs[k].gates)
        if m[best] * defs[best].gates < fn.gates * 0.5:
            break
        fn = defs[best]
        out.append((fn, m[best]))
    return out


def prims_of(fn, memo: dict) -> Counter:
    """The primitives fn reads, with multiplicity (a Boolean Definition counted as one leaf of its own)."""
    if fn.id in memo:
        return memo[fn.id]
    c = Counter()
    if isinstance(fn, PrimitiveDefinition):
        c[fn.id] += 1
    elif is_boolean(fn):
        c[f"<boolean {family(fn.id)}>"] += 1
    else:
        for nd in fn.body.nodes:
            for k, v in prims_of(nd.fn, memo).items():
                c[k] += v * mult(nd)
    memo[fn.id] = c
    return c


def missing_pieces(fn, memo: dict) -> list[str]:
    from verity_flock import ir_lower as IL
    out = []
    for p in prims_of(fn, memo):
        if p.startswith("<boolean") or p.startswith("Const") or input_id(p) is not None:
            continue
        try:
            if p not in IL.PIECES:
                out.append(p)
        except Exception:  # noqa: BLE001
            out.append(p)
    return sorted(out)


def leaf_bits(t) -> int:
    return sum(t.leaf_widths())


def rows_of(fn) -> list[tuple[str, str, int, list[str]]] | None:
    """The unit's parameters as V2 rows: all in one row when their bits are whole u16 words, else cut where the running
    sum is; None when no cut makes every row whole words."""
    params, _ret = fn.signature()
    bits = [leaf_bits(t) for _n, t in params]
    if sum(bits) % 16:
        return None
    rows, cur, cur_b = [], [], 0
    for (n, _t), b in zip(params, bits):
        cur.append(n)
        cur_b += b
        if cur_b % 16 == 0 and cur_b >= 8192:
            rows.append((f"r{len(rows)}", V2, cur_b, cur))
            cur, cur_b = [], 0
    if cur:
        if cur_b % 16:
            return None
        rows.append((f"r{len(rows)}", V2, cur_b, cur))
    return rows


def try_lower(fn) -> dict:
    """M0's `UnitStager` on fn (rows as `rows_of`, one V2 output row of every output): the composed circuit's size, or why
    it was refused."""
    from verity_flock import class_statement as CS
    from verity_flock import ir_lower as IL
    rows = rows_of(fn)
    if rows is None:
        return {"outcome": "shape", "why": "the parameters' bits are not whole u16 words"}
    _params, ret = fn.signature()
    n_out = len(ret.fields) if hasattr(ret, "fields") else 1
    t = time.time()
    try:
        st = CS.UnitStager(fn, f"a8-probe/{family(fn.id)}", rows, [(V2, tuple(range(n_out)))])
    except IL.NoPiece as e:
        return {"outcome": "no-piece", "piece": str(e)[:300], "seconds": round(time.time() - t, 1)}
    except ValueError as e:
        msg = str(e)
        kind = ("too-big" if "at most 2^" in msg else "shape" if ("compression before its last block" in msg or "whole u16" in msg)
                else "refused")
        return {"outcome": kind, "why": msg[:400], "seconds": round(time.time() - t, 1)}
    except MemoryError:
        return {"outcome": "memory", "why": "MemoryError under the child's address-space cap", "seconds": round(time.time() - t, 1)}
    c, meta = st.low.layout.counts, st.meta
    k_log = meta["k_log"]
    cap = meta["vus_per_block"] << (M_MAX - k_log) if M_MAX - k_log >= 3 else 0
    return {"outcome": "lowers" if cap else "too-big", "ands": c["ands"], "useful": st.low.layout.useful, "k_log": k_log,
            "vus_per_block": meta["vus_per_block"], "chunk_cap": cap, "rows": [[r, b, ps] for r, _s, b, ps in rows],
            "out_words": st.out_words, "seconds": round(time.time() - t, 1)}


def probe_kind(fam: str, fn, unit_gates: int) -> dict:
    memo: dict = {}
    ch = chain(fn, unit_gates)
    per = 1
    path = []
    for f, m in ch:
        per *= m
        path.append({"id": f.id, "gates": f.gates, "times": m, "per_call": per})
    attempts = []
    i = len(ch) - 1
    while i >= 0:
        u = ch[i][0]
        if isinstance(u, PrimitiveDefinition):
            i -= 1
            continue
        r = try_lower(u)
        attempts.append({"unit": u.id, "gates": u.gates, "per_call": path[i]["per_call"], **r})
        if r["outcome"] != "shape":
            break
        i -= 1
    last = attempts[-1] if attempts else {"outcome": "none", "why": "no composite Definition on the chain"}
    rec = {"kind": fam, "call": fn.id, "call_gates": fn.gates, "path": path, "attempts": attempts, "outcome": last["outcome"],
           "unit": last.get("unit"), "unit_gates": last.get("gates"), "units_per_call": last.get("per_call")}
    if last["outcome"] == "no-piece" and last.get("unit"):
        u = next(f for f, _m in ch if f.id == last["unit"])
        rec["missing"] = missing_pieces(u, memo)
        best, seen, frontier = None, set(), [(u, 1)]
        while frontier:
            f, m = frontier.pop()
            if f.id in seen:
                continue
            seen.add(f.id)
            cm, defs = callees(f)
            for k, n in cm.items():
                d = defs[k]
                if missing_pieces(d, memo):
                    frontier.append((d, m * n))
                elif d.gates >= 16 and (best is None or m * n * d.gates > best[1] * best[0].gates):
                    best = (d, m * n)
        if best is not None:
            sub = try_lower(best[0])
            rec["sub_unit"] = {"unit": best[0].id, "gates": best[0].gates, "per_unit": best[1], **sub}
    return rec


def census(prog) -> tuple[dict, int]:
    calls, gates, specs = Counter(), Counter(), {}
    for nd in prog.root.body.nodes:
        if is_input(nd.fn):
            continue
        fam = family(nd.fn.id)
        calls[fam] += mult(nd)
        gates[fam] += mult(nd) * nd.fn.gates
        specs.setdefault(fam, Counter())[nd.fn.id] += mult(nd)
    comp = sum(gates.values())
    return {f: {"calls": calls[f], "gates": gates[f], "share": gates[f] / comp if comp else 0.0, "specs": len(specs[f]),
                "spec": specs[f].most_common(1)[0][0]} for f in gates}, comp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--mem-gb", type=int, default=96, help="each child's address-space cap")
    ap.add_argument("--timeout-s", type=int, default=2400, help="each child's budget")
    ap.add_argument("--unit-gates", type=int, default=4096)
    ap.add_argument("--only", default=None, help="comma-separated kinds")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prog, digest = load_program(a.descriptor)
    kinds, comp = census(prog)
    print(f"A8 census: {len(kinds)} kinds, {comp} computing gates of {prog.gates}, decoded in {time.time() - t0:.0f} s", flush=True)
    by_id = {}
    for nd in prog.root.body.nodes:
        by_id.setdefault(nd.fn.id, nd.fn)
    todo = [f for f in sorted(kinds, key=lambda f: -kinds[f]["gates"]) if kinds[f]["gates"] > 0]
    if a.only:
        todo = [f for f in todo if f in a.only.split(",")]
    tmp = out.parent / f".{out.name}.d"
    tmp.mkdir(exist_ok=True)
    running: dict[int, tuple[str, float]] = {}
    results: dict[str, dict] = {}

    def reap(block: bool) -> None:
        while running:
            pid, status = os.waitpid(-1, 0 if block else os.WNOHANG)
            if pid == 0:
                break
            fam, started = running.pop(pid)
            f = tmp / f"{fam}.json"
            results[fam] = json.loads(f.read_text()) if f.exists() else {
                "kind": fam, "outcome": "crashed", "why": f"child exit status {status} after {time.time() - started:.0f} s"}
            print(f"A8 kind {fam}: {results[fam]['outcome']} {json.dumps(results[fam].get('attempts', [])[-1:], default=str)[:600]}",
                  flush=True)
            if block:
                break

    for fam in todo:
        while len(running) >= a.jobs:
            for pid, (f, started) in list(running.items()):
                if time.time() - started > a.timeout_s:
                    os.kill(pid, signal.SIGKILL)
                    (tmp / f"{f}.json").write_text(json.dumps({"kind": f, "outcome": "timeout", "why": f"over {a.timeout_s} s"}))
            reap(block=False)
            if len(running) >= a.jobs:
                time.sleep(2)
        pid = os.fork()
        if pid == 0:
            try:
                cap = a.mem_gb << 30
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
                rec = probe_kind(fam, by_id[kinds[fam]["spec"]], a.unit_gates)
            except BaseException as e:  # noqa: BLE001
                rec = {"kind": fam, "outcome": "error", "why": f"{type(e).__name__}: {str(e)[:400]}", "trace": traceback.format_exc()[-2000:]}
            (tmp / f"{fam}.json").write_text(json.dumps(rec, default=str))
            os._exit(0)
        running[pid] = (fam, time.time())
    while running:
        for pid, (f, started) in list(running.items()):
            if time.time() - started > a.timeout_s:
                os.kill(pid, signal.SIGKILL)
                (tmp / f"{f}.json").write_text(json.dumps({"kind": f, "outcome": "timeout", "why": f"over {a.timeout_s} s"}))
        reap(block=False)
        if running:
            time.sleep(2)
    doc = {"descriptor": a.descriptor, "program_digest": digest, "gates": prog.gates, "computing_gates": comp,
           "kinds": {f: {**kinds[f], **results.get(f, {"outcome": "not probed"})} for f in sorted(kinds, key=lambda f: -kinds[f]["gates"])},
           "seconds": round(time.time() - t0, 1)}
    out.write_text(json.dumps(doc, indent=1, default=str))
    tally = Counter(r.get("outcome") for r in doc["kinds"].values() if r["gates"] > 0)
    print(f"A8 kinds: {dict(tally)} in {doc['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
