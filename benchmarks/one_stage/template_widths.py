"""CIR-57 (measurement only, scratch branch `cursor/template-widths-21f3`): every template-query Definition a served row's
Programs offer, with the bits an instance commits under 9431e36d39's held rule (`partition._check_units` on a template
query: one activation's Input gates and returned gates, their declared widths summed), and the re-cut cost at 16-bit
units.

    PYTHONPATH=<tree>:<tree>/integrations/vllm python benchmarks/one_stage/template_widths.py \\
      --row ROW --build DIR [--record config_record.json] --out OUT.json [--parts step,request] [--ranks 0,1]

``DIR`` is a row's Build tree (``build/rank<r>/build_<part>/descriptor.json.gz``, or ``build_<part>/`` for TP1). A row's
templates are the ones its served paths register: each root Call's own Definition (`pipeline.serving_rows`, "any row's own
Definition"), the templates a ``CUT`` family's body batches (``RoPEHead_v1``, ``GemmCoordinate_v1``), and A11's unit chain
(`a11_kinds.chain` at 4,096 gates: the Call's dominant callees, the GLM one-stage audit's units).

* **Bits.** ``bits`` is the rule's count read off the descriptor: the Input gates' widths summed through the body (batch and
  scan multiplied), plus each distinct returned gate's width (an Input gate returned is counted once), so it costs the
  descriptor and the returns, not the unrolled activation. Where the activation has at most ``--literal-max`` gates the
  ported rule itself runs (``partition.evaluate`` of a one-instance population under ``Q_template_instance``, refused or
  not) and ``_committed_bits`` gives ``bits_literal``; the two must agree.
* **Re-cut.** For an over-16 template of at most ``--cone-max`` gates, the activation's gates are walked once and each
  returned gate's cone found (reach bitsets). ``units16`` is the 16-bit pieces its returns and Input gates need, packed in
  return order (a run of narrow gates up to 16 bits is one piece, so a Boolean Definition's bits go 16 to a unit; a wider
  gate is ``ceil(w / 16)`` pieces). ``ands_naive`` puts every piece in its own unit, recomputing its whole cone; ``ands_2level``
  commits each shared value a private cone reads (the frontier) as its own pieces, each recomputing its own cone, and
  the outputs' private cones once a piece. Gate ANDs are the circuit-check pins (`*.circuit-check.json`, `pins/`): a
  primitive's piece ANDs, and a Boolean Definition's where pinned; ``ands_coverage`` is the share of the activation's
  gates with a pinned weight.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

sys.setrecursionlimit(100000)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402
from verity.ml.circuits.codec import descriptor_id  # noqa: E402
from verity.ml.circuits.cut import QueryInapplicable, call_scope  # noqa: E402
from verity.ml.circuits.defs import PrimitiveDefinition  # noqa: E402
from verity.ml.circuits.layout import resolve  # noqa: E402
from verity.ml.circuits.program import is_input  # noqa: E402
from verity.ml.circuits.types import Value  # noqa: E402
from verity_vllm.program.boolean import load_program  # noqa: E402

import a11_kinds as K  # noqa: E402

CUT = ("RoPE_v1", "Gemm_v1")
UNIT_GATES = 4096
ROOT = HERE.parent.parent


def pins() -> tuple[dict[str, int], dict[str, int]]:
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.circuit-check.json",
                            "tools/verity/circuit_check/src/circuit_check/pins/*.json"], capture_output=True, text=True).stdout.split()
    pieces, defs = {}, {}
    for f in files:
        d = json.loads((ROOT / f).read_text())
        for k, v in (d.get("pieces") or {}).items():
            if isinstance(v, dict) and "and" in v:
                pieces[k] = int(v["and"])
        for k, v in (d.get("definitions") or {}).items():
            if isinstance(v, dict) and "and" in v:
                defs[k] = int(v["and"])
    return pieces, defs


PIECE_ANDS, DEF_ANDS = pins()


def width(prim) -> int:
    return prim.ret.w if isinstance(getattr(prim, "ret", None), Value) else 0


def input_bits(fn, memo: dict) -> int:
    k = id(fn)
    if k in memo:
        return memo[k]
    if isinstance(fn, PrimitiveDefinition):
        r = width(fn) if is_input(fn) else 0
    else:
        r = sum(K.mult(nd) * input_bits(nd.fn, memo) for nd in fn.body.nodes)
    memo[k] = r
    return r


def returned(fn) -> list[tuple[int, int, bool]]:
    """(local gate, width, is Input) of each distinct returned gate inside one activation of fn, in return order."""
    c, call = call_scope(fn)
    lo, n = call.offset, fn.gates
    rs = [0] if not hasattr(fn, "body") else [resolve(call, r) - lo for r in fn.body.ret]
    out, seen = [], set()
    for r in rs:
        if 0 <= r < n and r not in seen:
            seen.add(r)
            g = c.gate(lo + r)
            out.append((r, width(g.prim), is_input(g.prim)))
    return out


def pack(gates: list[int], wid: list[int]) -> tuple[list[int], list[int]]:
    """16-bit pieces over gates in order: a run of gates of at most 16 bits together is one piece (weight 1), a wider gate is
    ceil(w / 16) pieces on its own (one group of that weight). Returns each gate's group and each group's weight."""
    group, weight, run = [], [], 17
    for g in gates:
        w = max(wid[g], 1)
        if w > 16:
            group.append(len(weight))
            weight.append(math.ceil(w / 16))
            run = 17
            continue
        if run + w > 16:
            weight.append(1)
            run = 0
        run += w
        group.append(len(weight) - 1)
    return group, weight


def literal(fn) -> dict:
    """The ported rule on a one-instance population of fn, and `_committed_bits` of its activation."""
    t = time.time()
    try:
        P.evaluate(P.population_program(fn, 1), P.template_instance_query(fn))
        verdict = "accepted"
    except QueryInapplicable as e:
        verdict = f"refused: {e}"
    except (TypeError, ValueError) as e:
        verdict = f"no population: {e}"
    (b,) = P._committed_bits(fn, [(0, fn.gates)])
    return {"verdict": verdict, "bits_literal": b, "literal_s": round(time.time() - t, 2)}


def recut(fn, rets: list[tuple[int, int, bool]]) -> dict:
    """Cones of the returned gates and the 16-bit re-cut estimates (module docstring)."""
    t = time.time()
    c, call = call_scope(fn)
    lo, n = call.offset, fn.gates
    ops: list[list[int]] = [None] * n  # type: ignore[list-item]
    wts = [0] * n
    wid = [0] * n
    known = 0
    inputs = []
    for j in range(n):
        g = c.gate(lo + j)
        ops[j] = [o - lo for o in g.operands if lo <= o < lo + j]
        wid[j] = width(g.prim)
        pid = g.prim.id
        if is_input(g.prim):
            inputs.append(j)
        a = PIECE_ANDS.get(pid)
        if a is None and (pid.startswith("Const") or is_input(g.prim)):
            a = 0
        if a is not None:
            known += 1
            wts[j] = a
    outs = [r for r, _w, inp in rets if not inp] + inputs

    def cones(gates: list[int]) -> tuple[list[int], list[int], Callable[[int], int]]:
        group, weight = pack(gates, wid)
        reach = [0] * n
        for g, k in zip(gates, group):
            reach[g] |= 1 << k
        for j in range(n - 1, -1, -1):
            if reach[j]:
                for o in ops[j]:
                    reach[o] |= reach[j]
        by_w: dict[int, int] = defaultdict(int)
        for k, w in enumerate(weight):
            by_w[w] |= 1 << k
        return reach, weight, lambda mask: sum(w * (mask & m).bit_count() for w, m in by_w.items())

    reach, pieces, weighted = cones(outs)
    live = [j for j in range(n) if reach[j]]
    today = sum(wts[j] for j in live)
    naive = sum(wts[j] * weighted(reach[j]) for j in live)
    shared = {j for j in live if reach[j].bit_count() >= 2}
    frontier = sorted({o for j in live if j not in shared for o in ops[j] if o in shared})
    reach_f, fpieces, fweighted = cones(frontier)
    shared_cost = sum(wts[j] * fweighted(reach_f[j]) for j in live if reach_f[j])
    private_cost = sum(wts[j] * weighted(reach[j]) for j in live if j not in shared)
    return {"units16": sum(pieces), "frontier_values": len(frontier), "units16_2level": sum(pieces) + sum(fpieces),
            "ands_today_pinned": today, "ands_naive": naive, "ands_2level": shared_cost + private_cost,
            "naive_x": round(naive / today, 3) if today else None,
            "x_2level": round((shared_cost + private_cost) / today, 3) if today else None,
            "shared_gate_share": round(len(shared) / len(live), 4) if live else None,
            "ands_coverage": round(known / n, 4) if n else None, "live_gates": len(live), "recut_s": round(time.time() - t, 2)}


def describe(fn, memo: dict) -> dict:
    """The rule's count off the descriptor and the returns (no unrolled walk)."""
    t = time.time()
    rets = returned(fn)
    ib = input_bits(fn, memo)
    rb = sum(w for _r, w, inp in rets if not inp)
    return {"id": fn.id, "family": K.family(fn.id), "gates": fn.gates, "bits": ib + rb, "input_bits": ib,
            "returned_bits": rb, "returned_gates": len(rets), "returned_widths": dict(Counter(w for _r, w, _i in rets)),
            "boolean": K.is_boolean(fn), "pinned_ands": DEF_ANDS.get(fn.id), "describe_s": round(time.time() - t, 2)}


def deep(fn, rec: dict, a: argparse.Namespace, *, recut_too: bool) -> None:
    """The ported rule and `_committed_bits` (at most `--literal-max` gates), and the re-cut (over 16 bits, at most
    `--cone-max` gates)."""
    t = time.time()
    if fn.gates <= a.literal_max and "bits_literal" not in rec and "literal_error" not in rec:
        try:
            rec.update(literal(fn))
            if rec["bits_literal"] != rec["bits"]:
                rec["disagree"] = True
        except Exception as e:  # noqa: BLE001
            rec["literal_error"] = repr(e)[:300]
    if recut_too and (rec.get("bits") or 0) > P.UNIT_BITS and fn.gates <= a.cone_max and "recut" not in rec:
        try:
            rec["descriptor_id"] = descriptor_id(fn)
        except Exception as e:  # noqa: BLE001
            rec["descriptor_id"] = f"error: {e!r}"[:200]
        try:
            rec["recut"] = recut(fn, returned(fn))
        except Exception as e:  # noqa: BLE001
            rec["recut"] = {"error": repr(e)[:300]}
    rec["deep_s"] = round(rec.get("deep_s", 0) + time.time() - t, 2)


def candidates(root_fn) -> dict[str, tuple[object, list[str], int]]:
    """{template id: (Definition, hows, instances per Call of root_fn)} of the templates a Call of root_fn offers; a template
    two ways reach (a CUT member that is also the chain's) is counted once, at the larger multiplicity."""
    out: dict[str, tuple[object, list[str], int]] = {}

    def add(f, how: str, per: int) -> None:
        g, hows, p = out.get(f.id, (f, [], 0))
        out[f.id] = (g, hows + ([how] if how not in hows else []), max(p, per))

    add(root_fn, "row", 1)
    if K.family(root_fn.id) in CUT:
        for nd in root_fn.body.nodes:
            if nd.form in ("call", "batch") and not is_input(nd.fn):
                add(nd.fn, "cut", int(nd.n) if nd.form == "batch" else 1)
    per = 1
    for f, m in K.chain(root_fn, UNIT_GATES)[1:]:
        per *= m
        if not isinstance(f, PrimitiveDefinition):
            add(f, "a11-chain", per)
    return out


def descriptors(build: Path) -> list[tuple[int, str, Path]]:
    found = sorted(build.glob("build/rank*/build_*/descriptor.json.gz")) or sorted(build.glob("build_*/descriptor.json.gz"))
    out = []
    for p in found:
        rank = int(p.parent.parent.name[4:]) if p.parent.parent.name.startswith("rank") else 0
        out.append((rank, p.parent.name[len("build_"):], p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--pins", default=None, help="records.json's {rank: {step, workload, requests}} digests to check")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--parts", default=None, help="comma-separated parts (step, request_..., workload); default all")
    ap.add_argument("--ranks", default=None)
    ap.add_argument("--literal-max", type=int, default=400_000)
    ap.add_argument("--literal-cheap", type=int, default=20_000, help="every template this small gets the ported rule")
    ap.add_argument("--cone-max", type=int, default=3_000_000)
    ap.add_argument("--deep-per-family", type=int, default=3, help="specs per (family, how) given the rule and the re-cut")
    a = ap.parse_args()
    t0 = time.time()
    want_parts = set(a.parts.split(",")) if a.parts else None
    want_ranks = {int(r) for r in a.ranks.split(",")} if a.ranks else None
    pinned = json.loads(Path(a.pins).read_text()) if a.pins else None
    memo: dict = {}
    seen: dict[str, dict] = {}
    deep_count: Counter = Counter()
    programs, units = [], defaultdict(lambda: defaultdict(int))
    for rank, part, path in descriptors(a.build):
        if (want_ranks is not None and rank not in want_ranks) or (want_parts is not None and not any(part.startswith(w) for w in want_parts)):
            continue
        t = time.time()
        prog, digest = load_program(path)
        want = None
        if pinned is not None:
            pr = pinned.get(str(rank), {})
            reqs = pr.get("requests") or {}
            want = pr.get(part) if part in ("step", "workload") else reqs.get(f"build_{part}", reqs.get(part))
        roots = Counter()
        fns = {}
        for nd in prog.root.body.nodes:
            if is_input(nd.fn):
                continue
            roots[nd.fn.id] += K.mult(nd)
            fns.setdefault(nd.fn.id, nd.fn)
        key = f"r{rank}/{part}"
        here: dict[str, object] = {}
        for fid, calls in roots.items():
            for tid, (f, hows, per) in candidates(fns[fid]).items():
                units[tid][key] += calls * per
                here.setdefault(tid, f)
                if tid not in seen:
                    try:
                        seen[tid] = describe(f, memo)
                    except Exception as e:  # noqa: BLE001
                        seen[tid] = {"id": tid, "family": K.family(tid), "gates": f.gates, "bits": None,
                                     "describe_error": repr(e)[:300]}
                    seen[tid]["first_seen"] = key
                rec = seen[tid]
                rec["hows"] = sorted(set(rec.get("hows", [])) | set(hows))
                rec["root_families"] = sorted(set(rec.get("root_families", [])) | {K.family(fid)})
        for tid, f in here.items():
            if f.gates <= a.literal_cheap:
                deep(f, seen[tid], a, recut_too=False)
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for tid in here:
            for how in seen[tid]["hows"]:
                groups[(seen[tid]["family"], how)].append(tid)
        for (fam, how), tids in groups.items():
            for tid in sorted(tids, key=lambda t: -units[t][key]):
                if deep_count[(fam, how)] >= a.deep_per_family:
                    break
                if seen[tid].get("deep_done"):
                    continue
                deep(here[tid], seen[tid], a, recut_too=True)
                seen[tid]["deep_done"] = True
                deep_count[(fam, how)] += 1
        del here
        programs.append({"rank": rank, "part": part, "digest": digest, "pinned": want, "digest_ok": (want == digest) if want else None,
                         "root_calls": sum(roots.values()), "root_definitions": len(roots), "gates": prog.gates,
                         "seconds": round(time.time() - t, 1)})
        print(f"[widths] {a.row} {key}: {len(roots)} root Definitions, {len(seen)} templates so far, {time.time() - t:.0f} s "
              f"(digest {'ok' if programs[-1]['digest_ok'] else programs[-1]['digest_ok']})", flush=True)
        del prog, fns
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"row": a.row, "programs": programs, "templates": seen,
                                     "units": {k: dict(v) for k, v in units.items()}, "partial": True}, indent=1, default=str))
    refused = sorted((r for r in seen.values() if (r["bits"] or 0) > P.UNIT_BITS), key=lambda r: -r["bits"])
    doc = {"row": a.row, "rule": "9431e36d39: a template instance commits at most 16 bits (Input + returned gates)",
           "programs": programs, "templates": seen, "units": {k: dict(v) for k, v in units.items()},
           "refused": [r["id"] for r in refused], "disagreements": [r["id"] for r in seen.values() if r.get("disagree")],
           "errors": [r["id"] for r in seen.values() if r.get("describe_error") or r.get("literal_error")],
           "seconds": round(time.time() - t0, 1)}
    a.out.write_text(json.dumps(doc, indent=1, default=str))
    print(f"[widths] {a.row}: {len(seen)} templates, {len(refused)} over 16 bits, {len(doc['disagreements'])} disagreements, "
          f"{doc['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
