"""b-e2e-v2 CLI: the end-to-end B-Ligero proof at the checker-v2 shape.

    python -m backends.direct.ligero.v2.run gate  --root fixtures/bench-instances/v1 --device cuda --vus 8
    python -m backends.direct.ligero.v2.run bench --root fixtures/bench-instances/v1 --device cuda --l 16384 --reps 2
    python -m backends.direct.ligero.v2.run shape --l 16384

``gate``: (1) the first ``--vus`` honest ``vu-k1536`` instances in one proof (every recorded accumulator checked
against the packed word) -> accept, plus the prover's own constraint self-check; (2) every ``vu-k1536-neg`` instance
as a VU whose last unit claims the wrong word (``reject``-class operands: the prover cannot form a witness and the
verifier rejects a foreign proof for the statement) -> reject; (3) the a-v2 style unit negatives (one column of one
unit mutated before commitment: every hint, every state-out link, one lookup output per table, operand off the
domain, hint swap) and the four chain negatives -> reject; (4) the BabyBear battery's ``chain`` and ``epilogue``
classes mapped onto the v2 columns (the accumulator handed on as c + p; RN-even tie hints toggled).  An accepted
negative exits 2.

``bench``: ``--vus`` VUs per proof at ``--l`` (default: as many as fit) for ``--reps`` proofs, and ``--total-vus``
covered by ``N_subbatches`` proofs with the union bound; per-bucket ms, proof bytes, verifier seconds, peak memory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from verity_numerical.bench import instances
from verity_numerical.checker import Reject
from verity_numerical.checker.params import REAL
from verity_numerical.security import accounting as acc

from ..field import P
from . import protocol as pr
from .system import STATE, UnitArrays, V2System, build_system, unit_arrays, vu_units, zero_state

STEPS = 96
K = 1536

_SYS: V2System | None = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# witness generation (pool)
# ---------------------------------------------------------------------------------------------------------------------

def _init(sysm):
    global _SYS
    _SYS = sysm


def _vu_worker(task) -> UnitArrays | str:
    x, w, y16, accs = task
    try:
        return unit_arrays(_SYS, vu_units(_SYS, x, w, y16, accs))
    except Reject as e:
        return f"Reject: {e}"


def build_vus(sysm: V2System, tier, ids, procs: int, claimed=None, check_accs: bool = True):
    tasks = []
    for k, i in enumerate(ids):
        x, w, y16 = tier.instance(i)
        accs = tier.accs(i) if (check_accs and tier.acc is not None) else None
        tasks.append((x, w, int(y16) if claimed is None else int(claimed[k]), accs))
    if procs <= 1:
        _init(sysm)
        return [_vu_worker(t) for t in tasks]
    with Pool(procs, initializer=_init, initargs=(sysm,)) as pool:
        return pool.map(_vu_worker, tasks, chunksize=1)


# ---------------------------------------------------------------------------------------------------------------------
# negatives
# ---------------------------------------------------------------------------------------------------------------------

def _mutations(sysm: V2System, units: list[list[int]]) -> list[tuple[str, list[list[int]]]]:
    """a-v2 style unit negatives on a real VU: each returns a mutated copy of the VU's units (before derivation of
    nothing -- the columns are committed as given, the prover's constraint check is skipped)."""
    ci = sysm.col_idx
    out = []
    u = len(units) // 2                      # a middle unit
    last = len(units) - 1

    def mut(name, uidx, col, delta=1):
        cp = [list(v) for v in units]
        cp[uidx][col] = cp[uidx][col] + delta
        out.append((name, cp))

    for path in sysm.hint_paths:                       # every hint +1
        mut(f"hint {path}+1", u, ci[path])
    for path in sysm.epi_paths:                        # every epilogue hint +1 (last unit: the one that binds y16)
        mut(f"epi {path}+1", last, ci[path])
    for n in STATE:                                    # state-out links (+1 on y.* of unit u, x.* of u+1 untouched)
        mut(f"state y.{n}+1", u, ci[f"y.{n}"])
    for n in STATE:                                    # incoming state re-linked: both sides shifted (chain break)
        cp = [list(v) for v in units]
        cp[u][ci[f"y.{n}"]] += 1
        cp[u + 1][ci[f"x.{n}"]] += 1
        out.append((f"chain relink {n}", cp))
    # one lookup output column per table (the first non-key column of the first query into the table)
    seen = set()
    for tname, lins in sysm.lookups:
        if tname in seen or len(lins) < 2:
            continue
        seen.add(tname)
        coef, _ = lins[1]
        if coef:
            r = next(iter(coef))
            mut(f"lookup {tname} output col {sysm.ctx.cols[r]}+1", u, r)
    # operand off the domain (NaN/Inf exponent) with the other columns kept
    mut("operand a[0] -> 0x7f80 (Inf)", u, ci["x.a[0]"], 0x7F80 - units[u][ci["x.a[0]"]])
    mut("operand b[3] -> 0x7fc0 (NaN)", u, ci["x.b[3]"], 0x7FC0 - units[u][ci["x.b[3]"]])
    # hint swap: unit u takes unit u+1's hints
    cp = [list(v) for v in units]
    for path in sysm.hint_paths:
        cp[u][ci[path]] = units[u + 1][ci[path]]
    out.append(("hint swap u <- u+1", cp))
    # false overflow flag on the last unit
    for path in sysm.epi_paths:
        if path.endswith("ovf"):
            mut(f"false {path}", last, ci[path], 1 - units[last][ci[path]])
            break
    # BabyBear battery 'chain' class (the accumulator word handed on as c + p) is VACUOUS under v2: the state is
    # chained as (s, e, M, z) field elements, never as one 33-bit word, so "M + p" is the same element (the same
    # committed tableau, the same proof); the range/lookup rows bind M in [0, 2^24).  The 4 cases are recorded as
    # not applicable (the same finding as b-air-v2's battery_adapter --v2).
    return out


def _epilogue_tie_negatives(sysm: V2System, units: list[list[int]]) -> list[tuple[str, list[list[int]]]]:
    """BabyBear battery 'epilogue' class: RN-even tie with the rounding hints toggled (lsb/carry/rem style)."""
    ci = sysm.col_idx
    last = len(units) - 1
    out = []
    for path in sysm.epi_paths:
        low = path.rsplit(".", 1)[-1]
        if low in ("lsb", "carry", "rnd", "round", "tie", "half"):
            cp = [list(v) for v in units]
            cp[last][ci[path]] = 1 - cp[last][ci[path]]
            out.append((f"epilogue tie toggle {path}", cp))
    return out


def _neg_proof(sysm, cfg, units, y16, device):
    """Prove a (mutated) single VU without the self-check; the column integers are committed as given."""
    return pr.prove(sysm, cfg, units, np.asarray([y16], dtype=np.int64), device, check=False)


# ---------------------------------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------------------------------

def cmd_shape(a):
    sysm = build_system(REAL)
    lay = sysm.layout(a.l)
    cfg = pr.config_for(a.l, steps=STEPS)
    info = {"l": a.l, "n": cfg.n, "t": cfg.t, "n_vus": cfg.n_vus, "n_cols": lay.n_cols, "n_prods": lay.n_prods,
            "m1": lay.m1, "m2": lay.m2, "m": lay.m, "mult_rows": lay.mult_rows,
            "committed_bytes_int32": lay.m * cfg.n * 4, "tables": {k: v.size for k, v in sysm.tables.items()},
            "lookups": len(sysm.lookups), "asserts": len(sysm.asserts), "system_id": sysm.system_id.hex()[:16]}
    info["soundness"] = pr.soundness(cfg, sysm, sysm.n_prods + 6 * len(sysm.lookups))
    print(json.dumps(info, indent=1, default=str))


def cmd_gate(a):
    device = torch.device(a.device)
    root = Path(a.root)
    sysm = build_system(REAL)
    log(f"system {sysm.system_id.hex()[:16]}: {len(sysm.ctx.cols)} cols, {sysm.n_prods} prods, {len(sysm.lookups)} lookups, "
        f"{len(sysm.asserts)} asserts; tables {dict((k, v.size) for k, v in sysm.tables.items())}")
    report = {"system_id": sysm.system_id.hex(), "positives": [], "negatives": [], "mutations": [], "battery": []}
    failures = 0
    # (1) honest
    pos = instances.load("vu-k1536", root)
    ids = list(range(min(a.vus, pos.n)))
    l_pos = a.l_honest or 1 << (len(ids) * STEPS - 1).bit_length()
    cfg = pr.config_for(l_pos, steps=STEPS, n_vus=len(ids))
    t0 = time.perf_counter()
    blocks = build_vus(sysm, pos, ids, a.procs)
    for i, b in zip(ids, blocks):
        if isinstance(b, str):
            raise SystemExit(f"honest instance {i} rejected by the witness builder: {b}")
    log(f"honest witnesses for {len(ids)} VUs: {time.perf_counter() - t0:.1f}s (accumulators checked)")
    y16 = np.asarray([pos.instance(i)[2] for i in ids], dtype=np.int64)
    proof = pr.prove(sysm, cfg, UnitArrays.concat(blocks), y16, device, check=True)
    ok, why = pr.verify(sysm, cfg, y16, proof, device)
    log(f"honest proof: l={l_pos} t={cfg.t} n_vus={len(ids)} accepted={ok} ({why}) prover {proof.timings['total']:.2f}s "
        f"bytes {proof.n_bytes()['total'] / 1e6:.2f} MB")
    report["positives"].append({"ids": ids, "l": l_pos, "accepted": ok, "why": why, "timings": proof.timings,
                                "bytes": proof.n_bytes()})
    if not ok:
        failures += 1
    # wrong claimed word on the honest witness (statement binding)
    y_bad = y16.copy()
    y_bad[0] ^= 1
    ok, why = pr.verify(sysm, cfg, y_bad, proof, device)
    report["negatives"].append({"id": "honest proof, claimed word flipped", "accepted": ok, "why": why})
    failures += ok
    # (2) vu-k1536-neg
    neg = instances.load("vu-k1536-neg", root)
    cols = {n: i for i, n in enumerate(neg.columns)}
    nids = list(range(neg.n if a.neg_limit <= 0 else min(neg.n, a.neg_limit)))
    cfg1 = pr.config_for(a.l, steps=STEPS, n_vus=1)
    nblocks = build_vus(sysm, neg, nids, a.procs, check_accs=False)
    foreign = None
    for i, b in zip(nids, nblocks):
        row = neg.index[i]
        meta = {"id": i, "mistake": row[cols["mistake"]], "verdict": row[cols["verdict"]]}
        claimed = np.asarray([neg.instance(i)[2]], dtype=np.int64)
        if isinstance(b, str):
            if foreign is None:
                foreign = pr.prove(sysm, cfg1, blocks[0], np.asarray([y16[0]], dtype=np.int64), device)
            ok, why = pr.verify(sysm, cfg1, claimed, foreign, device)
            why = f"witness builder: {b}; verifier on a foreign proof: {why}"
        else:
            proof = pr.prove(sysm, cfg1, b, claimed, device, check=False)
            ok, why = pr.verify(sysm, cfg1, claimed, proof, device)
        report["negatives"].append({**meta, "accepted": ok, "why": why})
        if ok:
            log(f"!! NEGATIVE ACCEPTED: {meta}")
            failures += 1
    log(f"vu-k1536-neg: {sum(not r['accepted'] for r in report['negatives'])}/{len(report['negatives'])} rejected")
    # (3) unit + chain mutations on honest VU 0
    x, w, y0 = pos.instance(ids[0])
    units0 = vu_units(sysm, x, w, y0, pos.accs(ids[0]))
    muts = _mutations(sysm, units0)
    if a.mut_limit > 0:
        muts = muts[:a.mut_limit]
    for name, mu in muts:
        try:
            proof = pr.prove(sysm, cfg1, mu, np.asarray([y0], dtype=np.int64), device, check=False)
            ok, why = pr.verify(sysm, cfg1, np.asarray([y0]), proof, device)
        except Exception as e:  # noqa: BLE001
            ok, why = False, f"prover raised {type(e).__name__}: {e}"
        report["mutations"].append({"mutation": name, "accepted": ok, "why": why})
        if ok:
            log(f"!! MUTATION ACCEPTED: {name}")
            failures += 1
    log(f"mutations: {sum(not r['accepted'] for r in report['mutations'])}/{len(report['mutations'])} rejected")
    # (4) battery epilogue class
    for name, mu in _epilogue_tie_negatives(sysm, units0):
        proof = pr.prove(sysm, cfg1, mu, np.asarray([y0], dtype=np.int64), device, check=False)
        ok, why = pr.verify(sysm, cfg1, np.asarray([y0]), proof, device)
        report["battery"].append({"case": name, "accepted": ok, "why": why})
        failures += ok
    report["failures"] = int(failures)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    log(f"gate: {'PASS' if not failures else 'FAIL'} ({failures} failures) -> {out}")
    if failures:
        sys.exit(2)


def cmd_bench(a):
    device = torch.device(a.device)
    root = Path(a.root)
    sysm = build_system(REAL)
    pos = instances.load("vu-k1536", root)
    cfg = pr.config_for(a.l, steps=STEPS, n_vus=a.vus)
    lay = sysm.layout(a.l)
    log(f"l={cfg.l} n={cfg.n} t={cfg.t} n_vus={cfg.n_vus} m1={lay.m1} m2={lay.m2} m={lay.m} "
        f"committed {lay.m * cfg.n * 4 / 2**30:.2f} GiB (int32) / {lay.m * cfg.n * 8 / 2**30:.2f} GiB (int64 as run)")
    total = a.total_vus or cfg.n_vus
    n_sub = -(-total // cfg.n_vus)
    results = []
    all_ids = [(i % pos.n) for i in range(n_sub * cfg.n_vus)]
    for s in range(n_sub if a.all_subbatches else min(n_sub, a.reps)):
        ids = all_ids[s * cfg.n_vus:(s + 1) * cfg.n_vus]
        t0 = time.perf_counter()
        blocks = build_vus(sysm, pos, ids, a.procs, check_accs=not a.no_acc_check)
        bad = [b for b in blocks if isinstance(b, str)]
        if bad:
            raise SystemExit(bad[0])
        blk = UnitArrays.concat(blocks)
        t_wit = time.perf_counter() - t0
        y16 = np.asarray([pos.instance(i)[2] for i in ids], dtype=np.int64)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        proof = pr.prove(sysm, cfg, blk, y16, device, check=(s == 0 and not a.no_check))
        t1 = time.perf_counter()
        ok, why = pr.verify(sysm, cfg, y16, proof, device)
        t_ver = time.perf_counter() - t1
        nb = proof.n_bytes()
        r = {"subbatch": s, "n_vus": cfg.n_vus, "accepted": ok, "why": why, "witness_gen_host_s": t_wit,
             "prover_s": proof.timings, "verifier_s": t_ver, "proof_bytes": nb,
             "peak_device_GiB": proof.timings.get("peak_device_bytes", 0) / 2**30}
        results.append(r)
        log(f"sub-batch {s}: accepted={ok} ({why}); host witness {t_wit:.1f}s; prover {proof.timings['total']:.2f}s "
            f"({', '.join(f'{k} {v:.2f}' for k, v in proof.timings.items() if k not in ('total', 'peak_device_bytes'))}); "
            f"verifier {t_ver:.2f}s; proof {nb['total'] / 1e6:.1f} MB; peak {r['peak_device_GiB']:.2f} GiB")
        if not ok:
            raise SystemExit("bench proof rejected")
    snd = pr.soundness(cfg, sysm, sysm.n_prods + 6 * len(sysm.lookups))
    per_proof = 2.0 ** snd["total_log2"]
    union_log2 = acc.log2(per_proof * n_sub) if hasattr(acc, "log2") else snd["total_log2"] + np.log2(n_sub)
    steady = [r for r in results[1:]] or results
    prover_ms = float(np.mean([r["prover_s"]["total"] for r in steady])) * 1e3
    summary = {"l": cfg.l, "n": cfg.n, "t": cfg.t, "vus_per_proof": cfg.n_vus, "total_vus": total, "N_subbatches": n_sub,
               "prover_ms_per_proof": prover_ms, "us_per_vu": prover_ms * 1e3 / cfg.n_vus,
               "prover_ms_per_4096_vus": prover_ms * 4096 / cfg.n_vus,
               "verifier_s": float(np.mean([r["verifier_s"] for r in steady])),
               "proof_MB": results[0]["proof_bytes"]["total"] / 1e6,
               "peak_device_GiB": max(r["peak_device_GiB"] for r in results),
               "buckets_ms": {k: float(np.mean([r["prover_s"].get(k, 0.0) for r in steady])) * 1e3 for k in steady[0]["prover_s"]
                              if k not in ("peak_device_bytes",)},
               "witness_gen_host_s": float(np.mean([r["witness_gen_host_s"] for r in results])),
               "soundness_per_proof": snd, "soundness_union_log2": float(union_log2),
               "layout": {"m1": lay.m1, "m2": lay.m2, "m": lay.m, "n_cols": lay.n_cols, "n_prods": lay.n_prods},
               "device": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else str(device),
               "results": results}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=str))
    log(f"bench -> {out}: {prover_ms:.1f} ms/proof, {summary['us_per_vu']:.1f} us/VU, "
        f"{summary['prover_ms_per_4096_vus']:.1f} ms per 4096 VUs, union 2^{union_log2:.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shape")
    s.add_argument("--l", type=int, default=16384)
    s.set_defaults(fn=cmd_shape)
    g = sub.add_parser("gate")
    g.add_argument("--root", default="fixtures/bench-instances/v1")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--vus", type=int, default=8)
    g.add_argument("--l", type=int, default=128, help="l for the single-VU negatives")
    g.add_argument("--l-honest", type=int, default=0)
    g.add_argument("--neg-limit", type=int, default=0)
    g.add_argument("--mut-limit", type=int, default=0)
    g.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    g.add_argument("--out", default="backends/direct/ligero/v2/results/gate.json")
    g.set_defaults(fn=cmd_gate)
    b = sub.add_parser("bench")
    b.add_argument("--root", default="fixtures/bench-instances/v1")
    b.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    b.add_argument("--l", type=int, default=16384)
    b.add_argument("--vus", type=int, default=0)
    b.add_argument("--total-vus", type=int, default=0)
    b.add_argument("--reps", type=int, default=2)
    b.add_argument("--all-subbatches", action="store_true")
    b.add_argument("--no-check", action="store_true")
    b.add_argument("--no-acc-check", action="store_true")
    b.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    b.add_argument("--out", default="backends/direct/ligero/v2/results/bench.json")
    b.set_defaults(fn=cmd_bench)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
