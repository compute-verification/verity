"""Gate and bench for the Ligerito backend (lane ligerito-relation), mirroring ``backends/direct/ligero/relchain.py``.

    python -m backends.direct.ligerito.run gate  --relation fp8-ada --vus 12 --l 256 [--dump-dir D] [--out gate.json]
    python -m backends.direct.ligerito.run bench --relation fp8-ada --total-vus 4096 --reps 3 --coins local [--dump-dir D]
                                                 [--rust-verifier backends/ligerito-verify/target/release/ligerito-verify]
    python -m backends.direct.ligerito.run verify-dir D           # the Python file verifier on a dump (cold, from bytes)

Dump layout (what ``ligerito-verify batch --dir`` reads): ``key.bin`` (LGVK0001), per proof ``<stem>.stmt`` (LGTS0001) +
``<stem>.lgto`` (LGTO0001, must ACCEPT) or ``<stem>.lgto.neg`` (must REJECT), ``<stem>.coins`` for live coins
(``u32 count | 32 B per coin``), ``manifest.json`` ({"files": [{"proof", "stmt", "coins", "expect", "python_verdict", ...}]}).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from backends.direct.ligero import live as live_mod
from backends.direct.ligero.compile import P
from backends.direct.ligero import relchain
from backends.direct.ligero.relchain import K_VU

from . import prove as prove_mod
from .proof import Proof, Statement, read_key, read_proof, read_statement, sha256
from .prove import Prover, Result, make_coins
from .transcript import LiveCoins

BACKEND = "ligerito"
STUBBED = "zk.py not wired (no masks, no salted leaves, no Libra sumcheck masks, no per-column RS padding)"
ZK_PARTIAL = "ligerito-zk row masks (8 uniform + product triple + Libra row, in free rows) under the bit-interleaved index map z_to_f"
ZK_MISSING = ("Libra masks in the three sumchecks, the shift-claim reduction and the product-row constraint (ligerito-sumcheck-2), "
              "per-column RS padding + sparse round-1 terms (ligerito-pcs-fast)")
#: where a cold verify-dir gets each coin mode's coins from (red-team F12 (b)): only the FS transcript is the verifier's own
COIN_PROVENANCE = {"fiat-shamir": "recomputed from the transcript (verifier-derived)",
                   "local": "rebuilt from the manifest's coins_seed (prover-side: no soundness)",
                   "live": "the dump's .coins file, stream binding re-derived (prover-side record: no soundness without the verifier's own log)"}


def instances(rel, n: int, procs: int = 8, cache=None) -> list:
    if rel.name == "fp4-nvf4":
        from backends.direct.ligero.fp4.chain import instances_fp4
        return instances_fp4(n)
    return relchain.instances(rel, n, procs=procs, cache=cache)


def instances_digest(rel, n: int) -> str:
    if rel.name == "fp4-nvf4":
        from backends.direct.ligero.fp4 import chain as fc
        return fc.instances_digest(n)
    return relchain.instances_digest(rel, n)


def _git_commit() -> str:
    if os.environ.get("RESEARCH_GIT_COMMIT"):
        return os.environ["RESEARCH_GIT_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _claim_str(sc: dict | None) -> str:
    if not sc:
        return "n/a"
    c = sc["claimed_log2"]
    return f"{'none' if c is None else f'2^{c:.1f}'} [{sc['basis']}; interactive union 2^{sc['interactive_union_log2']:.2f}]"


def _coins_pair(kind: str, seed: int):
    """(prover coins, verifier coins) replaying the same randomness."""
    return make_coins(kind, seed), make_coins(kind, seed)


def _mutate_proof(proof: Proof, what: str) -> Proof:
    pf = copy.deepcopy(proof)
    pcs = pf.pcs
    if what == "opened row (round 1)":
        pcs.rows[0][0, 0] = (pcs.rows[0][0, 0] + 1) % P
    elif what == "opened row (last round)":
        pcs.rows[-1][-1, 3] = (pcs.rows[-1][-1, 3] + 1) % P
    elif what == "root 1":
        pcs.roots[0] = bytes([pcs.roots[0][0] ^ 1]) + pcs.roots[0][1:]
    elif what == "root 2":
        pcs.roots[1] = bytes([pcs.roots[1][0] ^ 1]) + pcs.roots[1][1:]
    elif what == "final vector":
        pcs.final_y[5, 2] = (pcs.final_y[5, 2] + 1) % P
    elif what == "Merkle sibling":
        s = bytearray(pcs.siblings[0])
        s[7] ^= 0x10
        pcs.siblings[0] = bytes(s)
    elif what == "PCS sumcheck message":
        pcs.sumcheck[0][1][2, 0] = (pcs.sumcheck[0][1][2, 0] + 1) % P
    elif what == "zero-check message":
        b = bytearray(pf.sc_bytes)
        b[13 + 4 * 6 * 9 + 8] ^= 1                      # rounds1[0] (after the 9-element opening round)
        pf.sc_bytes = bytes(b)
    elif what == "claim value":
        b = bytearray(pf.sc_bytes)
        b[-24] = (b[-24] + 1) % 256                     # values[-1] coordinate 0
        pf.sc_bytes = bytes(b)
    else:
        raise ValueError(what)
    return pf


PROOF_TAMPERS = ("opened row (round 1)", "opened row (last round)", "root 1", "root 2", "final vector", "Merkle sibling",
                 "PCS sumcheck message", "zero-check message", "claim value")


def _stmt_with(stmt: Statement, **kw) -> Statement:
    st = copy.deepcopy(stmt)
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _flip_operand(rel, a: np.ndarray) -> np.ndarray:
    """``a[10][1]`` with an exponent bit flipped (a word that still decodes)."""
    a = a.copy()
    a[10, 1] ^= 1 << (rel.params.frac if getattr(rel.params, "word_bits", 8) == 16 else 3)
    return a


def v1_forgeries(pv: Prover, bt) -> list[tuple[str, dict, str]]:
    """Red-team V1 forgeries, one per virtual-row class, as ``(name, Prover.prove kwargs, stage)``: the sumchecks run honestly
    on the prover's ``z`` and the commitment is non-zero on the virtual rows so that the verifier's reconstruction from the
    claimed statement gives that ``z``.  ``stage``: where the verifier must reject -- the PCS (the zero claims) for every
    public class; ``next:*`` rows are pinned by the shift, so the combined sumcheck rejects those (without the zero claims
    the verifier accepts every forgery but the ``next:*`` one: ``redteam_v1_test.py``)."""
    rel, steps, st, lay = pv.rel, pv.steps, bt.stmt, bt.lay
    y0, nv0 = st.y[0], st.n_vus[0]

    def bump(col: int) -> np.ndarray:
        y = y0.copy()
        y[col] = (int(y[col]) + 1) % 2 ** rel.y_bits
        return y
    vu_cols = slice((nv0 - 1) * steps, nv0 * steps)

    def zero_last_vu(z, lay_):
        z[:, vu_cols] = 0                   # all-zero columns satisfy every constraint; c_x = 0 at the neighbours' starts

    next0 = torch.zeros(lay.C, dtype=torch.int64)
    next0[5] = 12345
    return [("end row(s): claimed word +1 ulp, true witness", dict(forge_stmt=_stmt_with(st, y=[bump(steps - 1)] + st.y[1:])), "PCS"),
            ("pub:* rows: operand a[10][1] exponent bit flipped, true witness",
             dict(forge_stmt=_stmt_with(st, a=[_flip_operand(rel, st.a[0])] + st.a[1:])), "PCS"),
            ("start/link/end rows: statement n_vus[0] - 1 (same words), true witness",
             dict(forge_stmt=_stmt_with(st, n_vus=[nv0 - 1] + st.n_vus[1:])), "PCS"),
            ("const + every public row: last VU of sub-batch 0 all-zero in z, its y claimed +1",
             dict(forge_stmt=_stmt_with(st, y=[bump(nv0 * steps - 1)] + st.y[1:]), forge_z=zero_last_vu), "PCS"),
            ("next:0 row +12345 in column 5 committed (pinned by the shift)", dict(forge_rows={lay.virt["next:0"]: next0}),
             "combined final")]


# ---------------------------------------------------------------------------------------------------------------------
# Dumps
# ---------------------------------------------------------------------------------------------------------------------

class Dump:
    def __init__(self, root: Path | None, prover: Prover):
        self.root = root
        self.files: list[dict] = []
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            (root / "key.bin").write_bytes(prover.key_bytes)

    def coins_file(self, stem: str, slots: list) -> dict:
        if self.root is None:
            return {}
        (self.root / f"{stem}.coins").write_bytes(len(slots).to_bytes(4, "little") + b"".join(slots))
        return {"coins_file": f"{stem}.coins"}

    def add(self, stem: str, stmt: Statement, proof: Proof, expect: str, verdict: tuple, coins: str, extra=None, seed: int = 1):
        if self.root is None:
            return
        sb, pb = stmt.to_bytes(), proof.to_bytes()
        (self.root / f"{stem}.stmt").write_bytes(sb)
        name = f"{stem}.lgto" if expect == "accept" else f"{stem}.lgto.neg"
        (self.root / name).write_bytes(pb)
        e = {"proof": name, "stmt": f"{stem}.stmt", "expect": expect, "coins": coins, "coins_seed": seed, "proof_bytes": len(pb),
             "proof_sha256": sha256(pb).hex(), "stmt_sha256": sha256(sb).hex(),
             "python_verdict": {"accepted": verdict[0], "reason": verdict[1]}}
        if extra:
            e.update(extra)
        self.files.append(e)

    def close(self, meta: dict):
        if self.root is None:
            return
        (self.root / "manifest.json").write_text(json.dumps({"format": {"proof": "LGTO0001", "statement": "LGTS0001",
                                                                        "key": "LGVK0001"}, **meta, "files": self.files},
                                                            indent=1, default=str))


def rust_batch(verifier: str | None, d: Path) -> dict | None:
    if not verifier or d is None:
        return None
    out = d / "verify_rust.json"
    t0 = time.perf_counter()
    p = subprocess.run([verifier, "batch", "--dir", str(d), "--json", str(out)], capture_output=True, text=True)
    res = {"exit": p.returncode, "wall": time.perf_counter() - t0, "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-2000:]}
    if out.exists():
        try:
            res["json"] = json.loads(out.read_text())
        except ValueError:
            pass
    print(f"rust batch --dir {d}: exit {p.returncode} in {res['wall']:.2f}s", flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------------------------------------------------

def gate(args) -> int:
    pv = Prover(args.relation, args.device, args.l)
    rel, sys_, steps = pv.rel, pv.sys, pv.steps
    log = print
    log(f"{rel.name}: system {sys_.summary()}; key {pv.key_sha256.hex()[:16]} (m={pv.key.m}, n_i={pv.key.n_i}, n_k={pv.key.n_k}, "
        f"K={pv.key.K}); compiled {pv.compile_seconds:.2f}s")
    data = instances(rel, args.vus)
    dump = Dump(Path(args.dump_dir) if args.dump_dir else None, pv)
    report = {"relation": rel.name, "l": args.l, "vus": args.vus, "positives": [], "negatives": [], "system": sys_.summary()}
    failures = 0
    for kind in args.coins:
        bt = pv.batch(data, kind, zk=args.zk)
        lay = bt.lay
        log(f"[{kind}] layout {lay.summary()}; dims {bt.dims.describe()}")
        pc, vc = _coins_pair(kind, 1)
        res = pv.prove(bt, pc)
        ok, why, info = pv.verify(bt.stmt, res.proof, vc)
        nb = res.proof.n_bytes()
        log(f"[{kind}] honest: prover {res.timings['total']:.2f}s, {nb['total']} B, {res.coins_rounds} coins -> {ok} {why} "
            f"(verify {info['seconds']:.3f}s); soundness {_claim_str(info.get('soundness'))}")
        report["positives"].append({"coins": kind, "ok": ok, "why": why, "bytes": nb, "timings": res.timings, "verify": info})
        failures += not ok
        dump.add(f"honest_{kind}", bt.stmt, res.proof, "accept", (ok, why), kind)
        # round trip through bytes
        rt = read_proof(res.proof.to_bytes())
        st_rt = read_statement(bt.stmt.to_bytes())
        ok2, why2, _ = pv.verify(st_rt, rt, make_coins(kind, 1))
        report["positives"].append({"coins": kind, "name": "bytes round trip", "ok": ok2, "why": why2})
        failures += not ok2
        negs: list[tuple[str, tuple, Statement, Proof]] = []

        def neg(name: str, stmt: Statement, proof: Proof, seed: int = 1, stage: str | None = None):
            """``stage``: the reject reason must contain it (a forgery rejected earlier did not test what it names)."""
            v = pv.verify(stmt, proof, make_coins(kind, seed))
            negs.append((name, v, stmt, proof, stage))

        def neg_prove(name: str, mutate=None, cheat=None, stmt=None, ypub=None):
            b2 = bt if stmt is None else copy.copy(bt)
            if stmt is not None:
                b2.stmt, b2.stmt_digest = stmt, stmt.digest()
            r2 = pv.prove(b2, make_coins(kind, 1), mutate=mutate, cheat=cheat, ypub_override=ypub)
            neg(name, b2.stmt, r2.proof)
        # (1) wrong claimed word (the prover claims it: witness honest, y16 row + statement changed)
        y0 = bt.stmt.y[0]
        for bit in (0, 7, rel.y_bits - 1):
            yb = y0.copy()
            yb[steps - 1] ^= 1 << bit
            neg_prove(f"wrong y: claimed word bit {bit} flipped (proved)", stmt=_stmt_with(bt.stmt, y=[yb] + bt.stmt.y[1:]), ypub=yb)
        yb = y0.copy()
        yb[steps - 1] = (int(yb[steps - 1]) + 1) % 2 ** rel.y_bits
        neg(f"wrong y: +1 ulp in the statement (honest proof)", _stmt_with(bt.stmt, y=[yb] + bt.stmt.y[1:]), res.proof)
        # (2) off-domain / changed operand in the statement
        a_bad = bt.stmt.a[0].copy()
        a_bad[3, 5] = rel.off_domain[0]
        neg(f"off-domain operand 0x{rel.off_domain[0]:x} in a", _stmt_with(bt.stmt, a=[a_bad] + bt.stmt.a[1:]), res.proof)
        neg("operand a[10][1] exponent bit flipped (honest proof)", _stmt_with(bt.stmt, a=[_flip_operand(rel, bt.stmt.a[0])] + bt.stmt.a[1:]),
            res.proof)
        # (2b) red-team V1: one forgery per virtual-row class, each must fail at its stage
        for name, kw, stage in v1_forgeries(pv, bt):
            r2 = pv.prove(bt, make_coins(kind, 1), **kw)
            neg(f"V1 {name}", kw.get("forge_stmt", bt.stmt), r2.proof, stage=stage)
        # (3) chain: broken link, nonzero c_0, accumulator bits (relchain.negatives' mutations)
        c_rows = sys_.chain["c"]

        def break_link(W):
            W[c_rows[2], 20] = (W[c_rows[2], 20] + 1) % P

        def nonzero_c0(W):
            W[c_rows[1], steps] = 1

        def acc_bit(W):
            W[c_rows[2], 30] = (W[c_rows[2], 30] ^ (1 << 12)) % P

        def acc_low_bit(W):
            W[c_rows[2], 31] = (W[c_rows[2], 31] ^ 1) % P
        neg_prove("violated chain link c.f[20] != y.f[19]", mutate=break_link)
        neg_prove("c_0 != 0 at a chain start", mutate=nonzero_c0)
        neg_prove("accumulator fraction bit 12 flipped in a linked unit", mutate=acc_bit)
        neg_prove("accumulator fraction bit 0 flipped in a linked unit", mutate=acc_low_bit)
        # (4) witness rows: one of every row class first (prod = quadratic, sel = lookup one-hot selector, bit, hint, inv),
        # then one per (gadget scope, row class)
        pins = {idx for idx, _ in sys_.pins}
        seen, picks, kinds = set(), [], set()
        for r in sys_.rows:
            if r.idx not in pins and r.kind not in kinds:
                kinds.add(r.kind)
                seen.add((".".join(r.name.split(".")[:2]), r.kind))
                picks.append(r)
        for r in sys_.rows:
            key = (".".join(r.name.split(".")[:2]), r.kind)
            if r.idx not in pins and key not in seen:
                seen.add(key)
                picks.append(r)
        rng = np.random.default_rng(11)
        row_muts = []
        for r in picks[: max(len(kinds), args.row_negatives)]:
            col = int(rng.integers(0, 2 * steps))
            delta = int(rng.integers(1, 3))

            def mutate(W, idx=r.idx, col=col, delta=delta):
                W[idx, col] = (W[idx, col] + delta) % P
            name = f"row {r.name} ({r.kind}{', quadratic' if r.kind == 'prod' else ', lookup' if r.kind == 'sel' else ''}) +{delta} in column {col}"
            neg_prove(name, mutate=mutate)
            row_muts.append((r.kind, name, mutate))
        # (5) dishonest zero-check provers on a violated witness: the forged messages pass the round checks, the final
        # claim (bound to the committed w through the PCS) catches it
        neg_prove("violated chain link + zero-check opening round forged (cheat=round1)", mutate=break_link, cheat="round1")
        neg_prove("violated chain link + every zero-check round forged (cheat=all)", mutate=break_link, cheat="all")
        for kind_ in ("prod", "sel"):
            hit = [(n_, m_) for k_, n_, m_ in row_muts if k_ == kind_]
            if hit:
                neg_prove(f"{hit[0][0]} + every zero-check round forged (cheat=all)", mutate=hit[0][1], cheat="all")
        # (6) tampered proof bytes
        for what in PROOF_TAMPERS:
            neg(f"tampered {what}", bt.stmt, _mutate_proof(res.proof, what))
        neg("statement of another batch size (n_vus)", _stmt_with(bt.stmt, n_vus=[bt.stmt.n_vus[0] - 1] + bt.stmt.n_vus[1:]), res.proof)
        for i, (name, (okn, whyn, _), st, pf, stage) in enumerate(negs):
            early = not okn and stage is not None and stage not in whyn
            log(f"  [{kind}] negative {name}: {'ACCEPTED (BUG)' if okn else f'REJECTED BEFORE {stage!r} (BUG)' if early else 'REJECTED'} {whyn}")
            report["negatives"].append({"coins": kind, "name": name, "accepted": okn, "why": whyn, "stage": stage, "early": early})
            failures += okn or early
            if args.dump_all_negatives or i < 64:
                try:
                    dump.add(f"neg_{kind}_{i:02d}", st, pf, "reject", (okn, whyn), kind, {"name": name})
                except ValueError as e:        # a statement the writer refuses (e.g. a word wider than its slot)
                    log(f"    (not dumped: {e})")
    report["failures"] = failures
    dump.close({"relation": rel.name, "l": args.l, "vus": args.vus, "commit": _git_commit(), "kind": "gate"})
    if args.dump_dir:
        report["python_dump_verify"] = verify_dir(Path(args.dump_dir), args.device)
        report["rust"] = rust_batch(args.rust_verifier, Path(args.dump_dir))
    log(f"{rel.name} gate: {len(report['positives'])} positives, {len(report['negatives'])} negatives, {failures} failures")
    if args.out:
        Path(os.path.expandvars(args.out)).write_text(json.dumps(report, indent=1, default=str))
    return 2 if failures else 0


# ---------------------------------------------------------------------------------------------------------------------
# Python file verifier over a dump
# ---------------------------------------------------------------------------------------------------------------------

_PROVERS: dict = {}


def verify_dir(d: Path, device: str = "cuda") -> dict:
    man = json.loads((d / "manifest.json").read_text())
    key = read_key((d / "key.bin").read_bytes())
    k = (key.relation, man.get("l"), device)
    if k not in _PROVERS:
        _PROVERS[k] = Prover(key.relation, device, man["l"])
    pv = _PROVERS[k]
    if pv.key_bytes != (d / "key.bin").read_bytes():
        raise SystemExit("key.bin differs from the key this code derives from the system")
    items, t0 = [], time.perf_counter()
    for e in man["files"]:
        st = read_statement((d / e["stmt"]).read_bytes())
        try:
            pf = read_proof((d / e["proof"]).read_bytes())
            vc = make_coins(e["coins"], e.get("coins_seed", 1)) if e["coins"] != "live" else _file_coins(d, e)
        except (ValueError, KeyError) as ex:
            ok, why, info = False, f"unreadable: {ex}", {"seconds": 0.0}
        else:
            ok, why, info = pv.verify(st, pf, vc)
        good = ok == (e["expect"] == "accept")
        items.append({"proof": e["proof"], "expect": e["expect"], "accepted": ok, "reason": why, "seconds": info["seconds"], "agrees": good,
                      "coin_provenance": COIN_PROVENANCE.get(e["coins"], e["coins"]), "soundness": info.get("soundness")})
        if not good:
            print(f"!! {e['proof']}: expected {e['expect']}, python says {ok} ({why})", flush=True)
    kinds = sorted({e["coins"] for e in man["files"]})
    summ = {"verifier": "python (backends.direct.ligerito.prove.Prover.verify; shares the claim algebra with the prover)",
            "independent": False, "total": len(items), "agree": sum(i["agrees"] for i in items), "wall": time.perf_counter() - t0,
            "coin_provenance": {k: COIN_PROVENANCE.get(k, k) for k in kinds}}
    for k in kinds:
        print(f"  coins {k}: {COIN_PROVENANCE.get(k, k)}", flush=True)
    (d / "verify_python.json").write_text(json.dumps({**summ, "items": items}, indent=1))
    print(f"python verify-dir {d}: {summ['agree']}/{summ['total']} verdicts as expected in {summ['wall']:.2f}s", flush=True)
    return summ


def _file_coins(d: Path, e: dict):
    raw = (d / e["coins_file"]).read_bytes()
    n = int.from_bytes(raw[:4], "little")
    slots = [raw[4 + 32 * i:4 + 32 * (i + 1)] for i in range(n)]
    b = e.get("stream_binding")
    if b is not None:
        # the coins are the stream's: every slot re-derived from the verifier's commitments and openings
        stmt, ccom = bytes.fromhex(b["statement"]), [bytes.fromhex(c) for c in b["commitments"]]
        ctx = live_mod.batch_context(stmt, ccom)
        for k, ((r, s_), label) in enumerate(zip(b["openings"], b["labels"])):
            r, s_ = bytes.fromhex(r), bytes.fromhex(s_)
            if live_mod.pm.coin_commit(r, s_) != ccom[k] or live_mod.expand_challenge(ctx, k, label.encode(), r, 32) != slots[k]:
                raise ValueError(f"coin slot {k} is not the stream's")
        if len(b["openings"]) != n:
            raise ValueError("coin count differs from the stream's openings")
    return LiveCoins(slots)


# ---------------------------------------------------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------------------------------------------------

def _buckets(t: dict) -> dict:
    """proto / prove.py timer buckets -> the contract's t.* buckets (seconds)."""
    g = lambda *ks: sum(t.get(k, 0.0) for k in ks)  # noqa: E731
    return {"witness": g("witness"),
            "encoding_commitment": g("commit_r1_alloc", "commit_r1_encode", "commit_r1_merkle", "recursion_encode", "recursion_merkle"),
            "arithmetic": g("zero_check", "sumcheck_r1_contract", "sumcheck_host", "sumcheck_r1_fold", "recursion_contract", "recursion_fold"),
            "lookup": 0.0, "zk_additional": g("zk_masks"),
            "serialization": g("serialize_openings", "serialize_final", "serialize_proof", "transcript")}


def bench(args) -> int:
    from verity_numerical.bench import contract
    pv = Prover(args.relation, args.device, args.l)
    rel = pv.rel
    total_vus = args.total_vus
    t0 = time.perf_counter()
    data = instances(rel, total_vus, procs=args.instance_procs, cache=args.instances_cache)
    inst_digest = instances_digest(rel, total_vus)
    print(f"{total_vus} VUs in {time.perf_counter() - t0:.1f}s; system {pv.sys.summary()}", flush=True)
    bv = args.batch_vus or total_vus
    if total_vus % bv:
        raise SystemExit(f"--batch-vus {bv} must divide --total-vus {total_vus}")
    bts = [pv.batch(data[i:i + bv], args.coins, zk=args.zk) for i in range(0, total_vus, bv)]
    nb_ = len(bts)
    bt = bts[0]
    lay = bt.lay
    print(f"{nb_} batch(es) of {bv} VUs; layout {lay.summary()}; dims {bt.dims.describe()}", flush=True)
    dump = Dump(Path(args.dump_dir) if args.dump_dir else None, pv)
    live = args.coins == "live"
    stream_info: dict = {}
    if live:
        stmts = [prove_mod.statement_for_stream(b, pv.key_sha256) for b in bts]
        # the session's round count: one local-stream proof (also the warm-up of the compiled kernels)
        warm = prove_mod.StreamCoins(live_mod.LocalChallenges(4096), stmts[0])
        pv.prove(bt, warm)
        R = warm.rounds
        sess = None
        if args.verifier:
            host, port = live_mod.parse_url(args.verifier)
            sess = live_mod.ChallengeSession(host, port, {"run_id": os.environ.get("RESEARCH_RUN_ID"), "lane": "ligerito-relation",
                                                           "prover_host": platform.node(), "prover_commit": _git_commit(),
                                                           "note": f"ligerito {rel.name} {total_vus} VUs"},
                                             args.reps * nb_, R, args.window)
        stream_info = {"verifier": args.verifier or "local (live.LocalChallenges: same stream code, no network)",
                       "rounds_per_batch": R, "window": args.window, "batches": args.reps * nb_}
        print(f"live coins: {R} rounds per batch, window {args.window}, verifier {stream_info['verifier']}", flush=True)

    def one(rep: int, j: int):
        b = bts[j]
        if live:
            st = sess.stream(rep * nb_ + j) if sess is not None else live_mod.LocalChallenges(R, rep * nb_ + j)
            pc = prove_mod.StreamCoins(st, stmts[j])
            r = pv.prove(b, pc)
            return r, pc
        pc = make_coins(args.coins, 1000 + rep * nb_ + j)
        return pv.prove(b, pc), None

    runs = []          # per timed rep: (wall, [(Result, verify info, stream coins)])
    res = None
    for rep in range(args.reps + (0 if live else 1)):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if live and args.window > 1:
            with ThreadPoolExecutor(max_workers=args.window) as ex:
                outs = list(ex.map(lambda j: one(rep, j), range(nb_)))
        else:
            outs = [one(rep, j) for j in range(nb_)]
        wall = time.perf_counter() - t0
        items = []
        for j, (r, pc) in enumerate(outs):
            vc = LiveCoins(pc.slots_out) if live else make_coins(args.coins, 1000 + rep * nb_ + j)
            ok, why, info = pv.verify(bts[j].stmt, r.proof, vc)
            assert ok, why
            items.append((r, info, pc))
        res = items[0][0]
        warmup = rep == 0 and not live
        tb = {k: round(sum(_buckets(r.timings)[k] for r, _, _ in items), 3) for k in _buckets({})}
        lat = [pc.stream.latency for _, _, pc in items if pc is not None and pc.stream.latency is not None]
        print(f"rep {rep}{' (warm-up)' if warmup else ''}: wall {wall:.3f}s over {nb_} batch(es) (sum of prover totals "
              f"{sum(r.timings['total'] for r, _, _ in items):.3f}s) {tb} verifier {sum(i['seconds'] for _, i, _ in items):.3f}s; "
              f"{sum(r.proof.n_bytes()['total'] for r, _, _ in items)} B; peak {max(r.peak_device_bytes for r, _, _ in items) / 2**30:.1f} GiB"
              + (f"; batch latency {max(lat):.3f}s, stream wait {sum(pc.stream.wait for _, _, pc in items):.3f}s" if lat else ""),
              flush=True)
        if warmup:
            continue
        runs.append((wall, items))
        if len(runs) == 1:
            for j, (r, info, pc) in enumerate(items):
                extra = {"prover_seconds": r.timings["total"], "vus": [j * bv, (j + 1) * bv]}
                if pc is not None:
                    extra.update(dump.coins_file(f"batch_{args.coins}_{j}", pc.slots_out))
                    extra["stream_binding"] = pc.binding()
                dump.add(f"batch_{args.coins}_{j}", bts[j].stmt, r.proof, "accept", (True, "ok"), args.coins, extra,
                         seed=1000 + len(runs) * nb_ + j)
    if live and sess is not None:
        stream_info["verdict"] = sess.end({"wall_s": sum(w for w, _ in runs)})
        stream_info["round_ms_median"] = statistics.median(sess.rtts) * 1e3 if sess.rtts else None
        stream_info["bytes_out"], stream_info["bytes_in"] = sess.bytes_out, sess.bytes_in
        print(f"live verdict: {stream_info['verdict']}", flush=True)
    dump.close({"relation": rel.name, "l": args.l, "vus": total_vus, "commit": _git_commit(), "kind": "bench"})
    dumps = verify_dir(Path(args.dump_dir), args.device) if args.dump_dir else None
    rust = rust_batch(args.rust_verifier, Path(args.dump_dir)) if args.dump_dir else None
    med_total = statistics.median(w for w, _ in runs)
    def rep_buckets(it):
        """t.* of one rep: sequential batches sum; windowed batches overlap, so the per-batch mean (the wall is the
        joint).  Live: the challenge-stream waits stay inside the zero-check / PCS timers they happen in (reported
        separately as live.stream_wait_seconds)."""
        bs = [_buckets(r.timings) for r, _, _ in it]
        agg = (lambda xs: sum(xs) / len(xs)) if live and args.window > 1 else sum
        return {k: agg([b[k] for b in bs]) for k in _buckets({})}
    med_b = {k: statistics.median(rep_buckets(it)[k] for _, it in runs) for k in _buckets({})}
    med_wait = statistics.median(max(pc.stream.wait for _, _, pc in it) for _, it in runs) if live else 0.0
    med_lat = statistics.median(max(pc.stream.latency or 0.0 for _, _, pc in it) for _, it in runs) if live else None
    vmed = statistics.median(sum(i["seconds"] for _, i, _ in it) for _, it in runs)
    nbytes = {k: sum(r.proof.n_bytes()[k] for r, _, _ in runs[0][1]) for k in res.proof.n_bytes()}
    snd = pv.soundness(bt.dims)
    peak = max(r.peak_device_bytes for _, it in runs for r, _, _ in it)
    rust_ok = bool(rust and rust["exit"] == 0)
    live_verified = bool(live and args.verifier and (stream_info.get("verdict") or {}).get("accepted"))
    claim = prove_mod.soundness_claim(snd, args.coins, verifier_record=live_verified)
    claimed = claim["claimed_log2"]
    zk_on = False                                   # complete ZK: needs the sumcheck masks and the PCS column padding
    zk_partial = f"partial: {ZK_PARTIAL}; missing: {ZK_MISSING}" if args.zk else None
    if zk_on and live_verified and rust_ok:
        proof_class, zk_mode = "COMPLETE_ZK_BACKEND", "ligerito-zk (masks + per-column RS padding + Libra sumcheck masks), live coins"
        omitted = "external authentication"
    else:
        proof_class, zk_mode = "NON_ZK_PROOF_DIAGNOSTIC", "none"
        omitted = "external authentication; ZK masking (" + (ZK_MISSING if args.zk else STUBBED) + ")"
        if live and not live_verified:
            omitted += "; coins from live.LocalChallenges (no network verifier)" if not args.verifier else "; live session not accepted"
    if claimed is None:
        omitted += f"; soundness ({claim['basis']})"
    hw = contract.probe_hardware()
    sw = {"backend": {"name": BACKEND, "commit": _git_commit(), "hash": "blake3", "challenge_expander": "SHAKE-256",
                      "pcs": "Ligerito (ligerito-pcs-fast proto/pcs.commit + open, multi-claim)" if pv.pcs_impl == "pcs-fast"
                             else "Ligerito (proto/pcs.py pieces; multi-claim open in prove.py)", "coins": args.coins, "live": stream_info or None},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    inst = ({"dataset": rel.instances_dataset, "tier": rel.instances_tier, "range": [0, total_vus], "manifest_sha256": inst_digest,
             "seed": rel.instance_seed} if not rel.frozen_tier else contract.instances_ref(rel.instances_tier, 0, total_vus, inst_digest,
                                                                                            dataset=rel.instances_dataset))
    if not rel.frozen_tier and rel.instances_recipe:
        inst["recipe"] = rel.instances_recipe
    d = bt.dims
    fp = contract.fingerprint(
        profile=rel.target.name, domain=contract.DOMAIN, K=K_VU, B=total_vus, N_subbatches=nb_,
        proof_class=proof_class, authentication="excluded", zk_mode=zk_mode,
        security={"target": -128 if claimed is not None and claimed <= -128 else None,
                  "parameter_target": -128, "field": "BabyBear", "challenge_field": "F_{p^6} = F_p[x]/(x^6 - 31)", "pcs": "ligerito",
                  "parameter_set": "B (pow2: level-1 rate 1/2, later 1/4)", "n": d.n, "kprime": d.kprime, "queries": d.queries,
                  "rate_log2": d.rates, "final_len": 1 << d.k[-1], "coins": args.coins, "zk": zk_on, "zk_partial": zk_partial,
                  "bound_model": "interactive union: paper (17) per level with the sampling base ((1+rho)/2)^|S| + field terms, the "
                                 "three sumchecks, the batching coins and the virtual-row zero claims over F_{p^6} "
                                 "(params.LigeritoParams.soundness); claimed per coin mode by prove.soundness_claim (red-team F12)",
                  "hash_assumption": "BLAKE3 Merkle (collision resistance); Fiat-Shamir: BLAKE2b/SHAKE-256 transcript" if args.coins == "fiat-shamir"
                                     else "BLAKE3 Merkle (collision resistance); coins independent of the prover's messages",
                  "achieved_log2": claimed, "achieved_bits": None if claimed is None else -claimed, "soundness_basis": claim["basis"],
                  "interactive_union_log2": snd["total_log2"], "terms_log2": snd["terms_log2"],
                  **({"fs_queries_log2": claim["queries_log2"], "fs_worst_term": claim["worst_term"]} if "worst_term" in claim else {}),
                  "union_over": 1, "stubbed": STUBBED},
        hardware=hw, software=sw,
        preprocessing=f"constraint compilation + layout constraints + verification key (functions of the relation only, {pv.compile_seconds:.2f}s)",
        retention="retained", instances=inst, scope="vu", omitted=omitted,
        relation=rel.relation_name, model=rel.model_name, commit=_git_commit(), step_k=rel.k, steps_per_vu=rel.steps,
        rows=pv.sys.m, committed_rows=lay.m, layout=lay.summary(), witness_generation=rel.witness_generation,
        statement_format="LGTS0001", proof_format="LGTO0001", key_format="LGVK0001", key_sha256=pv.key_sha256.hex(),
        note=f"{total_vus} VUs as {nb_} Ligerito batch(es) of {lay.n_sub} sub-batches x l = {args.l} columns (S = {lay.S}), N = 2^{lay.n_vars} "
             f"cells each; t.total = wall over all batches{f' ({args.window} in flight)' if live and args.window > 1 else ''}; proof bytes / verify summed; "
             f"{'live coins: the t.* buckets include the challenge-stream waits (live.stream_wait_seconds, R x RTT per batch); ' if live else ''}"
             f"cells; prover time includes witness generation (Ligero's hints + build_witness, marshal cached across reps as in relchain); "
             f"{rel.claim_note}")
    meas = [contract.measurement(f"t.{k}", v) for k, v in med_b.items()]
    meas += [
        contract.measurement("t.total", med_total),
        contract.measurement("verifier.seconds", vmed),
        contract.measurement("transcript.bytes", nbytes["total"]),
        contract.measurement("rounds.protocol", res.coins_rounds),
        contract.measurement("rounds.sequential_depth", res.coins_rounds),
        contract.measurement("mem.peak_device_bytes", peak),
        contract.measurement("census.witness_elements", lay.m * lay.C),
        contract.measurement("census.committed_elements", 1 << lay.n_vars),
        contract.measurement("census.bytes_encoded", 4 * (1 << lay.n_vars) << d.rates[0]),
        contract.measurement("vu.seconds_per_vu", med_total / total_vus, "s"),
        contract.measurement("vu.count", total_vus, "count"),
        *contract.rate_measurements(med_total, total_vus, K_VU, rel.target.name),
        contract.measurement("proof_bytes", nbytes["total"], "B"),
        contract.measurement("proof.sumcheck_bytes", nbytes["sumcheck"], "B"),
        contract.measurement("proof.pcs_rows_bytes", nbytes["pcs_rows"], "B"),
        contract.measurement("proof.pcs_merkle_bytes", nbytes["pcs_merkle"], "B"),
        contract.measurement("proof.pcs_sumcheck_bytes", nbytes["pcs_sumcheck"], "B"),
        contract.measurement("proof.pcs_final_bytes", nbytes["pcs_final"], "B"),
        contract.measurement("prove_wall", med_total, "s"),
        contract.measurement("verify_wall", vmed, "s"),
        *([contract.measurement("soundness_bits", -claimed, "bits")] if claimed is not None else []),
        contract.measurement("split.zero_check_seconds", statistics.median(sum(r.timings.get("zero_check", 0.0) for r, _, _ in it) for _, it in runs), "s"),
        contract.measurement("split.witness_seconds", med_b["witness"], "s"),
    ]
    if live:
        meas += [contract.measurement("live.stream_wait_seconds", med_wait, "s"),
                 contract.measurement("live.batch_latency_seconds", med_lat, "s"),
                 contract.measurement("live.rounds_per_batch", stream_info["rounds_per_batch"], "count"),
                 contract.measurement("live.window", args.window, "count")]
        if stream_info.get("round_ms_median") is not None:
            meas.append(contract.measurement("live.round_trip_ms_median", stream_info["round_ms_median"], "ms"))
    if rust and "json" in rust:
        secs = [it.get("seconds") for it in rust["json"].get("items", []) if isinstance(it, dict) and it.get("seconds") is not None]
        if secs:
            meas.append(contract.measurement("verify_rust_seconds", statistics.median(secs), "s"))
    status = "passed" if (dumps is None or dumps["agree"] == dumps["total"]) else "failed"
    checker = (f"backends.direct.ligerito.run bench[{rel.name}] (the in-process Python verifier accepted every rep; shares the claim "
               "algebra with the prover" + ("; Python file verifier cold from the dumped bytes" if dumps else "")
               + (f"; ligerito-verify batch --dir exit {rust['exit']}" if rust else "") + ")")
    detail = f"{args.reps} reps at {total_vus} VUs ({nb_} batch(es)), coins {args.coins}; proof {nbytes['total']} B"
    result = {"schema": contract.RESULT_SCHEMA,
              "run_id": args.run_id or os.environ.get("RESEARCH_RUN_ID") or f"ligerito-{rel.name}-{args.coins}-{int(time.time())}",
              "workload_fingerprint": fp,
              "validation": {"status": status, "checker": checker, "detail": detail,
                             "evidence": {"soundness": snd, "soundness_claim": claim, "dumps": dumps, "rust": rust,
                                          "live": stream_info or None,
                                          "reps": [{"wall": w, "batches": [{"prover": r.timings, "verifier": i,
                                                                            "stream_latency": pc.stream.latency if pc else None,
                                                                            "stream_wait": pc.stream.wait if pc else None}
                                                                           for r, i, pc in it]} for w, it in runs]}},
              "measurements": meas, "measurement_files": [], "artifacts": []}
    problems = contract.validate(result)
    if problems:
        print("contract problems:", problems, flush=True)
    print(json.dumps({"relation": rel.name, "t.total": med_total, "buckets": med_b, "verifier": vmed, "bytes": nbytes,
                      "coins": res.coins_rounds, "peak_device_bytes": peak, "soundness_log2": claimed, "soundness_basis": claim["basis"],
                      "interactive_union_log2": snd["total_log2"], "proof_class": proof_class,
                      "rust_exit": rust["exit"] if rust else None, "validation": status}, indent=1, default=str), flush=True)
    if args.out:
        Path(os.path.expandvars(args.out)).write_text(json.dumps(result, indent=1, default=str))
    return 0 if status == "passed" and not problems else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backends.direct.ligerito.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gate", "bench"):
        s = sub.add_parser(name)
        s.add_argument("--relation", default="fp8-ada")
        s.add_argument("--device", default="cuda")
        s.add_argument("--l", type=int, default=None if name == "gate" else 16384)
        s.add_argument("--dump-dir")
        s.add_argument("--out")
        s.add_argument("--rust-verifier")
        s.add_argument("--zk", action="store_true", help="ligerito-zk's row masks + index map (partial ZK, see the fingerprint)")
        if name == "gate":
            s.add_argument("--vus", type=int, default=12)
            s.add_argument("--coins", nargs="+", default=["local", "fiat-shamir"])
            s.add_argument("--row-negatives", type=int, default=16)
            s.add_argument("--dump-all-negatives", action="store_true")
        else:
            s.add_argument("--total-vus", type=int, default=4096)
            s.add_argument("--reps", type=int, default=3)
            s.add_argument("--coins", default="local", choices=["local", "fiat-shamir", "live"])
            s.add_argument("--instance-procs", type=int, default=8)
            s.add_argument("--instances-cache")
            s.add_argument("--run-id")
            s.add_argument("--batch-vus", type=int, default=None, help="split --total-vus into proofs of this many VUs")
            s.add_argument("--verifier", help="tcp://host:port of a live challenge-stream verifier (live-2's `live serve`)")
            s.add_argument("--window", type=int, default=1, help="live: batches in flight")
    s = sub.add_parser("verify-dir")
    s.add_argument("dir")
    s.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.cmd == "gate":
        if args.l is None:
            args.l = 256 if args.vus <= 64 else 16384
        return gate(args)
    if args.cmd == "bench":
        return bench(args)
    summ = verify_dir(Path(args.dir), args.device)
    return 0 if summ["agree"] == summ["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
