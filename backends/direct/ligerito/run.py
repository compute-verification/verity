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
from pathlib import Path

import numpy as np
import torch

from backends.direct.ligero.compile import P
from backends.direct.ligero.relchain import K_VU, census, instances, instances_digest

from .proof import Proof, Statement, read_key, read_proof, read_statement, sha256
from .prove import Prover, Result, make_coins

BACKEND = "ligerito"
STUBBED = "zk.py not wired (no masks, no salted leaves, no Libra sumcheck masks); coins not live"


def _git_commit() -> str:
    if os.environ.get("RESEARCH_GIT_COMMIT"):
        return os.environ["RESEARCH_GIT_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


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

    def add(self, stem: str, stmt: Statement, proof: Proof, expect: str, verdict: tuple, coins: str, extra=None):
        if self.root is None:
            return
        sb, pb = stmt.to_bytes(), proof.to_bytes()
        (self.root / f"{stem}.stmt").write_bytes(sb)
        name = f"{stem}.lgto" if expect == "accept" else f"{stem}.lgto.neg"
        (self.root / name).write_bytes(pb)
        e = {"proof": name, "stmt": f"{stem}.stmt", "expect": expect, "coins": coins, "proof_bytes": len(pb),
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
        bt = pv.batch(data, kind)
        lay = bt.lay
        log(f"[{kind}] layout {lay.summary()}; dims {bt.dims.describe()}")
        pc, vc = _coins_pair(kind, 1)
        res = pv.prove(bt, pc)
        ok, why, info = pv.verify(bt.stmt, res.proof, vc)
        nb = res.proof.n_bytes()
        log(f"[{kind}] honest: prover {res.timings['total']:.2f}s, {nb['total']} B, {res.coins_rounds} coins -> {ok} {why} "
            f"(verify {info['seconds']:.3f}s)")
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

        def neg(name: str, stmt: Statement, proof: Proof, seed: int = 1):
            v = pv.verify(stmt, proof, make_coins(kind, seed))
            negs.append((name, v, stmt, proof))

        def neg_prove(name: str, mutate=None, cheat=None, stmt=None, ypub=None):
            b2 = bt if stmt is None else copy.copy(bt)
            if stmt is not None:
                b2.stmt = stmt
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
        a_bad = bt.stmt.a[0].copy()
        a_bad[10, 1] ^= 1 << (rel.params.frac if rel.params.word_bits == 16 else 3)
        neg("operand a[10][1] exponent bit flipped (honest proof)", _stmt_with(bt.stmt, a=[a_bad] + bt.stmt.a[1:]), res.proof)
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
        for i, (name, (okn, whyn, _), st, pf) in enumerate(negs):
            log(f"  [{kind}] negative {name}: {'REJECTED' if not okn else 'ACCEPTED (BUG)'} {whyn}")
            report["negatives"].append({"coins": kind, "name": name, "accepted": okn, "why": whyn})
            failures += okn
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
        except (ValueError, KeyError) as ex:
            ok, why, info = False, f"unreadable: {ex}", {"seconds": 0.0}
        else:
            ok, why, info = pv.verify(st, pf, make_coins(e["coins"], 1) if e["coins"] != "live" else _file_coins(d, e))
        good = ok == (e["expect"] == "accept")
        items.append({"proof": e["proof"], "expect": e["expect"], "accepted": ok, "reason": why, "seconds": info["seconds"], "agrees": good})
        if not good:
            print(f"!! {e['proof']}: expected {e['expect']}, python says {ok} ({why})", flush=True)
    summ = {"verifier": "python (backends.direct.ligerito.prove.Prover.verify; shares the claim algebra with the prover)",
            "independent": False, "total": len(items), "agree": sum(i["agrees"] for i in items), "wall": time.perf_counter() - t0}
    (d / "verify_python.json").write_text(json.dumps({**summ, "items": items}, indent=1))
    print(f"python verify-dir {d}: {summ['agree']}/{summ['total']} verdicts as expected in {summ['wall']:.2f}s", flush=True)
    return summ


def _file_coins(d: Path, e: dict):
    from .transcript import LiveCoins
    raw = (d / e["coins_file"]).read_bytes()
    n = int.from_bytes(raw[:4], "little")
    return LiveCoins([raw[4 + 32 * i:4 + 32 * (i + 1)] for i in range(n)])


# ---------------------------------------------------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------------------------------------------------

def _buckets(t: dict) -> dict:
    """proto / prove.py timer buckets -> the contract's t.* buckets (seconds)."""
    g = lambda *ks: sum(t.get(k, 0.0) for k in ks)  # noqa: E731
    return {"witness": g("witness"),
            "encoding_commitment": g("commit_r1_alloc", "commit_r1_encode", "commit_r1_merkle", "recursion_encode", "recursion_merkle"),
            "arithmetic": g("zero_check", "sumcheck_r1_contract", "sumcheck_host", "sumcheck_r1_fold", "recursion_contract", "recursion_fold"),
            "lookup": 0.0, "zk_additional": 0.0,
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
    bt = pv.batch(data, args.coins)
    lay = bt.lay
    print(f"layout {lay.summary()}; dims {bt.dims.describe()}", flush=True)
    dump = Dump(Path(args.dump_dir) if args.dump_dir else None, pv)
    runs = []
    res = None
    for rep in range(args.reps + 1):
        pc, vc = _coins_pair(args.coins, 1000 + rep)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        res = pv.prove(bt, pc)
        ok, why, info = pv.verify(bt.stmt, res.proof, vc)
        assert ok, why
        b = _buckets(res.timings)
        print(f"rep {rep}{' (warm-up)' if rep == 0 else ''}: prover {res.timings['total']:.3f}s "
              f"{ {k: round(v, 3) for k, v in b.items()} } verifier {info['seconds']:.3f}s; {res.proof.n_bytes()['total']} B; "
              f"peak {res.peak_device_bytes / 2**30:.1f} GiB", flush=True)
        if rep == 0:
            continue
        runs.append((res, info))
        if rep == 1:
            dump.add(f"batch_{args.coins}_rep1", bt.stmt, res.proof, "accept", (ok, why), args.coins,
                     {"prover_seconds": res.timings["total"], "vus": [0, total_vus]})
    dump.close({"relation": rel.name, "l": args.l, "vus": total_vus, "commit": _git_commit(), "kind": "bench"})
    dumps = verify_dir(Path(args.dump_dir), args.device) if args.dump_dir else None
    rust = rust_batch(args.rust_verifier, Path(args.dump_dir)) if args.dump_dir else None
    med_total = statistics.median(r.timings["total"] for r, _ in runs)
    med_b = {k: statistics.median(_buckets(r.timings)[k] for r, _ in runs) for k in _buckets({})}
    vmed = statistics.median(i["seconds"] for _, i in runs)
    nbytes = res.proof.n_bytes()
    snd = pv.soundness(bt.dims)
    peak = max(r.peak_device_bytes for r, _ in runs)
    rust_ok = bool(rust and rust["exit"] == 0)
    zk_on, live = False, args.coins == "live"
    if zk_on and live and rust_ok:
        proof_class, zk_mode = "COMPLETE_ZK_BACKEND", "ligerito-zk (masks + per-column RS padding + Libra sumcheck masks), live coins"
        omitted = "external authentication"
    else:
        proof_class, zk_mode = "NON_ZK_PROOF_DIAGNOSTIC", "none"
        omitted = "external authentication; ZK masking (" + STUBBED + ")"
    hw = contract.probe_hardware()
    sw = {"backend": {"name": BACKEND, "commit": _git_commit(), "hash": "blake3", "challenge_expander": "SHAKE-256",
                      "pcs": "Ligerito (proto/pcs.py pieces; multi-claim open in prove.py)", "coins": args.coins},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    inst = ({"dataset": rel.instances_dataset, "tier": rel.instances_tier, "range": [0, total_vus], "manifest_sha256": inst_digest,
             "seed": rel.instance_seed} if not rel.frozen_tier else contract.instances_ref(rel.instances_tier, 0, total_vus, inst_digest,
                                                                                            dataset=rel.instances_dataset))
    if not rel.frozen_tier and rel.instances_recipe:
        inst["recipe"] = rel.instances_recipe
    d = bt.dims
    fp = contract.fingerprint(
        profile=rel.target.name, domain=contract.DOMAIN, K=K_VU, B=total_vus, N_subbatches=1,
        proof_class=proof_class, authentication="excluded", zk_mode=zk_mode,
        security={"target": -128, "field": "BabyBear", "challenge_field": "F_{p^6} = F_p[x]/(x^6 - 31)", "pcs": "ligerito",
                  "parameter_set": "B (pow2: level-1 rate 1/2, later 1/4)", "n": d.n, "kprime": d.kprime, "queries": d.queries,
                  "rate_log2": d.rates, "final_len": 1 << d.k[-1], "coins": args.coins, "zk": zk_on,
                  "bound_model": "union bound: paper (17) per level with the sampling base ((1+rho)/2)^|S| + field terms, the three "
                                 "sumchecks and the batching coins over F_{p^6} (params.LigeritoParams.soundness)",
                  "hash_assumption": "BLAKE3 Merkle (collision resistance); Fiat-Shamir: BLAKE2b/SHAKE-256 transcript" if args.coins == "fiat-shamir"
                                     else "BLAKE3 Merkle (collision resistance); coins independent of the prover's messages",
                  "achieved_log2": snd["total_log2"], "achieved_bits": -snd["total_log2"], "terms_log2": snd["terms_log2"],
                  "union_over": 1, "stubbed": STUBBED},
        hardware=hw, software=sw,
        preprocessing=f"constraint compilation + layout constraints + verification key (functions of the relation only, {pv.compile_seconds:.2f}s)",
        retention="retained", instances=inst, scope="vu", omitted=omitted,
        relation=rel.relation_name, model=rel.model_name, commit=_git_commit(), step_k=rel.k, steps_per_vu=rel.steps,
        rows=pv.sys.m, committed_rows=lay.m, layout=lay.summary(), witness_generation=rel.witness_generation,
        statement_format="LGTS0001", proof_format="LGTO0001", key_format="LGVK0001", key_sha256=pv.key_sha256.hex(),
        note=f"{total_vus} VUs as ONE Ligerito batch of {lay.n_sub} sub-batches x l = {args.l} columns (S = {lay.S}), N = 2^{lay.n_vars} "
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
        contract.measurement("soundness_bits", -snd["total_log2"], "bits"),
        contract.measurement("split.zero_check_seconds", statistics.median(r.timings.get("zero_check", 0.0) for r, _ in runs), "s"),
        contract.measurement("split.witness_seconds", med_b["witness"], "s"),
    ]
    if rust and "json" in rust:
        secs = [it.get("seconds") for it in rust["json"].get("items", []) if isinstance(it, dict) and it.get("seconds") is not None]
        if secs:
            meas.append(contract.measurement("verify_rust_seconds", statistics.median(secs), "s"))
    status = "passed" if (dumps is None or dumps["agree"] == dumps["total"]) else "failed"
    checker = (f"backends.direct.ligerito.run bench[{rel.name}] (the in-process Python verifier accepted every rep; shares the claim "
               "algebra with the prover" + ("; Python file verifier cold from the dumped bytes" if dumps else "")
               + (f"; ligerito-verify batch --dir exit {rust['exit']}" if rust else "") + ")")
    detail = f"{args.reps} reps at {total_vus} VUs, coins {args.coins}; proof {nbytes['total']} B"
    result = {"schema": contract.RESULT_SCHEMA,
              "run_id": args.run_id or os.environ.get("RESEARCH_RUN_ID") or f"ligerito-{rel.name}-{args.coins}-{int(time.time())}",
              "workload_fingerprint": fp,
              "validation": {"status": status, "checker": checker, "detail": detail,
                             "evidence": {"soundness": snd, "dumps": dumps, "rust": rust,
                                          "reps": [{"prover": r.timings, "verifier": i} for r, i in runs]}},
              "measurements": meas, "measurement_files": [], "artifacts": []}
    problems = contract.validate(result)
    if problems:
        print("contract problems:", problems, flush=True)
    print(json.dumps({"relation": rel.name, "t.total": med_total, "buckets": med_b, "verifier": vmed, "bytes": nbytes,
                      "coins": res.coins_rounds, "peak_device_bytes": peak, "soundness_log2": snd["total_log2"],
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
