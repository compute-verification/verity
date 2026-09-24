"""Gate and bench for the Ligerito backend (lane ligerito-relation), mirroring ``backends/direct/ligero/relchain.py``.

    python -m backends.direct.ligerito.run gate  --relation fp8-ada --vus 12 --l 256 [--dump-dir D] [--out gate.json]
    python -m backends.direct.ligerito.run bench --relation fp8-ada --total-vus 4096 --reps 3 --coins local [--dump-dir D]
                                                 [--rust-verifier backends/ligerito-verify/target/release/ligerito-verify]
    python -m backends.direct.ligerito.run verify-dir D           # the Python file verifier on a dump (cold, from bytes)
    python -m backends.direct.ligerito.run verify-session D S...  # a live dump's coins vs the verifier's session dirs S

Dump layout (what ``ligerito-verify batch --dir`` reads): ``key.bin`` (LGVK0001), per proof ``<stem>.stmt`` (LGTS0001) +
``<stem>.lgto`` (LGTO0001, must ACCEPT) or ``<stem>.lgto.neg`` (must REJECT), ``<stem>.coins`` for live coins
(``u32 count | 32 B per coin``), ``manifest.json`` ({"files": [{"proof", "stmt", "coins", "expect", "python_verdict", ...}]}).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
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

from . import proof as proof_mod
from . import prove as prove_mod
from .layout import layout_for
from .proof import Proof, Statement, read_key, read_proof, read_statement, sha256
from .prove import Prover, Result, make_coins
from .transcript import LiveCoins

BACKEND = "ligerito"
STUBBED = "zk.py not wired (no masks, no salted leaves, no Libra sumcheck masks, no per-column RS padding)"
ZK_PARTIAL = ("LGSC0004 (ligerito-sumcheck-3): Libra masks on every monomial of the zero-check, combined and row-reduction rounds, "
              "committed next rows + mask link, the enforced product triple i1 i2 = i3; ligerito-zk's 8 uniform rows and per-column "
              "RS padding (t_pad) under the bit-interleaved index map z_to_f; the three mask claims as sparse PCS terms")
ZK_MISSING = ("a PCS zero-knowledge argument that covers the V1 zero claims (boolean high row bits: they never reach the uniform "
              "rows) and LGSC0004's sparse claims on the g rows (red-team-ligerito-3, LGSC0004 checklist item 5)")
#: where a cold verify-dir gets each coin mode's coins from (red-team F12 (b)): only the FS transcript is the verifier's own
COIN_PROVENANCE = {"fiat-shamir": "recomputed from the transcript (verifier-derived)",
                   "local": "rebuilt from the manifest's coins_seed (prover-side: no soundness)",
                   "live": "the dump's .coins file, stream binding re-derived (prover-side record: no soundness until verify-session "
                           "matches it to the verifier's own session record)"}


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


def _reencode_framing(pb: bytes) -> bytes:
    """The same proof with its (unabsorbed) framing JSON re-encoded: an explicit ``"t_pad":0`` without ZK padding (the
    red-team's ``framing_t_pad_zero``), the keys reordered with it."""
    r = proof_mod._R(pb)
    r.take(8)
    r.blob()
    at = r.pos
    fraw = r.blob()
    fr = json.loads(fraw)
    alt = json.dumps({"final_len": fr["final_len"], "sib_len": fr["sib_len"], "t_pad": fr.get("t_pad", 0)},
                     separators=(",", ":")).encode()
    return pb[:at] + len(alt).to_bytes(4, "little") + alt + pb[at + 4 + len(fraw):]


def _stmt_with(stmt: Statement, **kw) -> Statement:
    st = copy.deepcopy(stmt)
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _stmt_nvus(pv: Prover, stmt: Statement, n_vus: list[int]) -> Statement:
    """``stmt`` claiming other sub-batch sizes, kept canonical (y = 0 off the new chain ends, operand words 0 in the new pad
    units), so the verifier gets past the statement checks to the claim itself."""
    st = _stmt_with(stmt, n_vus=n_vus)
    lay = layout_for(pv.sys, pv.l, [nv for nv in n_vus if nv > 0], pv.steps)
    st.y = [np.where(lay.chain_layout(s).masks("cpu")[2].numpy() != 0, y, 0) for s, y in enumerate(st.y)]
    for s, nv in enumerate(n_vus[:len(st.a)]):
        st.a[s][nv * pv.steps:] = 0
        st.b[s][nv * pv.steps:] = 0
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

    if lay.zk is None:
        next0 = torch.zeros(lay.C, dtype=torch.int64)
        next0[5] = 12345
        shift = [("next:0 row +12345 in column 5 committed (pinned by the shift)", dict(forge_rows={lay.virt["next:0"]: next0}),
                  "combined final")]
    else:
        # LGSC0004 commits next:<x> and the mask link M_next; the shift pins both (red-team-ligerito-3's ZK must-rejects).
        # next:0 is edited where link = 0 (a VU's last unit), so the zero-check's link constraint cannot see it
        col = int(np.flatnonzero(lay.chain_layout(0).masks("cpu")[1].numpy() == 0)[0])

        def bump_row(row_of, c, d):
            def edit(z, lay_):
                r = row_of(lay_)
                z[r, c] = ((z[r, c].to(torch.int64) + d) % P).to(z.dtype)
            return edit
        shift = [(f"committed next:0 +12345 in column {col} (link = 0 there: only the shift pins it)",
                  dict(edit_zk=bump_row(lambda l_: l_.zk.next[0], col, 12345)), "combined"),
                 ("committed M_next +1 in column 5 (the mask link: only the shift pins it)",
                  dict(edit_zk=bump_row(lambda l_: l_.zk.mask_next, 5, 1)), "combined")]
    return [("end row(s): claimed word +1 ulp, true witness", dict(forge_stmt=_stmt_with(st, y=[bump(steps - 1)] + st.y[1:])), "PCS"),
            ("pub:* rows: operand a[10][1] exponent bit flipped, true witness",
             dict(forge_stmt=_stmt_with(st, a=[_flip_operand(rel, st.a[0])] + st.a[1:])), "PCS"),
            ("start/link/end rows: statement n_vus[0] - 1 (the dropped VU's end word zeroed), true witness",
             dict(forge_stmt=_stmt_nvus(pv, st, [nv0 - 1] + st.n_vus[1:])), "PCS"),
            ("const + every public row: last VU of sub-batch 0 all-zero in z, its y claimed +1",
             dict(forge_stmt=_stmt_with(st, y=[bump(nv0 * steps - 1)] + st.y[1:]), forge_z=zero_last_vu), "PCS")] + shift


# ---------------------------------------------------------------------------------------------------------------------
# Dumps
# ---------------------------------------------------------------------------------------------------------------------

class Dump:
    def __init__(self, root: Path | None):
        self.root = root
        self.files: list[dict] = []
        self.key: bytes | None = None
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    def set_key(self, key_bytes: bytes) -> None:
        """The batches' key (``Batch.key_bytes``): one per dump, since the reader pins one ``key.bin``."""
        if self.key is not None and self.key != key_bytes:
            raise ValueError("batches with different keys in one dump")
        self.key = key_bytes
        if self.root is not None:
            (self.root / "key.bin").write_bytes(key_bytes)

    def coins_file(self, stem: str, slots: list) -> dict:
        if self.root is None:
            return {}
        (self.root / f"{stem}.coins").write_bytes(len(slots).to_bytes(4, "little") + b"".join(slots))
        return {"coins_file": f"{stem}.coins"}

    def add(self, stem: str, stmt: Statement, proof: Proof | bytes, expect: str, verdict: tuple, coins: str, extra=None,
            seed: int = 1):
        if self.root is None:
            return
        sb, pb = stmt.to_bytes(), proof if isinstance(proof, bytes) else proof.to_bytes()
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
                                                                        "key": "LGVK0001"}, **meta,
                                                             "key_sha256": sha256(self.key).hex() if self.key else None,
                                                             "files": self.files}, indent=1, default=str))


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
    j = res.get("json") or {}
    res["summary"] = {k: j.get(k) for k in ("accepted", "n_proofs", "n_accepted", "n_negatives", "manifest_checked",
                                            "union_soundness_log2", "claimed_union_log2", "problems")}
    print(f"rust batch --dir {d}: exit {p.returncode} in {res['wall']:.2f}s {res['summary']}", flush=True)
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
    dump = Dump(Path(args.dump_dir) if args.dump_dir else None)
    report = {"relation": rel.name, "l": args.l, "vus": args.vus, "positives": [], "negatives": [], "system": sys_.summary()}
    failures = 0
    for kind in args.coins:
        bt = pv.batch(data, kind, zk=args.zk)
        dump.set_key(bt.key_bytes)
        lay = bt.lay
        log(f"[{kind}] layout {lay.summary()}; dims {bt.dims.describe()}; key sha256 {bt.key_sha256.hex()}")
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

        def neg(name: str, stmt: Statement, proof: Proof | bytes, seed: int = 1, stage: str | None = None):
            """``stage``: the reject reason must contain it (a forgery rejected earlier did not test what it names).
            ``proof`` as bytes: a file the writer would not produce (read through ``read_proof`` as ``verify-dir`` does)."""
            if isinstance(proof, bytes):
                try:
                    v = pv.verify(stmt, read_proof(proof), make_coins(kind, seed))
                except ValueError as e:
                    v = (False, f"unreadable: {e}", {})
            else:
                v = pv.verify(stmt, proof, make_coins(kind, seed))
            negs.append((name, v, stmt, proof, stage))

        def neg_prove(name: str, mutate=None, cheat=None, stmt=None, ypub=None, stage=None):
            b2 = bt if stmt is None else copy.copy(bt)
            if stmt is not None:
                b2.stmt, b2.stmt_digest = stmt, stmt.digest()
            r2 = pv.prove(b2, make_coins(kind, 1), mutate=mutate, cheat=cheat, ypub_override=ypub)
            neg(name, b2.stmt, r2.proof, stage=stage)
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
        neg("statement of another batch size (n_vus)", _stmt_nvus(pv, bt.stmt, [bt.stmt.n_vus[0] - 1] + bt.stmt.n_vus[1:]), res.proof)
        _, _, end0 = bt.lay.chain_layout(0).masks("cpu")
        off = np.flatnonzero(end0.numpy() == 0)
        if off.size:
            # the end constraint is masked there, so this proof of the altered statement is otherwise valid
            y_off = [y.copy() for y in bt.stmt.y]
            y_off[0][off[0]] = 1
            neg_prove(f"statement word y[{off[0]}] = 1 off a chain end, proved for that statement (non-canonical)",
                      stmt=_stmt_with(bt.stmt, y=y_off), ypub=y_off[0], stage="statement")
        pc = bt.stmt.n_vus[0] * steps
        if pc < args.l:
            a_pad = [a.copy() for a in bt.stmt.a]
            a_pad[0][pc, 0] = 1
            neg(f"statement operand a[{pc}][0] = 1 in a pad unit (non-canonical)", _stmt_with(bt.stmt, a=a_pad), res.proof,
                stage="statement")
        # (7) second encodings of the same proof / statement (red-team-ligerito-3 R3-2, R3-8): both readers must refuse them
        neg("framing JSON re-encoded (same values; not absorbed)", bt.stmt, _reencode_framing(res.proof.to_bytes()),
            stage="framing")
        if pv.word_bytes < 4:
            neg(f"statement words re-encoded at {2 * pv.word_bytes} bytes (same values)",
                _stmt_with(bt.stmt, word_bytes=2 * pv.word_bytes), res.proof, stage="statement shape")
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
    dump.close({"relation": rel.name, "l": args.l, "vus": args.vus, "commit": _git_commit(), "kind": "gate", "n_proofs": 1})
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


def _dump_prover(d: Path, man: dict, device: str) -> Prover:
    key = read_key((d / "key.bin").read_bytes())
    k = (key.relation, man.get("l"), device)
    if k not in _PROVERS:
        _PROVERS[k] = Prover(key.relation, device, man["l"])
    pv, kb = _PROVERS[k], (d / "key.bin").read_bytes()
    if kb != pv.key_bytes:
        # a ZK key (no next:* virtual rows) follows the batch's column count: derive it from the first statement's layout
        first = next((e for e in man.get("files", []) if e.get("expect") == "accept"), None)
        zk = not any(n.startswith("next:") for n in key.virt)
        if not (zk and first and pv.key_for(pv.stmt_layout(read_statement((d / first["stmt"]).read_bytes()), True))[2] == kb):
            raise SystemExit("key.bin differs from the key this code derives from the system")
    return pv


def verify_dir(d: Path, device: str = "cuda") -> dict:
    man = json.loads((d / "manifest.json").read_text())
    pv = _dump_prover(d, man, device)
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


def _file_coins(d: Path, e: dict, cls=LiveCoins):
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
    return cls(slots)


def record_slots(stream_stmt: bytes, rec: dict, rounds: int) -> list[bytes]:
    """The coin slots a live verifier issued for one batch, from ITS session record alone (red-team-ligerito-3 R3-7): the
    batch context of the STMT (rebuilt by the caller from the statement / params / key it verifies, and equal to the
    record's ``stmt_sha256``) and the verifier's own R commitments ``coin_commit(r_k, s_k)``; slot k =
    ``expand_challenge(ctx, k, label_k, r_k, 32)`` with ``label_k`` the label the verifier received with MSG k.  Exactly
    ``rounds`` coins and rounds, in order.  Nothing comes from the dump (``stream_binding`` / ``.coins`` are prover-written:
    a label or commitment list chosen after the coin opened would grind the challenge)."""
    coins, rds = rec.get("coins") or [], rec.get("rounds") or []
    if rounds < 1 or len(coins) != rounds or len(rds) != rounds or [rd.get("k") for rd in rds] != list(range(rounds)):
        raise ValueError(f"the verifier's record has {len(coins)} coins / {len(rds)} rounds, its session {rounds} rounds")
    if rec.get("stmt_sha256") != sha256(stream_stmt).hex():
        raise ValueError("the verifier's record is for another STMT (statement / params / key)")
    ops = [(bytes.fromhex(c["r"]), bytes.fromhex(c["s"])) for c in coins]
    ctx = live_mod.batch_context(stream_stmt, [live_mod.pm.coin_commit(r, s) for r, s in ops])
    return [live_mod.expand_challenge(ctx, k, rd["label"].encode(), r, 32) for k, ((r, _), rd) in enumerate(zip(ops, rds))]


def _load_session(sd: Path) -> tuple[str, dict, dict, dict]:
    sj, vj = (sd / "session.json").read_bytes(), (sd / "verdict.json").read_bytes()
    sess, verdict = json.loads(sj), json.loads(vj)
    return verdict.get("session_id", sd.name), sess, verdict, {"dir": str(sd), "session_json_sha256": sha256(sj).hex(),
                                                               "verdict_json_sha256": sha256(vj).hex()}


def verify_session(d: Path, records: list[Path], device: str = "cuda", sessions_root: Path | None = None) -> dict:
    """Authenticate a live dump's coins against the issuing verifier's own session records (``session.json`` +
    ``verdict.json`` of ``live.ChallengeServerSession``, one dir per session).  Per accepted live proof: a batch of an
    ACCEPTED session whose ``stmt_sha256`` is this statement + params + key; the proof verifies with the coin slots derived
    from that record alone (``record_slots``: the verifier's commitments, openings and received labels, exact length) and
    consumes all of them; every round's recorded ``msg_sha256`` and label = those the proof's own absorbs give (the verifier
    opened coin k only after MSG k).  Each record batch authenticates at most one proof.
    Claims (red-team F12, F5, R3-10): per proof, the interactive union times the number of batches the verifier opened for
    that STMT across every session under ``sessions_root`` (the verifier's own store, aborted / refused sessions included:
    abort-and-retry); without ``sessions_root`` no interactive claim is made.  The job: the union over its distinct
    statements (at least the manifest's ``n_proofs``)."""
    man = json.loads((d / "manifest.json").read_text())
    pv = _dump_prover(d, man, device)
    by_stmt: dict = {}
    rec_meta = []
    for r in records:
        sid, sess, verdict, meta = _load_session(Path(r) if Path(r).is_dir() else Path(r).parent)
        rec_meta.append(meta)
        for b in sess["batches"]:
            if b.get("stmt_sha256"):
                by_stmt.setdefault(b["stmt_sha256"], []).append((sid, b, verdict))
    attempts: dict | None = None
    unrecorded: list[str] = []
    if sessions_root is not None:
        attempts = {}
        for sd in sorted(p for p in Path(sessions_root).iterdir() if p.is_dir()):
            if not (sd / "session.json").exists():
                if (sd / "hello.json").exists():
                    unrecorded.append(sd.name)      # opened, record never written: its statements are unknown
                continue
            for b in json.loads((sd / "session.json").read_text()).get("batches", []):
                if b.get("stmt_sha256"):
                    attempts[b["stmt_sha256"]] = attempts.get(b["stmt_sha256"], 0) + 1
        if unrecorded:
            attempts = None
    used: set = set()
    items = []
    for e in man["files"]:
        if e["coins"] != "live" or e["expect"] != "accept":
            continue
        item = {"proof": e["proof"], "authenticated": False}
        try:
            st, pf = read_statement((d / e["stmt"]).read_bytes()), read_proof((d / e["proof"]).read_bytes())
            key_sha = pv.key_for(pv.stmt_layout(st, bool(pf.params.get("zk"))))[3]
            stream_stmt = b"lgto-stmt|v1|" + sha256(pf.absorbed_params()) + st.digest() + key_sha
            digest = sha256(stream_stmt).hex()
            item["stmt_sha256"] = digest
            cands = [c for c in by_stmt.get(digest, []) if (c[0], c[1]["batch"]) not in used]
            if not cands:
                raise ValueError("no unused batch of the given verifier sessions has this STMT (statement / params / key)")
            hint = ((e.get("stream_binding") or {}).get("openings") or [[None]])[0][0]     # an index into the record only
            cands.sort(key=lambda c: not (c[1].get("coins") and c[1]["coins"][0]["r"] == hint))
            whys = []
            for sid, rec, verdict in cands:
                try:
                    if not verdict.get("accepted"):
                        raise ValueError(f"session {sid} was not accepted by its verifier: {verdict.get('errors')}")
                    slots = record_slots(stream_stmt, rec, int(verdict["rounds"]))
                    vc = prove_mod.SessionReplayCoins(slots)
                    try:
                        ok, why, info = pv.verify(st, pf, vc)
                    except StopIteration:
                        raise ValueError(f"the proof draws more than the {len(slots)} coins the verifier issued") from None
                    if not ok:
                        raise ValueError(f"proof rejected under the verifier's coins: {why}")
                    if len(vc.msgs) != len(slots):
                        raise ValueError(f"the proof drew {len(vc.msgs)} coins, the verifier issued {len(slots)}")
                    for k, (m, lb, rd) in enumerate(zip(vc.msgs, vc.labels, rec["rounds"])):
                        if rd["msg_sha256"] != sha256(m).hex() or rd["label"] != lb.decode():
                            raise ValueError(f"round {k}: the verifier opened its coin on another message / label than the proof's")
                except (ValueError, KeyError, IndexError) as ex:
                    whys.append(f"session {sid} batch {rec.get('batch')}: {ex}")
                    continue
                used.add((sid, rec["batch"]))
                n_att = None if attempts is None else max(1, attempts.get(digest, 0))
                claim = prove_mod.soundness_claim(info["union"], "live", verifier_record=True)
                if n_att is None:
                    claim = dict(claim, claimed_log2=None, per_session_log2=claim["claimed_log2"],
                                 basis=f"none: sessions {unrecorded} under --sessions-root have no record" if unrecorded else
                                 "none: the verifier's full session store was not given (--sessions-root), so abort-and-retry "
                                 "sessions for this STMT are not counted (R3-10)")
                else:
                    claim = dict(claim, per_session_log2=claim["claimed_log2"], attempts=n_att,
                                 claimed_log2=claim["claimed_log2"] + math.log2(n_att),
                                 basis=claim["basis"] + f"; x {n_att} batch(es) the verifier opened for this STMT (sessions root)")
                dump_slots = None
                if e.get("coins_file"):
                    raw = (d / e["coins_file"]).read_bytes()
                    dump_slots = [raw[4 + 32 * i:4 + 32 * (i + 1)] for i in range(int.from_bytes(raw[:4], "little"))]
                item.update(authenticated=True, session=sid, batch=rec["batch"], rounds=len(slots), claim=claim,
                            dump_coins_equal_record=dump_slots == slots)
                break
            if not item["authenticated"]:
                raise ValueError("; ".join(whys))
        except (ValueError, KeyError, IndexError) as ex:
            item["why"] = str(ex)
        items.append(item)
    stmts = {i["stmt_sha256"]: i["claim"] for i in items if i["authenticated"]}
    n_proofs = max(int(man.get("n_proofs") or 1), len(stmts))
    ok = bool(items) and all(i["authenticated"] for i in items) and len(stmts) == n_proofs
    per = [c["claimed_log2"] for c in stmts.values()]
    claimable = ok and all(p is not None for p in per)
    job = math.log2(sum(2.0 ** p for p in per)) if claimable else None
    out = {"checker": "backends.direct.ligerito.run verify-session (coins derived from the issuing verifier's session record; "
                      "proof MSG digests + labels vs that record)",
           "records": rec_meta, "sessions_root": str(sessions_root) if sessions_root else None, "unrecorded_sessions": unrecorded,
           "live_proofs": len(items), "authenticated": sum(i["authenticated"] for i in items), "n_proofs": n_proofs,
           "distinct_statements": len(stmts), "per_proof_claimed_log2": max(per) if claimable else None,
           "job_claimed_log2": job, "items": items}
    (d / "verify_session.json").write_text(json.dumps(out, indent=1, default=str))
    for i in items:
        print(f"  {i['proof']}: {'authenticated' if i['authenticated'] else 'NOT authenticated: ' + i['why']}"
              f"{' (session ' + str(i.get('session')) + ' batch ' + str(i.get('batch')) + ')' if i.get('session') else ''}", flush=True)
    print(f"verify-session {d}: {out['authenticated']}/{out['live_proofs']} live proofs authenticated; job claim "
          f"{out['job_claimed_log2']} (n_proofs {n_proofs}; attempts {'from ' + str(sessions_root) if sessions_root else 'not counted: no claim'})",
          flush=True)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------------------------------------------------

def _buckets(t: dict) -> dict:
    """proto / prove.py timer buckets -> the contract's t.* buckets (seconds)."""
    g = lambda *ks: sum(t.get(k, 0.0) for k in ks)  # noqa: E731
    return {"witness": g("witness"),
            "encoding_commitment": g("commit_r1_alloc", "commit_r1_encode", "commit_r1_merkle", "commit_r1_stream", "commit_r1_recompute",
                                     "recursion_encode", "recursion_merkle"),
            "arithmetic": g("zero_check", "sumcheck_r1_contract", "sumcheck_host", "sumcheck_r1_fold", "sumcheck_device", "recursion_contract",
                            "recursion_fold"),
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
    # an n-proof job: each proof sized for 2^-(128 + ceil log2 n) so the union over the distinct statements meets 2^-128 (F5).
    # Against a live verifier every rep proves the same statements again: it opens reps x n batches, each an attempt that
    # verify-session counts (R3-10), so the proofs are sized for 2^-(128 + ceil log2 (reps n))
    attempts = -(-total_vus // bv) * (args.reps if args.coins == "live" and args.verifier else 1)
    per_proof_target = -128.0 - math.ceil(math.log2(attempts))
    bts = [pv.batch(data[i:i + bv], args.coins, zk=args.zk, target_log2=per_proof_target) for i in range(0, total_vus, bv)]
    nb_ = len(bts)
    bt = bts[0]
    lay = bt.lay
    print(f"{nb_} batch(es) of {bv} VUs; layout {lay.summary()}; dims {bt.dims.describe()}; key sha256 {bt.key_sha256.hex()}",
          flush=True)
    dump = Dump(Path(args.dump_dir) if args.dump_dir else None)
    dump.set_key(bt.key_bytes)
    live = args.coins == "live"
    stream_info: dict = {}
    if live:
        stmts = [prove_mod.statement_for_stream(b, b.key_sha256) for b in bts]
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
    # n_proofs: distinct statements of one job; each proof must reach 2^-(128 + log2 n_proofs) (red-team F5)
    dump.close({"relation": rel.name, "l": args.l, "vus": total_vus, "commit": _git_commit(), "kind": "bench", "n_proofs": nb_})
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
    snd = pv.soundness(bt.dims, int(bt.params.get("t_pad", 0)), bt.lay)
    peak = max(r.peak_device_bytes for _, it in runs for r, _, _ in it)
    rust_ok = bool(rust and rust["exit"] == 0)
    # the verdict frame proves the round order only; the coins count once verify-session matched the proofs' messages to
    # the verifier's own session record, which the prover does not hold during the run
    session_accepted = bool(live and args.verifier and (stream_info.get("verdict") or {}).get("accepted"))
    live_verified = False
    claim = prove_mod.soundness_claim(snd, args.coins, verifier_record=live_verified)
    per_proof = claim["claimed_log2"]
    claimed = None if per_proof is None else per_proof + math.log2(nb_)      # union over the job's nb_ distinct statements
    zk_on = False                                   # complete ZK: needs the sumcheck masks and the PCS column padding
    zk_partial = f"partial: {ZK_PARTIAL}; missing: {ZK_MISSING}" if args.zk else None
    if zk_on and live_verified and rust_ok:
        proof_class, zk_mode = "COMPLETE_ZK_BACKEND", "ligerito-zk (masks + per-column RS padding + Libra sumcheck masks), live coins"
        omitted = "external authentication"
    else:
        proof_class, zk_mode = "NON_ZK_PROOF_DIAGNOSTIC", "none"
        omitted = "external authentication; ZK masking (" + (ZK_MISSING if args.zk else STUBBED) + ")"
        if live and not live_verified:
            omitted += ("; coins from live.LocalChallenges (no network verifier)" if not args.verifier
                        else "; live coins pending `run.py verify-session` against the verifier's session record" if session_accepted
                        else "; live session not accepted")
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
                  "per_proof_target_log2": per_proof_target, "per_proof_claimed_log2": per_proof, "n_proofs": nb_,
                  "interactive_union_log2": snd["total_log2"], "terms_log2": snd["terms_log2"],
                  **({"fs_queries_log2": claim["queries_log2"], "fs_worst_term": claim["worst_term"]} if "worst_term" in claim else {}),
                  "union_over": nb_, "stubbed": STUBBED},
        hardware=hw, software=sw,
        preprocessing=f"constraint compilation + layout constraints + verification key (functions of the relation only, {pv.compile_seconds:.2f}s)",
        retention="retained", instances=inst, scope="vu", omitted=omitted,
        relation=rel.relation_name, model=rel.model_name, commit=_git_commit(), step_k=rel.k, steps_per_vu=rel.steps,
        rows=pv.sys.m, committed_rows=lay.m, layout=lay.summary(), witness_generation=rel.witness_generation,
        statement_format="LGTS0001", proof_format="LGTO0001", key_format="LGVK0001", key_sha256=bt.key_sha256.hex(),
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
        secs = [it.get("seconds") for it in rust["json"].get("proofs", []) if isinstance(it, dict) and it.get("seconds") is not None
                and it.get("accepted")]
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
    s = sub.add_parser("verify-session")
    s.add_argument("dir")
    s.add_argument("records", nargs="+", help="the verifier's session dirs (session.json + verdict.json)")
    s.add_argument("--sessions-root", help="the verifier's whole session store (every session it opened): counts the "
                                           "batches per statement, aborted ones included; required for a claim")
    s.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.cmd == "verify-session":
        out = verify_session(Path(args.dir), [Path(r) for r in args.records], args.device,
                             Path(args.sessions_root) if args.sessions_root else None)
        return 0 if out["live_proofs"] and out["authenticated"] == out["live_proofs"] else 1
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
