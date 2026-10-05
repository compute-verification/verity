"""Red-team 3: forged proofs against the GPU A-GKR *Python verifier* (``backends/gkr/gpu/prover.py::verify``).

Wave 2 (``vu_forge_a``) attacked the Rust prover/verifier over Goldilocks; the GPU port (BabyBear^6, Ligero
``k=4096, n=16384, t=192``, SHA-256 Merkle) has its own verifier, a Python port, which is what accepts the
4096/4096 positives and rejects the 52 negatives.  Every case below builds the *prover's* instance (forged
witness) and the *verifier's* instance (its statement: circuit hashes, unit counts, public words) separately,
runs the real ``prover.prove`` on the former (with the prover's own sanity checks disabled where the attack needs
it) and the real ``prover.verify`` on the latter, and records ACCEPTED / REJECTED with the verifier's error.

Classes (``expected`` is ``reject`` unless stated):

* chain linkage (wave-2 classes re-run against the Python verifier): ``splice``, ``c0_nonzero``,
  ``epilogue_y32``, ``epilogue_claim``, ``public_word``, ``swap_words``;
* LogUp: ``logup_oot_mult`` (an out-of-table R16 value whose multiplicity is credited to a table row so the
  prover's fractional sum still vanishes -- it cannot, so the prover's root check is bypassed and the sumcheck is
  run on the false leaves), ``logup_table_subst`` (prover proves against a table with one extra row, circuit
  hash is the verifier's because R16 is implicit in the parser, not in the hashed text); a multiplicity ``m + p``
  is the same field element as ``m`` (not a distinct witness: argued, no case);
* proof-object malleation against the verifier's parsing: ``proof_noncanonical_msg`` (a transcript message
  coordinate ``v + p``), ``opening_column_swap`` (two opened columns exchanged), ``opening_path_truncated``,
  ``opening_column_value_plus_p``, ``opening_fewer_columns``;
* ``honest`` control (expected ``accept``).

    PYTHONPATH=packages/verity/src:backends/numerical/python:backends/gkr \\
        python -m verity_numerical.redteam.gpu_a_forge --out DIR [--vus 2] [--device cpu] [--cases a,b]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

from verity_numerical.checker import REAL, Params
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.gadgets import Epilogue
from verity_numerical.checker.step import step_output
from verity_numerical.redteam.babybear import gen_step_bb
from verity_numerical.redteam.vu_forge_a import _fp32_with_cast, load_real_vus


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _import_gpu():
    sys.path.insert(0, str(_repo_root() / "backends" / "gkr"))
    from gpu import bb_export, circuit, ligero, logup, prover

    return bb_export, circuit, ligero, logup, prover


class BBChain:
    """One VU over BabyBear as unit rows + epilogue row, re-chainable from any step with any incoming word."""

    def __init__(self, bb, unit, epi, a: list[int], b: list[int], p: Params = REAL):
        self.bb, self.unit, self.epi, self.a, self.b, self.p = bb, unit, epi, a, b, p
        self.steps = len(a) // bb.STEP
        self.rows: list[list[int]] = []
        self.words: list[int] = []
        self.rechain(0, 0)

    def clone(self) -> BBChain:
        """Copy the rows/words; share the (unpicklable) module handles."""
        c = object.__new__(BBChain)
        c.bb, c.unit, c.epi, c.a, c.b, c.p, c.steps = self.bb, self.unit, self.epi, self.a, self.b, self.p, self.steps
        c.rows, c.words = [list(r) for r in self.rows], list(self.words)
        return c

    def unit_row(self, i: int, c: int) -> tuple[list[int], int]:
        s = self.bb.STEP
        ai, bi = self.a[i * s:(i + 1) * s], self.b[i * s:(i + 1) * s]
        h = gen_step_bb(self.p, c, ai, bi)
        y = step_output(self.p, c, ai, bi)
        row = self.bb.unit_witness(self.unit[0], self.unit[1], c, ai, bi, y, h)
        bad = self.bb.verify_rows(self.unit[0], row)
        assert bad is None, f"a forged unit must still satisfy the unit circuit: {bad}"
        return row, y

    def rechain(self, start: int, c: int) -> None:
        rows, words = self.rows[:start], self.words[:start]
        for i in range(start, self.steps):
            row, y = self.unit_row(i, c)
            rows.append(row)
            words.append(y)
            c = y
        self.rows, self.words = rows, words

    @property
    def y32(self) -> int:
        return self.words[-1]

    def cast(self, y32: int | None = None) -> int:
        y32 = self.y32 if y32 is None else y32
        return int(Epilogue.check(IntCtx(), self.p, y32, Epilogue.gen(self.p, y32)))

    def epilogue_row(self, y32: int | None = None, y16: int | None = None) -> list[int]:
        y32 = self.y32 if y32 is None else y32
        y16 = self.cast(y32) if y16 is None else y16
        return self.bb.epilogue_witness(self.epi[0], self.epi[1], y32, y16, Epilogue.gen(self.p, y32))


def make_instance(prover, circuit, bb, unit, epi, chains, epi_rows, public, device, utext=None, etext=None):
    import torch

    P = bb.P
    utext = bb.circuit_text(unit[0]) if utext is None else utext
    etext = bb.circuit_text(epi[0]) if etext is None else etext
    ucirc = circuit.parse_circuit(utext)
    ecirc = circuit.parse_circuit(etext)
    units = torch.tensor([[v % P for v in r] for ch in chains for r in ch.rows], dtype=torch.int64, device=device)
    epis = torch.tensor([[v % P for v in r] for r in epi_rows], dtype=torch.int64, device=device)
    cc = bb.chain_columns(unit[0], epi[0])
    chain = prover.Chain(chains[0].steps, cc["col_c"], cc["col_y"], cc["epi_y32"], cc["epi_y16"], [int(y) for y in public])
    segs = [
        prover.Segment("unit", ucirc, circuit.layers(ucirc), units, ucirc.hash),
        prover.Segment("epilogue", ecirc, circuit.layers(ecirc), epis, ecirc.hash),
    ]
    return prover.Instance(segs, chain)


def _sha_rows(rows) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(",".join(str(int(v)) for v in r).encode())
        h.update(b"\n")
    return h.hexdigest()


def build_cases(n_vus: int, root: Path, device, only: set[str] | None = None, p: Params = REAL):
    """Yields ``(name, expected, mechanism, prover_inst, verifier_inst, prove_kwargs, mutate_proof)``."""
    bb, circuit, _ligero, _logup, prover = _import_gpu()
    unit = bb.build_unit_circuit()
    epi = bb.build_epilogue_circuit()
    vus = load_real_vus(root, n_vus + 2)[:n_vus]
    assert len(vus) == n_vus, f"need {n_vus} real VUs with correct words"
    chains = [BBChain(bb, unit, epi, a, b, p) for a, b, _y in vus]
    for ch, (_a, _b, y) in zip(chains, vus):
        assert ch.cast() == y, "the recomputed word must equal the recorded one"
    honest_epi = [ch.epilogue_row() for ch in chains]
    honest_pub = [ch.cast() for ch in chains]

    def inst(chs=chains, epis=honest_epi, pub=honest_pub, **kw):
        return make_instance(prover, circuit, bb, unit, epi, chs, epis, pub, device, **kw)

    def want(name):
        return only is None or name in only

    if want("honest"):
        yield "honest", "accept", "control", inst(), inst(), {}, None

    # ---- chain linkage --------------------------------------------------------------------------------------
    if want("splice") and n_vus >= 2:
        k = 40
        f = [ch.clone() for ch in chains]
        f[1].rechain(k, chains[0].words[k - 1])
        pub = honest_pub[:]
        pub[1] = f[1].cast()
        epis = [f[0].epilogue_row(), f[1].epilogue_row()] + honest_epi[2:]
        yield "splice", "reject", "X[(1,40),c] - X[(1,39),y] = 0 (add_chain step constraint)", inst(f, epis, pub), inst(pub=pub), {}, None
    if want("c0_nonzero") and n_vus >= 2:
        f = [ch.clone() for ch in chains]
        f[1].rechain(0, chains[0].y32)
        pub = honest_pub[:]
        pub[1] = f[1].cast()
        epis = [f[0].epilogue_row(), f[1].epilogue_row()] + honest_epi[2:]
        yield "c0_nonzero", "reject", "X[(1,0),c] = 0 (add_chain first-step constraint)", inst(f, epis, pub), inst(pub=pub), {}, None
    if want("epilogue_y32"):
        y16f = (honest_pub[0] + 1) & 0xFFFF
        y32f = _fp32_with_cast(y16f)
        epis = [chains[0].epilogue_row(y32=y32f, y16=y16f)] + honest_epi[1:]
        pub = honest_pub[:]
        pub[0] = y16f
        yield "epilogue_y32", "reject", "X_epi[0,y32] - X[(0,95),y] = 0", inst(epis=epis, pub=pub), inst(pub=pub), {}, None
    if want("epilogue_claim"):
        y16f = (honest_pub[0] + 1) & 0xFFFF
        epis = [chains[0].epilogue_row(y16=y16f)] + honest_epi[1:]
        pub = honest_pub[:]
        pub[0] = y16f
        yield "epilogue_claim", "reject", "epilogue circuit assertion y16 = hi + carry", inst(epis=epis, pub=pub), inst(pub=pub), {}, None
    if want("public_word"):
        pub = honest_pub[:]
        pub[0] ^= 1
        yield "public_word", "reject", "X_epi[0,y16] = public word", inst(pub=pub), inst(pub=pub), {}, None
    if want("swap_words") and n_vus >= 2 and honest_pub[0] != honest_pub[1]:
        pub = honest_pub[:]
        pub[0], pub[1] = pub[1], pub[0]
        yield "swap_words", "reject", "X_epi[v,y16] = public word_v", inst(pub=pub), inst(pub=pub), {}, None

    # ---- LogUp ------------------------------------------------------------------------------------------------
    def bare_r16_column():
        # a query whose linear form is a single column with coefficient 1 and no constant
        ucirc = circuit.parse_circuit(bb.circuit_text(unit[0]))
        for qq in ucirc.queries:
            if qq.table == "R16" and len(qq.cols[0].terms) == 1 and qq.cols[0].konst == 0 and qq.cols[0].terms[0][1] == 1:
                return qq.cols[0].terms[0][0], ucirc.col_names[qq.cols[0].terms[0][0]]
        raise AssertionError("no bare R16 query column")

    if want("logup_oot_mult"):
        # row 3 of VU 0: push an R16-checked column to 65536 (out of T16), and credit the multiplicity of
        # table row 65535 so the prover's histogram still sums to the query count.  The prover's own checks
        # (`multiplicities` raises, `prove_table` raises on p_root != 0) are bypassed; the GKR runs on the false leaves.
        f = [ch.clone() for ch in chains]
        col, cname = bare_r16_column()
        f[0].rows[3] = list(f[0].rows[3])
        f[0].rows[3][col] = 65536
        yield "logup_oot_mult", "reject", f"LogUp sumcheck (false fractional sum; column {cname} := 65536)", inst(f), inst(), {"forge_logup": ("oot", 65536, 65535)}, None
    if want("logup_table_subst"):
        # the prover's R16 has a 65537th row (65536 is 'in the table' for the prover: its fractional sum vanishes
        # honestly); the circuit hash is the verifier's (R16 is implicit, not in the hashed text).
        f = [ch.clone() for ch in chains]
        col, cname = bare_r16_column()
        f[0].rows[3] = list(f[0].rows[3])
        f[0].rows[3][col] = 65536
        ip = inst(f)
        for t in ip.tables:
            if t.name == "R16":
                t.rows.append([65536])
        yield "logup_table_subst", "reject", f"verifier's own R16 rows in add_lookup_claim ({cname} := 65536, prover's R16 has 65537 rows)", ip, inst(), {}, None

    # ---- proof malleation -----------------------------------------------------------------------------------
    def mut_noncanonical(proof):
        pr = copy.deepcopy(proof)
        m = list(pr.msgs[5])
        m[0] += bb.P
        pr.msgs[5] = tuple(m)
        return pr

    def mut_col_swap(proof):
        pr = copy.deepcopy(proof)
        c = pr.opening.columns
        c[0], c[1] = c[1], c[0]
        return pr

    def mut_path_trunc(proof):
        pr = copy.deepcopy(proof)
        vals, path = pr.opening.columns[0]
        pr.opening.columns[0] = (vals, path[:-1])
        return pr

    def mut_col_plus_p(proof):
        pr = copy.deepcopy(proof)
        vals, path = pr.opening.columns[0]
        vals = list(vals)
        vals[0] += bb.P
        pr.opening.columns[0] = (vals, path)
        return pr

    def mut_fewer(proof):
        pr = copy.deepcopy(proof)
        pr.opening.columns = pr.opening.columns[:-1]
        return pr

    for name, fn, mech in [
        ("proof_noncanonical_msg", mut_noncanonical, "transcript bytes differ (to_bytes of v+p) -> challenges differ"),
        ("opening_column_swap", mut_col_swap, "merkle path at the verifier's index"),
        ("opening_path_truncated", mut_path_trunc, "len(path) != depth"),
        ("opening_column_value_plus_p", mut_col_plus_p, "non-canonical column value check"),
        ("opening_fewer_columns", mut_fewer, "len(columns) != t"),
    ]:
        if want(name):
            yield name, "reject", mech, inst(), inst(), {}, fn


def prove_forged(prover, logup, inst, pcs: bool, forge_logup):
    """The real prover, with its own sanity checks disabled for the LogUp forgeries."""
    import torch

    if forge_logup is None:
        return prover.prove(inst, pcs)
    kind, _oot_value, credit_row = forge_logup
    assert kind == "oot"
    orig_mult = logup.multiplicities
    orig_prove_table = logup.prove_table
    P = prover.P

    def forged_mult(t, trows, values):
        if t.name != "R16":
            return orig_mult(t, trows, values)
        m = torch.zeros((trows.shape[0],), dtype=torch.int64, device=trows.device)
        for v in values:
            key = v[:, 0].clone()
            if kind == "oot":
                key[key >= trows.shape[0]] = credit_row      # credit the out-of-table query to a table row
            m += torch.bincount(key, minlength=trows.shape[0])
        return m

    def forged_prove_table(p_leaf, q_leaf, n, tr, out):
        # identical to logup.prove_table but without the `root_p != 0` abort
        dev = q_leaf.device
        levels_p, levels_q = [p_leaf], [q_leaf]
        for _k in range(n, 0, -1):
            np_, nq = logup._combine_level(levels_p[-1], levels_q[-1])
            levels_p.append(np_)
            levels_q.append(nq)
        root_q = logup.py_ext(levels_q[n][0].to(torch.int64))
        out.append(root_q)
        tr.absorb_ext(root_q)
        lam = tr.challenge()
        rho = []
        value = logup.e_mul(lam, root_q)
        for k in range(n):
            cp, cq = levels_p[n - k - 1], levels_q[n - k - 1]
            p0, p1 = logup._split(cp)
            q0, q1 = logup._split(cq)
            e = logup.eq_table(rho, dev)
            lam_t = logup.t_ext(lam, dev)
            x_star = []
            length = 1 << k
            while length > 1:
                half = length // 2
                h = logup.round_poly(p0, p1, q0, q1, e, half, lam_t)
                out.extend(h)
                tr.absorb_exts(h)
                r = tr.challenge()
                x_star.append(r)
                rt = logup.t_ext(r, dev)
                p0 = logup.fold_base(p0, rt) if p0.ndim == 1 else logup.fold(p0, rt)
                p1 = logup.fold_base(p1, rt) if p1.ndim == 1 else logup.fold(p1, rt)
                q0, q1, e = logup.fold(q0, rt), logup.fold(q1, rt), logup.fold(e, rt)
                length = half
            vals = []
            for v in (p0, p1, q0, q1):
                if v.ndim == 1:
                    vals.append((int(v[0]) % P,) + (0,) * (logup.DEG - 1))
                else:
                    vals.append(logup.py_ext(v[0].to(torch.int64)))
            out.extend(vals)
            tr.absorb_exts(vals)
            mu = tr.challenge()
            pv = logup.e_add(vals[0], logup.e_mul(mu, logup.e_sub(vals[1], vals[0])))
            qv = logup.e_add(vals[2], logup.e_mul(mu, logup.e_sub(vals[3], vals[2])))
            lam = tr.challenge()
            value = logup.e_add(pv, logup.e_mul(lam, qv))
            rho = x_star + [mu]
        return logup.LeafClaim(rho, lam, value)

    logup.multiplicities = forged_mult
    logup.prove_table = forged_prove_table
    try:
        return prover.prove(inst, pcs)
    finally:
        logup.multiplicities = orig_mult
        logup.prove_table = orig_prove_table


def run(out: Path, n_vus: int, root: Path, device_name: str, only: set[str] | None, pcs: bool = True) -> dict:
    import torch

    _bb, _circuit, ligero, logup, prover = _import_gpu()
    device = torch.device(device_name)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, expected, mech, inst_p, inst_v, kw, mutate in build_cases(n_vus, root, device, only):
        rec = {"case": name, "expected": expected, "mechanism": mech, "vus": n_vus, "pcs": pcs,
               "prover_units_sha256": _sha_rows(inst_p.segs[0].cols.cpu().tolist()),
               "prover_epilogue_sha256": _sha_rows(inst_p.segs[1].cols.cpu().tolist()),
               "verifier_public_words": inst_v.chain.public_y16, "prover_public_words": inst_p.chain.public_y16}
        t0 = time.perf_counter()
        try:
            proof, st = prove_forged(prover, logup, inst_p, pcs, kw.get("forge_logup"))
            rec["prover_s"] = time.perf_counter() - t0
            rec["proof_elems"] = st.proof_elems
            if mutate is not None:
                proof = mutate(proof)
        except Exception as e:  # noqa: BLE001
            rec.update({"verdict": "PROVER_ABORT", "error": f"{type(e).__name__}: {e}", "prover_s": time.perf_counter() - t0})
            results.append(rec)
            print(json.dumps(rec), flush=True)
            continue
        t1 = time.perf_counter()
        try:
            prover.verify(inst_v, proof, pcs)
            rec["verdict"] = "ACCEPTED"
        except prover.VerifyError as e:
            rec["verdict"] = "REJECTED"
            rec["error"] = str(e)
        except Exception as e:  # noqa: BLE001
            rec["verdict"] = "REJECTED_EXCEPTION"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-800:]
        rec["verifier_s"] = time.perf_counter() - t1
        rec["ok"] = (rec["verdict"] == "ACCEPTED") == (expected == "accept")
        results.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "traceback"}), flush=True)
    summary = {"schema": "redteam-3/gpu-a-forge/v1", "device": device_name, "vus": n_vus, "root": str(root),
               "ligero": {"k": ligero.DEFAULT.k, "n": ligero.DEFAULT.n, "t": ligero.DEFAULT.t},
               "forged_accepted": [r["case"] for r in results if r.get("verdict") == "ACCEPTED" and r["expected"] == "reject"],
               "results": results}
    (out / "gpu_a_forge.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vus", type=int, default=2)
    ap.add_argument("--root", default=str(_repo_root() / "fixtures" / "bench-instances" / "v1"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cases", default=None, help="comma-separated subset")
    a = ap.parse_args(argv)
    only = set(a.cases.split(",")) if a.cases else None
    s = run(Path(a.out), a.vus, Path(a.root), a.device, only)
    print(f"forged accepted: {s['forged_accepted']}")
    return 1 if s["forged_accepted"] else 0


if __name__ == "__main__":
    sys.exit(main())
