"""GemmAccumulator<K=1536> over ``sm120.mma.m16n8k64.e2m1.nvf4`` on the B-Ligero chain prover: 24 chained NVFP4 units per VU
from ``c_0 = +0``, the final FP32 accumulator word public (its three components, ``relation``: the chain ends on bit
fields, not on a field element), the accumulators witness rows linked by the chain test (``..chain``; statement binding as
PROTOCOL.md: the claimed words and the operand bytes -- 64 nibbles + 4 scale bytes per operand per unit -- enter the
statement digest before any challenge).

    python -m backends.direct.ligero.run --relation fp4-nvf4 gate-vu  --device cuda --vus 2048 --out result.json
    python -m backends.direct.ligero.run --relation fp4-nvf4 bench-vu --device cuda --batch 4096 --total-vus 4096 --reps 3 --out result.json

Instances are synthetic and deterministic (``instances_fp4``: VU ``i`` draws its operands from edge family
``VU_FAMILIES[i % len]`` -- uniform, zero / subnormal / top / spanning scales, sparse and saturated codes, exact
cancellation, per-step mixtures -- from a seed; the accumulators recorded by the model ``BLACKWELL_SM120_NVF4.step``).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from verity.verification.target import NVFP4_SM120
from verity_numerical.bench import contract

from .. import protocol as protocol_mod
from ..chain import Layout
from ..compile import P
from ..protocol import Config, RelationHooks, config_for, n_rows, prove, soundness, verify
from ..witness import tables
from .census import census_fp4
from .relation import K, SCALES, STEPS, WORDS, NVFP4, compile_fp4_unit
from .witness import FAMILIES, MODEL, VARIANTS, hints_fp4, public_vectors_fp4, random_units, unit_hints

__all__ = ["FP4_HOOKS", "FP4ChainRunner", "INSTANCE_SEED", "MODEL_NAME", "RELATION_NAME", "VU_FAMILIES", "bench_vu_fp4",
           "gate_vu_fp4", "instances_digest", "instances_fp4"]

RELATION_NAME = "GemmAccumulator<1536> over sm120.mma.m16n8k64.e2m1.nvf4"
MODEL_NAME = MODEL.name
INSTANCE_SEED = 20260922
#: The operand families a VU's 24 steps are drawn from (``witness.FAMILIES`` minus the accumulator kind, which the chain
#: supplies), plus ``mixed``: every step its own family.
VU_FAMILIES = ("random", "zero-scales", "subnormal-scales", "top-scales", "top-scales-zero-acc", "span-scales", "sparse",
               "sparse-subnormal-acc", "all-zero-codes", "half-codes", "max-codes", "cancel", "cancel-acc", "mixed")


def _public_vectors_torch(p, c, a, b, y, device="cpu") -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.int64, device=device) for k, v in public_vectors_fp4(a, b, c, y).items()}


FP4_HOOKS = RelationHooks(_public_vectors_torch, word_dtype="<u1", y16_dtype="<u4", y16_bits=32, tag=b"|rel=fp4-nvf4|v1")


def vu_operands(rng: np.random.Generator, family: str) -> tuple[np.ndarray, np.ndarray]:
    """(A, B) each (STEPS, WORDS): the 24 steps' operand words of one VU of ``family``."""
    if family == "mixed":
        fams = list(FAMILIES)
        rows = [random_units(rng, 1, fams[int(rng.integers(0, len(fams)))]) for _ in range(STEPS)]
        A = np.concatenate([r[1] for r in rows]); B = np.concatenate([r[2] for r in rows])
    else:
        _, A, B = random_units(rng, STEPS, family)
    return A, B


def instances_fp4(n: int, seed: int = INSTANCE_SEED, start: int = 0) -> list[tuple[np.ndarray, np.ndarray, list[int], int]]:
    """VUs ``start .. start + n``: (A (24, 68), B (24, 68), accs (24,), y_final) with the accumulators recorded from the model."""
    out = []
    for i in range(start, start + n):
        rng = np.random.default_rng([seed, i])
        A, B = vu_operands(rng, VU_FAMILIES[i % len(VU_FAMILIES)])
        accs, c = [], 0
        for s in range(STEPS):
            c = MODEL.step(c, [int(x) for x in A[s]], [int(x) for x in B[s]])
            accs.append(c)
        out.append((A, B, accs, accs[-1]))
    return out


def instances_digest(n: int, seed: int = INSTANCE_SEED) -> str:
    return hashlib.sha256(f"fp4-nvf4 synthetic|seed={seed}|n={n}|K={STEPS * K}|families={','.join(VU_FAMILIES)}".encode()).hexdigest()


def vu_units(A: np.ndarray, B: np.ndarray, accs: list[int]) -> list[tuple]:
    units, c = [], 0
    for s in range(STEPS):
        units.append((c, A[s], B[s], accs[s]))
        c = accs[s]
    return units


def _variant_chains(data: list[tuple], variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Every VU of ``data`` as a prover following the wrong rule ``variant`` computes it: the accumulator fed forward under
    the wrong rule from c_0 = 0.  Returns the wrong accumulators (n, STEPS) and a mask of the VUs on which the wrong rule
    could not form a witness at some step (its hints leave the relation's shape)."""
    n = len(data)
    A = np.stack([np.asarray(d[0], dtype=np.int64) for d in data])
    B = np.stack([np.asarray(d[1], dtype=np.int64) for d in data])
    c, bad = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
    accs = np.zeros((n, STEPS), dtype=np.int64)
    for s in range(STEPS):
        try:
            _, yv = unit_hints(c, A[:, s], B[:, s], variant=variant)
        except (AssertionError, ValueError):
            yv = np.zeros(n, dtype=np.int64)
            for i in range(n):                                    # isolate the offenders
                if bad[i]:
                    continue
                try:
                    _, yi = unit_hints(c[i:i + 1], A[i:i + 1, s], B[i:i + 1, s], variant=variant)
                    yv[i] = yi[0]
                except (AssertionError, ValueError):
                    bad[i] = True
        accs[:, s] = np.where(bad, 0, yv)
        c = accs[:, s]
    return accs, bad


def _variant_base(data: list[tuple], variant: str) -> tuple[list[tuple], int, int] | None:
    """A pair of VUs as a prover following the wrong rule ``variant`` would present them -- wrong accumulators fed forward,
    the wrong FINAL word as the claim -- with (VU index, first differing step) of the first VU whose final word differs from
    the model's (a finite one when there is one), or None when the wrong rule reaches the model's final word on every VU
    (then the claim is true and no chain-level negative exists; the unit tier exercises the rule)."""
    accs_v, bad = _variant_chains(data, variant)
    honest = np.stack([np.asarray(d[2], dtype=np.int64) for d in data])
    final_differs = (accs_v[:, -1] != honest[:, -1]) & ~bad
    finite = ((accs_v[:, -1] >> 23) & 0xFF) != 0xFF
    cands = np.nonzero(final_differs & finite)[0]
    if not len(cands):
        cands = np.nonzero(final_differs)[0]
    if not len(cands):
        return None
    i = int(cands[0])
    step = int(np.nonzero(accs_v[i] != honest[i])[0][0])
    j = next((k for k in range(len(data)) if k != i and not bad[k]), None)
    vus = [(data[i][0], data[i][1], [int(x) for x in accs_v[i]], int(accs_v[i, -1]))]
    if j is not None:
        vus.append((data[j][0], data[j][1], [int(x) for x in accs_v[j]], int(accs_v[j, -1])))
    return vus, i, step


def _pad_unit():
    return (0, np.zeros(WORDS, dtype=np.int64), np.zeros(WORDS, dtype=np.int64), 0)


class FP4ChainRunner:
    """``vu.ChainRunner`` for the NVFP4 unit: the same protocol (``prove`` / ``verify``) on ``compile_fp4_unit(chain=True)``."""

    hooks = FP4_HOOKS
    relation = "fp4-nvf4"

    def __init__(self, device: str, target: float, rate_log2: int = 2, n_proofs: int = 1, zk: bool = False,
                 mode: str = protocol_mod.MODE_INTERACTIVE):
        self.p = NVFP4
        self.zk, self.mode, self.device, self.target, self.rate_log2, self.n_proofs = zk, mode, device, target, rate_log2, n_proofs
        t0 = time.perf_counter()
        self.sys = compile_fp4_unit(self.p, chain=True)
        self.tb = tables(self.sys, self.device)
        self.compile_seconds = time.perf_counter() - t0
        self._cfg: dict[int, Config] = {}

    def cfg(self, l: int) -> Config:
        if l not in self._cfg:
            self._cfg[l] = config_for(l, self.target, self.rate_log2, self.n_proofs, zk=self.zk, mode=self.mode)
        return self._cfg[l]

    @property
    def interactive(self) -> bool:
        return self.mode == protocol_mod.MODE_INTERACTIVE

    def layout(self, n_vus: int, min_l: int = 256) -> Layout:
        l = max(min_l, 1 << (n_vus * STEPS - 1).bit_length())
        return Layout(l, n_vus, STEPS)

    def prove_vus(self, vus: list[tuple], check=True, mutate=None, min_l: int = 256, coins=None, variant: str | None = None):
        """``vus``: (A, B, accs, y_final) per VU.  The public claim per VU is its final FP32 word at its last column.
        ``variant``: hints generated under a WRONG rule (``witness.VARIANTS``; a negative), the claim and links honest."""
        if coins is None and self.interactive:
            coins = protocol_mod.Coins.sample()
        lay = self.layout(len(vus), min_l)
        units, y32 = [], torch.zeros(lay.l, dtype=torch.int64)
        for v, (A, B, accs, y) in enumerate(vus):
            units += vu_units(A, B, accs)
            y32[STEPS * v + STEPS - 1] = int(y)
        while len(units) < lay.l:
            units.append(_pad_unit())
        a = np.stack([np.asarray(u[1], dtype=np.int64) for u in units])
        b = np.stack([np.asarray(u[2], dtype=np.int64) for u in units])
        c_in = np.asarray([u[0] for u in units], dtype=np.int64)
        t0 = time.perf_counter()
        pub = _public_vectors_torch(self.p, None, a, b, None, device=self.device)
        if variant is None:
            hints_np = hints_fp4(self.sys, c_in, a, b)
        else:
            H, _ = unit_hints(c_in, a, b, variant=variant)
            hints_np = np.stack([H[n] for _, n in self.sys.hints])
        hints = torch.as_tensor(hints_np, dtype=torch.int64, device=self.device)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        hints_seconds = time.perf_counter() - t0
        proof = prove(self.p, self.sys, self.tb, self.cfg(lay.l), pub, hints, self.device, check=check, mutate=mutate,
                      chain=(lay, y32), words={"a": a, "b": b}, coins=coins, rel=FP4_HOOKS)
        proof.timings["hints_host"] = hints_seconds
        return proof, (a, b, y32), lay

    def verify_vus(self, proof, pubs, lay: Layout, coins=None):
        a, b, y32 = pubs
        if coins is None and self.interactive:
            coins = proof.coins
        return verify(self.p, self.sys, self.tb, self.cfg(lay.l), None, a, b, None, proof, self.device, chain=(lay, y32),
                      coins=coins, rel=FP4_HOOKS)


def _git_commit():
    from ..run import _git_commit as _git_commit_repo
    return _git_commit_repo() or os.environ.get("VERITY_COMMIT")


def _proof_class(cfg, interactive: bool):
    from ..vu import ZK_MODE, ZK_MODE_FS, ZK_MODE_HM96, ZK_VACUOUS_NOTE
    if cfg.zk and interactive:
        return ("COMPLETE_ZK_BACKEND", ZK_MODE + ZK_MODE_HM96 + ZK_VACUOUS_NOTE,
                "external authentication; unopened columns hidden by Blake3 of a uniform column")
    if cfg.zk:
        return ("COMPLETE_HVZK_BACKEND", ZK_MODE + ZK_MODE_FS + ZK_VACUOUS_NOTE,
                "external authentication; malicious-verifier ZK outside the programmable random-oracle model")
    return "NON_ZK_PROOF_DIAGNOSTIC", "none", "external authentication, ZK masking"


def _software(R, proof) -> dict:
    return {"backend": {"name": "b-ligero chain, fp4-nvf4 relation (backends/direct/ligero/fp4)", "commit": _git_commit(),
                        "hash": proof.hash_name, "challenge_expander": "SHAKE-256 (placeholder)"},
            "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}


def _security(cfg, snd, n_proofs: int, target: float) -> dict:
    return {"target": round(target), "field": "BabyBear", "challenge_coords": cfg.D, "rs_n": cfg.n, "rs_k": cfg.k, "rs_l": cfg.l,
            "zk": cfg.zk, "t_pad": cfg.t_pad, "mode": cfg.mode, "bound_model": snd["bound_model"],
            "hash_assumption": snd["hash_assumption"], "zk_statement": snd["zk_statement"],
            "fiat_shamir_queries_log2": snd["fiat_shamir_queries_log2"],
            "opened_columns": cfg.t, "achieved_log2": snd["total_log2"], "terms_log2": snd["terms_log2"],
            "per_proof_target_log2": snd["per_proof_target_log2"], "per_proof_total_log2": snd["per_proof_total_log2"],
            "union_over": n_proofs, "hash_placeholder": True}


def _instances(total_vus: int) -> dict:
    return {"dataset": contract.NVFP4_INSTANCES_DATASET, "tier": contract.NVFP4_INSTANCES_TIER, "range": [0, total_vus],
            "manifest_sha256": instances_digest(total_vus), "seed": INSTANCE_SEED,
            "recipe": f"fp4-nvf4 synthetic (edge families {', '.join(VU_FAMILIES)} cycling per VU; model-recorded accumulators)"}


# ---------------------------------------------------------------------------------------------------------------------
# Gate: honest VUs of every family accepted, a negative battery rejected (on the device); a contract-valid result
# ---------------------------------------------------------------------------------------------------------------------

def gate_vu_fp4(args) -> int:
    R = FP4ChainRunner(args.device, args.target, args.rate_log2, zk=args.zk, mode=args.mode)
    log = print
    log(f"fp4 chain system: {R.sys.summary()} compiled in {R.compile_seconds:.2f}s; device {args.device}")
    cen = census_fp4(R.sys)
    report = {"system": R.sys.summary(), "census": cen, "positives": [], "negatives": [], "unit_negatives": {}}
    failures = 0
    per_proof = args.batch // STEPS
    t0 = time.perf_counter()
    data = instances_fp4(args.vus)
    log(f"{args.vus} synthetic VUs ({len(VU_FAMILIES)} families, model-recorded accumulators) in {time.perf_counter() - t0:.1f}s")
    agg: dict[str, float] = {}
    honest_ok = 0
    for lo in range(0, args.vus, per_proof):
        vus = data[lo:lo + per_proof]
        proof, pubs, lay = R.prove_vus(vus, check=True)
        t1 = time.perf_counter()
        ok, why = R.verify_vus(proof, pubs, lay)
        vsec = time.perf_counter() - t1
        log(f"fp4 vu-k1536 [{lo}, {lo + len(vus)}): l={lay.l} prover {proof.timings['total']:.2f}s -> {ok} {why}")
        report["positives"].append({"range": [lo, lo + len(vus)], "l": lay.l, "ok": ok, "why": why, "timings": proof.timings,
                                    "verifier_seconds": vsec})
        for k_, v_ in proof.timings.items():
            agg[k_] = agg.get(k_, 0.0) + v_
        agg["verifier"] = agg.get("verifier", 0.0) + vsec
        honest_ok += len(vus) if ok else 0
        failures += not ok

    base = data[:2]
    negs = []
    proof, (a, b, y32), lay = R.prove_vus(base, check=False)
    # (1) wrong claimed words: every bit of the final FP32 word of VU 0 (sign, 8 exponent bits, 23 fraction bits)
    for bit in range(32):
        y_bad = y32.clone()
        y_bad[STEPS - 1] ^= 1 << bit
        negs.append((f"claimed word bit {bit} flipped", R.verify_vus(proof, (a, b, y_bad), lay)))
    # (2) a claimed word off the honest one by one ulp (the RZ normalisation's last place)
    for delta in (1, -1):
        y_bad = y32.clone()
        y_bad[STEPS - 1] = (int(y32[STEPS - 1]) + delta) % 2**32
        negs.append((f"claimed word {'+' if delta > 0 else '-'}1 ulp", R.verify_vus(proof, (a, b, y_bad), lay)))
    # (3) off-domain operands smuggled into the statement: NaN scale, padding bit, a nibble out of range
    for name, arr_name, pos, code in (("UE4M3 NaN scale 0x7f in a", "a", (3, K + 1), 0x7F), ("UE4M3 padding bit 0x80 in b", "b", (7, K), 0x80),
                                      ("E2M1 code 16 in a", "a", (5, 9), 16)):
        a_bad, b_bad = a.copy(), b.copy()
        (a_bad if arr_name == "a" else b_bad)[pos] = code
        negs.append((name, R.verify_vus(proof, (a_bad, b_bad, y32), lay)))
    # (4) an operand changed (the proof is of another statement): a code's sign bit, a scale's exponent bit
    a_bad = a.copy()
    a_bad[10, 1] ^= 0x08
    negs.append(("operand a[10][1] sign bit flipped", R.verify_vus(proof, (a_bad, b, y32), lay)))
    b_bad = b.copy()
    b_bad[11, K + 2] ^= 0x08
    negs.append(("scale b[11][2] exponent bit flipped", R.verify_vus(proof, (a, b_bad, y32), lay)))
    # (5) chain: broken links (each component), nonzero c_0 (each component), a linked accumulator bit
    c_rows = R.sys.chain["c"]
    for x, comp in enumerate(("s", "t", "f")):
        def break_link(W, row=c_rows[x]):
            W[row, 20] = (W[row, 20] + 1) % P
        proof2, pubs2, lay2 = R.prove_vus(base, check=False, mutate=break_link)
        negs.append((f"link c.{comp}[20] != y.{comp}[19]", R.verify_vus(proof2, pubs2, lay2)))

        def nonzero_c0(W, row=c_rows[x]):
            W[row, STEPS] = 1
        proof2, pubs2, lay2 = R.prove_vus(base, check=False, mutate=nonzero_c0)
        negs.append((f"c_0.{comp} != 0 at a chain start", R.verify_vus(proof2, pubs2, lay2)))

    def acc_bit(W):
        W[c_rows[2], 30] = (W[c_rows[2], 30] ^ (1 << 12)) % P
    proof2, pubs2, lay2 = R.prove_vus(base, check=False, mutate=acc_bit)
    negs.append(("accumulator fraction bit 12 flipped in a linked unit", R.verify_vus(proof2, pubs2, lay2)))
    # (6) wrong-rule witnesses, own claim: a prover following a WRONG model line feeds its accumulator forward and claims
    #     ITS final word -- on a VU where that word differs from the model's (where it agrees the claim is true, and a
    #     witness whose hints differ only in bits the relation ignores, e.g. the sign of a zero, is rightly accepted).
    not_exercised = []
    for v in VARIANTS:
        vb = _variant_base(data, v)
        if vb is None:
            not_exercised.append(v)
            continue
        vus_v, vu_i, step = vb
        name = f"wrong-rule witness, own claim: {v} (VU {vu_i}, first differing step {step})"
        try:
            proof2, pubs2, lay2 = R.prove_vus(vus_v, check=False, variant=v)
            negs.append((name, R.verify_vus(proof2, pubs2, lay2)))
        except (AssertionError, ValueError) as e:          # no witness in the relation's shape / the claim is off-domain
            negs.append((name, (False, f"rejected before the linear test: {type(e).__name__}: {str(e)[:80]}")))
    if not_exercised:
        log(f"wrong-rule variants that reach the model's final word on every synthetic VU (no chain-level negative; the unit "
            f"tier exercises them): {not_exercised}")
    report["variants_not_exercised_at_chain_level"] = not_exercised
    # (7) one witness row per (gadget scope, row class), as gate_vu does
    seen, picks = set(), []
    for r in R.sys.rows:
        key = (".".join(r.name.split(".")[:3]), r.kind)
        if key not in seen:
            seen.add(key)
            picks.append(r)
    rng = np.random.default_rng(11)
    picks = picks[: max(0, getattr(args, "row_negatives", 60))]
    for r in picks:
        col = int(rng.integers(0, 2 * STEPS))
        delta = int(rng.integers(1, 3))

        def mutate(W, idx=r.idx, col=col, delta=delta):
            W[idx, col] = (W[idx, col] + delta) % P
        proof2, pubs2, lay2 = R.prove_vus(base, check=False, mutate=mutate)
        negs.append((f"row {r.name} ({r.kind}) +{delta} in column {col}", R.verify_vus(proof2, pubs2, lay2)))
    for name, (ok, why) in negs:
        report["negatives"].append({"name": name, "accepted": ok, "why": why})
        if ok:
            log(f"!! NEGATIVE ACCEPTED: {name} -- soundness bug")
            failures += 1
    n_neg_ok = sum(not r["accepted"] for r in report["negatives"])
    log(f"negatives: {n_neg_ok}/{len(report['negatives'])} rejected (l={lay.l})")
    # (8) unit-level wrong-rule witnesses on the device: every unit whose wrong word differs from the model's is rejected
    #     by the constraint tables (the prover's self-check, no proof) -- the tu-k64-neg tier
    unit_neg = _unit_negatives(R, args)
    report["unit_negatives"] = unit_neg
    tu_ok, tu_tot = unit_neg["rejected"], unit_neg["differing"]
    failures += tu_tot - tu_ok
    log(f"unit negatives (wrong-rule witnesses, own claim): {tu_ok}/{tu_tot} rejected over {unit_neg['units']} units x {len(VARIANTS)} variants")
    report["failures"] = failures
    detail = f"vu-k1536 {honest_ok}/{args.vus} ok; vu-k1536-neg {n_neg_ok}/{len(report['negatives'])} ok; tu-k64-neg {tu_ok}/{tu_tot} ok"
    log(detail)
    if args.out:
        result = _gate_result(args, R, cen, agg, honest_ok, detail, report, failures)
        Path(os.path.expandvars(args.out)).write_text(json.dumps(result, indent=1, default=str))
    return 2 if failures else 0


def _unit_negatives(R: FP4ChainRunner, args) -> dict:
    """The unit-mode system on the device: for every wrong rule, the witness it generates with its OWN claimed word; count
    the units whose word differs from the model's and how many of them the constraints reject (all of them, or a bug)."""
    from ..witness import build_witness, tables as _tables
    from .relation import compile_fp4_unit as _compile
    from .witness import oracle_words
    s = _compile(R.p, chain=False)
    tb = _tables(s, R.device)
    rng = np.random.default_rng(getattr(args, "unit_seed", 5))
    per = max(1, getattr(args, "unit_negatives", 48))
    cs, as_, bs = [], [], []
    for fam in FAMILIES:
        c, a, b = random_units(rng, per, fam)
        cs.append(c); as_.append(a); bs.append(b)
    c, a, b = np.concatenate(cs), np.concatenate(as_), np.concatenate(bs)
    y = oracle_words(c, a, b)
    out = {"units": int(len(c)), "differing": 0, "rejected": 0, "by_variant": {}}
    for v in VARIANTS:
        H, yv = unit_hints(c, a, b, variant=v)
        nonfin = ((yv >> 23) & 0xFF) == 0xFF
        diff = (yv != y)
        n_diff = int(diff.sum())
        rej = int((diff & nonfin).sum())                       # rejected by the verifier's decode itself
        keep = diff & ~nonfin
        if keep.any():
            idx = np.nonzero(keep)[0]
            pub = _public_vectors_torch(R.p, c[idx], a[idx], b[idx], yv[idx], device=R.device)
            hints = torch.as_tensor(np.stack([H[n][idx] for _, n in s.hints]), dtype=torch.int64, device=R.device)
            W = build_witness(s, pub, hints, R.device)
            bad_cols = _violating_columns(s, tb, W, pub)
            rej += int(len(bad_cols))
        out["by_variant"][v] = {"differing": n_diff, "rejected": rej}
        out["differing"] += n_diff
        out["rejected"] += rej
    return out


def _violating_columns(s, tb, W, pub) -> set:
    """Columns (units) at which at least one constraint fails (torch)."""
    from ..compile import Expr
    from ..witness import eval_expr
    bad: set[int] = set()
    for lin in s.linear:
        v = eval_expr(Expr(None, lin.lin, lin.c), W)
        tgt = (pub[lin.pub] % P) if lin.pub else torch.zeros_like(v)
        bad.update(int(i) for i in torch.nonzero(v != tgt).flatten().tolist())
    for q in s.quadratic:
        v = (eval_expr(q.a, W) * eval_expr(q.b, W)) % P
        bad.update(int(i) for i in torch.nonzero(v != eval_expr(q.c, W)).flatten().tolist())
    return bad


def _gate_result(args, R, cen, agg, honest_ok, detail, report, failures) -> dict:
    """The gate as a contract-valid ``bench-result/v1``: the honest sweep's timings (sum over its proofs) and, in
    ``validation.detail``, the tier counts ``verity.verification.trust`` reads (criterion Tc)."""
    interactive = R.interactive
    lay = R.layout(min(args.vus, args.batch // STEPS))
    cfg = R.cfg(lay.l)
    n_proofs = -(-args.vus // (args.batch // STEPS))
    snd = soundness(cfg, R.tb.L, R.tb.Q, n_proofs)
    proof_class, zk_mode, omitted = _proof_class(cfg, interactive)
    total = agg.get("total", 0.0) + agg.get("hints_host", 0.0)
    hw = contract.probe_hardware()
    fp = contract.fingerprint(
        profile=NVFP4_SM120.name, domain=contract.DOMAIN, K=K * STEPS, B=args.vus, N_subbatches=n_proofs,
        proof_class=proof_class, authentication="excluded", zk_mode=zk_mode,
        security=_security(cfg, snd, n_proofs, args.target), hardware=hw,
        software={"backend": {"name": "b-ligero chain, fp4-nvf4 relation (backends/direct/ligero/fp4)", "commit": _git_commit()},
                  "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}},
        preprocessing=f"constraint compilation (a function of NVFP4Params only, {R.compile_seconds:.2f}s)", retention="retained",
        instances=_instances(args.vus), scope="vu", omitted=omitted,
        relation=RELATION_NAME, model=MODEL_NAME, instruction=NVFP4_SM120.instruction, commit=_git_commit(), step_k=K, steps_per_vu=STEPS,
        rows=R.sys.m, committed_rows=n_rows(cfg, R.sys.m, True), linear_constraints=R.tb.L, quadratic_constraints=R.tb.Q,
        census=cen, checker="prove+verify (b-ligero chain) + constraint tables on wrong-rule witnesses",
        witness_generation="numpy hint generator (fp4/witness.py, one line per constraint) + torch program (derived rows)",
        B_proof=args.batch // STEPS,
        negatives={"chain": [n["name"] for n in report["negatives"]], "unit_variants": report["unit_negatives"].get("by_variant")},
        note=f"gate: {args.vus} honest VUs of {len(VU_FAMILIES)} edge families proved and verified as {n_proofs} proofs; "
             f"{len(report['negatives'])} chain-level negatives; unit-level wrong-rule witnesses on {report['unit_negatives'].get('units')} units",
    )
    meas = [
        contract.measurement("t.witness", agg.get("witness", 0.0) + agg.get("hints_host", 0.0)),
        contract.measurement("t.encoding_commitment", agg.get("encode", 0.0) + agg.get("transfer", 0.0) + agg.get("merkle", 0.0)),
        contract.measurement("t.arithmetic", agg.get("tests", 0.0)),
        contract.measurement("t.lookup", 0.0),
        contract.measurement("t.zk_additional", agg.get("zk_masks", 0.0)),
        contract.measurement("t.serialization", agg.get("openings", 0.0) + agg.get("statement", 0.0)),
        contract.measurement("t.total", total),
        contract.measurement("verifier.seconds", agg.get("verifier", 0.0)),
        contract.measurement("vu.count", args.vus, "count"),
        contract.measurement("vu.seconds_per_vu", total / args.vus, "s"),
        *contract.rate_measurements(total, args.vus, K * STEPS, NVFP4_SM120.name),
        contract.measurement("soundness_bits", -snd["total_log2"], "bits"),
        contract.measurement("rows_per_unit", R.sys.m, "count"),
        contract.measurement("units_per_vu", STEPS, "count"),
        contract.measurement("gate.honest_vus_ok", honest_ok, "count"),
        contract.measurement("gate.chain_negatives", len(report["negatives"]), "count"),
        contract.measurement("gate.chain_negatives_rejected", sum(not n["accepted"] for n in report["negatives"]), "count"),
        contract.measurement("gate.unit_negatives", report["unit_negatives"].get("differing", 0), "count"),
        contract.measurement("gate.unit_negatives_rejected", report["unit_negatives"].get("rejected", 0), "count"),
    ]
    for k_, v_ in cen.items():
        if isinstance(v_, int) and not isinstance(v_, bool):
            meas.append(contract.measurement(f"relation.census.{k_}", v_, "count"))
    result = {"schema": contract.RESULT_SCHEMA,
              "run_id": getattr(args, "run_id", None) or os.environ.get("RESEARCH_RUN_ID") or f"b-ligero-fp4-gate-{int(time.time())}",
              "workload_fingerprint": fp,
              "validation": {"status": "passed" if not failures else "failed",
                             "checker": "backends.direct.ligero.fp4.chain gate_vu_fp4 (prove+verify every honest sub-batch with the prover's "
                                        "self-check; chain negatives through verify; wrong-rule unit witnesses through the constraint tables)",
                             "detail": detail, "evidence": {"soundness": snd, "negatives": report["negatives"], "unit_negatives": report["unit_negatives"]}},
              "measurements": meas, "measurement_files": [], "artifacts": []}
    problems = contract.validate(result)
    if problems:
        print("contract problems:", problems)
    return result


# ---------------------------------------------------------------------------------------------------------------------
# Bench: the same result envelope as vu.bench_vu / fp8.chain.bench_vu_fp8, with --dump-dir proof files
# ---------------------------------------------------------------------------------------------------------------------

def bench_vu_fp4(args) -> int:
    per_proof = args.batch // STEPS
    total_vus = args.total_vus or per_proof
    n_proofs = -(-total_vus // per_proof)
    interactive = args.mode == protocol_mod.MODE_INTERACTIVE
    depth = 3 if interactive else 1
    R = FP4ChainRunner(args.device, args.target, args.rate_log2, n_proofs=n_proofs, zk=args.zk, mode=args.mode)
    cen = census_fp4(R.sys)
    print("census:", json.dumps(cen))
    subs = [(lo, min(lo + per_proof, total_vus)) for lo in range(0, total_vus, per_proof)]
    t0 = time.perf_counter()
    data = instances_fp4(total_vus)
    print(f"{total_vus} synthetic VUs ({len(VU_FAMILIES)} families, model-recorded accumulators) in {time.perf_counter() - t0:.1f}s")
    lay = R.layout(per_proof)
    l = lay.l
    cfg = R.cfg(l)
    print(f"config {cfg}; layout {lay}; zk {cfg.zk}, mode {cfg.mode}; system {R.sys.summary()}; {total_vus} VUs as {n_proofs} sub-batches of <= {per_proof}")
    runs = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    keys = None
    dump_dir = Path(os.path.expandvars(args.dump_dir)) if getattr(args, "dump_dir", None) else None
    dump_manifest = None
    if dump_dir is not None:
        from ..serialize import _manifest, _statement_of, _write_system, proof_bytes, sha256_file, statement_bytes, verify_files
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_manifest = _manifest(dump_dir, args, R)
        _write_system(dump_dir, R, dump_manifest)
        dump_manifest["set"] = {"total_vus": total_vus, "batch": args.batch, "per_proof": per_proof, "n_proofs": n_proofs,
                                "zk": args.zk, "mode": args.mode, "reps": args.reps, "relation": R.relation,
                                "instances": _instances(total_vus)}
    dump_reps = getattr(args, "dump_reps", None)
    dump_reps = args.reps if dump_reps is None or dump_reps < 0 else dump_reps
    for rep in range(args.reps + 1):
        agg: dict[str, float] = {}
        vsum = 0.0
        t_pass = time.perf_counter()
        # Prove every sub-batch of the rep first, then verify (and dump) them: the verifier's host work and the file
        # round trips must not sit between two timed prover sub-batches (vu.bench_vu / fp8.chain discipline).
        proved = []
        for si, (lo, hi) in enumerate(subs):
            V = protocol_mod.Coins.sample() if interactive else None
            proof, pubs, lay_i = R.prove_vus(data[lo:hi], check=(rep == 0 and lo == 0), min_l=l, coins=V)
            for k_, v_ in proof.timings.items():
                agg[k_] = agg.get(k_, 0.0) + v_
            proved.append((si, lo, hi, V, proof, pubs, lay_i))
            if rep == 0:
                break                                        # warm-up: one sub-batch (compile, graph capture, self-check)
        wall_prove = time.perf_counter() - t_pass
        for si, lo, hi, V, proof, pubs, lay_i in proved:
            t1 = time.perf_counter()
            ok, why = R.verify_vus(proof, pubs, lay_i, coins=V)
            vsum += time.perf_counter() - t1
            assert ok, f"sub-batch [{lo}, {hi}): {why}"
            if dump_dir is not None and 0 < rep <= dump_reps:
                rd = dump_dir / f"rep{rep}"
                rd.mkdir(exist_ok=True)
                stem = f"sub_{si:02d}"
                st = _statement_of(R, pubs, lay_i, R.cfg(lay_i.l), n_proofs)
                (rd / f"{stem}.stmt").write_bytes(statement_bytes(st))
                (rd / f"{stem}.proof").write_bytes(proof_bytes(proof))
                # the Python verifier on the FILES (the self-check a file-level verifier is compared against), outside the timings
                ok_f, why_f, secs_f = verify_files(rd / f"{stem}.stmt", rd / f"{stem}.proof", args.device, runner=R)
                entry = {"rep": rep, "sub": si, "vus": [lo, hi], "stmt": f"rep{rep}/{stem}.stmt", "proof": f"rep{rep}/{stem}.proof",
                         "stmt_sha256": sha256_file(rd / f"{stem}.stmt"),
                         "proof_bytes": (rd / f"{stem}.proof").stat().st_size, "proof_sha256": sha256_file(rd / f"{stem}.proof"),
                         "python_verdict": {"accepted": ok, "reason": why},
                         "python_file_verdict": {"accepted": ok_f, "reason": why_f, "seconds": secs_f},
                         "prover_seconds": proof.timings.get("total")}
                assert ok_f, f"file-level Python verifier rejected rep{rep}/{stem}: {why_f}"
                if V is not None:
                    (rd / f"{stem}.coins").write_bytes(V.r1 + V.s1 + V.r2 + V.s2)
                    entry["coins"] = f"rep{rep}/{stem}.coins"
                dump_manifest["files"].append(entry)
        del proved
        if rep == 0:
            print("warm-up (one sub-batch, with the prover's self-check):", {k_: round(v_, 3) for k_, v_ in agg.items()})
            continue
        keys = agg.keys()
        runs.append((agg, vsum))
        print(f"rep {rep}: prover {agg['total']:.3f}s (+hints {agg['hints_host']:.3f}s) over {n_proofs} proofs; "
              f"verifier {vsum:.3f}s; prove wall {wall_prove:.3f}s ({(agg['total'] + agg['hints_host']) / total_vus * 1e3:.3f} ms/VU)", flush=True)
    if dump_dir is not None:
        (dump_dir / "manifest.json").write_text(json.dumps(dump_manifest, indent=1, default=str))
        print(f"dumped {len(dump_manifest['files'])} proofs to {dump_dir}", flush=True)
    med = {k_: statistics.median(r[0][k_] for r in runs) for k_ in keys}
    vmed = statistics.median(r[1] for r in runs)
    peak_dev = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    nbytes = {k_: v_ * n_proofs for k_, v_ in proof.n_bytes().items()}
    snd = soundness(cfg, R.tb.L, R.tb.Q, n_proofs)
    total = med["total"] + med["hints_host"]
    per_vu = total / total_vus
    hw = contract.probe_hardware()
    proof_class, zk_mode, omitted = _proof_class(cfg, interactive)
    fp = contract.fingerprint(
        profile=NVFP4_SM120.name, domain=contract.DOMAIN, K=K * STEPS, B=total_vus, N_subbatches=n_proofs,
        proof_class=proof_class, authentication="excluded", zk_mode=zk_mode,
        security=_security(cfg, snd, n_proofs, args.target), hardware=hw, software=_software(R, proof),
        preprocessing=f"constraint compilation (a function of NVFP4Params only, {R.compile_seconds:.2f}s)",
        retention="retained",
        # the frozen NVFP4 set (contract.NVFP4_INSTANCES_*): the canonical tables accept exactly this dataset/tier at
        # range [0, 4096) with manifest_sha256 == contract.NVFP4_INSTANCES_DIGEST == instances_digest(4096)
        instances=_instances(total_vus),
        scope="vu", omitted=omitted,
        relation=RELATION_NAME, model=MODEL_NAME, instruction=NVFP4_SM120.instruction, commit=_git_commit(), step_k=K, steps_per_vu=STEPS,
        rows=R.sys.m, committed_rows=n_rows(cfg, R.sys.m, True), linear_constraints=R.tb.L, quadratic_constraints=R.tb.Q,
        census=cen,
        witness_generation="numpy hint generator (fp4/witness.py, one line per constraint) + torch program (derived rows)",
        B_proof=per_proof,
        proof_dumps=None if dump_dir is None else {"dir": str(dump_dir), "files": len(dump_manifest["files"]),
                                                   "formats": dump_manifest["format"], "system_sha256": dump_manifest["system_file"]["sha256"]},
        note=f"{total_vus} VUs as {n_proofs} independently committed proofs of <= {per_proof} VUs each (l = {l} columns, "
             f"{l - per_proof * STEPS} pad columns per proof); prover time = sum over the proofs; soundness = union bound; "
             f"public claim per VU = the final FP32 word, compared as three bit-field components (sign, exponent, fraction)",
    )
    meas = [
        contract.measurement("t.witness", med["witness"] + med["hints_host"]),
        contract.measurement("t.encoding_commitment", med["encode"] + med["transfer"] + med["merkle"]),
        contract.measurement("t.arithmetic", med["tests"]),
        contract.measurement("t.lookup", 0.0),
        contract.measurement("t.zk_additional", med.get("zk_masks", 0.0)),
        contract.measurement("t.serialization", med["openings"] + med.get("statement", 0.0)),
        contract.measurement("t.total", total),
        contract.measurement("verifier.seconds", vmed),
        contract.measurement("transcript.bytes", nbytes["total"]),
        contract.measurement("rounds.protocol", depth),
        contract.measurement("rounds.sequential_depth", depth),
        contract.measurement("mem.peak_device_bytes", peak_dev),
        contract.measurement("census.witness_elements", R.sys.m * l * n_proofs),
        contract.measurement("census.committed_elements", R.sys.m * cfg.n * n_proofs),
        contract.measurement("census.bytes_encoded", R.sys.m * cfg.n * 4 * n_proofs),
        contract.measurement("vu.seconds_per_vu", per_vu, "s"),
        contract.measurement("vu.count", total_vus, "count"),
        *contract.rate_measurements(total, total_vus, K * STEPS, NVFP4_SM120.name),
        contract.measurement("split.subbatches", n_proofs, "count"),
        contract.measurement("split.per_proof_vus", per_proof, "count"),
        contract.measurement("split.hints_seconds", med["hints_host"], "s"),
        contract.measurement("split.witness_torch_seconds", med["witness"], "s"),
        contract.measurement("split.encode_seconds", med["encode"], "s"),
        contract.measurement("split.transfer_seconds", med["transfer"], "s"),
        contract.measurement("split.merkle_seconds", med["merkle"], "s"),
        contract.measurement("split.tests_seconds", med["tests"], "s"),
        contract.measurement("split.openings_seconds", med["openings"], "s"),
        contract.measurement("proof.opened_columns_bytes", nbytes["opened_columns"], "B"),
        contract.measurement("proof.merkle_paths_bytes", nbytes["merkle_paths"], "B"),
        contract.measurement("proof.w_h_q_bytes", nbytes["w"] + nbytes["h"] + nbytes["q_chain"] + nbytes["v_linear"], "B"),
        contract.measurement("split.zk_masks_seconds", med.get("zk_masks", 0.0), "s"),
        contract.measurement("prove_wall", total, "s"),
        contract.measurement("verify_wall", vmed, "s"),
        contract.measurement("proof_bytes", nbytes["total"], "B"),
        contract.measurement("depth", depth, "count"),
        contract.measurement("soundness_bits", -snd["total_log2"], "bits"),
        contract.measurement("rows_per_unit", R.sys.m, "count"),
        contract.measurement("units_per_vu", STEPS, "count"),
    ]
    for k_, v_ in cen.items():
        if isinstance(v_, int) and not isinstance(v_, bool):
            meas.append(contract.measurement(f"relation.census.{k_}", v_, "count"))
    result = {"schema": contract.RESULT_SCHEMA, "run_id": args.run_id or os.environ.get("RESEARCH_RUN_ID") or f"b-ligero-fp4-vu-{int(time.time())}",
              "workload_fingerprint": fp,
              "validation": {"status": "passed", "checker": "backends.direct.ligero.fp4.chain bench_vu_fp4 (verifier accepted every rep; "
                                                            "prover self-check on the warm-up sub-batch)",
                             "detail": f"{args.reps} reps, {total_vus} VUs as {n_proofs} proofs at l={l}", "evidence": {"soundness": snd}},
              "measurements": meas, "measurement_files": [], "artifacts": []}
    problems = contract.validate(result)
    if problems:
        print("contract problems:", problems)
    print(json.dumps({"seconds_per_vu": per_vu, "prover_total": total, "verifier": vmed, "bytes": nbytes, "n_proofs": n_proofs,
                      "per_proof_vus": per_proof, "t": cfg.t, "zk": cfg.zk, "k": cfg.k, "peak_device_bytes": peak_dev,
                      "soundness_log2": snd["total_log2"], "per_proof_log2": snd["per_proof_total_log2"], "rows": R.sys.m,
                      "medians": med}, indent=1))
    if args.out:
        Path(os.path.expandvars(args.out)).write_text(json.dumps(result, indent=1, default=str))   # $RESEARCH_RUN_DIR/result.json
    return 0
