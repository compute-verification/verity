"""Price of privacy: what the honest-verifier and the malicious-verifier zero-knowledge layers add to every design
(zk-construction lane, 2026-09-22; the construction is ``backends/ZK_CONSTRUCTION.md``).

Every number is a cost-model ESTIMATE on top of ``explore.tensor`` (limb MACs, hash bytes, PRG bytes at the module's
stated rates) and ``security.accounting`` (rows, pads, coins, depth).  Three privacy levels per design
(``explore.tensor.ZK_LEVELS``):

* ``none``      -- the non-ZK diagnostic a kernel measurement corresponds to: payload ``k`` per Ligero row, no
                   blinding rows, no transcript pads, plain Merkle leaves, no verifier-coin commitment;
* ``hvzk``      -- the protocol documents' layer: ``k - l`` randomizers per row, 3 blinding rows per tableau, one
                   committed pad per transcript element (A-GKR), HM96 leaf commitments (statistical hiding
                   ``n 2^-160``), prover randomness from a PRG.  ``COMPLETE_HVZK_BACKEND`` when implemented;
* ``malicious`` -- ``hvzk`` plus step 0: the verifier commits (HM96, ``k_V = 160``) to every coin before the first
                   prover message and opens each on the challenge message it replaces; +1 sequential depth,
                   ``R`` commitments (``rounds.protocol``) verified by the prover.  ``COMPLETE_ZK_BACKEND`` when
                   implemented and red-team reviewed.

Designs: A-GKR (GKR + logUp-GKR + Ligero) in its two hint-free branches; B-Ligero as direct Ligero (the
``B`` row of tensor.md); B-Ligero as AIR/FRI (the current Plonky3 implementation's shape: randomised trace,
selector column, blinding codeword, salted leaves; :func:`air_fri_zk`, relative deltas only because tensor.md has
no absolute B-AIR row); the VOLE candidate (QuickSilver, ``explore.vole``) as the comparison row whose privacy is
native (designated verifier; malicious-verifier ZK in the sVOLE-hybrid model; nothing to add).

Overhead deltas are stated at the H100 realistic utilisation of tensor.md (0.172, TensorZKP) and, for A, at the
``tc_sumcheck_4090`` re-pricing (model-equivalent tensor-bucket utilisation 0.0143), Blake3 at 50 GB/s.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from fractions import Fraction

from verity_numerical.explore import fields as fl
from verity_numerical.explore import pcs
from verity_numerical.explore import tensor as t
from verity_numerical.security import accounting as acc

__all__ = [
    "AIR_LAYOUTS",
    "TC_SUMCHECK_4090_UTILISATION",
    "air_fri_zk",
    "design_zk_rows",
    "fiat_shamir_reparameterisation",
    "main",
    "report",
    "soundness_with_zk",
    "vole_row",
]

#: ``tc_sumcheck_4090.md``: model-equivalent utilisation of the tensor bucket measured on the RTX 4090 (peak-ratio
#: scaled to the H100 in that report; an HBM-bound kernel does not scale with peak, so this is a labelled point).
TC_SUMCHECK_4090_UTILISATION = 0.0143
HASH_RATE = t.HASH_RATES["blake3_simt"]["bytes_per_s"]
UTILISATIONS = (0.172, TC_SUMCHECK_4090_UTILISATION)

#: B-AIR trace layouts (base-field columns per unit row).  ``plonky3_current``: ``archive/direct/README.md`` (2,794
#: columns: range checks as 2-bit limbs, fixed tables as one-hot selectors; no lookup argument, ``MAX_DEG = 5``).
#: ``census_lookup``: the census layout with a logUp helper column per lookup (1,064 witness slots + 306 helpers in
#: ``F_c``), the shape an AIR with a lookup argument would have (degree 3 helper constraints; taken at ``MAX_DEG = 5``).
AIR_LAYOUTS: dict[str, dict] = {
    "plonky3_current": {"base_columns": 2794, "helper_columns_fc": 0, "constraint_degree": 5,
                        "basis": "archive/direct/README.md layout census (uni-stark 0.4.3, Goldilocks, log_blowup=2)"},
    "census_lookup": {"base_columns": 1064, "helper_columns_fc": 306, "constraint_degree": 5,
                      "basis": "security.census: 1,064 committed slots + 306 logUp helpers per unit"},
}


def _bucket_delta(base: dict, other: dict) -> dict:
    keys = ("t.matmul", "t.simt_field", "t.hash", "t.hint_gen", "t.prg", "t.total")
    out = {k: other[k] - base[k] for k in keys}
    out["overhead_delta"] = other["overhead.vs_native_peak"] - base["overhead.vs_native_peak"]
    out["relative"] = other["t.total"] / base["t.total"] - 1.0
    return out


def design_zk_rows(design_name: str, field: str, target_log2: int = -128, utilisations: Sequence[float] = UTILISATIONS,
                   gpu: str = "H100") -> dict:
    """One Ligero-committed design (A, A-hintfree, B) on one field: the three privacy levels' per-VU census
    (committed / encoded / hashed / PRG / communication bytes, rows, depth) and the bucket-wise time deltas of
    ``hvzk`` and ``malicious`` over ``none`` at each utilisation."""
    design, fs = t.DESIGNS[design_name], t.FIELDS[field]
    g = t.GPUS[gpu]
    levels = {}
    for zk in t.ZK_LEVELS:
        p = t.price_design(design, fs, target_log2, zk=zk)
        levels[zk] = {
            "priced": p,
            "committed_elements_per_vu": p.committed_elements / t.B_TARGET,
            "committed_bytes_per_vu": p.committed_bytes / t.B_TARGET,
            "encoded_bytes_per_vu": p.encoded_bytes / t.B_TARGET,
            "hash_bytes_per_vu": p.hash_bytes / t.B_TARGET,
            "prg_bytes_per_vu": p.prg_bytes / t.B_TARGET,
            "communication_bytes_per_vu": p.communication_bytes / t.B_TARGET,
            "ligero_rows": p.rows, "rounds.sequential_depth": p.depth,
            "limb_macs_per_vu": p.matmul_total / t.B_TARGET,
            "times": {f"{gpu}@{u:g}": p.times(g, gpu, u, HASH_RATE, fs=fs) for u in utilisations},
        }
    base = levels["none"]
    out = {"design": design_name, "label": design.label, "field": field, "target_log2": target_log2,
           "extension_degree": base["priced"].extension_degree, "B": t.B_TARGET, "K": t.K_TARGET, "gpu": gpu,
           "hash_rate_bytes_per_s": HASH_RATE, "levels": {}, "deltas": {}}
    for zk, lv in levels.items():
        out["levels"][zk] = {k: v for k, v in lv.items() if k != "priced"}
        out["levels"][zk]["notes"] = lv["priced"].notes
    for zk in ("hvzk", "malicious"):
        lv = levels[zk]
        d = {
            "committed_elements_per_vu": lv["committed_elements_per_vu"] - base["committed_elements_per_vu"],
            "encoded_bytes_per_vu": lv["encoded_bytes_per_vu"] - base["encoded_bytes_per_vu"],
            "hash_bytes_per_vu": lv["hash_bytes_per_vu"] - base["hash_bytes_per_vu"],
            "prg_bytes_per_vu": lv["prg_bytes_per_vu"] - base["prg_bytes_per_vu"],
            "communication_bytes_per_vu": lv["communication_bytes_per_vu"] - base["communication_bytes_per_vu"],
            "limb_macs_per_vu": lv["limb_macs_per_vu"] - base["limb_macs_per_vu"],
            "ligero_rows": lv["ligero_rows"] - base["ligero_rows"],
            "rounds.sequential_depth": lv["rounds.sequential_depth"] - base["rounds.sequential_depth"],
            "times": {k: _bucket_delta(base["times"][k], lv["times"][k]) for k in lv["times"]},
        }
        out["deltas"][zk] = d
    return out


def air_fri_zk(layout: str = "census_lookup", field: str = "goldilocks", target_log2: int = -128, B: int = t.B_TARGET,
               log_blowup: int = 2, regime: str = "proven_udr") -> dict:
    """The ZK layer of B-Ligero as AIR/FRI (relative deltas; tensor.md has no absolute B-AIR row).

    Trace: one row per unit, ``U = 96 B`` real rows in a domain of ``2^ceil(log2 U)`` rows; LDE at rate
    ``2^-log_blowup``; FRI queries ``s`` from ``explore.pcs.fri_queries`` at the target for ``rows_log2 + 2``
    field-term applications (alpha, DEEP zeta, one beta per fold).  Randomised trace: ``revealed`` random rows per
    column (DEEP ``zeta, g zeta`` + 2 points per query), one ``is_real`` selector column, one blinding codeword
    (``d`` base columns), quotient chunks doubled by the selector's degree, 32-byte salts on the leaves (or HM96
    leaves).  Non-interactive Plonky3 (Fiat-Shamir) is honest-verifier ZK = ZK in the random-oracle model at depth 1;
    the interactive variant has depth ``rows_log2 + 3`` and step 0 adds one."""
    lay = AIR_LAYOUTS[layout]
    fs = t.FIELDS[field]
    d = t.extension_degree(fs, target_log2, "B")
    Fc = fs.option.trace.extension(d)
    units = B * t.UNITS_PER_VU
    rows_log2 = math.ceil(math.log2(units))
    N = 2**rows_log2
    n = N * 2**log_blowup
    rate = Fraction(1, 2**log_blowup)
    fri_rounds = rows_log2
    s, _m, terms = pcs.fri_queries(target_log2, rate, regime, n, Fc, applications=fri_rounds + 2)
    req = acc.fri_zk_requirements(rows_log2, units, s, ood_points=2, arity_log2=1,
                                  constraint_degree=lay["constraint_degree"], F=Fc)
    cols_base = lay["base_columns"] + lay["helper_columns_fc"] * d
    quotient_cols_before = req["quotient_chunks_before"] * d
    quotient_cols_after = req["quotient_chunks_after"] * d
    extra_cols = req["selector_columns"] + req["blinding_codewords"] * d + (quotient_cols_after - quotient_cols_before)
    eb = fs.element_bytes
    committed_before = (cols_base + quotient_cols_before) * N
    committed_after = committed_before + extra_cols * N
    encoded_before = committed_before * 2**log_blowup * eb
    encoded_after = committed_after * 2**log_blowup * eb
    # Merkle leaves are LDE rows (all columns of one row); 2 trees (trace, quotient) + the FRI fold trees
    fold_bytes = sum((n >> i) * d * eb + 2 * (n >> i) * acc.DIGEST_BYTES for i in range(1, fri_rounds))
    hash_before = encoded_before + 2 * 2 * n * acc.DIGEST_BYTES + fold_bytes
    salt_bytes = 2 * n * acc.DIGEST_BYTES               # 32 B salt per leaf of both trees
    hash_after = encoded_after + 2 * 2 * n * acc.DIGEST_BYTES + fold_bytes + salt_bytes
    prg_bytes = req["random_rows_needed"] * cols_base * eb + N * d * eb + salt_bytes
    lookup_tables = len(acc.CENSUS_TABLES) if lay["helper_columns_fc"] else 0
    coins = lookup_tables + 2 + fri_rounds + 1          # z_T tuples, alpha, zeta, betas, query set
    depth_interactive = 3 + fri_rounds                  # z -> com_2/alpha -> zeta ... betas -> queries (helper round adds 1)
    return {
        "design": "B-AIR", "layout": layout, "layout_basis": lay["basis"], "field": field, "extension_degree": d,
        "target_log2": target_log2, "B": B, "K": t.K_TARGET,
        "trace": {"real_rows": units, "rows_log2": rows_log2, "lde_n": n, "rate": str(rate), "base_columns": cols_base,
                  "quotient_columns_before": quotient_cols_before, "quotient_columns_after": quotient_cols_after},
        "fri": {"queries": s, "regime": regime, "rounds": fri_rounds, "field_term_log2": acc.log2(terms["field"]),
                "query_term_log2": acc.log2(terms["query"])},
        "zk_requirements": {k: (str(v) if isinstance(v, Fraction) else v) for k, v in req.items()},
        "extra_columns": {"selector": req["selector_columns"], "blinding_codeword": req["blinding_codewords"] * d,
                          "quotient_chunk_doubling": quotient_cols_after - quotient_cols_before, "total": extra_cols},
        "per_vu": {
            "committed_elements_none": committed_before / B, "committed_elements_hvzk": committed_after / B,
            "encoded_bytes_none": encoded_before / B, "encoded_bytes_hvzk": encoded_after / B,
            "hash_bytes_none": hash_before / B, "hash_bytes_hvzk": hash_after / B,
            "prg_bytes_hvzk": prg_bytes / B,
            "coin_commitment_bytes_malicious": coins * (acc.HM_LEAF_BYTES + acc.HM_DECOMMIT_BYTES) / B,
        },
        "relative": {"committed": committed_after / committed_before - 1, "encoded": encoded_after / encoded_before - 1,
                     "hash": hash_after / hash_before - 1,
                     "prg_vs_hash_time_at_100GBps_50GBps": (prg_bytes / t.PRG_BYTES_PER_S) / (hash_before / HASH_RATE)},
        "depth": {"fiat_shamir_plonky3": 1, "interactive_hvzk": depth_interactive, "interactive_malicious": depth_interactive + 1,
                  "coins": coins},
        "soundness": {"deep_degree_term_before_log2": acc.log2(req["deep_degree_term_before"]),
                      "deep_degree_term_after_log2": acc.log2(req["deep_degree_term_after"]),
                      "note": "the selector raises the DEEP/quotient degree by one: the Schwartz-Zippel term grows by (deg+1)/deg; "
                              "the blinding codeword has the same degree bound as the trace, no change to the FRI terms"},
        "notes": [
            "hiding is perfect when the padding rows cover the revealed points (padding_covers); otherwise the trace domain doubles",
            ("Fiat-Shamir (Plonky3) makes the honest-verifier layer ZK against any verifier in the random-oracle model; the interactive "
            "variant with step 0 needs collision resistance only"),
            "the current Plonky3 implementation (FRI + Keccak, unsalted row leaves) has none of this: NON_ZK_PROOF_DIAGNOSTIC",
        ],
    }


def fiat_shamir_reparameterisation(field: str = "goldilocks", target_log2: int = -128, adversary_log2_queries: int = 60) -> dict:
    """What making a candidate non-interactive by Fiat-Shamir costs in parameters, under the generic
    state-restoration / round-by-round bound ``eps_FS <= q x eps_round`` (BCS16; for FRI: Block et al. 2024/1161):
    the interactive per-round error must fall to ``2^target / q``.  For the Ligero query terms that means more
    opened columns ``t``; for the field terms a larger extension.  Labelled comparison only: the campaign's
    protocols are interactive and no FS loss is charged."""
    fs = t.FIELDS[field]
    shifted = target_log2 - adversary_log2_queries
    n, k, l = fl.N_CODE, fl.K_CODE, fl.L_PAYLOAD
    t_now = acc.ligero_queries_for(target_log2 - 1, n, k, l)
    t_fs = acc.ligero_queries_for(shifted - 1, n, k, l)
    d_now = t.extension_degree(fs, target_log2, "A")
    # field terms: the largest per-round statistical term of A is the logUp identity (~2^-165 at degree d_now); shifting
    # by 2^60 needs the extension that first gives that term <= 2^shifted: scan degrees
    d_fs = None
    Ft = fs.option.trace
    for d in range(d_now, 13):
        Fc = Ft.extension(d)
        try:
            ident = acc.logup_identity_error(111_673_344, 65_536, Fc)
        except acc.AccountingError:
            continue
        if ident <= Fraction(2) ** shifted:
            d_fs = d
            break
    return {
        "field": field, "target_log2": target_log2, "adversary_log2_queries": adversary_log2_queries,
        "ligero_t_interactive": t_now, "ligero_t_fiat_shamir": t_fs, "opened_columns_factor": t_fs / t_now,
        "extension_degree_interactive": d_now, "extension_degree_fiat_shamir": d_fs,
        "challenge_limbs_factor": (d_fs * fs.L) ** 2 / (d_now * fs.L) ** 2 if d_fs else None,
        "note": "k > l + t must still hold for the ZK Ligero: k - l = 256 opened columns is the ceiling at l = 3840; "
                f"t = {t_fs} {'fits' if t_fs < k - l else 'does NOT fit (payload must shrink)'}",
    }


def soundness_with_zk(target_log2: int = -128) -> dict:
    """Budgets of A and B at the first configuration of each field with the ZK terms switched on: step-0 coin
    commitments (statistical hiding, a soundness term), the libra masking alternative for A (``1/|F|`` per masked
    instance), and whether the operating point still holds."""
    out = {}
    for field in ("goldilocks", "babybear"):
        fs = t.FIELDS[field]
        sfc = fl.small_field_census(fs.option.trace)
        tables = sfc.tables()
        for cand in ("A", "B"):
            d, tq, _ = fl.first_small_field_configuration(target_log2, fs.option, cand, t.K_TARGET, t.B_TARGET)
            Fc = fs.option.trace.extension(d)
            lig = acc.LigeroParams(n=fl.N_CODE, k=fl.K_CODE, l=fl.L_PAYLOAD, t=tq)
            rows = {}
            if cand == "A":
                for name, kw in (("hvzk", {}), ("malicious", {"coin_commitment_bits": 160}),
                                 ("malicious+libra", {"coin_commitment_bits": 160, "sumcheck_masking": "libra"})):
                    c = acc.CandidateA(K=t.K_TARGET, B=t.B_TARGET, trace_field=fs.option.trace, challenge_field=Fc, ligero=lig,
                                       hash_queries=fl.HASH_QUERIES, tables=tables, inputs_per_unit=sfc.columns_committed_a,
                                       query_wires_per_unit=sfc.queries_on_computed_wires, **kw)
                    bud = acc.account_candidate_a(c)
                    rows[name] = _budget_row(bud)
            else:
                for name, kw in (("hvzk", {}), ("malicious", {"coin_commitment_bits": 160})):
                    c = acc.CandidateB(K=t.K_TARGET, B=t.B_TARGET, trace_field=fs.option.trace, challenge_field=Fc, ligero=lig,
                                       hash_queries=fl.HASH_QUERIES, tables=tables, witness_per_unit=sfc.columns_committed_b,
                                       linear_per_unit=sfc.lin_total + 2 * sfc.mul, quadratic_per_unit=sfc.mul, **kw)
                    bud = acc.account_candidate_b(c)
                    rows[name] = _budget_row(bud)
            out[f"{cand}/{field}^{d}/t={tq}"] = rows
    return out


def _budget_row(bud: acc.Budget) -> dict:
    zk_terms = [(tm.name, tm.log2) for tm in bud.terms if "coin" in tm.name or "libra" in tm.name or tm.kind == "zk"]
    return {"total_log2": acc.log2(bud.total()), "meets_2^-128": bud.meets(-128), "zk_class": bud.parameters["zk_class"],
            "depth": bud.parameters["rounds.sequential_depth"], "pads": bud.parameters.get("pad_elements"),
            "zk_related_terms": zk_terms, "zk_loss_log2": acc.log2(bud.zk_total())}


def vole_row(target_log2: int = -128) -> dict:
    """The VOLE candidate (QuickSilver, designated verifier): privacy is native and malicious-verifier in the sVOLE-hybrid
    model; the ZK delta is zero by construction.  Comparison row only (never compared against A/B without the
    ``zk_mode`` caveat, ``explore.vole.PROOF_CLASS_NOTE``)."""
    v = t.price_vole(target_log2, utilisations=(1.0, 0.172))
    return {"design": "C (QuickSilver VOLE)", "zk_mode": v["zk_mode"], "overhead_H100@0.172": v["times"]["H100@0.172"]["overhead.vs_native_peak"],
            "zk_delta": 0.0, "p2v_bytes_per_vu": v["per_vu"]["p2v_bytes_quicksilver"],
            "privacy": "IT-MAC hiding is perfect; ZK against a malicious verifier in the F_sVOLE-hybrid (QuickSilver Thm 2); the "
                       "sVOLE protocol itself must be maliciously secure (Wolverine/Ferret are); no coin commitment needed: the "
                       "verifier's chi is answered with a fresh VOLE mask",
            "assumptions": ["LPN regular noise", "base OT", "correlation-robust hash"], "proof_class_note": v["proof_class_note"]}


def report(target_log2: int = -128) -> dict:
    rows = []
    for dn in ("A", "A-hintfree-depth630", "B"):
        for field in ("babybear", "goldilocks"):
            rows.append(design_zk_rows(dn, field, target_log2))
    air = [air_fri_zk(lay, field, target_log2) for lay in AIR_LAYOUTS for field in ("goldilocks", "babybear")]
    summary = []
    for r in rows:
        for zk in ("hvzk", "malicious"):
            for uk, dv in r["deltas"][zk]["times"].items():
                summary.append({"design": r["design"], "field": r["field"], "zk": zk, "utilisation": uk,
                                "overhead_none": r["levels"]["none"]["times"][uk]["overhead.vs_native_peak"],
                                "overhead": r["levels"][zk]["times"][uk]["overhead.vs_native_peak"],
                                "relative": dv["relative"], "overhead_delta": dv["overhead_delta"],
                                "depth": r["levels"][zk]["rounds.sequential_depth"]})
    return {
        "meta": {"lane": "zk-construction", "doc": "backends/ZK_CONSTRUCTION.md", "K": t.K_TARGET, "B": t.B_TARGET,
                 "target_log2": target_log2, "hash_rate_bytes_per_s": HASH_RATE, "prg_bytes_per_s": t.PRG_BYTES_PER_S,
                 "utilisations": {"tensor_md": 0.172, "tc_sumcheck_4090_model_equivalent": TC_SUMCHECK_4090_UTILISATION},
                 "levels": t.ZK_LEVELS},
        "summary": summary,
        "ligero_designs": rows,
        "air_fri": air,
        "vole": vole_row(target_log2),
        "fiat_shamir": {f: fiat_shamir_reparameterisation(f, target_log2) for f in ("goldilocks", "babybear")},
        "soundness": soundness_with_zk(target_log2),
        "assumptions": ASSUMPTIONS,
    }


ASSUMPTIONS: tuple[str, ...] = (
    "Every figure is an estimate from explore.tensor's limb price list and security.accounting's row/pad/coin counts; nothing is measured.",
    ("The 'none' level keeps the same committed witness (inputs, hints, query wires, multiplicities, helpers) and removes only what exists "
    "for privacy: pads, randomizers (payload k instead of l), blinding rows, HM96 leaves, coin commitments."),
    "HM96 leaves and coin commitments are hashed at the Merkle hash rate; the universal-hash evaluations are not priced (bit operations).",
    "Prover randomness is PRG output at 100 GB/s (AES-CTR on SIMT, unmeasured); an HM96 x_j is 121 B; a salt 32 B.",
    "Communication is the accountant's transcript estimate at 8 B per trace element (Goldilocks bytes even for 31-bit fields).",
    ("B-AIR: one row per unit, 2^19 rows at B=4096, rate 1/4 FRI in the proven unique-decoding regime with rows_log2 + 2 field-term "
    "applications; the quotient is 2^ceil(log2(deg-1)) chunks of F_c; leaves are LDE rows.  Relative deltas only."),
)


def _pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


def main(argv: Sequence[str] | None = None) -> None:
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    rep = report()
    if args and args[0] == "--report":
        from pathlib import Path

        Path(args[1]).write_text(json.dumps(rep, indent=1, default=str) + "\n")
        return
    print(f"{'design':22} {'field':10} {'zk':10} {'util':12} {'ovh none':>10} {'ovh zk':>10} {'delta':>9} depth")
    for s in rep["summary"]:
        print(f"{s['design']:22} {s['field']:10} {s['zk']:10} {s['utilisation']:12} {s['overhead_none']:10.3g} {s['overhead']:10.3g} "
              f"{_pct(s['relative']):>9} {s['depth']}")
    for a in rep["air_fri"]:
        print(f"B-AIR {a['layout']:16} {a['field']:10} committed {_pct(a['relative']['committed'])} encoded {_pct(a['relative']['encoded'])} "
              f"hash {_pct(a['relative']['hash'])} queries {a['fri']['queries']} depth FS 1 / interactive {a['depth']['interactive_hvzk']} "
              f"/ +step0 {a['depth']['interactive_malicious']}")
    print("C:", rep["vole"]["overhead_H100@0.172"], "zk delta", rep["vole"]["zk_delta"])
    print("FS:", json.dumps(rep["fiat_shamir"], indent=1))
    for k, v in rep["soundness"].items():
        print(k, {n: (round(r["total_log2"], 1), r["meets_2^-128"], r["depth"]) for n, r in v.items()})


if __name__ == "__main__":       # pragma: no cover
    main()
