"""Live soundness budget of the GPU prover's proof: ``security.accounting.account_candidate_a`` on the instance proved.

Model: the interactive protocol of PROTOCOL.md (Candidate A, "Fiat-Shamir is not used"): the verifier's coins are
live.  The benchmark harness derives them as SHA-256(statement, transcript prefix) so that a proof file can be
re-checked offline (``verity-gkr-verify``); no Fiat-Shamir soundness is claimed for that file (under the campaign's
``q x eps`` rule every statistical term below would be multiplied by 2^60).  Terms, BabyBear^6 challenges
(``|F| = p^6 ~ 2^185.6``):

  * commitment binding: SHA-512 Merkle tree over the Ligero columns, generic ``q^2 / 2^512`` at ``q = 2^64``, i.e.
    2^-384.  SHA-256 made this term 2^-128 by itself, which is why the recorded figure was 2^-127.7;
  * Ligero ``(n, k, t) = (16384, 4096, 192)``, non-ZK (``l = k - 1`` stands in for ``l = k``: ``validate`` needs
    ``l < k``), Appendix-C query terms; the proximity test combines the ``m`` committed rows by powers of one ``r``
    (``gpu/ligero.row_coeffs``), booked as the BCIKS20 curve bound ``((m-1) n + 1) / |F|``;
  * the unit segment's data-parallel GKR layers and LogUp tables taken from the instance (table rows, columns and
    queries per unit / per VU), per-table LogUp; the epilogue segment's layers are added here (assertions booked on
    every layer, conservative);
  * the chain linkage folded into the Ligero functional by powers of one challenge: ``(N - 1) / |F|``.

The total is dominated by the Ligero IRS query term ``(1 - e/n)^t = 0.625^192 = 2^-130.19``.
"""
from __future__ import annotations

import math
from fractions import Fraction

HASH_BITS = 512
HASH_QUERIES = 2**64


def budget(inst, ligero_rows: int, linkage_constraints: int, n: int = 16384, k: int = 4096, t: int = 192,
           K: int | None = None, bind_constraints: int = 0):
    """``K``: the VU's reduction length (default 16 products per unit)."""
    from verity_numerical.security import accounting as acc

    from gpu.field import DEG, P

    Ft = acc.Field.prime("BabyBear", P)
    Fc = Ft.extension(DEG)
    units, *rest = inst.segs
    vus = inst.vus()
    steps = inst.chain.steps

    def nq(seg, name: str) -> int:
        return sum(1 for q in seg.circ.queries if q.table == name)

    tables = tuple(acc.LookupTable(tb.name, tb.nrows, nq(units, tb.name), tb.cols,
                                   queries_per_vu=sum(nq(s, tb.name) * (s.units // vus) for s in rest))
                   for tb in inst.tables)
    layers = tuple(acc.CheckerLayer(L.s_out, L.s_in, assertions=True) for L in units.layers)
    cand = acc.CandidateA(K=16 * steps if K is None else K, B=vus, trace_field=Ft, challenge_field=Fc,
                          ligero=acc.LigeroParams(n=n, k=k, l=k - 1, t=t, zk=False), layers=layers, tables=tables,
                          lookup_mode="per_table", hash_bits=HASH_BITS, hash_queries=HASH_QUERIES,
                          leaf_commitment_bits=None, irs_row_combination="powers", ligero_rows=ligero_rows)
    B = acc.account_candidate_a(cand)
    for s in rest:
        b = max(0, math.ceil(math.log2(s.units)))
        B.add("batching", f"{s.name}: output layer all-zero claim, random z in F^{s.layers[0].s_out + b}",
              acc.schwartz_zippel_error(s.layers[0].s_out + b, Fc))
        for j, L in enumerate(s.layers):
            e = acc.data_parallel_layer_error(L.s_out, L.s_in, b, Fc, alpha=True, assertions=True)
            B.add("batching", f"{s.name} layer {j}: alpha", e["alpha"])
            B.add("batching", f"{s.name} layer {j}: beta", e["beta"])
            B.add("sumcheck", f"{s.name} layer {j}: {e['rounds']} rounds", e["sumcheck"])
    B.add("batching", f"chain linkage: {linkage_constraints} constraints folded by powers of one challenge",
          Fraction(max(0, linkage_constraints - 1), Fc.size))
    if bind_constraints:
        # rho_bind * sum_c geo_bind^c r_c: a nonzero residual vector leaves a nonzero polynomial of degree <= N in
        # (rho_bind, geo_bind), both drawn after every message the residuals depend on
        B.add("batching", f"operand binding: {bind_constraints} constraints, fresh rho_bind times powers of fresh geo_bind",
              Fraction(bind_constraints, Fc.size))
    return B


def summary(B) -> dict:
    by = {c: round(math.log2(v), 2) for c, v in B.by_component().items() if v}
    top = max(B.soundness_terms(), key=lambda tm: tm.error)
    return {"achieved_log2": math.log2(B.total()), "by_component_log2": by,
            "dominant_term": {"component": top.component, "name": top.name, "log2": round(top.log2, 2)},
            "hash": f"SHA-512 Merkle (generic q^2/2^{HASH_BITS}, q = 2^{int(math.log2(HASH_QUERIES))})",
            "model": "interactive (PROTOCOL.md A); proof files derive the coins as SHA-256(transcript) for offline "
                     "re-checking, no Fiat-Shamir soundness is claimed",
            "source": "backends/gkr/soundness.py -> security.accounting.account_candidate_a on the proved instance"}
