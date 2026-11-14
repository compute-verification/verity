"""The honest prover's side of check 3 through the opening unit (`verity.protocols.accounting.poub.verifier.opening`): a
sampled receipt's arguments from the prover's store and pass (`witness`), and a session's windows decided by `opening.check`
on them (`reference`), the stand-in for the units' proofs.
"""

from __future__ import annotations

from collections.abc import Sequence

from verity.protocols.accounting.poub.prover import audit_prover
from verity.protocols.accounting.poub.verifier import audit, opening
from verity.protocols.accounting.poub.verifier.commitments.gates import block_check as BC
from verity.protocols.accounting.poub.verifier.receipts import PER_LEAF


def witness(d: audit.PositionSample, ps: audit_prover.Pass, store: audit_prover.Store, plan_row: Sequence[int], S,
            values: opening.Values) -> dict[str, list[int]]:
    """The honest prover's arguments for sample ``d`` of pass ``ps``: its block and payload from its store, the receipt leaf
    under its salt and path in the pass's tree, the payload under its salt and path in its append's tree (or the
    registration's), the step's certified plan row, and, appended, the payload's path in the canonical payload commitment."""
    j, c = d.block, d.leaf
    pay_salt, pay_path, _d = store.committed(j)
    leaf = b"".join(ps.receipts[PER_LEAF * c:PER_LEAF * (c + 1)])
    vpath = values.path(j - values.base) if d.coin is not None else ()
    return BC.opening_args(S, opening.public(d, store.registration.codec_seed, S, values), store.blocks[j], store.payload(j), leaf,
                           list(plan_row), ps.salts[c], pay_salt, tuple(ps.tree.open(c).path), tuple(pay_path), vpath)


def reference(session: audit.AuditRun, store: audit_prover.Store, passes: dict, c3) -> list[audit.Verdict]:
    """Every window of a session whose receipts are sampled, decided by `opening.check` on the honest prover's witness: the stand-in
    for the units' proofs. Stops at the first rejected window. ``c3`` holds the codec, the plan row's P and the depths DR,
    DP and DV, each step's plan row (``plan_rows``), its salt (``plan_salts``) and certified ``b || c`` (``plan_bcs``), and
    the value commitment (``values``)."""
    out = []
    for t in range(len(session.windows)):
        fails = []
        for d in session.drawn(t):
            S = opening.statics(c3.codec, d.leaf_size, c3.P, c3.DR, c3.DP, c3.DV)
            a = witness(d, passes[d.window, d.index], store, c3.plan_rows[d.step], S, c3.values)
            fails.append(opening.check(d, a, store.registration.codec_seed, S, c3.values, c3.plan_salts[d.step], c3.plan_bcs[d.step]))
        out.append(session.decide(t, fails))
        if not out[-1].accepted:
            break
    return out


__all__ = ["reference", "witness"]
