"""reqval — the fine query on `Lift(P)` (R17-42 / LIFTING-SPEC A4, pad M-0955 rule of record `Q_fine(Lift(P)) := Lift(Q_fine(P))`, revb M-0965
R2/R3): the pytest form of `out/gen/sweep/evidence/reqval/lift_q_fine/wmax_lift.py` (PROMOTION 6.1).

REFERENCE, not the production object: `lift_query` below is the deliberately small statement of what `Lift(Q)` must do — (a) the width
predicate of `default(W)` reads the UNLIFTED declared width (lifted width minus one activity bit per lifted returned leaf), (b) every
Definition id in the query text is matched modulo the `Lifted[<id>]_v2` wrapper (padded-v1, LIFTING-SPEC v0.6 B7; was `_v1` before the enable).  Whatever lands in `gateset` / `vu_canonical` (R1: a versioned
Query object with its own digest embedding Q_fine_v1's) must agree with it set for set; until it lands, the refusal of R3 (an unlifted query on a
lifted Program is REFUSED by name) is the strict xfail at the bottom, so its arrival flips this file rather than passing silently."""
from __future__ import annotations

import collections
import re

import pytest

from verity.ir import query_ast as GS
from verity.ir.program import Program
from verity.ir.query import parse as parse_query
from verity_vllm.registry import b1
from verity_vllm.registry import lifted as L
from verity_capture import vu_canonical as VQ

LP, CAP = 8, 3
C = b1.B0_CONFIG
ATT = b1.attention_fa2(C["D"])
GUARD_IDS = ("LStepLive", "LSelect", "LBot", "LIsEOS", "LEmitNext", "Const")   # the root body's guard + controller rows (sampled_replay.GUARD_PREFIXES + the ⊥ / eos constants)
_ID = re.compile(r"([A-Za-z_][A-Za-z0-9_\.\-]*_v\d+)")


class _PayloadWidthEvaluator(GS._Evaluator):
    """`default(W)` on the UNLIFTED width: for a `Lifted[...]` Definition the declared width minus one bit per distinct returned leaf."""

    def member_width(self, fn) -> int:
        w = super().member_width(fn)
        if not (getattr(fn, "id", "") or "").startswith("Lifted["):
            return w
        if isinstance(fn, GS.PrimitiveDefinition):
            return w - 1
        return w - len(set(fn.body.ret))


def lift_query(profile: str = "ampere") -> tuple[GS.Query, GS.Query]:
    """(the query as written, Lift(Q)): Lift(Q) = the same override table with every id in the TEXT also matched under the wrapper."""
    cases, lifted = {}, {}
    for key, text in VQ.canonical_cases_text(profile):
        cases[key] = parse_query(text)
        lifted[key] = cases[key]
        lifted[_ID.sub(r"Lifted[\1]_v2", key)] = parse_query(_ID.sub(r"Lifted[\1]_v2", text))
    return GS.override(GS.default(VQ.W_MAX), cases), GS.override(GS.default(VQ.W_MAX), lifted)


def evaluate_lifted(program: Program, q: GS.Query) -> GS.Family:
    ev = _PayloadWidthEvaluator(program)
    return GS.PlanFamily(ev, ev.root_scope, ev.plan(program.root, q))


def _by_body(fam: GS.Family) -> collections.Counter:
    """sets per body, bodies named modulo the wrapper (so the two Programs' rows line up)."""
    out: collections.Counter = collections.Counter()
    for c in fam.classes():
        body = str(c.body).split("{")[0]
        if body.startswith("Lifted["):
            body = body[len("Lifted["):body.index("]")]
        elif body.startswith("Root[Lifted["):
            body = "Root[" + body[len("Root[Lifted["):body.index("]")]
        out[body] += c.total
    return out


@pytest.fixture(scope="module")
def twins():
    p_u = Program(b1.serve_v2(C, LP, CAP - 1, attention=ATT))                # D_MAX = CAP - 1 (pad's deletion-rule pairing)
    p_l = Program(L.lifted_request(C, LP, CAP, (0,), attention=ATT, lag=0))
    return p_u, p_l


def test_q_fine_v1_evaluated_afresh_on_lift_p_re_tiles_at_32_and_at_33(twins):
    p_u, p_l = twins
    q, _ = lift_query()
    n_u = GS.evaluate(p_u, q).count()
    n_32 = GS.evaluate(p_l, q).count()
    n_33 = GS.evaluate(p_l, GS.override(GS.default(VQ.W_MAX + 1), dict(VQ.canonical_cases("ampere")))).count()
    assert n_32 > n_u + 10_000        # by WIDTH: payload-32 composites are 33+ bits lifted and get descended (A4.2's re-tiling, in reverse)
    assert n_u + 10_000 < n_33 < n_32  # by NAME: the id-keyed case table does not match Lifted[<id>]_v2 -- a threshold cannot repair it
    # and the unlifted Program does not care about the threshold between 32 and 33 (nothing of P's is exactly 33 wide)
    assert GS.evaluate(p_u, GS.override(GS.default(VQ.W_MAX + 1), dict(VQ.canonical_cases("ampere")))).count() == n_u


def test_lift_q_reproduces_the_active_partition_and_the_residual_is_the_named_guard_class_only(twins):
    """revb R2: the guard / controller rows are a NAMED class outside the bijection -- the correspondence is delta 0 everywhere else."""
    p_u, p_l = twins
    q, lq = lift_query()
    fam_u = GS.evaluate(p_u, q)
    fam_l = evaluate_lifted(p_l, lq)
    bu, bl = _by_body(fam_u), _by_body(fam_l)
    root_u = next(k for k in bu if k.startswith("Root["))
    root_l = next(k for k in bl if k.startswith("Root["))
    assert {k: v for k, v in bu.items() if k != root_u} == {k: v for k, v in bl.items() if k != root_l}   # every non-root body: delta 0
    n_guard = fam_l.count() - fam_u.count()
    assert n_guard == L.continuation_gate_count(CAP, 1, 0) == 2 * (CAP - 1) + 1 + 2 + 3 * CAP  # 6 guard + 9 controller at CAP 3, |E| 1, LAG 0
    assert bl[root_l] - bu[root_u] == n_guard                                                   # the whole residual sits in the root body ...
    # ... and the root body's non-Lifted nodes are exactly the guard / controller ids: the class is named by id, one gate = one set
    extra = collections.Counter(nd.fn.id.split("{")[0] for nd in p_l.root.body.nodes if not nd.fn.id.startswith(("Lifted[", "Input")))
    assert extra and all(any(k.startswith(g) for g in GUARD_IDS) for k in extra), extra
    guard_gates = sum(c.hi - c.lo for c in p_l.circuit.root_calls() if any(c.id.startswith(g) for g in GUARD_IDS))
    assert guard_gates == n_guard


def test_lift_q_needs_both_halves_the_width_rule_and_the_wrapper_match(twins):
    p_u, p_l = twins
    q, lq = lift_query()
    n_u = GS.evaluate(p_u, q).count()
    only_width = evaluate_lifted(p_l, q).count()                          # payload width, ids as written
    only_names = GS.evaluate(p_l, lq).count()                             # ids modulo wrapper, declared (lifted) width
    both = evaluate_lifted(p_l, lq).count()
    assert only_width != n_u and only_names != n_u and both == n_u + L.continuation_gate_count(CAP, 1, 0)


@pytest.mark.xfail(strict=True, reason="revb M-0965 R3: gateset does not yet REFUSE an unlifted query on a lifted Program by name; "
                                       "when Lift(Q) lands as a versioned Query object this must raise and the reference above must agree with it")
def test_r3_an_unlifted_query_on_a_lifted_program_is_refused_by_name(twins):
    _, p_l = twins
    q, _ = lift_query()
    with pytest.raises(GS.QueryError, match="lifted"):
        GS.evaluate(p_l, q).count()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
