"""REDTEAM phase 6, target 2: ENC-REPLAY's Python replay (#3555), and the claim that "advice v1's replay makes v0
valid by construction".

`decode(cert, code)` replays vLLM V1's scheduler from the residual advice (`code`) and the certified record (`cert`).
The certified ingress and egress buckets enter `decode` only through `_coder`/`_Clock`/`_Egress` as the arithmetic
coder's *context* (they bias `model.freqs`), never as a constraint on the replayed schedule. So `decode` replays
whatever admission decisions the residual's bits carry, and nothing checks the decoded schedule against the record
beyond the per-request I/O and the engine config.

The claim holds narrowly: `v0_advice` builds a v0 that v0 validates from any schedule the replay emits, because the
replay only emits schedules vLLM could run. But validity is not fidelity. These cases show, executably, that a residual
the replay *accepts* (no error) decodes to a different valid schedule than the one that served, and to a valid v0 that
is not what served — on the identical certified record. The binding to the served schedule is not in the replay; it must
come from the downstream compute/commitment proof (ENC-REPLAY's replay-step circuit, not yet up).

Red-team evidence, not a merge-gate suite. Loads replay.py by file path (its package isn't a workspace member) and
remaps `verity` to this worktree for advice.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[5]
try:
    from editables.redirector import RedirectingFinder as _F

    _F.map_module("verity", str(_ROOT / "verity" / "__init__.py"))
except Exception:  # pragma: no cover
    pass


def _load(name: str):
    key = f"enc_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


R = _load("replay")
from verity.core.protocols.certifier import advice as V0  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CASE = "async-light"


def _trace():
    return R.read_step_trace(FIXTURES / f"{CASE}.jsonl", tokens=None)


def _trace_from_replay(t, rp):
    """A self-consistent Trace from a decoded schedule: before_step from the decode's admissions, a trivial clock. The
    replay of this before_step reproduces these steps, so `v0_advice` builds its v0 without diverging."""
    n = len(t.requests)
    return R.Trace(engine=t.engine, ids=list(t.ids), requests=list(t.requests),
                   before_step=[a if a is not None else 0 for a in rp.admitted], steps=rp.steps,
                   emits=[bool(s.rows) for s in rp.steps], ingress_t=[0.0] * n,
                   release_t=[float(k) for k in range(len(rp.steps))])


def test_honest_residual_is_the_served_schedule_and_a_valid_v0():
    """Baseline: the honest residual decodes to the served schedule, and `v0_advice` builds a v0 that v0 validates."""
    t = _trace()
    code = R.encode(t)
    m = R.compare(R.decode(t.certified(code.bucket_s), code).steps, t.steps)
    assert m.first_mismatch is None and m.matched == len(t.steps)
    adv = R.v0_advice(t, V0)
    assert isinstance(adv, V0.Advice) and len(adv.steps) == len(t.steps)


def test_decode_accepts_residuals_that_did_not_serve():
    """CLAIM: the certified record pins the served schedule; the residual only fills what the record leaves open.
    REALITY: the certified ingress/egress buckets are the arithmetic coder's context, not a constraint. Every single-bit
    flip of the honest residual decodes WITHOUT error, and most decode to a different valid schedule than the one that
    served, on the identical `Certified` record. `decode` cannot tell a served residual from a forged one.
    ONE-LINE FIX: make the certified ingress clock a hard constraint in the replay (reject a decoded admission whose
    bucket contradicts `cert.ingress`), not merely an arithmetic-coding context.
    """
    t = _trace()
    code = R.encode(t)
    cert = t.certified(code.bucket_s)
    (w, bits), = code.bits.items()
    raised = different = same = 0
    for i in range(len(bits)):
        flipped = R.Code(code.contexts, code.bucket_s, code.window_buckets, {w: bits[:i] + [1 - bits[i]] + bits[i + 1:]},
                         code.ideal)
        try:
            rp = R.decode(cert, flipped)
        except Exception:
            raised += 1
            continue
        if R.compare(rp.steps, t.steps).first_mismatch is None:
            same += 1
        else:
            different += 1
    assert raised == 0, "decode never rejects a residual: it has no validity check against the record"
    assert different >= 1, "a forged residual decodes to a schedule that did not serve"
    # the record the verifier holds is identical whatever the residual says.
    assert cert.requests == t.certified(code.bucket_s).requests and cert.engine == t.certified(code.bucket_s).engine


def test_a_residual_decodes_to_a_valid_v0_that_is_not_what_served():
    """CLAIM: "advice v1's replay makes v0 valid by construction" — so the v0 the verifier reconstructs is the served
    one.
    REALITY: "valid by construction" is true but not sufficient. A forged residual decodes to a different valid vLLM
    schedule, and `v0_advice` builds from it a v0 that v0 validates and that differs from the served v0 — same certified
    requests and engine. Validity does not pin fidelity; the residual, not the record, chooses the schedule.
    ONE-LINE FIX: the binding must be the committed values of the executed steps (the replay-step circuit), shown to
    reject a v0 whose batching did not run; the Python replay alone cannot certify fidelity.
    """
    t = _trace()
    code = R.encode(t)
    cert = t.certified(code.bucket_s)
    served = R.v0_advice(t, V0)
    (w, bits), = code.bits.items()
    found = None
    for i in range(len(bits)):
        flipped = R.Code(code.contexts, code.bucket_s, code.window_buckets, {w: bits[:i] + [1 - bits[i]] + bits[i + 1:]},
                         code.ideal)
        rp = R.decode(cert, flipped)
        if R.compare(rp.steps, t.steps).first_mismatch is None:
            continue
        forged = R.v0_advice(_trace_from_replay(t, rp), V0)
        if forged != served:
            found = (len(rp.steps), forged)
            break
    assert found is not None, "a forged residual yields a valid v0 that is not the served one"
    steps, forged = found
    assert isinstance(forged, V0.Advice)
    assert steps != len(t.steps) and forged != served
    # same requests served, different batching reconstructed: the record does not distinguish them.
    assert {q.id for q in forged.requests} == {q.id for q in served.requests}
