"""REDTEAM phase 7, target 3: N2-ADVICE's fix, re-running phase 6 target 1 against the advised unit now that
cursor/n2-advice-061a moved past d74de53fd9 (head 622571d4f2: `advised.PromptTie`, `ends` refuses a link without it).

Phase 6 target 1 (`test_advised_adversarial_phase6.test_split_prompt_lengths_defeat_the_floor`) defeated the ingress-end
floor on one certified record: `ends` read each request's prompt length from v0 alone and held only the *total* to the
record, so a prover who swapped a short and a long prompt between an early and a later request lowered the early
request's end and the floor, and an acausal egress was accepted. This re-runs that attack against the fix and confirms,
independently of n2-advice's own regression (`test_network_certifier_advised_phase6.py`):

  (1) the advised verdict refuses a link WITHOUT the tie (the lead's exact ask);
  (2) under the certified tie (the true prompt lengths and arrival order) the forged split and the order swap are both
      refused, where before they were accepted;
  (3) the honest advice under the matching tie restores the true, higher floor, so the acausal egress stays refused;

and names the residual gap this fix leaves (which closes phase 7 target 1's loop): `advised` checks the advice against
the tie, but NOTHING here checks the tie against the certified ingress record — that agreement is a host-side premise
until IO-TIE's in-proof `IoTieRequest` (confirmed absent in target 1). A tie that states the forged split still accepts
it, so the binding rests entirely on an honest host supplying the tie from the certified I/O (`derived.tie_fields`).

Red-team evidence, not a merge-gate suite; it remaps `verity` to the worktree so it runs regardless of the editable
install pointing at main."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

# <root>/verity/core/protocols/certifier/tests/<this file>  ->  parents[5] is the repo root.
_ROOT = Path(__file__).resolve().parents[5]
try:  # make `verity` resolve to this worktree, not the editable install's main checkout
    from editables.redirector import RedirectingFinder as _F

    _F.map_module("verity", str(_ROOT / "verity" / "__init__.py"))
except Exception:  # pragma: no cover - plain checkout, nothing to remap
    pass

import pytest  # noqa: E402

from verity.core.protocols.certifier import advice as I1  # noqa: E402
from verity.core.protocols.certifier import advised as AD  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet  # noqa: E402

T, R, RIN = 4, 32, 9
# CERT-VARLEN's id-request framing at B = 4096: a prompt of up to 337 tokens is one ingress frame, 338..399 is two.
RF = (1,) * 337 + (2,) * 62
SHORT, LONG = 16, 338
SYNC = ClockSync(0, Fraction(1))
SS = SyncSet((0,), (Fraction(1),))


def _one_prefill(prompt_lens, release):
    reqs = tuple(I1.RequestAdvice(f"r{i}", p, 1, "length") for i, p in enumerate(prompt_lens))
    steps = tuple(I1.StepAdvice((I1.Row(i, "prefill", 0, p),)) for i, p in enumerate(prompt_lens))
    return I1.Advice(reqs, steps, tuple(release))


def _tie(prompt_len, order=None):
    return AD.PromptTie(tuple(prompt_len), tuple(range(len(prompt_len))) if order is None else tuple(order))


def _accepts(link, ingress, egress, own=0):
    try:
        v = AD.view(link, 0, T, R, [SYNC], SS, RF)
    except ValueError:
        return False  # view/ends refused the link: no verdict, failing closed
    return AD.reference(T, RIN, own, ingress, egress, v.released, v.floor, v.received)


# the phase-6 record: req0 is a 338-token prompt (2 frames), req1 a 16-token one (1 frame); the frames land in buckets
# 0,1,2 and the egress releases an output in bucket 1 and one in bucket 3. req0's output in bucket 1 is acausal (its
# second frame only lands in bucket 1).
INGRESS, EGRESS = [1, 1, 1, 0], [0, 1, 0, 1]
CERTIFIED = _tie((LONG, SHORT))


def test_the_advised_verdict_refuses_a_link_without_the_tie():
    """The lead's exact ask. Phase 6's forged split (16 then 338), which d74de53fd9 accepted, gets no verdict now: `ends`
    refuses a link whose tie is None, so `view` raises and the link fails closed."""
    forged = AD.LinkAdvice(_one_prefill((SHORT, LONG), (1, 3)), (3,))  # no tie
    assert forged.tie is None
    with pytest.raises(ValueError, match="without the prompt-length tie"):
        AD.ends(forged, RF)
    with pytest.raises(ValueError, match="without the prompt-length tie"):
        AD.view(forged, 0, T, R, [SYNC], SS, RF)
    assert not _accepts(forged, INGRESS, EGRESS)  # the phase-6 accept is now a refusal


def test_the_forged_split_is_refused_under_the_certified_tie():
    """Under the certified tie (the true (338, 16) prompts in arrival order), the forged split's prompt lengths don't
    match, so `ends` refuses it before any floor is computed — the phase-6 accept (`_accepts is True` at d74de53fd9) is
    gone. Both advices are still valid v0 of equal charge, so only the tie catches it."""
    honest = AD.LinkAdvice(_one_prefill((LONG, SHORT), (1, 3)), (3,), CERTIFIED)
    forged = AD.LinkAdvice(_one_prefill((SHORT, LONG), (1, 3)), (3,), CERTIFIED)
    assert AD.ends(honest, RF) == (2, 3)  # the true, higher end
    with pytest.raises(ValueError, match="request 0's prompt_len 16 isn't its certified prompt's 338 ids"):
        AD.ends(forged, RF)
    assert not _accepts(forged, INGRESS, EGRESS)
    assert I1.total_bits(honest.advice) == I1.total_bits(forged.advice)


def test_the_order_swap_is_refused_under_the_certified_tie():
    """Phase 6's other seam: v0's first-row order isn't arrival order, so a later-arriving request numbered first lowered
    its end. The tie's `order` must be arrival order (identity, FCFS); a non-identity order is refused, and with equal
    prompt lengths — where the lengths alone can't tell — a swapped release is caught only by the order."""
    swapped = _one_prefill((SHORT, SHORT), (1, 3))
    with pytest.raises(ValueError, match="arrival order"):
        AD.view(AD.LinkAdvice(swapped, (2,), _tie((SHORT, SHORT), (1, 0))), 0, T, R, [SYNC], SS, RF)
    assert not _accepts(AD.LinkAdvice(swapped, (2,), _tie((SHORT, SHORT), (1, 0))), [1, 0, 1, 0], [0, 1, 0, 1])
    # the honest identity order on the same record is admitted (so the refusal is the order, not the shape).
    assert _accepts(AD.LinkAdvice(swapped, (2,), _tie((SHORT, SHORT))), [1, 0, 1, 0], [0, 1, 0, 1])


def test_the_honest_advice_restores_the_true_floor():
    """Under the certified tie the honest split's true end (2 frames for req0) sets the floor, so the acausal egress — an
    output in bucket 1 before req0's second frame landed — is refused. The fix doesn't over-refuse: a causal egress on
    the same record and tie is accepted."""
    acausal = AD.LinkAdvice(_one_prefill((LONG, SHORT), (1, 3)), (3,), CERTIFIED)
    assert not _accepts(acausal, INGRESS, EGRESS)  # req0's output in bucket 1, before its 2nd frame: refused
    causal = AD.LinkAdvice(_one_prefill((LONG, SHORT), (2, 3)), (3,), CERTIFIED)
    assert _accepts(causal, [1, 1, 1, 0], [0, 0, 1, 1])  # req0's output in bucket 2, after its 2nd frame: accepted


def test_residual_gap_the_tie_itself_is_not_checked_against_the_record():
    """The fix binds the advice to the tie, but NOTHING in `advised` binds the tie to the certified ingress: supply a tie
    that states the forged split (16 then 338) and the forged advice is accepted again, with the acausal release passing.
    So N2-ADVICE's ingress end rests on an honest host building the tie from the certified I/O (`derived.tie_fields`);
    the in-proof check is IO-TIE's `IoTieRequest`, which phase 7 target 1 confirmed is not yet built. This is the named
    host-side premise, measured here as: a prover who controls the tie reopens the whole phase-6 channel."""
    forged = AD.LinkAdvice(_one_prefill((SHORT, LONG), (1, 3)), (3,), _tie((SHORT, LONG)))
    assert AD.ends(forged, RF) == (1, 3)  # the lowered end the tie now states, unchecked against the record
    assert _accepts(forged, INGRESS, EGRESS)  # the phase-6 acausal accept, reopened by a prover-supplied tie
