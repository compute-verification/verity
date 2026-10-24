"""Red-team (REDTEAM-ADVICE, phase 9 target 3): confirm #3618 (NET-E2E varied follow-up) closes the phase 5 report
findings. The two that had live executable breaks are checked here against the real `e2e.negative`; findings 2 and 3
(the OK no longer passes an unbound advice; the prove-path column relabelled and excluded when unbound) are closed in
`e2e.report`/`chain.prove_path` by code the checkpoint cites.

Phase 5 finding 1 (CLOSED): on the advice profile (N2 undecided) a negative is "refused" only when the rehearsal's own
check names the negative's OWN_CODE violation; a refusal for any other reason no longer counts.
Phase 5 finding 4 (CLOSED): a run with no `server.json` (shim_requests None) reads as not-run, so it cannot be a
"refused" negative.
"""
from __future__ import annotations

import benchmarks.network_e2e.e2e as E


def _run(accepted, codes, shim):
    return {"check": {"verdict": "refused", "codes": list(codes)} if codes is not None else None,
            "network": {"accepted": accepted, "windows": None, "refused": None, "digest": None},
            "proofs": None, "shim_requests": shim}


def test_finding1_advice_negative_refused_only_by_its_own_code():
    own = E.OWN_CODE["extra"]                       # "io:longer than max_tokens"
    # undecided (advice profile); a refusal for an UNRELATED code is no longer this negative's refusal.
    wrong = E.negative(_run(None, ["audit.status:some-coverage-fault"], 5), [], "extra")
    assert wrong["refused"] is False and wrong["own_code"] == own
    # the same check that names the negative's own violation IS its refusal.
    right = E.negative(_run(None, [own], 5), [], "extra")
    assert right["refused"] is True


def test_finding4_a_negative_with_no_server_json_reads_as_not_run():
    own = E.OWN_CODE["extra"]
    # shim_requests None (no server.json): not-run, so not a refused negative even with the right code.
    none = E.negative(_run(None, [own], None), [], "extra")
    assert none["ran"] is False and none["refused"] is False
    # a real run with the right code is refused.
    real = E.negative(_run(None, [own], 7), [], "extra")
    assert real["ran"] is True and real["refused"] is True
