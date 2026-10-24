"""REDTEAM phase 12, target 3 (REDTEAM-ADVICE, note:network-accounting/20261024T0558Z-finding-redteam-v3) against #3728,
CERT-VARLEN's compact EOS fix. Two things:

- **The fix holds.** The phase-10 cases (`test_compact_phase10`) pass against #3728: an EOS id in a compact response,
  marked or unmarked, is refused ("an EOS id in a compact response"), so the server can no longer extend a response past a
  sampled EOS by sending the id as an ordinary token.

- **A second way a server decides where a compact response stops, which the fix does not close.** A compact stop record is
  `COMPACT_EOS` and carries no token id (the stop token is hidden). `certified_io`'s compact branch accepts a `COMPACT_EOS`
  last mark as finish `eos` at *any* length up to `max_tokens`, with no check that the model sampled an EOS there — there is
  nothing to check, the token being hidden. So a response that honestly ran to `max_tokens` (finish `length`, no EOS
  sampled) can be truncated at an arbitrary earlier position with a `COMPACT_EOS` stop, and `certified_io` accepts it as
  `eos`: the server shortens the response and charges fewer tokens, claiming a stop it never sampled.

  The id-events rule binds the stop to a revealed EOS token (``stops[0] == len(out) - 1``, then the I/O tie to the
  committed sampled id); compact hides the stop token, so `certified_io` loses that binding entirely. The true length is
  bound only by the committed sampled ids (IoTieWindow / `active.compact_finish_violations` holding each response's
  `output_len` and finish to the advice), which on the advice profile is a stand-in that need not run (target 2). So
  `certified_io`, the verdict of record there, takes the server's word for where a compact response stops.

One-line fix: the hidden `COMPACT_EOS` stop can't be verified by `certified_io` alone, so on the compact advice profile the
response length and finish must be a required gate of the I/O tie (`active.compact_finish_violations` against the advice),
not `certified_io`'s to accept. Owner: CERT-VARLEN (with NET-E2E making the tie a gate, target 2).
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

# test_certifier_serve puts the benchmark on the path; test_compact_phase10 shares the honest compact run and its helpers.
from test_certifier_serve import VARIED_TOKENS, _no_sleep, _Varied, _varied_runs, ar, cs
from verity.core.protocols.certifier import active

B = 8


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    endpoint = _Varied(merges=lambda r: r % 2 == 1)
    args = argparse.Namespace(model=None, api_key_env=None, traffic="varied", requests=16, rate=2.0,
                              prompt_range="16:1024", tokens=VARIED_TOKENS, seed=3, probes=4, eos_ids=None, compact=B)
    tmp, _probed, report, plan, runs = _varied_runs(tmp_path_factory.mktemp("compact-eos-p12"), endpoint,
                                                    lambda url: cs.plan_of(args, url, lambda: 0.0, _no_sleep), ("honest",))
    setup = ar.Setup(report)
    e, i = setup.egress_params, setup.ingress_params
    ins = [ar.unpack_record(b, i.frame_bytes) for b in ar.read_blobs(tmp / "honest" / "requests.records")]
    outs = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(tmp / "honest" / "responses.records")]
    return SimpleNamespace(plan=plan, setup=setup, ins=ins, outs=outs, check=runs["honest"]["check"])


def _truncate_with_stop(outs, stream: int, k: int):
    """`outs` with ``stream``'s response truncated at output ``k``: outputs 0..k-1 kept, output k replaced by a
    `COMPACT_EOS` stop record, outputs k+1.. dropped — the server claiming the model sampled EOS at position k."""
    seen, out = 0, []
    for rec in outs:
        grid = []
        for row in rec.grid:
            new = []
            for slot in row:
                got = None if slot is None else active.compact_record_data(slot)
                if got is not None and got[0] == stream:
                    if seen == k:
                        slot = active.compact_cell(active.compact_record(stream, None, "stop"), B)
                    elif seen > k:
                        slot = None
                    seen += 1
                new.append(slot)
            grid.append(new)
        out.append(SimpleNamespace(window=rec.window, grid=grid))
    assert seen > k
    return out


def test_the_honest_compact_run_is_accepted(run):
    honest = cs.certified_io(run.plan, run.setup, run.ins, run.outs)
    assert honest["accepted"] and run.check["verdict"] == "accepted"


def test_a_compact_eos_stop_shortens_a_max_tokens_response_and_is_accepted_as_eos(run):
    """A response that honestly ran to max_tokens (finish `length`, no EOS sampled) is truncated to two outputs with a
    `COMPACT_EOS` stop. `certified_io` accepts it as `eos`: the server stopped the response where it chose, charging
    fewer tokens for a stop it never sampled. The stop token is hidden, so `certified_io` has nothing to refuse — the
    binding is the I/O tie's (`active.compact_finish_violations` against the advice), a stand-in on the advice profile."""
    honest = cs.certified_io(run.plan, run.setup, run.ins, run.outs)
    assert honest["accepted"]
    target = next(r for r in honest["records"] if r["finish"] == "length" and len(r["output"]) >= 3)
    stream, full = target["stream"], len(target["output"])
    eos = set(run.plan["varied"]["eos_token_ids"])
    assert not any(t in eos for t in target["output"])  # it really ran to max_tokens, no EOS among its ids

    forged = _truncate_with_stop(run.outs, stream, 1)
    got = cs.certified_io(run.plan, run.setup, run.ins, forged)
    mine = next(x for x in got["records"] if x["stream"] == stream)
    assert got["accepted"] and got["violations"] == []
    assert mine["finish"] == "eos" and len(mine["output"]) == 2 < full
    # the compact finish rule placed no constraint on where the stop landed:
    assert got["tokens"] < honest["tokens"]


def test_a_compact_eos_stop_is_accepted_at_every_length_up_to_max_tokens(run):
    """The stop may land anywhere up to max_tokens: for each truncation position the response is accepted as `eos`, so
    `certified_io` binds the compact stop's position to nothing — the server alone decides where it falls."""
    honest = cs.certified_io(run.plan, run.setup, run.ins, run.outs)
    target = next(r for r in honest["records"] if r["finish"] == "length" and len(r["output"]) >= 4)
    stream, full = target["stream"], len(target["output"])
    for k in range(full):
        got = cs.certified_io(run.plan, run.setup, run.ins, _truncate_with_stop(run.outs, stream, k))
        mine = next(x for x in got["records"] if x["stream"] == stream)
        assert got["accepted"] and mine["finish"] == "eos" and len(mine["output"]) == k + 1, k
