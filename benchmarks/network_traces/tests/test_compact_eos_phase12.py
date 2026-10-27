"""Adversarial cases against the compact framing's EOS rule, kept as regressions of its fix:

- **The fix holds.** `test_compact_phase10`'s cases pass: an EOS id in a compact record that isn't its
  response's stop is refused, so the server can't extend a response past a sampled EOS by sending the id as an
  ordinary token.
- **A second way a server decided where a compact response stops.** A compact stop record was `COMPACT_EOS` (all ones)
  and carried no token id, so `certified_io` accepted it as finish `eos` at any length up to `max_tokens`, with nothing
  to check: a response that honestly ran to `max_tokens` (finish `length`, no EOS sampled) could be cut at any earlier
  position and charged fewer tokens for a stop it never sampled. The forged stop was accepted at every position. Now a stop record carries the EOS id the model sampled (`COMPACT_LAST | COMPACT_STOP | id`), so:
  - the old cell is a stop on id 2^30 − 1, and `certified_io` refuses a stop on an id that isn't EOS at every position;
  - a stop on an EOS id the model never sampled there is `certified_io`'s to accept, as it accepts the same id events
    (the server writes what it likes): the ids of record are the Commit's, and `active.compact_finish_violations`,
    which the I/O tie requires, holds each response's ids, length and finish to them, refusing the cut at every
    position.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

# test_certifier_serve puts the benchmark on the path
from test_certifier_serve import VARIED_TOKENS, _answer, _no_sleep, _Varied, _varied_runs, ar, cs
from test_compact_phase10 import _as_id_events
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.grid import row_count

B = 8
NOT_EOS = "a stop on an id that isn't EOS"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    endpoint = _Varied(merges=lambda r: r % 2 == 1)
    args = argparse.Namespace(model=None, api_key_env=None, traffic="varied", requests=16, rate=2.0,
                              prompt_range="16:1024", tokens=VARIED_TOKENS, seed=3, probes=4, eos_ids=None, compact=B)
    tmp, _probed, report, plan, runs = _varied_runs(tmp_path_factory.mktemp("compact-eos-p12"), endpoint,
                                                    lambda url: cs.plan_of(args, url, lambda: 0.0, _no_sleep),
                                                    ("honest",))
    setup = ar.Setup(report)
    e, i = setup.egress_params, setup.ingress_params
    ins = [ar.unpack_record(b, i.frame_bytes) for b in ar.read_blobs(tmp / "honest" / "requests.records")]
    outs = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(tmp / "honest" / "responses.records")]
    honest = cs.certified_io(plan, setup, ins, outs)
    target = next(r for r in honest["records"] if r["finish"] == "length" and len(r["output"]) >= 4)
    return SimpleNamespace(plan=plan, setup=setup, ins=ins, outs=outs, check=runs["honest"]["check"], honest=honest,
                           eos=frozenset(plan["varied"]["eos_token_ids"]), target=target)


def _sampled(io: dict) -> dict[int, list[int]]:
    """Each request's ids as the model sampled them, by its ingress ordinal: the stub's greedy answer (`_answer`) to
    the request's fields, the ids the Commit records."""
    return {r["stream"]: _answer(r["prompt"], r["max_tokens"])[0] for r in io["records"]}


def _caps(io: dict) -> dict[int, int]:
    """Each request's `max_tokens` by its ingress ordinal."""
    return {r["stream"]: r["max_tokens"] for r in io["records"]}


def _cells(outs) -> list[bytes]:
    """The link's frames in canonical order, padding left out."""
    return [s for rec in sorted(outs, key=lambda r: r.window) for row in rec.grid for s in row[:row_count(row)]]


def _cut(outs, stream: int, k: int, record: bytes):
    """`outs` with `stream`'s response cut at output `k`: outputs 0 … k − 1 kept, output k's cell `record`'s, the rest
    dropped, each row keeping its width."""
    seen, out = 0, []
    for rec in outs:
        grid = []
        for row in rec.grid:
            new = []
            for slot in row:
                got = None if slot is None else active.compact_record_data(slot)
                if got is not None and got[0] == stream:
                    seen += 1
                    if seen - 1 > k:
                        continue
                    if seen - 1 == k:
                        slot = active.compact_cell(record, B)
                new.append(slot)
            grid.append(new + [None] * (len(row) - len(new)))
        out.append(SimpleNamespace(window=rec.window, grid=grid))
    assert seen > k
    return out


def test_the_honest_compact_run_is_accepted_and_its_cells_are_the_commits_sampled_ids(run):
    """The honest run's certified I/O is the model's sampled ids, each stop carrying its EOS id, and its cells are
    `compact_records_of` them, so the I/O tie's finish rule finds nothing."""
    assert run.honest["accepted"] and run.check["verdict"] == "accepted"
    sampled = _sampled(run.honest)
    assert {r["stream"]: r["output"] for r in run.honest["records"]} == sampled
    assert any(r["finish"] == "eos" for r in run.honest["records"]) and not run.eos & set(run.target["output"])
    assert active.compact_finish_violations(sampled, run.eos, _cells(run.outs), max_tokens=_caps(run.honest)) == []


def test_the_hidden_stop_is_refused_at_every_length(run):
    """The response that ran to `max_tokens` (finish `length`, no EOS among its ids) cut at output k by
    the old `COMPACT_EOS` cell, which was accepted as `eos` at every k. That cell is a stop on id 2^30 − 1, and so is
    refused at every k, as is a stop on the id the model did sample there: a stop is only on an EOS id."""
    stream, full = run.target["stream"], len(run.target["output"])
    old = active.COMPACT_RECORD.pack(stream, 0xFFFF_FFFF)
    assert active.compact_record_data(old) == (stream, active.COMPACT_STOP - 1, "stop")
    for k in range(full):
        for record in (old, active.compact_record(stream, run.target["output"][k], "stop")):
            got = cs.certified_io(run.plan, run.setup, run.ins, _cut(run.outs, stream, k, record))
            mine = next(x for x in got["records"] if x["stream"] == stream)
            assert not got["accepted"] and got["violations"] == [[stream, NOT_EOS]] and len(mine["output"]) == k + 1, k


def test_a_stop_the_model_never_sampled_is_refused_by_the_tie_at_every_length(run):
    """A server that cuts the response at output k now has to write an EOS id there, which the model didn't sample.
    `certified_io` accepts it as `eos`, as it accepts the same records as id events: where a response stops is the
    server's word on the link. The I/O tie's rule, `compact_finish_violations` against the Commit's sampled ids,
    refuses it at every k: output k is the EOS id where the Commit sampled another."""
    stream, full = run.target["stream"], len(run.target["output"])
    sampled, eos_id = _sampled(run.honest), min(run.eos)
    ids = {**run.plan, "compact": None}
    for k in range(full):
        forged = _cut(run.outs, stream, k, active.compact_record(stream, eos_id, "stop"))
        got = cs.certified_io(run.plan, run.setup, run.ins, forged)
        mine = next(x for x in got["records"] if x["stream"] == stream)
        assert got["accepted"] and mine["finish"] == "eos" and len(mine["output"]) == k + 1, k
        assert got["tokens"] == run.honest["tokens"] - (full - 1 - k)
        assert cs.certified_io(ids, run.setup, run.ins, _as_id_events(forged))["accepted"], k
        assert active.compact_finish_violations(sampled, run.eos, _cells(forged), max_tokens=_caps(run.honest)) == \
            [(stream, f"output {k} is id {eos_id} where the Commit sampled {run.target['output'][k]}")], k
