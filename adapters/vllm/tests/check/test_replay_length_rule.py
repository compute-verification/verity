"""The replay of a request's length rule (`check.replay.checks`): the `StopCheck_v1` its Program returns as its last check,
evaluated by its row kernel (`numerics.stop_rows`) from the sampled ids committed at the request's own steps.  An honest finish
is accepted; a stream cut short before its first stop, or run past one, is refused by name.  The rule closes the request's last
step (`manifest.format.segment_steps`), so a request served through it has the rule evaluated, never counted as not executed, and
the Match reads no instance of it (`instance_sequence.RootView`), since the runtime never runs it."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from tests.program.predict import test_steps as ST
from verity.definitions.numerics.rows import SAMPLED_MEMBER
from verity_vllm.check.match.instance_sequence import RootView
from verity_vllm.check.match.program_loaders import check_rows
from verity_vllm.check.replay import checks as CK, index as IX
from verity_vllm.check.replay.index import SAMPLER_OP
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.program.instances_file import read_instances
from verity_vllm.program.predict.naming import write_build
from verity_vllm.program.predict.steps import compile_steps
from verity_vllm.query.program_view import from_instances
from verity_vllm.query.required import request_manifest

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")
RES = {"program": {"digest": "d" * 64, "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}}}


class Sampled:
    """A committed store holding one request's sampled ids, `ids[s]` at its step s."""

    def __init__(self, ids: list[int]):
        self.ids = ids

    def words(self, rid: str, step: int, op_path: str, member: str, lo: int, hi: int):
        if (op_path, member, lo, hi) != (SAMPLER_OP, SAMPLED_MEMBER, 0, 1) or not 0 <= step < len(self.ids):
            return None, f"nothing committed at {op_path}/{member} [{lo}, {hi}), step {step}"
        return np.array([self.ids[step]], dtype=np.int64), None


def _build(tmp_path, finish: int):
    """The named Build of `ST._stops`' request that finished on `finish`, 1 a stop id or 2 its max_tokens (T = 2: three outputs)."""
    reg, (advice, ingress) = ST._registered(), ST._stops()
    c = compile_steps(reg, advice, ingress, named=True)
    u = next(u for u in c.requests if u.stop["finish"] == finish)
    d = tmp_path / str(finish)
    write_build(str(d), c.programs[u.digest], c.named[u.digest])
    return d


def _checks(d, ids: list[int], n_exec: int = 3) -> dict:
    pi = IX.ProgramIndex(read_instances(str(d / "instances.json.gz"), expand=False), program_dir=str(d))
    return CK.checks_of({"r0": pi}, Sampled(ids), lambda *a: None, {"r0": 0}, {"r0": {"n_exec": n_exec, "cap": 3}})


@pytest.mark.parametrize("finish, ids, result", [
    (1, [11, 12, 255], True),            # an honest eos finish
    (2, [11, 12, 13], True),             # an honest length finish
    (1, [11, 12, 13], False),            # cut short before its first stop
    (1, [11, 255, 255], False),          # run past a stop to finish on another
    (2, [11, 255, 13], False),           # run past a stop to its cap
])
def test_the_replay_evaluates_the_length_rule_from_the_committed_ids(tmp_path, finish, ids, result):
    rec = _checks(_build(tmp_path, finish), ids)
    assert (rec["result"], rec["n"], rec["not_executed"]) == (result, 1, 0)
    if not result:
        assert rec["why"].startswith(f"check is 0, not 1: the length rule (StopCheck_v1 of StopRule_v1): the outputs stop at the last "
                                     f"one, finish {finish}")


def test_the_length_rule_closes_the_last_step(tmp_path):
    """The rule's row is in step 2, the request's last: three steps, not a fourth; executed through step 1 only, it is not evaluated;
    the manifest requires no capture of it (the runtime never runs it) and lists it unreturned in step 2; and the Match's instance
    sequence has no instance of it."""
    d = _build(tmp_path, 1)
    inst = read_instances(str(d / "instances.json.gz"), expand=False)
    pi = IX.ProgramIndex(inst, program_dir=str(d))
    (_what, i, _lo, _hi), = CK.check_rows(pi)
    assert (pi.n_steps, pi.step_of(i), i) == (3, 2, len(inst["rows"]) - 1)
    assert (lambda r: (r["accepted"], r["not_executed"]))(_checks(d, [11, 12, 255], n_exec=2)) == (0, 1)
    P = from_instances(read_instances(str(d / "instances.json.gz")))
    m = request_manifest(P, Correspondence.of(P, None), RES, request_id="r0").manifest
    assert m["steps"] == [0, 1, 2] and not any(r.get("spec", "").startswith("StopCheck") for r in m["identities"])
    assert [(r["spec"], r["step"]) for r in m["unreturned"]] == [("StopCheck_v1", 2)]
    with gzip.open(d / "descriptor.json.gz", "rt") as f:
        desc = json.load(f)
    assert check_rows(str(d)) == [i]
    for compact in (False, True):
        view = RootView("built", desc, {}, tok_node=0, compact=compact, checks=check_rows(str(d)))
        assert len(view.inst) == len(inst["rows"]) - 1 and len(view.steps) == 3
        assert not any(r["spec"].startswith("StopCheck") for r in view.inst)


def test_an_id_word_past_32_bits_is_not_evaluated(tmp_path):
    """No I32 output holds a committed id of 2^32 + 255: the row kernel declines it, so the check is unevaluated, never wrapped onto
    the stop id."""
    rec = _checks(_build(tmp_path, 1), [11, 12, (1 << 32) + 255])
    assert rec["result"] is None and "past 32 bits" in rec["why"]
