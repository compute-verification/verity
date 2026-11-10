"""The replay of a stochastic request's seed rule (`check.replay.checks`): the `SeedCheck_v2` its Program returns beside its length
rule, evaluated from the sampler's seed word committed at the rule's step and the request's certified inputs (`checks_of`'s
``certified``). The seed its body names, or the one derived from the window's seed salt and its id, is accepted; any other seed
the sampler read is refused by name; an unseeded request with no seed salt, or one with no certified inputs, isn't evaluated."""
from __future__ import annotations

import numpy as np
import pytest

from tests.program.predict import test_steps as ST
from verity.definitions.numerics.rows import SAMPLED_MEMBER
from verity_vllm.check.replay import checks as CK, index as IX
from verity_vllm.check.replay.index import SAMPLER_OP, SEED_PARAM
from verity_vllm.engine.served_seed import derived
from verity_vllm.program.instances_file import read_instances
from verity_vllm.program.predict.naming import write_build
from verity_vllm.program.predict.steps import compile_steps

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")


class Served:
    """A committed store holding one request's sampled ids, `ids[s]` at its step s, and the seed its sampler read at each."""

    def __init__(self, ids: list[int], seed: int):
        self.ids, self.seed = ids, seed

    def words(self, rid: str, step: int, op_path: str, member: str, lo: int, hi: int):
        if op_path != SAMPLER_OP or (lo, hi) != (0, 1) or member not in (SAMPLED_MEMBER, SEED_PARAM) or not 0 <= step < len(self.ids):
            return None, f"nothing committed at {op_path}/{member} [{lo}, {hi}), step {step}"
        return np.array([self.ids[step] if member == SAMPLED_MEMBER else self.seed], dtype=np.int64), None


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> IX.ProgramIndex:
    """The named Build of one seeded top-p request `q0` (LP 5, three outputs to its max_tokens)."""
    advice, ingress = ST._session([(5, 3, ST.STOCH)], [[(0, 0, 5)], [(0, 5, 1)], [(0, 6, 1)]])
    c = compile_steps(ST._registered(), advice, ingress, named=True)
    (u,) = c.requests
    d = tmp_path_factory.mktemp("seeded")
    write_build(str(d), c.programs[u.digest], c.named[u.digest])
    return IX.ProgramIndex(read_instances(str(d / "instances.json.gz"), expand=False), program_dir=str(d))


def _seed(pi: IX.ProgramIndex, served: int, certified: dict | None) -> dict:
    rec = CK.checks_of({"q0": pi}, Served([11, 12, 13], served), lambda *a: None, {"q0": 0}, None,
                       None if certified is None else {"q0": certified})
    (seed,) = [r for r in rec["refused"] + rec["not_evaluated"] if r["check"].startswith("the seed rule")] or [None]
    return {"result": rec["result"], "accepted": rec["accepted"], "n": rec["n"], "seed": seed}


@pytest.mark.slow
def test_the_replay_evaluates_the_seed_rule_from_the_committed_seed_and_the_certified_inputs(built):
    """Both checks evaluated in the request's last step: its body's seed and the seed derived from a salt pass; a seed the sampler
    read that is neither is refused as the seed rule; with no salt for an unseeded request, or no certified inputs, it is named
    not evaluated, never passed."""
    rows = CK.check_rows(built)
    assert [w.split(" (")[0] for w, *_ in rows] == ["the seed rule", "the length rule"]
    assert {built.step_of(i) for _, i, _, _ in rows} == {2}
    assert _seed(built, 11, {"request_id": "q0", "seed": 11}) == {"result": True, "accepted": 2, "n": 2, "seed": None}
    salt = bytes(range(64))
    assert _seed(built, derived(salt, "q0"), {"request_id": "q0", "seed": None, "seed_salt": salt})["result"] is True
    forged = _seed(built, 12, {"request_id": "q0", "seed": 11})
    assert forged["result"] is False and forged["seed"]["why"].startswith("check is 0, not 1: the seed rule (SeedCheck_v2{COIN=64,RID=2}")
    assert _seed(built, derived(salt, "q0"), {"request_id": "q0", "seed": None, "seed_salt": bytes(64)})["result"] is False
    for certified, why in (({"request_id": "q0", "seed": None}, "request q0 is unseeded, and the replay has no window seed salt"),
                           (None, "request q0's certified seed and id aren't given to the replay"),
                           ({"request_id": "q00", "seed": 11}, "the seed rule reads an id of 2 bytes, and request q0's certified id")):
        got = _seed(built, 11, certified)
        assert got["result"] is None and got["accepted"] == 1 and got["seed"]["why"].startswith(why)
