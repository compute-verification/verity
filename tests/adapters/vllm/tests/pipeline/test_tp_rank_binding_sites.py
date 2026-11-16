"""`verity_vllm.pipeline.tp.rank_worker`: where a rank's binding takes its collective sites from when the rank holds no
collective source (`TP2CaptureWorkerExtension._tp2_binding`).  In a world of more than one rank the sites are the model's
structure: the rank reads them off its engine's model, hooking nothing, so a collector that committed no partial fails its
coverage by every site.  A site that is also a module the committer hooked owes its member beside the module's own.  And a
rank whose binding modules cannot be imported says so by the error and binds nothing.  CPU-only: the committer is the
library's host committer over a stream the test names, and the model's two site modules carry the class names a site is
found by."""
from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from verity_vllm.commit.committer.native_host import NativeHostCommitter
from verity_vllm.pipeline.tp import rank_worker as RW

PROGRAM = b"the rank's program"
PD = hashlib.sha512(PROGRAM).hexdigest()
RANK, STEPS = 1, 3
#: the member each collective site of the stand-in model owes on rank 1: the logits site its shard, the other its part
MEMBERS = [("m.proj", f"part_rank{RANK}"), ("m.logits", f"shard_rank{RANK}")]
ABSENT: Any = object()
BINDER = "verity_vllm.query.manifest.bind"


class RowParallelLinear(torch.nn.Module):
    """A module of the class name a row-parallel site is found by; it states no size of its own, so it reduces over the world's."""


class LogitsProcessor(torch.nn.Module):
    """A module of the class name the logits site is found by."""


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj, self.act, self.logits = RowParallelLinear(), torch.nn.Identity(), LogitsProcessor()


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m = _Block()


def _bind(members: list[tuple[str, str]], *, hooked: list[str] | None = None, **state: Any) -> tuple[dict[str, Any], dict[str, Any], _Model]:
    """(the binding of rank 1 of two over a run of three steps that committed ``members`` at each, its state after, its model).
    The rank holds no collective source unless ``state`` gives one; ``hooked`` names the model's modules the committer hooked."""
    com = NativeHostCommitter(None, run_id="run", hash_threads=1, program_digest=hashlib.sha512(PROGRAM).digest())
    rc = com.commit_offline([[(f"{op}/{member}", bytes([step + 1]) * (100 + 50 * i)) for i, (op, member) in enumerate(members)]
                             for step in range(STEPS)])
    model = _Model()
    com._modules = [(name, model.get_submodule(name)) for name in hooked or []]
    rank = RW.TP2CaptureWorkerExtension()
    given = {"program_digest": PD, "com": com, "rank": RANK, "world": 2, "eng": SimpleNamespace(model=model), **state}
    rank._tp2_commit = {k: v for k, v in given.items() if v is not ABSENT}
    answer = rank._tp2_binding(rc)
    assert answer is not None
    return answer, rank._tp2_commit, model


def _population(st: dict[str, Any]) -> set[tuple[int, str, str]]:
    """(step, op_path, member) of every identity the rank held its map to."""
    return {(i[2], i[4], i[5]) for i in st["pop"]}


def _over_the_steps(members: list[tuple[str, str]]) -> set[tuple[int, str, str]]:
    return {(step, op, member) for step in range(STEPS) for op, member in members}


@pytest.mark.parametrize("source", [ABSENT, None], ids=["absent", "none"])
def test_a_rank_with_no_collective_source_reads_its_sites_off_its_engines_model_and_hooks_nothing(source):
    answer, st, model = _bind(MEMBERS, tps=source)
    assert "error" not in answer and answer["coverage"]["ok"] is True
    assert (answer["population"]["tp_partial_members"], answer["population"]["n"], answer["structure_modules"]) == (2, 6, 2)
    assert _population(st) == _over_the_steps(MEMBERS)
    assert st.get("tps", ABSENT) is source                                          # the sites were read; the rank still holds no source
    assert sum(len(m._forward_pre_hooks) + len(m._forward_hooks) for m in model.modules()) == 0


def test_a_collector_that_committed_no_partial_fails_its_coverage_by_every_site_of_the_model():
    """The sites are required by the model's structure, not by what a source bound: outputs alone leave both members missing."""
    answer, st, _ = _bind([("m.proj", "out"), ("m.logits", "out")])
    assert _population(st) == _over_the_steps(MEMBERS) and answer["population"]["tp_partial_members"] == 2
    assert answer["coverage"]["ok"] is False and answer["coverage"]["missing_n"] == 2 * STEPS
    assert {(m["op_path"], m["output_member"]) for m in answer["coverage"]["missing"]} == set(MEMBERS)


def test_a_model_with_no_site_owes_no_collective_member_and_a_population_of_nothing_is_not_covered():
    answer, st, _ = _bind([("m.act", "out")], eng=SimpleNamespace(model=torch.nn.Sequential(torch.nn.Identity())))
    assert (answer["population"]["tp_partial_members"], answer["population"]["n"], answer["structure_modules"]) == (0, 0, 0)
    assert _population(st) == set() and (answer["coverage"]["ok"], answer["coverage"]["expected_n"]) == (False, 0)


def test_a_site_that_is_a_module_the_committer_hooked_owes_its_member_beside_the_modules_own():
    """The committer hooked `m.proj`, so the structure already holds it with the member its class owes; the site adds the rank's
    part to that module and replaces nothing, and the logits site, which the committer did not hook, is a module of its own."""
    own = [("m.proj", "0"), *MEMBERS]                                               # a linear's own member is its output, "0"
    answer, st, _ = _bind(own, hooked=["m.proj"])
    assert "error" not in answer and answer["coverage"]["ok"] is True and _population(st) == _over_the_steps(own)
    assert (answer["population"]["tp_partial_members"], answer["population"]["n"], answer["structure_modules"]) == (2, 9, 2)
    answer, st, _ = _bind(MEMBERS, hooked=["m.proj"])
    assert _population(st) == _over_the_steps(own) and answer["coverage"]["ok"] is False
    assert {(m["op_path"], m["output_member"]) for m in answer["coverage"]["missing"]} == {("m.proj", "0")}


def test_binding_modules_that_cannot_be_imported_are_said_by_the_error_and_nothing_is_bound(monkeypatch):
    monkeypatch.setitem(sys.modules, BINDER, None)                                  # what an import of it finds when it is not there
    answer, st, _ = _bind(MEMBERS)
    assert list(answer) == ["skipped"] and answer["skipped"].startswith("binding module unavailable: ModuleNotFoundError(")
    assert BINDER in answer["skipped"] and "bmap" not in st and "pop" not in st
