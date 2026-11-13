"""`c2_replay.run`, the replay and linkage of one instrumented run, on a stubbed evaluation: what it records when the replay
cannot run, when this host cannot hold it and when an earlier pair's record is reused, and what `free_programs` and `record`
put on the run's row, with the lines each logs.  CPU only; the replay itself is `tests/check/test_sampled_replay.py`."""
from __future__ import annotations

import json
import logging
import struct
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.c2_replay as C2
from verity_vllm.check.replay.driver import PoolDoesNotFit

SAMPLED_REPLAY, BOUNDARY_LINKAGE = "verity-sweep/sampled-replay/v1", "verity-sweep/boundary-linkage/v1"
CACHED: dict[str, Any] = {"replay_cache": True, "replay_seed": 7, "replay_per_stratum": 1, "replay_max": 0, "replay_k": None}


Record = dict[str, Any]
Cache = dict[str, Record]
Lines = Callable[[], list[str]]


@pytest.fixture
def logged(caplog: pytest.LogCaptureFixture) -> Lines:
    """The lines `c2_replay` has logged so far in this test."""
    caplog.set_level(logging.INFO, logger=C2.__name__)
    return lambda: [r.getMessage() for r in caplog.records if r.name == C2.__name__]


class Spans:
    """The timeline's two calls, kept in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def start(self, name: str, **kw: Any) -> None:
        self.calls.append(("start", name, kw))

    def end(self, name: str, **kw: Any) -> None:
        self.calls.append(("end", name, kw))


def sources(out: Path, pair: int = 0, vc_result: bool | None = None, **args: Any) -> C2.ReplaySources:
    (out / f"binding_map_p{pair}.json").write_text("{}")
    return C2.ReplaySources(
        args=SimpleNamespace(**{"replay_cache": False} | args), out=out, pair=pair, com=SimpleNamespace(name="stub"),
        rc=SimpleNamespace(run_root=bytes(64), weight_registration=None),
        row={"commit": {"value_correspondence": {"method": "by reference", "result": None}}}, reqs=[], sampling={}, tokens={},
        manifest=None, stop_of_record={}, ids_full=None, versions={"git_head": "0" * 40}, components={}, vc_result=vc_result,
        weights_of=lambda prefix: lambda *a: None, weights_check=lambda: {"equal": True}, tl=Spans())


def evaluated(result: bool | None = True) -> tuple[Record, Record]:
    return ({"schema": SAMPLED_REPLAY, "result": result, "grade": "exact", "sample": {"seed": 7, "evaluated": 3, "evaluated_here": 3},
             "strata": {"a": {}, "b": {}}},
            {"schema": BOUNDARY_LINKAGE, "result": True, "pairs": 2, "linked": 2})


def evaluates(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> list[int]:
    """`evaluate` gives ``outcome`` (raised if it is an exception); the pairs it was called for."""
    pairs: list[int] = []

    def evaluate(s: C2.ReplaySources, held: dict[str, Any]) -> Any:
        pairs.append(s.pair)
        held["programs"] = object()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(C2, "evaluate", evaluate)
    return pairs


def test_a_replay_that_cannot_run_is_recorded_as_not_evaluated_and_never_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                                logged: Lines) -> None:
    evaluates(monkeypatch, RuntimeError("no Programs of record"))
    readings(monkeypatch, 4096, 2048, 1024)
    s = sources(tmp_path)
    C2.run(s, {})
    art = tmp_path / "sampled_replay_p0.json"
    doc = json.loads(art.read_text())
    why = "RuntimeError('no Programs of record')"
    assert doc == {"sampled_replay": {"schema": SAMPLED_REPLAY, "method": "sampled-exact-replay", "result": None, "grade": "not-evaluated",
                                      "why": f"sampled replay not run: {why}", "error": why},
                   "boundary_linkage": {"schema": BOUNDARY_LINKAGE, "result": None, "why": f"boundary linkage not run: {why}", "pairs": 0,
                                        "linked": 0}}
    commit = s.row["commit"]
    assert commit["value_correspondence"] == {"method": "sampled-exact-replay(not-evaluated) + boundary-linkage + by reference", "result": None}
    freed = commit["sampled_replay"].pop("programs_freed")
    assert commit["sampled_replay"] == doc["sampled_replay"] | {"artifact": str(art), "strata_n": 0}
    assert commit["boundary_linkage"] == doc["boundary_linkage"]
    assert freed["when"] == "after the sampled-replay + boundary-linkage records, before openings-after-release"
    assert (s.components["sampled_replay"]["result"], s.components["sampled_replay"]["grade"]) == (None, "not-evaluated")
    assert (s.components["boundary_linkage"]["result"], s.components["boundary_linkage"]["checked"]) == (None, 0)
    assert s.tl.calls == [("start", "validate.sampled_replay", {"pair": 0, "cached": False}),
                          ("end", "validate.sampled_replay", {"pair": 0, "ok": None, "n": None, "linkage": None, "reused": False})]
    freed_line, replay_line, linkage_line = logged()
    assert freed_line.startswith("[delta] C2 PROGRAMS FREED (pair 0): RSS 4 -> 2 (gc) -> 1 MiB (malloc_trim ")
    assert freed_line.endswith("): 3 MiB back before the openings-after-release replay")
    assert replay_line == ("[delta] C2 SAMPLED REPLAY (pair 0): NOT-EVALUATED evaluated None equal None mismatch None not-evaluated None of "
                           "population None VUs in None strata (per-stratum None, seed None); not-evaluable None -- sampled replay not run: "
                           f"{why}")
    assert linkage_line == ("[delta] boundary_linkage (C2 BOUNDARY LINKAGE, pair 0): NOT RUN 0/0 (step, request) boundaries linked -- "
                            f"boundary linkage not run: {why}")


def test_a_replay_this_host_cannot_hold_is_raised_and_nothing_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluates(monkeypatch, PoolDoesNotFit("the largest pick does not fit"))
    s = sources(tmp_path)
    with pytest.raises(PoolDoesNotFit, match="the largest pick does not fit"):
        C2.run(s, {})
    assert not (tmp_path / "sampled_replay_p0.json").exists()
    assert s.row == {"commit": {"value_correspondence": {"method": "by reference", "result": None}}} and s.components == {}
    assert s.tl.calls == [("start", "validate.sampled_replay", {"pair": 0, "cached": False})]


def test_a_later_pair_with_the_same_key_reuses_the_record_and_one_with_another_seed_replays(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, logged: Lines) -> None:
    pairs, cache = evaluates(monkeypatch, evaluated()), Cache()
    first, second, third = (sources(tmp_path, 0, **CACHED), sources(tmp_path, 1, **CACHED),
                            sources(tmp_path, 2, **CACHED | {"replay_seed": 8}))
    C2.run(first, cache)
    (key, kept), = cache.items()
    assert pairs == [0] and (kept["pair"], kept["seed"], kept["run_root"]) == (0, 7, bytes(64).hex())
    C2.run(second, cache)
    assert pairs == [0], "the second pair's key is the first's: nothing is evaluated again"
    sr, lk = second.row["commit"]["sampled_replay"], second.row["commit"]["boundary_linkage"]
    assert sr["reused_from"] == lk["reused_from"] and sr["sample"] == {"seed": 7, "evaluated": 3, "evaluated_here": 0}
    assert {k: sr["reused_from"][k] for k in ("pair", "root_equal", "seed", "artifact")} == {
        "pair": 0, "root_equal": True, "seed": 7, "artifact": str(tmp_path / "sampled_replay_p0.json")}
    assert first.row["commit"]["sampled_replay"]["sample"]["evaluated_here"] == 3 and "reused_from" not in first.row["commit"]["sampled_replay"]
    assert kept["shared_by"] == [{"pair": 1, "run_root": bytes(64).hex(), "root_equal": True,
                                  "artifact": str(tmp_path / "sampled_replay_p1.json")}]
    assert second.tl.calls[0][2] == {"pair": 1, "cached": True} and second.tl.calls[1][2]["reused"] is True
    assert second.row["commit"]["value_correspondence"]["result"] is True
    C2.run(third, cache)
    assert pairs == [0, 2] and len(cache) == 2 and "reused_from" not in third.row["commit"]["sampled_replay"]
    said = logged()
    assert ("[delta] C2 SAMPLED REPLAY (pair 1): REUSED pair 0's record (identical key: run root 0000000000000000, seed 7, code 000000000); "
            "no VU evaluated here") in said
    assert "[delta] replay cache: pair 2 key differs from pair 0's in ['seed']: own replay (root 0000000000000000 vs 0000000000000000)" in said
    assert ("[delta] C2 SAMPLED REPLAY (pair 2): EXACT evaluated 3 equal None mismatch None not-evaluated None of population None VUs in "
            "None strata (per-stratum None, seed 7); not-evaluable None") in said
    assert "[delta] boundary_linkage (C2 BOUNDARY LINKAGE, pair 2): OK 2/2 (step, request) boundaries linked" in said


def test_a_run_with_no_cache_key_replays_and_keeps_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, logged: Lines) -> None:
    pairs, cache = evaluates(monkeypatch, evaluated()), Cache()
    s = sources(tmp_path, **CACHED)
    (tmp_path / "binding_map_p0.json").unlink()
    assert C2.cache_key(s) is None
    assert logged() == ["[delta] replay cache: no key for pair 0 (FileNotFoundError(2, 'No such file or directory')); replay runs"]
    C2.run(s, cache)
    assert pairs == [0] and cache == {}
    assert C2.cache_key(sources(tmp_path)) is None, "the cache is off unless asked for"


def test_a_replay_with_no_result_is_not_kept_for_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pairs, cache = evaluates(monkeypatch, evaluated(None)), Cache()
    C2.run(sources(tmp_path, 0, **CACHED), cache)
    C2.run(sources(tmp_path, 1, **CACHED), cache)
    assert pairs == [0, 1] and cache == {}


@pytest.mark.parametrize("entries, live", [(["pouw:ncp-v2"], "pouw:ncp-v2"), (["pous:any", "pouw:ncp-v2-shift24"], "pouw:ncp-v2-shift24"),
                                           (["pouw:another-scheme"], None), (["pous:ncp-v2"], None), ([], None), (None, None)])
def test_a_run_is_live_only_when_a_protocol_entry_s_replay_reads_the_live_run(tmp_path: Path, entries: list[str] | None,
                                                                              live: str | None) -> None:
    protocols = None if entries is None else SimpleNamespace(entries=entries)
    s = sources(tmp_path)
    s.protocols = protocols
    assert C2.live_only_entry(protocols) == live and C2.live_only(s) == live


def readings(monkeypatch: pytest.MonkeyPatch, *kb: int | None) -> None:
    left = list(kb)
    monkeypatch.setattr(C2, "rss_kb", lambda: (left.pop(0), None))


def test_the_freed_programs_are_measured_by_three_readings_of_the_resident_set(monkeypatch: pytest.MonkeyPatch) -> None:
    readings(monkeypatch, 4096, 2048, 1024)
    sr, held = Record(), {"programs": object()}
    C2.free_programs(sr, held, 0)
    trim = sr["programs_freed"].pop("malloc_trim")
    assert held == {} and isinstance(trim, (int, str))
    assert sr["programs_freed"] == {"rss_before_kb": 4096, "rss_after_gc_kb": 2048, "rss_after_kb": 1024, "freed_mib": 3, "freed_by_gc_mib": 2,
                                    "freed_by_malloc_trim_mib": 1,
                                    "when": "after the sampled-replay + boundary-linkage records, before openings-after-release"}


def test_the_programs_are_freed_where_the_resident_set_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    readings(monkeypatch, None, None, None)
    sr, held = Record(), {"programs": object()}
    C2.free_programs(sr, held, 0)
    assert held == {}
    assert {k: v for k, v in sr["programs_freed"].items() if k not in ("malloc_trim", "when")} == {
        "rss_before_kb": None, "rss_after_gc_kb": None, "rss_after_kb": None, "freed_mib": None, "freed_by_gc_mib": None,
        "freed_by_malloc_trim_mib": None}


@pytest.mark.parametrize("replay, linkage, before, after", [
    (True, True, None, True), (True, True, True, True), (True, True, False, None), (False, True, None, False), (True, False, None, False),
    (None, True, None, None), (True, None, None, None)])
def test_the_value_correspondence_passes_only_when_the_replay_the_linkage_and_what_came_before_do(
        tmp_path: Path, logged: Lines, replay: bool | None, linkage: bool | None, before: bool | None, after: bool | None) -> None:
    s = sources(tmp_path, vc_result=before)
    sr, lk = evaluated(replay)
    sr |= {"sample": {"evaluated": 3, "equal": 2, "mismatch_n": 1}, "mismatch_by": {"by_family_request_phase": {"attention|r0|decode": 1}},
           "moe_planes": {"alias_check": {"compared": 4, "identical_n": 1, "result": False, "per_layer": {"0": {}}, "why": "one triple aliases"}}}
    C2.record(s, sr, lk | {"result": linkage}, tmp_path / "sampled_replay_p0.json")
    assert s.row["commit"]["value_correspondence"] == {"method": "sampled-exact-replay(exact) + boundary-linkage + by reference", "result": after}
    assert "strata" not in s.row["commit"]["sampled_replay"] and s.row["commit"]["sampled_replay"]["strata_n"] == 2
    assert (s.components["sampled_replay"]["n"], s.components["sampled_replay"]["equal"], s.components["sampled_replay"]["mismatch_n"]) == (3, 2, 1)
    assert (s.components["boundary_linkage"]["result"], s.components["boundary_linkage"]["linked"]) == (linkage, 2)
    linked = "OK" if linkage else "NOT RUN" if linkage is None else "FAIL"
    assert logged() == [
        ("[delta] C2 SAMPLED REPLAY (pair 0): EXACT evaluated 3 equal 2 mismatch 1 not-evaluated None of population None VUs in None strata "
         "(per-stratum None, seed None); not-evaluable None"),
        "[delta] C2 SAMPLED REPLAY (pair 0): mismatches by family|request|phase {'attention|r0|decode': 1}",
        f"[delta] boundary_linkage (C2 BOUNDARY LINKAGE, pair 0): {linked} 2/2 (step, request) boundaries linked",
        ("[delta] C2 MOE PLANE ALIAS CHECK (pair 0): ALIAS up != down bytes on 3/4 (request, step, layer) triples, 1 layers, plane words None"
         " -- one triple aliases")]


def test_the_live_replay_reads_a_registered_weight_from_the_engine_s_model() -> None:
    """`live_weights` is the in-process replay's reader of registered inputs: a field's words from the model's own tensor,
    under the Program's prefix or without it, and None for a field the model does not hold."""
    torch = pytest.importorskip("torch")
    values = [1.0, -2.0, 0.5, 3.0, -0.25, 8.0]
    model = torch.nn.Linear(2, 3, bias=False)
    model.weight = torch.nn.Parameter(torch.tensor(values).reshape(3, 2))
    words = list(struct.unpack("<6I", struct.pack("<6f", *values)))
    read = C2.live_weights(model, "model.")
    assert read("model.weight", 0, 6).tolist() == words and read("weight", 2, 5).tolist() == words[2:5]
    assert read("model.bias", 0, 1) is None and read("model.weight", 0, 7) is None
