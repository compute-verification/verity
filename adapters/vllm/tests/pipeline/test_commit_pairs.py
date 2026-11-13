"""`verity_vllm.pipeline.commit`: what `main` does with one run of a pair after the run has ended, where the run's committer,
its binding record or an option makes it more than `tests.pipeline.test_commit_main`'s plain run.

Real: all that `tests.pipeline.test_commit_main` runs for real, the run's verdict where a test gives none
(`check.commit_pass.commit_verdict`), and the retained bytes' digests (`commit.tensor_digests`).

Stood in: that module's stand-ins; the committer, `Kept`, which is its committer with a retention mode, families left out
of what it retains, a release of its staging, a fault of the retained bytes and retained blocks, each only where a test gives
it; the openings after the release (`commit.openings_after_release` replays the workload on the engine), which answer what
a test states; and the binding record where a test states one.

Held: what the committer is asked, in order; the challenged positions `main` asks it about and the ones it is handed to
open; the run's row; every line `main` prints for what the test is about; the files it writes; and how the Commit ends.
A run's challenge is this process's own flips after the run's root, so a test reads the positions from what the committer
was asked and states the rest from them.
CPU only."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("torch")

import torch

from tests.pipeline import test_commit_main as M
from tests.pipeline import test_commit_weights_of_record as W
from tests.pipeline.test_commit_main import RUN, WARMUP, Committer, Ended, Record, run
from verity_vllm.pipeline import commit

#: `test_commit_main`'s fixture and `test_commit_weights_of_record`'s, under their names in this module.
stand_ins, pinned = M.stand_ins, W.pinned

KIND = f"plugin:{__name__}:committer"
MIB = 1 << 20
#: A run's calls where its openings are not asked for in the run.
RUN_UNOPENED = [c for c in RUN if not c.startswith("open ")]
BINDING_LINE = ("[delta] BINDING (pair {pair}): map 3 entries digest mmmmmmmmmmmmmmmm; population 3 (digest pppppppppppppppp); coverage {ok} bound "
                "3/3 missing {missing} extra 0; population openings 0/0 (unbound 0, late step True)")
NONE_KEPT = "retain=none: values by replay (openings_after_release)"
NO_RECORD = "openings skipped ({skipped}) and no openings-after-release (replay) record"
#: The real verdict's word on a run it finds nothing else against: these Commits name no Program to replay.
NO_C2 = "C2 of record not established: sampled replay None -- not run (no --program-dir / --programs-root); the row's Commit is C1 + C3 only"
#: The openings after the release as far as `main` reads them: every pick opened from the host's copy.
AFTER_RELEASE: Record = {"source": "host", "n": 4, "verified": 4, "freed": {"raw_device_bytes": 3 * MIB, "levels_device_bytes": MIB},
                         "forced_n": 1, "forced_families": ["instance_outputs"], "host_n": 4, "replay_n": 0, "replay_families": [],
                         "replay_wall_s": None, "replay_root_equal": None, "retained_device_after": 2 * MIB, "replay_retain": None,
                         "host_rss_gb_after": 1.5, "host_rss_hwm_gb": 2.5}
AFTER_RELEASE_LINE = ("[delta] OPENINGS AFTER RELEASE (pair {pair}): freed 3 MiB raw + 1 MiB levels from the device; source {source}; 4/4 verified "
                      "(1 forced: ['instance_outputs']; host 4 / replay {replay_n} {families} in {wall} s); replay root equal {equal}; device retained "
                      "after 2 MiB; replay retain None; host rss after 1.5 GB (process VmHWM 2.5 GB)")
REPLAY_METHOD = " + replay (openings_after_release.replay_root_equal)"
REPLAY_REFERENCE = "replay of the committed values (openings_after_release)"
REPLAY_DIFFERS = "replay root != committed root"
SAME_DIGESTS_LINE = ("[delta] TENSOR DIGESTS pair 1 vs pair 0: 0 steps differ []; differing tensors by family {{}}; steps whose digest MULTISET is "
                     "equal (pure permutation) 0; 0.0s")


class Kept(Committer):
    """`Committer` with what `main` reads of a committer's retention after a run.  ``retain`` is its mode; ``exclude`` the
    families it leaves out of the host's copy, of which the host holds every leaf (``all_kept``) or every leaf but the first
    one a run asks about, so that each run's challenge leaves one to the replay; ``releases`` gives it a staging to release;
    ``flip`` is what its fault of the retained bytes reports; ``blocks`` and ``layouts`` are its retained blocks and their
    members by step; ``sources`` are its sources; ``dumps`` is what it dumps of its layouts and ``step_rows`` the requests'
    rows it recorded by step.  ``asked`` keeps the leaves `main` asked about and ``opened`` the ones it was handed to open, a
    list a call."""

    #: The leaves `main` leaves to the replay, which it writes on its committer: a `Kept` has none until then.
    _oar_deferred: list[tuple[int, int]]

    def __init__(self, eng: SimpleNamespace, run_id: str, *, retain: str = "host", exclude: tuple[str, ...] = (), all_kept: bool = True,
                 releases: bool = False, flip: Record | None = None, blocks: Any = None, layouts: Any = None, sources: Any = None,
                 dumps: Any = None, step_rows: Any = None, **kw: Any) -> None:
        super().__init__(eng, run_id, **kw)
        if sources is not None:
            self.sources = sources
        if dumps is not None:
            def layouts_dump() -> Any:
                self.calls.append("layouts_dump")
                return copy.deepcopy(dumps)

            self.layouts_dump = layouts_dump
        if step_rows is not None:
            self.step_rows = step_rows
        self.retain, self.retain_exclude, self.all_kept, self.flip = retain, frozenset(exclude), all_kept, flip
        self.first: tuple[int, int] | None = None
        self.asked: list[tuple[int, int]] = []
        self.opened: list[list[tuple[int, int]]] = []
        self.faults: list[str] = []
        if releases:
            self.release_staging = lambda **kw: pytest.fail("the stand-in's staging is released by `openings_after_release` alone")
        if blocks is not None:
            self._gpu_blocks, self._layouts = blocks, layouts

    def reset(self) -> None:
        super().reset()
        self.first = None

    def leaf_retained(self, step: int, leaf: int) -> bool:
        self.asked.append((step, leaf))
        self.first = self.first or (step, leaf)
        return self.all_kept or (step, leaf) != self.first

    def open(self, sample: list[tuple[int, int]]) -> list[Any]:
        self.opened.append(list(sample))
        return super().open(sample)

    def fault_retained_flip(self, spec: str) -> Record:
        self.faults.append(spec)
        return dict(self.flip or {})


def committer(eng: SimpleNamespace, run_id: str) -> Kept:
    """``--committer plugin:<this module>:committer``."""
    com = Kept(eng, run_id, **M.MAKE)
    M.MADE.append(com)
    return com


def made() -> list[Kept]:
    """`M.MADE` as what this module's `committer` put there."""
    return cast("list[Kept]", M.MADE)


def kept(tmp_path: Any, capsys: Any, *argv: str) -> Ended:
    """`run` with the committer of this module."""
    return run(tmp_path, capsys, "--committer", KIND, *argv)


def instrumented(ended: Ended) -> list[Record]:
    return [row for row in ended.rows() if row["arm"] != "control"]


def lines(ended: Ended, *words: str) -> list[str]:
    """What `main` printed that holds one of ``words``, in order."""
    return [p for p in ended.printed if any(w in p for w in words)]


def with_coverage(**coverage: Any) -> Record:
    return {**copy.deepcopy(M.BINDING), "coverage": {**M.BINDING["coverage"], **coverage}}


@pytest.mark.parametrize(("coverage", "more"), [
    ({"composition": {"ok": True, "checked_steps": 3}}, ["[delta] COMPOSITION (pair {pair}): OK engine steps 3 problem steps 0"]),
    ({"composition": {"ok": False, "checked_steps": 3, "problem_steps_n": 2, "skipped": "(1 step skipped)", "error": "step 1 holds 3 rows"}},
     ["[delta] COMPOSITION (pair {pair}): FAIL engine steps 3 problem steps 2 (1 step skipped) -- step 1 holds 3 rows"]),
    ({"composition": {}}, ["[delta] COMPOSITION (pair {pair}): FAIL engine steps None problem steps 0"]),
    ({"ok": False, "error": "population not derived"}, ["[delta] required_value_coverage: identity coverage ERROR (pair {pair}): population not derived"]),
    ({"ok": False, "missing_n": 4, "missing": [{"step": i} for i in range(4)]}, ["[delta]   first missing: [{{'step': 0}}, {{'step': 1}}, {{'step': 2}}]"]),
], ids=["composition-ok", "composition-fails", "composition-says-nothing", "coverage-error", "missing"])
def test_a_binding_records_composition_error_and_missing_are_each_printed_under_its_line(tmp_path, stand_ins, capsys, monkeypatch, coverage, more):
    """The binding record is the run's as it was given; `main` prints its line, then one for a composition it carries, one
    for a coverage error and one for the first three it misses."""
    monkeypatch.setattr(commit, "binding_record", lambda com, rc, reqs, sampling, args, pair, **kw: with_coverage(**coverage))
    stand_ins.verdicts = [(True, None)]
    ended = run(tmp_path, capsys)
    binding = with_coverage(**coverage)
    ok, missing = "OK" if binding["coverage"]["ok"] else "FAIL", binding["coverage"]["missing_n"]
    assert ended.exit is None and [row["commit"]["binding"] for row in instrumented(ended)] == [binding, binding]
    assert lines(ended, "BINDING", "COMPOSITION", "required_value_coverage", "first missing") == [
        line.format(pair=pair) for pair in (0, 1) for line in [BINDING_LINE.format(pair=pair, ok=ok, missing=missing), *more]]


def test_a_binding_record_that_raises_is_recorded_with_where_it_raised_and_fails_the_run(tmp_path, stand_ins, capsys, monkeypatch):
    """The record is then the error, the frames it came through and the rules' version, the row names no rules version of its
    own, and the real verdict fails the run on the map that was not built."""
    def binding_record(com, rc, reqs, sampling, args, pair, **kw):
        raise ValueError(f"no map for pair {pair}")

    monkeypatch.setattr(commit, "binding_record", binding_record)
    ended = run(tmp_path, capsys)
    rows = instrumented(ended)
    sites = [row["commit"]["binding"]["site"] for row in rows]
    assert [row["commit"]["binding"] for row in rows] == [
        {"error": f"ValueError('no map for pair {pair}')", "site": site, "rules_version": "r14-v1"} for pair, site in enumerate(sites)]
    assert all(site.endswith("in binding_record") and __file__ in site and len(site) <= 400 for site in sites)
    assert ["rules_version" in row["commit"] for row in rows] == [False, False]
    assert lines(ended, "BINDING") == [f"[delta] BINDING ERROR (pair {pair}): ValueError('no map for pair {pair}') at {site}"
                                       for pair, site in enumerate(sites)]
    whys = [f"binding map not built: ValueError('no map for pair {pair}')" for pair in (0, 1)]
    assert stand_ins.answers == [(False, why) for why in whys]
    assert ended.exit == 3 and ended.printed[-1] == M.FAIL_LINE.format(failed=2, n=2, pair=0, arm="instrumented", why=whys[0])


def test_a_committer_that_retains_nothing_is_refused_where_nothing_opens_after_the_release(tmp_path, stand_ins, capsys):
    """``--no-open-after-release`` with a committer whose values are nowhere after the run: the first instrumented run ends
    the Commit with the option's sentence, before any opening is asked for."""
    M.MAKE.update(retain="none")
    ended = kept(tmp_path, capsys, "--no-open-after-release")
    [com] = made()
    assert ended.exit == "--retain none needs --open-after-release (the only opening path is replay)"
    assert com.calls == ["register_weights", *WARMUP, "reset", "install", "finalize", "uninstall"] and com.opened == []
    assert not (ended.out / "runs.jsonl").exists() and not (ended.out / "verdict.json").exists()


def test_a_committer_that_retains_nothing_and_releases_nothing_fails_each_run_on_its_skipped_openings(tmp_path, stand_ins, capsys):
    """The openings are left to the ones after the release, which a committer with no staging to release never makes: the
    row says they were skipped and the real verdict fails the run on that."""
    M.MAKE.update(retain="none", declare=M.CLASSES)
    ended = kept(tmp_path, capsys)
    [com] = made()
    why = NO_RECORD.format(skipped=NONE_KEPT)
    assert com.calls == ["register_weights", *WARMUP, *RUN_UNOPENED, *RUN_UNOPENED, "close"] and com.opened == []
    for row in instrumented(ended):
        assert "openings_after_release" not in row["commit"]
        assert row["commit"]["openings"] == {"n": 0, "verified": 0, "generate_s": 0.0, "verify_s": 0.0, "skipped": NONE_KEPT, "deferred_replay_n": 0,
                                             "challenge_n": 4, "deferred_positions": [], "deferred_digest": None, "bytes_mean": None, "failed": []}
        assert (row["commit"]["commit_pass"], row["commit"]["commit_fail_reason"]) == (False, why)
    assert ended.exit == 3 and ended.printed[-1] == M.FAIL_LINE.format(failed=2, n=2, pair=0, arm="instrumented", why=why)


def test_excluded_families_that_hold_no_challenged_leaf_change_nothing_of_the_run(tmp_path, stand_ins, capsys):
    """Every challenged leaf is asked about twice, each is on the host, and all are opened in the run: the row is the plain
    run's, with nothing left to the openings after the release."""
    M.MAKE.update(exclude=("fa3_hidden_m1",))
    stand_ins.verdicts = [(True, None)]
    ended = kept(tmp_path, capsys, "--no-open-after-release")
    [com] = made()
    sample_0, sample_1 = com.opened
    assert ended.exit is None and com.calls == ["register_weights", *WARMUP, *RUN, *RUN, "close"]
    assert len(sample_0) == len(sample_1) == 4 and com.asked == [*sample_0, *sample_0, *sample_1, *sample_1]
    assert not hasattr(com, "_oar_deferred")
    assert M.less_coins(ended.rows()) == [M.control_row(tmp_path, 0, 0), M.instrumented_row(tmp_path, 0, 1, com.roots[1], kind=KIND),
                                          M.instrumented_row(tmp_path, 1, 0, com.roots[2], kind=KIND), M.control_row(tmp_path, 1, 1)]


def split(com: Kept, run: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """The challenge of ``com``'s run ``run`` as `main` splits it, read from what it asked: the leaves on the host and the
    ones left to the replay, which are the picks of the first leaf it asked about."""
    sample = com.asked[8 * run:8 * run + 4]
    assert len(sample) == 4 and com.asked[8 * run + 4:8 * run + 8] == sample
    return [p for p in sample if p != sample[0]], [p for p in sample if p == sample[0]]


def test_a_challenged_leaf_of_an_excluded_family_is_refused_where_nothing_opens_after_the_release(tmp_path, stand_ins, capsys):
    """The host's share is opened first; the leaves left to the replay are put on the committer for the openings after the
    release, and with ``--no-open-after-release`` the Commit ends there with the option's sentence."""
    M.MAKE.update(exclude=("fa3_hidden_m1",), all_kept=False)
    ended = kept(tmp_path, capsys, "--no-open-after-release")
    [com] = made()
    held, left = split(com, 0)
    assert ended.exit == "--retain-exclude needs --open-after-release (the excluded families open by replay)"
    assert com.opened == [held] and left and com._oar_deferred == left and len(com.asked) == 8
    assert com.calls == ["register_weights", *WARMUP, "reset", "install", "finalize", "uninstall", f"open {len(held)}"]


def after_release(s: SimpleNamespace, monkeypatch: Any, **answer: Any) -> Record:
    """`commit.openings_after_release` stood in: it answers `AFTER_RELEASE` with ``answer``, and ``s.released`` keeps what it
    was handed, a call a dict, with the left leaves the committer carried then."""
    doc = {**AFTER_RELEASE, **answer}
    s.released = []

    def openings_after_release(com, rc, eng, reqs, sampling, base, args, pair, tokens_ref=None):
        assert eng is s.engine and isinstance(args, commit.Options) and [r["request_id"] for r in reqs] == ["a", "b"]
        carried = list(getattr(com, "_oar_deferred", []))
        s.released.append({"com": com, "root": rc.run_root.hex(), "pair": pair, "tokens_ref": tokens_ref, "deferred": carried})
        return copy.deepcopy(doc)

    monkeypatch.setattr(commit, "openings_after_release", openings_after_release)
    return doc


def test_the_leaves_left_to_the_replay_are_recorded_by_position_and_carried_to_the_openings_after_the_release(tmp_path, stand_ins, capsys, monkeypatch):
    """A challenged leaf the host does not hold: the run opens the host's share and records how many it left and which, under
    the keys the verdict reads them by, and the committer carries every run's left leaves to the openings after the release."""
    M.MAKE.update(exclude=("fa3_hidden_m1", "fa2_hidden_m1"), all_kept=False, releases=True)
    doc = after_release(stand_ins, monkeypatch, source="host+replay", replay_n=1, replay_families=["fa3_hidden_m1"], replay_wall_s=0.5,
                        replay_root_equal=True)
    stand_ins.verdicts = [(True, None)]
    ended = kept(tmp_path, capsys)
    [com] = made()
    splits = [split(com, 0), split(com, 1)]
    carried = [splits[0][1], [*splits[0][1], *splits[1][1]]]
    assert ended.exit is None and com.opened == [held for held, _ in splits] and len(com.asked) == 16 and com._oar_deferred == carried[1]
    assert [r["deferred"] for r in stand_ins.released] == carried
    assert com.calls == ["register_weights", *WARMUP, *[c if not c.startswith("open ") else f"open {len(held)}" for held, _ in splits for c in RUN], "close"]
    for row, (held, left) in zip(instrumented(ended), splits):
        openings = row["commit"]["openings"]
        assert left and (openings["n"], openings["verified"], openings["challenge_n"], openings["deferred_replay_n"]) == (len(held), len(held), 4, len(left))
        assert openings["skipped"] == (f"retain=host excluding ['fa2_hidden_m1', 'fa3_hidden_m1']: {len(left)} of 4 challenged leaves lie in excluded "
                                       "families -> opened by replay (openings_after_release)")
        assert openings["deferred_positions"] == [list(p) for p in left]
        assert openings["deferred_digest"] == commit._positions_digest(left) and len(openings["deferred_digest"]) == 64
        assert row["commit"]["openings_after_release"] == doc


@pytest.mark.parametrize(("answer", "argv", "result", "why", "replay"), [
    ({}, (), None, None, None),
    ({"source": "replay", "replay_root_equal": True}, (), True, None, True),
    ({"source": "host+replay", "replay_root_equal": False}, (), False, "replay: " + REPLAY_DIFFERS, False),
    ({"source": "replay", "replay_root_equal": None}, (), False, "replay: " + REPLAY_DIFFERS, False),
    ({"source": "replay", "replay_root_equal": True}, ("--expect-root", "ab" * 64), False, None, True),
], ids=["from-the-host", "replay-equal", "replay-differs", "replay-says-nothing", "replay-equal-after-another-root"])
def test_the_openings_after_the_release_are_recorded_and_a_replay_among_them_is_a_component_of_the_value_correspondence(
        tmp_path, stand_ins, capsys, monkeypatch, answer, argv, result, why, replay):
    """A committer with a staging to release: after each run's value correspondence the openings after the release are made
    once, with the first run's tokens, recorded whole and printed.  Where their source is a replay, the replay's root against
    the committed one is a component: it sets a result no component had set, and never turns a failed one."""
    M.MAKE.update(releases=True)
    doc = after_release(stand_ins, monkeypatch, **answer)
    stand_ins.verdicts = [(True, None)]
    ended = kept(tmp_path, capsys, *argv)
    [com] = made()
    rows = instrumented(ended)
    assert ended.exit is None and com.calls == ["register_weights", *WARMUP, *RUN, *RUN, "close"]
    assert stand_ins.released == [{"com": com, "root": root, "pair": pair, "tokens_ref": M.TOKENS, "deferred": []} for pair, root in enumerate(com.roots[1:])]
    assert [row["commit"]["openings_after_release"] for row in rows] == [doc, doc]
    assert lines(ended, "OPENINGS AFTER RELEASE") == [
        AFTER_RELEASE_LINE.format(pair=pair, source=doc["source"], replay_n=doc["replay_n"], families=doc["replay_families"],
                                  wall=doc["replay_wall_s"], equal=doc["replay_root_equal"]) for pair in (0, 1)]
    method = M.VALUE_CORRESPONDENCE["method"] + (" + root-compare-vs-independent-run (--expect-root)" if argv else "")
    for row in rows:
        correspondence = row["commit"]["value_correspondence"]
        assert (correspondence["result"], correspondence["why"]) == (result, why)
        assert correspondence["method"] == method + (REPLAY_METHOD if replay is not None else "")
        assert correspondence["components"] == ({} if replay is None else {"replay": {
            "result": replay, "reference": REPLAY_REFERENCE, "why": None if replay else REPLAY_DIFFERS}})


def test_a_fault_of_the_retained_bytes_is_made_after_each_run_and_before_its_challenge(tmp_path, stand_ins, capsys, monkeypatch):
    """``VERITY_FAULT=retained_flip``, a negative test's fault: the committer is handed the fault's text once a run, after the
    run's root and before any opening, and the row and a line say what it reports."""
    flip = {"flipped": "head.out", "step": 0, "byte": 3}
    M.MAKE.update(flip=flip)
    monkeypatch.setenv("VERITY_FAULT", "retained_flip:head")
    stand_ins.verdicts = [(True, None)]
    ended = kept(tmp_path, capsys)
    [com] = made()
    assert ended.exit is None and com.faults == ["retained_flip:head"] * 2 and len(com.opened) == 2
    assert [row.get("fault_retained_flip") for row in ended.rows()] == [None, flip, flip, None]
    assert lines(ended, "VERITY_FAULT") == [f"[delta] VERITY_FAULT retained_flip (negative test): {flip}"] * 2


def test_another_fault_asks_the_committer_for_no_flip(tmp_path, stand_ins, capsys, monkeypatch):
    monkeypatch.setenv("VERITY_FAULT", "late_read")
    stand_ins.verdicts = [(True, None)]
    ended = kept(tmp_path, capsys)
    assert ended.exit is None and made()[0].faults == [] and lines(ended, "VERITY_FAULT") == []
    assert ["fault_retained_flip" in row for row in ended.rows()] == [False] * 4


def member(name: str, offset: int, nbytes: int) -> SimpleNamespace:
    return SimpleNamespace(name=name, dev_off=offset, nbytes=nbytes)


@pytest.mark.parametrize(("hidden", "zero_rows"), [(7, 0), (0, 1)], ids=["hidden-row-written", "hidden-row-all-zero"])
def test_the_retained_blocks_tensors_are_digested_on_request_and_a_hidden_row_of_zeros_fails_the_run(tmp_path, stand_ins, capsys, hidden, zero_rows):
    """``--tensor-digests`` with a committer that retains blocks: each run's digests are written by pair and held against the
    pair before, and the count of hidden rows retained as zeros goes into the run's record, where the real verdict reads it:
    a run with such a row fails on it, and a run with none gets as far as the C2 of record, which these Commits have none of."""
    block = torch.cat([torch.arange(32, dtype=torch.uint8), torch.full((32,), hidden, dtype=torch.uint8)])
    M.MAKE.update(blocks={0: (block, 64), 1: (None, 0)}, layouts={0: [member("head.out", 0, 32), member("fa3.m1/L0", 32, 32)]},
                  declare=M.CLASSES)
    ended = kept(tmp_path, capsys, "--tensor-digests")
    rows = instrumented(ended)
    assert [row["commit"]["fa2_zero_rows"] for row in rows] == [zero_rows, zero_rows]
    zeros = ["[delta] FA2 ROWS BOUND AS ZEROS (pair {pair}): 1 e.g. ['0:fa3.m1/L0']"] if zero_rows else []
    assert lines(ended, "TENSOR DIGESTS", "BOUND AS ZEROS") == [line.format(pair=pair) for pair, line in [
        *((0, z) for z in zeros), (1, SAME_DIGESTS_LINE), *((1, z) for z in zeros)]]
    digests = [ended.doc(f"tensor_digests_pair{pair}.json") for pair in (0, 1)]
    assert digests[0] == digests[1] and list(digests[0]) == ["0"] and list(digests[0]["0"]) == ["head.out", "fa3.m1/L0"]
    assert all(len(bytes.fromhex(d)) == 8 for d in digests[0]["0"].values())
    why = "1 FA2 hidden rows bound as all-zero bytes (tap staged after it was zeroed: ordering fault)" if zero_rows else NO_C2
    assert stand_ins.answers == [(False, why)] * 2 and ended.exit == 3
    assert ended.printed[-1] == M.FAIL_LINE.format(failed=2, n=2, pair=0, arm="instrumented", why=why)


def test_no_digest_is_taken_of_a_committer_that_retains_no_block(tmp_path, stand_ins, capsys):
    stand_ins.verdicts = [(True, None)]
    ended = kept(tmp_path, capsys, "--tensor-digests")
    assert ended.exit is None and ["fa2_zero_rows" in row["commit"] for row in instrumented(ended)] == [False, False]
    assert not [name for name in ended.files() if name.startswith("tensor_digests")]


#: What the compare against the Match record answers, as far as `main` reads it: every committed value equal.
COMPARED: Record = {"schema": "verity-sweep/oracle-compare/v1", "method": "oracle-compare", "result": True, "why": None, "grade": None, "complete": True,
                    "compared": 3, "equal": 3, "mismatch_n": 0, "steps_compared": 3, "steps_not_snapshotted": 0,
                    "compared_by_family": {"instance_outputs": 3}, "not_compared": {}, "workload_check": {"ok": True},
                    "attribution": {"complete": True, "pairs": 5, "unverified_n": 0, "ambiguous_steps": []}, "oracle": {"dir": "the Match's"}}
OPENED: Record = {"reads": 3, "leaves": 3, "bytes": 320, "seconds": 0.5, "steps_bound": 3, "not_retained": 0, "failed": 0}
OPENED_LINE = ("[delta] C2 ORACLE COMPARE opened values (pair {pair}): 3 reads, 3 leaves, 320 B verified against the run root in 0.5 s (3 steps bound); "
               "not retained 0, failed 0")
COMPARED_LINE = ("[delta] C2 ORACLE COMPARE (pair {pair}): {head} attribution True (5 step/request pairs, unverified 0, ambiguous []) compared 3 equal "
                 "{equal} mismatch {mismatch} steps 3 not-snapshotted 0 by-family {{'instance_outputs': 3}} not-compared {{}}{why}")
#: The keys of a compare that the value correspondence repeats.
REPEATED = ("result", "why", "grade", "compared", "equal", "mismatch_n", "steps_compared", "steps_not_snapshotted", "complete", "attribution",
            "workload_check", "value_source")
#: `WORKLOAD` as the compare is handed it, to hold the Match record's requests against by name; its requests are token ids
#: and state no prompt length.
COMPARED_WORKLOAD: Record = {"requests": [{"request_id": "a", "index": 0, "prompt_len": None, "max_tokens": 3},
                                          {"request_id": "b", "index": 1, "prompt_len": None, "max_tokens": 2}]}


@pytest.fixture
def oracle(stand_ins, monkeypatch):
    """`stand_ins` with the compare against a Match record stood in whole: the record's reader, the manifest's three readers,
    the producer facts of the Programs of record, their merge, the opened values' reader and the compare itself.  The binding
    record writes the run's binding map, which `main` reads back.  ``handed`` keeps what each was handed, in order;
    ``compared`` is what the compare answers and ``facts`` the producer facts."""
    from verity_vllm.check import oracle_compare
    from verity_vllm.pipeline import commit_plan

    s = stand_ins
    s.handed, s.compared, s.facts = [], copy.deepcopy(COMPARED), ({("head", "out"): {"consumers": []}}, {"members": 1})

    def binding_record(com, rc, reqs, sampling, args, pair, **kw):
        (commit.Path(args.out) / f"binding_map_p{pair}.json").write_text(f'{{"map of pair": {pair}}}')
        return copy.deepcopy(M.BINDING)

    def keep(name: str, answer: Any) -> Any:
        def call(*args: Any, **kw: Any) -> Any:
            s.handed.append((name, args, kw))
            return answer(*args, **kw)

        return call

    class Reader:
        def __init__(self, com: Any, rc: Any) -> None:
            s.handed.append(("OpenedReader", (com, rc.run_root.hex()), {}))

        def record(self) -> Record:
            return dict(OPENED)

    monkeypatch.setattr(commit, "binding_record", binding_record)
    monkeypatch.setattr(commit_plan, "derived", keep("derived", lambda args, manifest, reqs, sampling: s.facts))
    for name, answer in [("MatchOracle", lambda path: ("the oracle of", path)), ("producers_of", lambda manifest: {"of": "the manifest"}),
                         ("merge_producers", lambda of_manifest, facts: {"merged": (of_manifest, facts)}),
                         ("query_families_of", lambda manifest: {"families of": manifest}), ("runtime_rows_of", lambda manifest: {"rows of": manifest}),
                         ("oracle_compare", lambda *args, **kw: copy.deepcopy(s.compared))]:
        monkeypatch.setattr(oracle_compare, name, keep(name, answer))
    monkeypatch.setattr(oracle_compare, "OpenedReader", Reader)
    return s


def handed(s: SimpleNamespace, name: str) -> list[tuple[tuple[Any, ...], Record]]:
    return [(args, kw) for n, args, kw in s.handed if n == name]


@pytest.mark.parametrize(("answer", "argv", "result", "method", "head"), [
    ({}, (), True, "oracle-compare", "OK complete"),
    ({"result": False, "why": "step 1 head.out differs", "equal": 2, "mismatch_n": 1}, (), False, "oracle-compare", "MISMATCH"),
    ({"complete": False, "grade": "partial (2 of 3 steps)"}, (), True, "oracle-compare(partial)", "partial (2 of 3 steps)"),
    ({"result": None, "complete": False}, (), None, "oracle-compare(partial)", "partial"),
    ({}, ("--expect-root", "ab" * 64), False, "oracle-compare", "OK complete"),
], ids=["equal", "mismatch", "partial-with-a-grade", "no-result", "equal-after-another-root"])
def test_a_compare_against_the_match_record_is_recorded_whole_and_is_a_component_of_the_value_correspondence(
        tmp_path, oracle, capsys, answer, argv, result, method, head):
    """``--match-dir`` with a record to compare against: each run's compare is recorded as it was answered, with the producer
    facts' summary and the opened values' record, and repeated in the run's value correspondence, whose result a mismatch
    fails and an equal compare sets only where no component had failed it."""
    s = oracle
    s.compared.update(answer)
    s.verdicts = [(True, None)]
    ended = run(tmp_path, capsys, "--match-dir", str(tmp_path / "match"), *argv)
    [com] = made()
    doc = {**s.compared, "producer_facts": {"members": 1}, "value_source": OPENED}
    why = f" -- {doc['why']}" if doc["why"] else ""
    assert ended.exit is None
    assert lines(ended, "ORACLE COMPARE", "producer facts") == [
        "[delta] form (B) producer facts from the Programs of record: {'members': 1}",
        *(line for pair in (0, 1) for line in [OPENED_LINE.format(pair=pair), COMPARED_LINE.format(
            pair=pair, head=head, equal=doc["equal"], mismatch=doc["mismatch_n"], why=why)])]
    root_compare = " + root-compare-vs-independent-run (--expect-root)" if argv else ""
    for row in instrumented(ended):
        correspondence = row["commit"]["value_correspondence"]
        assert row["commit"]["oracle_compare"] == doc and row["commit"]["opened_values"]["oracle_compare"] == OPENED
        assert correspondence["method"] == f"{method} + {M.VALUE_CORRESPONDENCE['method']}{root_compare}"
        assert correspondence["oracle_compare"] == {k: doc.get(k) for k in REPEATED} and correspondence["oracle_provenance"] == doc["oracle"]
        assert correspondence["components"] == {"oracle_compare": {
            **correspondence["oracle_compare"], "reference": "live Match oracle of this attempt (--match-dir)", "provenance": doc["oracle"]}}
        assert (correspondence["result"], correspondence["why"]) == (result, f"oracle_compare: {doc['why']}" if doc["result"] is False else None)
    merged = {"merged": ({"of": "the manifest"}, s.facts[0])}
    later = ["merge_producers", "OpenedReader", "query_families_of", "runtime_rows_of", "oracle_compare"]
    assert [n for n, _, _ in s.handed] == ["MatchOracle", "producers_of", "derived", *later, "MatchOracle", "producers_of", *later]
    assert handed(s, "MatchOracle") == [((str(tmp_path / "match"),), {})] * 2 and handed(s, "producers_of") == [((None,), {})] * 2
    [((args, manifest, reqs, sampling), _)] = handed(s, "derived")
    assert isinstance(args, commit.Options) and manifest is None and [r["request_id"] for r in reqs] == ["a", "b"] and sampling["variant"] == "greedy"
    assert handed(s, "merge_producers") == [(({"of": "the manifest"}, s.facts[0]), {})] * 2
    assert handed(s, "OpenedReader") == [((com, root), {}) for root in com.roots[1:]]
    compares = handed(s, "oracle_compare")
    assert [(a[0], a[1]) for a, _ in compares] == [(("the oracle of", str(tmp_path / "match")), {"map of pair": pair}) for pair in (0, 1)]
    assert all(kw == {"producers": merged, "workload": COMPARED_WORKLOAD, "aliases": None, "query_families": {"families of": None},
                      "runtime_rows": {"rows of": None}, "own": frozenset()} for _, kw in compares)


def test_a_compare_with_no_producer_facts_is_handed_the_manifests_own_and_the_names_the_committers_sources_own(tmp_path, oracle, capsys):
    """Facts that could not be derived leave the manifest's producers as they are, and are asked for again by no later run;
    the names the committer's sources say are their own are handed on as one set."""
    s = oracle
    s.facts, s.verdicts = (None, {"used": False, "why": "no Program of record"}), [(True, None)]
    M.MAKE.update(sources=[SimpleNamespace(own_names=("head.out", "head.in")), SimpleNamespace(), SimpleNamespace(own_names=("tail.out",))])
    ended = kept(tmp_path, capsys, "--match-dir", str(tmp_path / "match"))
    assert ended.exit is None and len(handed(s, "derived")) == 1 and handed(s, "merge_producers") == []
    assert [(kw["producers"], kw["own"]) for _, kw in handed(s, "oracle_compare")] == [
        ({"of": "the manifest"}, frozenset({"head.out", "head.in", "tail.out"}))] * 2
    assert [row["commit"]["oracle_compare"]["producer_facts"] for row in instrumented(ended)] == [s.facts[1]] * 2


class Session:
    """The Commit's remote session as `main` and `pipeline.remote_session` read it: asked before the engine's build and once
    it is built, begun at the first instrumented run, each of whose engine steps is then a pass of its lock, and ended after
    that run.  ``accepts`` is its answer; ``calls`` keeps what it was asked, in order."""

    def __init__(self, out: Any, accepts: bool = True, error: str | None = None) -> None:
        self.out, self.accepts, self.error = out, accepts, error
        self.calls: list[Any] = []
        self.said: list[str] = []
        self.lockstep: Any = None
        self.record: Record | None = None

    def log(self, line: str) -> None:
        self.said.append(line)

    def before_build(self, engine_kwargs: Record) -> None:
        self.calls.append(("before_build", sorted(engine_kwargs)))

    def check(self, llm: Any, runner: Any) -> None:
        self.calls.append(("check", llm, runner))

    def begin(self) -> Any:
        self.calls.append("begin")
        self.lockstep = SimpleNamespace(audited=0, unaudited=0, step=self.step)
        return self.lockstep

    def step(self, engine_step: Any) -> Any:
        self.calls.append("step")
        self.lockstep.audited += 1
        return engine_step()

    def end(self) -> Record:
        self.calls.append("end")
        self.record = {"verifier": "auditor:7000", "plan": "the plan", "accepted": self.accepts, "error": self.error, "encode": {"s": 1.0},
                       "memory_map": {"agrees": True}, "run": {"passes": self.lockstep.audited}, "not in the verdict": True}
        return self.record


@pytest.mark.parametrize(("accepts", "error", "ended_as"), [(True, None, None), (False, "pass 2 late", 3)], ids=["accepted", "rejected"])
def test_a_remote_session_audits_the_first_instrumented_run_alone_a_pass_an_engine_step(tmp_path, stand_ins, capsys, monkeypatch, accepts, error, ended_as):
    """The session is begun at the first instrumented run, runs each of its engine steps and is ended after it, before the
    run's root is finalized; that run's row carries its entry, no other run is audited, the verdict carries the session's
    record, and a session that rejected ends the Commit with exit 3 after the verdict is written."""
    from verity_vllm.pipeline import remote_session

    session = Session(tmp_path / "out", accepts, error)
    monkeypatch.setattr(remote_session, "session_of", lambda args: session)
    stand_ins.verdicts = [(True, None)]
    ended = run(tmp_path, capsys)
    [com] = made()
    eng = stand_ins.engine
    assert ended.exit == ended_as and ended.printed[-1] == M.PASS_LINE.format(n=2)
    assert session.calls == [("before_build", sorted(session.calls[0][1])), ("check", eng.llm, eng.runner), "begin", "step", "step", "step", "end"]
    entry = {"audited_steps": 3, "unaudited_steps": 0, "accepted": accepts}
    assert [row.get("remote_session") for row in ended.rows()] == [None, entry, None, None]
    word = "ACCEPTED" if accepts else "REJECTED"
    assert session.said == [
        f"[delta] remote session over pair 0's instrumented run: 3 of its 3 engine steps a pass each, {word}" + (f" ({error})" if error else ""),
        f"[delta] REMOTE SESSION {word} (the plan at auditor:7000; map agrees True): {tmp_path / 'out' / 'remote_session.json'}"]
    assert ended.doc("verdict.json") == M.verdict_doc(com.roots[1:], remote_session={
        "verifier": "auditor:7000", "plan": "the plan", "accepted": accepts, "error": error, "encode": {"s": 1.0}, "memory_map": {"agrees": True},
        "run": {"passes": 3}})


PENDING = "PENDING: the C2 sampled replay + boundary linkage are deferred to the replay bundle"


@pytest.fixture
def bundles(stand_ins, monkeypatch):
    """`stand_ins` with the replay bundle's writing and the C2 site's two hooks stood in: the store's early write, which starts
    nothing; the deferral, which puts the bundle's records on the run's row and its context under the run's pair, or answers
    that the replay runs in this process (``defers`` False); the bundles' sealing; the in-process replay and the serving
    rows.  ``handed`` keeps each call's name and what it was handed."""
    from verity_vllm.pipeline import c2_replay, replay_bundle

    s = stand_ins
    s.handed, s.defers = [], True

    def start(out, pair, com, rc, args, protocols):
        s.handed.append(("start", pair))

    def defer(out, pair, com, rc, args, *, row, deferred, started, plan, **kw):
        s.handed.append(("defer", pair, {"out": out, "com": com, "root": rc.run_root.hex(), "started": started, "plan": plan, **kw}))
        if s.defers:
            row["commit"]["sampled_replay"], row["commit"]["boundary_linkage"] = replay_bundle.deferred_records(out / f"bundle_{pair}")
            deferred[pair] = {"pair": pair}
        return s.defers

    def seal_all(out, deferred, rows, verdict, pending, *, tl):
        assert tl is commit.TL and not (out / "verdict.json").exists()
        s.handed.append(("seal_all", copy.deepcopy(deferred), [row["pair"] for row in rows], pending))
        if deferred:
            verdict["replay_deferred"] = {"pending_pairs": pending, "bundles": [{"pair": pair, "path": f"bundle_{pair}"} for pair in deferred]}

    for owner, name, value in [(replay_bundle, "start", start), (replay_bundle, "defer", defer), (replay_bundle, "seal_all", seal_all),
                               (c2_replay, "run", lambda sources, cache: s.handed.append(("run", sources.pair, cache))),
                               (c2_replay, "serving_rows", lambda sources: s.handed.append(("serving_rows", sources.pair, sources.com)))]:
        monkeypatch.setattr(owner, name, value)
    return s


def names(s: SimpleNamespace) -> list[tuple[str, int]]:
    return [(name, pair) for name, pair, *_ in s.handed if name != "seal_all"]


@pytest.mark.parametrize("serving", [False, True], ids=["bundle-alone", "with-serving-rows"])
def test_a_deferred_replay_leaves_each_run_pending_and_the_commit_names_the_bundles_it_runs_from(tmp_path, bundles, capsys, serving):
    """``--replay-deferred``: at each run's C2 site the replay is handed to its bundle and no replay runs in this process; the
    serving rows are committed there only where ``--serving-rows`` asks.  The run's verdict is then the deferred one's, which
    passes no run and fails none that passes its own checks, and the Commit ends pending, naming the bundles."""
    s = bundles
    M.MAKE.update(declare=M.CLASSES)
    ended = kept(tmp_path, capsys, "--replay-deferred", "--program-dir", str(tmp_path / "program"),
                 *(("--serving-rows", str(tmp_path / "request.json")) if serving else ()))
    [com] = made()
    rows = instrumented(ended)
    assert ended.exit is None
    assert names(s) == [(name, pair) for pair in (0, 1) for name in ("start", "defer", *(["serving_rows"] if serving else []))]
    assert [h[2] for h in s.handed if h[0] == "serving_rows"] == ([com, com] if serving else [])
    for (_, pair, kw), root in zip([h for h in s.handed if h[0] == "defer"], com.roots[1:]):
        said = M.declaration("stand_in", M.CLASSES, run=M.run_of(tmp_path))
        assert kw == {"out": ended.out, "com": com, "root": root, "started": None, "plan": None, "tl": commit.TL, "arm": "instrumented",
                      "versions": kw["versions"], "reqs": kw["reqs"], "sampling": kw["sampling"], "tokens": M.TOKENS, "stop_of_record": M.STOP_RULE,
                      "ids_full": None, "declaration": said, "required_classes": M.CLASSES, "vc_result": None}
        assert kw["versions"]["stop_rule_of_record"] == M.STOP and [r["request_id"] for r in kw["reqs"]] == ["a", "b"]
    assert [(row["commit"]["commit_pass"], row["commit"]["commit_fail_reason"]) for row in rows] == [(None, PENDING)] * 2
    assert [row["commit"]["sampled_replay"]["grade"] for row in rows] == ["deferred", "deferred"]
    assert [name for name, _ in s.calls if name == "commit_verdict"] == []
    assert s.handed[-1] == ("seal_all", {0: {"pair": 0}, 1: {"pair": 1}}, [0, 0, 1, 1], [0, 1])
    assert lines(ended, "COMMIT P", "COMMIT F") == [
        f"[delta] COMMIT PENDING (pair 0, instrumented): {PENDING}", f"[delta] COMMIT PENDING (pair 1, instrumented): {PENDING}",
        "[delta] COMMIT PENDING: 2/2 instrumented runs pass every check of this process; their C2 replay runs from ['bundle_0', 'bundle_1']"]
    verdict = ended.doc("verdict.json")
    assert (verdict["commit_pass"], verdict["runs_failed"]) == (None, 0)
    assert verdict["replay_deferred"] == {"pending_pairs": [0, 1], "bundles": [{"pair": 0, "path": "bundle_0"}, {"pair": 1, "path": "bundle_1"}]}


def test_a_replay_its_bundle_does_not_take_runs_in_this_process_and_the_run_gets_the_whole_verdict(tmp_path, bundles, capsys):
    """A deferral that answers no (a store over the bundle's limit) leaves the run to the in-process replay, with the Commit's
    one replay cache, and to the whole verdict; nothing is committed in the serving rows' format for it."""
    s = bundles
    s.defers, s.verdicts = False, [(True, None)]
    ended = kept(tmp_path, capsys, "--replay-deferred", "--program-dir", str(tmp_path / "program"), "--serving-rows", str(tmp_path / "request.json"))
    assert ended.exit is None and ended.printed[-1] == M.PASS_LINE.format(n=2)
    assert names(s) == [(name, pair) for pair in (0, 1) for name in ("start", "defer", "run")]
    caches = [h[2] for h in s.handed if h[0] == "run"]
    assert caches[0] is caches[1] and caches[0] == {}
    assert [name for name, _ in s.calls if name == "commit_verdict"] == ["commit_verdict"] * 2
    assert s.handed[-1] == ("seal_all", {}, [0, 0, 1, 1], []) and "replay_deferred" not in ended.doc("verdict.json")


@pytest.mark.parametrize("digest", [W.PROGRAM, "c" * 128], ids=["the-records-program", "another-program"])
def test_a_program_digest_of_record_is_the_one_each_pairs_weights_pin_is_held_to(tmp_path, pinned, capsys, digest):
    """``--program-digest`` with the weights of record: each pair's pin is handed the digest as the set of Programs the record
    may be of, and the real check then refuses a record of another Program by name."""
    pinned.verdicts = [(True, None)]
    file = W.record_at(tmp_path, W.record_of(M.Model()))
    ended = run(tmp_path, capsys, "--weights-of-record", str(file), "--program-digest", digest)
    assert ended.exit is None
    assert [kw["program_digests_of_record"] for name, _, kw in pinned.handed if name == "check"] == [{digest}, {digest}]
    pin = W.pin_of(pinned, str(file))
    same = digest == W.PROGRAM
    why = None if same else f"record built for Program {W.PROGRAM[:16]} != Program of record ['{digest[:16]}']"
    assert W.pins(ended) == [pin, pin] and (pin["result"], pin["program_of_record"], pin["why"]) == (same, same, why)
    assert W.pin_lines(ended) == [W.pin_line(pair) if same else W.pin_line(pair, "FAIL [mismatch]", why) for pair in (0, 1)]


#: A committer's layouts as it dumps them: a step's tensors in layout order.
LAYOUTS: Record = {str(step): [{"name": "head.out", "numel": n // 4}] for step, n in enumerate(M.STEP_BYTES)}
#: What the manifest's coverage check answers: every identity it was handed is covered.
COVERED: Record = {"ok": True, "checked": 5, "missing_n": 0, "missing_by_family": {}, "families_missing_entirely": [], "committed_unmatched_n": 0}
COVERAGE_LINE = ("[delta] required_value_coverage: MANIFEST COVERAGE (pair {pair}, instrumented): {ok} checked {checked} missing {missing} {{}} "
                 "families-missing-entirely [] unmatched-committed {unmatched}")
DECLARED_BASIS = "manifest engine_step annotation (declared arrive_steps)"
OBSERVED_BASIS = "engine schedule of record (request step + observed arrival)"


def identity(request: int, step: int, **more: Any) -> Record:
    """A manifest identity of request ``request``'s step ``step``, annotated with the engine step its declared arrival puts it at."""
    return {"request_id": f"r{request}", "step": step, "engine_step": step, "family": "instance_outputs", "op_path": "head",
            "output_member": "out", **more}


def composition(step: int, *requests: int) -> Record:
    """A manifest's declared composition of engine step ``step``: a row a request."""
    return {"engine_step": step, "members": [{"request_id": f"r{r}", "rows": 1} for r in requests], "rows": len(requests), "n_seq": len(requests)}


#: The identities of `WORKLOAD` run to each request's ``max_tokens``: three steps of request a and two of request b.
IDENTITIES = [identity(0, 0), identity(1, 0), identity(0, 1), identity(1, 1), identity(0, 2)]
COMPOSITIONS = [composition(0, 0, 1), composition(1, 0, 1), composition(2, 0)]
#: The executed prefix of record of `WORKLOAD` served to each request's ``max_tokens`` under `STOP_RULE`, as the coverage keeps it.
PREFIX: Record = {"r0": {"served_output_len": 3, "executed_prefix_len": 3, "cap": 3, "cap_exhausted": True, "eos_stop": False},
                  "r1": {"served_output_len": 2, "executed_prefix_len": 2, "cap": 2, "cap_exhausted": True, "eos_stop": False},
                  "lag": 1, "ok": True, "problems": []}


@pytest.fixture
def manifest(stand_ins, monkeypatch):
    """`stand_ins` for a Commit under a required-value manifest: the Programs' two readers before the engine answer that
    there is no plan and no fact, the attachment of record takes the manifest, and the manifest's coverage check
    (`query.manifest.bind.coverage_check`) answers ``coverage``, or raises ``raises``; ``checked`` keeps what it was handed."""
    from verity_vllm.pipeline import boundary_plan, commit_plan
    from verity_vllm.query.manifest import bind

    s = stand_ins
    s.coverage, s.raises, s.checked = copy.deepcopy(COVERED), None, []

    def attach_of_record(com, eng, args, manifest, out, **kw):
        s.calls.append(("attach_of_record", {"com": com, "manifest": manifest}))
        return SimpleNamespace(required_classes=kw["required_classes"], moe_attached=False, hidden_required=None)

    def coverage_check(manifest, layouts, **kw):
        s.checked.append((copy.deepcopy(manifest), layouts, kw))
        if s.raises:
            raise s.raises
        return copy.deepcopy(s.coverage)

    for owner, name, value in [(boundary_plan, "use", lambda args, manifest, *, tl: (None, None)),
                               (commit_plan, "producer_facts", lambda args, manifest, reqs, sampling, *, tl: ((None, {"members": 0}), None)),
                               (commit, "attach_of_record", attach_of_record), (bind, "coverage_check", coverage_check)]:
        monkeypatch.setattr(owner, name, value)
    return s


def under(tmp_path: Any, capsys: Any, doc: Record, *argv: str) -> Ended:
    """`kept` under the manifest ``doc`` and its Program's digest, with a committer that dumps `LAYOUTS`."""
    M.MAKE.setdefault("dumps", LAYOUTS)
    return kept(tmp_path, capsys, "--required-manifest", str(M.manifest_at(tmp_path, doc)), "--program-digest", doc["program_digest"], *argv)


def coverages(ended: Ended) -> list[Record]:
    """Each instrumented run's coverage, after its row is seen to keep it in both places the verdict reads it from."""
    rows = instrumented(ended)
    assert all(row["commit"]["coverage"] == row["commit"]["extra"]["coverage"] for row in rows)
    return [row["commit"]["coverage"] for row in rows]


def dumped(ended: Ended, pair: int) -> Any:
    import gzip

    with gzip.open(ended.out / f"layouts_pair{pair}_instrumented.json.gz", "rt") as f:
        return json.load(f)


def test_a_manifest_that_lists_no_identity_is_checked_as_it_is_against_the_committers_layouts(tmp_path, manifest, capsys):
    """Each run's layouts are dumped once, checked against the manifest of record itself and written beside the run; the
    coverage is the check's answer, kept in the run's record and in its committer's own."""
    s = manifest
    s.verdicts = [(True, None)]
    ended = under(tmp_path, capsys, M.MANIFEST)
    [com] = made()
    assert ended.exit is None and com.calls == ["register_weights", *WARMUP, *[c for _ in (0, 1) for c in (*RUN, "layouts_dump")], "close"]
    assert s.checked == [(M.MANIFEST, LAYOUTS, {"request_id": None, "rank": 0, "ports": {}})] * 2
    assert coverages(ended) == [COVERED, COVERED] and [dumped(ended, pair) for pair in (0, 1)] == [LAYOUTS, LAYOUTS]
    assert lines(ended, "MANIFEST COVERAGE", "executed prefix") == [
        COVERAGE_LINE.format(pair=pair, ok="OK", checked=5, missing=0, unmatched=0) for pair in (0, 1)]
    assert ended.doc("verdict.json")["required_manifest"] == {
        "path": str(tmp_path / "manifest.json"), "manifest_digest": "e" * 64, "program_digest": "d" * 128, "populations": M.MANIFEST["populations"],
        "query": None, "coverage": [COVERED, COVERED]}


def test_a_coverage_check_that_raises_is_recorded_as_a_coverage_that_failed_with_its_error(tmp_path, manifest, capsys):
    """The run keeps its record: the coverage is then not ok, with the error and no count, and the layouts are still written."""
    s = manifest
    s.raises, s.verdicts = TypeError("a row of every step"), [(True, None)]
    ended = under(tmp_path, capsys, M.MANIFEST)
    failed = {"ok": False, "error": "TypeError('a row of every step')", "checked": 0, "missing_n": -1, "missing_by_family": {},
              "families_missing_entirely": [], "committed_unmatched_n": -1, "manifest_source": str(tmp_path / "manifest.json")}
    assert ended.exit is None and coverages(ended) == [failed, failed] and dumped(ended, 1) == LAYOUTS
    assert lines(ended, "MANIFEST COVERAGE") == [line for pair in (0, 1) for line in [
        f"[delta] MANIFEST COVERAGE ERROR (pair {pair}, instrumented): TypeError('a row of every step')",
        COVERAGE_LINE.format(pair=pair, ok="FAIL", checked=0, missing=-1, unmatched=-1)]]


def test_a_committer_that_dumps_no_layouts_is_given_no_manifest_coverage(tmp_path, manifest, capsys):
    s = manifest
    s.verdicts = [(True, None)]
    M.MAKE.update(dumps=None)
    ended = under(tmp_path, capsys, M.MANIFEST)
    assert ended.exit is None and s.checked == [] and ["coverage" in row["commit"] for row in instrumented(ended)] == [False, False]
    assert not [name for name in ended.files() if name.startswith("layouts_")]


def placed(doc: Record, identities: list[Record], compositions: list[Any], *, basis: str = OBSERVED_BASIS, schedule: Record,
           not_executed: Record | None = None, prefix: Record = PREFIX) -> tuple[Record, Record]:
    """What the coverage check is handed of ``doc`` once its requirement is placed, and the coverage `main` makes of `COVERED`."""
    classified = {"rule": not_executed_rule(), "n": 0, "by_family": {}, "by_request": {}, "kind_by_family": {}, "scope": not_executed_scope(),
                  **(not_executed or {})}
    return ({**doc, "identities": identities, "engine_steps": compositions},
            {**COVERED, "schedule": schedule, "engine_step_basis": basis, "executed_prefix_of_record": prefix, "not_executed_by_rule": classified,
             "manifest_of_record": {"digest": doc["manifest_digest"], "identities_n": len(doc["identities"]), "required_n": len(identities),
                                    "not_executed_n": classified["n"], "edited": False}})


def not_executed_rule() -> str:
    from verity_vllm.commit import executed_prefix

    return executed_prefix.RULE


def not_executed_scope() -> Any:
    from verity_vllm.commit import executed_prefix

    return executed_prefix.ACTIVE_REFERENCE_ONLY


def declared_schedule(**more: Any) -> Record:
    """The schedule of a committer that recorded no step's rows: the declared arrivals stand."""
    return {"use": "declared", "basis": SCHEDULE_BASIS, "why": "no per-step request rows (single request or none recorded): declared arrive_steps",
            "all_equal_declared": False, "declared_arrive_steps": {"r0": 0, "r1": 0}, "observed_arrive_steps": {}, **more}


SCHEDULE_BASIS = ("engine schedule of record: per-committed-step request rows (the scheduler's row order as the collector recorded it); "
                  "arrival = the first committed step the request ran in")
PLACED_LINE = ("[delta] manifest requirement placed over the executed prefix of record (pair {pair}): identities not-executed {dropped} (by family "
               "{families}; manifest of record untouched: {n} identities, digest eeeeeeeeeeeeeeee), composition members dropped {members} over "
               "{changed} steps, steps not executed {steps}")


def test_a_manifests_identities_are_placed_over_the_executed_prefix_of_record_before_they_are_checked(tmp_path, manifest, capsys):
    """A manifest that lists its identities, all of them executed: the check is handed a copy of the manifest with the same
    identities and compositions, and the coverage says how they were placed, under which schedule, and that the manifest
    of record was not edited.  Nothing was left out, so no line says so."""
    s = manifest
    s.verdicts = [(True, None)]
    doc = {**M.MANIFEST, "identities": IDENTITIES, "engine_steps": COMPOSITIONS}
    ended = under(tmp_path, capsys, doc)
    handed_on, coverage = placed(doc, IDENTITIES, COMPOSITIONS, schedule=declared_schedule())
    assert ended.exit is None and [m for m, _, _ in s.checked] == [handed_on, handed_on] and coverages(ended) == [coverage, coverage]
    assert lines(ended, "MANIFEST COVERAGE", "executed prefix") == [
        COVERAGE_LINE.format(pair=pair, ok="OK", checked=5, missing=0, unmatched=0) for pair in (0, 1)]


def test_identities_and_composition_members_past_a_requests_executed_prefix_are_left_out_by_name_and_counted(tmp_path, manifest, capsys):
    """A manifest built for longer runs than the served ones: an identity at a step its request did not execute is left out
    of the requirement and counted by family and request, a composition's member likewise, a step no request executes is
    dropped whole with its members, and a line says how many of each.  The manifest of record keeps its count and its digest."""
    s = manifest
    s.verdicts = [(True, None)]
    late = [identity(1, 2), identity(0, 3, family="sampled_token_ids")]
    doc = {**M.MANIFEST, "identities": [*IDENTITIES, *late], "engine_steps": [composition(0, 0, 1), composition(1, 0, 1), composition(2, 0, 1),
                                                                               composition(3, 0)]}
    ended = under(tmp_path, capsys, doc)
    classified = {"n": 2, "by_family": {"instance_outputs": 1, "sampled_token_ids": 1}, "by_request": {"r1": 1, "r0": 1},
                  "kind_by_family": {"instance_outputs": "row", "sampled_token_ids": "row"}}
    handed_on, coverage = placed(doc, IDENTITIES, COMPOSITIONS, schedule=declared_schedule(), not_executed=classified)
    assert ended.exit is None and [m for m, _, _ in s.checked] == [handed_on, handed_on] and coverages(ended) == [coverage, coverage]
    assert lines(ended, "executed prefix") == [PLACED_LINE.format(pair=pair, dropped=2, families=classified["by_family"], n=7, members=2, changed=1,
                                                                  steps=1) for pair in (0, 1)]


def test_a_manifest_whose_engine_steps_are_numbers_is_restricted_to_the_steps_a_request_executes(tmp_path, manifest, capsys):
    """A manifest of one request lists its engine steps as plain numbers, with no composition: the steps no request executes
    at are dropped, and with no identity left out no line is printed."""
    s = manifest
    s.verdicts = [(True, None)]
    doc = {**M.MANIFEST, "identities": IDENTITIES, "engine_steps": [0, 1, 2, 3, 4]}
    ended = under(tmp_path, capsys, doc)
    handed_on, coverage = placed(doc, IDENTITIES, [0, 1, 2], schedule=declared_schedule())
    assert ended.exit is None and [m for m, _, _ in s.checked] == [handed_on, handed_on] and coverages(ended) == [coverage, coverage]
    assert lines(ended, "executed prefix") == []


def test_a_request_the_committer_saw_arrive_later_than_declared_has_its_identities_placed_by_the_observed_arrival(tmp_path, manifest, capsys):
    """The committer's rows by step say request b first ran at step 1: the schedule of record is the observed one, each
    identity is placed at its request's step past the observed arrival, b's one engine step later, with the declared engine
    step kept beside it, and the compositions are restricted to the steps of that schedule."""
    s = manifest
    s.verdicts = [(True, None)]
    M.MAKE.update(step_rows={0: [("a", 0, 1)], 1: [("a", 0, 1), ("b", 1, 2)], 2: [("a", 0, 1), ("b", 1, 2)]})
    doc = {**M.MANIFEST, "identities": IDENTITIES, "engine_steps": COMPOSITIONS}
    ended = under(tmp_path, capsys, doc)
    moved = [identity(0, 0, engine_step_declared=0), identity(1, 0, engine_step=1, engine_step_declared=0), identity(0, 1, engine_step_declared=1),
             identity(1, 1, engine_step=2, engine_step_declared=1), identity(0, 2, engine_step_declared=2)]
    schedule = {"use": "observed", "basis": SCHEDULE_BASIS, "why": None, "all_equal_declared": False, "declared_arrive_steps": {"r0": 0, "r1": 0},
                "observed_arrive_steps": {"r0": 0, "r1": 1}}
    handed_on, coverage = placed(doc, moved, [composition(0, 0), composition(1, 0, 1), composition(2, 0)], schedule=schedule)
    assert ended.exit is None and [m for m, _, _ in s.checked] == [handed_on, handed_on] and coverages(ended) == [coverage, coverage]
    assert lines(ended, "executed prefix") == [PLACED_LINE.format(pair=pair, dropped=0, families={}, n=5, members=1, changed=1, steps=0)
                                               for pair in (0, 1)]


def test_an_executed_prefix_that_cannot_be_placed_fails_a_coverage_the_check_passed(tmp_path, manifest, capsys, monkeypatch):
    """A request served short of its limit under a Program that declares no stop rule: its executed prefix is not of
    record, so the coverage is failed in place with the prefix's own sentence, whatever the check answered."""
    s = manifest
    s.verdicts = [(True, None)]
    monkeypatch.setattr(M.va, "sampling_params", lambda req, sampling: SimpleNamespace(max_tokens=int(req["max_tokens"]) - (req["request_id"] == "a")))
    doc = {**M.MANIFEST, "identities": IDENTITIES, "engine_steps": COMPOSITIONS}
    ended = under(tmp_path, capsys, doc)
    for coverage in coverages(ended):
        [why] = coverage["executed_prefix_of_record"]["problems"]
        assert why.startswith("r0: served 2 of its ") and why.endswith("the Program declares no stop rule (eos_token_ids): the early stop cannot be verified, fail closed")
        assert (coverage["ok"], coverage["error"]) == (False, "executed prefix of record: " + why)
        assert coverage["executed_prefix_of_record"] == {
            **PREFIX, "r0": {**PREFIX["r0"], "served_output_len": 2, "cap_exhausted": False, "eos_stop": True}, "ok": False, "problems": [why]}
    assert ended.exit is None and lines(ended, "MANIFEST COVERAGE") == [
        COVERAGE_LINE.format(pair=pair, ok="FAIL", checked=5, missing=0, unmatched=0) for pair in (0, 1)]


def test_under_a_manifest_the_weights_pin_is_held_to_every_program_digest_the_manifest_names(tmp_path, manifest, pinned, capsys):
    """The Programs a record may be of are then the Build's digest of record, the manifest's component digests and each of
    its components' own; a component that is no record, or names no digest, adds none."""
    pinned.verdicts = [(True, None)]
    doc = {**M.MANIFEST, "component_digests": ["1" * 128, W.PROGRAM], "components": {
        "decoder": {"program_digest": "2" * 128}, "sampler": {"program_digest": None}, "embedder": "not a record"}}
    file = W.record_at(tmp_path, W.record_of(M.Model()))
    ended = under(tmp_path, capsys, doc, "--weights-of-record", str(file))
    assert ended.exit is None
    assert [kw["program_digests_of_record"] for name, _, kw in pinned.handed if name == "check"] == [{"d" * 128, "1" * 128, W.PROGRAM, "2" * 128}] * 2
    assert [row["commit"]["weights_pin"]["result"] for row in instrumented(ended)] == [True, True]
    assert W.pin_lines(ended) == [W.pin_line(0), W.pin_line(1)]


SERVED_METHOD = "sampled-exact-replay (the served Build's Program on the committed values; served_check.py) + boundary linkage"
#: What the served source's check of a run answers, as far as `main` reads it.
SERVED_C2: Record = {"ok": True, "line": "3/3 values exact", "components": {"sampled_replay": {"result": True}, "boundary_linkage": {"result": True}},
                     "value_correspondence": {"compiled_check": {"ok": True, "equal": 3, "checked": 3, "steps_unrun": [], "error": None}}}
SOURCE_SUMMARY: Record = {"slots": 1, "plans": 2, "identities": 3}


class Through(W.Served):
    """`test_commit_weights_of_record`'s served source, which goes on past the committers' setup: its rebuild answers the
    Build of record's digest, each run is begun on its source, and its check of a run answers ``answers``' next, the last
    for every later run.  ``calls`` keeps what it was asked after the setup, in order."""

    made: list[Through] = []
    answers: list[Record] = []

    def __init__(self, build: Record, fault: str | None) -> None:
        super().__init__(build, fault)
        self.calls: list[Any] = []
        self.src.begin_run = lambda: self.calls.append("begin_run")
        self.src.summary = lambda: dict(SOURCE_SUMMARY)
        Through.made.append(self)

    def rebuild(self, workload: Record, out: str) -> Record:
        self.calls.append(("rebuild", workload, out))
        return {"program_digest": self.build["program_digest"], "complete": True}

    def c2(self, com: Any, rc: Any, reqs: list[Record], challenge: str, seed: Any) -> Record:
        n = sum(1 for c in self.calls if c[0] == "c2")
        self.calls.append(("c2", com, rc.run_root.hex(), [r["request_id"] for r in reqs], challenge, seed))
        return copy.deepcopy(Through.answers[min(n, len(Through.answers) - 1)])


@pytest.fixture
def serving(manifest, monkeypatch):
    """`manifest` on the served row, its source `Through`."""
    from verity_vllm.pipeline import served_match

    Through.made, Through.answers = [], [SERVED_C2]
    W.ORDER.clear()
    monkeypatch.setattr(served_match, "ServedCommit", Through)
    return manifest


def served(tmp_path: Any, capsys: Any, *argv: str) -> Ended:
    return kept(tmp_path, capsys, "--compiled-taps", "served", "--required-manifest", str(M.manifest_at(tmp_path)), "--program-digest",
                M.MANIFEST["program_digest"], *argv)


@pytest.mark.parametrize(("argv", "result", "method"), [
    ((), True, SERVED_METHOD + " + " + M.VALUE_CORRESPONDENCE["method"]),
    (("--expect-root", "ab" * 64), False, f"{SERVED_METHOD} + {M.VALUE_CORRESPONDENCE['method']} + root-compare-vs-independent-run (--expect-root)"),
], ids=["alone", "after-another-root"])
def test_the_served_source_begins_each_run_and_its_check_of_the_run_is_the_value_correspondences_first_method(
        tmp_path, serving, capsys, argv, result, method):
    """``--compiled-taps served``: the source is told of each instrumented run before the committer is installed and, after
    the run, checks it under the run's own replay coin; the row keeps the source's summary and the compile settings, and the
    value correspondence takes the check's components, its record and its result, which never turns a failed one."""
    s = serving
    s.verdicts = [(True, None)]
    ended = served(tmp_path, capsys, *argv)
    [com], [source] = made(), Through.made
    rows = instrumented(ended)
    coins = [row["commit"]["extra"]["coins"]["keys"][M.CC.REPLAY] for row in rows]
    assert ended.exit is None and source.build == M.MANIFEST
    assert source.calls == [("rebuild", M.WORKLOAD, str(ended.out)),
                            *(c for root, coin in zip(com.roots[1:], coins) for c in ["begin_run", ("c2", com, root, ["a", "b"], coin, None)])]
    assert lines(ended, "served Build", "SERVED C2") == [
        "[delta] served Build of this process: program dddddddddddddddd complete True, equal to the Build of record True",
        "[delta] SERVED C2 (pair 0): 3/3 values exact", "[delta] SERVED C2 (pair 1): 3/3 values exact"]
    for row in rows:
        assert row["commit"]["compiled_source"] == SOURCE_SUMMARY and row["commit"]["compilation"] == ended.doc("versions.json")["compilation"]
        assert row["commit"]["value_correspondence"] == {
            **M.VALUE_CORRESPONDENCE, "method": method, "result": result, **SERVED_C2["value_correspondence"],
            "components": {**SERVED_C2["components"], "replay_challenge": {"source": M.CC.IN_PROCESS, "family": M.CC.REPLAY, "index": None}}}
        assert (row["commit"]["commit_pass"], row["commit"]["commit_fail_reason"]) == (True, None)


def test_a_replay_challenge_given_on_the_command_line_is_the_one_the_served_check_is_handed(tmp_path, serving, capsys):
    """``--replay-challenge``, a measurement's: where the run's coins are this process's own the given challenge takes the
    coin's place, and the component says it came from the flag."""
    serving.verdicts = [(True, None)]
    ended = served(tmp_path, capsys, "--replay-challenge", "5eed")
    assert ended.exit is None and [c[4] for c in Through.made[0].calls if c[0] == "c2"] == ["5eed", "5eed"]
    assert [row["commit"]["value_correspondence"]["components"]["replay_challenge"] for row in instrumented(ended)] == [{"source": "flag"}] * 2


@pytest.mark.parametrize(("verdict", "why"), [
    ((True, None), "served C2 FAIL: 2/3 values exact, steps unrun [1] step 1 not replayed"),
    ((False, "openings verified 3 != checked 4"), "openings verified 3 != checked 4"),
], ids=["a-run-the-verdict-passed", "a-run-the-verdict-failed"])
def test_a_served_check_that_is_not_ok_fails_a_run_the_verdict_passed_and_leaves_a_failed_runs_reason(tmp_path, serving, capsys, verdict, why):
    """The served check's own record is read after the verdict: a run that passed the verdict fails on the check's counts,
    and a run the verdict failed keeps the verdict's reason."""
    failed = {**SERVED_C2, "ok": False, "line": "2/3 values exact", "value_correspondence": {"compiled_check": {
        "ok": False, "equal": 2, "checked": 3, "steps_unrun": [1], "error": "step 1 not replayed"}}}
    Through.answers = [failed, SERVED_C2]
    serving.verdicts = [verdict, (True, None)]
    ended = served(tmp_path, capsys)
    rows = instrumented(ended)
    assert [(row["commit"]["commit_pass"], row["commit"]["commit_fail_reason"]) for row in rows] == [(False, why), (True, None)]
    assert [row["commit"]["value_correspondence"]["result"] for row in rows] == [False, True]
    assert lines(ended, "SERVED C2", "COMMIT F") == [
        "[delta] SERVED C2 (pair 0): 2/3 values exact", f"[delta] COMMIT FAIL (pair 0, instrumented): {why}", "[delta] SERVED C2 (pair 1): 3/3 values exact",
        M.FAIL_LINE.format(failed=1, n=2, pair=0, arm="instrumented", why=why)]
    assert ended.exit == 3
