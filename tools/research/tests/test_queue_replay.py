"""`research.queue_replay`: the verified queue replayed under other rules, on a few recorded hours of it.

The fixture is a collected document cut small: `python3 -m research.queue_replay --doc DOC --hours 5 --cut FILE`."""
import collections
import json
from pathlib import Path

import pytest
from research import queue_replay as qr
from research import queue_table as qt

FIXTURE = Path(__file__).parent / "fixtures" / "queue_replay.json"
DOC = json.loads(FIXTURE.read_text())
CHAIN = [r for r in qr.RULES if not r.side and not r.first]


@pytest.fixture(scope="module")
def tr() -> qr.Trace:
    return qr.trace(DOC)


def rule(start: str) -> qr.Rule:
    return next(r for r in qr.RULES if r.name.startswith(start))


def walk(tr: qr.Trace, r: qr.Rule, seed: int = 0):
    """The replay after each tick."""
    run = qr.Replay(tr, r, seed)
    for t in tr.ticks:
        run.step(t)
        yield run


def test_the_trace_is_what_the_recorded_hours_hold(tr):
    assert (len(tr.ticks), len(tr.changes)) == (107, 116)
    assert sorted((f.how, len(f.changes), f.suites) for f in tr.faults.values()) == [
        ("found by a hold or a mark another withdrew", 2, ()),
        ("named by the loop", 1, (".",)), ("named by the loop", 1, ("adapters/vllm",)), ("named by the loop", 1, ("tools/research",))]
    assert (len(tr.passes), len(tr.fails), tr.records, tr.flakes, tr.hands) == (3, 6, [4598.0], [], [])
    assert tr.flags[-1][1] == {"batch": 32, "depth": 6, "lean": 0}
    assert tr.start < tr.records_from < tr.blame_from < tr.end
    m = qr.measured(tr)
    assert (m["landed"], m["n_lean"], m["built"], m["reds"], m["moves"]) == (50, 18, 13, 3, 3)
    assert m["by_cause"] == {"red": 82, "withdrawn mark": 33, "dropped base": 13}
    assert m["within"] == 0.0 and round(m["p50"]) == 377


def test_a_change_is_plain_when_it_has_no_lean_input_and_nothing_on_the_soundness_path(tr):
    kinds = collections.Counter((ch.lean, ch.audit, ch.sound, ch.plain) for ch in tr.changes.values())
    assert sum(n for (lean, audit, _, _), n in kinds.items() if lean and not audit) == 0   # a Lean change is a Lean input
    assert all(plain == (not audit and not sound) for _, audit, sound, plain in kinds)
    assert sum(ch.lean for ch in tr.changes.values()) == 24 and sum(ch.plain for ch in tr.changes.values()) == 81


def test_a_rule_replays_the_same_way_each_time(tr):
    for r in (rule("1."), rule("2."), rule("3."), rule("5.")):
        assert qr.row(tr, r, 2) == qr.row(tr, r, 2)


@pytest.mark.parametrize("r", qr.RULES, ids=lambda r: r.name.split(" ")[0])
def test_no_change_is_in_two_candidates_or_in_one_once_on_main(tr, r):
    for run in walk(tr, r):
        held = collections.Counter(ch.key for c in [*run.cands.values(), *run.side] for ch in c.changes)
        assert max(held.values(), default=1) == 1
        assert not set(held) & set(run.landed)


@pytest.mark.parametrize("r", CHAIN, ids=lambda r: r.name.split(" ")[0])
def test_the_candidates_are_one_chain_on_main(tr, r):
    for run in walk(tr, r):
        bases = [c.base for c in run.cands.values()]
        assert len(set(bases)) == len(bases)
        assert all(b == run.main or b in run.cands for b in bases)


@pytest.mark.parametrize("r", qr.RULES, ids=lambda r: r.name.split(" ")[0])
def test_a_change_reaches_main_no_sooner_than_the_queue_could_take_it(tr, r):
    run = qr.Replay(tr, r).run()
    assert run.landed
    for key, at in run.landed.items():
        ch = tr.changes[key]
        assert ch.after is not None or at >= ch.spans[0][0]


@pytest.mark.parametrize("r", [r for r in qr.RULES if not r.first], ids=lambda r: r.name.split(" ")[0])
def test_a_change_a_real_check_failed_on_passes_no_check(tr, r):
    run = qr.Replay(tr, r).run()
    assert not [key for key in run.landed if tr.changes[key].fault is not None]
    assert not run.red and not run.counts["reverts"]


def test_landing_first_puts_a_change_at_fault_on_main_until_its_check_fails(tr):
    run = qr.Replay(tr, rule("5.")).run()
    m = qr.measure(tr, run)
    assert m["reverts"] == len(run.red) == 4 and m["red_times"] == 2 and m["on_red"] > 0
    assert all(a < b for a, b in run.red)
    assert not [key for key in run.landed if tr.changes[key].fault is not None]
    # only plain changes land before their check: every other went through a candidate, and no plain one did
    checked = {ch.key for c in run.built for ch in c.changes}
    assert all(tr.changes[key].plain != (key in checked) for key in run.landed)


def test_reverting_every_change_a_failure_names_reverts_changes_not_at_fault(tr):
    one, each = (qr.measure(tr, qr.Replay(tr, rule(n)).run()) for n in ("5.", "5b."))
    assert each["reverts"] > one["reverts"] and each["lost"] > one["lost"]


def test_nothing_is_built_while_a_lean_batch_waits_for_its_records_unless_the_rule_builds_beside_it(tr):
    def built_beside(r: qr.Rule) -> int:
        n, seen = 0, 0
        for run in walk(tr, r):
            waits = any(c.start is None for c in [*run.cands.values(), *run.side])
            new = [c for c in run.built[seen:] if c.start is not None]
            seen = len(run.built)
            n += bool(waits and new)
        return n

    assert built_beside(rule("1b.")) == 0
    assert built_beside(rule("2a.")) > 0


def test_side_by_side_a_failed_batch_drops_no_other(tr):
    run = qr.Replay(tr, rule("3.")).run()
    assert "dropped base" not in run.lost and run.counts["unions"] > 0
    assert all(c.base == "main" or c.base in {b.sha for b in run.built} for c in run.built)
    chain = qr.Replay(tr, rule("2a.")).run()
    assert chain.lost.get("dropped base", 0) > 0


def test_a_check_alone_runs_only_on_a_place_nothing_else_needs(tr):
    twins = 0
    for run in walk(tr, rule("4.")):
        running = [c for c in run.cands.values() if c.twin is not None and run.t < c.twin[1]]
        twins += len(running)
        assert not running or run._busy() <= qr.PLACES
    assert twins > 0


def test_the_places_held_at_once_are_counted_over_time():
    got = qr.peak([(0.0, 10.0, 3), (5.0, 20.0, 3), (5.0, 6.0, 20)], 0.0, 20.0)
    assert got["peak"] == 26 and got["p90_places"] == 6
    assert got["over"] == pytest.approx(1 / 20)


def test_the_loops_flags_are_read_as_its_parser_reads_them():
    assert qr._flags("--batch 16\n# --depth 9\n--depth 3 --lean=2\n--batch 32\n") == {"batch": 32, "depth": 3, "lean": 2}
    assert qr._flags("--no-records\n")["records"] is False


def test_a_lean_change_compiles_when_a_lean_source_and_a_lock_spec_or_locked_package_change():
    locks = ["verity/protocols/flock/verifier"]
    assert qr.lean_compiles(["security/Definitions/Core.lean"], locks)
    assert qr.lean_compiles(["verity/protocols/flock/verifier/Flock/Main.lean"], locks)
    assert not qr.lean_compiles(["security/Proofs/Core/Lemma.lean"], locks)
    assert not qr.lean_compiles(["security/lean-audit.json"], locks)
    assert qr.lean_compiles(["security/lean-audit.json", "security/Proofs/Core/Lemma.lean"], locks)
    assert not qr.lean_compiles(["tools/research/src/research/queue.py"], locks)


def test_a_file_reruns_the_suites_whose_inputs_it_is_under():
    rep = {"suites": {".": {"name": "root", "inputs": ["."], "cut": ["pkg"], "prose": True},
                      "pkg": {"name": "pkg", "inputs": ["pkg", "lib/shared.py"], "cut": ["pkg/inner"], "prose": False},
                      "pkg/inner": {"name": "inner", "inputs": ["pkg/inner"], "cut": [], "prose": False}},
           "runner": ["uv.lock"], "seconds": {"root": 600.0, "pkg": 120.0, "inner": 60.0}}
    su = qr.Suites(rep, 100.0, 400.0)
    assert su.whole == {"."}
    assert su.touched(frozenset({"pkg/a.py"})) == {".", "pkg"}
    assert su.touched(frozenset({"pkg/inner/a.py"})) == {".", "pkg/inner"}
    assert su.touched(frozenset({"lib/shared.py"})) == {".", "pkg"}
    assert su.touched(frozenset({"pkg/README.md"})) == {"."}
    assert su.touched(frozenset({"uv.lock"})) == {".", "pkg", "pkg/inner"}
    assert su.speed == pytest.approx(780 / 300)
    assert su.wall({"."}) == pytest.approx(700.0)
    assert su.wall(set()) == 100.0
    assert su.inputs(".") == frozenset() and su.inputs("pkg") == {"pkg", "lib/shared.py"} and su.inputs("gone") is None


def test_two_real_batches_are_compared_by_the_suites_both_rerun(tr):
    got = qr.pairs(tr, DOC)
    assert (got["pairs"], got["none_shared"], got["suites"]) == (13, 0, 122)
    assert got["whole_min"] < got["rerun_p50_min"] <= got["rerun_p90_min"]


def test_each_lean_batch_built_records_first_says_how_long_no_plain_batch_followed(tr):
    got = qr.lean_first(tr, DOC)
    assert [(round(h["minutes"]), h["in_queue"], h["after_drop"]) for h in got] == [
        (98, 1, True), (84, 0, True), (39, 2, False), (20, 2, False), (2, 0, True)]
    assert all(h["plain_waiting"] > 0 for h in got)


def test_the_cut_of_a_cut_document_is_itself():
    hours = (DOC["now"] - qt.ts(DOC["day"])) / 3600
    assert qr.cut(DOC, hours) == DOC
    assert FIXTURE.stat().st_size < 128 * 1024


def test_the_table_has_a_row_for_what_happened_and_one_for_each_rule(tr, capsys):
    rows = [qr.row(tr, r, 1) for r in qr.RULES]
    text = qr.render(tr, rows, qr.pairs(tr, DOC), qr.lean_first(tr, DOC))
    table = [line for line in text.splitlines() if line.startswith("|")]
    assert len(table) == 3 + len(qr.RULES) and table[2].startswith("| Measured | 0% |")
    assert all(line.count("|") == table[0].count("|") for line in table)
    assert "Assumptions:" in text and all(f"- {a}" in text for a in qr.ASSUMPTIONS)
    assert qr.main(["--doc", str(FIXTURE), "--seeds", "1", "--rules", "1.,5."]) == 0
    out = capsys.readouterr().out
    assert out.count("\n| ") == 4 and "reverts" in out
