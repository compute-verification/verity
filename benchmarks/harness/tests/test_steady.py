"""`verity_bench.steady` on made-up files: a run whose files show all ten checks, each check failing alone, each check the
files can't show, a value that isn't a number passing nothing, a run's rows read from its rate file and from a directory,
the window and its thirds on stated numbers, and a window cut at whole bursts where a row declares its sessions' period,
with no session of it left unjudged."""

from __future__ import annotations

import json
import math

import pytest

from verity_bench import steady as S

TYPE, K, WARM, CARD = "gemm64", 2, 2, "3"
SESSIONS = f"out/rate/{TYPE}-k{K}/sessions.jsonl"
START, END = f"out/cotenants-k{K}-start.txt", f"out/cotenants-k{K}-end.txt"
#: a field a made-up row is left without
DROP = object()
#: seconds between one timed session's filing and the next: 2 s a statement throughout
EVEN = (2.0,) * 15
#: what else was on the node: another run's lease and its compute process, on another card
NODE = "epoch 1700000000\nloadavg 1.00 1.00 1.00 1/100 1\nlease 0 held 5m pid=11 run=r-other who=other\napp GPU 0 pid 12 100MiB\n"


def lease(*cards: int) -> str:
    return json.dumps({"schema": "gpu-lease/usage/v1", "run": "r-own", "rc": 0,
                       "gpus": [{"index": card, "uuid": f"GPU-{card}", "held_s": 600} for card in cards]}) + "\n"


def made(changes: dict | None = None, *, gaps=EVEN, **texts: str | None) -> tuple[dict, S.Files]:
    """A made-up run that passes all ten, and its one row: two provers on card 3 with ten CPUs of one NUMA node, two
    warm sessions each and the timed ones filed `gaps` apart (16 of them unless `gaps` says otherwise), one lease of
    card 3, and the card clear at the row's start and end. The row's window is the sessions' own: from the second timed session's filing to the third
    from last's, or every timed session where that leaves under two. `changes`: fields of the row replaced, or left out
    (`DROP`); `texts`: a file's text replaced (`rates`, `leases`, `start`, `end`, `sessions`), or the file left out
    (None)."""

    filed = [100.0]
    for gap in gaps:
        filed.append(filed[-1] + gap)
    timed = len(filed)
    lines = [{"i": i, "filed": float(i), "prover": f"prover-gpu{CARD}-{i % K}"} for i in range(K * WARM)]
    lines += [{"i": K * WARM + j, "filed": t, "prover": f"prover-gpu{CARD}-{j % K}"} for j, t in enumerate(filed)]
    first, last = (K - 1, timed - 1 - K) if timed - 2 * K >= 2 else (0, timed - 1)
    span = filed[last] - filed[first]
    row = {"type": TYPE, "provers": K, "gpus": 1, "device": CARD, "warm": WARM, "sessions": timed,
           "accepted": K * WARM + timed, "refused_n": 0, "unfiled": 0,
           "exits": {f"prover-gpu{CARD}-0": 0, f"prover-gpu{CARD}-1": 0, "verifier-ab-0": 0, "verifier-ab-1": 0},
           "statements_per_s": round((last - first) / span, 4), "window_sessions": last - first, "window_s": round(span, 2),
           "flat": True, "throttled": {"nr_throttled": 0, "throttled_usec": 0},
           "warm_per_prover": {f"prover-gpu{CARD}-0": WARM, f"prover-gpu{CARD}-1": WARM}, "idle_provers": 0,
           "cores": {"count": 10, "list": "34-43", "numa_nodes": [1]},
           "stage": {"definition": "Gemm_v1{K=64}", "k_log": 20, "blocks": 8, "full": 8}} | (changes or {})
    row = {k: v for k, v in row.items() if v is not DROP}
    files = {S.RATES: json.dumps([row]), S.LEASES: lease(int(CARD)), START: NODE, END: NODE,
             SESSIONS: "".join(json.dumps(line) + "\n" for line in lines)}
    files |= {{"rates": S.RATES, "leases": S.LEASES, "start": START, "end": END, "sessions": SESSIONS}[name]: text
              for name, text in texts.items()}
    return row, {path: text for path, text in files.items() if text is not None}.get


def test_a_row_whose_files_show_all_ten_checks_passes():
    row, files = made()
    got = S.verdict(row, files)
    assert [c.name for c in got.checks] == list(S.CHECKS) and len(S.CHECKS) == 10
    assert all(c == S.Check(c.name, S.PASS) for c in got.checks)
    assert (got.status, got.passed, got.why) == (S.PASS, True, None)
    assert got.record() == {"type": TYPE, "provers": K, "status": "pass", "level": 1.25,
                            "checks": {name: {"status": "pass", "why": None} for name in S.CHECKS}}


#: each check failing alone: the check, what is changed in the made-up run, and what the check then says
FAILS = [
    ("accepted", {"changes": {"accepted": 19}}, "19 of its 20 sessions accepted"),
    ("accepted", {"changes": {"refused_n": 1}}, "1 sessions refused"),
    ("accepted", {"changes": {"unfiled": 1}}, "1 sessions never filed by their verifier"),
    ("exits", {"changes": {"exits": {"prover-gpu3-0": 0, "prover-gpu3-1": 139}}}, "exited nonzero: prover-gpu3-1 rc 139"),
    ("exits", {"changes": {"exits": {"prover-gpu3-0": None}}}, "exited nonzero: prover-gpu3-0 rc None"),
    ("not_void", {"changes": {"void": ["out of device memory: prover-gpu3-0"], "statements_per_s": None}},
     "void: out of device memory: prover-gpu3-0"),
    ("not_void", {"changes": {"statements_per_s": None}}, "no rate: its `statements_per_s` is None"),
    ("not_void", {"changes": {"statements_per_s": math.inf}}, "no rate: its `statements_per_s` is inf"),
    ("not_void", {"changes": {"statements_per_s": math.nan}}, "no rate: its `statements_per_s` is nan"),
    ("flat", {"changes": {"flat": False}}, "its statement cache's hit rate wasn't flat"),
    ("no_throttling", {"changes": {"throttled": {"nr_throttled": 3, "throttled_usec": 250000}}},
     "its CPUs were throttled 3 times while it ran, for 0.25 s"),
    ("no_throttling", {"changes": {"throttled": {"nr_throttled": 3, "throttled_usec": math.nan}}},
     "its CPUs were throttled 3 times while it ran"),
    ("one_lease_own_card", {"leases": lease(3) + lease(3)}, "the run took 2 leases"),
    ("one_lease_own_card", {"leases": lease(4)}, "its lease held GPU 4; its provers ran on GPU 3"),
    ("one_lease_own_card", {"leases": lease(3, 4)}, "its lease held GPU 3, 4; its provers ran on GPU 3"),
    ("one_lease_own_card", {"changes": {"gpus": 2}}, "its row counts 2 GPUs on a lease of one"),
    ("card_clear_start_end", {"start": NODE + "lease 3 held 9m pid=13 run=r-other who=other\n"},
     "GPU 3 wasn't its own: at its start, another run's lease (3 held 9m pid=13 run=r-other who=other)"),
    ("card_clear_start_end", {"end": NODE + "app GPU 3 pid 14 2048MiB\n"},
     "GPU 3 wasn't its own: at its end, a compute process (pid 14 2048MiB)"),
    ("level", {"gaps": (2.0,) * 9 + (2.75,) * 6},
     "its window isn't level: a statement took 2, 2, 2.75 s in its thirds, the slowest over 1.25 times the fastest"),
    ("level", {"changes": {"period": 2}, "gaps": (2.0,) * 9 + (2.75,) * 6},
     "its window isn't level: a statement took 2, 2, 2.75 s in its parts of whole bursts of 4 sessions, the slowest "
     "over 1.25 times the fastest"),
    # 31 sessions in bursts of 8: three whole bursts at 2 s a statement, then seven sessions at 10 s
    ("level", {"changes": {"period": 4}, "gaps": (1.0, *(2.0,) * 24, *(10.0,) * 7, 1.0, 1.0)},
     "its window isn't level: a statement took 2, 2, 9 s in its parts of whole bursts of 8 sessions counted back from "
     "its end, the slowest over 1.25 times the fastest"),
    ("every_prover_warmed", {"changes": {"warm_per_prover": {"prover-gpu3-0": 4}}},
     "1 of its 2 provers finished a warm session before the timed ones"),
    ("every_prover_warmed", {"changes": {"idle_provers": 1}}, "1 of its 2 provers filed no session in the window"),
    ("one_numa_node", {"changes": {"cores": {"count": 10, "list": "20-23,24-29", "numa_nodes": [0, 1]}}},
     "its CPUs (20-23,24-29) are on 2 NUMA nodes (0, 1)"),
    # the verifiers placed on a third node's CPUs: the row's CPUs are the provers' and theirs
    ("one_numa_node", {"changes": {"cores": {"count": 12, "list": "34-43,72-73", "numa_nodes": [3, 1]},
                                   "verifier_cpus": "72-73"}}, "its CPUs (34-43,72-73) are on 2 NUMA nodes (1, 3)"),
    ("one_numa_node", {"changes": {"cores": {"count": 96, "numa_nodes": [0, 1, 2, 3]}}},
     "its CPUs are on 4 NUMA nodes (0, 1, 2, 3)"),
]


@pytest.mark.parametrize(("name", "change", "why"), FAILS, ids=[f"{name}-{n}" for n, (name, _, _) in enumerate(FAILS)])
def test_each_check_fails_alone(name, change, why):
    got = S.verdict(*made(**change))
    assert [(c.name, c.status, c.why) for c in got.checks if not c.ok] == [(name, S.FAIL, why)]
    assert (got.status, got.passed, got.why) == (S.FAIL, False, f"`{name}` (fail): {why}")


#: each check the run's files can't show: the check, what the made-up run lacks, and what the check then says
UNTOLD = [
    ("accepted", {"changes": {"unfiled": DROP}}, "its row lacks `unfiled`"),
    ("accepted", {"changes": {"refused_n": None}}, "`refused_n` (None) of its row isn't a count"),
    ("exits", {"changes": {"exits": DROP}}, "its row lacks `exits`"),
    ("exits", {"changes": {"exits": {}}}, "its row's `exits` ({}) names no process"),
    ("not_void", {"changes": {"statements_per_s": DROP}}, "its row lacks `statements_per_s`"),
    ("flat", {"changes": {"flat": DROP}}, "its row lacks `flat`"),
    ("no_throttling", {"changes": {"throttled": DROP}}, "its row lacks `throttled.nr_throttled`"),
    ("no_throttling", {"changes": {"throttled": {}}}, "its row lacks `throttled.nr_throttled`"),
    ("one_lease_own_card", {"changes": {"gpus": DROP}}, "its row lacks `gpus`"),
    ("one_lease_own_card", {"changes": {"gpus": None}}, "`gpus` (None) of its row isn't a count"),
    ("one_lease_own_card", {"leases": None}, "the run holds no `gpu-lease-usage.jsonl`, which gpu-lease writes when a lease ends"),
    ("one_lease_own_card", {"leases": ""}, "the run's `gpu-lease-usage.jsonl` holds no lease"),
    ("one_lease_own_card", {"leases": "not json\n"}, "line 1 of `gpu-lease-usage.jsonl` isn't JSON (Expecting value)"),
    ("one_lease_own_card", {"leases": json.dumps({"run": "r-own"}) + "\n"},
     "the lease in `gpu-lease-usage.jsonl` doesn't say which cards it held"),
    ("card_clear_start_end", {"end": None}, "the run holds no record of its card at the row's end (`out/cotenants-k2-end.txt`)"),
    ("card_clear_start_end", {"start": ""}, "`out/cotenants-k2-start.txt` isn't a record of the node: it has no `epoch` line"),
    ("card_clear_start_end", {"end": "epoch 1700000000\napp nvidia-smi gave no answer\n"},
     "`out/cotenants-k2-end.txt` has no list of compute processes: nvidia-smi gave no answer"),
    ("card_clear_start_end", {"end": "epoch 1700000000\nlease 0 held 5m\nnvidia-smi gave no answer\n"},
     "`out/cotenants-k2-end.txt` has no list of compute processes: nvidia-smi gave no answer"),
    ("card_clear_start_end", {"start": "epoch 1700000000\ngpu-lease status gave no answer (exit 1)\napp GPU 0 pid 12 100MiB\n"},
     "`out/cotenants-k2-start.txt` has no list of other runs' leases: gpu-lease status gave no answer (exit 1)"),
    ("card_clear_start_end", {"start": "epoch 1700000000\nthe node gave no answer\n"},
     "`out/cotenants-k2-start.txt` has no list of what else was on the node: the node gave no answer"),
    ("card_clear_start_end", {"rates": None},
     "the run holds no `out/rate/rate.json`: it would say whether the record of its card is this row's alone"),
    ("card_clear_start_end", {"rates": "[{"},
     "the run's `out/rate/rate.json` isn't JSON (Expecting property name enclosed in double quotes): it would say whether "
     "the record of its card is this row's alone"),
    ("card_clear_start_end", {"start": NODE + "app GPU ? pid 15 1MiB\n"},
     "`out/cotenants-k2-start.txt` names a compute process on a card it couldn't name (pid 15 1MiB)"),
    ("level", {"sessions": None}, "the run holds no `out/rate/gemm64-k2/sessions.jsonl`"),
    ("level", {"sessions": json.dumps({"i": 0, "filed": 1.0}) + "\n"},
     "`out/rate/gemm64-k2/sessions.jsonl` holds 1 sessions; its row ran 20"),
    ("level", {"changes": {"window_s": 30.0}},
     "the window `out/rate/gemm64-k2/sessions.jsonl` gives (12 sessions over 24.00 s) isn't its row's (12 over 30.0 s)"),
    ("level", {"changes": {"window_sessions": 11}},
     "the window `out/rate/gemm64-k2/sessions.jsonl` gives (12 sessions over 24.00 s) isn't its row's (11 over 24.0 s)"),
    ("level", {"changes": {"window_s": DROP}}, "its row lacks `window_s`"),
    ("level", {"changes": {"window_s": math.nan}}, "`window_s` (nan) of its row isn't a window's seconds"),
    ("level", {"changes": {"window_s": math.inf}}, "`window_s` (inf) of its row isn't a window's seconds"),
    ("level", {"changes": {"period": 0}}, "`period` (0) of its row isn't a count of sessions a prover, at least 1"),
    ("level", {"changes": {"period": 4.0}}, "`period` (4.0) of its row isn't a count of sessions a prover, at least 1"),
    ("level", {"changes": {"period": 4}},
     "its window of 12 sessions can't be cut into three parts of whole bursts of 8 sessions that each took time"),
    ("every_prover_warmed", {"changes": {"warm_per_prover": DROP}}, "its row lacks `warm_per_prover`"),
    ("every_prover_warmed", {"changes": {"idle_provers": DROP}}, "its row lacks `idle_provers`"),
    ("one_numa_node", {"changes": {"cores": DROP}}, "its row lacks `cores.numa_nodes`"),
    # a count alone, or the CPUs without their nodes, as a row completed before the script wrote them may hold
    ("one_numa_node", {"changes": {"cores": 10}}, "its row lacks `cores.numa_nodes`"),
    ("one_numa_node", {"changes": {"cores": {"count": 10, "list": "34-43"}}}, "its row lacks `cores.numa_nodes`"),
    ("one_numa_node", {"changes": {"cores": {"count": 10, "list": "0-9", "numa_nodes": []}}},
     "`cores.numa_nodes` of its row names no NUMA node: its machine listed none"),
    ("one_numa_node", {"changes": {"cores": {"count": 10, "list": "34-43", "numa_nodes": None}}},
     "`cores.numa_nodes` (None) of its row isn't a list of NUMA nodes"),
    ("one_numa_node", {"changes": {"cores": {"count": 10, "list": "34-43", "numa_nodes": 1}}},
     "`cores.numa_nodes` (1) of its row isn't a list of NUMA nodes"),
    ("one_numa_node", {"changes": {"cores": {"count": 10, "list": "34-43", "numa_nodes": "1"}}},
     "`cores.numa_nodes` ('1') of its row isn't a list of NUMA nodes"),
    ("one_numa_node", {"changes": {"cores": {"count": 10, "list": "34-43", "numa_nodes": [1, -1]}}},
     "`cores.numa_nodes` ([1, -1]) of its row isn't a list of NUMA nodes"),
]


@pytest.mark.parametrize(("name", "lack", "why"), UNTOLD, ids=[f"{name}-{n}" for n, (name, _, _) in enumerate(UNTOLD)])
def test_a_check_the_runs_files_cannot_show_is_not_a_pass(name, lack, why):
    got = S.verdict(*made(**lack))
    assert [(c.name, c.status, c.why) for c in got.checks if not c.ok] == [(name, S.CANNOT_TELL, why)]
    assert (got.status, got.passed, got.why) == (S.CANNOT_TELL, False, f"`{name}` (cannot tell): {why}")


def test_every_check_has_a_case_that_fails_it_and_one_its_files_cannot_show():
    assert {name for name, _, _ in FAILS} == {name for name, _, _ in UNTOLD} == set(S.CHECKS)


#: every number a check reads in a row, by its place in the row
NUMBERS = [("provers",), ("gpus",), ("warm",), ("sessions",), ("accepted",), ("refused_n",), ("unfiled",),
           ("exits", "prover-gpu3-0"), ("statements_per_s",), ("window_sessions",), ("window_s",),
           ("throttled", "nr_throttled"), ("warm_per_prover", "prover-gpu3-1"), ("idle_provers",), ("period",),
           ("cores", "numa_nodes", 0)]
#: what isn't a number, though JSON or Python lets it stand where one goes
NOT_NUMBERS = [math.nan, math.inf, -math.inf, True, "2", [2]]


@pytest.mark.parametrize("place", NUMBERS, ids=[".".join(map(str, place)) for place in NUMBERS])
def test_a_value_that_is_not_a_number_passes_no_check(place):
    row, files = made({"period": 2})
    assert S.verdict(row, files).passed
    for value in NOT_NUMBERS:
        changed = json.loads(json.dumps(row))
        at = changed
        for key in place[:-1]:
            at = at[key]
        at[place[-1]] = value
        got = S.verdict(changed, files)
        assert not got.passed, (place, value)
        # and the same value read back from the run's own rate file, as JSON's reader gives it
        own = {S.RATES: json.dumps([changed])}
        [read] = S.verdicts(lambda path, own=own: own.get(path, files(path)))
        assert read.status == got.status, (place, value)


def test_an_integer_no_float_holds_is_no_seconds_and_no_rate():
    huge = 10 ** 400
    assert [c.name for c in S.verdict(*made({"statements_per_s": huge})).checks if c.status == S.FAIL] == ["not_void"]
    assert [c.name for c in S.verdict(*made({"window_s": huge})).checks if c.status == S.CANNOT_TELL] == ["level"]
    throttled = S.verdict(*made({"throttled": {"nr_throttled": 3, "throttled_usec": huge}}))
    assert throttled.why == "`no_throttling` (fail): its CPUs were throttled 3 times while it ran"


@pytest.mark.parametrize("field", ["i", "filed"])
def test_a_session_whose_index_or_filing_time_is_not_a_number_shows_no_level_window(field):
    row, files = made()
    lines = [json.loads(line) for line in files(SESSIONS).splitlines()]
    for value in (*NOT_NUMBERS, *([10 ** 400] if field == "filed" else [])):
        text = "".join(json.dumps(line | ({field: value} if n == 9 else {})) + "\n" for n, line in enumerate(lines))
        got = S.verdict(row, lambda path, text=text: text if path == SESSIONS else files(path))
        assert [(c.name, c.status, c.why) for c in got.checks if not c.ok] == [(
            "level", S.CANNOT_TELL, "`out/rate/gemm64-k2/sessions.jsonl` isn't each session's index `i` and its `filed` time")]


def test_a_fail_outranks_what_cannot_be_told_and_the_reason_names_both():
    got = S.verdict(*made({"flat": DROP, "throttled": {"nr_throttled": 1}}))
    assert (got.status, got.passed) == (S.FAIL, False)
    assert got.why == ("`flat` (cannot tell): its row lacks `flat`; "
                       "`no_throttling` (fail): its CPUs were throttled 1 times while it ran")
    # a card another run held outranks a record that is missing at the row's other end
    other = S.verdict(*made(start=NODE + "lease 3 held 9m pid=13 run=r-other who=other\n", end=None))
    assert [(c.name, c.status) for c in other.checks if not c.ok] == [("card_clear_start_end", S.FAIL)]


def test_a_runs_rows_are_judged_from_its_rate_file(tmp_path):
    row, files = made()
    assert [(v.type, v.provers, v.status) for v in S.verdicts(files)] == [(TYPE, K, S.PASS)]
    # the same files in a directory, as a run record's tree is
    for path in (S.RATES, S.LEASES, START, END, SESSIONS):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(files(path))
    read = S.files_in(tmp_path)
    assert read(SESSIONS) == files(SESSIONS) and read("out/none.txt") is None and read("out") is None
    assert [v.record() for v in S.verdicts(read)] == [S.verdict(row, files).record()]
    (tmp_path / S.LEASES).unlink()
    [got] = S.verdicts(read)
    assert (got.status, [c.name for c in got.checks if not c.ok]) == (S.CANNOT_TELL, ["one_lease_own_card"])
    # a row that failed has no rate and nothing else to show
    failed = {S.RATES: json.dumps([{"type": TYPE, "provers": 4, "error": "the server exited"}]), S.LEASES: lease(3)}.get
    [got] = S.verdicts(failed)
    assert got.status == S.FAIL and {c.name: c.status for c in got.checks} == {
        name: S.FAIL if name == "not_void" else S.CANNOT_TELL for name in S.CHECKS}
    assert got.checks[S.CHECKS.index("not_void")].why == "failed: the server exited"
    # a run with no rate file, or one that isn't a list of rows, has no row to judge
    with pytest.raises(ValueError, match=r"the run holds no `out/rate/rate\.json`: it has no row to judge"):
        S.verdicts({}.get)
    with pytest.raises(ValueError, match=r"the run's `out/rate/rate\.json` isn't a list of rows: it has no row to judge"):
        S.verdicts({S.RATES: json.dumps({"rows": [row]})}.get)
    with pytest.raises(ValueError, match=r"the run's `out/rate/rate\.json` isn't JSON \(Expecting value\): it has no row"):
        S.rows({S.RATES: "[" + json.dumps(row) + ","}.get)
    assert S.rows(files) == [row]


def test_rows_at_one_prover_count_share_a_record_of_the_card_that_clears_none_of_them():
    row, files = made()
    twin = row | {"type": "norm"}
    both = {S.RATES: json.dumps([row, twin]), f"out/rate/norm-k{K}/sessions.jsonl": files(SESSIONS)}
    shared = lambda path: both.get(path, files(path))  # noqa: E731
    why = "the run holds 2 rows at 2 provers and one record of its card for them (`out/cotenants-k2-start.txt`)"
    # one row judged against the run's files, as a caller with a row in hand does, and every row of the run
    for got in (S.verdict(row, shared), S.verdict(twin, shared), *S.verdicts(shared)):
        assert got.status == S.CANNOT_TELL
        assert [(c.name, c.why) for c in got.checks if not c.ok] == [("card_clear_start_end", why)]
    # the row as completed after the run is judged against the same files, and shares the same record
    assert S.verdict(row | {"flat": True, "verify_s": 0.5}, shared).why == f"`card_clear_start_end` (cannot tell): {why}"
    # a card the shared record shows taken clears no row either, and fails none: the record is of one row's start
    taken = lambda path: NODE + "lease 3 held 9m pid=13 run=r-other who=other\n" if path == START else shared(path)  # noqa: E731
    assert [c.why for c in S.verdict(row, taken).checks if not c.ok] == [why]
    # a row at another prover count in the same run has its own record
    other = {S.RATES: json.dumps([row, row | {"provers": 4}])}
    assert S.verdict(row, lambda path: other.get(path, files(path))).passed


def test_the_window_is_the_scripts_and_its_thirds_are_equal_counts_of_sessions():
    # K provers: from the K-th timed session's filing to the last but K's
    assert S.window([float(t) for t in range(16)], 2) == [float(t) for t in range(1, 14)]
    assert S.window([float(t) for t in range(24)], 4) == [float(t) for t in range(3, 20)]
    # where that leaves under two sessions, every timed session
    assert S.window([0.0, 1.0, 2.0, 3.0, 4.0], 2) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert S.window([], 1) == []
    # twelve sessions in three fours; thirteen in 4, 4 and 5; two sessions give two figures
    assert S.thirds([0.0, 2.0, 4.0, 6.0, 8.0, 11.0, 14.0, 17.0, 20.0, 24.0, 28.0, 32.0, 36.0]) == [2.0, 3.0, 4.0]
    assert S.thirds([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0]) == [1.0, 2.0, 3.0]
    assert S.thirds([0.0, 1.0, 3.0]) == [1.0, 2.0]


def test_the_slowest_third_may_take_level_times_the_fastest_and_no_more():
    assert S.LEVEL == 1.25
    at = S.verdict(*made(gaps=(2.0,) * 9 + (2.5,) * 6))
    assert at.passed
    # the thirds are cut by the filing times in order, whichever provers filed them, and only the timed sessions count
    first = S.verdict(*made(gaps=(2.75,) * 5 + (2.0,) * 10))
    assert [c.why for c in first.checks if not c.ok] == [
        "its window isn't level: a statement took 2.75, 2, 2 s in its thirds, the slowest over 1.25 times the fastest"]
    # a window too short for three thirds isn't shown level
    row, files = made(gaps=(2.0,) * 3)
    lines = files(SESSIONS).splitlines()
    assert (len(lines), row["window_sessions"], row["window_s"]) == (8, 3, 6.0) and S.verdict(row, files).passed
    short = row | {"sessions": 3, "accepted": 7, "window_sessions": 2, "window_s": 4.0}
    got = S.verdict(short, {SESSIONS: "\n".join(lines[:7]) + "\n"}.get)
    assert got.checks[S.CHECKS.index("level")] == S.Check(
        "level", S.CANNOT_TELL, "its window of 2 sessions can't be cut into three thirds that each took time")


#: two provers that each file in a cycle of four sessions with one long gap a cycle, so a burst is eight sessions: the 36
#: gaps of a window, in seconds. Both provers keep the cycle, and a third of twelve gaps holds one long gap or two
KEPT = (1.07, 0.69, 4.08, 0.2, 1.0, 0.42, 10.43, 1.17, 0.11, 0.58, 4.08, 0.37,
        0.89, 0.52, 10.73, 0.05, 1.66, 0.03, 2.57, 1.55, 0.01, 1.32, 9.26, 1.73,
        0.41, 0.49, 4.06, 0.0, 0.3, 1.35, 8.62, 1.74, 0.49, 1.51, 2.05, 1.57)
#: the same two provers where one leaves the cycle halfway: the window's last burst comes in about half the others' time
LEFT = (0.34, 0.04, 8.59, 0.17, 1.25, 0.06, 18.09, 0.67, 0.09, 1.57, 5.68, 0.45,
        7.32, 0.08, 11.22, 2.61, 4.06, 0.1, 0.28, 6.75, 1.42, 5.87, 2.24, 13.59,
        0.04, 2.1, 0.01, 5.19, 1.35, 5.24, 0.0, 4.74, 7.53, 5.25, 0.77, 1.48)


def around(window: tuple[float, ...]) -> tuple[float, ...]:
    """The gaps of a made-up run of two provers whose window's gaps are `window`: one filing before it and two after."""

    return (1.0, *window, 1.0, 1.0)


def spread(row: dict, files: S.Files, burst: int | None = None, *, back: bool = False) -> float:
    """The slowest part of a row's window over its fastest: its thirds, or its parts of whole bursts of `burst`, counted
    from its start or `back` from its end."""

    lines = [json.loads(line) for line in files(SESSIONS).splitlines()]
    times = S.window(sorted(line["filed"] for line in lines if line["i"] >= K * WARM), K)
    took = S.thirds(times) if burst is None else S.burst_parts(times, burst, back=back)
    return max(took) / min(took)


def test_a_row_that_declares_its_period_has_its_window_cut_at_whole_bursts():
    # no period: thirds, as ever. They cut a burst of eight, and the middle one holds two long gaps where the others hold one
    row, files = made(gaps=around(KEPT))
    assert row["window_sessions"] == 36 and spread(row, files) == pytest.approx(1.342, abs=0.001)
    cut = S.verdict(row, files)
    assert [(c.name, c.status) for c in cut.checks if not c.ok] == [("level", S.FAIL)]
    assert cut.why.endswith("s in its thirds, the slowest over 1.25 times the fastest")
    assert S.verdict(row | {"period": None}, files).record() == cut.record()
    # a period of four sessions a prover: the window's four whole bursts, each a part, are within the line, counted from
    # its start and, for the four sessions after them, back from its end
    row, files = made({"period": 4}, gaps=around(KEPT))
    assert spread(row, files, 8) == pytest.approx(1.123, abs=0.001) and S.verdict(row, files).passed
    assert spread(row, files, 8, back=True) == pytest.approx(1.051, abs=0.001)
    # a row that isn't level by whole bursts fails with its period declared as it does without
    row, files = made({"period": 4}, gaps=around(LEFT))
    assert spread(row, files) == pytest.approx(1.648, abs=0.001) and spread(row, files, 8) == pytest.approx(1.838, abs=0.001)
    left = S.verdict(row, files)
    assert [(c.name, c.status) for c in left.checks if not c.ok] == [("level", S.FAIL)]
    assert left.why.endswith("s in its parts of whole bursts of 8 sessions, the slowest over 1.25 times the fastest")
    assert not S.verdict(*made(gaps=around(LEFT))).passed


def test_the_parts_of_a_window_cut_at_whole_bursts_are_a_third_of_its_bursts_each():
    def times(*bursts: float, burst: int = 4, odd: int = 0) -> list[float]:
        """Filing times of a window whose bursts take the given seconds a statement, with `odd` sessions after them."""
        out = [0.0]
        for took in (*[t for t in bursts for _ in range(burst)], *[100.0] * odd):
            out.append(out[-1] + took)
        return out

    # three, six and nine bursts: the window's thirds
    assert S.burst_parts(times(1.0, 2.0, 3.0), 4) == [1.0, 2.0, 3.0]
    assert S.burst_parts(times(1.0, 3.0, 2.0, 2.0, 3.0, 5.0), 4) == [2.0, 2.0, 4.0]
    nine = times(1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0)
    assert S.burst_parts(nine, 4) == S.thirds(nine) == [2.0, 2.0, 3.0]
    # four or five bursts: each burst; seven: three parts of two and the one left over
    assert S.burst_parts(times(1.0, 2.0, 3.0, 4.0), 4) == [1.0, 2.0, 3.0, 4.0]
    assert S.burst_parts(times(1.0, 2.0, 3.0, 4.0, 5.0), 4) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert S.burst_parts(times(1.0, 3.0, 2.0, 2.0, 3.0, 5.0, 7.0), 4) == [2.0, 2.0, 4.0, 7.0]
    # the sessions after the last whole burst are in no part, and a window of under three bursts gives under three figures
    assert S.burst_parts(times(1.0, 2.0, 3.0, odd=3), 4) == [1.0, 2.0, 3.0]
    assert S.burst_parts(times(1.0, 2.0, odd=3), 4) == [1.0, 2.0] and S.burst_parts(times(odd=3), 4) == []
    # counted back from the window's end, the parts hold those sessions and leave out as many at its start; the shorter
    # part of the bursts left over comes first
    assert S.burst_parts(times(1.0, 2.0, 3.0, odd=3), 4, back=True) == [1.75, 2.75, 75.75]
    assert S.burst_parts(times(1.0, 3.0, 2.0, 2.0, 3.0, 5.0, 7.0), 4, back=True) == [1.0, 2.5, 2.5, 6.0]
    assert S.burst_parts(times(1.0, 2.0, odd=3), 4, back=True) == [1.75, 75.5]
    whole = times(1.0, 3.0, 2.0, 2.0, 3.0, 5.0)
    assert S.burst_parts(whole, 4, back=True) == S.burst_parts(whole, 4)
    # a window that is whole bursts, a multiple of three of them, reads the same by its thirds and by its bursts
    for provers, period, bursts in ((1, 4, 30), (2, 4, 3), (4, 4, 9), (3, 5, 6)):
        window = times(*[1.0 + (i * 7 % 5) / 10 for i in range(bursts)], burst=period * provers)
        assert S.burst_parts(window, period * provers) == pytest.approx(S.thirds(window))


def test_a_window_that_is_not_whole_bursts_is_also_cut_back_from_its_end():
    # three whole bursts of eight at 2 s a statement, then seven sessions at 10 s: by thirds the last third shows them
    tail = around((2.0,) * 24 + (10.0,) * 7)
    row, files = made(gaps=tail)
    assert row["window_sessions"] == 31 and spread(row, files) == pytest.approx(78 / 11 / 2)
    assert [(c.name, c.status) for c in S.verdict(row, files).checks if not c.ok] == [("level", S.FAIL)]
    # with the period declared, the whole bursts from the window's start are level, and the cut back from its end isn't
    row, files = made({"period": 4}, gaps=tail)
    assert (spread(row, files, 8), spread(row, files, 8, back=True)) == (1.0, 4.5)
    got = S.verdict(row, files)
    assert [(c.name, c.status) for c in got.checks if not c.ok] == [("level", S.FAIL)]
    assert got.why.endswith("took 2, 2, 9 s in its parts of whole bursts of 8 sessions counted back from its end, the "
                            "slowest over 1.25 times the fastest")
    # the same seven sessions at the window's start fail the cut from its start
    head = S.verdict(*made({"period": 4}, gaps=around((10.0,) * 7 + (2.0,) * 24)))
    assert head.why.endswith("took 9, 2, 2 s in its parts of whole bursts of 8 sessions, the slowest over 1.25 times the "
                             "fastest")
    # sessions after the last whole burst that keep the pace pass, and a window of whole bursts is cut once
    assert S.verdict(*made({"period": 4}, gaps=around((2.0,) * 31))).passed
    assert S.verdict(*made({"period": 4}, gaps=around((2.0,) * 16 + (2.5,) * 8))).passed
    # under three whole bursts neither cut has three parts
    short = S.verdict(*made({"period": 4}, gaps=around((2.0,) * 23)))
    assert short.why == ("`level` (cannot tell): its window of 23 sessions can't be cut into three parts of whole bursts "
                         "of 8 sessions that each took time")


@pytest.mark.parametrize("period", [None, 2], ids=["thirds", "bursts"])
def test_no_session_of_a_window_is_left_out_of_its_cut(period):
    """One slow session anywhere in a window fails `level`, whatever the window's length: no cut leaves one unjudged."""

    for sessions in range(12, 25):
        for slow in range(sessions):
            gaps = [1.0] * sessions
            gaps[slow] = 100.0
            row, files = made({} if period is None else {"period": period}, gaps=around(tuple(gaps)))
            got = S.verdict(row, files)
            assert [(c.name, c.status) for c in got.checks if not c.ok] == [("level", S.FAIL)], (sessions, slow)
