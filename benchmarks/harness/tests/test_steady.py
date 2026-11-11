"""`verity_bench.steady` on made-up files: a run whose files show all nine checks, each check failing alone, each check the
files can't show, a run's rows read from its rate file and from a directory, the window and its thirds on stated numbers,
and the four conditions the reader of these rows still holds itself."""

from __future__ import annotations

import json

import pytest

from verity_bench import prove_records as R
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
    """A made-up run that passes all nine, and its one row: two provers on card 3, two warm sessions each and the timed
    ones filed `gaps` apart (16 of them unless `gaps` says otherwise), one lease of card 3, and the card clear at the
    row's start and end. The row's window is the sessions' own: from the second timed session's filing to the third
    from last's, or every timed session where that leaves under two. `changes`: fields of the row replaced, or left out
    (`DROP`); `texts`: a file's text replaced (`leases`, `start`, `end`, `sessions`), or the file left out (None)."""

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
           "stage": {"definition": "Gemm_v1{K=64}", "k_log": 20, "blocks": 8, "full": 8}} | (changes or {})
    row = {k: v for k, v in row.items() if v is not DROP}
    files = {S.RATES: json.dumps([row]), S.LEASES: lease(int(CARD)), START: NODE, END: NODE,
             SESSIONS: "".join(json.dumps(line) + "\n" for line in lines)}
    files |= {{"leases": S.LEASES, "start": START, "end": END, "sessions": SESSIONS}[name]: text for name, text in texts.items()}
    return row, {path: text for path, text in files.items() if text is not None}.get


def test_a_row_whose_files_show_all_nine_checks_passes():
    row, files = made()
    got = S.verdict(row, files)
    assert [c.name for c in got.checks] == list(S.CHECKS) and len(S.CHECKS) == 9
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
    ("flat", {"changes": {"flat": False}}, "its statement cache's hit rate wasn't flat"),
    ("no_throttling", {"changes": {"throttled": {"nr_throttled": 3, "throttled_usec": 250000}}},
     "its CPUs were throttled 3 times while it ran, for 0.25 s"),
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
    ("every_prover_warmed", {"changes": {"warm_per_prover": {"prover-gpu3-0": 4}}},
     "1 of its 2 provers finished a warm session before the timed ones"),
    ("every_prover_warmed", {"changes": {"idle_provers": 1}}, "1 of its 2 provers filed no session in the window"),
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
    ("one_lease_own_card", {"leases": None}, "the run holds no `gpu-lease-usage.jsonl`, which gpu-lease writes when a lease ends"),
    ("one_lease_own_card", {"leases": ""}, "the run's `gpu-lease-usage.jsonl` holds no lease"),
    ("one_lease_own_card", {"leases": "not json\n"}, "line 1 of `gpu-lease-usage.jsonl` isn't JSON (Expecting value)"),
    ("one_lease_own_card", {"leases": json.dumps({"run": "r-own"}) + "\n"},
     "the lease in `gpu-lease-usage.jsonl` doesn't say which cards it held"),
    ("card_clear_start_end", {"end": None}, "the run holds no record of its card at the row's end (`out/cotenants-k2-end.txt`)"),
    ("card_clear_start_end", {"start": ""}, "`out/cotenants-k2-start.txt` isn't a record of the node: it has no `epoch` line"),
    ("card_clear_start_end", {"end": "epoch 1700000000\napp nvidia-smi gave no answer\n"},
     "`out/cotenants-k2-end.txt` has no list of compute processes: nvidia-smi gave no answer"),
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
    ("every_prover_warmed", {"changes": {"warm_per_prover": DROP}}, "its row lacks `warm_per_prover`"),
    ("every_prover_warmed", {"changes": {"idle_provers": DROP}}, "its row lacks `idle_provers`"),
]


@pytest.mark.parametrize(("name", "lack", "why"), UNTOLD, ids=[f"{name}-{n}" for n, (name, _, _) in enumerate(UNTOLD)])
def test_a_check_the_runs_files_cannot_show_is_not_a_pass(name, lack, why):
    got = S.verdict(*made(**lack))
    assert [(c.name, c.status, c.why) for c in got.checks if not c.ok] == [(name, S.CANNOT_TELL, why)]
    assert (got.status, got.passed, got.why) == (S.CANNOT_TELL, False, f"`{name}` (cannot tell): {why}")


def test_every_check_has_a_case_that_fails_it_and_one_its_files_cannot_show():
    assert {name for name, _, _ in FAILS} == {name for name, _, _ in UNTOLD} == set(S.CHECKS)


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
    with pytest.raises(ValueError, match=r"the run's `out/rate/rate\.json` isn't a list of rows"):
        S.verdicts({S.RATES: json.dumps({"rows": [row]})}.get)
    with pytest.raises(ValueError, match=r"the run's `out/rate/rate\.json` isn't JSON \(Expecting value\)"):
        S.verdicts({S.RATES: "[" + json.dumps(row) + ","}.get)


def test_rows_at_one_prover_count_share_a_record_of_the_card_that_clears_none_of_them():
    row, files = made()
    twin = row | {"type": "norm"}
    both = {S.RATES: json.dumps([row, twin]), f"out/rate/norm-k{K}/sessions.jsonl": files(SESSIONS)}
    got = S.verdicts(lambda path: both.get(path, files(path)))
    assert [(v.type, v.status) for v in got] == [(TYPE, S.CANNOT_TELL), ("norm", S.CANNOT_TELL)]
    for v in got:
        assert [(c.name, c.why) for c in v.checks if not c.ok] == [(
            "card_clear_start_end", "the run holds 2 rows at 2 provers and one record of its card for them "
                                    "(`out/cotenants-k2-start.txt`)")]


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


def test_the_reader_holds_four_of_the_checks_itself_and_gives_the_same_reasons():
    """`prove_records.row_why` reads the same rows and holds `accepted`, `not_void`, `exits` and `flat` as conditions of
    its own. Until it calls this module's, the two must give one answer and one reason on a row."""

    row, _ = made()
    assert R.RATES == S.RATES and R.row_why(row) is None
    assert all(check(row).ok for check in (S.accepted, S.not_void, S.exits, S.flat))
    for check, change in ((S.accepted, {"accepted": 19}), (S.accepted, {"refused_n": 2}),
                          (S.not_void, {"void": ["2 of 20 sessions refused", "exited nonzero: prover-gpu3-0 rc 1"]}),
                          (S.exits, {"exits": {"prover-gpu3-0": 0, "prover-gpu3-1": 137}}),
                          (S.exits, {"exits": [0, 1]}), (S.flat, {"flat": False}), (S.flat, {"flat": None})):
        got = check(row | change)
        assert got.status == S.FAIL and got.why == R.row_why(row | change), change
