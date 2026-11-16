"""A steady-state point's passing run: which run or labelled row of the store is the point's, and what plan says of the rest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from steady_standins import CUDA, POINT, RECORD, STAGE, Store, tree
from test_steady import DROP, made

from verity_bench import prove_records as R
from verity_bench import steady as S
from verity_bench import steady_node as N
from verity_bench import steady_plan as P
from verity_bench import steady_spec as SP


@pytest.fixture
def planned(tmp_path):
    """The made-up tree's spec, its point of two provers, that point's identity, and a store."""

    root, path = tree(tmp_path)
    spec = SP.load_spec(root / path, root=root)
    item = SP.point(spec, POINT)
    identity = SP.point_identity(spec, item, "f" * 64)
    return spec, item, identity, Store(tmp_path, root)


def test_a_run_of_the_point_whose_files_pass_the_checks_is_its_passing_run(planned):
    spec, item, identity, store = planned
    run = store.ran("r20231115-100000-0001", store.files(), identity=identity)
    assert store.lookup()(spec, item, identity) == (run, None)
    assert store.lookup(local_only=True)(spec, item, identity) == (None, None), "this machine has read nothing"
    assert store.lookup()(spec, SP.point(spec, "gemm64-k1"), "e" * 64) == (None, None), "another point's identity"


def test_a_verdict_is_recomputed_from_the_runs_files_never_read_from_the_run(planned):
    """A run that said it passed and whose files fail a check isn't the passing run; one that said it didn't pass
    (`NOT_PASSED`) and whose files pass by the rules as they are now is."""

    spec, item, identity, store = planned
    passed_then = store.ran("r20231115-100000-0001", store.files({"flat": False}), identity=identity, rc=0)
    assert store.lookup()(spec, item, identity) == (None, f"{passed_then}: fail: `flat` (fail): its statement cache's hit rate wasn't flat")
    failed_then = store.ran("r20231115-110000-0002", store.files(), identity=identity, rc=SP.NOT_PASSED)
    assert store.lookup()(spec, item, identity) == (failed_then, None)


def test_the_newest_passing_run_is_taken_and_the_runs_that_are_not_it_are_said(planned):
    spec, item, identity, store = planned
    older = store.ran("r20231115-100000-0001", store.files(), identity=identity)
    store.ran("r20231115-110000-0002", store.files({"refused_n": 1}), identity=identity, rc=SP.NOT_PASSED)
    assert store.lookup()(spec, item, identity) == (older, None)
    newer = store.ran("r20231115-120000-0003", store.files(), identity=identity)
    assert store.lookup()(spec, item, identity) == (newer, None)


#: a run of the point that isn't its passing run: its exit, its tree, what its files hold, and what `plan` says
NOT_IT = [
    ({"rc": 143}, {}, None),
    ({"rc": N.REFUSED}, {}, None),
    ({"dirty": True}, {}, None),
    ({}, {"leases": None}, "cannot tell: `one_lease_own_card` (cannot tell): the run holds no `gpu-lease-usage.jsonl`"),
    ({}, {"start": None}, "cannot tell: `card_clear_start_end` (cannot tell): the run holds no record of its card at the row's start"),
    ({}, {"changes": {"flat": DROP}}, "cannot tell: `flat` (cannot tell): "),
    ({}, {"changes": {"cores": {"count": 8, "list": "0-7", "numa_nodes": [0]}}},
     "fail: it ran on 8 cores; the spec's `cores` is 10"),
    ({}, {"changes": {"cores": DROP}}, "cannot tell: its row doesn't say which cores it ran on"),
    ({}, {"changes": {"cores": {"count": 10, "list": "0-9"}}},
     "cannot tell: `one_numa_node` (cannot tell): its row lacks `cores.numa_nodes`"),
    ({}, {"changes": {"cores": {"count": 10, "list": "0-4,24-28", "numa_nodes": [0, 1]}}},
     "fail: `one_numa_node` (fail): its CPUs (0-4,24-28) are on 2 NUMA nodes (0, 1)"),
    ({}, {"changes": {"pin": "cd" * 64}}, "fail: its `pin` is another circuit's (cdcdcdcdcdcdcdcd...)"),
    ({}, {"changes": {"n": 2, "stage": STAGE | {"blocks": 2}}}, "fail: its statement holds 2 blocks; the point's `n` is 1"),
    ({}, {"changes": {SP.VERSIONS: {"driver": "570.86.16", "cuda_toolkit": CUDA}}}, "cannot tell: its row's `versions`"),
    ({}, {"changes": {SP.VERSIONS: DROP}}, "cannot tell: its row's `versions` (None) aren't the spec's"),
    ({}, {"changes": {"sessions": 15}}, "fail: it timed 15 sessions after 2 warm ones a prover; the point's are 16 and 2"),
    ({}, {"changes": {"warm": 1}}, "fail: it timed 16 sessions after 1 warm ones a prover; the point's are 16 and 2"),
    ({}, {"changes": {S.PERIOD: 2}},
     "cannot tell: its row's `period` (2) isn't the point's (None), which sizes its timed sessions and cuts its window"),
    ({}, {"changes": {"type": "gemm32"}}, "cannot tell: the run's `out/rate/rate.json` holds 0 rows of `gemm64` at 2 provers"),
]


@pytest.mark.parametrize("ran, files, says", NOT_IT, ids=[str(says or ran)[:48] for ran, _, says in NOT_IT])
def test_a_run_that_did_not_end_or_whose_files_do_not_show_a_passed_point_is_planned_again(planned, ran, files, says):
    spec, item, identity, store = planned
    run = store.ran("r20231115-100000-0001", store.files(files.get("changes"), **{k: v for k, v in files.items()
                                                                                 if k != "changes"}),
                    identity=identity, **ran)
    found, why = store.lookup()(spec, item, identity)
    assert found is None
    assert why is None if says is None else why.startswith(f"{run}: {says}"), why


def test_a_run_that_kept_no_files_cannot_be_told(planned):
    spec, item, identity, store = planned
    run = store.ran("r20231115-100000-0001", None, identity=identity)
    assert store.lookup()(spec, item, identity) == (None, f"{run}: cannot tell: the run kept no files")


def test_the_lookup_pulls_the_labels_of_runs_and_arts_alone_and_says_so_before_it_starts(planned):
    """The remote's labels are mostly of other things than runs and arts (the check's tasks, pull requests' heads),
    and a record's targets are runs and arts: a machine reads those, and says what it is reading before it reads."""

    spec, item, identity, store = planned
    files = store.files({SP.VERSIONS: DROP, "flat": DROP})
    run = store.ran("r20231114-100000-0001", files)
    store.label(run, "run_of_record", RECORD)
    art = completed(store, run, files, {"flat": True})
    others = ["task:" + "a" * 128, "pr:12@" + "b" * 40, "commit:" + "c" * 40]
    for other in others:
        store.label(other, "run_of_record", RECORD)
    lookup = store.lookup()
    assert lookup(spec, item, identity) == (run, None)
    assert [bool(lookup.store.labels(target)) for target in (run, art, *others)] == [True, True, False, False, False]
    assert store.said[0] == ("pulling the labels of the store's runs and arts: two listings of the remote, then each "
                             "label this store lacks")
    assert store.said[1] == "pulled 2 labels; the remote holds labels of 2 runs and arts"
    # a second lookup on the same machine reads no label again
    again = P.SteadyCompletion(lookup.store, root=store.root, since=lambda: 1_700_000_000, log=store.said.append)
    assert again(spec, item, identity) == (run, None) and "pulled 0 labels; " in store.said[-2], store.said


def test_a_labelled_row_is_a_passed_point_only_when_its_pin_count_and_cores_are_the_points_and_its_run_passes(planned):
    """A row timed outside the harness, labelled into the workload's steady-state record. Its row holds no `versions`,
    which only a run of the point's own identity is held to."""

    spec, item, identity, store = planned
    outside = {SP.VERSIONS: DROP}
    run = store.ran("r20231114-100000-0001", store.files(outside))
    assert store.lookup()(spec, item, identity) == (None, None), "not labelled: no run of the record"
    store.label(run, "run_of_record", RECORD)
    assert store.lookup()(spec, item, identity) == (run, None)
    assert store.lookup()(spec, SP.point(spec, "gemm64-k1"), identity) == (None, None), "its row is at two provers"
    assert store.lookup()(spec, SP.point(spec, "head_block-k2"), identity) == (None, None), "another circuit's pin"

    store.label(run, "superseded_by", "r20231116-100000-0009")
    assert store.lookup()(spec, item, identity) == (None, None)


#: a labelled row at the point's pin and count that isn't a passed point, and what `plan` says of it
LABELLED_NOT_IT = [
    ({"cores": DROP}, {}, "cannot tell: its row doesn't say which cores it ran on"),
    ({"cores": {"count": 16, "list": "0-15", "numa_nodes": [0]}}, {}, "fail: it ran on 16 cores; the spec's `cores` is 10"),
    ({"cores": {"count": 10, "list": "0-9"}}, {}, "cannot tell: `one_numa_node` (cannot tell): its row lacks `cores.numa_nodes`"),
    ({"cores": {"count": 10, "list": "0-4,24-28", "numa_nodes": [0, 1]}}, {},
     "fail: `one_numa_node` (fail): its CPUs (0-4,24-28) are on 2 NUMA nodes (0, 1)"),
    ({"flat": DROP}, {}, "cannot tell: `flat` (cannot tell): "),
    ({}, {"leases": None}, "cannot tell: `one_lease_own_card` (cannot tell): "),
    ({"accepted": 19}, {}, "fail: `accepted` (fail): 19 of its 20 sessions accepted"),
    ({"n": DROP, "stage": DROP}, {}, "cannot tell: its row doesn't say its statement's blocks (`n`, `stage.blocks`"),
    ({"stage": STAGE | {"blocks": 2}}, {},
     "cannot tell: its row's `n` (1) and `stage.blocks` (2) aren't one count of its statement's blocks"),
]


@pytest.mark.parametrize("changes, texts, says", LABELLED_NOT_IT, ids=[says[:40] for *_, says in LABELLED_NOT_IT])
def test_a_labelled_row_whose_files_cannot_show_a_check_is_timed_again_and_plan_says_why(planned, changes, texts, says):
    spec, item, identity, store = planned
    run = store.ran("r20231114-100000-0001", store.files({SP.VERSIONS: DROP} | changes, **texts))
    store.label(run, "run_of_record", RECORD)
    found, why = store.lookup()(spec, item, identity)
    assert found is None and why.startswith(f"{run} (labelled `{RECORD}`): {says}"), why


def completed(store: Store, run: str, files: Path, changes: dict, *, runs: list[str] | None = None,
              derivation: bool | str = True, names: list[str] | None = None, rows: list[dict] | None = None) -> str:
    """An art of the record that holds a run's rows as completed after it: the run's own rows with `changes` (a value
    of `DROP` leaves the field out; or `rows`), the run named in the art's `runs` (or `runs`), and how fields were
    derived: for each row the fields of `changes` (or `names`), named together in one key. `derivation`: False for
    none, or the file's text."""

    own = json.loads((files / S.RATES).read_text())
    if rows is None:
        rows = [{k: v for k, v in (row | changes).items() if v is not DROP} for row in own]
    store._n += 1
    (directory := store.tmp / f"completed{store._n}").mkdir()
    (directory / R.ART_RATES).write_text(json.dumps(rows))
    if derivation is True:
        named = ", ".join(sorted(changes) if names is None else names)
        derivation = json.dumps({"rows": [{"type": row["type"], "provers": row["provers"],
                                           "fields": {named: {"how": "made up"}} if named else {}} for row in own]})
    if derivation:
        (directory / P.DERIVATION).write_text(derivation)
    art = store.art("evidence/v1", {"runs": [run] if runs is None else runs}, directory)
    store.label(art, "run_of_record", RECORD)
    return art


def test_a_run_from_before_a_field_counts_on_its_completed_row_in_a_labelled_art(planned):
    """A run from before the timing script wrote `flat`. Its own row can't show the check. An art of the record that
    names the run, holds its rows with the field and says how the field was derived gives the row that is judged,
    against the run's own files; and that row is held to every check as the run's own would be."""

    spec, item, identity, store = planned
    files = store.files({SP.VERSIONS: DROP, "flat": DROP})
    run = store.ran("r20231114-100000-0001", files)
    store.label(run, "run_of_record", RECORD)
    found, why = store.lookup()(spec, item, identity)
    assert found is None and why.startswith(f"{run} (labelled `{RECORD}`): cannot tell: `flat` (cannot tell): "), why

    art = completed(store, run, files, {"flat": True})
    assert store.lookup()(spec, item, identity) == (run, None)

    unnamed = store.ran("r20231114-110000-0002", store.files({SP.VERSIONS: DROP, "flat": DROP}))
    assert store.lookup()(spec, item, identity) == (run, None), "a run no label and no labelled art names"

    store.label(art, "superseded_by", newer := completed(store, run, files, {"flat": True, "accepted": 19}))
    found, why = store.lookup()(spec, item, identity)
    assert found is None and why == (f"{run} (labelled `{RECORD}`, its row as completed in {newer[:16]}…): fail: "
                                     "`accepted` (fail): 19 of its 20 sessions accepted"), why
    assert unnamed not in why


#: an art that holds a labelled run's rows completed and isn't taken as them, and what `plan` then says of the run
UNTAKEN = [
    ({"runs": ["r20231114-100000-0001", "r20231114-090000-0000"]},
     "holds `rate.json` and names 2 runs: an art's rows are one run's"),
    ({"derivation": False}, "holds its rows without their derivation from the run's records (`derivation.json`)"),
    ({"superseded": True}, "is superseded"),
]
#: the exit codes of the made-up run's row
EXITS = made()[0]["exits"]
#: an art whose rows are more than the run's own rows completed: its rows, what its derivation names, and what `plan`
#: says the art differs in (None: nothing, the art's row is taken)
NOT_THE_RUNS = [
    ({"changes": {"flat": True, "accepted": 20, "exits": dict(EXITS)}, "names": ["flat"]}, None),
    ({"changes": {"flat": True, "accepted": 19}, "names": ["flat"]}, "`accepted` (changed)"),
    ({"changes": {"flat": True, "exits": EXITS | {"prover-gpu3-1": 1}}, "names": ["flat"]},
     "`exits.prover-gpu3-1` (changed)"),
    ({"changes": {"flat": True, "exits": EXITS | {"prover-gpu3-1": 1}}, "names": ["flat", "exits.prover-gpu3-0"]},
     "`exits.prover-gpu3-1` (changed)"),
    ({"changes": {"flat": True, "unfiled": DROP}, "names": ["flat", "unfiled"]}, "`unfiled` (dropped)"),
    ({"changes": {"flat": True, "accepted": 20.0}, "names": []}, "`accepted` (changed)"),
    ({"changes": {"flat": True, "accepted": 19, "exits": {}, "unfiled": 2, "sessions": 21, "warm": 1}, "names": []},
     ("`accepted` (changed), `exits.prover-gpu3-0` (dropped), `exits.prover-gpu3-1` (dropped), `exits.verifier-ab-0` "
      "(dropped) and 4 more")),
]


@pytest.mark.parametrize("how, says", UNTAKEN, ids=[says[:28] for _, says in UNTAKEN])
def test_a_completed_row_counts_only_from_an_art_that_names_the_run_carries_its_derivation_and_stands(planned, how, says):
    """Each condition alone: the art's row would pass, it isn't taken, and the run is judged on its own row, which
    can't show `flat`: "cannot tell", so the point is timed again, and `plan` says why the art's row wasn't taken."""

    spec, item, identity, store = planned
    files = store.files({SP.VERSIONS: DROP, "flat": DROP})
    run = store.ran("r20231114-100000-0001", files)
    store.label(run, "run_of_record", RECORD)
    how = dict(how)
    superseded = how.pop("superseded", False)
    art = completed(store, run, files, {"flat": True}, **how)
    if superseded:
        store.label(art, "superseded_by", "art:" + "9" * 64)
    found, why = store.lookup()(spec, item, identity)
    assert found is None and why == (f"{run} (labelled `{RECORD}`, its own row: {art[:16]}… {says}): cannot tell: "
                                     "`flat` (cannot tell): its row lacks `flat`"), why


@pytest.mark.parametrize("art, differs", NOT_THE_RUNS, ids=[str(differs)[:32] for _, differs in NOT_THE_RUNS])
def test_a_completed_row_is_the_runs_own_row_with_only_what_it_lacked_or_the_derivation_names(planned, art, differs):
    """An art that would pass and that changes a field the run's own row holds, without its derivation naming that
    field, isn't the run's row completed: the run is judged on its own row, and `plan` says what the art changed."""

    spec, item, identity, store = planned
    files = store.files({SP.VERSIONS: DROP, "flat": DROP})
    run = store.ran("r20231114-100000-0001", files)
    art = completed(store, run, files, **art)
    found, why = store.lookup()(spec, item, identity)
    if differs is None:
        assert (found, why) == (run, None)
        return
    assert found is None and why == (
        f"{run} (labelled `{RECORD}`, its own row: {art[:16]}… differs from the run's own row of `gemm64` at 2 provers in "
        f"{differs}: a completed row keeps every field of the run's and changes only those its `derivation.json` names): "
        "cannot tell: `flat` (cannot tell): its row lacks `flat`"), why


def test_a_derivation_names_a_field_of_a_field_and_a_name_covers_the_fields_under_it(planned):
    spec, item, identity, store = planned
    files = store.files({SP.VERSIONS: DROP, "flat": DROP, "stage": {"definition": "Gemm_v1{K=64}", "k_log": 20}})
    run = store.ran("r20231114-100000-0001", files)
    stage = {"definition": "Gemm_v1{K=64}", "k_log": 21, "blocks": 1}

    def lookup(names):
        art = completed(store, run, files, {"flat": True, "stage": stage}, names=names)
        found = store.lookup()(spec, item, identity)
        store.label(art, "superseded_by", "art:" + "9" * 64)
        return found

    # `stage.blocks` is a field the run's row lacks; `stage.k_log` is one it holds, changed
    assert lookup(["flat", "stage"]) == (run, None)
    assert lookup(["flat, stage.k_log", "stage.blocks"]) == (run, None), "several names in one key"
    for names in (["flat", "stage.blocks"], ["flat", "stage.k"], ["flat", "k_log"]):
        found, why = lookup(names)
        assert found is None and "in `stage.k_log` (changed): a completed row keeps every field of the run's" in why, why


#: an art whose rows or derivation can't be held to the run's own rows, and what `plan` says of the art
NOT_ROW_FOR_ROW = [
    ({"rows": []}, "holds 0 rows that aren't the run's 1, one for one by `type` and `provers`"),
    ({"rows": [{"type": "gemm64", "provers": 4, "flat": True}]},
     "holds 1 rows that aren't the run's 1, one for one by `type` and `provers`"),
    ({"derivation": json.dumps({"rows": [{"fields": ["flat"]}]})},
     "isn't a record of the fields it derived for each row"),
    ({"derivation": "{"}, "`derivation.json` of art:"),
    ({"derivation": "{"}, "isn't JSON (Expecting property name enclosed in double quotes)"),
    ({"derivation": json.dumps(["flat"])}, "isn't a record of the fields it derived for each row"),
]


@pytest.mark.parametrize("art, says", NOT_ROW_FOR_ROW, ids=[f"{says[:24]}-{n}" for n, (_, says) in enumerate(NOT_ROW_FOR_ROW)])
def test_a_completed_art_whose_rows_or_derivation_are_not_the_runs_row_for_row_is_not_taken(planned, art, says):
    spec, item, identity, store = planned
    files = store.files({SP.VERSIONS: DROP, "flat": DROP})
    run = store.ran("r20231114-100000-0001", files)
    art = completed(store, run, files, {"flat": True}, **art)
    found, why = store.lookup()(spec, item, identity)
    assert found is None and why.startswith(f"{run} (labelled `{RECORD}`, its own row: ") and says in why, why
    assert f"{art[:16]}…" in why and why.endswith("cannot tell: `flat` (cannot tell): its row lacks `flat`")


def test_two_standing_arts_that_complete_one_run_complete_it_for_neither(planned):
    spec, item, identity, store = planned
    files = store.files({SP.VERSIONS: DROP, "flat": DROP})
    run = store.ran("r20231114-100000-0001", files)
    first = completed(store, run, files, {"flat": True})
    second = completed(store, run, files, {"flat": True, "verify_s": 0.5})
    found, why = store.lookup()(spec, item, identity)
    assert found is None and why == (
        f"{run} (labelled `{RECORD}`, its own row: 2 standing arts hold its rows completed ({first[:16]}…, "
        f"{second[:16]}…): supersede all but one): cannot tell: `flat` (cannot tell): its row lacks `flat`"), why
    store.label(first, "superseded_by", second)
    assert store.lookup()(spec, item, identity) == (run, None)


def test_a_labelled_run_whose_rows_cannot_be_read_is_named_in_each_point_it_could_have_been(planned):
    """A run of the record that kept no files, or whose rate file is no list of rows: its rows aren't known, so it
    could have been any point, and `plan` says so for each point without a passing run."""

    spec, item, identity, store = planned
    bare = store.ran("r20231114-100000-0001", None)
    store.label(bare, "run_of_record", RECORD)
    no_files = f"{bare} (labelled `{RECORD}`): cannot tell: the run kept no files"
    for point in (item, SP.point(spec, "gemm64-k1"), SP.point(spec, "head_block-k2")):
        assert store.lookup()(spec, point, identity) == (None, no_files)

    unread = store.ran("r20231114-110000-0002", store.files(rates="{}"))
    store.label(unread, "run_of_record", RECORD)
    assert store.lookup()(spec, item, identity) == (None, (
        f"{unread} (labelled `{RECORD}`): cannot tell: the run's `out/rate/rate.json` isn't a list of rows: it has no "
        f"row to judge; {no_files}"))
    # an art that completes a run with no files completes nothing: there is no row of the run's to hold it to
    completed(store, bare, store.files({SP.VERSIONS: DROP}), {"flat": True})
    assert no_files in store.lookup()(spec, item, identity)[1]
    # a passing run of the point is still found, and a superseded run that kept no files is no run of the record
    passing = store.ran("r20231115-100000-0003", store.files({SP.VERSIONS: DROP}))
    store.label(passing, "run_of_record", RECORD)
    assert store.lookup()(spec, item, identity) == (passing, None)
    store.label(passing, "superseded_by", "r20231116-100000-0009")
    store.label(bare, "superseded_by", "r20231116-100000-0009")
    found, why = store.lookup()(spec, item, identity)
    assert found is None and bare not in why and unread in why, why


def test_a_run_is_cannot_tell_where_neither_its_own_row_nor_its_completed_one_holds_the_field(planned):
    spec, item, identity, store = planned
    files = store.files({SP.VERSIONS: DROP, "flat": DROP})
    run = store.ran("r20231114-100000-0001", files)
    art = completed(store, run, files, {"stage": STAGE | {"k_log": 21}})
    found, why = store.lookup()(spec, item, identity)
    assert found is None and why == (f"{run} (labelled `{RECORD}`, its row as completed in {art[:16]}…): cannot tell: "
                                     "`flat` (cannot tell): its row lacks `flat`"), why
    # the run is the record's through the art alone, and stops being its when the art is superseded
    store.label(art, "superseded_by", "art:" + "9" * 64)
    assert store.lookup()(spec, item, identity) == (None, None)


def test_a_run_of_the_point_is_preferred_and_plan_names_only_the_newest_few(planned):
    spec, item, identity, store = planned
    labelled = store.ran("r20231114-100000-0001", store.files({SP.VERSIONS: DROP}))
    store.label(labelled, "run_of_record", RECORD)
    own = store.ran("r20231115-100000-0002", store.files(), identity=identity)
    assert store.lookup()(spec, item, identity) == (own, None)

    other = SP.point_identity(spec, item, "0" * 64)
    runs = [store.ran(f"r20231115-11000{i}-000{i}", store.files({"flat": False}), identity=other) for i in range(5)]
    store.label(labelled, "superseded_by", runs[0])
    found, why = store.lookup()(spec, item, other)
    assert found is None and why.count("`flat` (fail)") == P.WHYS and why.endswith("; and 2 more"), why
    assert why.startswith(f"{runs[4]}: ") and runs[1] not in why
