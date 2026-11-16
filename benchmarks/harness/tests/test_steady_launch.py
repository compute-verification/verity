"""A type at its full statement and at a quarter of one, and plan, check and run over a steady-state spec's points."""

from __future__ import annotations

import json

import pytest
from steady_standins import (
    ART,
    FULL,
    IN_ART,
    OWN,
    PIN,
    QUARTER,
    RECORD,
    SETUP,
    SPEC,
    STAGE,
    TIMED,
    Store,
    git,
    tree,
)
from test_steady import DROP, made

from verity_bench import jobs, served_table
from verity_bench import steady_plan as P
from verity_bench import steady_spec as SP
from verity_bench.cli import _parser, main
from verity_bench.tool import STEADY_POINT


@pytest.fixture
def sized(tmp_path):
    """A spec that names `gemm64` at its full statement and at a quarter of it, with a store."""

    root, path = tree(tmp_path, spec={"row": [FULL, QUARTER, SPEC["row"][1]]})
    return root, path, SP.load_spec(root / path, root=root), Store(tmp_path, root)


def test_a_spec_names_a_type_at_its_full_statement_and_a_quarter_side_by_side_and_launches_both(sized):
    root, path, spec, _ = sized
    assert [item.id for item in spec.points] == ["gemm64-n8-k1", "gemm64-n8-k2", "gemm64-n2-k2", "head_block-k2"]
    assert [(item.row.n, item.row.pin) for item in spec.points[1:3]] == [(8, PIN), (2, PIN)]
    plan = P.make_plan(root / path, root=root, lookup=lambda spec, item, identity: (None, None))
    assert len({item.identity for item in plan.points}) == 4, "the statement's units are in a point's identity"
    commands = []
    assert P.dispatch(plan, root / path, root=root, direct="vy-dc-h200",
                      execute=lambda command, cwd: commands.append(command) or 0) == 0
    assert [STEADY_POINT.parse(command) for command in commands] == [
        {"spec": path, "point": item.point.id, "identity": item.identity} for item in plan.points]
    assert [SP.point(spec, STEADY_POINT.parse(command)["point"]).row.n for command in commands] == [8, 8, 2, 4]


def test_a_full_row_is_no_passed_point_of_its_quarter_nor_a_quarters_row_of_the_full_statement(sized):
    """The pin is the block circuit's digest, so a quarter's row and its full statement's hold one `pin`: the
    statement's blocks tell them apart."""

    _, _, spec, _ = sized
    full, quarter = SP.point(spec, "gemm64-n8-k2"), SP.point(spec, "gemm64-n2-k2")
    row, files = made(OWN | {SP.VERSIONS: DROP} | TIMED[8])
    assert SP.point_why(row, files, spec, full) is None
    assert SP.point_why(row, files, spec, quarter) == "fail: its statement holds 8 blocks; the point's `n` is 2"
    row, files = made(OWN | {SP.VERSIONS: DROP} | TIMED[2])
    assert SP.point_why(row, files, spec, quarter) is None
    assert SP.point_why(row, files, spec, full) == "fail: its statement holds 2 blocks; the point's `n` is 8"


def test_the_full_point_and_the_quarter_each_match_their_own_labelled_run(sized):
    _, _, spec, store = sized
    full, quarter = SP.point(spec, "gemm64-n8-k2"), SP.point(spec, "gemm64-n2-k2")
    identity = {item.id: SP.point_identity(spec, item, "f" * 64) for item in (full, quarter)}

    def lookup(item):
        return store.lookup()(spec, item, identity[item.id])

    full_run = store.ran("r20231114-100000-0001", store.files({SP.VERSIONS: DROP} | TIMED[8]))
    store.label(full_run, "run_of_record", RECORD)
    assert lookup(full) == (full_run, None)
    assert lookup(quarter) == (None, None), "the full statement's row is no run of the quarter's"
    quarter_run = store.ran("r20231114-110000-0002", store.files({SP.VERSIONS: DROP} | TIMED[2]))
    store.label(quarter_run, "run_of_record", RECORD)
    assert lookup(full) == (full_run, None), "the newer quarter's row is no run of the full statement's"
    assert lookup(quarter) == (quarter_run, None)

    store.label(quarter_run, "superseded_by", "r20231116-100000-0009")
    wrong = store.ran("r20231115-100000-0003", store.files(TIMED[8]), identity=identity[quarter.id])
    assert lookup(quarter) == (None, f"{wrong}: fail: its statement holds 8 blocks; the point's `n` is 2"), (
        "a run of the quarter's identity that timed the full statement; the full statement's labelled row isn't named")
    own = store.ran("r20231115-110000-0004", store.files(TIMED[2]), identity=identity[quarter.id])
    assert lookup(quarter) == (own, None) and lookup(full) == (full_run, None)


def test_a_run_of_both_sizes_at_one_count_gives_each_point_its_own_row():
    full, quarter = (SP.SteadyPoint(SP.SteadyRow("gemm64", n, PIN, (2,), 2), 2, 16, sized=True) for n in (8, 2))
    rows = [made(OWN | TIMED[n])[0] for n in (8, 2)]
    files = made(rates=json.dumps(rows))[1]
    assert (SP.own_row(files, full), SP.own_row(files, quarter)) == tuple(rows)
    assert SP.own_row(made(rates=json.dumps([rows[0], rows[0]]))[1], full) == (
        "cannot tell: the run's `out/rate/rate.json` holds 2 rows of `gemm64` of 8 blocks at 2 provers")


#: a row's statement as it states it, and the blocks `row_blocks` reads, or why it reads none
ROW_BLOCKS = [
    ({"n": 8}, 8),
    ({"n": 8, "stage": {"blocks": 8}}, 8),
    ({"stage": {"blocks": 8}}, 8),
    ({"n": 8, "stage": {"blocks": 2}}, "its row's `n` (8) and `stage.blocks` (2) aren't one count of its statement's blocks"),
    ({"n": 8, "stage": {"blocks": 8.0}}, ("its row's `n` (8) and `stage.blocks` (8.0) aren't one count of its statement's "
                                          "blocks")),
    ({"n": True}, "its row's `n` (True) isn't one count of its statement's blocks"),
    ({"n": 0, "stage": "8"}, "its row's `n` (0) isn't one count of its statement's blocks"),
    ({}, f"its row doesn't say its statement's blocks (`n`, `stage.blocks`, which {SP.TIMER} writes)"),
]


@pytest.mark.parametrize("row, blocks", ROW_BLOCKS, ids=[str(row) for row, _ in ROW_BLOCKS])
def test_a_rows_statement_blocks_are_its_n_and_its_stage_blocks_where_both_say_one_count(row, blocks):
    assert SP.row_blocks(row) == blocks


# ---- plan, check and run ---------------------------------------------------------------------------------------------------


def test_plan_and_check_take_a_steady_spec_and_say_why_a_point_has_no_passing_run(tmp_path, monkeypatch, capsys):
    root, path = tree(tmp_path)
    git(root)
    store = Store(tmp_path, root)
    points = P.make_plan(root / path, root=root, lookup=lambda spec, item, identity: (None, None)).points
    run = store.ran("r20231115-100000-0001", store.files(), identity=points[1].identity)
    head = {"type": "head_block", "pin": "ab" * 64, "n": 4, "stage": STAGE | {"blocks": 4}}
    failed = store.ran("r20231115-110000-0002", store.files(head | {"flat": False}), identity=points[2].identity,
                       rc=SP.NOT_PASSED)
    monkeypatch.chdir(root)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "here"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(store.config))

    assert main(["check", path, "--local-only"]) == 1
    assert "0/3 points" in capsys.readouterr().out
    assert main(["check", path]) == 1
    out = capsys.readouterr().out
    assert "1/3 points" in out
    assert f"gemm64-k1\t{points[0].identity}\tmissing\n" in out
    assert f"gemm64-k2\t{points[1].identity}\t{run}\n" in out
    assert f"head_block-k2\t{points[2].identity}\tmissing\t{failed}: fail: `flat` (fail): its statement cache's hit rate wasn't flat" in out

    assert main(["plan", path, "--local-only"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["resolved_spec"]["schema"] == SP.SCHEMA and plan["toolchain"]["files"] == 5
    assert [(point["point"]["name"], point["point"]["provers"], point["passing_run"]) for point in plan["points"]] == [
        ("gemm64", 1, None), ("gemm64", 2, run), ("head_block", 2, None)]
    assert plan["points"][2]["why"].startswith(f"{failed}: fail: `flat` (fail)") and "why" not in plan["points"][1]


def test_a_spec_outside_the_specs_directory_or_without_history_is_looked_up_on_this_machine_alone(tmp_path, monkeypatch):
    """The remote's attempts are pulled since the spec was added, which git says of a spec under `benchmarks/harness/specs`."""

    root, path = tree(tmp_path)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "here"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(Store(tmp_path, root).config))
    (elsewhere := root / "elsewhere.toml").write_text((root / path).read_text())
    with pytest.raises(ValueError, match=f"elsewhere.toml isn't a spec of {served_table.SPECS}, .*--local-only"):
        P.make_plan(elsewhere, root=root)
    with pytest.raises(ValueError, match="git has no commit adding benchmarks/harness/specs/steady.toml"):
        P.make_plan(root / path, root=root)
    assert [point.passing_run for point in P.make_plan(elsewhere, root=root, local_only=True).points] == [None] * 3


def test_run_launches_each_point_without_a_passing_run_through_a_gpu_slot_of_the_node(tmp_path):
    """The command is the slot-only launcher's form; the node side takes the card itself, so no lease wraps it."""

    from research.pods.nebius.slot_only import through_slot

    root, path = tree(tmp_path)
    plan = P.make_plan(root / path, root=root,
                       lookup=lambda spec, item, identity: ("r-passed", None) if item.id == "gemm64-k1" else (None, "why"))
    assert [point.passing_run for point in plan.points] == ["r-passed", None, None]
    commands = []
    assert P.dispatch(plan, root / path, root=root, direct="vy-dc-h200",
                      execute=lambda command, cwd: commands.append((list(command), cwd)) or 0) == 0
    assert [cwd for _, cwd in commands] == [root, root]
    for (command, _), item in zip(commands, plan.points[1:], strict=True):
        flags, workload = command[:command.index("--")], command[command.index("--") + 1:]
        assert flags[:4] == ["research", "run", "--on", "vy-dc-h200"] and "--queue" not in flags and "--gpus" not in flags
        assert flags[flags.index("--source") + 1] == str(root) and flags[flags.index("--cwd") + 1] == "source"
        assert flags[flags.index("--timeout") + 1] == f"{jobs.DIRECT_WAIT_S + 5400}" and "--require-result" in flags
        assert flags[flags.index("--tool") + 1] == "verity_bench.tool:STEADY_POINT"
        assert [flags[i + 1] for i, token in enumerate(flags) if token == "--env"] == [
            "RUST_MIN_STACK=1073741824", f"VERITY_BENCH_SETUP={SETUP}"]
        assert through_slot(workload) and "gpu-lease" not in command
        assert workload[:4] == ["env", f"CHECK_SLOTS={jobs.GPU_SLOTS}", "python3", "tools/check/slot.py"]
        assert workload[workload.index("--wait-max") + 1] == f"{jobs.DIRECT_WAIT_S}"
        assert workload[workload.index("--run-max") + 1] == "5400"
        assert workload[workload.index("--", 4) + 1:] == [
            "bash", "benchmarks/harness/steady_point.sh", "--spec", path, "--point", item.point.id,
            "--identity", item.identity]
        assert STEADY_POINT.parse(command) == {"spec": path, "point": item.point.id, "identity": item.identity}


def test_a_run_given_the_booked_periods_end_never_ends_after_it_and_one_that_could_not_finish_is_not_launched(tmp_path):
    """`run --end`: slot.py waits only as long as the spec's max_minutes and slot.py's last try (`SLOT_GRACE_S`) still
    fit before the end, the run's timeout is that wait and max_minutes, and a point whose max_minutes don't fit at its
    launch is refused, saying so."""

    root, path = tree(tmp_path)
    plan = P.make_plan(root / path, root=root, lookup=lambda spec, item, identity: (None, "why"))
    commands = []
    assert P.dispatch(plan, root / path, root=root, direct="vy-dc-h200", end=1_007_200, now=lambda: 1_000_000.4,
                      execute=lambda command, cwd: commands.append(list(command)) or 0) == 0
    assert len(commands) == 3
    for command in commands:
        flags, workload = command[:command.index("--")], command[command.index("--") + 1:]
        assert flags[flags.index("--timeout") + 1] == "7139"
        assert workload[workload.index("--wait-max") + 1] == "1739"
        assert workload[workload.index("--run-max") + 1] == "5400"
    launched = []
    with pytest.raises(ValueError, match=r"up to 90 min .* don't fit before its end, 1970-01-12T15:16:40Z: it would "
                                         r"not finish, so it isn't launched"):
        P.dispatch(plan, root / path, root=root, direct="vy-dc-h200", end=1_005_400, now=lambda: 1_000_000.0,
                   execute=lambda command, cwd: launched.append(command) or 0)
    assert launched == []
    assert _parser().parse_args(["run", "x", "--direct", "n", "--end", "1970-01-12T15:46Z"]).end == 1_007_160
    with pytest.raises(SystemExit):
        _parser().parse_args(["run", "x", "--end", "15:46"])


def test_a_point_named_again_is_launched_though_a_passing_run_stands_for_it_and_plan_says_so(tmp_path):
    """A labelled row stands for a point by its pin, provers and cores, so without `--again` a first run of the harness
    at a point the record already has is never launched."""

    root, path = tree(tmp_path)
    plan = P.make_plan(root / path, root=root, lookup=lambda spec, item, identity: ("r-passed", None))
    ran = []
    assert P.dispatch(plan, root / path, root=root, direct="vy-dc-h200", execute=lambda c, cwd: ran.append(c) or 0) == 0
    assert ran == [], "every point passed: nothing is launched"
    again = jobs.again(plan, ["gemm64-k2"])
    assert [item.again for item in again.points] == [False, True, False]
    assert [point.get("again") for point in jobs.plan_json(again)["points"]] == [None, "passed by r-passed", None]
    assert [point["passing_run"] for point in jobs.plan_json(again)["points"]] == ["r-passed"] * 3, "nothing superseded"
    assert P.dispatch(again, root / path, root=root, direct="vy-dc-h200", execute=lambda c, cwd: ran.append(c) or 0) == 0
    assert [STEADY_POINT.parse(command)["point"] for command in ran] == ["gemm64-k2"]
    with pytest.raises(ValueError, match=r"--again gemm64-k3: not a point of spec 'steady', whose points are gemm64-k1, "):
        jobs.again(plan, ["gemm64-k3"])
    with pytest.raises(SystemExit):
        main(["run", path, "--direct", "vy-dc-h200", "--again"])


def test_a_point_on_a_booked_card_is_launched_through_the_booked_slot_and_names_the_card(tmp_path):
    """The calendar books a card and its slot's cores; the first free slot, and either card of its NUMA node, may be
    another lane's booking."""

    root, path = tree(tmp_path)
    plan = P.make_plan(root / path, root=root, lookup=lambda spec, item, identity: (None, None))
    ran = []
    assert P.dispatch(plan, root / path, root=root, direct="vy-spot-glm53", card=3, cores="34-43",
                      execute=lambda c, cwd: ran.append(c) or 0) == 0
    for command, item in zip(ran, plan.points, strict=True):
        workload = command[command.index("--") + 1:]
        assert workload[:7] == ["taskset", "-c", "34-43", "env", f"CHECK_SLOTS={jobs.GPU_SLOTS}", "python3",
                                "tools/check/slot.py"]
        assert workload[workload.index("--", 7) + 1:] == [
            "bash", "benchmarks/harness/steady_point.sh", "--spec", path, "--point", item.point.id,
            "--identity", item.identity, "--card", "3"]
        assert STEADY_POINT.parse(command) == {"spec": path, "point": item.point.id, "identity": item.identity}
    with pytest.raises(ValueError, match=r"--card 3 needs --cores, the booked slot's cores"):
        P.dispatch(plan, root / path, root=root, direct="vy-spot-glm53", card=3, execute=lambda c, cwd: 0)


def test_a_point_whose_statement_is_in_the_store_is_launched_with_a_read_only_key_to_it(tmp_path):
    """Only that point's run reads the store: a row staged in the run, or from a directory on the node, gets no key."""

    root, path = tree(tmp_path)
    spec = root / path
    spec.write_text(spec.read_text().replace('staged = "/dev/shm/head"', f'staged = "{ART}/{IN_ART}"'))
    plan = P.make_plan(spec, root=root, lookup=lambda spec, item, identity: (None, None))
    assert [point.point.row.stored for point in plan.points] == [None, None, (ART, IN_ART)]
    commands = []
    assert P.dispatch(plan, spec, root=root, direct="vy-dc-h200",
                      execute=lambda command, cwd: commands.append(command) or 0) == 0
    flags = [command[:command.index("--")] for command in commands]
    assert ["--store-read" in f for f in flags] == [False, False, True]
    assert STEADY_POINT.parse(commands[2])["point"] == plan.points[2].point.id


def test_a_launch_that_fails_stops_the_launches_after_it(tmp_path):
    root, path = tree(tmp_path)
    plan = P.make_plan(root / path, root=root, lookup=lambda spec, item, identity: (None, None))
    ran = []

    def execute(command, cwd):
        ran.append(command[command.index("--point") + 1])
        return 0 if len(ran) < 2 else 75

    assert P.dispatch(plan, root / path, root=root, direct="vy-dc-h200", execute=execute) == 75
    assert ran == ["gemm64-k1", "gemm64-k2"]


def test_run_refuses_a_steady_spec_without_a_node_a_node_without_slots_and_a_spec_without_a_bound(tmp_path):
    root, path = tree(tmp_path)
    plan = P.make_plan(root / path, root=root, lookup=lambda spec, item, identity: (None, None))
    launched = []

    def execute(command, cwd):
        launched.append(command)
        return 0

    with pytest.raises(ValueError, match="is a steady-state spec.*`verity-bench run SPEC --direct NODE`"):
        P.dispatch(plan, root / path, root=root, execute=execute)
    with pytest.raises(ValueError, match="--direct vy-nebius-cpu-1: not a GPU node with gpu slots"):
        P.dispatch(plan, root / path, root=root, direct="vy-nebius-cpu-1", execute=execute)
    unbounded, path = tree(tmp_path, spec={"admission": None}, name="unbounded")
    plan = P.make_plan(unbounded / path, root=unbounded, lookup=lambda spec, item, identity: (None, None))
    with pytest.raises(ValueError, match="admission max_minutes"):
        P.dispatch(plan, unbounded / path, root=unbounded, direct="vy-dc-h200", execute=execute)
    assert launched == []
