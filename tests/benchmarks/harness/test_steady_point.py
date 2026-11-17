"""A steady-state spec's points on a made-up tree, a store with a directory remote and stand-ins for the node: the schema
loaded strictly, what a point's identity binds and what it leaves out, the sessions a point times where a row's sessions
file in bursts, which run `plan` counts as a point's passing run and why it counts no other, which labels it pulls,
the command `run --direct` launches, and the node side from its refusals to its verdict and to a signal that ends it.
No GPU, no node and no clock."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from steady_standins import ART, BURSTY, FILES, OWN, PIN, POINT, SETUP, SPEC, git, tree
from test_steady import made
from verity_config import ConfigError

from verity_bench import jobs, served_table, settings
from verity_bench import steady as S
from verity_bench import steady_plan as P
from verity_bench import steady_spec as SP


def test_a_spec_was_added_when_its_first_path_was(tmp_path, monkeypatch):
    """A spec that moved with its directory was added when the file was added under the former one: the runs recorded
    under that path started after it, and the pull of the store's attempts reaches back to them."""

    import subprocess

    from verity_bench.vllm.presets import FORMER_SPECS

    def commit(at: int, *argv: str) -> None:
        when = f"@{at} +0000"
        for step in (argv, ("add", "-A"), ("-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false",
                                           "commit", "-qm", "specs")):
            if step:
                subprocess.run(["git", *step], cwd=root, check=True,
                               env={**settings.environ(), "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when})

    monkeypatch.setattr(served_table, "SPECS", "elsewhere/specs")
    root = tmp_path / "tree"
    (root / FORMER_SPECS).mkdir(parents=True)
    (root / FORMER_SPECS / "moved.toml").write_text("moved\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    commit(1700000000)
    (root / "elsewhere" / "specs").mkdir(parents=True)
    (root / "elsewhere" / "specs" / "new.toml").write_text("new\n")
    commit(1700009000, "mv", f"{FORMER_SPECS}/moved.toml", "elsewhere/specs/moved.toml")
    assert not (root / FORMER_SPECS / "moved.toml").exists()
    assert served_table.spec_added(root, "moved") == 1700000000
    assert served_table.spec_added(root, "new") == 1700009000
    with pytest.raises(ValueError, match="git has no commit adding elsewhere/specs/neither.toml here"):
        served_table.spec_added(root, "neither")


def load(tmp_path: Path, **changes) -> SP.SteadySpec:
    root, spec = tree(tmp_path, **changes)
    return SP.load_spec(root / spec, root=root)


def identities(tmp_path: Path, name: str, **changes) -> dict[str, str]:
    """Each point's identity in a made-up tree, by the point's id."""

    root, spec = tree(tmp_path, name=name, **changes)
    plan = P.make_plan(root / spec, root=root, lookup=lambda spec, item, identity: (None, None))
    return {item.point.id: item.identity for item in plan.points}


def identity_of_the_first_row_at_its_last_count(tmp_path: Path, name: str, **changes) -> str:
    found = identities(tmp_path, name, **changes)
    return found[[point for point in found if not point.startswith("head_block")][-1]]


# ---- the schema ------------------------------------------------------------------------------------------------------------


def test_a_steady_spec_is_told_by_its_schema_and_its_points_are_each_row_at_each_count(tmp_path):
    root, path = tree(tmp_path)
    assert SP.is_steady(root / path)
    (other := tmp_path / "vllm.toml").write_text('schema = "verity-bench/vllm-spec/v0"\n')
    assert not SP.is_steady(other) and not SP.is_steady(tmp_path / "no-such-spec.toml")

    spec = SP.load_spec(root / path, root=root)
    assert [point.id for point in spec.points] == ["gemm64-k1", "gemm64-k2", "head_block-k2"]
    head = SP.point(spec, "head_block-k2")
    assert head.to_json() == {"type": "head-block", "name": "head_block", "n": 4, "pin": "ab" * 64, "warm": 2,
                              "period": None, "staged": "/dev/shm/head", "provers": 2, "sessions": 16}
    assert (spec.cores, spec.admission.max_minutes, dict(spec.env)) == (10, 90, {"RUST_MIN_STACK": "1073741824"})
    with pytest.raises(ValueError, match=r"point 'gemm64-k4' is not in spec 'steady' \(gemm64-k1, gemm64-k2, head_block-k2\)"):
        SP.point(spec, "gemm64-k4")


#: a spec the loader refuses: what is changed, and what the refusal says
REFUSED_SPECS = [
    ({"spec": {"gpus": 1}}, "gpus"),
    ({"spec": {"row0": {"device": "3"}}}, "device"),
    ({"spec": {"schema": "verity-bench/vllm-spec/v0"}}, "not a steady-state spec"),
    ({"spec": {"id": "Steady"}}, "`id` and `workload` are lowercase names"),
    ({"spec": {"card": "a100"}},
     r"`card` 'a100' is no GPU of verity_config's registry \(rtx-pro-6000, h200, h100, b200\)"),
    ({"spec": {"cores": 0}}, "`cores` is 0; it is a count of at least 1"),
    ({"spec": {"ahead": -1}}, "`ahead` is -1"),
    ({"spec": {"driver": "v570"}}, "`driver` 'v570' isn't a version"),
    ({"spec": {"cuda_toolkit": ""}}, "`cuda_toolkit` '' isn't a version"),
    ({"spec": {"cuda_toolkit": "13.3"}}, r"`cuda_toolkit` '13\.3' isn't the `V` version the node's nvcc gives"),
    ({"spec": {"setup": None}}, "setup"),
    ({"spec": {"setup": "benchmarks/harness/specs/none.sh"}}, "`setup` 'benchmarks/harness/specs/none.sh' is no script of the source tree"),
    ({"spec": {"setup": "../steady-setup.sh"}}, "is no script of the source tree"),
    ({"spec": {"env": {"CUDA_VISIBLE_DEVICES": "0"}}}, "`env` can't set 'CUDA_VISIBLE_DEVICES'"),
    ({"spec": {"env": {"VERITY_BENCH_STEADY_BIN": "/x"}}}, "`env` can't set 'VERITY_BENCH_STEADY_BIN'"),
    ({"spec": {"admission": {"max_minutes": 0}}}, r"`admission\.max_minutes` is 0"),
    ({"spec": {"row": []}}, "missing required key 'row'"),
    ({"spec": {"row0": {"type": "--"}}}, "its `type` names nothing"),
    ({"spec": {"row0": {"n": 0}}}, "`n` is 0"),
    ({"spec": {"row0": {"period": 0}}}, "`period` is 0; it is a count of at least 1"),
    ({"spec": {"row0": {"pin": PIN[:64]}}}, "`pin` isn't a SHA-512 in lowercase hexadecimal"),
    ({"spec": {"row0": {"pin": PIN.upper()}}}, "`pin` isn't a SHA-512"),
    ({"spec": {"row0": {"provers": []}}}, "`provers` is a list of distinct prover counts"),
    ({"spec": {"row0": {"provers": [2, 2]}}}, "`provers` is a list of distinct prover counts"),
    ({"spec": {"row0": {"provers": [0]}}}, r"`provers \(0\)` is 0"),
    ({"spec": {"row0": {"provers": [1, 4]}}}, "`sessions` is 16, under 8 a prover at 4 provers"),
    ({"spec": {"row0": {"staged": "z/gemm64"}}}, "`staged` 'z/gemm64' isn't an absolute path on the node"),
    ({"spec": {"row0": {"staged": "art:" + "1" * 63 + "/row"}}}, "nor a directory of an artifact in the store"),
    ({"spec": {"row0": {"staged": ART}}}, "nor a directory of an artifact in the store"),
    ({"spec": {"row0": {"staged": f"{ART}/out/../row"}}}, "nor a directory of an artifact in the store"),
    ({"spec": {"row0": {"staged": f"{ART}/out/*"}}}, "nor a directory of an artifact in the store"),
    ({"spec": {"row0": {"type": "head block", "n": 4}}}, "two rows are named head_block at n = 4 in the rate file"),
]


@pytest.mark.parametrize("changes, says", REFUSED_SPECS, ids=[says[:40] for _, says in REFUSED_SPECS])
def test_a_spec_is_loaded_strictly(tmp_path, changes, says):
    with pytest.raises(ConfigError, match=says):
        load(tmp_path, **changes)


# ---- the identity ----------------------------------------------------------------------------------------------------------

#: what a point's time depends on: each changes the point's identity
BOUND = {
    "the card": {"spec": {"card": "b200"}},
    "the cores": {"spec": {"cores": 12}},
    "the timed sessions": {"spec": {"sessions": 32}},
    "the sessions ahead": {"spec": {"ahead": 3}},
    "the driver's version": {"spec": {"driver": "570.86.16"}},
    "the CUDA toolkit's version": {"spec": {"cuda_toolkit": "12.9.41"}},
    "the run's environment": {"spec": {"env": {"RUST_MIN_STACK": "2147483648"}}},
    "which script is the setup": {"spec": {"setup": "benchmarks/harness/specs/other-setup.sh"},
                                  "files": {"benchmarks/harness/specs/other-setup.sh": FILES[SETUP]}},
    "the block type": {"spec": {"row0": {"type": "gemm32"}}},
    "the statement's units": {"spec": {"row0": {"n": 2}}},
    "the statement's pin": {"spec": {"row0": {"pin": "cd" * 64}}},
    "the warm sessions": {"spec": {"row0": {"warm": 3}}},
    "the period its sessions file in": {"spec": {"row0": {"period": 4}}},
    "the prover's source": {"files": {"verity/protocols/proofs/flock/prover/src/main.rs": "fn main() { loop {} }\n"}},
    "a kernel": {"files": {"verity/protocols/proofs/flock/prover/cuda/ntt.cu": "// another kernel\n"}},
    "a new source file": {"files": {"verity/protocols/proofs/flock/prover/src/extra.rs": "\n"}},
    "the build's flags": {"files": {"verity/protocols/proofs/flock/prover/pod/70-class-sweep.sh": "FLAGS=-O2\n"}},
    "the timing script": {"files": {"benchmarks/one_stage/server_rate.py": "# times otherwise\n"}},
    "the setup's text": {"files": {SETUP: "export VERITY_BENCH_STEADY_BIN=/other/prover\n"}},
}
#: what it doesn't depend on: none changes it
UNBOUND = {
    "the spec's name": {"spec": {"id": "steady-again"}},
    "the run's bound": {"spec": {"admission": {"max_minutes": 240}}},
    "the lease's memory": {"spec": {"admission": {"max_minutes": 90, "mem_gb": 300}}},
    "where a statement is staged": {"spec": {"row0": {"staged": "/dev/shm/gemm64"}}},
    "the row's other counts": {"spec": {"row0": {"provers": [2]}}},
    "another row": {"spec": {"row": [SPEC["row"][0], SPEC["row"][1] | {"n": 8, "warm": 4}]}},
    "the prover's tests": {"files": {"tests/verity/protocols/proofs/flock/prover/test_prover.py": "def test(): assert 1\n"}},
    "a document": {"files": {"verity/protocols/proofs/flock/prover/README.md": "the prover, again\n"}},
    "a file outside the closure": {"files": {"benchmarks/harness/verity_bench/cli.py": "# the harness\n"}},
}


@pytest.mark.parametrize("what", BOUND)
def test_the_identity_binds_everything_the_time_depends_on(tmp_path, what):
    assert (identity_of_the_first_row_at_its_last_count(tmp_path, "a")
            != identity_of_the_first_row_at_its_last_count(tmp_path, "b", **BOUND[what]))


@pytest.mark.parametrize("what", UNBOUND)
def test_the_identity_leaves_out_what_the_time_does_not_depend_on(tmp_path, what):
    assert (identity_of_the_first_row_at_its_last_count(tmp_path, "a")
            == identity_of_the_first_row_at_its_last_count(tmp_path, "b", **UNBOUND[what]))


def test_the_identity_names_no_commit_and_each_point_has_its_own(tmp_path):
    """The made-up tree is no git checkout at all: the plan and its identities need none."""

    found = identities(tmp_path, "a")
    assert list(found) == ["gemm64-k1", "gemm64-k2", "head_block-k2"] and len(set(found.values())) == 3
    assert all(re.fullmatch(r"[0-9a-f]{64}", identity) for identity in found.values())
    root, spec = tree(tmp_path)
    plan = P.make_plan(root / spec, root=root, lookup=lambda spec, item, identity: (None, None))
    covered = SP.closure(root, plan.spec)
    assert set(covered) == {"benchmarks/one_stage/server_rate.py", SETUP, "verity/protocols/proofs/flock/prover/pod/70-class-sweep.sh",
                            "verity/protocols/proofs/flock/prover/cuda/ntt.cu", "verity/protocols/proofs/flock/prover/src/main.rs"}
    assert len(covered) == 5
    assert plan.toolchain == P.Sources(plan.toolchain.files_sha256, 5)
    assert jobs.plan_json(plan)["points"][0] == {"point": plan.points[0].point.to_json(), "identity": found["gemm64-k1"],
                                                 "passing_run": None}


def test_in_a_checkout_only_tracked_files_count_so_a_shipped_tree_gives_the_same_identity(tmp_path):
    """A run ships `git archive HEAD`: what a build or an editor left in the launching checkout isn't on the node."""

    import subprocess

    shipped = identities(tmp_path, "shipped")
    root, spec = tree(tmp_path, name="checkout")
    git(root)
    assert identities(tmp_path, "checkout") == shipped
    for left in ("verity/protocols/proofs/flock/prover/target/release/flock-circuit", "verity/protocols/proofs/flock/prover/src/draft.rs",
                 "verity/protocols/proofs/flock/prover/pod/notes.sh"):
        (root / left).parent.mkdir(parents=True, exist_ok=True)
        (root / left).write_text("left by a build\n")
    assert identities(tmp_path, "checkout") == shipped, "untracked"
    assert identities(tmp_path, "checkout", files={"verity/protocols/proofs/flock/prover/src/main.rs": "fn main() { loop {} }\n"}
                      )[POINT] != shipped[POINT], "a tracked file as the working tree has it"
    subprocess.run(["git", "add", "verity/protocols/proofs/flock/prover/src/draft.rs"], cwd=root, check=True)
    assert identities(tmp_path, "checkout", files={"verity/protocols/proofs/flock/prover/src/main.rs": FILES[
        "verity/protocols/proofs/flock/prover/src/main.rs"]})[POINT] != shipped[POINT], "a newly tracked source"


@pytest.mark.parametrize("gone", ["benchmarks/one_stage/server_rate.py", "verity/protocols/proofs/flock/prover/pod/70-class-sweep.sh"])
def test_a_tree_without_a_path_the_identity_covers_is_refused(tmp_path, gone):
    root, spec = tree(tmp_path, files={gone: None})
    with pytest.raises(ValueError, match=f"holds no {re.escape(gone.rsplit('/', 1)[0] if 'pod' in gone else gone)}"):
        P.make_plan(root / spec, root=root, lookup=lambda spec, item, identity: (None, None))


def test_a_spec_of_a_workload_without_a_steady_state_record_is_refused(tmp_path):
    root, spec = tree(tmp_path, spec={"workload": "llama"})
    with pytest.raises(ValueError, match="the `workload` of spec 'steady', 'llama', has no steady-state record: the workloads are glm53"):
        P.make_plan(root / spec, root=root, lookup=lambda spec, item, identity: (None, None))


def test_a_declared_period_sizes_the_timed_sessions_so_the_window_and_its_thirds_hold_whole_bursts():
    assert [SP.timed_sessions(120, provers, None) for provers in (1, 2, 4)] == [120, 120, 120]
    assert [SP.timed_sessions(16, provers, None) for provers in (1, 2, 4)] == [16, 16, 32], "the timing script's floor"
    assert SP.timed_sessions(16, 4, 4) == 56
    assert [SP.timed_sessions(120, provers, 4) for provers in (1, 2, 4)] == [122, 124, 152]
    assert SP.timed_sessions(124, 2, 4) == 124 and SP.timed_sessions(125, 2, 4) == 148 and SP.timed_sessions(16, 2, 4) == 28
    for sessions, provers, period in [(120, 1, 4), (120, 2, 4), (120, 4, 4), (16, 2, 4), (40, 3, 5), (17, 2, 1)]:
        timed = SP.timed_sessions(sessions, provers, period)
        gaps = len(S.window([float(i) for i in range(timed)], provers)) - 1
        assert timed >= sessions and gaps == SP.window_gaps(timed, provers), "the window the timing script rates"
        assert gaps % (3 * period * provers) == 0 and timed - sessions < 3 * period * provers, "the next such count"


def test_thirds_that_cut_a_burst_read_uneven_and_the_points_count_reads_whole_bursts(tmp_path):
    """Two provers whose sessions file in bursts of eight. At 16 timed sessions the window is 12 gaps, a burst and a
    half, and its thirds read the burst's phase; at the count a period of four gives, 28, each third is one burst, and
    the row, which says its period, reads the same by its thirds and by its whole bursts."""

    spec = load(tmp_path, spec={"row0": {"period": 4}})
    item = SP.point(spec, POINT)
    assert item.sessions == 28 and item.to_json() | {"period": 4, "sessions": 28} == item.to_json()

    row, files = made(OWN, gaps=(BURSTY * 4)[:15])
    assert row["sessions"] == 16
    cut = S.verdict(row, files)
    assert [check.name for check in cut.checks if not check.ok] == ["level"] and "its window isn't level" in cut.why
    assert SP.point_why(row, files, spec, item, own=True).startswith("fail: it timed 16 sessions after 2 warm ones")

    row, files = made(OWN | {S.PERIOD: 4}, gaps=(BURSTY * 4)[:27])
    assert row["sessions"] == 28 and row["window_sessions"] == 24
    assert S.verdict(row, files).passed and SP.point_why(row, files, spec, item, own=True) is None
    assert S.verdict(row | {S.PERIOD: None}, files).passed, "its thirds are whole bursts"
    for shifted in range(1, 8):
        row, files = made(OWN | {S.PERIOD: 4}, gaps=(BURSTY * 5)[shifted:shifted + 27])
        assert SP.point_why(row, files, spec, item, own=True) is None, f"whatever the burst's phase ({shifted})"
    silent = {k: v for k, v in row.items() if k != S.PERIOD}
    assert SP.point_why(silent, files, spec, item, own=True) == (
        "cannot tell: its row's `period` (None) isn't the point's (4), which sizes its timed sessions and cuts its window")
    assert SP.point_why(silent, files, spec, item) is None, "a row timed outside the harness is judged as it declares"
