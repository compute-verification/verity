"""A steady-state spec's points on a made-up tree, a store with a directory remote and stand-ins for the node: the schema
loaded strictly, what a point's identity binds and what it leaves out, the sessions a point times where a row's sessions
file in bursts, which run `plan` counts as a point's passing run and why it counts no other, which labels it pulls,
the command `run --direct` launches, and the node side from its refusals to its verdict and to a signal that ends it.
No GPU, no node and no clock."""

from __future__ import annotations

import hashlib
import json
import re
import signal
import sys
from pathlib import Path

import pytest
from test_steady import DROP, END, SESSIONS, START, lease, made
from verity_config import ConfigError

from verity_bench import jobs, kit, served_table, settings
from verity_bench import prove_records as R
from verity_bench import steady as S
from verity_bench import steady_node as N
from verity_bench import steady_plan as P
from verity_bench import steady_spec as SP
from verity_bench.cli import main
from verity_bench.tool import STEADY_POINT

CIRCUIT = "the circuit of gemm64\n"
PIN = hashlib.sha512(CIRCUIT.encode()).hexdigest()
DRIVER, CUDA = "570.86.15", "12.8.93"
SETUP = "benchmarks/harness/specs/steady-setup.sh"
RECORD = "prove-glm53-flock-steady"
#: a made-up artifact of the store, and the directory in it that holds a staged statement
ART, IN_ART = "art:" + "5" * 64, "out/row1"
#: the made-up run of `test_steady.made`: two provers of `gemm64` on GPU 3, in the run `r-own`
POINT, RUN = "gemm64-k2", "r-own"
#: what a row of a run of the point holds beyond the script's own fields
OWN = {"pin": PIN, "cores": {"count": 10, "list": "0-9", "numa_nodes": [0]},
       SP.VERSIONS: {"driver": DRIVER, "cuda_toolkit": CUDA}}
#: the made-up tree's files beside its spec: the prover's source, its tests and its document, a build script, the timing
#: script and the spec's setup
FILES = {
    "uv.lock": "lock\n", "pyproject.toml": "",
    "verity/protocols/flock/prover/src/main.rs": "fn main() {}\n",
    "verity/protocols/flock/prover/cuda/ntt.cu": "// kernel\n",
    "verity/protocols/flock/prover/tests/test_prover.py": "def test(): pass\n",
    "verity/protocols/flock/prover/README.md": "the prover\n",
    "verity/protocols/flock/pod/70-class-sweep.sh": "FLAGS=-O3\n",
    "benchmarks/one_stage/server_rate.py": "# times\n",
    SETUP: "export VERITY_BENCH_STEADY_BIN=/built/prover\n",
}
SPEC = {
    "schema": SP.SCHEMA, "id": "steady", "workload": "glm53", "card": "h200", "cores": 10, "sessions": 16, "ahead": 2,
    "driver": DRIVER, "cuda_toolkit": CUDA, "setup": SETUP,
    "env": {"RUST_MIN_STACK": "1073741824"},
    "admission": {"max_minutes": 90, "mem_gb": 80},
    "row": [{"type": "gemm64", "n": 1, "pin": PIN, "provers": [1, 2], "warm": 2},
            {"type": "head-block", "n": 4, "pin": "ab" * 64, "provers": [2], "warm": 2, "staged": "/dev/shm/head"}],
}


def toml(doc: dict) -> str:
    def value(v):
        return json.dumps(v) if not isinstance(v, bool) else str(v).lower()

    top = {k: v for k, v in doc.items() if not isinstance(v, dict) and k != "row"}
    lines = [f"{k} = {value(v)}" for k, v in top.items()]
    for name, table in doc.items():
        if isinstance(table, dict):
            lines += ["", f"[{name}]", *(f"{k} = {value(v)}" for k, v in table.items())]
    for row in doc.get("row", ()):
        lines += ["", "[[row]]", *(f"{k} = {value(v)}" for k, v in row.items())]
    return "\n".join(lines) + "\n"


def tree(tmp_path: Path, spec: dict | None = None, files: dict | None = None, name: str = "tree") -> tuple[Path, str]:
    """A made-up source tree and its spec's path in it. `spec`: top-level keys replaced (`row0`: keys of the first row;
    a value of None leaves the key out); `files`: a file's text replaced, added, or left out (None)."""

    doc = json.loads(json.dumps(SPEC))
    for key, change in (spec or {}).items():
        if key == "row0":
            doc["row"][0] = {k: v for k, v in (doc["row"][0] | change).items() if v is not None}
        elif change is None:
            del doc[key]
        else:
            doc[key] = change
    root = tmp_path / name
    for path, text in ({**FILES, "benchmarks/harness/specs/steady.toml": toml(doc)} | (files or {})).items():
        if text is not None:
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text)
    return root, "benchmarks/harness/specs/steady.toml"


def git(root: Path) -> None:
    import subprocess

    subprocess.run("git init -q && git add . && GIT_AUTHOR_DATE='@1700000000 +0000' GIT_COMMITTER_DATE='@1700000000 +0000' "
                   "git -c user.name=t -c user.email=t@example.com -c commit.gpgsign=false commit -qm spec",
                   shell=True, cwd=root, check=True)


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
    ({"spec": {"card": "h100"}}, r"`card` 'h100' is no GPU of verity_config's registry \(rtx-pro-6000, h200, b200\)"),
    ({"spec": {"cores": 0}}, "`cores` is 0; it is a count of at least 1"),
    ({"spec": {"ahead": -1}}, "`ahead` is -1"),
    ({"spec": {"driver": "v570"}}, "`driver` 'v570' isn't a version"),
    ({"spec": {"cuda_toolkit": ""}}, "`cuda_toolkit` '' isn't a version"),
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
    ({"spec": {"row0": {"type": "head block"}}}, "two rows are named head_block in the rate file"),
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
    "the prover's source": {"files": {"verity/protocols/flock/prover/src/main.rs": "fn main() { loop {} }\n"}},
    "a kernel": {"files": {"verity/protocols/flock/prover/cuda/ntt.cu": "// another kernel\n"}},
    "a new source file": {"files": {"verity/protocols/flock/prover/src/extra.rs": "\n"}},
    "the build's flags": {"files": {"verity/protocols/flock/pod/70-class-sweep.sh": "FLAGS=-O2\n"}},
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
    "the prover's tests": {"files": {"verity/protocols/flock/prover/tests/test_prover.py": "def test(): assert 1\n"}},
    "a document": {"files": {"verity/protocols/flock/prover/README.md": "the prover, again\n"}},
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
    assert set(covered) == {"benchmarks/one_stage/server_rate.py", SETUP, "verity/protocols/flock/pod/70-class-sweep.sh",
                            "verity/protocols/flock/prover/cuda/ntt.cu", "verity/protocols/flock/prover/src/main.rs"}
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
    for left in ("verity/protocols/flock/prover/target/release/flock-circuit", "verity/protocols/flock/prover/src/draft.rs",
                 "verity/protocols/flock/pod/notes.sh"):
        (root / left).parent.mkdir(parents=True, exist_ok=True)
        (root / left).write_text("left by a build\n")
    assert identities(tmp_path, "checkout") == shipped, "untracked"
    assert identities(tmp_path, "checkout", files={"verity/protocols/flock/prover/src/main.rs": "fn main() { loop {} }\n"}
                      )[POINT] != shipped[POINT], "a tracked file as the working tree has it"
    subprocess.run(["git", "add", "verity/protocols/flock/prover/src/draft.rs"], cwd=root, check=True)
    assert identities(tmp_path, "checkout", files={"verity/protocols/flock/prover/src/main.rs": FILES[
        "verity/protocols/flock/prover/src/main.rs"]})[POINT] != shipped[POINT], "a newly tracked source"


@pytest.mark.parametrize("gone", ["benchmarks/one_stage/server_rate.py", "verity/protocols/flock/pod/70-class-sweep.sh"])
def test_a_tree_without_a_path_the_identity_covers_is_refused(tmp_path, gone):
    root, spec = tree(tmp_path, files={gone: None})
    with pytest.raises(ValueError, match=f"holds no {re.escape(gone.rsplit('/', 1)[0] if 'pod' in gone else gone)}"):
        P.make_plan(root / spec, root=root, lookup=lambda spec, item, identity: (None, None))


def test_a_spec_of_a_workload_without_a_steady_state_record_is_refused(tmp_path):
    root, spec = tree(tmp_path, spec={"workload": "llama"})
    with pytest.raises(ValueError, match="the `workload` of spec 'steady', 'llama', has no steady-state record: the workloads are glm53"):
        P.make_plan(root / spec, root=root, lookup=lambda spec, item, identity: (None, None))


# ---- the timed sessions ----------------------------------------------------------------------------------------------------

#: a prover's filings where one session in four waits: seven sessions of two provers a second apart, then one after nine
BURSTY = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0)


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


# ---- a point's passing run -------------------------------------------------------------------------------------------------


def run_tree(directory: Path, changes: dict | None = None, **texts) -> Path:
    """The files of `test_steady.made`'s run as a run keeps them, its row a run of the point's (`OWN`) with `changes`."""

    _, files = made(OWN | (changes or {}), **texts)
    for path in (S.RATES, S.LEASES, START, END, SESSIONS):
        if (text := files(path)) is not None:
            (directory / path).parent.mkdir(parents=True, exist_ok=True)
            (directory / path).write_text(text)
    return directory


class Store:
    """A machine's store with a directory remote, each run's attempt and files on the remote."""

    def __init__(self, tmp: Path, root: Path):
        from research.store.local import LocalStore

        self.tmp, self.root, self.remote = tmp, root, tmp / "remote"
        (self.remote / "attempts").mkdir(parents=True)
        self.config = tmp / "store.toml"
        self.config.write_text(f'[remote]\ntype = "fs"\npath = "{self.remote}"\n')
        self.store = LocalStore(tmp / "producer", config_path=self.config)
        self._n = 0

    def fresh(self):
        from research.store.local import LocalStore

        self._n += 1
        return LocalStore(self.tmp / f"fresh{self._n}", config_path=self.config)

    def files(self, changes: dict | None = None, **texts) -> Path:
        self._n += 1
        return run_tree(self.tmp / "runs" / str(self._n), changes, **texts)

    def art(self, kind: str, meta: dict, directory: Path) -> str:
        art = self.store.put_artifact(kind, meta, tree=directory)
        self.store.push(art)
        return art

    def ran(self, run: str, files: Path | None, *, identity: str | None = None, rc: int = 0, dirty: bool = False) -> str:
        """A run's attempt on the remote: a run of the point `identity` as `research run` records it, or (None) a run
        from before the harness; its `files` a run record (None: it kept none)."""

        from research.store import canon, derivation

        tool = STEADY_POINT.identity() if identity else {"name": "bash", "version": None}
        params = {"identity": identity} if identity else {}
        drv = derivation.derivation_id(tool, {}, params, {}, []) if identity else "drv:" + "0" * 64
        attempt = {
            "schema": "research/attempt/v0.1", "id": run, "tool": tool, "derivation": drv, "reuse_key": None, "argv": [],
            "params": params, "key_params": sorted(params), "inputs": {}, "conditions": {}, "key_conditions": [],
            "source": {"commit": "c" * 40, "tree": "d" * 40, "dirty": dirty}, "closure": None,
            "execution": {"start_utc": f"{run[1:5]}-{run[5:7]}-{run[7:9]}T{run[10:12]}:{run[12:14]}:{run[14:16]}Z"},
            "cost": None, "status": {"state": "done" if rc == 0 else "failed", "rc": rc,
                                     "validation": "passed" if rc == 0 else "failed"},
            "outputs": {"result": None, "run_record": self.art("run-record/v1", {}, files) if files else None},
            "telemetry": None,
        }
        (self.remote / "attempts" / f"{run}.json").write_bytes(canon.dumps(attempt))
        return run

    def label(self, target: str, key: str, value: str) -> None:
        self._n += 1
        self.store.add_label(target, key, value, by="benchmarks", ts=f"2026-01-01T00:00:{self._n:02d}Z")
        self.store.push_labels([target])

    def lookup(self, **options) -> P.SteadyCompletion:
        """The lookup of a machine that has read nothing yet, pulling the remote's attempts since the made-up day; what
        it says while it pulls is in `said`."""

        self.said: list[str] = []
        return P.SteadyCompletion(self.fresh(), root=self.root, since=lambda: 1_700_000_000, log=self.said.append,
                                  **options)


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
     "`accepted` (changed), `exits.prover-gpu3-0` (dropped), `exits.prover-gpu3-1` (dropped), `exits.verifier-ab-0` "
     "(dropped) and 4 more"),
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
    stage = {"definition": "Gemm_v1{K=64}", "k_log": 21, "blocks": 8}

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
    art = completed(store, run, files, {"stage": {"definition": "Gemm_v1{K=64}", "k_log": 20, "blocks": 8, "full": 8}})
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


# ---- plan, check and run ---------------------------------------------------------------------------------------------------


def test_plan_and_check_take_a_steady_spec_and_say_why_a_point_has_no_passing_run(tmp_path, monkeypatch, capsys):
    root, path = tree(tmp_path)
    git(root)
    store = Store(tmp_path, root)
    points = P.make_plan(root / path, root=root, lookup=lambda spec, item, identity: (None, None)).points
    run = store.ran("r20231115-100000-0001", store.files(), identity=points[1].identity)
    failed = store.ran("r20231115-110000-0002", store.files({"type": "head_block", "pin": "ab" * 64, "flat": False}),
                       identity=points[2].identity, rc=SP.NOT_PASSED)
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
        assert flags[flags.index("--timeout") + 1] == "90m" and "--require-result" in flags
        assert flags[flags.index("--tool") + 1] == "verity_bench.tool:STEADY_POINT"
        assert [flags[i + 1] for i, token in enumerate(flags) if token == "--env"] == [
            "RUST_MIN_STACK=1073741824", f"VERITY_BENCH_SETUP={SETUP}"]
        assert through_slot(workload) and "gpu-lease" not in command
        assert workload[:4] == ["env", f"CHECK_SLOTS={jobs.GPU_SLOTS}", "python3", "tools/verity/check/slot.py"]
        assert workload[workload.index("--run-max") + 1] == "5400"
        assert workload[workload.index("--", 4) + 1:] == [
            "bash", "benchmarks/harness/steady_point.sh", "--spec", path, "--point", item.point.id,
            "--identity", item.identity]
        assert STEADY_POINT.parse(command) == {"spec": path, "point": item.point.id, "identity": item.identity}


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


# ---- the node side ---------------------------------------------------------------------------------------------------------

NVCC = ("nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2025 NVIDIA Corporation\n"
        f"Built on a day\nCuda compilation tools, release 12.8, V{CUDA}\nBuild cuda_12.8.r12.8/compiler.0_0\n")
#: the kit's host stamp inside the lease: the leased card, with the driver's version and the newest CUDA it serves,
#: which is another number than the toolkit's
STAMP = {"card": "NVIDIA H200", "gpus": [{"uuid": "GPU-3", "sm_mhz": 1980, "mem_mhz": 3201}], "driver": DRIVER,
         "cuda": "12.9", "node": "the-node", "commit": "c" * 40}
#: what the node answers: the toolkit, the leases (the run's own, and another run's on GPU 0) and the compute processes
#: (one, on GPU 0)
ANSWERS = {
    "nvcc --version": NVCC,
    "gpu-lease status": (f"3 held 1m pid=7 run={RUN} who=benchmarks cmd=python -m verity_bench.steady_node\n"
                         "0 held 5m pid=11 run=r-other who=other cmd=python train.py\n1 free\n"),
    "nvidia-smi --query-gpu=index,uuid --format=csv,noheader": "0, GPU-0\n3, GPU-3\n",
    "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader": "GPU-0, 12, 100 MiB\n",
}


class Node:
    """A stand-in for the node under one run of the point: the stager, gpu-lease, the timing script and what the node
    answers. `row`: fields of the row the script writes; `texts`: the run's files as `test_steady.made` takes them;
    `answers`: an answer replaced (None: the command fails); `stamp`: fields of the host stamp replaced; `circuit`,
    `said` (its pin.txt) and `units`: what the stager stages."""

    def __init__(self, tmp_path: Path, monkeypatch, *, row: dict | None = None, answers: dict | None = None,
                 circuit: str = CIRCUIT, said: str = PIN, units: int = 1, cpus: int = 10, lease_exit: int | None = None,
                 spec: dict | None = None, stamp: dict | None = None, **texts):
        self.root, self.spec = tree(tmp_path, spec=spec)
        self.host = STAMP | (stamp or {})
        self.run_dir = tmp_path / "run"
        self.run_dir.mkdir()
        (scratch := tmp_path / "scratch").mkdir()
        self.scratch_root = scratch
        (binary := tmp_path / "prover").write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        self.binary = binary
        for name, value in {"RESEARCH_RUN_DIR": str(self.run_dir), "RESEARCH_RUN_ID": RUN,
                            "VERITY_BENCH_STEADY_BIN": str(binary), "VERITY_BENCH_STEADY_SCRATCH": str(scratch),
                            "VERITY_BENCH_SETUP": SETUP, "RUST_MIN_STACK": "1073741824"}.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.chdir(self.root)
        self.monkeypatch = monkeypatch
        #: what the node lists of its NUMA nodes and its PCI devices (`numa`): nothing, unless a test says
        self.sys = tmp_path / "sys"
        monkeypatch.setattr(N, "NODES", self.sys / "node")
        monkeypatch.setattr(N, "DEVICES", self.sys / "pci")
        self.row, self.files = made(OWN | {SP.VERSIONS: DROP} | (row or {}), **texts)
        self.answers = ANSWERS | (answers or {})
        self.circuit, self.said, self.units, self.cpus, self.lease_exit = circuit, said, units, cpus, lease_exit
        self.ran: list[list[str]] = []
        #: the store's stand-in (`stored`): the files a fetch gives, what its listing answers, the fetch's exit code, the
        #: listings asked, and what the timed step saw of the scratch
        self.fetch, self.listed, self.fetch_exit, self.asked, self.seen = {}, None, 0, [], None
        spec = SP.load_spec(self.root / self.spec, root=self.root)
        self.identity = SP.point_identity(spec, SP.point(spec, POINT), sha256_of(SP.closure(self.root, spec)))

    def args(self, **changes):
        return N.parse(["--spec", self.spec, "--point", POINT, "--identity", changes.get("identity", self.identity)])

    def ask(self, argv):
        if list(argv[1:5]) == list(N.FETCH):
            self.asked.append(list(argv))
            return (0, self.listed) if self.listed else (1, "no remote configured")
        answer = self.answers.get(" ".join(argv))
        return (1, "") if answer is None else (0, answer)

    def run(self, argv, log: Path) -> int:
        argv = list(argv)
        self.ran.append(argv)
        log.write_text("log\n")
        if argv[1:2] == [SP.STAGER]:
            stage = Path(argv[4])
            stage.mkdir(parents=True)
            (stage / "circuit.txt").write_text(self.circuit)
            (stage / "pin.txt").write_text(self.said + "\n")
            (stage / "stage.json").write_text(json.dumps({"n": self.units, "definition": "Gemm_v1{K=64}", "k_log": 20}))
            return 0
        if argv[1:5] == list(N.FETCH):
            store, to = (Path(argv[argv.index(flag) + 1]) for flag in ("--store", "--to"))
            (store / "objects").mkdir(parents=True)
            (store / "objects" / "blob").write_text("the fetch's own copy\n")
            (to / IN_ART).mkdir(parents=True)
            for name, text in self.fetch.items():
                if text is not None:
                    (to / IN_ART / name).write_text(text)
            return self.fetch_exit
        if argv[0] == "gpu-lease":
            if self.lease_exit is not None:
                return self.lease_exit
            inside = argv[argv.index("--") + 1:]
            assert inside[:3] == [sys.executable, "-m", "verity_bench.steady_node"]
            self.monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
            rc = N.point(N.parse(inside[3:]), run=self.run, ask=self.ask, now=lambda: 1_700_000_000.0,
                         stamp=lambda: dict(self.host))
            self.monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
            if (usage := self.files(S.LEASES)) is not None:
                (self.run_dir / S.LEASES).write_text(usage)
            return rc
        assert argv[1] == SP.TIMER, argv
        out = Path(argv[argv.index("--out") + 1])
        (step := out / f"gemm64-k{argv[argv.index('--provers') + 1]}").mkdir(parents=True)
        self.timed = argv
        [stage] = Path(argv[argv.index("--staged") + 1]).glob("*/c*/stage")
        self.seen = {"store": (out.parent / "store").exists(), "files": sorted(path.name for path in stage.iterdir())}
        (out / "rate.json").write_text(json.dumps([self.row]))
        if (sessions := self.files(SESSIONS)) is not None:
            (step / "sessions.jsonl").write_text(sessions)
        (step / "prover-gpu3-0.log").write_text("x" * (N.KEPT_LOG_BYTES + 10))
        return 0

    def point(self, **changes) -> int:
        return N.point(self.args(**changes), run=self.run, ask=self.ask, now=lambda: 1_700_000_000.0,
                       cpus=lambda: list(range(20, 20 + self.cpus)), stamp=self.no_stamp)

    def no_stamp(self):
        raise AssertionError("the host is stamped inside the lease, where the stamp is the leased card's")

    def read(self, path: str):
        return json.loads((self.run_dir / path).read_text())


def sha256_of(doc) -> str:
    from verity_config import sha256

    return sha256(doc)


def test_the_node_side_times_the_point_under_one_lease_and_computes_its_verdict_after_it(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    assert node.point() == 0

    stager, lease_argv, timer = node.ran
    scratch = Path(lease_argv[-1])
    assert scratch.parent == node.scratch_root and not scratch.exists(), "the scratch is removed"
    assert stager == [sys.executable, SP.STAGER, "gemm64", "1", str(scratch / "z" / "gemm64" / "c1" / "stage"), "--unbound"]
    assert kit.lease(1, 90) == ["gpu-lease", "1", "--wait", "--max-min", "90", "--"], "the kit's lease, and beside it:"
    assert lease_argv == ["gpu-lease", "1", "--wait", "--max-min", "90", "--preemptible", "--mem-gb", "80", "--",
                          sys.executable, "-m", "verity_bench.steady_node", "--spec", node.spec, "--point", POINT,
                          "--identity", node.identity, "--inside", str(scratch)]
    assert timer == [sys.executable, SP.TIMER, "--bin", str(node.binary), "--staged", str(scratch / "z"),
                     "--types", "gemm64", "--provers", "2", "--out", str(scratch / "rate"), "--device", "3",
                     "--sessions", "16", "--warm", "2", "--ahead", "2"]

    [row] = node.read(S.RATES)
    assert row[SP.VERSIONS] == {"driver": DRIVER, "cuda_toolkit": CUDA}, "read on the node and written in the row"
    assert row == node.row | {SP.VERSIONS: row[SP.VERSIONS]}
    start = (node.run_dir / START).read_text()
    assert start == (node.run_dir / END).read_text() == (
        "epoch 1700000000\nlease 0 held 5m pid=11 run=r-other who=other\napp GPU 0 pid 12 100MiB\n")
    kept = sorted(p.relative_to(node.run_dir).as_posix() for p in (node.run_dir / "out" / "rate").rglob("*") if p.is_file())
    assert kept == ["out/rate/gemm64-k2/prover-gpu3-0.log", SESSIONS, S.RATES, "out/rate/server-rate.log"]
    assert (node.run_dir / "out/rate/gemm64-k2/prover-gpu3-0.log").stat().st_size == N.KEPT_LOG_BYTES

    point = node.read(N.POINT_FILE)
    assert point["identity"] == node.identity and point["point"]["provers"] == 2 and "refused" not in point
    assert point["prover"] == {"path": str(node.binary), "sha256": hashlib.sha256(b"#!/bin/sh\n").hexdigest()}
    assert point["cores"] == {"count": 10, "list": "20-29"}
    assert point["statement"]["pin"] == PIN and point["statement"]["from"] == SP.STAGER
    assert point["timed"] == {"refused": None, "device": "3", "host": STAMP,
                              SP.VERSIONS: {"driver": DRIVER, "cuda_toolkit": CUDA}, "exit": 0, "rows": 1}
    assert STAMP["cuda"] != CUDA, "the stamp's `cuda` is the driver's newest, never the toolkit's version"
    verdict = node.read(N.VERDICT_FILE)
    assert (verdict["identity"], verdict["point"], verdict["passed"], verdict["why"]) == (node.identity, POINT, True, None)
    assert list(verdict["verdict"]["checks"]) == list(S.CHECKS)
    result = node.read("result.json")
    assert result["kind"] == N.KIND and result["validation"]["status"] == "passed" and result["run_id"] == RUN
    assert result["workload_fingerprint"]["identity"] == node.identity
    assert result["schema"] == kit.RESULT_SCHEMA
    assert result["measurements"] == [{
        "name": "statements_per_s", "value": row["statements_per_s"], "unit": "1/s", "commit": STAMP["commit"],
        "run": RUN, "ci95": None, "target": None, "verdict": None, "inputs": [], "host": STAMP}], "the kit's writer's"

    spec = SP.load_spec(node.root / node.spec, root=node.root)
    assert SP.point_why(row, S.files_in(node.run_dir), spec, SP.point(spec, POINT), own=True) is None


#: a made-up node of two NUMA nodes, two cards on each: the run's CPUs (20 to 29) are on the second
CARDS = "0, 00000000:17:00.0\n1, 00000000:65:00.0\n2, 00000000:B1:00.0\n3, 00000000:CA:00.0\n"
NUMA = {"node/node0/cpulist": "0-15\n", "node/node1/cpulist": "16-31\n", "node/node1/meminfo": "",
        "pci/0000:17:00.0/numa_node": "0\n", "pci/0000:65:00.0/numa_node": "0\n",
        "pci/0000:b1:00.0/numa_node": "1\n", "pci/0000:ca:00.0/numa_node": "1\n"}


def numa(node: Node, cards: str | None = CARDS, **changes: str | None) -> None:
    """What the node says of its NUMA nodes and its cards: `NUMA`, with files replaced or left out (None), their
    names' `/`, `:` and `.` written `_`."""

    node.answers = node.answers | {" ".join(N.CARDS): cards}
    by_key = {re.sub(r"[/:.]", "_", path): path for path in NUMA}
    for path, text in (NUMA | {by_key[key]: text for key, text in changes.items()}).items():
        if text is not None:
            (node.sys / path).parent.mkdir(parents=True, exist_ok=True)
            (node.sys / path).write_text(text)


def test_the_lease_is_asked_of_the_cards_on_the_numa_node_of_the_runs_cpus(tmp_path, monkeypatch):
    """A slot gives the run its CPUs and no card: without this the lease takes the first free card of the node."""

    node = Node(tmp_path, monkeypatch)
    numa(node)
    assert node.point() == 0
    lease = node.ran[1]
    assert lease[:lease.index("--")] == ["gpu-lease", "1", "--wait", "--max-min", "90", "--on", "2,3", "--preemptible",
                                         "--mem-gb", "80"]
    assert node.read(N.POINT_FILE)["lease_on"] == {"numa_nodes": [1], "cards": [2, 3]}


def test_a_node_that_cannot_run_gpu_lease_is_refused_before_a_statement_is_staged_with_the_script_to_install(
        tmp_path, monkeypatch):
    """The kit's lease calls `gpu-lease` by its bare name. A node without it on its path would stage or fetch the
    statement and then fail to take the lease; the point says so first, with what to put on the path."""
    node = Node(tmp_path, monkeypatch)
    answered = node.ask
    node.ask = lambda argv: ((N.NOT_RUN, "FileNotFoundError: [Errno 2] No such file or directory: 'gpu-lease'")
                             if list(argv) == ["gpu-lease", "status"] else answered(argv))

    assert node.point() == N.REFUSED
    assert node.ran == [], "nothing is staged, fetched or leased"
    refused = node.read(N.POINT_FILE)["refused"]
    assert refused == ("`gpu-lease` can't be run on this node (FileNotFoundError: [Errno 2] No such file or directory: "
                       "'gpu-lease'), and the timed step runs under its lease of one card: put "
                       "tools/research/src/research/pods/sh/gpu_lease.sh of this tree on the node's path as "
                       "`gpu-lease`. A copy on the node's own disk goes when the node is reset")
    assert node.read(kit.RESULT)["validation"] == {
        "status": "failed", "checker": "verity_bench.steady_spec.point_why",
        "detail": f"refused, nothing timed: {refused}", "evidence": [N.POINT_FILE]}
    assert not list(node.scratch_root.iterdir()), "no scratch was made"


def test_a_command_the_node_cannot_run_is_asked_with_the_exit_code_a_shell_gives(tmp_path):
    rc, text = N._ask([str(tmp_path / "no-such-program")])
    assert rc == N.NOT_RUN == 127 and text.startswith("FileNotFoundError: ")


@pytest.mark.parametrize("answer", [(0, "0 free\n"), (1, ""), (2, "gpu-lease: no state directory")])
def test_a_gpu_lease_that_runs_is_not_refused_at_the_start_whatever_its_status_says(answer):
    """Only a tool that can't be run at all is refused there: what a lease that can't be taken says is the lease's own."""
    N.lease_tool(lambda argv: answer)


#: a node that doesn't say which cards are on the NUMA node of the run's CPUs: what it says instead, and why the lease
#: is asked of any card
ANY_CARD = [
    ({"node_node0_cpulist": "0-24\n", "node_node1_cpulist": "25-31\n"}, [0, 1], "the node lists 2 NUMA nodes for the run's CPUs"),
    ({"node_node0_cpulist": None, "node_node1_cpulist": None}, [], "the node lists 0 NUMA nodes for the run's CPUs"),
    ({"cards": None}, [1], "`nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader` exited 1"),
    ({"pci_0000_b1_00_0_numa_node": "-1\n", "pci_0000_ca_00_0_numa_node": None}, [1],
     "none of the node's 3 cards says it is on NUMA node 1"),
]


@pytest.mark.parametrize("says, nodes, why", ANY_CARD, ids=[why for *_, why in ANY_CARD])
def test_a_node_that_does_not_say_which_cards_are_near_the_runs_cpus_leases_any_card_and_the_point_says_why(
        tmp_path, monkeypatch, says, nodes, why):
    node = Node(tmp_path, monkeypatch)
    numa(node, **({"cards": CARDS} | says))
    assert node.point() == 0
    assert "--on" not in node.ran[1]
    assert node.read(N.POINT_FILE)["lease_on"] == {"numa_nodes": nodes, "cards": None, "why": why}


def test_a_node_that_lists_nothing_of_its_numa_nodes_leases_any_card(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    assert node.point() == 0 and "--on" not in node.ran[1]
    assert node.read(N.POINT_FILE)["lease_on"]["cards"] is None


def test_the_result_lists_every_file_plan_judges_the_run_by(tmp_path, monkeypatch):
    """The store keeps the files a run's result lists (`artifacts` and `measurement_files`) and no other, so a verdict
    recomputed from the store reads only those: a result that listed only its records left a passed run "cannot tell"."""

    import shutil

    node = Node(tmp_path, monkeypatch)
    assert node.point() == 0
    result = node.read("result.json")
    listed = sorted({*result["artifacts"], *result["measurement_files"]})
    assert {S.RATES, S.LEASES, START, END, SESSIONS, N.VERDICT_FILE, N.POINT_FILE, N.TIMED_FILE} <= set(listed)
    assert all((node.run_dir / path).is_file() for path in listed)
    for path in listed:
        (tmp_path / "stored" / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(node.run_dir / path, tmp_path / "stored" / path)
    spec = SP.load_spec(node.root / node.spec, root=node.root)
    item, files = SP.point(spec, POINT), S.files_in(tmp_path / "stored")
    assert SP.point_why(SP.own_row(files, item), files, spec, item, own=True) is None


def test_the_node_side_times_a_row_with_a_period_at_the_points_count_and_records_the_period(tmp_path, monkeypatch):
    period = {"row0": {"period": 4}}
    node = Node(tmp_path, monkeypatch, spec=period, gaps=(BURSTY * 4)[:27])
    assert node.point() == 0
    assert node.timed[node.timed.index("--sessions") + 1] == "28", "not the spec's 16"
    [row] = node.read(S.RATES)
    assert row[S.PERIOD] == 4 and row["sessions"] == 28, "the field `level` reads"
    assert node.read(N.POINT_FILE)["point"] | {"period": 4, "sessions": 28} == node.read(N.POINT_FILE)["point"]

    short = Node(tmp_path / "short", monkeypatch, spec=period)
    assert short.point() == SP.NOT_PASSED, "a script that timed another count than the point's"
    assert short.read(N.VERDICT_FILE)["why"].startswith("fail: it timed 16 sessions after 2 warm ones a prover; the "
                                                        "point's are 28 and 2")
    plain = Node(tmp_path / "plain", monkeypatch)
    assert plain.point() == 0 and S.PERIOD not in plain.read(S.RATES)[0], "a row without a period records none"


def test_a_statement_already_on_the_node_is_proved_from_a_copy_checked_against_the_pin(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    (staged := tmp_path / "staged").mkdir()
    for name, text in {"circuit.txt": CIRCUIT, "pin.txt": PIN, "stage.json": json.dumps({"n": 1})}.items():
        (staged / name).write_text(text)
    path = node.root / node.spec
    path.write_text(path.read_text().replace("warm = 2\n", f'warm = 2\nstaged = "{staged}"\n', 1))
    assert node.point() == 0, "`staged` is no part of the identity"
    assert [argv[0] for argv in node.ran] == ["gpu-lease", sys.executable] and node.ran[1][1] == SP.TIMER
    assert node.read(N.POINT_FILE)["statement"]["from"] == str(staged)

    (staged / "circuit.txt").write_text("another circuit\n")
    node.ran.clear()
    assert node.point() == N.REFUSED and node.ran == []
    assert "another circuit takes no GPU time" in node.read("result.json")["validation"]["detail"]


def stored(node: Node, listed: str | None = None, **files) -> None:
    """The first row's `staged` names the store, and the node's stand-in fetches `files` for it (the staged statement's,
    unless replaced or left out with None); `listed`: what the store's listing answers instead."""

    path = node.root / node.spec
    path.write_text(path.read_text().replace("warm = 2\n", f'warm = 2\nstaged = "{ART}/{IN_ART}"\n', 1))
    node.fetch = {"circuit.txt": CIRCUIT, "pin.txt": PIN, "stage.json": json.dumps({"n": 1}), "inst.bin": "inst"} | files
    size = sum(len(text) for text in node.fetch.values() if text is not None)
    node.listed = json.dumps([{"path": f"{IN_ART}/x", "bytes": size}]) if listed is None else listed
    # wherever the suite keeps its temporary files, the stand-in's scratch counts as disk
    node.monkeypatch.setattr(N, "MEMORY", node.root / "memory")


def test_a_statement_in_the_store_is_fetched_into_the_scratch_and_leaves_no_copy_on_the_node(tmp_path, monkeypatch):
    """The fetch reads the artifact's one directory through a store of its own under the scratch, which is removed
    before the statement is hashed; the statement is then held to the pin like any other, and goes with the scratch."""

    node = Node(tmp_path, monkeypatch)
    stored(node)
    assert node.point() == 0, "`staged` is no part of the identity"
    fetch, lease_argv, timer = node.ran
    scratch = Path(lease_argv[-1])
    assert fetch == [sys.executable, "-m", "research", "data", "fetch", ART, "--store", str(scratch / "store"),
                     "--path", IN_ART, "--to", str(scratch / "fetched")]
    assert node.asked[0] == [*fetch[:-2], "--list", "--json"], "the listing is asked of the same store, before the fetch"
    assert node.seen == {"store": False, "files": ["circuit.txt", "inst.bin", "pin.txt", "stage.json"]}, (
        "at the timed step the fetch's store is gone and the statement is where the timing script looks for it")
    assert not scratch.exists() and list(node.scratch_root.iterdir()) == []
    statement = node.read(N.POINT_FILE)["statement"]
    assert (statement["from"], statement["pin"], statement["n"]) == (f"{ART}/{IN_ART}", PIN, 1)
    assert "out/fetch.log" in node.read("result.json")["artifacts"]


#: a statement in the store that is refused: how the stand-in differs, and what the refusal says
NOT_FETCHED = [
    ({"listed": ""}, "the store lists no file of the `staged` of row 'gemm64'.*`research run --store-read`"),
    ({"listed": "[]"}, "the store lists no file of the `staged` of row 'gemm64'"),
    ({"listed": "not a listing"}, r"the store's listing of the `staged` of row 'gemm64'.*bytes \(JSONDecodeError"),
    ({"listed": json.dumps([{"path": "out/row1/x"}])}, r"isn't each file with its bytes \(KeyError"),
    ({"listed": json.dumps([{"path": "out/row1/x", "bytes": 1 << 62}])}, "held twice while it is fetched, and the disk"),
    ({"fetch_exit": 1}, r"fetching the `staged` of row 'gemm64', art:5+/out/row1, failed \(exit 1\): out/fetch.log"),
    ({"pin.txt": None}, "holds no staged statement in the store"),
    ({"circuit.txt": "another circuit\n"}, "another circuit takes no GPU time"),
    ({"stage.json": json.dumps({"n": 2})}, "the statement holds 2 units"),
]


@pytest.mark.parametrize("change, says", NOT_FETCHED, ids=[says[:40] for _, says in NOT_FETCHED])
def test_a_statement_the_store_does_not_give_whole_is_refused_before_any_card(tmp_path, monkeypatch, change, says):
    node = Node(tmp_path, monkeypatch)
    node.fetch_exit = change.get("fetch_exit", 0)
    stored(node, **{key: value for key, value in change.items() if key != "fetch_exit"})
    assert node.point() == N.REFUSED and not any(argv[0] == "gpu-lease" for argv in node.ran)
    assert re.search(says, node.read("result.json")["validation"]["detail"])
    assert list(node.scratch_root.iterdir()) == [], "what was fetched goes with the scratch"


def test_a_statement_from_the_store_is_never_fetched_into_the_nodes_memory(tmp_path, monkeypatch):
    """The scratch's default is the node's memory, where a statement staged in the run may be: one from the store is
    refused there before the store is asked anything."""

    assert settings.REGISTRY.setting("VERITY_BENCH_STEADY_SCRATCH").default == str(N.MEMORY)
    node = Node(tmp_path, monkeypatch)
    stored(node)
    monkeypatch.setattr(N, "MEMORY", node.scratch_root.resolve())
    assert node.point() == N.REFUSED and node.ran == [] and node.asked == []
    assert re.search("is fetched from the store onto the node's disk, and the scratch .* is in its memory: the spec's "
                     "setup exports VERITY_BENCH_STEADY_SCRATCH", node.read("result.json")["validation"]["detail"])
    assert list(node.scratch_root.iterdir()) == []


#: the node side refusing before any card is asked for: the stand-in's change and what the refusal says
BEFORE_THE_CARD = [
    ({"cpus": 16}, r"the run has 16 CPUs \(20-35\), and the `cores` of spec 'steady' is 10"),
    ({"circuit": "another circuit\n"}, "the statement's circuit has SHA-512 .*, and the `pin` of row 'gemm64' is"),
    ({"said": "ab" * 64}, r"the statement's pin.txt, which the timing script writes in the row, says abababababababab\.\.\."),
    ({"units": 2}, "the statement holds 2 units, and the `n` of row 'gemm64' is 1"),
]


@pytest.mark.parametrize("change, says", BEFORE_THE_CARD, ids=[says[:30] for _, says in BEFORE_THE_CARD])
def test_the_node_side_refuses_another_point_before_any_card_is_asked_for(tmp_path, monkeypatch, change, says):
    node = Node(tmp_path, monkeypatch, **change)
    assert node.point() == N.REFUSED
    assert not any(argv[0] == "gpu-lease" for argv in node.ran), "no card was asked for"
    result, point = node.read("result.json"), node.read(N.POINT_FILE)
    assert result["validation"]["status"] == "failed" and result["measurements"] == []
    assert re.search(says, point["refused"]) and result["validation"]["detail"] == f"refused, nothing timed: {point['refused']}"
    assert not (node.run_dir / N.VERDICT_FILE).exists() and not list(node.scratch_root.iterdir())


def test_the_node_side_refuses_a_tree_whose_point_is_not_the_one_it_was_launched_as(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=f"point identity changed: this tree gives {POINT} {node.identity}"):
        node.point(identity="0" * 64)
    (node.root / "verity/protocols/flock/prover/src/main.rs").write_text("fn main() { /* edited on the node */ }\n")
    with pytest.raises(ValueError, match="a file its time depends on differs from the launching checkout's"):
        node.point()
    (node.root / "verity/protocols/flock/prover/src/main.rs").write_text(FILES["verity/protocols/flock/prover/src/main.rs"])
    # a shipped tree is no checkout: a file a build left under the closure's directories counts, and the refusal says so
    (left := node.root / "verity/protocols/flock/prover/target/built.o").parent.mkdir()
    left.write_text("object\n")
    with pytest.raises(ValueError) as shipped:
        node.point()
    assert ("This tree is no checkout, so every file under verity/protocols/flock/prover, verity/protocols/flock/pod "
            "counts: one the launching checkout doesn't track (what a build or a setup left there) changes the identity "
            "too. Its closure here is ") in str(shipped.value)
    assert re.search(r"Its closure here is \d+ files \(SHA-256 [0-9a-f]{16}\.\.\.\); `plan` prints the launching "
                     r"checkout's as its `toolchain`$", str(shipped.value))
    # in a checkout the same file is untracked and no part of the point, and a tracked file that differs is
    left.unlink()
    git(node.root)
    left.write_text("object\n")
    (node.root / "verity/protocols/flock/prover/src/main.rs").write_text("fn main() { /* edited in the checkout */ }\n")
    with pytest.raises(ValueError) as checkout:
        node.point()
    assert "differs from the launching checkout's. Its closure here is " in str(checkout.value)
    assert "no checkout" not in str(checkout.value)
    monkeypatch.setenv("RUST_MIN_STACK", "1")
    (node.root / "verity/protocols/flock/prover/src/main.rs").write_text(FILES["verity/protocols/flock/prover/src/main.rs"])
    with pytest.raises(ValueError, match=r"differs from spec 'steady''s env and setup in \['RUST_MIN_STACK'\]"):
        node.point()
    assert node.ran == [] and not (node.run_dir / "result.json").exists()
    assert N.main(["--spec", node.spec, "--point", POINT, "--identity", node.identity]) == 2


def test_the_node_side_refuses_a_run_without_the_prover_its_setup_builds(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    monkeypatch.setenv("VERITY_BENCH_STEADY_BIN", str(tmp_path / "no-prover"))
    assert node.point() == N.REFUSED and node.ran == []
    assert "the spec's setup left no prover" in node.read(N.POINT_FILE)["refused"]


#: the timed step refusing inside the lease: what the node answers otherwise, and what the refusal says
INSIDE_THE_LEASE = [
    ({"stamp": {"driver": "570.86.16"}},
     "the node's driver and CUDA toolkit are {'driver': '570.86.16', 'cuda_toolkit': '12.8.93'}, and spec 'steady' says"),
    ({"answers": {"nvcc --version": NVCC.replace(CUDA, "12.9.41")}},
     "Set the spec's `driver` and `cuda_toolkit` to the node's"),
    # the stamp's `cuda`, the newest the driver serves, is never taken for the toolkit's version
    ({"stamp": {"cuda": CUDA}, "answers": {"nvcc --version": NVCC.replace(CUDA, "12.9.41")}},
     "the node's driver and CUDA toolkit are {'driver': '570.86.15', 'cuda_toolkit': '12.9.41'}, and spec 'steady' says"),
    ({"stamp": {"card": "NVIDIA B200"}},
     "the lease's card is a NVIDIA B200, not the `card` of spec 'steady' (h200: NVIDIA H200)"),
    ({"stamp": {"card": "none", "gpus": [], "driver": None, "cuda": None, "why": "no nvidia-smi on PATH"}},
     "the host stamp doesn't say what the lease's card is (no nvidia-smi on PATH)"),
    ({"stamp": {"gpus": [{"uuid": "GPU-3", "sm_mhz": 1980, "mem_mhz": 3201}, {"uuid": "GPU-4", "sm_mhz": 0, "mem_mhz": 0}]}},
     "the host stamp doesn't say what the lease's card is (it names 2 GPUs and the driver '570.86.15'): inside a "
     "gpu-lease of one card it is that card's, which gpu-lease names in GPU_LEASE_UUID"),
    ({"answers": {"nvcc --version": None}}, "the nvcc on the path the spec's setup leaves doesn't say its version"),
    ({"answers": {"nvcc --version": "nvcc: no version here\n"}},
     "the nvcc on the path the spec's setup leaves doesn't say its version"),
]


@pytest.mark.parametrize("change, says", INSIDE_THE_LEASE, ids=[says[:40] for _, says in INSIDE_THE_LEASE])
def test_the_timed_step_refuses_a_card_or_versions_that_are_not_the_specs(tmp_path, monkeypatch, change, says):
    node = Node(tmp_path, monkeypatch, **change)
    assert node.point() == N.REFUSED
    assert [argv[0] for argv in node.ran] == [sys.executable, "gpu-lease"] and node.ran[0][1] == SP.STAGER, "nothing timed"
    assert says in node.read(N.POINT_FILE)["refused"] and says in node.read("result.json")["validation"]["detail"]
    assert not (node.run_dir / S.RATES).exists() and not (node.run_dir / START).exists()


def test_a_lease_that_never_ran_the_timed_step_is_a_refusal(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch, lease_exit=75)
    assert node.point() == N.REFUSED
    assert node.read(N.POINT_FILE)["refused"] == "the timed step never began: gpu-lease exited 75 (out/lease.log)"


def test_a_timed_step_stopped_before_its_end_is_a_refusal_that_says_it_began(tmp_path, monkeypatch):
    """A lease preempted mid-step: the step had recorded the node at its start and wrote no record of its end."""

    node = Node(tmp_path, monkeypatch)
    timer = node.run

    def preempted(argv, log):
        if argv[0] == "gpu-lease":
            assert timer(argv, log) == 0
            (node.run_dir / N.TIMED_FILE).unlink()
            return 143
        return timer(argv, log)

    node.run = preempted
    assert node.point() == N.REFUSED
    assert (node.run_dir / START).is_file()
    assert node.read(N.POINT_FILE)["refused"] == (
        "the timed step was stopped before its end and left no row, as a preempted lease stops it: gpu-lease exited "
        "143 (out/lease.log)")
    assert node.read("result.json")["validation"]["status"] == "failed"


@pytest.fixture
def signals():
    """TERM's and INT's handlers, put back after a test that lets the node side set its own."""

    kept = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    yield
    for number, handler in kept.items():
        signal.signal(number, handler)


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT], ids=["TERM", "INT"])
def test_a_cancelled_or_timed_out_point_removes_its_scratch_and_exits_143(tmp_path, monkeypatch, signals, number):
    """A cancel or a timeout reaches the node side while it waits for its lease: the run ends with 143 and no result,
    and the statement it staged is gone from the scratch."""

    node = Node(tmp_path, monkeypatch)
    seen = {}

    def lease(argv, log):
        if argv[0] != "gpu-lease":
            return Node.run(node, argv, log)
        seen["scratch"] = scratch = Path(argv[-1])
        seen["staged"] = (scratch / "z" / "gemm64" / "c1" / "stage" / "circuit.txt").is_file()
        assert signal.getsignal(number) is N._stopped, "the signal would end the tests' own process"
        signal.raise_signal(number)
        return 0

    node.run = lease
    N.stop_on_signals()
    with pytest.raises(SystemExit) as stopped:
        node.point()
    assert stopped.value.code == N.STOPPED == 143
    assert seen["staged"] and not seen["scratch"].exists() and list(node.scratch_root.iterdir()) == []
    assert not (node.run_dir / "result.json").exists() and not (node.run_dir / N.POINT_FILE).exists()


def test_main_ends_on_a_signal_and_a_command_it_waits_for_is_ended_first(tmp_path, monkeypatch, signals):
    """`main` sets the handlers; and `_run`, ended while it waits, sends its command TERM and waits for it, so a lease's
    holder stops its step and releases its card before the scratch goes. The command here signals its parent, the
    tests' process, once it can be signalled itself, and writes a file when TERM reaches it."""

    node = Node(tmp_path, monkeypatch)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, signal.SIG_DFL)
    assert N.main(["--spec", node.spec, "--point", POINT, "--identity", "0" * 64]) == 2
    assert [signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)] == [N._stopped, N._stopped]

    ended = tmp_path / "ended"
    command = ("import os, signal, sys\n"
               "def end(*_):\n"
               "    open(sys.argv[1], 'w').write('ended by TERM')\n"
               "    sys.exit(0)\n"
               "signal.signal(signal.SIGTERM, end)\n"
               "os.kill(os.getppid(), signal.SIGTERM)\n"
               "while True:\n"
               "    signal.pause()\n")
    with pytest.raises(SystemExit) as stopped:
        N._run([sys.executable, "-c", command, str(ended)], tmp_path / "command.log")
    assert stopped.value.code == 143 and ended.read_text() == "ended by TERM"
    assert (tmp_path / "command.log").read_text() == ""


#: a point that was timed and isn't a passed point: the stand-in's change and what the verdict says
TIMED_NOT_PASSED = [
    ({"row": {"flat": False}}, "fail: `flat` (fail): its statement cache's hit rate wasn't flat"),
    ({"row": {"cores": {"count": 8, "list": "0-7", "numa_nodes": [0]}}}, "fail: it ran on 8 cores; the spec's `cores` is 10"),
    ({"row": {"cores": DROP}}, "cannot tell: its row doesn't say which cores it ran on"),
    ({"row": {"cores": {"count": 10, "list": "20-29"}}},
     "cannot tell: `one_numa_node` (cannot tell): its row lacks `cores.numa_nodes`"),
    ({"row": {"cores": {"count": 10, "list": "20-23,24-29", "numa_nodes": [0, 1]}}},
     "fail: `one_numa_node` (fail): its CPUs (20-23,24-29) are on 2 NUMA nodes (0, 1)"),
    ({"row": {"flat": DROP}}, "cannot tell: `flat` (cannot tell): "),
    ({"leases": None}, "cannot tell: `one_lease_own_card` (cannot tell): the run holds no `gpu-lease-usage.jsonl`"),
    ({"leases": lease(0)}, "fail: `one_lease_own_card` (fail): its lease held GPU 0; its provers ran on GPU 3"),
    ({"answers": {"gpu-lease status": None}}, "cannot tell: `card_clear_start_end` (cannot tell): "),
    ({"answers": {"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader": "GPU-3, 99, 5 MiB\n"}},
     "fail: `card_clear_start_end` (fail): GPU 3 wasn't its own: at its start, a compute process (pid 99 5MiB)"),
    ({"answers": {"gpu-lease status": ANSWERS["gpu-lease status"] + "3 held 9m pid=5 run=r-third who=x cmd=y\n"}},
     "fail: `card_clear_start_end` (fail): GPU 3 wasn't its own: at its start, another run's lease (3 held 9m pid=5 run=r-third who=x)"),
    ({"row": {"type": "gemm32"}}, "cannot tell: the run's `out/rate/rate.json` holds 0 rows of `gemm64` at 2 provers"),
]


@pytest.mark.parametrize("change, says", TIMED_NOT_PASSED, ids=[says[:44] for _, says in TIMED_NOT_PASSED])
def test_a_timed_point_whose_files_do_not_show_all_the_checks_is_not_passed(tmp_path, monkeypatch, change, says):
    node = Node(tmp_path, monkeypatch, **change)
    assert node.point() == SP.NOT_PASSED
    verdict, result = node.read(N.VERDICT_FILE), node.read("result.json")
    assert verdict["passed"] is False and verdict["why"].startswith(says), verdict["why"]
    assert result["validation"] == result["validation"] | {"status": "failed", "detail": verdict["why"]}
    assert result["measurements"] == []


def test_a_point_that_runs_again_in_its_run_directory_leaves_no_earlier_attempts_records(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch)
    assert node.point() == 0
    monkeypatch.setenv("VERITY_BENCH_STEADY_BIN", str(tmp_path / "no-prover"))
    assert node.point() == N.REFUSED
    left = sorted(p.relative_to(node.run_dir).as_posix() for p in node.run_dir.rglob("*") if p.is_file())
    assert left == [S.LEASES, "out/lease.log", N.POINT_FILE, "out/stage.log", "result.json"]
    assert node.read("result.json")["validation"]["status"] == "failed"


def test_the_node_side_reads_the_environment_through_declared_settings_and_names_the_tools_it_runs():
    """The stager and the timing script are named by their paths in the tree, where the tool's closure covers them."""

    root = Path(__file__).resolve().parents[3]
    assert (root / SP.STAGER).is_file() and (root / SP.TIMER).is_file() and SP.TIMER in SP.CLOSURE
    assert f"timed_n = max(a.sessions, {SP.FLOOR} * K)" in (root / SP.TIMER).read_text(), "the floor a point's count takes"
    assert (root / "benchmarks/harness/steady_point.sh").is_file() and STEADY_POINT.command[-1].endswith("steady_point.sh")
    assert {SP.STAGER, SP.TIMER} <= set(STEADY_POINT.closure)
    assert all(any(path.startswith(pattern.removesuffix("/**")) for pattern in STEADY_POINT.closure) for path in SP.CLOSURE)
    assert all(settings.REGISTRY.known(name) for name in (
        "VERITY_BENCH_STEADY_BIN", "VERITY_BENCH_STEADY_PY", "VERITY_BENCH_STEADY_SCRATCH", "CUDA_VISIBLE_DEVICES"))


def test_the_readme_states_a_steady_state_spec_as_the_code_keeps_it():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    section = readme[readme.index("## A steady-state spec"):readme.index("## Settings")]
    for name in (SP.SCHEMA, "verity_bench.tool:STEADY_POINT", "VERITY_BENCH_STEADY_BIN", "VERITY_BENCH_STEADY_PY",
                 "benchmarks/harness/steady_point.sh", SP.STAGER, SP.TIMER, "steady_plan.completed_rows", P.DERIVATION, SP.VERSIONS,
                 *SP.CLOSURE[:2], "out/verdict.json", "out/point.json"):
        assert f"`{name}`" in section, name
    assert (N.VERDICT_FILE, N.POINT_FILE) == ("out/verdict.json", "out/point.json")
    assert (f"exits 0 for a passed point, {SP.NOT_PASSED} for a point that was timed and didn't pass, {N.REFUSED} where "
            f"nothing was timed, and {N.STOPPED} where\nTERM or INT ended it") in section
    assert "`steady_plan.completed_rows`, `completed`" in section and "runs and arts" in section
    assert all(f"`{name}`" in section for name in ("kit.lease", "kit.host_stamp", "kit.Writer", "jobs.dispatch"))
    spec_keys = {"workload", "card", "cores", "sessions", "ahead", "driver", "cuda_toolkit", "setup", "[env]", "[admission]",
                 "max_minutes", "mem_gb", "[[row]]", "type", "n", "pin", "provers", "warm", "period", "staged"}
    assert all(f"`{key}`" in section for key in spec_keys), [key for key in spec_keys if f"`{key}`" not in section]
