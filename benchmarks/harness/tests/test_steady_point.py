"""A steady-state spec's points on a made-up tree, a store with a directory remote and stand-ins for the node: the schema
loaded strictly, what a point's identity binds and what it leaves out, which run `plan` counts as a point's passing run
and why it counts no other, the command `run --direct` launches, and the node side from its refusals to its verdict. No
GPU, no node and no clock."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
from test_steady import DROP, END, SESSIONS, START, lease, made
from verity_config import ConfigError

from verity_bench import jobs, settings
from verity_bench import steady as S
from verity_bench import steady_node as N
from verity_bench import steady_plan as P
from verity_bench import steady_spec as SP
from verity_bench.cli import main
from verity_bench.tool import STEADY_POINT

CIRCUIT = "the circuit of gemm64\n"
PIN = hashlib.sha512(CIRCUIT.encode()).hexdigest()
DRIVER, CUDA = "570.86.15", "12.8.93"
SETUP = "benchmarks/specs/steady-setup.sh"
RECORD = "prove-glm53-flock-steady"
#: the made-up run of `test_steady.made`: two provers of `gemm64` on GPU 3, in the run `r-own`
POINT, RUN = "gemm64-k2", "r-own"
#: what a row of a run of the point holds beyond the script's own fields
OWN = {"pin": PIN, "cores": {"count": 10, "list": "0-9"}, SP.VERSIONS: {"driver": DRIVER, "cuda": CUDA}}
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
    "driver": DRIVER, "cuda": CUDA, "setup": SETUP,
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
    for path, text in ({**FILES, "benchmarks/specs/steady.toml": toml(doc)} | (files or {})).items():
        if text is not None:
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text)
    return root, "benchmarks/specs/steady.toml"


def git(root: Path) -> None:
    import subprocess

    subprocess.run("git init -q && git add . && GIT_AUTHOR_DATE='@1700000000 +0000' GIT_COMMITTER_DATE='@1700000000 +0000' "
                   "git -c user.name=t -c user.email=t@example.com -c commit.gpgsign=false commit -qm spec",
                   shell=True, cwd=root, check=True)


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
                              "staged": "/dev/shm/head", "provers": 2}
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
    ({"spec": {"cuda": ""}}, "`cuda` '' isn't a version"),
    ({"spec": {"setup": None}}, "setup"),
    ({"spec": {"setup": "benchmarks/specs/none.sh"}}, "`setup` 'benchmarks/specs/none.sh' is no script of the source tree"),
    ({"spec": {"setup": "../steady-setup.sh"}}, "is no script of the source tree"),
    ({"spec": {"env": {"CUDA_VISIBLE_DEVICES": "0"}}}, "`env` can't set 'CUDA_VISIBLE_DEVICES'"),
    ({"spec": {"env": {"VERITY_BENCH_STEADY_BIN": "/x"}}}, "`env` can't set 'VERITY_BENCH_STEADY_BIN'"),
    ({"spec": {"admission": {"max_minutes": 0}}}, r"`admission\.max_minutes` is 0"),
    ({"spec": {"row": []}}, "missing required key 'row'"),
    ({"spec": {"row0": {"type": "--"}}}, "its `type` names nothing"),
    ({"spec": {"row0": {"n": 0}}}, "`n` is 0"),
    ({"spec": {"row0": {"pin": PIN[:64]}}}, "`pin` isn't a SHA-512 in lowercase hexadecimal"),
    ({"spec": {"row0": {"pin": PIN.upper()}}}, "`pin` isn't a SHA-512"),
    ({"spec": {"row0": {"provers": []}}}, "`provers` is a list of distinct prover counts"),
    ({"spec": {"row0": {"provers": [2, 2]}}}, "`provers` is a list of distinct prover counts"),
    ({"spec": {"row0": {"provers": [0]}}}, r"`provers \(0\)` is 0"),
    ({"spec": {"row0": {"provers": [1, 4]}}}, "`sessions` is 16, under 8 a prover at 4 provers"),
    ({"spec": {"row0": {"staged": "z/gemm64"}}}, "`staged` 'z/gemm64' isn't an absolute path on the node"),
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
    "the CUDA version": {"spec": {"cuda": "12.9.41"}},
    "the run's environment": {"spec": {"env": {"RUST_MIN_STACK": "2147483648"}}},
    "which script is the setup": {"spec": {"setup": "benchmarks/specs/other-setup.sh"},
                                  "files": {"benchmarks/specs/other-setup.sh": FILES[SETUP]}},
    "the block type": {"spec": {"row0": {"type": "gemm32"}}},
    "the statement's units": {"spec": {"row0": {"n": 2}}},
    "the statement's pin": {"spec": {"row0": {"pin": "cd" * 64}}},
    "the warm sessions": {"spec": {"row0": {"warm": 3}}},
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
    assert sorted(covered) == ["benchmarks/one_stage/server_rate.py", SETUP, "verity/protocols/flock/pod/70-class-sweep.sh",
                               "verity/protocols/flock/prover/cuda/ntt.cu", "verity/protocols/flock/prover/src/main.rs"]
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
        """The lookup of a machine that has read nothing yet, pulling the remote's attempts since the made-up day."""

        return P.SteadyCompletion(self.fresh(), root=self.root, since=lambda: 1_700_000_000, log=lambda message: None,
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
    ({}, {"changes": {"cores": {"count": 8, "list": "0-7"}}}, "fail: it ran on 8 cores; the spec's `cores` is 10"),
    ({}, {"changes": {"cores": DROP}}, "cannot tell: its row doesn't say which cores it ran on"),
    ({}, {"changes": {"pin": "cd" * 64}}, "fail: its `pin` is another circuit's (cdcdcdcdcdcdcdcd...)"),
    ({}, {"changes": {SP.VERSIONS: {"driver": "570.86.16", "cuda": CUDA}}}, "cannot tell: its row's `versions`"),
    ({}, {"changes": {SP.VERSIONS: DROP}}, "cannot tell: its row's `versions` (None) aren't the spec's"),
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
    ({"cores": {"count": 16, "list": "0-15"}}, {}, "fail: it ran on 16 cores; the spec's `cores` is 10"),
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


def test_which_row_of_a_labelled_run_is_judged_is_one_constant(planned, monkeypatch):
    """A run from before the script wrote `flat`, and an art of the record that holds its row completed. By default the
    run's own row is judged, and it can't show the check; `COMPLETED` judges the art's row against the run's files."""

    spec, item, identity, store = planned
    assert P.LABELLED_ROW == P.OWN
    files = store.files({SP.VERSIONS: DROP, "flat": DROP})
    run = store.ran("r20231114-100000-0001", files)
    completed = json.loads((files / S.RATES).read_text())
    completed[0]["flat"] = True
    (rows := store.tmp / "completed").mkdir()
    (rows / "rate.json").write_text(json.dumps(completed))
    art = store.art("prove-steady/v1", {"runs": [run]}, rows)
    store.label(art, "run_of_record", RECORD)

    found, why = store.lookup()(spec, item, identity)
    assert found is None and why.startswith(f"{run} (labelled `{RECORD}`): cannot tell: `flat` (cannot tell): "), why
    monkeypatch.setattr(P, "LABELLED_ROW", P.COMPLETED)
    assert store.lookup()(spec, item, identity) == (run, None)

    completed[0]["accepted"] = 19
    (again := store.tmp / "completed-again").mkdir()
    (again / "rate.json").write_text(json.dumps(completed))
    store.label(art, "superseded_by", newer := store.art("prove-steady/v1", {"runs": [run]}, again))
    store.label(newer, "run_of_record", RECORD)
    found, why = store.lookup()(spec, item, identity)
    assert found is None and "fail: `accepted` (fail): 19 of its 20 sessions accepted" in why, "the completed row is checked too"


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
    """The remote's attempts are pulled since the spec was added, which git says of a spec under `benchmarks/specs`."""

    root, path = tree(tmp_path)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "here"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(Store(tmp_path, root).config))
    (elsewhere := root / "elsewhere.toml").write_text((root / path).read_text())
    with pytest.raises(ValueError, match="elsewhere.toml isn't a spec of benchmarks/specs.*--local-only"):
        P.make_plan(elsewhere, root=root)
    with pytest.raises(ValueError, match="git has no commit adding benchmarks/specs/steady.toml"):
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
#: what the node answers: the leased card, the toolkit, the leases (the run's own, and another run's on GPU 0) and the
#: compute processes (one, on GPU 0)
ANSWERS = {
    "nvidia-smi -i 3 --query-gpu=driver_version,name,uuid --format=csv,noheader": f"{DRIVER}, NVIDIA H200, GPU-3\n",
    "nvcc --version": NVCC,
    "gpu-lease status": (f"3 held 1m pid=7 run={RUN} who=benchmarks cmd=python -m verity_bench.steady_node\n"
                         "0 held 5m pid=11 run=r-other who=other cmd=python train.py\n1 free\n"),
    "nvidia-smi --query-gpu=index,uuid --format=csv,noheader": "0, GPU-0\n3, GPU-3\n",
    "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader": "GPU-0, 12, 100 MiB\n",
}


class Node:
    """A stand-in for the node under one run of the point: the stager, gpu-lease, the timing script and what the node
    answers. `row`: fields of the row the script writes; `texts`: the run's files as `test_steady.made` takes them;
    `answers`: an answer replaced (None: the command fails); `circuit`, `said` (its pin.txt) and `units`: what the
    stager stages."""

    def __init__(self, tmp_path: Path, monkeypatch, *, row: dict | None = None, answers: dict | None = None,
                 circuit: str = CIRCUIT, said: str = PIN, units: int = 1, cpus: int = 10, lease_exit: int | None = None,
                 **texts):
        self.root, self.spec = tree(tmp_path)
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
        self.row, self.files = made(OWN | {SP.VERSIONS: DROP} | (row or {}), **texts)
        self.answers = ANSWERS | (answers or {})
        self.circuit, self.said, self.units, self.cpus, self.lease_exit = circuit, said, units, cpus, lease_exit
        self.ran: list[list[str]] = []
        spec = SP.load_spec(self.root / self.spec, root=self.root)
        self.identity = SP.point_identity(spec, SP.point(spec, POINT), sha256_of(SP.closure(self.root, spec)))

    def args(self, **changes):
        return N.parse(["--spec", self.spec, "--point", POINT, "--identity", changes.get("identity", self.identity)])

    def ask(self, argv):
        answer = self.answers.get(" ".join(argv))
        return (1, "") if answer is None else (0, answer)

    def run(self, argv, log: Path) -> int:
        argv = list(argv)
        self.ran.append(argv)
        log.write_text("log\n")
        if argv[1:2] == [SP.STAGER]:
            stage = Path(argv[-1])
            stage.mkdir(parents=True)
            (stage / "circuit.txt").write_text(self.circuit)
            (stage / "pin.txt").write_text(self.said + "\n")
            (stage / "stage.json").write_text(json.dumps({"n": self.units, "definition": "Gemm_v1{K=64}", "k_log": 20}))
            return 0
        if argv[0] == "gpu-lease":
            if self.lease_exit is not None:
                return self.lease_exit
            inside = argv[argv.index("--") + 1:]
            assert inside[:3] == [sys.executable, "-m", "verity_bench.steady_node"]
            self.monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
            rc = N.point(N.parse(inside[3:]), run=self.run, ask=self.ask, now=lambda: 1_700_000_000.0)
            self.monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
            if (usage := self.files(S.LEASES)) is not None:
                (self.run_dir / S.LEASES).write_text(usage)
            return rc
        assert argv[1] == SP.TIMER, argv
        out = Path(argv[argv.index("--out") + 1])
        (step := out / f"gemm64-k{argv[argv.index('--provers') + 1]}").mkdir(parents=True)
        (out / "rate.json").write_text(json.dumps([self.row]))
        if (sessions := self.files(SESSIONS)) is not None:
            (step / "sessions.jsonl").write_text(sessions)
        (step / "prover-gpu3-0.log").write_text("x" * (N.KEPT_LOG_BYTES + 10))
        return 0

    def point(self, **changes) -> int:
        return N.point(self.args(**changes), run=self.run, ask=self.ask, now=lambda: 1_700_000_000.0,
                       cpus=lambda: list(range(20, 20 + self.cpus)))

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
    assert stager == [sys.executable, SP.STAGER, "gemm64", "1", str(scratch / "z" / "gemm64" / "c1" / "stage")]
    assert lease_argv == ["gpu-lease", "1", "--wait", "--max-min", "90", "--preemptible", "--mem-gb", "80", "--",
                          sys.executable, "-m", "verity_bench.steady_node", "--spec", node.spec, "--point", POINT,
                          "--identity", node.identity, "--inside", str(scratch)]
    assert timer == [sys.executable, SP.TIMER, "--bin", str(node.binary), "--staged", str(scratch / "z"),
                     "--types", "gemm64", "--provers", "2", "--out", str(scratch / "rate"), "--device", "3",
                     "--sessions", "16", "--warm", "2", "--ahead", "2"]

    [row] = node.read(S.RATES)
    assert row[SP.VERSIONS] == {"driver": DRIVER, "cuda": CUDA}, "read on the node and written in the row"
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
    assert point["timed"] == {"refused": None, "device": "3", "card": {"name": "NVIDIA H200", "uuid": "GPU-3"},
                              SP.VERSIONS: {"driver": DRIVER, "cuda": CUDA}, "exit": 0, "rows": 1}
    verdict = node.read(N.VERDICT_FILE)
    assert (verdict["identity"], verdict["point"], verdict["passed"], verdict["why"]) == (node.identity, POINT, True, None)
    assert list(verdict["verdict"]["checks"]) == list(S.CHECKS)
    result = node.read("result.json")
    assert result["kind"] == N.KIND and result["validation"]["status"] == "passed" and result["run_id"] == RUN
    assert result["workload_fingerprint"]["identity"] == node.identity
    assert result["measurements"] == [{"name": "statements_per_s", "value": row["statements_per_s"], "unit": "1/s"}]

    spec = SP.load_spec(node.root / node.spec, root=node.root)
    assert SP.point_why(row, S.files_in(node.run_dir), spec, SP.point(spec, POINT), own=True) is None, "as plan judges it"


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
    ({"nvidia-smi -i 3 --query-gpu=driver_version,name,uuid --format=csv,noheader": "570.86.16, NVIDIA H200, GPU-3\n"},
     "the node's driver and CUDA toolkit are {'driver': '570.86.16', 'cuda': '12.8.93'}, and spec 'steady' says"),
    ({"nvcc --version": NVCC.replace(CUDA, "12.9.41")}, "Set the spec's `driver` and `cuda` to the node's"),
    ({"nvidia-smi -i 3 --query-gpu=driver_version,name,uuid --format=csv,noheader": f"{DRIVER}, NVIDIA B200, GPU-3\n"},
     "the lease's card is a NVIDIA B200, not the `card` of spec 'steady' (h200: NVIDIA H200)"),
    ({"nvidia-smi -i 3 --query-gpu=driver_version,name,uuid --format=csv,noheader": None},
     "nvidia-smi doesn't say what GPU 3 is (exit 1: '')"),
    ({"nvcc --version": None}, "the nvcc on the path the spec's setup leaves doesn't say its version"),
    ({"nvcc --version": "nvcc: no version here\n"}, "the nvcc on the path the spec's setup leaves doesn't say its version"),
]


@pytest.mark.parametrize("answers, says", INSIDE_THE_LEASE, ids=[says[:36] for _, says in INSIDE_THE_LEASE])
def test_the_timed_step_refuses_a_card_or_versions_that_are_not_the_specs(tmp_path, monkeypatch, answers, says):
    node = Node(tmp_path, monkeypatch, answers=answers)
    assert node.point() == N.REFUSED
    assert [argv[0] for argv in node.ran] == [sys.executable, "gpu-lease"] and node.ran[0][1] == SP.STAGER, "nothing timed"
    assert says in node.read(N.POINT_FILE)["refused"] and says in node.read("result.json")["validation"]["detail"]
    assert not (node.run_dir / S.RATES).exists() and not (node.run_dir / START).exists()


def test_a_lease_that_never_ran_the_timed_step_is_a_refusal(tmp_path, monkeypatch):
    node = Node(tmp_path, monkeypatch, lease_exit=75)
    assert node.point() == N.REFUSED
    assert "the timed step never ran: gpu-lease exited 75 (out/lease.log)" in node.read(N.POINT_FILE)["refused"]


#: a point that was timed and isn't a passed point: the stand-in's change and what the verdict says
TIMED_NOT_PASSED = [
    ({"row": {"flat": False}}, "fail: `flat` (fail): its statement cache's hit rate wasn't flat"),
    ({"row": {"cores": {"count": 8, "list": "0-7"}}}, "fail: it ran on 8 cores; the spec's `cores` is 10"),
    ({"row": {"cores": DROP}}, "cannot tell: its row doesn't say which cores it ran on"),
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
    assert (root / "benchmarks/harness/steady_point.sh").is_file() and STEADY_POINT.command[-1].endswith("steady_point.sh")
    assert {SP.STAGER, SP.TIMER} <= set(STEADY_POINT.closure)
    assert all(any(path.startswith(pattern.removesuffix("/**")) for pattern in STEADY_POINT.closure) for path in SP.CLOSURE)
    assert all(settings.REGISTRY.known(name) for name in (
        "VERITY_BENCH_STEADY_BIN", "VERITY_BENCH_STEADY_PY", "VERITY_BENCH_STEADY_SCRATCH", "CUDA_VISIBLE_DEVICES"))


def test_the_readme_states_a_steady_state_spec_as_the_code_keeps_it():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    section = readme[readme.index("## A steady-state spec"):readme.index("## Settings")]
    for name in (SP.SCHEMA, "verity_bench.tool:STEADY_POINT", "VERITY_BENCH_STEADY_BIN", "VERITY_BENCH_STEADY_PY",
                 "benchmarks/harness/steady_point.sh", SP.STAGER, SP.TIMER, "steady_plan.LABELLED_ROW", SP.VERSIONS,
                 *SP.CLOSURE[:2], "out/verdict.json", "out/point.json"):
        assert f"`{name}`" in section, name
    assert (N.VERDICT_FILE, N.POINT_FILE) == ("out/verdict.json", "out/point.json")
    assert (f"exits 0 for a passed point, {SP.NOT_PASSED} for a point that was timed and didn't pass, and {N.REFUSED} where "
            "nothing was timed") in section
    spec_keys = {"workload", "card", "cores", "sessions", "ahead", "driver", "cuda", "setup", "[env]", "[admission]",
                 "max_minutes", "mem_gb", "[[row]]", "type", "n", "pin", "provers", "warm", "staged"}
    assert all(f"`{key}`" in section for key in spec_keys), [key for key in spec_keys if f"`{key}`" not in section]
