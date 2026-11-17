"""The stand-ins the steady-state point tests share: a made-up tree with its spec, setup and record, a store of runs and
labels, and a node the point's node side runs on."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from test_steady import DROP, END, SESSIONS, START, made

from verity_bench import steady as S
from verity_bench import steady_cards as C
from verity_bench import steady_node as N
from verity_bench import steady_plan as P
from verity_bench import steady_spec as SP
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
#: the staged statement of the spec's `gemm64` row, of `n` 1, as a row states it
STAGE = {"definition": "Gemm_v1{K=64}", "k_log": 20, "blocks": 1, "full": 8}
#: what a row of a run of the point holds beyond the script's own fields, and its statement's units as the script writes
#: them (`n` and `stage.blocks`)
OWN = {"pin": PIN, "n": 1, "stage": STAGE, "cores": {"count": 10, "list": "0-9", "numa_nodes": [0]}, S.CARD_NUMA: 0,
       SP.VERSIONS: {"driver": DRIVER, "cuda_toolkit": CUDA}}
#: the made-up tree's files beside its spec: the prover's source, its tests and its document, a build script, the timing
#: script and the spec's setup
FILES = {
    "uv.lock": "lock\n", "pyproject.toml": "",
    "verity/protocols/proofs/flock/prover/src/main.rs": "fn main() {}\n",
    "verity/protocols/proofs/flock/prover/cuda/ntt.cu": "// kernel\n",
    "tests/verity/protocols/proofs/flock/prover/test_prover.py": "def test(): pass\n",
    "verity/protocols/proofs/flock/prover/README.md": "the prover\n",
    "verity/protocols/proofs/flock/prover/pod/70-class-sweep.sh": "FLAGS=-O3\n",
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


# ---- the timed sessions ----------------------------------------------------------------------------------------------------

#: a prover's filings where one session in four waits: seven sessions of two provers a second apart, then one after nine
BURSTY = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0)


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


# ---- a type at its full statement and at a quarter of one ------------------------------------------------------------------

#: the spec's `gemm64` at its full statement of 8 blocks and at a quarter of it: one circuit, so one pin
FULL, QUARTER = SPEC["row"][0] | {"n": 8}, SPEC["row"][0] | {"n": 2, "provers": [2]}
#: what a row of a run of each holds of its statement, by its `n`, as the timing script writes it
TIMED = {n: {"n": n, "stage": STAGE | {"blocks": n}} for n in (8, 2)}


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
    `said` (its pin.txt) and `units`: what the stager stages; `point_id`: the point the run is of; `card_node`: the NUMA
    node the timed step reads of the leased card (`steady_cards.card_numa_node`, which `numa` and its own test read)."""

    def __init__(self, tmp_path: Path, monkeypatch, *, row: dict | None = None, answers: dict | None = None,
                 circuit: str = CIRCUIT, said: str = PIN, units: int = 1, cpus: int = 10, lease_exit: int | None = None,
                 spec: dict | None = None, stamp: dict | None = None, point_id: str = POINT, card_node: int | None = 0,
                 **texts):
        self.root, self.spec, self.point_id = *tree(tmp_path, spec=spec), point_id
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
        monkeypatch.setattr(C, "NODES", self.sys / "node")
        monkeypatch.setattr(C, "DEVICES", self.sys / "pci")
        monkeypatch.setattr(C, "card_numa_node", lambda device, ask: card_node)
        self.row, self.files = made(OWN | {SP.VERSIONS: DROP} | (row or {}), **texts)
        self.answers = ANSWERS | (answers or {})
        self.circuit, self.said, self.units, self.cpus, self.lease_exit = circuit, said, units, cpus, lease_exit
        self.ran: list[list[str]] = []
        #: the store's stand-in (`stored`): the files a fetch gives, what its listing answers, the fetch's exit code, the
        #: listings asked, and what the timed step saw of the scratch
        self.fetch, self.listed, self.fetch_exit, self.asked, self.seen = {}, None, 0, [], None
        spec = SP.load_spec(self.root / self.spec, root=self.root)
        self.identity = SP.point_identity(spec, SP.point(spec, point_id), sha256_of(SP.closure(self.root, spec)))

    def args(self, **changes):
        return N.parse(["--spec", self.spec, "--point", self.point_id, "--identity", changes.get("identity", self.identity),
                        *(["--card", str(changes["card"])] if "card" in changes else [])])

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


#: a made-up node of two NUMA nodes, two cards on each: the run's CPUs (20 to 29) are on the second
CARDS = "0, 00000000:17:00.0\n1, 00000000:65:00.0\n2, 00000000:B1:00.0\n3, 00000000:CA:00.0\n"
NUMA = {"node/node0/cpulist": "0-15\n", "node/node1/cpulist": "16-31\n", "node/node1/meminfo": "",
        "pci/0000:17:00.0/numa_node": "0\n", "pci/0000:65:00.0/numa_node": "0\n",
        "pci/0000:b1:00.0/numa_node": "1\n", "pci/0000:ca:00.0/numa_node": "1\n"}


def numa(node: Node, cards: str | None = CARDS, **changes: str | None) -> None:
    """What the node says of its NUMA nodes and its cards: `NUMA`, with files replaced or left out (None), their
    names' `/`, `:` and `.` written `_`."""

    node.answers = node.answers | {" ".join(C.CARDS): cards}
    by_key = {re.sub(r"[/:.]", "_", path): path for path in NUMA}
    for path, text in (NUMA | {by_key[key]: text for key, text in changes.items()}).items():
        if text is not None:
            (node.sys / path).parent.mkdir(parents=True, exist_ok=True)
            (node.sys / path).write_text(text)


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
