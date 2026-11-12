"""A steady-state spec (`SCHEMA`), the harness's second schema beside a vLLM spec: the proof server's rate at steady
state, a point one staged block type at K provers with C cores on one card, timed by
`benchmarks/one_stage/server_rate.py`. `plan`, `run` and `check` take either schema, told apart by the file's `schema`.
Standard library and `verity_config` only: the node side reads it too.

    schema = "verity-bench/steady-spec/v0"
    id = "..."            # the spec's name
    workload = "glm53"    # the workload whose steady-state record its rows belong to
    card = "h200"         # the GPU, by its id in verity_config's registry
    cores = 10            # C: the CPUs a point's provers and verifiers run on
    sessions = 120        # the timed sessions of a point, at least 8 a prover
    ahead = 2             # the sessions a prover runs ahead of its verifier
    driver = "..."        # the GPU driver's version, as nvidia-smi gives it on the node
    cuda_toolkit = "..."  # the CUDA toolkit's version, as its nvcc gives it on the node
    setup = "..."         # a script of the tree the launcher sources: it builds the prover and names it and its Python
    [env]                 # variables the point's run is given
    [admission]           # max_minutes: the run's bound, its slot's and its lease's; mem_gb: the lease's host memory
    [[row]]               # a block type
    type = "..."          # the stager's word for it (`benchmarks/served_units/stage.py`)
    n = 1                 # the units of its full statement
    pin = "..."           # the SHA-512 of the statement's circuit: another circuit takes no GPU time
    provers = [1, 2, 4]   # a point a count
    warm = 2              # warm sessions a prover, before any timed one
    period = 4            # optional: the sessions a prover's filings repeat over, where they file in bursts
    staged = "/..."       # optional: a statement staged already, proved from a copy checked against `pin`: a directory
                          # on the node, or one of an artifact in the store (`art:ID/DIR`)

**A point's timed sessions** (`timed_sessions`) are the spec's `sessions`, at least the eight a prover the timing script
times whatever it is told (`FLOOR`), and for a row with a `period` the next count whose rated window holds a whole
number of bursts, a multiple of three: a burst is `period` sessions a prover, and a
window or a third of one that cuts a burst reads the burst's phase and not the rate. The period is declared, never read
off the run's first sessions: the count is then fixed before the run and is the point's. The node side writes the
period in the row (`steady.PERIOD`), where `level` reads it and cuts the window at whole bursts.

**A point's identity** (`point_identity`) is a digest of everything its time depends on, and of nothing else: its row's
`type`, `n`, `pin`, `warm` and `period`, its prover count, the spec's `card`, `cores`, `sessions`, `ahead`, `driver`,
`cuda_toolkit`, `env` and `setup`, and the files of `closure`: the prover's sources with the scripts that build them
(their flags), the script that times them, and the spec's setup. It leaves out the commit, the spec's other rows, its `id`, `workload` and
`admission`, and a row's `staged`: a new commit or an edit to another row times no row again. The driver and the
toolkit are the spec's because the node can't be asked before the run: the node side reads both on the node, refuses a
node whose versions aren't the spec's, and writes them in the row (`VERSIONS`). The toolkit's is `cuda_toolkit`, in the
spec and in the row: `cuda` is the harness kit's name for the newest CUDA the driver serves (`kit.host_stamp`), another
number.

**A row's `staged`** names a statement staged before the run, for a type whose staging takes longer than a point: a
directory on the node, which the node side copies, or a directory of an artifact in the store (`stored`), which it
fetches into its scratch with the read-only key the launch gives such a point's run. Either copy is held to the row's
`pin` and `n` like a statement staged in the run, and goes with the scratch.

**A passed point** (`point_why`): a row whose `pin`, `provers` and `cores.count` are the point's and its spec's, and
that passes `steady`'s ten checks against the files of the run that measured it. A row of a run of the point itself
also holds the point's load (its timed and warm sessions, its period) and the versions read on the node.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from verity_config import ConfigError, load, sha256

from verity_bench import steady
from verity_bench.vllm.presets import ENV_NAME, GPUS, RESERVED_ENV, SOURCE_ROOT, SPEC_ROOT

SCHEMA = "verity-bench/steady-spec/v0"
#: what a point's time depends on in the tree, beside its spec's setup: the prover (its sources, its patches of
#: upstream and its Python, which drives the provers), the scripts that build it, and the script that times it
CLOSURE = ("verity/protocols/flock/prover", "verity/protocols/flock/pod", "benchmarks/one_stage/server_rate.py")
#: what of `CLOSURE`'s directories is no source: tests, documents and the interpreter's caches
NOT_SOURCE = re.compile(r"(^|/)(tests|__pycache__)/|\.(md|pyc)$")
#: the row's field for the versions the node side read on the node: the driver's (`driver`) and the toolkit's
#: (`cuda_toolkit`)
VERSIONS = "versions"
STAGER = "benchmarks/served_units/stage.py"
TIMER = "benchmarks/one_stage/server_rate.py"
#: a row's `staged` in the store: an artifact's whole id, then the directory in it that holds the statement
STORED = re.compile(r"(art:[0-9a-f]{64})/([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)")
#: the node side's exit code for a point that was timed and whose row isn't a passed point; 0 is a passed point's
NOT_PASSED = 3
#: the sessions a prover the timing script times at least, whatever its `--sessions` says
FLOOR = 8


@dataclass(frozen=True)
class SteadyAdmission:
    """Where and how long a point may run; never its identity."""

    max_minutes: int | None = field(default=None, metadata={"identity": False})
    mem_gb: int | None = field(default=None, metadata={"identity": False})


@dataclass(frozen=True)
class SteadyRow:
    type: str
    n: int
    pin: str
    provers: tuple[int, ...]
    warm: int
    period: int | None = None
    staged: str | None = field(default=None, metadata={"identity": False})

    @property
    def name(self) -> str:
        """The row's `type` in the rate file and in its files' names: the stager's word with every run of characters
        other than letters, digits and `_` as one `_`."""

        return re.sub(r"[^A-Za-z0-9_]+", "_", self.type).strip("_")

    @property
    def stored(self) -> tuple[str, str] | None:
        """The artifact and the directory in it that hold the row's statement, where its `staged` names the store."""

        found = STORED.fullmatch(self.staged or "")
        return None if found is None or ".." in found[2].split("/") else (found[1], found[2])


@dataclass(frozen=True)
class SteadySpec:
    schema: str
    id: str
    workload: str
    card: str
    cores: int
    sessions: int
    ahead: int
    driver: str
    cuda_toolkit: str
    setup: str
    row: tuple[SteadyRow, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    admission: SteadyAdmission = field(default_factory=SteadyAdmission)

    @property
    def points(self) -> tuple[SteadyPoint, ...]:
        """Each row at each of its prover counts, in the spec's order."""

        return tuple(SteadyPoint(row, provers, timed_sessions(self.sessions, provers, row.period))
                     for row in self.row for provers in row.provers)


@dataclass(frozen=True)
class SteadyPoint:
    """One row of a spec at one prover count, with the sessions it times (`timed_sessions`)."""

    row: SteadyRow
    provers: int
    sessions: int

    @property
    def id(self) -> str:
        return f"{self.row.name}-k{self.provers}"

    def to_json(self) -> dict[str, Any]:
        row = self.row
        return {"type": row.type, "name": row.name, "n": row.n, "pin": row.pin, "warm": row.warm, "period": row.period,
                "staged": row.staged, "provers": self.provers, "sessions": self.sessions}


def window_gaps(sessions: int, provers: int) -> int:
    """The sessions the timing script rates of `sessions` timed ones at K provers: the gaps of its window, which runs
    from the K-th timed session's filing to the last but K's (`steady.window`)."""

    return sessions - 2 * provers


def timed_sessions(sessions: int, provers: int, period: int | None) -> int:
    """The sessions a point times: `sessions`, at least the timing script's `FLOOR` a prover, and with a `period` the
    next count whose rated window is a whole number of bursts and a multiple of three of them, so the window and each
    third of it (`steady.thirds`) hold whole bursts."""

    sessions = max(sessions, FLOOR * provers)
    if period is None:
        return sessions
    three = 3 * period * provers
    return 2 * provers + -(-window_gaps(sessions, provers) // three) * three


def is_steady(source: str | Path) -> bool:
    """Whether a spec, by its path or its name under `benchmarks/specs`, says it is a steady-state spec."""

    try:
        text = spec_path(source).read_text(encoding="utf-8")
    except OSError:
        return False
    return re.search(rf'^schema\s*=\s*"{re.escape(SCHEMA)}"\s*(#.*)?$', text, re.M) is not None


def spec_path(source: str | Path) -> Path:
    path = Path(source)
    return path if path.is_file() else SPEC_ROOT / f"{source}.toml"


def _positive(spec: str, **counts: int) -> None:
    for name, count in counts.items():
        if count < 1:
            raise ConfigError(f"{spec}: `{name}` is {count}; it is a count of at least 1")


def load_spec(source: str | Path, *, root: Path = SOURCE_ROOT) -> SteadySpec:
    """The spec at `source`, strictly: every key declared, every figure a point can be timed at."""

    path = spec_path(source)
    spec = load(path, SteadySpec)
    if spec.schema != SCHEMA:
        raise ConfigError(f"{path}: not a steady-state spec ({spec.schema!r} isn't {SCHEMA!r})")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", spec.id) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", spec.workload):
        raise ConfigError(f"{path}: `id` and `workload` are lowercase names of letters, digits and `-`")
    if spec.card not in GPUS:
        raise ConfigError(f"{path}: `card` {spec.card!r} is no GPU of verity_config's registry ({', '.join(GPUS)})")
    _positive(str(path), cores=spec.cores, sessions=spec.sessions)
    if spec.ahead < 0:
        raise ConfigError(f"{path}: `ahead` is {spec.ahead}; it is a count of sessions")
    for name in ("driver", "cuda_toolkit"):
        if not re.fullmatch(r"\d+(\.\d+)*", getattr(spec, name)):
            raise ConfigError(f"{path}: `{name}` {getattr(spec, name)!r} isn't a version (digits and dots), as the node "
                              "gives it: the node side refuses a node whose version is another")
    setup = PurePosixPath(spec.setup)
    if setup.is_absolute() or ".." in setup.parts or not (root / setup).is_file():
        raise ConfigError(f"{path}: `setup` {spec.setup!r} is no script of the source tree")
    for name in spec.env:
        if not ENV_NAME.fullmatch(name) or RESERVED_ENV.fullmatch(name):
            raise ConfigError(f"{path}: `env` can't set {name!r}: not a variable's name, or one the dispatcher, the "
                              "launcher or the lease sets")
    admission = spec.admission
    _positive(str(path), **{f"admission.{name}": count for name in ("max_minutes", "mem_gb")
                            if (count := getattr(admission, name)) is not None})
    if not spec.row:
        raise ConfigError(f"{path}: no `[[row]]`: a spec times at least one block type")
    for row in spec.row:
        where = f"{path}: row {row.type!r}"
        if not row.name:
            raise ConfigError(f"{where}: its `type` names nothing")
        _positive(where, n=row.n, warm=row.warm, **({"period": row.period} if row.period is not None else {}))
        if not re.fullmatch(r"[0-9a-f]{128}", row.pin):
            raise ConfigError(f"{where}: `pin` isn't a SHA-512 in lowercase hexadecimal, the circuit's")
        if not row.provers or len(set(row.provers)) != len(row.provers):
            raise ConfigError(f"{where}: `provers` is a list of distinct prover counts, a point each")
        _positive(where, **{f"provers ({k})": k for k in row.provers})
        if spec.sessions < 8 * max(row.provers):
            raise ConfigError(f"{where}: `sessions` is {spec.sessions}, under 8 a prover at {max(row.provers)} provers, "
                              f"where {TIMER} times 8 a prover instead: say the count it times")
        if row.staged is not None and row.stored is None and not PurePosixPath(row.staged).is_absolute():
            raise ConfigError(f"{where}: `staged` {row.staged!r} isn't an absolute path on the node, nor a directory "
                              "of an artifact in the store (`art:` and the artifact's 64 hexadecimal digits, `/`, the "
                              "directory)")
    names = [row.name for row in spec.row]
    if twice := sorted({name for name in names if names.count(name) > 1}):
        raise ConfigError(f"{path}: two rows are named {', '.join(twice)} in the rate file: a block type is one row")
    return spec


def point(spec: SteadySpec, point_id: str) -> SteadyPoint:
    found = [candidate for candidate in spec.points if candidate.id == point_id]
    if not found:
        raise ValueError(f"point {point_id!r} is not in spec {spec.id!r} ({', '.join(p.id for p in spec.points)})")
    return found[0]


def _tracked(root: Path, entries: tuple[str, ...]) -> set[str] | None:
    """The files git tracks under `entries` where `root` is a checkout, else None: a shipped tree (`git archive`) holds
    the tracked files and nothing else."""

    if not (root / ".git").exists():
        return None
    process = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", *entries], capture_output=True, text=True,
                             check=False)
    if process.returncode != 0:
        raise ValueError(f"git can't list {root}'s files ({process.stderr.strip()[:200]}), which a steady-state point's "
                         "identity covers")
    return set(process.stdout.split("\0")) - {""}


def closure(root: Path, spec: SteadySpec) -> dict[str, str]:
    """The SHA-256 of each file a point's time depends on, by its path in the tree: `CLOSURE`'s files and every source
    file under its directories, and the spec's setup. In a checkout only the files git tracks count, as they are in
    the working tree: what a build or an editor left beside them is no source, and isn't in the tree a run ships. A path
    of `CLOSURE` the tree lacks is refused: the identity would cover less than it says."""

    entries = (*CLOSURE, spec.setup)
    tracked = _tracked(root, entries)
    files: dict[str, str] = {}
    for entry in entries:
        path = root / entry
        found = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path] if path.is_file() else []
        kept = {p.relative_to(root).as_posix(): p for p in found}
        kept = {name: p for name, p in kept.items() if not (path.is_dir() and NOT_SOURCE.search(name))
                and (tracked is None or name in tracked)}
        if not kept:
            raise ValueError(f"{root} holds no {entry}, which a steady-state point's identity covers")
        files |= {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in kept.items()}
    return dict(sorted(files.items()))


def point_identity(spec: SteadySpec, item: SteadyPoint, files_sha256: str) -> str:
    """The point's identity: `files_sha256` is the digest of the spec's `closure` in the tree the point runs from."""

    row = item.row
    return sha256({
        "schema": SCHEMA, "card": spec.card, "cores": spec.cores, "sessions": spec.sessions, "ahead": spec.ahead,
        "driver": spec.driver, "cuda_toolkit": spec.cuda_toolkit, "env": dict(spec.env), "setup": spec.setup,
        "files_sha256": files_sha256,
        "row": {"type": row.type, "n": row.n, "pin": row.pin, "warm": row.warm, "period": row.period},
        "provers": item.provers,
    })


def versions(spec: SteadySpec) -> dict[str, str]:
    return {"driver": spec.driver, "cuda_toolkit": spec.cuda_toolkit}


def rows(files: steady.Files) -> list[Mapping[str, Any]] | str:
    """The rows of a run's rate file (`steady.rows`), or why it has none to read."""

    try:
        return steady.rows(files)
    except ValueError as error:
        return f"{steady.CANNOT_TELL}: {error}"


def own_row(files: steady.Files, item: SteadyPoint) -> Mapping[str, Any] | str:
    """The point's row in the files of a run of the point, or why the run has none: a run of a point times one row."""

    found = rows(files)
    if isinstance(found, str):
        return found
    mine = [row for row in found if row.get("type") == item.row.name and row.get("provers") == item.provers]
    if len(mine) != 1:
        return (f"{steady.CANNOT_TELL}: the run's `{steady.RATES}` holds {len(mine)} rows of `{item.row.name}` at "
                f"{item.provers} provers")
    return mine[0]


def point_why(row: Mapping[str, Any], files: steady.Files, spec: SteadySpec, item: SteadyPoint, *,
              own: bool = False) -> str | None:
    """Why a row of a run isn't a passed point of the spec, or None when it is one: its `pin`, `provers` and
    `cores.count` are the point's, and it passes the ten checks against the run's `files`. `own`: the row is a run of
    the point's own identity, so it also holds the point's load (its timed and warm sessions, its period) and the
    versions the node side read, the spec's."""

    if row.get("pin") != item.row.pin:
        return f"{steady.FAIL}: its `pin` is another circuit's ({str(row.get('pin'))[:16]}...)"
    if row.get("provers") != item.provers:
        return f"{steady.FAIL}: its row is at {row.get('provers')!r} provers"
    cores = row.get("cores")
    count = cores.get("count") if isinstance(cores, Mapping) else None
    if not isinstance(count, int) or isinstance(count, bool):
        return (f"{steady.CANNOT_TELL}: its row doesn't say which cores it ran on (`cores.count`, which {TIMER} writes "
                "from its own affinity)")
    if count != spec.cores:
        return f"{steady.FAIL}: it ran on {count} cores; the spec's `cores` is {spec.cores}"
    if own and (row.get("sessions"), row.get("warm")) != (item.sessions, item.row.warm):
        return (f"{steady.FAIL}: it timed {row.get('sessions')!r} sessions after {row.get('warm')!r} warm ones a prover; "
                f"the point's are {item.sessions} and {item.row.warm}")
    if own and row.get(steady.PERIOD) != item.row.period:
        return (f"{steady.CANNOT_TELL}: its row's `{steady.PERIOD}` ({row.get(steady.PERIOD)!r}) isn't the point's "
                f"({item.row.period!r}), which sizes its timed sessions and cuts its window")
    if own and row.get(VERSIONS) != versions(spec):
        return (f"{steady.CANNOT_TELL}: its row's `{VERSIONS}` ({row.get(VERSIONS)!r}) aren't the spec's driver and "
                "toolkit, read on the node")
    verdict = steady.verdict(row, files)
    return None if verdict.passed else f"{verdict.status}: {verdict.why}"
