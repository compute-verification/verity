"""The integration's typed configuration, and where its run definitions live.

* **Run definitions.**  ``manifests/`` (checkpoint pins, semantic profiles) and ``workloads/`` (row workloads) sit beside
  the package in the source tree; they are run definitions, not package data.  Library code locates them, and the source
  tree itself, through ``ROOT``, ``MANIFESTS``, ``WORKLOADS``, ``CHECKPOINTS`` and ``RECORDS``; a row id finds its file through
  :func:`row_file`, which also reads the directories ``$VERITY_WORKLOADS`` names.  A workload's bytes are a fixture in the
  evidence store: the tree keeps its name, its digest (``workloads/SHA256SUMS``) and its art id (``fixtures/artifacts.json``),
  and every reader takes the file through :func:`workload_file`, which fetches it by that id and checks the digest.
* **What runs.**  :class:`RowConfig`, a row's one config (its file ``workloads/<row>.json``: the workload and the pinned
  :class:`RowEngine`), the done source a row of record must declare (``DONE_SOURCES``, :func:`done_source_refusal`: its public
  cap), the row-id grammar (:class:`RowId`), the dtype vocabulary (``DTYPES``), the target table
  (``TARGETS``), the protocol-set grammar (``PROTOCOLS``, :func:`parse_protocols`), and :class:`EnginePin`, the pinned
  engine record every engine is built from: model, full revision and checkpoint role, checked against
  ``manifests/checkpoints.json``.
* **How a stage runs.**  :class:`BuildConfig`, :class:`MatchConfig` and :class:`CommitConfig`: defaults in code,
  overrides from flags or the environment of ``verity-vllm row``.
* **Where it runs.**  :class:`RunEnv`: interpreters, caches and roots.  ``pipeline/cli.py`` fills it from flags and the
  environment (the machine-path defaults live there); it is recorded as telemetry and never hashed.
* **Subcommand options.**  Every ``verity-vllm`` subcommand takes one frozen dataclass whose fields are declared with
  :func:`option`; ``pipeline/cli.py`` is the only parser.  A subcommand refuses a command line by raising
  :class:`UsageError`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any

from verity_vllm import settings
from verity_vllm.records import workload as workload_record

#: The source tree holding the ``verity_vllm`` package, ``manifests/`` and ``workloads/``.
ROOT = Path(str(resources.files("verity_vllm"))).resolve().parent
MANIFESTS = ROOT / "manifests"
WORKLOADS = ROOT / "workloads"
CHECKPOINTS = MANIFESTS / "checkpoints.json"
#: The circuit database's index (`pipeline.config_record`, `verity-vllm records`); the rows it lists are the rows of record.
RECORDS = MANIFESTS / "records.json"
#: The remote bandwidth session's processes (not a package), whose prover runs in the Commit's (`store.remote`).
REMOTE_SESSION = ROOT.parent.parent / "benchmarks" / "remote_session"
#: kv.cu's launchers over torch (``paged.py``, with ``word_codec/kv_bench.py``), the remote session's device encode.
VLLM_ENCODED = ROOT.parent.parent / "benchmarks" / "vllm_encoded"
#: The directory holding ``p2_v1``, P2's encoded weights store (``p2_v1/store.py``) and its CUDA library's wrapper, which
#: POUS encodes with (`protocols.pous`).
P2_POUS = ROOT.parent.parent / "benchmarks" / "pous"
#: The staging decode's directory (``stage.py``, loaded as the package ``p2_stage``), POUS's decode at use on the GPU.
P2_STAGE = ROOT.parent.parent / "benchmarks" / "torch_gpu" / "p2_stage"


def checkpoints_hf_home() -> str | None:
    """The HF cache the checkpoint pins name (``checkpoints.json`` ``hf_home``); None without the manifest."""
    try:
        return json.loads(CHECKPOINTS.read_text()).get("hf_home") or None
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------------------- options

class UsageError(ValueError):
    """A command line the subcommand refuses; ``pipeline/cli.py`` reports it as a usage error (exit 2)."""


def option(*flags: str, env: str | None = None, env_empty: bool = False, **kw: Any) -> Any:
    """One command-line option of a subcommand's options dataclass.

    ``flags`` and ``kw`` are ``argparse`` ``add_argument`` arguments, applied by ``pipeline/cli.py``, which also takes
    ``action="boolean_optional"`` (``argparse.BooleanOptionalAction``: ``--flag`` / ``--no-flag``).  ``kw["default"]``
    is also the field's default; a field without one is required (a required flag, or a positional).  ``env`` names the
    environment variable whose value the CLI takes as the default, converted by ``type`` like a flag value; an empty
    variable counts as unset unless ``env_empty``.
    """
    md = {"flags": flags, "kw": kw, "env": env, "env_empty": env_empty}
    if "default" not in kw:
        return field(metadata=md)
    d = kw["default"]
    if isinstance(d, (list, dict, set)):
        return field(default_factory=lambda: copy.copy(d), metadata=md)
    return field(default=d, metadata=md)


def group(cls: type) -> Any:
    """A nested options dataclass whose options join the subcommand's own (``verity-vllm row``'s stage configs)."""
    return field(default_factory=cls, metadata={"group": cls})


def environ_snapshot() -> Any:
    """The environment the CLI was started with, as a field: filled by ``pipeline/cli.py``, never by a flag."""
    return field(default_factory=dict, metadata={"environ": True})


# ---------------------------------------------------------------------------------------------------------------- what runs

#: Row-id dtype token -> the checkpoint weight format the row serves.
DTYPES: dict[str, str] = {"bf16": "bfloat16", "fp8": "float8_e4m3fn", "int8": "int8"}

#: Row-id target token -> the device class's compute capability.  Extend by name; an unknown token is a refusal, not a guess.
TARGETS: dict[str, tuple[int, int]] = {
    "l40s": (8, 9), "l40": (8, 9), "l4": (8, 9), "rtx4090": (8, 9),
    "a100": (8, 0),
    "a10": (8, 6), "a10g": (8, 6), "a40": (8, 6),
    "h100": (9, 0), "h200": (9, 0), "h20": (9, 0),
    "b200": (10, 0), "gb200": (10, 0),
    "rtxpro6000": (12, 0),
}

#: Row-id target token -> the device class's SM count, for the classes where it is fixed by the class: such a row declares its whole
#: Build target (capability + SM count) by its id, so its Build derives for it on any host, a GPU-free one included.  A class whose SM
#: count varies by part (h100 SXM 132 / PCIe 114) is not listed: its Build target is the host GPU's unless the workload declares one.
#: The H200 is 132 SMs as SXM and as NVL.
TARGET_NUM_SMS: dict[str, int] = {"rtxpro6000": 188, "h200": 132}

ROW_FIELDS = ("model", "dtype", "target", "world", "batch", "max_input", "max_output", "arrivals", "sampling", "execution")
_ROW_INT_PREFIX = {"world": "tp", "batch": "b", "max_input": "i", "max_output": "o"}
_TOKEN = re.compile(r"__(tp|b|i|o)(\d+)(?=__)")


@dataclass(frozen=True)
class RowId:
    """A sweep row id: ``<model>__<dtype>__<target>__tp<N>__b<N>__i<N>__o<N>__<arrivals>__<sampling>__<execution>``.

    ``str(row)`` gives the id back; ``RowId.parse`` refuses anything else.
    """

    model: str
    dtype: str
    target: str
    world: int
    batch: int
    max_input: int
    max_output: int
    arrivals: str
    sampling: str
    execution: str

    @classmethod
    def parse(cls, row: str) -> RowId:
        parts = str(row).split("__")
        if len(parts) != len(ROW_FIELDS) or not all(parts):
            raise ValueError(f"row id {row!r}: expected {len(ROW_FIELDS)} '__'-separated fields ({', '.join(ROW_FIELDS)})")
        vals: dict[str, Any] = {}
        for name, tok in zip(ROW_FIELDS, parts):
            pfx = _ROW_INT_PREFIX.get(name)
            if pfx is None:
                vals[name] = tok
            elif tok.startswith(pfx) and tok[len(pfx):].isdigit():
                vals[name] = int(tok[len(pfx):])
            else:
                raise ValueError(f"row id {row!r}: field {name} is {tok!r}, expected {pfx}<N>")
        return cls(**vals)

    def __str__(self) -> str:
        return "__".join((self.model, self.dtype, self.target, f"tp{self.world}", f"b{self.batch}", f"i{self.max_input}",
                          f"o{self.max_output}", self.arrivals, self.sampling, self.execution))


#: A row id inside text (a workload path, a log line): ten ``__``-separated fields, the four integer ones by their prefix.
ROW_ID_PATTERN = r"[\w\-]+__[\w\-]+__[\w\-]+__tp\d+__b\d+__i\d+__o\d+__\w+__\w+__[\w\-]+"


def row_tokens(row: str) -> list[str]:
    """A row id's ``__``-separated fields as written, unvalidated (`RowId.parse` validates them)."""
    return row.split("__")


def row_int(row: str, prefix: str, default: int = 1) -> int:
    """The ``__<prefix><N>__`` field of a row id (``tp``, ``b``, ``i``, ``o``), ``default`` when the id has none."""
    for m in _TOKEN.finditer(str(row)):
        if m.group(1) == prefix:
            return int(m.group(2))
    return default


def row_world(row: str) -> int:
    """The tensor-parallel world a row id declares (``__tp<N>__``), 1 when absent."""
    return row_int(row, "tp")


def row_batch(row: str) -> int:
    """The batch a row id declares (``__b<N>__``), 1 when absent."""
    return row_int(row, "b")


def row_execution(row: str) -> str | None:
    """The last ``__`` field of a row id (its execution label), None for an id without one."""
    return str(row).rsplit("__", 1)[1] if "__" in str(row) else None


def row_target(row: str) -> str | None:
    """The device token of a row id (its third field when the fourth is ``tp<N>``), None when the id lacks the grammar."""
    parts = str(row).split("__")
    return parts[2] if len(parts) >= 4 and parts[3].startswith("tp") else None


#: The protocols a run can enable (`verity_vllm.protocols`), in install order, innermost first: PoUW replaces the
#: linear layers' GEMMs, POUS decodes the weights those GEMMs read, sampled proofs commits what the model computed.
PROTOCOLS = ("pouw", "pous", "sampled-proofs")
#: The set a row runs when its target declares none: sampled proofs alone, the Commit under its committer's scheme
#: (rows-sha512 by default, `pipeline.commit --commit-scheme`).
DEFAULT_ROW_PROTOCOLS = ("sampled-proofs:rows-sha512",)
_SCHEME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+-]*")


def parse_protocols(value: str | Mapping[str, str] | Any | None) -> tuple[str, ...]:
    """A protocol set in its one spelling: ``name:scheme`` entries, one per protocol, in `PROTOCOLS` order.

    ``value`` is None, ``""`` or ``"none"`` (the empty set), a comma-separated string, an iterable of entries, or a mapping
    ``{name: scheme}``.  Refuses (ValueError) an unknown protocol, an entry without a scheme, a malformed scheme and two
    entries of one protocol.  Whether a scheme is registered is the protocol package's to say
    (`verity_vllm.protocols.refusals`)."""
    if value is None or (isinstance(value, str) and value.strip() in ("", "none")):
        return ()
    if isinstance(value, Mapping):
        items = [f"{k}:{v}" for k, v in value.items()]
    elif isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    else:
        items = [str(x).strip() for x in value]
    got: dict[str, str] = {}
    for item in items:
        name, sep, scheme = item.partition(":")
        if name not in PROTOCOLS:
            raise ValueError(f"protocol entry {item!r}: unknown protocol {name!r} (known: {', '.join(PROTOCOLS)})")
        if not sep or not _SCHEME.fullmatch(scheme):
            raise ValueError(f"protocol entry {item!r}: expected {name}:<scheme>, the scheme a name its protocol registers")
        if name in got:
            raise ValueError(f"protocol {name!r} is enabled twice ({got[name]!r} and {scheme!r}); a run enables one scheme per protocol")
        got[name] = scheme
    return tuple(f"{n}:{got[n]}" for n in PROTOCOLS if n in got)


#: pins kept beside a checkpoint manifest in `<stem>-<name>.json`, same schema and append-only too (the int8 recipe pins: the
#: manifest itself sits at the repository's 256 KiB file cap); `checkpoint_entries` reads them after the manifest's own pins
SIBLING_MANIFESTS = ("int8",)


def sibling_manifests(checkpoints: str | Path = CHECKPOINTS) -> list[Path]:
    p = Path(checkpoints)
    return [q for n in SIBLING_MANIFESTS if (q := p.with_name(f"{p.stem}-{n}.json")).is_file()]


def checkpoint_entries(checkpoints: str | Path = CHECKPOINTS) -> list[dict[str, Any]]:
    """Every pin of the manifest `checkpoints` and of its sibling manifests (`SIBLING_MANIFESTS`)."""
    out = list(json.loads(Path(checkpoints).read_text())["checkpoints"])
    for q in sibling_manifests(checkpoints):
        out += json.loads(q.read_text())["checkpoints"]
    return out


@dataclass(frozen=True)
class EnginePin:
    """The pinned engine a row or :class:`verity_vllm.LLM` runs.

    ``model`` is the HF repo id and ``revision`` its full commit sha, both as ``manifests/checkpoints.json`` pins them;
    ``case`` is the pin's role (the engine profile's key) and ``local_path`` the snapshot on this host.  ``world`` is the
    tensor-parallel degree.  Build it with :meth:`resolve`, which refuses an unpinned model or another revision.
    """

    model: str
    revision: str
    case: str
    local_path: str | None
    world: int = 1

    @classmethod
    def resolve(cls, model: str, revision: str, *, world: int = 1, checkpoints: str | Path = CHECKPOINTS) -> EnginePin:
        entries = [c for c in checkpoint_entries(checkpoints) if c.get("repo") == model]
        if not entries:
            pinned = sorted({str(c.get("repo")) for c in checkpoint_entries(checkpoints)})
            raise ValueError(f"model {model!r} is not pinned in {Path(checkpoints).name} (pinned: {', '.join(pinned)})")
        hits = [c for c in entries if str(c.get("revision", "")) == str(revision)]
        if not hits:
            revs = sorted({str(c.get("revision")) for c in entries})
            raise ValueError(f"model {model!r} revision {revision!r} is not the pinned revision ({', '.join(revs)}); "
                             f"pass the full commit sha")
        # a repo can be pinned under several roles; the downloaded one is the engine's
        hit = next((c for c in hits if c.get("downloaded") and c.get("local_path")), hits[0])
        return cls(model=model, revision=str(hit["revision"]), case=str(hit.get("role")), local_path=hit.get("local_path"),
                   world=int(world))


def checkpoint_dir(model: str, revision: str, hf_home: str, checkpoints: str | Path = CHECKPOINTS) -> str | None:
    """The local snapshot of a workload's (model, revision): the first pin whose revision starts with ``revision[:10]``
    and whose ``local_path`` exists, else the HF hub snapshot under ``hf_home`` of that revision prefix; None when absent."""
    local = None
    for c in checkpoint_entries(checkpoints):
        if c.get("repo") == model and str(c.get("revision", "")).startswith(revision[:10]):
            local = c.get("local_path")
            break
    if not local or not os.path.exists(local):
        base = f"{hf_home}/hub/models--{model.replace('/', '--')}/snapshots"
        for d in sorted(os.listdir(base)) if os.path.exists(base) else []:
            if d.startswith(revision[:10]):
                local = f"{base}/{d}"
    return local


#: The vLLM commit this tree derives and serves on: ``vllm.__version__`` must carry ``+g<VLLM>`` (``TargetProfile.vllm_revision``).
VLLM = "d9105ea80"

#: A row file (``workloads/<row>.json``): the workload and its pinned ``engine`` block (:class:`RowConfig`), the typed record
#: ``records.workload.Workload``.
WORKLOAD_SCHEMA = workload_record.SCHEMA


#: The repository root: its ``fixtures/artifacts.json`` registers each stored workload file by its art id.
REPO = ROOT.parent.parent
#: A workload directory's digests, ``<sha256>  <name>`` a line (``sha256sum``'s format): each file it lists is a fixture in the
#: evidence store, gitignored here, and every reader takes it through :func:`workload_file`.
DIGESTS = "SHA256SUMS"


def stored_workloads(d: Path = WORKLOADS) -> dict[str, str]:
    """{file name: SHA-256} of the workloads the directory ``d`` keeps in the store (its ``SHA256SUMS``); {} without one."""
    p = Path(d) / DIGESTS
    if not p.is_file():
        return {}
    pairs = [line.split(None, 1) for line in p.read_text().splitlines() if line.strip()]
    return {name.strip().lstrip("*"): h for h, name in pairs}


def _fetch(paths: list[Path]) -> None:
    """Fetch ``paths`` with ``research data fetch-fixtures`` in the repository holding each (the nearest ancestor with a registry),
    by that tree's own ``research``."""
    by_repo: dict[Path, list[Path]] = {}
    for p in paths:
        repo = next((d for d in p.resolve().parents if (d / "fixtures" / "artifacts.json").is_file()), REPO)
        by_repo.setdefault(repo, []).append(p)
    for repo, ps in by_repo.items():
        rels = [os.path.relpath(p.resolve(), repo) for p in ps]
        env = settings.environ()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo / "tools" / "research" / "src"), env.get("PYTHONPATH")]))
        r = subprocess.run([sys.executable, "-m", "research", "data", "fetch-fixtures", "--repo", str(repo), "--json", *rels],
                           capture_output=True, text=True, env=env)
        if missing := [rel for p, rel in zip(ps, rels) if not p.is_file()]:
            raise RuntimeError(f"{', '.join(missing)}: stored workloads, not here, and `research data fetch-fixtures` could not fetch "
                               f"them (rc {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}); fetch them where the evidence store "
                               "is reachable, or name the public export with $RESEARCH_PUBLIC_URL")


def workload_file(path: str | Path) -> Path:
    """``path``, made present and checked when its directory's ``SHA256SUMS`` lists it: an absent file is fetched by the art id
    ``fixtures/artifacts.json`` gives its path (``research data fetch-fixtures``), and a file whose SHA-256 is not the listed one is
    refused.  A file its directory doesn't list is returned as it is."""
    p = Path(path)
    want = stored_workloads(p.parent).get(p.name)
    if want is None:
        return p
    if not p.is_file():
        _fetch([p])
    if (got := hashlib.sha256(p.read_bytes()).hexdigest()) != want:
        raise RuntimeError(f"{p}: SHA-256 {got} is not the {want} its {DIGESTS} lists: a changed workload is a new fixture "
                           "(`research data put --kind fixture/v1 --file PATH --preserve`, its id in fixtures/artifacts.json, "
                           f"its digest in {DIGESTS}); `research data fetch-fixtures PATH` restores the stored one")
    return p


def workload_arg(s: str) -> str:
    """The ``type`` of an option naming a workload file: ``s`` as given, its file made present and checked (:func:`workload_file`;
    its refusal a :class:`UsageError`)."""
    try:
        workload_file(s)
    except RuntimeError as e:
        raise UsageError(str(e)) from e
    return s


def workload_files(d: Path = WORKLOADS) -> list[Path]:
    """Every workload of the directory ``d``, its stored ones (``SHA256SUMS``) and any other ``*.json`` in it, each through
    :func:`workload_file`; the absent stored ones are fetched in one call."""
    names = sorted({*stored_workloads(d), *(p.name for p in Path(d).glob("*.json"))})
    if absent := [p for n in names if not (p := Path(d) / n).is_file()]:
        _fetch(absent)
    return [workload_file(Path(d) / n) for n in names]


def row_file(row: str) -> Path:
    """Row id ``row``'s file: ``<dir>/<row>.json`` in ``WORKLOADS`` or in a directory ``$VERITY_WORKLOADS`` names (rows that live
    outside the integration, which it finds only there), else ``WORKLOADS/<row>.json``; a stored one is fetched and checked
    (:func:`workload_file`).  A row id with a file in two is refused."""
    dirs = (WORKLOADS, *map(Path, settings.VERITY_WORKLOADS))
    found = list(dict.fromkeys(p.resolve() for d in dirs if (p := d / f"{row}.json").is_file() or p.name in stored_workloads(d)))
    if len(found) > 1:
        raise UsageError(f"row {row} has a file in more than one of workloads/ and $VERITY_WORKLOADS: {', '.join(map(str, found))}")
    return workload_file(found[0]) if found else WORKLOADS / f"{row}.json"


def row_path(row: str) -> str:
    """:func:`row_file` from ``ROOT``, where a row's Build, Match and Commit run: ``workloads/<row>.json`` for a row of the
    integration's own."""
    return os.path.relpath(row_file(row), ROOT)

#: How a row's requests are done: at their public cap (``sampling.ignore_eos``: every request runs its ``max_tokens`` steps), or at an
#: EOS the model samples before it.  A row of record is done only at its public cap, so its served length is
#: public; the EOS a request samples there is a value, which decides no step.
DONE_SOURCES = ("public-cap", "eos-token")


def rows_of_record() -> frozenset[str]:
    """The row ids ``RECORDS`` lists (none without the file)."""
    return frozenset(json.loads(RECORDS.read_text()).get("rows") or {}) if RECORDS.is_file() else frozenset()


def done_source(workload: Mapping[str, Any]) -> str:
    """The done source (``DONE_SOURCES``) a workload doc declares."""
    s = workload.get("sampling")
    return "public-cap" if isinstance(s, Mapping) and s.get("ignore_eos") is True else "eos-token"


def done_source_refusal(workload: Mapping[str, Any], *, of_record: bool = False) -> str | None:
    """Why a row of record (its ``sweep.row_id`` in ``RECORDS``, or any row with ``of_record``: one about to be indexed) is not done
    only at its public cap; None for a row that is, and for every research row (its EOS stops are its own).  Row files
    (:meth:`RowConfig.of`), the row precheck and `pipeline.global_program` refuse with it, and so does whatever indexes a row
    (``pipeline.config_record.index_refusal``: ``records publish``, ``pull`` and ``switch``, and ``row chain`` when it publishes)."""
    sw = workload.get("sweep")
    row = sw.get("row_id") if isinstance(sw, Mapping) else None
    if not of_record and row not in rows_of_record():
        return None
    s = workload.get("sampling") if isinstance(workload.get("sampling"), Mapping) else {}
    if done_source(workload) != "public-cap":
        why = (f"its done source is 'eos-token' (sampling.ignore_eos is {s.get('ignore_eos')!r}): declare sampling.ignore_eos: true, "
               f"so every request runs its public max_tokens")
    elif stops := [k for k in ("stop_token_ids", "stop") if s.get(k)]:
        why = f"sampling.{stops[0]} {s[stops[0]]!r} stops a request before its cap: drop it"
    elif uncapped := [str(r.get("request_id", i)) for i, r in enumerate(workload.get("requests") or [])
                      if isinstance(r.get("max_tokens"), bool) or not isinstance(r.get("max_tokens"), int) or r["max_tokens"] < 1]:
        why = f"request(s) {', '.join(uncapped[:4])} declare no public cap (a positive integer max_tokens)"
    else:
        return None
    return f"{row}: a row of record ({RECORDS.name}) is done only at its public cap, and {why}"


_SHA = re.compile(r"[0-9a-f]{40}")
#: The flat v1 keys the ``engine`` block replaced, by their place in it.
_V1_KEYS = {"engine_args_required": "args", "env_required": "env", "target": "target", "tokenizer_revision": "revision",
            "checkpoint_revision": "revision"}


@dataclass(frozen=True)
class RowEngine:
    """What, besides its requests, determines a row's circuit and its run: the checkpoint's full ``revision`` (its repo is the row's
    ``model``), the ``vllm`` commit, the engine ``args``, the ``env`` its engines run under (kernel selection: ``VLLM_BATCH_INVARIANT``,
    ``NCCL_P2P_DISABLE``), the declared ``target`` (``TargetProfile`` fields; None: the device's) and ``reasons``: why the row sets a
    switch away from vLLM's default, by the switch's name (``engine.served.DEFAULTS``; a served Commit refuses one without).  The κ
    profile a Match folds with is the role's (``observe.fold.profiles.generic.module_name``), never a field."""

    revision: str | None = None
    vllm: str = VLLM
    args: Mapping[str, Any] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    target: Mapping[str, Any] | None = None
    reasons: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def of(cls, doc: Mapping[str, Any]) -> RowEngine:
        """The ``engine`` block of a row file (or any workload doc); refuses the flat v1 keys and another vLLM."""
        stale = sorted(k for k in _V1_KEYS if k in doc)
        if stale:
            raise ValueError(f"workload keys {', '.join(stale)} are the v1 layout: regenerate the file (`verity-vllm workload`) or move "
                             f"them into its `engine` block ({', '.join(f'{k} -> engine.{_V1_KEYS[k]}' for k in stale)})")
        e = doc.get("engine") or {}
        unknown = sorted(set(e) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"engine block: unknown field(s) {', '.join(unknown)} (known: {', '.join(f.name for f in fields(cls))})")
        eng = cls(revision=e.get("revision"), vllm=str(e.get("vllm", VLLM)), args=dict(e.get("args") or {}),
                  env={str(k): str(v) for k, v in (e.get("env") or {}).items()}, target=dict(e["target"]) if e.get("target") else None,
                  reasons={str(k): str(v) for k, v in (e.get("reasons") or {}).items()})
        if eng.vllm != VLLM:
            raise ValueError(f"engine.vllm {eng.vllm!r}: this tree derives and serves on vLLM {VLLM} only (config.VLLM)")
        if eng.target and eng.target.get("vllm_revision", eng.vllm) != eng.vllm:
            raise ValueError(f"engine.target.vllm_revision {eng.target['vllm_revision']!r} disagrees with engine.vllm {eng.vllm!r}")
        return eng

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"revision": self.revision, "vllm": self.vllm, "args": dict(self.args)}
        if self.env:
            out["env"] = dict(self.env)
        if self.target:
            out["target"] = dict(self.target)
        if self.reasons:
            out["reasons"] = dict(self.reasons)
        return out


@dataclass(frozen=True)
class RowConfig:
    """A row's one config, its file ``workloads/<row>.json``: the checkpoint's repo (``model``) and role (``case``), the pinned
    :class:`RowEngine` and the workload (everything else in the file: requests, sampling, the sweep's parameters).

    The flags and environment variables of ``verity-vllm row`` override how its stages run, never this; what a run used is its
    effective config (``pipeline.row.effective_config``, ``<row dir>/effective_config.json``)."""

    model: str
    case: str | None
    engine: RowEngine
    workload: Mapping[str, Any]

    @classmethod
    def of(cls, doc: Mapping[str, Any]) -> RowConfig:
        if doc.get("schema") != WORKLOAD_SCHEMA:
            raise ValueError(f"workload schema {doc.get('schema')!r}, expected {WORKLOAD_SCHEMA!r}")
        rest = {k: v for k, v in doc.items() if k not in ("schema", "model", "case", "engine")}
        cfg = cls(model=str(doc["model"]), case=doc.get("case"), engine=RowEngine.of(doc), workload=rest)
        if cfg.engine.revision is not None and not _SHA.fullmatch(cfg.engine.revision):
            raise ValueError(f"engine.revision {cfg.engine.revision!r}: the checkpoint's full commit sha (manifests/checkpoints.json)")
        if cfg.row and row_world(cfg.row) > 1 and not isinstance(cfg.engine.args.get("disable_custom_all_reduce"), bool):
            raise ValueError(f"{cfg.row}: a tensor-parallel row states engine.args.disable_custom_all_reduce (true on a SYS box, "
                             f"`engine.profiles.transport.row_custom_all_reduce_off`)")
        if why := done_source_refusal(doc):
            raise ValueError(why)
        return cfg

    @classmethod
    def load(cls, row_or_path: str | Path) -> RowConfig:
        """A row's config by row id (:func:`row_file`) or by file (:func:`workload_file`)."""
        p = Path(row_or_path)
        return cls.of(json.loads((workload_file(p) if p.suffix == ".json" else row_file(str(row_or_path))).read_text()))

    @property
    def row(self) -> str | None:
        """The row id the requests were sampled for (``sweep.row_id``); a variant file (``olmoe-1b-7b-padded__…``) keeps its base row's."""
        sw = self.workload.get("sweep")
        return sw.get("row_id") if isinstance(sw, Mapping) else None

    def pin(self, world: int = 1, checkpoints: str | Path = CHECKPOINTS) -> EnginePin:
        if not self.engine.revision:
            raise ValueError(f"{self.row or self.model}: the row pins no checkpoint revision (engine.revision)")
        return EnginePin.resolve(self.model, self.engine.revision, world=world, checkpoints=checkpoints)

    def to_json(self) -> dict[str, Any]:
        return {"schema": WORKLOAD_SCHEMA, "model": self.model, **({"case": self.case} if self.case is not None else {}),
                "engine": self.engine.to_json(), **self.workload}


# ---------------------------------------------------------------------------------------------------------------- stage configs

@dataclass(frozen=True, kw_only=True)
class BuildConfig:
    """How the Build stage of ``verity-vllm row`` runs.  None = the row's derived default."""

    build_jobs: str | None = option("--build-jobs", env="BUILD_JOBS", default=None,
                                    help="concurrent derives: N, or auto (default): 0.8 x the memory headroom (or --build-ram-budget-gb, if "
                                         "smaller) over the largest shape's planned derive peak, at most one per CPU and one per derive "
                                         "(TP: ranks x (shapes + 1); its per-rank composes are sized from memory too)")
    build_ram_budget_gb: float | None = option("--build-ram-budget-gb", env="BUILD_RAM_BUDGET_GB", type=float, default=None,
                                               help="auto's memory budget when the host's headroom is shared (a sweep's per-config share)")
    build_timeout: int | None = option("--build-timeout", env="BUILD_TIMEOUT", type=int, default=None,
                                       help="seconds for one derive (and for one workload Program compose; TP: one rank's global Program "
                                            "and the call-boundary check); default scales with the envelope (TP: 7200).  Never the TP "
                                            "manifest, which --manifest-timeout bounds")
    manifest_timeout: int | None = option("--manifest-timeout", env="MANIFEST_TIMEOUT", type=int, default=None,
                                          help="TP: seconds for the required-value manifest over the W ranks, the Build's and a Commit's "
                                               "rebuild (default 4 h + 1200 s a component build, W x declared requests: "
                                               "row_records.manifest_timeout; 0 = no limit).  manifest.log reports each component built")
    lp: int | None = option("--lp", env="LP", type=int, default=None,
                            help="request wrapper prompt length (default: the longest prompt; a stochastic row: r0's)")
    t: int | None = option("--t", env="T", type=int, default=None,
                           help="request wrapper decode steps (default: the largest cap - 1; a stochastic row: r0's)")
    program_cache: str | None = option("--program-cache", env="PROGRAM_CACHE", default=None,
                                       help="a directory of derived Programs shared by the host's runs (pipeline.program_cache): each derive "
                                            "restores a cached entry of the same key or publishes its own; default off")
    build_relabel: str = option("--build-relabel", env="BUILD_RELABEL", default="off", choices=("off", "on"),
                                help="TP: one template rank's request derives write the other ranks' (pipeline.tp.relabel_build), behind a "
                                     "step guard over every rank and a canary request derived in full; default off")
    build_programs_only: str = option("--build-programs-only", env="BUILD_PROGRAMS_ONLY", default="0",
                                      help="TP: 1 = the Build ends at the rank and global Programs: no required-value manifest or call-boundary "
                                           "check (the Commit builds the manifest it runs over); `verity-vllm neutrality`'s heavy tier")
    build_construction: str = option("--build-construction", env="BUILD_CONSTRUCTION", default="record", choices=("record", "trace"),
                                     help="TP: record = the row's construction of record, the AOT Build for a switched row (its `aot` in "
                                          "manifests/records.json, verity_vllm.program.predict.aot) and the trace for any other; trace = the trace "
                                          "(`verity-vllm neutrality`'s heavy tier)")
    aot_trace: str | None = option("--aot-trace", env="AOT_TRACE", default=None,
                                   help="TP, a switched row: the dir holding its `aot.trace` Build (build/rank<r>/build_*), e.g. a job's "
                                        "--send of it; default fetched from the store into <row dir>/aot_trace")


@dataclass(frozen=True, kw_only=True)
class MatchConfig:
    """How the Match stage of ``verity-vllm row`` runs.  None = the row's derived default."""

    match_impl: str = option("--match-impl", env="MATCH_IMPL", default="fast", help="global match checker: fast or record (the audit)")
    match_pipeline: str = option("--match-pipeline", env="MATCH_PIPELINE", default="shared",
                                 help="shared = one batch decomposition inside the global match (a TP row's: per rank)")
    match_phase: str = option("--match-phase", env="MATCH_PHASE", default="all", choices=("all", "gpu", "cpu"),
                              help="cpu = resume the CPU half over the capture already on disk; gpu (TP) = the capture and control alone, "
                                   "which need no Build, then the row stops")
    snapshot_steps: str | None = option("--snapshot-steps", env="MATCH_SNAP_STEPS", default=None,
                                        help="0,1 | all; default 0,1 greedy, all stochastic or when form (B) must be complete")
    snapshot_max_bytes: str | None = option("--snapshot-max-bytes", env="MATCH_SNAP_MAX_BYTES", default=None,
                                            help="explicit per-tensor snapshot cap (default: derived by the Match)")
    within_step: str = option("--within-step", env="WITHIN_STEP", default="dag", help="within-step criterion: dag or sequence")
    instances_form: str = option("--instances-form", env="VERITY_INSTANCES_FORM", default="progressions",
                                 help="stored form of the fold's instances.jsonl: progressions or runs")
    acquire_manifest: str | None = option("--acquire-manifest", env="ACQUIRE_MANIFEST", default=None,
                                          help="required-value manifest for the capture (default: the Build's manifest.json)")
    match_timeout: int | None = option("--match-timeout", env="MATCH_TIMEOUT", type=int, default=None,
                                       help="seconds for the capture + fold (default scales with B and the envelope; TP: 3600)")
    compare_timeout: int | None = option("--compare-timeout", env="COMPARE_TIMEOUT", type=int, default=None,
                                         help="seconds per comparator (default scales with B and the envelope)")
    fold_timeout: int | None = option("--fold-timeout", env="FOLD_TIMEOUT", type=int, default=None,
                                      help="TP: seconds for each leg of the per-rank fold Match (fold, compare, the global match; default "
                                           "350 s a GiB of a rank's summed request derive peaks, at least 7200: row_records.fold_plan); "
                                           "tp-fold-match as a whole gets the folds' rounds, the compare and global-match rounds and the "
                                           "cross-rank leg (row_records.fold_phase_s: four legs when every rank runs at once)")
    fold_mem_gb: float | None = option("--fold-mem-gb", env="RESEARCH_RUN_MEM_GB", type=float, default=None,
                                       help="TP: the memory (GB) the fold Match's concurrent legs are sized to (default: research run's "
                                            "--mem-gb, RESEARCH_RUN_MEM_GB; without one, the host's headroom); one that holds no planned "
                                            "global-match leg beside one fold runs one rank at a time, with a warning in row.log")
    fold_rank_workers: int = option("--fold-rank-workers", env="FOLD_RANK_WORKERS", type=int, default=0,
                                    help="TP: ranks whose compare and global-match legs run at once (0 = as many as --fold-mem-gb holds at "
                                         "a global-match leg's planned peak beside one fold, 5.5x a rank's summed request derive peaks: "
                                         "GLM-5.3 B8 I256's 339 GiB; every rank when the checkpoint's layers are unknown)")
    fold_workers: int = option("--fold-workers", env="FOLD_WORKERS", type=int, default=0,
                               help="TP: ranks whose fold leg runs at once (0 = as many as the memory those global-match legs leave holds "
                                    "at a fold's planned peak, 0.6x the same sum: B8 I256's 40 GB; as --fold-rank-workers when it is given "
                                    "or the layers are unknown)")
    match_reuse: str = option("--match-reuse", env="MATCH_REUSE", default="off", choices=("off", "fold", "legs"),
                              help="TP, --match-phase cpu, rerunning a CPU half that stopped part way under the same code: fold = keep each "
                                   "rank's persisted fold of the capture in place; legs = also each rank's written compare and global-match "
                                   "documents, so only the missing legs and the cross-rank leg run")
    fold_profile: str | None = option("--fold-profile", env="FOLD_PROFILE", env_empty=True, default=None,
                                      help="TP: rank profile module (default derived_<ROLE>_tp<W>; empty = collective-level Match only)")


@dataclass(frozen=True, kw_only=True)
class CommitConfig:
    """How the Commit stage of ``verity-vllm row`` runs.  None = the row's derived default."""

    pairs: int = option("--pairs", env="PAIRS", type=int, default=3, help="instrumented/control pairs")
    config_run: str = option("--config-run", env="CONFIG_RUN", default="0",
                             help="1 = the config run (docs/vllm-config-sweep-plan.md §2): Build and the strict word check, ONE instrumented "
                                  "Commit (no control arm, no pairs) under bounded staging, a k-unit replay (--replay-draw), then "
                                  "config_record.json; no Match, no manifest-verify rebuild, no VU export")
    replay_k: int = option("--replay-k", env="REPLAY_K", type=int, default=256, help="config run: the replay's k units")
    replay_deferred: str = option("--replay-deferred", env="REPLAY_DEFERRED", default="0",
                                  help="config run (any run of a TP row; at a world other than TP2 and TP8 its Commit replays in process all the "
                                       "same, `tp.c2.DEFER_WORLDS`): 1 = the Commit "
                                       "writes a replay bundle (per rank) and exits without "
                                       "the replay; the verdict stays pending until `row stage replay` (CPU) replays the bundle; slim = a "
                                       "slim bundle (single-GPU)")
    config_baseline: str = option("--config-baseline", env="CONFIG_BASELINE", default="0",
                                  help="config run: 1 = the Commit also runs the uninstrumented control arm (one pair), the baseline of "
                                       "the record's extraction slowdown")
    replay_draw: str = option("--replay-draw", env="REPLAY_DRAW", default="uniform", choices=("uniform", "family"),
                              help="config run: the k units sampled uniformly from every unit (k/k equal bounds the wrong fraction) or "
                                   "stratified by family (each family at least once, the rest split evenly)")
    commit_gpu_util: str | None = option("--commit-gpu-util", env="COMMIT_GPU_UTIL", default=None,
                                         help="TP: gpu_memory_utilization of the Commit engine (default --gpu-util)")
    window_mb: int = option("--window-mb", env="WINDOW_MB", type=int, default=256, help="device staging window (0 = per-step blocks)")
    window_slots: int = option("--window-slots", env="WINDOW_SLOTS", type=int, default=8)
    retain: str = option("--retain", env="RETAIN", default="host", help="where raw copies are retained: host, device or gpu")
    open_after_release: str = option("--open-after-release", env="OPEN_AFTER_RELEASE", default="1", help="0 opts out")
    tap_cap_mb: int = option("--tap-cap-mb", env="TAP_CAP_MB", type=int, default=2048, help="FA2 tap residency cap (0 = unbounded)")
    replay_cache: str = option("--replay-cache", env="REPLAY_CACHE", default="1", help="0 = every pair replays")
    required_classes: str | None = option("--required-classes", env="REQUIRED_CLASSES", default=None)
    required_manifest: str | None = option("--required-manifest", env="REQUIRED_MANIFEST", default=None,
                                           help="an explicit manifest (default: the row's own manifest.json)")
    program_digest: str | None = option("--program-digest", env="PROGRAM_DIGEST", default=None,
                                        help="explicit Program digest (default: read from build_summary.json)")
    extra_engine_args: str | None = option("--extra-engine-args", env="EXTRA_ENGINE_ARGS", default=None, help='"k=v k=v"')
    extra_commit_args: str | None = option("--commit-args", env="EXTRA_COMMIT_ARGS", default=None, help="extra Commit flags, verbatim")
    commit_args: str | None = option("--commit-args-tail", env="COMMIT_ARGS", default=None,
                                     help="extra Commit flags appended after --program-digest, verbatim")
    replay_args: str | None = option("--replay-args", env="REPLAY_ARGS", default=None, help="extra replay flags, verbatim")
    replay_workers: int = option("--replay-workers", env="REPLAY_WORKERS", type=int, default=32, help="TP: replay workers")
    hot_engine: str = option("--hot-engine", env="HOT_ENGINE", default="0", help="1 = run the Commit on the pod's hot worker")
    group: str = option("--commit-group", env="COMMIT_GROUP", default="", metavar="MEMBERS_JSON",
                        help="config run: the rows of one engine (this row included) committed in one process under one GPU lease, and "
                             "replayed by this row's replay stage (verity_vllm.pipeline.row_group)")
    commit_timeout: int | None = option("--commit-timeout", env="COMMIT_TIMEOUT", type=int, default=None,
                                        help="seconds (default scales with B and the envelope; TP: 7200)")
    vu_export: str = option("--vu-export", env="VU_EXPORT", default="1",
                            help="0 opts out: the Commit exports a sample of the row's VUs as instance sets (verity_vllm.pipeline.vu_export)")
    serving_rows: str | None = option("--serving-rows", env="SERVING_ROWS", default=None, metavar="PARTITION_JSON",
                                      help="opt in, never the record: the Commit also commits every unit of this partition file's templates "
                                           "in M0's serving row format (verity_vllm.pipeline.serving_rows), fixed before serving; a "
                                           "deferred Commit (REPLAY_DEFERRED=1; slim refused) commits them at its C2 site, before its "
                                           "bundle is sealed (c2_replay.serving_rows); a TP row's, one window per "
                                           "rank, from tp-commit itself (verity_vllm.pipeline.tp.serving_rows; needs REPLAY_DEFERRED=1)")
    remote_verifier: str | None = option("--remote-verifier", env="REMOTE_VERIFIER", default=None, metavar="HOST:PORT",
                                         help="the remote bandwidth session's verifier: the Commit proves its store to it in lockstep "
                                              "with its first instrumented run (verity_vllm.protocols.remote_bandwidth.remote; single-GPU rows)")
    norm_tap: str = option("--norm-tap", env="NORM_TAP", default="1",
                           help="0 opts out, for A/B only: by default the row commits one scale per norm Call row (verity_vllm.query."
                                "norm_scales) through the norm-scale tap (acquire.sources.norm_scale_source) when its manifest names norm scales")
    guarded_max_tap: str = option("--guarded-max-tap", env="GUARDED_MAX_TAP", default="1",
                                  help="0 opts out, for A/B only: by default the attention stream's ROW word 3 carries GuardNegInfZero(m) after a "
                                       "row's first key block and its MS class every block's max * scale (verity_vllm.query.guarded_max), written "
                                       "by the guarded-max FA2/FA3 tap build")
    router_tap: str = option("--router-tap", env="ROUTER_TAP", default="1",
                             help="0 opts out, for A/B only: by default the row commits the selection-order router's softmax per token row "
                                  "(verity_vllm.query.router_softmax) through the router-softmax tap (acquire.sources.router_softmax_source) "
                                  "when its manifest names MoE routers")
    vocab_tap: str = option("--vocab-tap", env="VOCAB_TAP", default="1",
                            help="0 opts out, for A/B only: by default each TP rank commits its vocabulary-range values per token row "
                                 "(verity_vllm.query.vocab_range) through the vocabulary-range source (acquire.sources.vocab_range_source) "
                                 "when its manifest names vocabulary ranges")


# ---------------------------------------------------------------------------------------------------------------- where it runs

@dataclass(frozen=True, kw_only=True)
class RunEnv:
    """Where a row runs: interpreters, caches and roots.  Filled by ``pipeline/cli.py`` from flags, the environment and
    the pod layout's machine-path defaults (``cli.MACHINE``); recorded as telemetry, never hashed.

    ``environ`` is the environment the CLI was started with; child processes get it plus the pins the driver sets.
    """

    py: str = option("--py", env="PY", help="the stages' interpreter")
    py312: str | None = option("--py312", env="PY312", default=None, help="the CPU stages' interpreter (default: --py)")
    sweep_dir: str = option("--sweep-dir", env="SWEEP_DIR", help="rows live in <sweep-dir>/<row>")
    hf_home: str = option("--hf-home", env="HF_HOME")
    cuda_home: str = option("--cuda-home", env="CUDA_HOME")
    cuda_bin: str = option("--cuda-bin", help="put on the stages' PATH after the interpreter's directory")
    native_collect_build: str | None = option("--native-collect-build", env="NATIVE_COLLECT_BUILD", default=None,
                                              help="collector build dir (default: the release's, else --native-collect-default)")
    native_collect_default: str = option("--native-collect-default")
    hidden_so: str | None = option("--hidden-so", env="HIDDEN_SO", default=None,
                                   help="M1 tap .so (default: the one of the row's attention implementation)")
    hidden_so_fa2: str = option("--hidden-so-fa2")
    hidden_so_fa3: str = option("--hidden-so-fa3")
    hidden_so_fa2_guarded: str = option("--hidden-so-fa2-guarded", help="the guarded-max FA2 tap build (ops/pod_fa2_tap.sh, FA2_TAP_ROW_GUARD=1), used unless --guarded-max-tap 0")
    hidden_so_fa3_guarded: str = option("--hidden-so-fa3-guarded", help="the guarded-max FA3 tap build (ops/pod_fa3_tap.sh, FA3_TAP_ROW_GUARD=1), used unless --guarded-max-tap 0")
    norm_tap_so: str = option("--norm-tap-so", help="the norm-scale tap op (ops/pod_norm_tap.sh), used unless --norm-tap 0")
    router_tap_so: str = option("--router-tap-so", help="the router-softmax tap op (ops/pod_router_tap.sh), used unless --router-tap 0")
    hot_root: str = option("--hot-root", env="HOT_ROOT")
    hot_idle: str = option("--hot-idle", env="HOT_IDLE", default="900")
    timeline: str | None = option("--timeline", env="VERITY_TIMELINE", default=None,
                                  help="stage telemetry file (default: <row dir>/timeline.jsonl)")
    verity_tree: str | None = option("--verity-tree", env="VERITY_TREE", default=None,
                                     help="must name the tree that runs when given (the mixed-tree refusal)")
    research_run_dir: str | None = option("--research-run-dir", env="RESEARCH_RUN_DIR", default=None)
    research_run_id: str | None = option("--research-run-id", env="RESEARCH_RUN_ID", default=None)
    guard_preserve_file: str = option("--guard-preserve-file", env="GUARD_PRESERVE_FILE")
    pod_guard: str = option("--pod-guard", help="the pod guard shell library (the tree's copy when absent)")
    environ: Mapping[str, str] = environ_snapshot()
