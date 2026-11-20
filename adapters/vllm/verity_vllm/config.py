"""The integration's typed configuration, and where its run definitions live.

* **Run definitions.**  ``manifests/`` (checkpoint pins, capture maps) and ``workloads/`` (row workloads) sit beside
  the package in the source tree; they are run definitions, not package data.  Library code locates them, and the source
  tree itself, through ``ROOT``, ``MANIFESTS``, ``WORKLOADS``, ``CHECKPOINTS`` and ``RECORDS``; a row id finds its file through
  :func:`row_file`, which also reads the directories ``$VERITY_WORKLOADS`` names.  A workload's bytes are a fixture in the
  evidence store: the tree keeps its name, its digest (``workloads/SHA256SUMS``) and its art id (``artifacts.json``),
  and every reader takes the file through :func:`workload_file`, which fetches it by that id and checks the digest.
* **What runs.**  :class:`RowConfig`, a row's one config (its file ``workloads/<row>.json``: the workload and the pinned
  :class:`RowEngine`), the done source a row of record must declare (``DONE_SOURCES``, :func:`done_source_refusal`: its public
  cap), the row-id grammar (:class:`RowId`), the dtype vocabulary (``DTYPES``), a row's target
  card (:func:`target_card`, the card files'), the protocol-set grammar (``PROTOCOLS``, :func:`parse_protocols`), and :class:`EnginePin`, the pinned
  engine record every engine is built from: model, full revision and checkpoint role, checked against
  ``manifests/checkpoints.json``.
* **How a stage runs.**  :class:`BuildConfig`, :class:`MatchConfig` and :class:`CommitConfig`: defaults in code,
  overrides from flags or the environment of ``verity-vllm row``.
* **Where it runs.**  :class:`RunEnv`: interpreters, caches and roots.  ``pipeline/cli.py`` fills it from flags and the
  environment (the machine-path defaults live there); it is recorded as telemetry and never hashed.
* **Subcommand options.**  Every ``verity-vllm`` subcommand takes one frozen dataclass whose fields are declared with
  :func:`option`; ``pipeline/cli.py`` is the only parser.  A subcommand refuses a command line by raising
  :class:`UsageError`, and what needs the team's run store refuses the same way on a tree without it (:func:`run_store`).
"""
from __future__ import annotations

import copy
import functools
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from importlib import resources
from pathlib import Path
from types import ModuleType
from typing import Any

from verity_vllm import settings
from verity_vllm.records import workload as workload_record

#: The names the library's other modules read here.
__all__ = [
    "CARD_FILES", "CHECKPOINTS", "DEFAULT_ROW_PROTOCOLS", "DTYPES", "MANIFESTS", "RECORDS", "ROOT", "VLLM",
    "WORKLOADS", "WORKLOAD_SCHEMA", "card_arch", "cuda_arch", "EnginePin", "RowConfig", "RowEngine",
    "RowId", "UsageError", "CONTENT", "checkpoint_digest", "checkpoint_dir", "checkpoint_entries", "checkpoints_hf_home", "done_source_refusal",
    "RETIRED", "group", "option", "parse_protocols", "retired", "row_batch", "row_execution", "row_file", "row_path", "row_target", "row_tokens",
    "row_world", "rows_of_record", "run_store", "target_card", "target_cc", "target_num_sms", "target_tokens", "vllm_revision", "workload_arg", "workload_file", "workload_files",
]

#: The source tree holding the ``verity_vllm`` package, ``manifests/`` and ``workloads/``.
ROOT = Path(str(resources.files("verity_vllm"))).resolve().parent
MANIFESTS = ROOT / "manifests"
WORKLOADS = ROOT / "workloads"
CHECKPOINTS = MANIFESTS / "checkpoints.json"
#: The circuit database's index (`pipeline.config_record`, `verity-vllm records`); the rows it lists are the rows of record.
RECORDS = MANIFESTS / "records.json"
CARD_FILES = ROOT.parents[1] / "calibration/hardware/cards"


def card_arch(sku: str) -> str:
    """Card `sku`'s `arch` (sm_90a), refused unless it is a GPU card's sm_<major><minor>[a|f]."""
    from verity.device import compute_capability, load_card

    card = load_card(sku, directory=CARD_FILES)
    compute_capability(card)
    return card["arch"]


def cuda_arch(sku: str) -> str:
    """The `TORCH_CUDA_ARCH_LIST` entry of card `sku`'s `arch` (sm_90a -> 9.0a), which an extension built for it compiles to."""
    m = re.fullmatch(r"sm_(\d+)(\d)([af]?)", card_arch(sku))
    return f"{m[1]}.{m[2]}{m[3]}"


def checkpoints_hf_home() -> str | None:
    """The HF cache the checkpoint pins name (``checkpoints.json`` ``hf_home``); None without the manifest."""
    try:
        return json.loads(CHECKPOINTS.read_text()).get("hf_home") or None
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------------------- options

class UsageError(ValueError):
    """A command line the subcommand refuses; ``pipeline/cli.py`` reports it as a usage error (exit 2)."""


def run_store(module: str, entry: str) -> ModuleType:
    """``module``, the team's run and store tool (``research``) or a module that runs it, loaded by name when ``entry`` needs it,
    so nothing needs it at import time.  Where the tree leaves ``module`` out, ``entry`` is refused by name (:class:`UsageError`),
    never skipped; a module missing from inside ``module`` is a broken environment and raises as itself."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        if not f"{module}.".startswith(f"{e.name}."):
            raise
        raise UsageError(f"{entry} is refused: {module} is not on this tree. It needs the run store (`research`), the tool that "
                         "records the source identity and runs a row's stages as Attempts: put tools/research/src on PYTHONPATH "
                         "(verity-vllm row exports it)") from e


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

@functools.cache
def _gpu_cards() -> tuple[dict, ...]:
    from verity.device import load_card, skus

    return tuple(load_card(sku, directory=CARD_FILES) for sku in skus("gpu", directory=CARD_FILES))


def _token(card: dict) -> str:
    return str(card.get("short", "")).replace("-", "")


def target_tokens() -> list[str]:
    """The row-id target tokens: each GPU card file's `short` without its hyphens (rtx-pro-6000 is ``rtxpro6000``)."""
    return sorted(_token(c) for c in _gpu_cards())


def target_card(token: str) -> dict:
    """The GPU card file a row id's target token names (`target_tokens`); a token no card file names is refused by name: its
    rows wait for a card file in ``CARD_FILES``. A row's whole Build target (capability and SM count) is its card's, so its Build
    derives on any host, a GPU-free one included."""
    found = [c for c in _gpu_cards() if _token(c) == token]
    if len(found) != 1:
        raise UsageError(f"row target {token!r}: no GPU card file names it (the targets are {target_tokens()}, each a card's `short` "
                         f"without hyphens); its rows wait for a card file under {CARD_FILES}")
    return found[0]


def target_cc(token: str) -> tuple[int, int]:
    from verity.device import compute_capability

    return compute_capability(target_card(token))


def target_num_sms(token: str) -> int:
    return int(target_card(token)["sm"]["count"]["value"])

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
            elif tok.startswith(pfx) and tok[len(pfx):].isdecimal():
                vals[name] = int(tok[len(pfx):])
            else:
                raise ValueError(f"row id {row!r}: field {name} is {tok!r}, expected {pfx}<N>")
        return cls(**vals)

    def __str__(self) -> str:
        return "__".join((self.model, self.dtype, self.target, f"tp{self.world}", f"b{self.batch}", f"i{self.max_input}",
                          f"o{self.max_output}", self.arrivals, self.sampling, self.execution))


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


#: The protocols a run can enable, each with its adapter module in `verity_vllm.protocols`, in install order, innermost
#: first: SPOT commits what the model computed.
PROTOCOLS = {"sampled-proofs": "spot"}
#: The protocols a run once enabled, each (name, why), refused by name (`retired`).
RETIRED = (("pouw", "the first stage is deleted; useful work is Pearl+'s, attached by its declared name"),)


def retired(name: str) -> str | None:
    """Why the protocol ``name`` is refused by name (`RETIRED`), or None."""
    return dict(RETIRED).get(name)
#: The set a row runs when its target declares none: SPOT alone, the Commit under its committer's scheme
#: (rows-sha512 by default, `pipeline.commit --commit-scheme`).
DEFAULT_ROW_PROTOCOLS = ("sampled-proofs:rows-sha512",)
_SCHEME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+-]*")


def parse_protocols(value: str | Mapping[str, str] | Any | None) -> tuple[str, ...]:
    """A protocol set in its one spelling: ``name:scheme`` entries, one per protocol, in `PROTOCOLS` order.

    ``value`` is None, ``""`` or ``"none"`` (the empty set), a comma-separated string, an iterable of entries, or a mapping
    ``{name: scheme}``.  Refuses (ValueError) a retired protocol by name (`RETIRED`), an unknown protocol, an entry without a scheme, a malformed scheme and two
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
        if why := retired(name):
            raise ValueError(f"protocol entry {item!r}: {why}")
        if name not in PROTOCOLS:
            raise ValueError(f"protocol entry {item!r}: unknown protocol {name!r} (known: {', '.join(PROTOCOLS)})")
        if not sep or not _SCHEME.fullmatch(scheme):
            raise ValueError(f"protocol entry {item!r}: expected {name}:<scheme>, the scheme a name its protocol registers")
        if name in got:
            raise ValueError(f"protocol {name!r} is enabled twice ({got[name]!r} and {scheme!r}); a run enables one scheme per protocol")
        got[name] = scheme
    return tuple(f"{n}:{got[n]}" for n in PROTOCOLS if n in got)


#: pins kept beside a checkpoint manifest in `<stem>-<name>.json`, same schema and append-only too (the int8 recipe pins, and
#: the node-local checkpoints bound by their content: the manifest itself sits at the repository's 256 KiB file cap);
#: `checkpoint_entries` reads them after the manifest's own pins
SIBLING_MANIFESTS = ("int8", "local")
#: the revision prefix of a node-local checkpoint's pin, whose ``repo`` and ``local_path`` are its directory and whose
#: revision is this prefix and its `checkpoint_digest`
CONTENT = "sha256:"


def checkpoint_files(root: Path) -> list[Path]:
    """The files that bind a node-local checkpoint: all but a Hub snapshot's repository metadata (dot-files such as
    ``.gitattributes``, and Markdown), which no loader reads and which a re-fetched snapshot can add."""
    return sorted(p for p in root.iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix != ".md")


def checkpoint_digest(root: str | Path) -> str:
    """SHA-256 over each of `checkpoint_files`' name, size and SHA-256, in order: what a node-local pin's revision binds."""
    files = checkpoint_files(Path(root))
    if not files:
        raise ValueError(f"checkpoint {root} has no files")
    digest = hashlib.sha256()
    for file in files:
        name = file.name.encode()
        digest.update(len(name).to_bytes(4, "big") + name + file.stat().st_size.to_bytes(8, "big"))
        content = hashlib.sha256()
        with file.open("rb") as stream:
            while chunk := stream.read(8 << 20):
                content.update(chunk)
        digest.update(content.digest())
    return digest.hexdigest()


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

    ``model`` is the HF repo id and ``revision`` its full commit sha, both as ``manifests/checkpoints.json`` pins them
    (a node-local checkpoint: its directory and `CONTENT` with its `checkpoint_digest`);
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


#: The vLLM commit this tree derives and serves on: the running vLLM's version with its commit (`vllm_revision`) must
#: carry ``+g<VLLM>`` (``TargetProfile.vllm_revision``).
#: A move of it rewrites, in one change: the wheel URL, its sha256 and its readiness check in ``ops/pod_bootstrap.sh``; this name,
#: which every workload file's ``engine.vllm`` states; benchmarks' ``VLLM_PIN`` (``verity_bench.vllm.presets``) and each benchmark
#: spec's ``vllm_pin``; and PoUW's serving venv (``benchmarks/pouw/pearl_c_vllm/setup.sh``).  ``git grep`` of the old short sha
#: lists the rest.
VLLM = "d9105ea80"


def vllm_revision() -> str:
    """The installed vLLM's version with its commit in it, which a target's ``vllm_revision`` is held to: its ``__version__``
    where that carries ``+g<commit>`` (the pin's does), and at a tagged build, whose version names none, with
    ``+g<commit>`` after it.  It imports vLLM when called."""
    import vllm

    version = str(vllm.__version__)
    if "+g" in version:
        return version
    from vllm._version import __commit_id__

    return f"{version}+g{__commit_id__.removeprefix('g')}"

#: A row file (``workloads/<row>.json``): the workload and its pinned ``engine`` block (:class:`RowConfig`), the typed record
#: ``records.workload.Workload``.
WORKLOAD_SCHEMA = workload_record.SCHEMA


#: The repository root: its ``artifacts.json`` registers each stored workload file by its art id.
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
        repo = next((d for d in p.resolve().parents if (d / "artifacts.json").is_file()), REPO)
        by_repo.setdefault(repo, []).append(p)
    for repo, ps in by_repo.items():
        rels = [os.path.relpath(p.resolve(), repo) for p in ps]
        env = settings.environ()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo / "tools" / "research" / "src"), env.get("PYTHONPATH")]))
        # absolute: fetch-fixtures reads a relative PATH from its working directory, not from --repo
        r = subprocess.run([sys.executable, "-m", "research", "data", "fetch-fixtures", "--repo", str(repo), "--json",
                            *(str(p.resolve()) for p in ps)], capture_output=True, text=True, env=env)
        if missing := [rel for p, rel in zip(ps, rels) if not p.is_file()]:
            raise RuntimeError(f"{', '.join(missing)}: stored workloads, not here, and `research data fetch-fixtures` could not fetch "
                               f"them (rc {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}); fetch them where the evidence store "
                               "is reachable, or name the store's public half with $RESEARCH_PUBLIC_URL")


def workload_file(path: str | Path) -> Path:
    """``path``, made present and checked when its directory's ``SHA256SUMS`` lists it: an absent file is fetched by the art id
    ``artifacts.json`` gives its path (``research data fetch-fixtures``), and a file whose SHA-256 is not the listed one is
    refused.  A file its directory doesn't list is returned as it is."""
    p = Path(path)
    want = stored_workloads(p.parent).get(p.name)
    if want is None:
        return p
    if not p.is_file():
        _fetch([p])
    if (got := hashlib.sha256(p.read_bytes()).hexdigest()) != want:
        raise RuntimeError(f"{p}: SHA-256 {got} is not the {want} its {DIGESTS} lists: a changed workload is a new fixture "
                           "(`research data put --kind fixture/v1 --file PATH --preserve`, its id in artifacts.json, "
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
    rows = json.loads(RECORDS.read_text()).get("rows") if RECORDS.is_file() else None
    return frozenset(rows) if rows else frozenset()


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
    sampling = workload.get("sampling")
    s: Mapping[str, Any] = sampling if isinstance(sampling, Mapping) else {}
    if done_source(workload) != "public-cap":
        why = (f"its done source is 'eos-token' (sampling.ignore_eos is {s.get('ignore_eos')!r}): declare sampling.ignore_eos: true, "
               f"so every request runs its public max_tokens")
    elif stops := [k for k in ("stop_token_ids", "stop") if s.get(k)]:
        why = f"sampling.{stops[0]} {s[stops[0]]!r} stops a request before its cap: drop it"
    elif (requests := workload.get("requests")) and (uncapped := [
            str(r.get("request_id", i)) for i, r in enumerate(requests)
            if isinstance(r.get("max_tokens"), bool) or not isinstance(r.get("max_tokens"), int) or r["max_tokens"] < 1]):
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
        engine = doc.get("engine")
        e = engine if engine else {}
        unknown = sorted(set(e) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"engine block: unknown field(s) {', '.join(unknown)} (known: {', '.join(f.name for f in fields(cls))})")
        eng = cls(revision=e.get("revision"), vllm=str(e.get("vllm", VLLM)),
                  args=dict(args) if (args := e.get("args")) else {},
                  env={str(k): str(v) for k, v in env.items()} if (env := e.get("env")) else {},
                  target=dict(e["target"]) if e.get("target") else None,
                  reasons={str(k): str(v) for k, v in reasons.items()} if (reasons := e.get("reasons")) else {})
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
