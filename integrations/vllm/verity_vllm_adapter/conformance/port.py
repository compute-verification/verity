"""Profile ports: load and validate `vllm-poc/profiles/<port>/` (contract Part 2, the five inputs).

Fail closed: every problem names the file and the field (`PortError`). Nothing is defaulted silently.

    port = load_port("b1-eager")                  # -> Port
    port.check_manifest(manifest)                 # -> [] or the list of declared fields the profile manifest violates
    port.program(LP=32, STEPS=7)                  # -> verity.ir.Program (or raises NotImplementedError: the port does not claim C2/L2;
                                                  #    a message starting with a taxonomy member classifies why — port.level2_taxonomy)
    port.config.axis                              # -> the one-sentence architecture question the profile answers (required)
    port.bindings_document()                      # -> the generated bindings.json content
    port.circuit_map(program)                     # -> the TA1 mapping object (or NotImplementedError)
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from . import SCHEMA_BINDINGS, SCHEMA_EXPECTATIONS, SCHEMA_PORT
from . import taxonomy as TX
from .families import (
    BINDING_STATUSES,
    DECODING_MODES,
    EXECUTION_MODES,
    EXPECTATION_STATUSES,
    HARDWARE_GENERATIONS,
    KNOWN_FAMILIES,
    REPLAY_EXECS,
    UNSUPPORTED_KINDS,
)

PORT_FILES = ("vllm_config.json", "static_config.py", "capture_mapping.py", "expectations.json")
GENERATED_FILES = ("bindings.json",)
PORT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
CC_RE = re.compile(r"^\d+\.\d+$")


class PortError(ValueError):
    """A port file is missing, malformed, or names something outside the registered vocabularies."""


def repo_root() -> Path:
    """`<root>` when this module is `<root>/verity_vllm_adapter/conformance/port.py`."""
    return Path(__file__).resolve().parents[3]


def profiles_dir() -> Path:
    return repo_root() / "vllm-poc" / "profiles"


def _ensure_import_roots() -> None:
    root = repo_root()
    for p in (root, root / "vllm-poc", root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_module(path: Path, name: str) -> ModuleType:
    _ensure_import_roots()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PortError(f"{path}: cannot import")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 - the port's own import errors are port errors
        raise PortError(f"{path}: import failed: {type(e).__name__}: {e}") from e
    return mod


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise PortError(f"{path}: missing")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise PortError(f"{path}: malformed JSON: {e}") from e


def _need(doc: dict, key: str, typ: type | tuple[type, ...], where: str, *, allow_none: bool = False) -> Any:
    if key not in doc:
        raise PortError(f"{where}: missing field {key!r}")
    v = doc[key]
    if v is None and allow_none:
        return None
    if not isinstance(v, typ):
        raise PortError(f"{where}: field {key!r} must be {getattr(typ, '__name__', typ)}, got {type(v).__name__}")
    return v


def _need_choice(doc: dict, key: str, choices: tuple[str, ...], where: str) -> str:
    v = _need(doc, key, str, where)
    if v not in choices:
        raise PortError(f"{where}: field {key!r} must be one of {list(choices)}, got {v!r}")
    return v


# ---------------------------------------------------------------------------------------------------------
# vllm_config.json
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareClass:
    gpu_generation: str
    compute_capability: tuple[str, ...]
    gpu_names: tuple[str, ...]


@dataclass(frozen=True)
class VllmConfig:
    port: str
    checkpoint_role: str
    model_repo: str
    engine_args: dict[str, Any]
    environment_flags: dict[str, str]
    hardware: HardwareClass
    execution_mode: str
    cuda_graphs: str  # "off" | "stock"
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel: bool
    decoding_mode: str
    draft_role: str | None
    quantization: str | None
    workload: dict[str, Any]
    adapter_case: str
    replay_exec: str
    capture_per_rank: bool
    known_profile_ids: tuple[str, ...]
    notes: str
    axis: str = ""  # the one-sentence architecture question this profile answers (coverage wave Part 3; required)

    def as_json(self) -> dict[str, Any]:
        return {
            "port": self.port,
            "axis": self.axis,
            "checkpoint": {"role": self.checkpoint_role, "repo": self.model_repo},
            "engine_args": self.engine_args,
            "environment_flags": self.environment_flags,
            "hardware_class": {"gpu_generation": self.hardware.gpu_generation,
                               "compute_capability": list(self.hardware.compute_capability),
                               "gpu_names": list(self.hardware.gpu_names)},
            "execution_mode": self.execution_mode,
            "cuda_graphs": self.cuda_graphs,
            "parallelism": {"tensor_parallel_size": self.tensor_parallel_size,
                            "pipeline_parallel_size": self.pipeline_parallel_size,
                            "expert_parallel": self.expert_parallel},
            "decoding": {"mode": self.decoding_mode, "draft_role": self.draft_role},
            "quantization": self.quantization,
            "workload": self.workload,
            "adapter": {"case": self.adapter_case, "replay_exec": self.replay_exec, "capture_per_rank": self.capture_per_rank},
            "known_profile_ids": list(self.known_profile_ids),
        }


def _parse_vllm_config(doc: Any, where: str, checkpoints: dict[str, dict] | None) -> VllmConfig:
    if not isinstance(doc, dict):
        raise PortError(f"{where}: must be a JSON object")
    if doc.get("schema") != SCHEMA_PORT:
        raise PortError(f"{where}: field 'schema' must be {SCHEMA_PORT!r}, got {doc.get('schema')!r}")
    port = _need(doc, "port", str, where)
    if not PORT_NAME_RE.match(port):
        raise PortError(f"{where}: field 'port' must match {PORT_NAME_RE.pattern}, got {port!r}")
    ck = _need(doc, "checkpoint", dict, where)
    role = _need(ck, "role", str, f"{where}:checkpoint")
    repo = _need(ck, "repo", str, f"{where}:checkpoint")
    from verity_vllm_adapter import profile as prof  # light import (no torch at module level)

    if role not in prof.CASES:
        raise PortError(f"{where}:checkpoint: field 'role' {role!r} is not an adapter case ({sorted(prof.CASES)})")
    if prof.CASES[role]["repo"] != repo:
        raise PortError(f"{where}:checkpoint: field 'repo' {repo!r} is not the adapter's repo for role {role!r} ({prof.CASES[role]['repo']!r})")
    if checkpoints is not None and repo not in checkpoints:
        raise PortError(f"{where}:checkpoint: field 'repo' {repo!r} is not in manifests/checkpoints.json")
    engine_args = _need(doc, "engine_args", dict, where)
    env = _need(doc, "environment_flags", dict, where)
    for k, v in env.items():
        if not isinstance(v, str):
            raise PortError(f"{where}:environment_flags: flag {k!r} must be a string value")
    hw = _need(doc, "hardware_class", dict, where)
    gen = _need_choice(hw, "gpu_generation", HARDWARE_GENERATIONS, f"{where}:hardware_class")
    ccs = _need(hw, "compute_capability", list, f"{where}:hardware_class")
    if not ccs or not all(isinstance(c, str) and CC_RE.match(c) for c in ccs):
        raise PortError(f"{where}:hardware_class: field 'compute_capability' must be a non-empty list of 'major.minor' strings")
    names = hw.get("gpu_names", [])
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise PortError(f"{where}:hardware_class: field 'gpu_names' must be a list of strings")
    mode = _need_choice(doc, "execution_mode", EXECUTION_MODES, where)
    graphs = doc.get("cuda_graphs", "off" if mode == "eager" else "stock")
    if graphs not in ("off", "stock"):
        raise PortError(f"{where}: field 'cuda_graphs' must be 'off' or 'stock', got {graphs!r}")
    if mode == "eager" and graphs != "off":
        raise PortError(f"{where}: execution_mode 'eager' requires cuda_graphs 'off'")
    par = _need(doc, "parallelism", dict, where)
    tp = _need(par, "tensor_parallel_size", int, f"{where}:parallelism")
    pp = _need(par, "pipeline_parallel_size", int, f"{where}:parallelism")
    ep = _need(par, "expert_parallel", bool, f"{where}:parallelism")
    if tp < 1 or pp < 1:
        raise PortError(f"{where}:parallelism: sizes must be >= 1")
    dec = _need(doc, "decoding", dict, where)
    dmode = _need_choice(dec, "mode", DECODING_MODES, f"{where}:decoding")
    draft = dec.get("draft_role")
    if dmode == "speculative" and not isinstance(draft, str):
        raise PortError(f"{where}:decoding: speculative decoding requires field 'draft_role'")
    quant = _need(doc, "quantization", str, where, allow_none=True)
    wl = _need(doc, "workload", dict, where)
    _need(wl, "id", str, f"{where}:workload")
    ad = _need(doc, "adapter", dict, where)
    case = _need(ad, "case", str, f"{where}:adapter")
    if case != role:
        raise PortError(f"{where}:adapter: field 'case' {case!r} must equal checkpoint.role {role!r}")
    rexec = _need_choice(ad, "replay_exec", REPLAY_EXECS, f"{where}:adapter")
    if mode == "eager" and rexec != "eager":
        raise PortError(f"{where}:adapter: execution_mode 'eager' requires replay_exec 'eager', got {rexec!r}")
    if mode != "eager" and rexec == "eager":
        raise PortError(f"{where}:adapter: a compiled execution_mode needs a compiled-preserving replay_exec, not 'eager'")
    per_rank = _need(ad, "capture_per_rank", bool, f"{where}:adapter")
    if tp > 1 and not per_rank:
        raise PortError(f"{where}:adapter: tensor_parallel_size {tp} > 1 requires capture_per_rank true (the per-rank seam, `s=<rank>`)")
    if tp == 1 and per_rank:
        raise PortError(f"{where}:adapter: capture_per_rank true needs tensor_parallel_size > 1")
    kpi = doc.get("known_profile_ids", [])
    if not isinstance(kpi, list) or not all(isinstance(x, str) and re.match(r"^[0-9a-f]{12}$", x) for x in kpi):
        raise PortError(f"{where}: field 'known_profile_ids' must be a list of 12-hex profile ids")
    notes = doc.get("notes", "")
    if not isinstance(notes, str):
        raise PortError(f"{where}: field 'notes' must be a string")
    # every profile answers one architecture question (coverage wave Part 3 / Part 4 item 3); the matrix's `axis` column
    if "axis" not in doc:
        raise PortError(f"{where}: missing field 'axis' (the one-sentence architecture question this profile answers, e.g. "
                        f"\"static config selects different exact primitives without changing program structure\")")
    axis = doc["axis"]
    if not isinstance(axis, str) or not axis.strip():
        raise PortError(f"{where}: field 'axis' must be a non-empty string (one sentence)")
    return VllmConfig(port, role, repo, dict(engine_args), {k: str(v) for k, v in env.items()},
                      HardwareClass(gen, tuple(ccs), tuple(names)), mode, graphs, tp, pp, ep, dmode, draft, quant,
                      dict(wl), case, rexec, per_rank, tuple(kpi), notes, axis.strip())


# ---------------------------------------------------------------------------------------------------------
# expectations.json
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyExpectation:
    family: str
    status: str
    reason: str
    kind: str | None  # unsupported only
    relations: tuple[str, ...]  # relation ids the bundle is expected to record for this family
    unsupported_reason: str | None = None  # unsupported only: one of taxonomy.UNSUPPORTED_REASONS (closed enum)


@dataclass(frozen=True)
class Expectations:
    families: dict[str, FamilyExpectation]
    unmaterialised: tuple[dict[str, str], ...]
    not_in_program: tuple[dict[str, str], ...]
    # functions/values the port declares unsupported, each with a closed-enum reason and the detail naming it
    # (`{"function", "reason", "detail", "rung"?}`); the first is the port's headline unsupported outcome
    unsupported_functions: tuple[dict[str, str], ...] = ()
    # why a rung did not advance, declared by the port (`{"rung", "class", "detail", "resolved"?}`; class from
    # ladder.BLOCKERS, GUIDANCE §24); a declaration whose rung is met is kept by the ladder as history
    blockers: tuple[dict[str, str], ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "families": {f: {"status": e.status, "reason": e.reason, **({"kind": e.kind} if e.kind else {}),
                             **({"unsupported_reason": e.unsupported_reason} if e.unsupported_reason else {}),
                             "relations": list(e.relations)} for f, e in self.families.items()},
            "unmaterialised": list(self.unmaterialised),
            "not_in_program": list(self.not_in_program),
            "unsupported_functions": list(self.unsupported_functions),
            "blockers": list(self.blockers),
        }


def _parse_expectations(doc: Any, where: str) -> Expectations:
    if not isinstance(doc, dict):
        raise PortError(f"{where}: must be a JSON object")
    if doc.get("schema") != SCHEMA_EXPECTATIONS:
        raise PortError(f"{where}: field 'schema' must be {SCHEMA_EXPECTATIONS!r}, got {doc.get('schema')!r}")
    fams = _need(doc, "families", dict, where)
    if not fams:
        raise PortError(f"{where}: field 'families' must name at least one family")
    out: dict[str, FamilyExpectation] = {}
    for name, e in fams.items():
        w = f"{where}:families.{name}"
        if name not in KNOWN_FAMILIES:
            raise PortError(f"{w}: unknown family {name!r} (registered families: {sorted(KNOWN_FAMILIES)})")
        if not isinstance(e, dict):
            raise PortError(f"{w}: must be an object")
        status = _need_choice(e, "status", EXPECTATION_STATUSES, w)
        reason = _need(e, "reason", str, w)
        if not reason.strip():
            raise PortError(f"{w}: field 'reason' must not be empty")
        kind = None
        ureason = None
        if status == "unsupported":
            kind = _need_choice(e, "kind", UNSUPPORTED_KINDS, w)
            # the closed taxonomy (taxonomy.py): an unsupported family must say which kind of unsupported it is
            if "unsupported_reason" not in e:
                raise PortError(f"{w}: status 'unsupported' requires field 'unsupported_reason', one of {list(TX.UNSUPPORTED_REASONS)} (taxonomy.py)")
            try:
                ureason = TX.check_unsupported_reason(e["unsupported_reason"], f"{w}: field 'unsupported_reason'")
            except TX.TaxonomyError as ex:
                raise PortError(str(ex)) from ex
        else:
            if "kind" in e:
                raise PortError(f"{w}: field 'kind' is only for status 'unsupported'")
            if "unsupported_reason" in e:
                raise PortError(f"{w}: field 'unsupported_reason' is only for status 'unsupported' (a {status!r} family has no unsupported outcome)")
        rels = e.get("relations", [])
        if not isinstance(rels, list) or not all(isinstance(r, str) for r in rels):
            raise PortError(f"{w}: field 'relations' must be a list of relation ids")
        out[name] = FamilyExpectation(name, status, reason, kind, tuple(rels), ureason)

    def _list_of_records(key: str, required: tuple[str, ...]) -> tuple[dict[str, str], ...]:
        lst = doc.get(key, [])
        if not isinstance(lst, list):
            raise PortError(f"{where}: field {key!r} must be a list")
        recs = []
        for i, r in enumerate(lst):
            if not isinstance(r, dict):
                raise PortError(f"{where}:{key}[{i}]: must be an object")
            for k in required:
                if not isinstance(r.get(k), str) or not r[k].strip():
                    raise PortError(f"{where}:{key}[{i}]: missing field {k!r}")
            recs.append({k: str(v) for k, v in r.items()})
        return tuple(recs)

    # `unsupported_functions`: each names the function/value, carries a closed-enum reason and the free-text detail
    # (Part 4 item 2: `unsupported:primitive-semantics-missing FA3 wgmma m64n8k16`); optional `rung` (C0..C5 / L1..L4)
    from .ladder import normalize_rung  # noqa: PLC0415  (ladder imports evidence, which imports this module)

    unsupported_functions = []
    for i, u in enumerate(_list_of_records("unsupported_functions", ("function", "reason", "detail"))):
        w = f"{where}:unsupported_functions[{i}]"
        try:
            TX.check_unsupported_reason(u["reason"], f"{w}: field 'reason'")
        except TX.TaxonomyError as ex:
            raise PortError(str(ex)) from ex
        if "rung" in u:
            try:
                u["rung"] = normalize_rung(u["rung"])
            except ValueError as ex:
                raise PortError(f"{w}: field 'rung': {ex}") from ex
        unsupported_functions.append(u)

    from .ladder import BLOCKERS  # noqa: PLC0415

    blockers = []
    seen_rungs: set[str] = set()
    for i, b in enumerate(_list_of_records("blockers", ("rung", "class", "detail"))):
        w = f"{where}:blockers[{i}]"
        try:
            b["rung"] = normalize_rung(b["rung"])
        except ValueError as ex:
            raise PortError(f"{w}: field 'rung': {ex}") from ex
        if b["class"] not in BLOCKERS:
            raise PortError(f"{w}: field 'class' must be one of {list(BLOCKERS)}, got {b['class']!r}")
        if b["rung"] in seen_rungs:
            raise PortError(f"{w}: rung {b['rung']} declared twice")
        seen_rungs.add(b["rung"])
        blockers.append(b)

    return Expectations(out, _list_of_records("unmaterialised", ("canonical", "reason")),
                        _list_of_records("not_in_program", ("pattern", "reason")), tuple(unsupported_functions), tuple(blockers))


# ---------------------------------------------------------------------------------------------------------
# static_config.py / bindings.json / capture_mapping.py
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Binding:
    function: str  # FunctionDefinition name (e.g. Gemm)
    registered: str  # id@version, e.g. Gemm@1 (the registry id is Gemm_v1)
    status: str  # one of BINDING_STATUSES
    family: str | None  # capture family this function's values are recorded under (None: structural)
    divergence_recorded: str | None  # text of a recorded production divergence attached to the record


def _parse_binding(fn: str, b: Any, where: str, registry) -> Binding:
    w = f"{where}:{fn}"
    if not isinstance(b, dict):
        raise PortError(f"{w}: must be an object")
    reg = _need(b, "registered", str, w)
    if "@" not in reg:
        raise PortError(f"{w}: field 'registered' must be 'Name@version', got {reg!r}")
    name, ver = reg.rsplit("@", 1)
    if not ver.isdigit():
        raise PortError(f"{w}: field 'registered' version must be an integer, got {reg!r}")
    rid = f"{name}_v{ver}"
    if rid not in registry:
        raise PortError(f"{w}: binding to unregistered version {reg!r} ({rid} is not in verity.ir.REGISTRY)")
    if name != fn:
        raise PortError(f"{w}: field 'registered' names {name!r}, not the function {fn!r}")
    status = _need_choice(b, "status", BINDING_STATUSES, w)
    fam = b.get("family")
    if fam is not None and fam not in KNOWN_FAMILIES:
        raise PortError(f"{w}: unknown family {fam!r}")
    div = b.get("divergence_recorded")
    if div is not None and not isinstance(div, str):
        raise PortError(f"{w}: field 'divergence_recorded' must be a string")
    return Binding(fn, reg, status, fam, div)


def split_conformance(text: str) -> tuple[str, str | None]:
    """`verity.ir` conformance strings are `<status>[; <recorded divergence>]` (typed-b1.md §3/§5)."""
    head, _, rest = text.partition(";")
    return head.strip(), (rest.strip() or None)


# ---------------------------------------------------------------------------------------------------------
# the port
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Port:
    name: str
    path: Path
    config: VllmConfig
    expectations: Expectations
    static_config: ModuleType = field(repr=False)
    capture_mapping: ModuleType = field(repr=False)
    bindings: dict[str, Binding] = field(default_factory=dict)
    level2_claimed: bool = True
    level2_reason: str | None = None  # why L2/C2 is not claimed (static_config.program raised NotImplementedError)
    # `(reason, detail)` when the NotImplementedError text starts with a taxonomy member (`unsupported:… <function>`);
    # None when the reason is free text (pending work, e.g. "typed M1 program: lane CM" — not an unsupported outcome)
    level2_taxonomy: tuple[str, str] | None = None
    mapping_implemented: bool = True
    mapping_reason: str | None = None

    # -- static configuration --------------------------------------------------------------------------
    def program(self, **shape: int):
        """The typed `Program` for this port at the given shape statics (`LP`, `STEPS`)."""
        return self.static_config.program(**shape)

    def config_digest(self) -> str | None:
        f = getattr(self.static_config, "config_digest", None)
        return f() if callable(f) else None

    def static_config_record(self) -> dict[str, Any]:
        f = getattr(self.static_config, "static_config", None)
        return dict(f()) if callable(f) else {}

    def bindings_document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_BINDINGS,
            "port": self.name,
            "generated_from": "static_config.bindings()",
            "bindings": {fn: {"registered": b.registered, "status": b.status,
                              **({"family": b.family} if b.family else {}),
                              **({"divergence_recorded": b.divergence_recorded} if b.divergence_recorded else {})}
                         for fn, b in sorted(self.bindings.items())},
        }

    def circuit_map(self, program):
        """The port's capture-v2 identity -> `FunctionCall` mapping (TA1's `CircuitMap` contract: `.locate(ident)`)."""
        return self.capture_mapping.circuit_map(program)

    def mapping_record(self) -> dict[str, Any]:
        f = getattr(self.capture_mapping, "identities", None)
        return {"implemented": self.mapping_implemented, "reason": self.mapping_reason,
                "identities": (f() if callable(f) else None)}

    # -- profile manifest ------------------------------------------------------------------------------
    def check_manifest(self, m: dict[str, Any]) -> list[str]:
        """Declared fields of `vllm_config.json` that the profile manifest (`verity-vllm/profile/v1`) violates."""
        c = self.config
        bad: list[str] = []
        if m.get("schema") != "verity-vllm/profile/v1":
            bad.append(f"manifest schema {m.get('schema')!r}")
        if (m.get("model") or {}).get("repo") != c.model_repo:
            bad.append(f"checkpoint.repo: port {c.model_repo!r}, manifest {(m.get('model') or {}).get('repo')!r}")
        if m.get("case") != c.checkpoint_role:
            bad.append(f"checkpoint.role: port {c.checkpoint_role!r}, manifest case {m.get('case')!r}")
        flags = m.get("engine_flags") or {}
        for k, v in c.environment_flags.items():
            if str(flags.get(k)) != v:
                bad.append(f"environment_flags.{k}: port {v!r}, manifest {flags.get(k)!r}")
        comp = m.get("compilation") or {}
        eager = bool(comp.get("enforce_eager", (m.get("engine_kwargs") or {}).get("enforce_eager", False)))
        if (c.execution_mode == "eager") != eager:
            bad.append(f"execution_mode: port {c.execution_mode!r}, manifest enforce_eager={eager}")
        if c.execution_mode != "eager":
            cg = str(comp.get("cudagraph_mode", "")).split(".")[-1]
            if c.cuda_graphs == "off" and cg not in ("NONE", ""):
                bad.append(f"cuda_graphs: port 'off', manifest cudagraph_mode {cg!r}")
            if c.cuda_graphs == "stock" and cg == "NONE":
                bad.append(f"cuda_graphs: port 'stock', manifest cudagraph_mode NONE")
        par = m.get("parallelism") or {}
        if int(par.get("tensor_parallel_size", 1)) != c.tensor_parallel_size:
            bad.append(f"parallelism.tensor_parallel_size: port {c.tensor_parallel_size}, manifest {par.get('tensor_parallel_size')}")
        if int(par.get("pipeline_parallel_size", 1)) != c.pipeline_parallel_size:
            bad.append(f"parallelism.pipeline_parallel_size: port {c.pipeline_parallel_size}, manifest {par.get('pipeline_parallel_size')}")
        gpu = m.get("gpu") or {}
        if str(gpu.get("compute_capability")) not in c.hardware.compute_capability:
            bad.append(f"hardware_class.compute_capability: port {list(c.hardware.compute_capability)}, manifest {gpu.get('compute_capability')!r}")
        if c.hardware.gpu_names and gpu.get("name") not in c.hardware.gpu_names:
            bad.append(f"hardware_class.gpu_names: port {list(c.hardware.gpu_names)}, manifest {gpu.get('name')!r}")
        if m.get("quantization") != c.quantization:
            bad.append(f"quantization: port {c.quantization!r}, manifest {m.get('quantization')!r}")
        ek = m.get("engine_kwargs") or {}
        for k, v in c.engine_args.items():
            mv = ek.get(k, False if isinstance(v, bool) else None)
            if mv != v:
                bad.append(f"engine_args.{k}: port {v!r}, manifest {mv!r}")
        return bad

    def as_json(self) -> dict[str, Any]:
        return {"port": self.name, "path": str(self.path), "axis": self.config.axis, "vllm_config": self.config.as_json(),
                "expectations": self.expectations.as_json(), "bindings": self.bindings_document()["bindings"],
                "level2_claimed": self.level2_claimed, "level2_reason": self.level2_reason,
                "level2_taxonomy": list(self.level2_taxonomy) if self.level2_taxonomy else None,
                "capture_mapping": self.mapping_record()}


def load_port(name_or_path: str | Path, *, checkpoints: Path | None = None, check_generated: bool = True) -> Port:
    """Load and validate a port; raise `PortError` naming the first offending file/field."""
    p = Path(name_or_path)
    path = p if p.is_dir() else profiles_dir() / str(name_or_path)
    if not path.is_dir():
        raise PortError(f"port directory {path} does not exist")
    for f in PORT_FILES:
        if not (path / f).exists():
            raise PortError(f"{path / f}: missing (a port has exactly {list(PORT_FILES)} plus the generated {list(GENERATED_FILES)})")
    ck_path = checkpoints or (repo_root() / "manifests" / "checkpoints.json")
    ck = None
    if ck_path.exists():
        ck = {c["repo"]: c for c in _read_json(ck_path)["checkpoints"]}
    cfg = _parse_vllm_config(_read_json(path / "vllm_config.json"), str(path / "vllm_config.json"), ck)
    if cfg.port != path.name:
        raise PortError(f"{path / 'vllm_config.json'}: field 'port' {cfg.port!r} must equal the directory name {path.name!r}")
    exp = _parse_expectations(_read_json(path / "expectations.json"), str(path / "expectations.json"))
    sc = _load_module(path / "static_config.py", f"verity_port_{path.name.replace('-', '_')}_static_config")
    for attr in ("program", "bindings"):
        if not callable(getattr(sc, attr, None)):
            raise PortError(f"{path / 'static_config.py'}: must define a callable {attr!r}")
    cm = _load_module(path / "capture_mapping.py", f"verity_port_{path.name.replace('-', '_')}_capture_mapping")
    if not callable(getattr(cm, "circuit_map", None)):
        raise PortError(f"{path / 'capture_mapping.py'}: must define a callable 'circuit_map(program)'")
    port = Port(path.name, path, cfg, exp, sc, cm)
    # bindings: data generated from static_config.bindings(); every binding must name a registered version
    from verity.ir import REGISTRY  # noqa: PLC0415

    try:
        raw = sc.bindings()
    except NotImplementedError as e:
        port.level2_claimed = False
        port.level2_reason = str(e) or "static_config.bindings() raised NotImplementedError"
        port.level2_taxonomy = TX.parse(port.level2_reason)
        raw = {}
    if not isinstance(raw, dict):
        raise PortError(f"{path / 'static_config.py'}: bindings() must return a dict FunctionDefinition -> binding record")
    port.bindings = {fn: _parse_binding(fn, b, f"{path / 'static_config.py'}:bindings()", REGISTRY) for fn, b in raw.items()}
    # program construction is the L2 claim: a port that raises NotImplementedError does not claim L2
    probe = getattr(sc, "PROBE_SHAPE", {"LP": 2, "STEPS": 1})
    try:
        prog = sc.program(**probe)
    except NotImplementedError as e:
        port.level2_claimed = False
        port.level2_reason = str(e) or "static_config.program() raised NotImplementedError"
        port.level2_taxonomy = TX.parse(port.level2_reason)
        prog = None
    else:
        from verity.ir import Program  # noqa: PLC0415

        if not isinstance(prog, Program):
            raise PortError(f"{path / 'static_config.py'}: program() must return a verity.ir.Program, got {type(prog).__name__}")
        if not port.bindings:
            raise PortError(f"{path / 'static_config.py'}: bindings() returned no bindings for a port that constructs a program")
        _check_bindings_cover_program(port, prog, path)
    # a generated bindings.json that disagrees with static_config.bindings() is a port error (it is data, regenerated)
    gen = path / "bindings.json"
    if gen.exists() and check_generated:  # `conformance bindings` regenerates a stale file instead of refusing to load it
        doc = _read_json(gen)
        if doc.get("schema") != SCHEMA_BINDINGS:
            raise PortError(f"{gen}: field 'schema' must be {SCHEMA_BINDINGS!r}")
        if doc.get("bindings") != port.bindings_document()["bindings"]:
            raise PortError(f"{gen}: field 'bindings' is stale: regenerate with `python -m verity_vllm_adapter.conformance bindings --port {path.name}`")
    # capture mapping: implemented or an explicit NotImplementedError with a reason
    if prog is not None:
        try:
            port.circuit_map(prog)
        except NotImplementedError as e:
            port.mapping_implemented = False
            port.mapping_reason = str(e) or "capture_mapping.circuit_map() raised NotImplementedError"
    else:
        port.mapping_implemented = False
        port.mapping_reason = port.level2_reason
    return port


def _check_bindings_cover_program(port: Port, prog, path: Path) -> None:
    """Every composite reachable from the program's root must be bound (name -> registered version), and the
    registered version must be the one the program actually uses."""
    from verity.ir.defs import PrimitiveDefinition, SpecializedDefinition  # noqa: PLC0415

    seen: dict[str, int] = {}

    def visit(fn) -> None:
        if isinstance(fn, PrimitiveDefinition):
            return
        assert isinstance(fn, SpecializedDefinition)
        if fn.name in seen:
            return
        seen[fn.name] = fn.version
        for n in fn.body.nodes:
            visit(n.fn)
        for v in fn.bindings.values():
            if isinstance(v, (PrimitiveDefinition, SpecializedDefinition)):
                visit(v)

    visit(prog.fn)
    for name, ver in sorted(seen.items()):
        if name.startswith("Root["):
            continue
        b = port.bindings.get(name)
        if b is None:
            raise PortError(f"{path / 'static_config.py'}:bindings(): the program uses {name}_v{ver} but bindings() does not bind {name!r}")
        if b.registered != f"{name}@{ver}":
            raise PortError(f"{path / 'static_config.py'}:bindings(): {name} is bound to {b.registered!r} but the program uses version {ver}")


def write_bindings(port: Port) -> Path:
    out = port.path / "bindings.json"
    out.write_text(json.dumps(port.bindings_document(), indent=1, sort_keys=False) + "\n")
    return out


def list_ports() -> list[str]:
    d = profiles_dir()
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / "vllm_config.json").exists()) if d.is_dir() else []
