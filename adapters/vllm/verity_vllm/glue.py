"""The glue inventory of a model and its time to record: the model-specific code a new model needed, and the hours from its
staged checkpoint to its published record, the two numbers the generality goal drives toward zero.

A model is its checkpoint's HF repo id, as `manifests/checkpoints.json` (`repo`), the row files (`model`) and the capture maps
(`pins.checkpoint.repo`) pin it.  Code is tied to a model by one marker, a module-level `__models__` naming the repo ids the module
exists for (after the docstring and `from __future__`, PEP 8's place for module dunders), read from the source's syntax tree and never
imported; P8 and the by-name lint skip it (it names, it decides nothing).  A Definition that belongs in Verity, whose vocabulary has no
models, carries no marker: its row in `manifests/definitions.json` (`DEFINITIONS`) ties it instead.  A module's circuit-check
sidecar (`<module>.circuit-check.py`) goes with it.  Everything else joins on a field that already exists:

  * files         the marked modules under `verity_vllm/` and `tests/` and the Definitions `DEFINITIONS` ties, with their non-blank
                  lines, by area (`program.frontend`, ...);
  * rules         the lowering rules (`RuleInfo`) the marked modules define;
  * allowlists    the entries of `tests/by_name_allowlist.json` and the lint allowlists (`tests/lint/allowlists/`) whose `file` is
                  marked for the model, or whose own `models` names it;
  * family facts  `family_facts.FAMILY_FACTS[model_type]` of the checkpoint's `config.json`, and the other checkpoints of that family;
  * profile       the patterns of the checkpoint role's derived engine profile (`profiles.profile_for_case`, at its rows' world)
                  that a marked module defines, or why the profile refuses its `config.json`;
  * capture maps  the maps pinned to the repo, by section;
  * hours         from `staged`, the checkpoint's first staging event (the label `staged` on `ckpt:<repo>@<revision>`, which the
                  step that stages its weights writes: `records.staged`), to `record`, the first `row` label on a PASS
                  `vllm-config-record/v1` of one of the model's rows (`verity-vllm records publish`).

The `config.json` is the one the Build reads (`hf_config`): the role's (`dense_generic.find_hf_config`), else the `$HF_HOME`
snapshot at the pinned revision, which glue fills with `config.json` alone from the hub when this host has none, refused unless its
sha256 is the pinned `config_sha256`.  The store is read through `research data sql` (the local catalog; `research data reindex
--remote` refreshes it).

    verity-vllm glue [REPO ...] [--json] [--no-store]   # default: every model a marker names
"""
from __future__ import annotations

import ast
import hashlib
import json
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

from verity_vllm.pipeline import circuit_db as RC
from verity_vllm.commit import capture_map as CM
from verity_vllm.config import CHECKPOINTS, MANIFESTS, ROOT, UsageError, option, workload_files
from verity_vllm.engine.profiles import dense_generic, family_facts as FF
from verity_vllm.pipeline import config_record as CR

MARKER = "__models__"
SCAN = (Path(__file__).parent, ROOT / "tests")
SIDECAR = ".circuit-check.py"
#: The models each Definition that belongs in Verity exists for, by its path: from `VERITY_HOME` under `verity/`, else from `ROOT`.
DEFINITIONS = MANIFESTS / "definitions.json"
#: the directory `verity` sits in (the repository root, in a checkout)
VERITY_HOME = Path(str(resources.files("verity"))).resolve().parent
ALLOWLISTS = (ROOT / "tests" / "by_name_allowlist.json", ROOT / "tests" / "lint" / "allowlists")

Sql = Callable[[str], list[dict[str, Any]]]
Fetch = Callable[[str, str], bytes]


def marked(path: Path, root: Path = ROOT) -> tuple[str, ...]:
    """The models a module's `__models__` names; () without one.  Anything but a tuple or list of strings refuses."""
    src = path.read_text()
    if MARKER not in src:
        return ()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == MARKER for t in node.targets):
            v = node.value
            if not (isinstance(v, (ast.Tuple, ast.List)) and v.elts and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                                                                               for e in v.elts)):
                raise ValueError(f"{path.relative_to(root)}:{node.lineno}: {MARKER} is a tuple of HF repo ids")
            return tuple(e.value for e in v.elts)
    return ()


def markers() -> dict[str, tuple[str, ...]]:
    """Every marked module under `SCAN`, by path relative to `ROOT`, and every Definition `DEFINITIONS` ties, by its row's path,
    each with its sidecar.  A `DEFINITIONS` row whose module is gone refuses."""
    out: dict[str, tuple[str, ...]] = {}
    for top in SCAN:
        for p in sorted(top.rglob("*.py")):
            if "__pycache__" in p.parts or p.name.endswith(SIDECAR):
                continue
            models = marked(p)
            if models:
                out[p.relative_to(ROOT).as_posix()] = models
                car = p.with_name(p.stem + SIDECAR)
                if car.exists():
                    out[car.relative_to(ROOT).as_posix()] = models
    for rel, models in json.loads(DEFINITIONS.read_text())["models"].items():
        p = source(rel)
        if not (rel.endswith(".py") and p.is_file()):
            raise ValueError(f"{DEFINITIONS.name} names {rel}, which is not a module: delete or move its row")
        out[rel] = tuple(models)
        if p.with_name(p.stem + SIDECAR).exists():
            out[rel.removesuffix(".py") + SIDECAR] = tuple(models)
    return out


def source(rel: str) -> Path:
    """The file a `markers()` path names."""
    return VERITY_HOME / rel if rel.split("/", 1)[0] == "verity" else ROOT / rel


def checkpoints() -> list[dict[str, Any]]:
    return json.loads(CHECKPOINTS.read_text())["checkpoints"]


def capture_maps(repo: str | None = None) -> list[dict[str, Any]]:
    maps = [CM.load(p) for p in sorted(CM.MAPS.glob("*.json"))]
    return [m for m in maps if repo is None or ((m.get("pins") or {}).get("checkpoint") or {}).get("repo") == repo]


def known() -> set[str]:
    """The models this tree pins: every checkpoint's repo and every capture map's."""
    return {c["repo"] for c in checkpoints()} | {m["pins"]["checkpoint"]["repo"] for m in capture_maps()}


def _model_type(role: str) -> str | None:
    hit = dense_generic.find_hf_config(role)
    return FF.family_of(json.loads(hit[0].read_text())) if hit else None


def hub_config(repo: str, revision: str) -> bytes:
    """`config.json` alone from the hub at `revision` (never a weight file), unauthenticated: a gated repo's refusal is the reason."""
    with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/config.json", timeout=60) as r:
        return r.read()


def hf_config(repo: str, role: str | None, revision: str | None, sha256: str | None,
              fetch: Fetch | None = hub_config) -> tuple[dict[str, Any] | None, str]:
    """The checkpoint's `config.json` as the Build reads it and where it was read, or None and why there is none (module docstring)."""
    hit = dense_generic.find_hf_config(role) if role else None
    if hit:
        return json.loads(hit[0].read_text()), hit[1]
    if not revision:
        return None, "no revision pinned"
    p = dense_generic.hf_snapshot(repo, revision) / "config.json"
    if not p.exists():
        if fetch is None or not sha256:
            return None, f"no {p}" + ("" if sha256 else ", and no config_sha256 pinned to fetch it against")
        try:
            b = fetch(repo, revision)
        except OSError as e:
            return None, f"no {p}, and the hub's config.json at {revision[:12]}: {e}"
        if hashlib.sha256(b).hexdigest() != sha256:
            return None, f"the hub's config.json at {revision[:12]} has sha256 {hashlib.sha256(b).hexdigest()[:12]}, not the pinned {sha256[:12]}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    elif sha256 and hashlib.sha256(p.read_bytes()).hexdigest() != sha256:
        return None, f"{p} is not the pinned config_sha256 {sha256[:12]}"
    return json.loads(p.read_text()), "hf_home:snapshot"


def _area(rel: str) -> str:
    parts = rel.split("/")
    if parts[0] == "verity":
        return ".".join(parts[2:-1][:2])
    return "tests" if parts[0] == "tests" else ".".join(parts[1:-1][:2]) or parts[0]


def _lines(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _rules(path: Path) -> list[str]:
    out = []
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "RuleInfo" and n.args \
                and isinstance(n.args[0], ast.Constant):
            v = n.args[1].value if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) else None
            out.append(f"{n.args[0].value}_v{v}" if v is not None else n.args[0].value)
    return out


def _allowlists(files: set[str], repo: str) -> list[dict[str, Any]]:
    paths = [ALLOWLISTS[0], *sorted(ALLOWLISTS[1].glob("*.json"))]
    return [{"allowlist": p.stem, **e} for p in paths for e in json.loads(p.read_text())["entries"]
            if e.get("file") in files or repo in (e.get("models") or ())]


def _family(model_type: str | None, repo: str) -> dict[str, Any] | None:
    fam = FF.FAMILY_FACTS.get(model_type or "")
    if fam is None:
        return None
    others = sorted({c["repo"] for c in checkpoints() if c["repo"] != repo and c.get("role") and _model_type(c["role"]) == model_type})
    params = sum(len(v) for k, v in (fam.get("params") or {}).items() if isinstance(v, dict))
    return {"model_type": model_type, "arch": fam["arch"], "facts": len(fam["facts"]), "params": params, "rules": fam["rules"],
            "shared_with": others}


def _profile(role: str | None, world: int, files: set[str], config: tuple[dict[str, Any] | None, str]) -> dict[str, Any]:
    from verity_vllm.observe.fold import profiles
    from verity_vllm.observe.fold.profiles import generic
    cfg, source = config
    if cfg is None:
        return {"refused": f"no config.json: {source}"}
    why = generic.structural_refusal(cfg)
    if why:
        return {"refused": f"its config.json ({source}): {why}"}
    if role is None:
        return {"refused": "no checkpoint role in manifests/checkpoints.json"}
    try:
        name = profiles.profile_module_for_case(role, world)
        pats = profiles.profile_for_case(role, world).patterns
    except (KeyError, ValueError) as e:
        return {"refused": str(e.args[0] if e.args else e)}
    mine = [{"pattern": p.name, "module": type(p).__module__} for p in pats
            if type(p).__module__.replace(".", "/") + ".py" in files]
    return {"module": name, "config": source, "patterns": len(pats), "entries": mine}


def _map_counts(m: dict[str, Any]) -> dict[str, Any]:
    return {"id": m["id"], **{k: len(m.get(k) or ()) for k in ("taps", "modules", "expect", "programs", "routes", "capture_work")}}


def _at(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def store_sql(query: str) -> list[dict[str, Any]]:
    return json.loads(RC.data("sql", "--json", query) or "[]")


def _quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def hours(ckpt: str | None, rows: list[str], sql: Sql) -> dict[str, Any]:
    """`staged`, the checkpoint's first staging event, and `record`, its rows' first PASS record, from the store (each None when it
    has none) and the hours between them."""
    staged = sql(f"SELECT value, ref FROM labels WHERE target = {_quote(ckpt)} AND key = {_quote(RC.STAGED)} ORDER BY value LIMIT 1") if ckpt else []
    record = sql("SELECT l.target, l.value, l.ts FROM labels l JOIN artifacts a ON a.id = l.target JOIN artifact_meta m ON m.id = l.target "
                 f"WHERE l.key = 'row' AND a.kind = {_quote(CR.KIND)} AND m.key = 'outcome' AND m.value = 'PASS' "
                 f"AND l.value IN ({', '.join(map(_quote, rows))}) ORDER BY l.ts LIMIT 1") if rows else []
    out: dict[str, Any] = {"checkpoint": ckpt,
                           "staged": {"at": staged[0]["value"], "ref": staged[0]["ref"]} if staged else None,
                           "record": {"artifact": record[0]["target"], "row": record[0]["value"], "at": record[0]["ts"]} if record else None}
    out["hours"] = (round((_at(out["record"]["at"]) - _at(out["staged"]["at"])).total_seconds() / 3600, 1)
                    if out["staged"] and out["record"] else None)
    return out


def inventory(repo: str, marks: dict[str, tuple[str, ...]], sql: Sql | None = store_sql, fetch: Fetch | None = hub_config) -> dict[str, Any]:
    """A model's glue inventory (module docstring); `sql` None leaves the hours out, `fetch` None reads no config.json from the hub."""
    ckpt = next((c for c in checkpoints() if c["repo"] == repo), {})
    maps = capture_maps(repo)
    pin = {**next((m["pins"]["checkpoint"] for m in maps), {}), **ckpt}
    role = ckpt.get("role")
    config = hf_config(repo, role, pin.get("revision"), pin.get("config_sha256"), fetch)
    model_type = (FF.family_of(config[0]) if config[0] else None) or next((m["model_type"] for m in maps), None)
    rows = {p.stem: json.loads(p.read_text()) for p in workload_files()}
    rows = {r: w for r, w in rows.items() if w.get("model") == repo}
    world = max((int(w["engine"]["args"].get("tensor_parallel_size", 1)) for w in rows.values()), default=1)
    files = {rel: ms for rel, ms in marks.items() if repo in ms}
    return {
        "model": repo, "role": role, "model_type": model_type, "rows": sorted(rows),
        "files": [{"path": rel, "area": _area(rel), "lines": _lines(source(rel)), "models": list(ms)} for rel, ms in sorted(files.items())],
        "rules": [{"path": rel, "rule": r} for rel in sorted(files) for r in _rules(source(rel))],
        "allowlists": _allowlists(set(files), repo),
        "family": _family(model_type, repo),
        "profile": _profile(role, world, set(files), config),
        "capture_maps": [_map_counts(m) for m in maps],
        "hours": hours(RC.checkpoint(repo, pin["revision"]) if pin.get("revision") else None, sorted(rows), sql) if sql else None,
    }


def render(inv: dict[str, Any]) -> str:
    files = inv["files"]
    only = [f for f in files if f["models"] == [inv["model"]]]
    areas: Counter[str] = Counter()
    for f in files:
        areas[f["area"]] += f["lines"]
    n_area = Counter(f["area"] for f in files)
    lines = [f"{inv['model']}  (role {inv['role']}, model_type {inv['model_type']}, rows {len(inv['rows'])})",
             f"  files         {len(files)}, {sum(f['lines'] for f in files):,} lines; {len(only)}, "
             f"{sum(f['lines'] for f in only):,} lines for this model alone",
             "                " + "  ".join(f"{a} {n_area[a]} ({areas[a]:,})" for a in sorted(areas)),
             f"  rules         {len(inv['rules'])}" + "".join(f"  {r['rule']}" for r in inv["rules"]),
             f"  allowlists    {len(inv['allowlists'])}" + "".join(f"  {k} {n}" for k, n in sorted(Counter(e['allowlist'] for e in inv['allowlists']).items()))]
    fam = inv["family"]
    lines.append(f"  family facts  {inv['model_type']}: none" if fam is None else
                 f"  family facts  {fam['model_type']} ({fam['arch']}): {fam['facts']} facts, {fam['params']} parameter names; "
                 f"shared with {', '.join(fam['shared_with']) or 'no other checkpoint'}")
    prof = inv["profile"]
    lines.append(f"  profile       refused: {prof['refused']}" if "refused" in prof else
                 f"  profile       {len(prof['entries'])} of {prof['patterns']} patterns of {prof['module']} ({prof['config']})"
                 + "".join(f"  {e['pattern']}" for e in prof["entries"]))
    lines += [f"  capture map   {m['id']}: " + ", ".join(f"{k} {m[k]}" for k in m if k != "id") for m in inv["capture_maps"]] or \
             ["  capture map   none"]
    h = inv["hours"]
    if h is not None:
        st = f"{h['staged']['at']} ({h['staged']['ref']})" if h["staged"] else f"no staging event on {h['checkpoint']}"
        rec = f"{h['record']['at']} ({h['record']['artifact'][:16]}…)" if h["record"] else "no PASS record"
        lines.append(f"  hours         {h['hours'] if h['hours'] is not None else '-'}  staged {st} -> record {rec}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Options:
    """Per model, the model-specific code its `__models__` markers and pins tie to it, and the hours from staged checkpoint to record."""

    models: list[str] = option("models", nargs="*", default=[], help="HF repo ids (default: every model a marker names)")
    json: bool = option("--json", action="store_true", default=False, help="print the inventories as JSON")
    store: bool = option("--store", action="boolean_optional", default=True, help="read the hours from the run store")


def main(o: Options) -> int:
    marks = markers()
    pinned = known()
    stray = {rel: m for rel, ms in marks.items() for m in ms if m not in pinned}
    if stray:
        raise UsageError(f"{MARKER} names models no checkpoint or capture map pins: {stray}")
    models = o.models or sorted({m for ms in marks.values() for m in ms})
    if set(models) - pinned:
        raise UsageError(f"{sorted(set(models) - pinned)}: no checkpoint or capture map pins it")
    invs = [inventory(m, marks, store_sql if o.store else None) for m in models]
    print(json.dumps(invs, indent=1) if o.json else "\n\n".join(map(render, invs)))
    return 0
