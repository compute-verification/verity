"""Split the dense family's registry modules into Definitions, twins and vocabulary (dense, activation, norms, RoPE, GEMM bias).

The family's Definition modules stay in `verity_vllm/program/registry/` and import only `verity` and other Definition modules
(`DEFINITION_MODULES`, `READS`). Its twins and vocabulary leave the registry for the integration package (`MOVES`):

    registry/rope_difftest.py        -> properties/difftests/rope_difftest.py   difftest adapter (admission imports it by name)
    registry/rope_gptj_difftest.py   -> properties/difftests/rope_gptj_difftest.py
    registry/rmsnorm_fused_sweep.py  -> program/kernels/rmsnorm_fused_sweep.py  numpy twins of the fused norm, and its B0 sweep
    registry/gemm_targets.py         -> program/frontend/gemm_targets.py        which GEMM Definition and step each target binds

`apply` moves each module (`git mv`) and rewrites every reference in the tracked tree: imports, dotted module names in strings
and docs, file paths in the lint allowlists. It is idempotent: on a tree where the split is done it changes nothing, and it
fails by name, changing nothing, on a tree it cannot split (a module in both places or in neither, a reference form it cannot
rewrite). `verify --before REV` regenerates circuit-check's task list and the descriptor digest of every target on REV and on
this tree and fails on any difference; `pin` writes the family's targets and digests from a snapshot to the fixture
`tools/verity/circuit_check/tests/fixtures/split/dense.json`.

    uv run python integrations/vllm/tools/migrate/split_dense.py plan
    uv run python integrations/vllm/tools/migrate/split_dense.py apply
    uv run python integrations/vllm/tools/migrate/split_dense.py verify --before origin/main
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
VLLM = "integrations/vllm"
PKG = "verity_vllm"
REGISTRY = f"{PKG}.program.registry"

#: the family's Definition modules (they stay in the registry)
DEFINITION_MODULES = ("dense", "act", "boolean_dense", "boolean_dense_norm", "boolean_norms", "boolean_activation",
                      "boolean_silu", "boolean_rope", "gemm_bias")
#: the other families' Definition modules the family's Definitions read
READS = ("prims", "b1", "sampling")


@dataclass(frozen=True)
class Move:
    name: str
    package: str   # the dotted package it moves to
    kind: str      # twin | vocabulary

    @property
    def old(self) -> str:
        return f"{REGISTRY}.{self.name}"

    @property
    def new(self) -> str:
        return f"{self.package}.{self.name}"

    @property
    def old_path(self) -> str:
        return self.old.replace(".", "/") + ".py"

    @property
    def new_path(self) -> str:
        return self.new.replace(".", "/") + ".py"


MOVES = (
    Move("rope_difftest", f"{PKG}.properties.difftests", "twin"),
    Move("rope_gptj_difftest", f"{PKG}.properties.difftests", "twin"),
    Move("rmsnorm_fused_sweep", f"{PKG}.program.kernels", "twin"),
    Move("gemm_targets", f"{PKG}.program.frontend", "vocabulary"),
)

#: packages this split creates, with their docstrings
NEW_PACKAGES = {f"{PKG}.properties.difftests": '"""Difftest adapters (`verity_vllm.properties.admission`): each runs a registered '
                                               'Definition and the production vLLM kernel it models on the same edge-kind inputs."""\n'}

#: never rewritten: this tool and its test name the old modules on purpose, and history stays as it was written
EXCLUDE = (f"{VLLM}/tools/migrate/", f"{VLLM}/tests/tools/test_split_dense.py", "CHANGELOG.md")
#: prose kept as written, so never rewritten, and refused if it imports a moved module: `b1.py` cites the sweep in two
#: `conformance=` strings, which `frontend.provenance.registry_version()` reads
KEPT = (f"{VLLM}/{PKG}/program/registry/b1.py",)
FIXTURE = "tools/verity/circuit_check/tests/fixtures/split/dense.json"
ROOTS = f"{VLLM}/tests/census_roots.txt"
BY_NAME = f"{VLLM}/tests/by_name_allowlist.json"
#: `tests/test_no_by_name_rules.py`'s SKIP_DIRS: a moved file under one leaves that scan, and its entries go stale
BY_NAME_UNSCANNED = ("kernels",)
#: census roots for what only the registry walk (`registry.catalog.load()`) reached: a moved module's own root folds into its
#: package's
ROOT_DIRS = {f"{PKG}.properties.difftests": "the difftest adapters, imported by name (properties-admission --adapter)"}


class SplitError(RuntimeError):
    """The tree is in a state this split cannot handle; nothing was changed."""


# --- the rewrite ------------------------------------------------------------------------------------------------------------

def _short(dotted: str) -> str:
    """The last two components (`registry.gemm_targets` in prose stands for the module)."""
    return ".".join(dotted.split(".")[-2:])


_IMPORT = re.compile(r"^(?P<indent>[ \t]*)from[ \t]+" + re.escape(REGISTRY) + r"[ \t]+import[ \t]+(?P<names>[^\n#]*?)[ \t]*(?P<comment>#[^\n]*)?$",
                     re.M)


def _rewrite_imports(text: str, where: str) -> str:
    """`from verity_vllm.program.registry import m [as X]` of a moved module, to its new package; a statement importing a moved
    module beside others is split in two."""
    by_name = {m.name: m for m in MOVES}

    def sub(mt: re.Match) -> str:
        names = mt["names"]
        if names.startswith("("):
            close = mt.string.find(")", mt.start())
            stmt = mt.string[mt.start():close + 1 if close >= 0 else len(mt.string)]
            if any(re.search(rf"\b{n}\b", stmt) for n in by_name):
                raise SplitError(f"{where}: a parenthesized import of a moved module; rewrite it by hand first: {stmt.strip()}")
            return mt[0]
        parts = [p.strip() for p in names.split(",") if p.strip()]
        moved = [p for p in parts if p.split()[0] in by_name]
        if not moved:
            return mt[0]
        kept = [p for p in parts if p not in moved]
        indent, comment = mt["indent"], f"  {mt['comment']}" if mt["comment"] else ""
        lines = [f"{indent}from {REGISTRY} import {', '.join(kept)}"] if kept else []
        for p in moved:
            lines.append(f"{indent}from {by_name[p.split()[0]].package} import {p}")
        lines[-1] += comment
        return "\n".join(lines)

    return _IMPORT.sub(sub, text)


def _rewrite_names(text: str) -> str:
    for m in MOVES:
        text = text.replace(m.old_path, m.new_path)
        text = re.sub(rf"(?<![\w.]){re.escape(m.old)}\b", m.new, text)
        text = re.sub(rf"(?<![\w.])program\.registry\.{m.name}\b", m.new.removeprefix(f"{PKG}."), text)
        text = re.sub(rf"(?<![\w.])registry\.{m.name}\b", _short(m.new), text)
    return text


def _absolute_relative_imports(text: str) -> str:
    """A moved module's relative imports (`from . import b1`, `from .prims import X`) made absolute against the registry."""
    text = re.sub(r"^([ \t]*)from \. import ", rf"\1from {REGISTRY} import ", text, flags=re.M)
    return re.sub(r"^([ \t]*)from \.(\w)", rf"\1from {REGISTRY}.\2", text, flags=re.M)


_CYCLE = re.compile(r'("kind": "(?:module|package)-cycle", "symbol": "<package>", "detail": ")([^"]+)(")')


def _sort_cycles(text: str) -> str:
    """P9 names a cycle by its members in sorted order, so a renamed member can move within its entry."""
    return _CYCLE.sub(lambda mt: mt[1] + " <-> ".join(sorted(mt[2].split(" <-> "))) + mt[3], text)


def _add_roots(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for pkg, why in ROOT_DIRS.items():
        entry = pkg.replace(".", "/") + "/"
        own = {m.new for m in MOVES if m.package == pkg}
        at = [i for i, ln in enumerate(lines) if ln.split("#")[0].strip() in own]
        if any(ln.split("#")[0].strip() == entry for ln in lines):
            continue
        lines = [ln for i, ln in enumerate(lines) if i not in at]
        lines.insert(at[0] if at else len(lines), f"{entry}      # {why}\n")
    return "".join(lines)


def _drop_unscanned(text: str) -> str:
    gone = {m.new_path for m in MOVES if set(Path(m.new_path).parts[:-1]) & set(BY_NAME_UNSCANNED)}
    out = "".join(ln for ln in text.splitlines(keepends=True) if not any(f'"file": "{g}"' in ln for g in gone))
    try:
        json.loads(out)
    except ValueError as e:
        raise SplitError(f"{BY_NAME}: dropping the entries of {sorted(gone)} leaves invalid JSON ({e}); drop them by hand") from None
    return out


def rewrite(text: str, where: str = "<text>") -> str:
    text = _rewrite_names(_rewrite_imports(text, where))
    if where.endswith("p09_layering.json"):
        text = _sort_cycles(text)
    if where.endswith(ROOTS):
        text = _add_roots(text)
    if where.endswith(BY_NAME):
        text = _drop_unscanned(text)
    return text


# --- the tree ---------------------------------------------------------------------------------------------------------------

def _git(*args: str, root: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def _tracked(root: Path) -> list[str]:
    if (root / ".git").exists():
        return [p for p in _git("ls-files", "-z", root=root).split("\0") if p]
    return [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]


def _text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw[:8192]:
        return None
    try:
        return raw.decode()
    except UnicodeDecodeError:
        return None


def plan(root: Path = REPO) -> dict:
    """What `apply` would do: the moves still to make and every file whose text it would change. Raises `SplitError` when the
    tree cannot be split."""
    vllm = root / VLLM
    if not (vllm / PKG / "program" / "registry").is_dir():
        raise SplitError(f"{root}: no {VLLM}/{PKG}/program/registry (not a verity tree, or the registry has moved: update MOVES)")
    moves = []
    for m in MOVES:
        old, new = vllm / m.old_path, vllm / m.new_path
        if old.exists() and new.exists():
            raise SplitError(f"{m.name}: both {m.old_path} and {m.new_path} exist; reconcile them by hand")
        if not old.exists() and not new.exists():
            raise SplitError(f"{m.name}: neither {m.old_path} nor {m.new_path} exists (moved elsewhere? update MOVES)")
        if not (new.parent).is_dir() and m.package not in NEW_PACKAGES:
            raise SplitError(f"{m.name}: the target package {m.package} does not exist")
        if old.exists():
            moves.append(m)
    for d in DEFINITION_MODULES:
        p = vllm / f"{REGISTRY.replace('.', '/')}/{d}.py"
        if not p.exists():
            raise SplitError(f"Definition module {d} is not at {p.relative_to(root)} (moved? update DEFINITION_MODULES)")
    for p in sorted((vllm / PKG / "program" / "registry").rglob("*.py")):
        t = _text(p) or ""
        for m in MOVES:
            if re.search(rf"^[ \t]*from \.{{1,}}{m.name}\b|^[ \t]*from \.+ import [^\n]*\b{m.name}\b", t, re.M):
                raise SplitError(f"{p.relative_to(root)}: a relative import of {m.name}; make it absolute first")
    edits = {}
    pending = {str(Path(VLLM) / m.old_path): str(Path(VLLM) / m.new_path) for m in moves}
    for rel in _tracked(root):
        if rel.startswith(EXCLUDE) or rel.endswith(EXCLUDE):
            continue
        text = _text(root / rel)
        if text is None:
            continue
        if rel in KEPT:
            if _rewrite_imports(text, rel) != text or any(
                    re.search(rf"^[ \t]*(?:from|import)[ \t]+{re.escape(m.old)}\b", text, re.M) for m in MOVES):
                raise SplitError(f"{rel} is kept as written but imports a moved module; edit it by hand first")
            continue
        new = rewrite(text, rel)
        if rel in pending:
            new = _absolute_relative_imports(new)
        if new != text:
            edits[pending.get(rel, rel)] = new
    return {"moves": moves, "edits": edits}


def apply(root: Path = REPO, log=print) -> dict:
    p = plan(root)
    vllm = root / VLLM
    for pkg, doc in NEW_PACKAGES.items():
        init = vllm / pkg.replace(".", "/") / "__init__.py"
        if not init.exists():
            init.parent.mkdir(parents=True, exist_ok=True)
            init.write_text(doc)
            if (root / ".git").exists():
                _git("add", str(init.relative_to(root)), root=root)
            log(f"created {init.relative_to(root)}")
    for m in p["moves"]:
        old, new = vllm / m.old_path, vllm / m.new_path
        if (root / ".git").exists():
            _git("mv", str(old.relative_to(root)), str(new.relative_to(root)), root=root)
        else:
            old.rename(new)
        log(f"moved {VLLM}/{m.old_path} -> {VLLM}/{m.new_path}")
    for rel, text in sorted(p["edits"].items()):
        (root / rel).write_text(text)
        log(f"rewrote {rel}")
    if not p["moves"] and not p["edits"]:
        log("nothing to do: the dense family is split")
    return p


# --- the invariants ---------------------------------------------------------------------------------------------------------

def _digest(fn) -> str | None:
    """A circuit-check target's canonical descriptor digest: `verity.ml.circuits.codec.program_digest` of the one-call Program at its
    statics (the shell `pipeline.program_graph._encoded` builds), for a primitive and a specialization alike."""
    from verity.ml.circuits import codec
    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
    from verity.ml.circuits.program import Program

    if not isinstance(fn, (SpecializedDefinition, PrimitiveDefinition)):
        return None
    params, ret = fn.signature()
    shell = CompositeDefinition(f"Show[{fn.id}]", 1, (), lambda S: (tuple(params), ret), lambda B, S, *a: B.call(fn, *a),
                                register=False)
    return codec.program_digest(codec.encode_program(Program(bind(shell))))


def family_ids() -> dict[str, str]:
    """Each Definition family (`Name_vN`) a family module defines (not one it imports), by the module that defines it."""
    import importlib

    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition

    out = {}
    for name in DEFINITION_MODULES:
        mod = importlib.import_module(f"{REGISTRY}.{name}")
        tree = ast.parse(Path(mod.__file__).read_text())
        imported = {a.asname or a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        for k, v in vars(mod).items():
            if k not in imported and isinstance(v, (CompositeDefinition, PrimitiveDefinition)):
                out.setdefault(v.id, name)
    return dict(sorted(out.items()))


def snapshot() -> dict:
    """circuit-check's task list on this tree (kind and label of every target, in order), the registered ids, pins.json's digest,
    and every target's descriptor digest."""
    from circuit_check import checks as CK
    from circuit_check import cli
    from circuit_check import targets as TG

    TG.load_registries()
    suite = TG.suite()
    real = TG.suite
    TG.suite = lambda: suite
    try:
        tasks = cli._tasks([], CK.Options(), True, None)
    finally:
        TG.suite = real
    fam = family_ids()
    return {"tasks": [[t[0], t[1]] for t in tasks],
            "registered": list(TG.REGISTERED),
            "root_errors": [list(e) for e in TG.ROOT_ERRORS],
            "pins_sha256": hashlib.sha256((Path(TG.__file__).parent / "pins.json").read_bytes()).hexdigest(),
            "digests": {fid: _digest(fn) for fid, fn in suite.items()},
            "families": {fid: TG.family(fn) for fid, fn in suite.items()},
            "dense_family": fam}


def compare(before: dict, after: dict) -> list[str]:
    out = []
    for key in ("tasks", "registered", "root_errors", "pins_sha256"):
        if before[key] != after[key]:
            b, a = before[key], after[key]
            if isinstance(b, list):
                bs, as_ = {json.dumps(x) for x in b}, {json.dumps(x) for x in a}
                out.append(f"{key}: {len(bs - as_)} gone {sorted(bs - as_)[:8]}, {len(as_ - bs)} new {sorted(as_ - bs)[:8]}"
                           + ("" if bs != as_ else ", same entries in another order"))
            else:
                out.append(f"{key}: {b} -> {a}")
    for fid in sorted(set(before["digests"]) | set(after["digests"])):
        b, a = before["digests"].get(fid), after["digests"].get(fid)
        if b != a:
            out.append(f"digest {fid}: {b} -> {a}")
    if before["dense_family"] != after["dense_family"]:
        out.append(f"dense_family: {before['dense_family']} -> {after['dense_family']}")
    return out


def _snapshot_tree(root: Path, out: Path) -> dict:
    """`snapshot` in the tree ``root``'s own environment."""
    subprocess.run(["uv", "sync", "--all-packages", "--extra", "torch-cpu", "-q"], cwd=root, check=True)
    subprocess.run(["uv", "run", "--no-sync", "python", str(Path(__file__).resolve()), "snapshot", str(out)], cwd=root, check=True)
    return json.loads(out.read_text())


def verify(before_rev: str, out_dir: Path) -> list[str]:
    """Snapshot ``before_rev`` (a fresh worktree) and this tree, and compare."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wt = Path(tempfile.mkdtemp(prefix="split-dense-before-"))
    _git("worktree", "add", "--detach", str(wt), before_rev)
    try:
        before = _snapshot_tree(wt, out_dir / "before.json")
    finally:
        _git("worktree", "remove", "--force", str(wt))
    after = _snapshot_tree(REPO, out_dir / "after.json")
    return compare(before, after)


def pin(snap: dict) -> dict:
    """The fixture: every circuit-check target of a dense-family Definition, with its digest, and the family's templates."""
    fam = set(snap["dense_family"])
    defs = {fid: snap["digests"][fid] for fid, f in snap["families"].items() if f in fam}
    return {"family": "dense", "tool": f"{VLLM}/tools/migrate/split_dense.py", "definitions": dict(sorted(defs.items())),
            "registered": sorted(i for i in snap["registered"] if i in fam)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="list what apply would change")
    sub.add_parser("apply", help="move the twins and vocabulary and rewrite every reference")
    s = sub.add_parser("snapshot", help="write circuit-check's task list and digests on this tree")
    s.add_argument("out", type=Path)
    v = sub.add_parser("verify", help="compare the task list and digests of REV with this tree")
    v.add_argument("--before", required=True)
    v.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "split-dense")
    pn = sub.add_parser("pin", help="write the family's fixture from a snapshot")
    pn.add_argument("snapshot", type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "plan":
            p = plan()
            for m in p["moves"]:
                print(f"move {VLLM}/{m.old_path} -> {VLLM}/{m.new_path} ({m.kind})")
            for rel in sorted(p["edits"]):
                print(f"rewrite {rel}")
            if not p["moves"] and not p["edits"]:
                print("nothing to do: the dense family is split")
        elif a.cmd == "apply":
            apply()
        elif a.cmd == "snapshot":
            a.out.write_text(json.dumps(snapshot(), indent=1, sort_keys=True) + "\n")
        elif a.cmd == "verify":
            diffs = verify(a.before, a.out)
            for d in diffs:
                print(d)
            if diffs:
                print(f"FAIL: {len(diffs)} differences between {a.before} and this tree (snapshots in {a.out})")
                return 1
            snap = json.loads((a.out / "after.json").read_text())
            print(f"ok: {len(snap['tasks'])} circuit-check tasks, {len(snap['registered'])} registered ids and "
                  f"{sum(d is not None for d in snap['digests'].values())} target digests are the same on {a.before} and this tree")
        elif a.cmd == "pin":
            path = REPO / FIXTURE
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pin(json.loads(a.snapshot.read_text())), indent=1) + "\n")
            print(f"wrote {FIXTURE}")
    except SplitError as e:
        print(f"split_dense: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
