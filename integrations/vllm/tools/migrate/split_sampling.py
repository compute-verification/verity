"""Split the sampling family's registry modules into Definitions, twins and vocabulary (the catalog split, one family per PR).

    uv run python integrations/vllm/tools/migrate/split_sampling.py            # split, rewrite every reference, verify
    uv run python integrations/vllm/tools/migrate/split_sampling.py --no-verify  # the same without the before/after snapshots
    uv run python integrations/vllm/tools/migrate/split_sampling.py --check      # exit 0 iff the tree is already split

The Definitions stay where they are, in `verity_vllm/program/registry/` (`sampling`, `sampling_boolean`, `gumbel_boolean`,
`topp_keep_builder`, `topp_split`, `topp_word_gates`, `topp_words`), the directory the catalog move maps to `catalog/`.  What
moves out of the registry:

* the twins, to `verity_vllm/program/kernels/` beside the other row twins and the row harness that calls them:
  `sampling_rows` (numpy twins of the sampler's Definitions) and `sampling_topp_difftest` (the G4 difftest adapter against
  vLLM's production sampler);
* the vocabulary, to `verity_vllm/program/sampling_vocab.py`: `SHARED_GREEDY` and `select_for`, which Definition a target's
  `sampler_construction` binds for a stochastic select.

Every reference in the tree follows: imports (module-level and function-local, `from` and plain), attribute uses of the
vocabulary through an alias of `registry.sampling`, dotted names and paths in strings, comments, docs and the lint allowlists,
and `tests/test_backend_boundaries.py`'s `KNOWN` sites.  The script reruns on a later main: an already-split tree is left as it
is (its references are rewritten again, which finds nothing new unless a later commit added one), and any state it does not
recognise (a twin in both places, the vocabulary in neither) fails by name, as does a reference it cannot rewrite.

`--verify` (the default) snapshots, before and after, every canonical descriptor digest of the family's Definitions at every
statics circuit-check's task list, its pins and the served bindings use, and circuit-check's whole task list (target id,
statics, pins), in fresh processes; the split fails if either differs.  The family's part of both is written to
`tests/program/data/split_sampling_digests.json` and `tools/circuit_check/tests/data/split_sampling_tasks.json`, which
`tests/program/test_split_sampling.py` and circuit-check's `test_split_sampling.py` replay.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
VLLM = REPO / "integrations" / "vllm"
PKG = VLLM / "verity_vllm"
REG = "verity_vllm.program.registry"
KERNELS = "verity_vllm.program.kernels"
VOCAB_MODULE = "verity_vllm.program.sampling_vocab"
SAMPLING = f"{REG}.sampling"

DEFINITION_MODULES = ("sampling", "sampling_boolean", "gumbel_boolean", "topp_keep_builder", "topp_split", "topp_word_gates",
                      "topp_words")
#: twin module -> the package it moves to
TWINS = {"sampling_rows": KERNELS, "sampling_topp_difftest": KERNELS}
VOCAB = ("SHARED_GREEDY", "select_for")
#: Definition ids the family makes on demand (`registry/__init__.py`'s lazy families), not module attributes
LAZY_IDS = re.compile(r"^(TopPMaskWordx\d+|BitAtx\d+)_v1$")
#: the vocabulary sizes of the served models (workload configs and manifests) the word-level selects are bound at
SERVED_V = (128256, 151936, 152064, 256000, 154880, 50304, 49152, 32768, 32064, 32000)
DIGESTS_OUT = VLLM / "tests" / "program" / "data" / "split_sampling_digests.json"
TASKS_OUT = REPO / "tools" / "circuit_check" / "tests" / "data" / "split_sampling_tasks.json"
SELF_DIR = Path(__file__).resolve().parent
TEXT_SUFFIXES = {".py", ".json", ".md", ".sh", ".toml", ".txt", ".yaml", ".yml", ".cfg", ".ini", ".lean", ".rst"}


class SplitError(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"split_sampling: {msg}")


def _path(module: str) -> Path:
    return REPO / "integrations" / "vllm" / (module.replace(".", "/") + ".py")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


# --- state ----------------------------------------------------------------------------------------------------------------------

def _sampling_defines_vocab() -> bool:
    tree = ast.parse(_path(SAMPLING).read_text(encoding="utf-8"))
    return any(n in VOCAB for n in _top_names(tree))


def _top_names(tree: ast.Module) -> dict[str, ast.stmt]:
    out: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node
    return out


def state() -> dict[str, str]:
    """Each piece of the split as `before`, `after`, or a reason the tree is in neither state (fails the split)."""
    out = {}
    for m in DEFINITION_MODULES:
        if not _path(f"{REG}.{m}").exists():
            out[m] = f"the Definition module registry/{m}.py is missing"
    for twin, dst in TWINS.items():
        old, new = _path(f"{REG}.{twin}").exists(), _path(f"{dst}.{twin}").exists()
        out[twin] = "before" if old and not new else "after" if new and not old else (
            f"both registry/{twin}.py and {dst}.{twin} exist" if old else f"neither registry/{twin}.py nor {dst}.{twin} exists")
    old, new = _sampling_defines_vocab(), _path(VOCAB_MODULE).exists()
    out["vocab"] = "before" if old and not new else "after" if new and not old else (
        f"registry/sampling.py defines {VOCAB} and {VOCAB_MODULE} exists" if old else f"{VOCAB} are in neither module")
    return out


# --- moving the vocabulary -------------------------------------------------------------------------------------------------------

VOCAB_HEAD = '''"""Which sampler Definition a target binds: each stochastic select under the `TargetProfile.sampler_construction` it is bound
under.  The Definitions are `registry.sampling`'s; this is the integration's vocabulary over them (`sampling_event.CONSTRUCTION_OF`
maps the other way, from a family to its construction)."""
from __future__ import annotations

'''


def _span(lines: list[str], node: ast.stmt) -> tuple[int, int]:
    """0-based [start, end) line range of a top-level statement with the `#:` comment lines right above it."""
    start = (node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno) - 1
    while start > 0 and lines[start - 1].lstrip().startswith("#:"):
        start -= 1
    return start, node.end_lineno


def move_vocab() -> None:
    src_path = _path(SAMPLING)
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    top = _top_names(tree)
    missing = [n for n in VOCAB if n not in top]
    if missing:
        raise SplitError(f"registry/sampling.py defines {[n for n in VOCAB if n in top]} but not {missing}: split by hand")
    lines = [ln.decode() for ln in src.encode().splitlines(keepends=True)]
    spans = sorted({_span(lines, top[n]) for n in VOCAB})
    moved = "".join("".join(lines[a:b]).rstrip("\n") + "\n\n\n" for a, b in spans).rstrip("\n") + "\n"
    moved_tree = ast.parse(moved)
    bound = set(_top_names(moved_tree))
    free = sorted({n.id for n in ast.walk(moved_tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                  & (set(top) - bound))
    keep = [i for i, _ in enumerate(lines) if not any(a <= i < b for a, b in spans)]
    rest = "".join(lines[i] for i in keep)
    rest = re.sub(r"\n{4,}", "\n\n\n", rest)
    left = ast.parse(rest)
    still = sorted({n.id for n in ast.walk(left) if isinstance(n, ast.Name) and n.id in VOCAB})
    if still:
        raise SplitError(f"registry/sampling.py still reads {still} after the move: a Definition depends on the vocabulary")
    rest = rest.replace("(`TargetProfile.sampler_construction`, `select_for`)",
                        "(`TargetProfile.sampler_construction`, `sampling_vocab.select_for`)")
    imports = f"from {SAMPLING} import {', '.join(free)}\n\n\n" if free else ""
    body = VOCAB_HEAD + imports + moved + f"\n\n__all__ = {json.dumps(list(VOCAB))}\n"
    _path(VOCAB_MODULE).write_text(body, encoding="utf-8")
    src_path.write_text(rest, encoding="utf-8")
    _git("add", str(_path(VOCAB_MODULE).relative_to(REPO)))


# --- rewriting references --------------------------------------------------------------------------------------------------------

def _alias_text(a: ast.alias) -> str:
    return a.name if a.asname is None else f"{a.name} as {a.asname}"


def _from(module: str, names: list[ast.alias]) -> str:
    return f"from {module} import {', '.join(_alias_text(a) for a in names)}"


def _twin_of(module: str) -> tuple[str, str] | None:
    """(old prefix, new prefix) when `module` is a moved twin or inside one."""
    for twin, dst in TWINS.items():
        old = f"{REG}.{twin}"
        if module == old or module.startswith(old + "."):
            return old, f"{dst}.{twin}"
    return None


class _Edits:
    def __init__(self, src: str):
        self.lines = src.encode().splitlines(keepends=True)     # bytes split as `ast` counts lines; columns are UTF-8 offsets
        self.edits: list[tuple[int, int, int, int, str]] = []

    def replace(self, node: ast.AST, text: str) -> None:
        self.edits.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, text))

    def apply(self) -> str:
        for l0, c0, l1, c1, text in sorted(self.edits, reverse=True):
            a, b = self.lines[l0 - 1], self.lines[l1 - 1]
            joined = a[:c0] + text.encode() + b[c1:]
            self.lines[l0 - 1:l1] = [joined]
        return b"".join(self.lines).decode()


def rewrite_python(src: str) -> str:
    tree = ast.parse(src)
    E = _Edits(src)
    aliases: dict[str, list[ast.stmt]] = {}      # name bound to `registry.sampling` -> the import statements binding it
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mod = node.module
            if mod == REG:
                twins = [a for a in node.names if a.name in TWINS]
                for a in node.names:
                    if a.name == "sampling":
                        aliases.setdefault(a.asname or a.name, []).append(node)
                if twins:
                    stay = [a for a in node.names if a.name not in TWINS]
                    by_dst: dict[str, list[ast.alias]] = {}
                    for a in twins:
                        by_dst.setdefault(TWINS[a.name], []).append(a)
                    stmts = ([_from(REG, stay)] if stay else []) + [_from(d, ns) for d, ns in by_dst.items()]
                    E.replace(node, ("\n" + " " * node.col_offset).join(stmts))
            elif _twin_of(mod):
                old, new = _twin_of(mod)
                E.replace(node, _from(new + mod[len(old):], node.names))
            elif mod == SAMPLING and any(a.name in VOCAB for a in node.names):
                stay = [a for a in node.names if a.name not in VOCAB]
                moved = [a for a in node.names if a.name in VOCAB]
                stmts = ([_from(SAMPLING, stay)] if stay else []) + [_from(VOCAB_MODULE, moved)]
                E.replace(node, ("\n" + " " * node.col_offset).join(stmts))
        elif isinstance(node, ast.Import):
            new_names, changed = [], False
            for a in node.names:
                t = _twin_of(a.name)
                if t:
                    new_names.append(ast.alias(t[1] + a.name[len(t[0]):], a.asname))
                    changed = True
                else:
                    new_names.append(a)
                if a.name == SAMPLING and a.asname:
                    aliases.setdefault(a.asname, []).append(node)
            if changed:
                if any(a.asname is None and _twin_of(a.name) for a in node.names):
                    raise SplitError(f"line {node.lineno}: `import {', '.join(a.name for a in node.names)}` without `as`: its uses "
                                     "name the old path; rewrite it by hand")
                E.replace(node, "import " + ", ".join(_alias_text(a) for a in new_names))
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in VOCAB and isinstance(n.value, ast.Name)
            and n.value.id in aliases]
    if uses:
        for n in uses:
            E.replace(n, f"sampling_vocab.{n.attr}")
        users = {n.value.id for n in uses}
        for alias in sorted(users):
            for stmt in aliases[alias]:
                eol = len(E.lines[stmt.end_lineno - 1].rstrip(b"\r\n"))     # after a trailing comment, which stays on its line
                E.edits.append((stmt.end_lineno, eol, stmt.end_lineno, eol,
                                "\n" + " " * stmt.col_offset + "from verity_vllm.program import sampling_vocab"))
    return E.apply() if E.edits else src


def _text_rules() -> list[tuple[re.Pattern, str]]:
    rules = []
    for twin, dst in TWINS.items():
        new_dir = dst.removeprefix("verity_vllm.program.")
        rules += [(re.compile(rf"\bverity_vllm\.program\.registry\.{twin}\b"), f"{dst}.{twin}"),
                  (re.compile(rf"\bverity_vllm/program/registry/{twin}\b"), f"verity_vllm/program/{new_dir}/{twin}"),
                  (re.compile(rf"(?<![\w/.])registry/{twin}\.py\b"), f"{new_dir}/{twin}.py"),
                  (re.compile(rf"(?<![\w.])registry\.{twin}\b"), f"{new_dir}.{twin}")]
    for name in VOCAB:
        rules += [(re.compile(rf"\b(?:verity_vllm\.program\.)?registry\.sampling\.{name}\b"), f"sampling_vocab.{name}")]
    return rules


def _tracked_text_files() -> list[Path]:
    out = []
    for rel in _git("ls-files", "-z").split("\0"):
        if not rel:
            continue
        p = REPO / rel
        if p.suffix not in TEXT_SUFFIXES or not p.is_file() or SELF_DIR in p.parents:
            continue
        if p in (DIGESTS_OUT, TASKS_OUT):
            continue
        out.append(p)
    return out


def rewrite_references() -> list[str]:
    rules = _text_rules()
    changed = []
    for p in _tracked_text_files():
        try:
            src = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = src
        if p.suffix == ".py":
            try:
                new = rewrite_python(new)
            except SyntaxError:
                pass
        for pat, rep in rules:
            new = pat.sub(rep, new)
        if new != src:
            p.write_text(new, encoding="utf-8")
            changed.append(p.relative_to(REPO).as_posix())
    return changed


def leftovers() -> list[str]:
    """References the rewrite could not follow: each fails the split by name."""
    pats = [re.compile(rf"\bregistry[./]{t}\b") for t in TWINS]
    out = []
    for p in _tracked_text_files():
        try:
            src = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = p.relative_to(REPO).as_posix()
        for i, line in enumerate(src.splitlines(), 1):
            if any(q.search(line) for q in pats):
                out.append(f"{rel}:{i}: {line.strip()[:160]}")
        if p.suffix == ".py":
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            aliases = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == REG
                       for a in n.names if a.name == "sampling"}
            aliases |= {a.asname for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
                        if a.name == SAMPLING and a.asname}
            for n in ast.walk(tree):
                if isinstance(n, ast.Attribute) and n.attr in VOCAB and (
                        (isinstance(n.value, ast.Name) and n.value.id in aliases) or ast.unparse(n.value).endswith(SAMPLING)):
                    out.append(f"{rel}:{n.lineno}: {ast.unparse(n)} (the vocabulary through registry.sampling)")
                if isinstance(n, ast.ImportFrom) and n.module == SAMPLING and any(a.name in VOCAB for a in n.names):
                    out.append(f"{rel}:{n.lineno}: imports {VOCAB} from registry.sampling")
    return out


def update_backend_known() -> bool:
    """Bring `tests/test_backend_boundaries.py`'s `KNOWN` in line with the backends' imports after the rewrite."""
    import importlib.util

    path = REPO / "tests" / "test_backend_boundaries.py"
    spec = importlib.util.spec_from_file_location("_backend_boundaries", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sites = mod._sites()
    want = (set(mod.KNOWN) | (sites - mod.KNOWN)) - {s for s in mod.KNOWN - sites if (REPO / s[0]).exists()}
    if want == set(mod.KNOWN):
        return False
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "KNOWN" for t in n.targets))
    lines = [ln.decode() for ln in src.encode().splitlines(keepends=True)]
    block = "".join(lines[node.lineno - 1:node.end_lineno])
    if "#" in block:
        raise SplitError("tests/test_backend_boundaries.py: KNOWN holds comments; update it by hand")
    body = "KNOWN = {\n" + "".join(f'    ("{f}", "{m}"),\n' for f, m in sorted(want)) + "}\n"
    path.write_text("".join(lines[:node.lineno - 1]) + body + "".join(lines[node.end_lineno:]), encoding="utf-8")
    return True


# --- the before/after snapshot -----------------------------------------------------------------------------------------------------

def _family_modules() -> list[str]:
    mods = [f"{REG}.{m}" for m in DEFINITION_MODULES]
    for twin, dst in TWINS.items():
        mods.append(f"{REG}.{twin}" if _path(f"{REG}.{twin}").exists() else f"{dst}.{twin}")
    return mods


def _direct_imports(module) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            out.add(n.module)
            out.update(f"{n.module}.{a.name}" for a in n.names)
        elif isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
    return out


def owned_definitions() -> dict[str, object]:
    """id -> Definition for every Definition the family's modules define (not one they import from another module)."""
    import importlib

    from verity.ir.defs import CompositeDefinition, PrimitiveDefinition

    family = _family_modules()
    out: dict[str, object] = {}
    for name in family:
        m = importlib.import_module(name)
        foreign = [sys.modules[d] for d in _direct_imports(m) if d in sys.modules and d not in family]
        for v in vars(m).values():
            if isinstance(v, (PrimitiveDefinition, CompositeDefinition)) and not any(
                    any(x is v for x in vars(f).values()) for f in foreign):
                out.setdefault(v.id, v)
    return out


def _owned_id(i: str, owned: dict) -> bool:
    return i in owned or bool(LAZY_IDS.match(i))


def _family_entries(desc: dict, owned: dict) -> dict[str, dict]:
    from verity.ir.annotations import definition_digest

    out = {}
    for fid, e in desc["definitions"].items():
        if _owned_id(e["id"], owned):
            out[fid] = {"id": e["id"], "bindings": e.get("bindings", {}), "digest": definition_digest(desc, fid)}
    return out


def _wrap(prim):
    """A static-free composite calling one primitive, so its descriptor entry can be encoded."""
    from verity.ir.defs import CompositeDefinition

    return CompositeDefinition(f"SplitSamplingWrap[{prim.id}]", 1, (), lambda S: (tuple(prim.params), prim.ret),
                               lambda B, S, *a: B.call(prim, *a), conformance="derived: split-sampling probe", register=False)


def served_roots(owned: dict) -> list:
    from verity.ir.defs import bind

    roots = []
    for fam in ("GumbelTokenSelect_v1", "GumbelTopPTokenSelect_v1", "GumbelTokenSelectSharedGreedy_v1",
                "GumbelTopPTokenSelectSharedGreedy_v1"):
        roots += [bind(owned[fam], V=v) for v in SERVED_V]
    roots += [bind(owned["TokenSelect_v2"], V=v) for v in (128256, 151936)]
    roots += [bind(owned["GumbelSelectF32_v2"], V=128256)]
    roots += [bind(owned["GumbelTopPTokenSelect_v2"], V=128256, S=32)]
    roots += [bind(owned["GumbelTopPTokenSelectDifftest_v1"], V=v, S=s) for v in (128256, 1024, 1025, 3000)
              for s in (1, 2, 4, 8, 16, 32)]
    return roots


def snapshot() -> dict:
    """The digests of every family Definition at every statics a root reaches, and circuit-check's task list and pins."""
    from verity.ir.codec import encode_program
    from verity.ir.defs import PrimitiveDefinition
    from verity.ir.program import Program

    from circuit_check import checks as CK
    from circuit_check import cli
    from circuit_check import targets as TG

    owned = owned_definitions()
    suite = TG.suite()
    TG.suite = lambda: suite                    # `_tasks` builds the same suite again otherwise (a minute)
    tasks = cli._tasks([], CK.Options(), True, None)
    pins = CK.load_pins()
    entries: dict[str, dict] = {}
    unencodable = []

    def visit(fn, label):
        try:
            desc = encode_program(Program(_wrap(fn) if isinstance(fn, PrimitiveDefinition) else fn))
        except Exception as e:  # noqa: BLE001 - a root circuit-check itself fails to encode is not this split's
            unencodable.append(f"{label}: {type(e).__name__}")
            return
        for fid, e in _family_entries(desc, owned).items():
            if fid in entries and entries[fid]["digest"] != e["digest"]:
                raise SplitError(f"{fid}: two descriptor digests in one tree ({label})")
            entries[fid] = e

    for fid, fn in suite.items():
        visit(fn, fid)
    for spec in pins["definitions"]:
        base = spec.split("{", 1)[0]
        if _owned_id(base, owned):
            visit(TG.definition(spec), spec)
    for d in owned.values():
        if isinstance(d, PrimitiveDefinition) or not getattr(d, "statics", ()):
            visit(d, d.id)
    for fn in served_roots(owned):
        visit(fn, fn.id)
    templates = [{"label": label, "template": sub.template.name, "instance_statics": sts}
                 for label, sub, sts in TG.templates()]
    return {"owned": sorted(owned), "entries": dict(sorted(entries.items())), "unencodable": sorted(unencodable),
            "tasks": [[kind, label] for kind, label, _ in tasks], "templates": templates, "pins": pins}


def _family_tasks(snap: dict) -> dict:
    owned = set(snap["owned"])
    fam = lambda spec: _owned_id(spec.split("{", 1)[0], owned)  # noqa: E731
    return {
        "definitions": [label for kind, label in snap["tasks"] if kind == "definition" and fam(label)],
        "templates": [t for t in snap["templates"] if _owned_id(t["template"], owned)],
        "pins": {k: {i: v for i, v in snap["pins"].get(k, {}).items() if fam(i)} for k in ("definitions", "pieces")},
    }


def _snapshot_subprocess(out: Path) -> dict:
    env = dict(os.environ)
    r = subprocess.run([sys.executable, __file__, "--snapshot", str(out)], cwd=REPO, env=env)
    if r.returncode != 0:
        raise SplitError(f"the snapshot failed ({r.returncode})")
    return json.loads(out.read_text())


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _compare(before: dict, after: dict) -> list[str]:
    bad = []
    if before["owned"] != after["owned"]:
        bad.append(f"the family's Definition ids moved: -{sorted(set(before['owned']) - set(after['owned']))} "
                   f"+{sorted(set(after['owned']) - set(before['owned']))}")
    for fid, e in before["entries"].items():
        got = after["entries"].get(fid)
        if got != e:
            bad.append(f"{fid}: {e['digest'][:16]} -> {got and got['digest'][:16]}")
    bad += [f"{fid}: new after the split" for fid in after["entries"].keys() - before["entries"].keys()]
    for k in ("tasks", "templates", "pins"):
        if before[k] != after[k]:
            bad.append(f"circuit-check's {k} differ")
    return bad


# --- main ----------------------------------------------------------------------------------------------------------------------

def split(verify: bool) -> None:
    st = state()
    broken = {k: v for k, v in st.items() if v not in ("before", "after")}
    if broken:
        raise SplitError("the tree is in neither state:\n  " + "\n  ".join(f"{k}: {v}" for k, v in broken.items()))
    tmp = Path(tempfile.mkdtemp(prefix="split-sampling-"))
    before = _snapshot_subprocess(tmp / "before.json") if verify else None
    for twin, dst in TWINS.items():
        if st[twin] == "before":
            _git("mv", str(_path(f"{REG}.{twin}").relative_to(REPO)), str(_path(f"{dst}.{twin}").relative_to(REPO)))
            print(f"moved registry/{twin}.py -> {dst.removeprefix('verity_vllm.program.')}/{twin}.py")
    if st["vocab"] == "before":
        move_vocab()
        print(f"moved {VOCAB} -> {VOCAB_MODULE}")
    changed = rewrite_references()
    print(f"rewrote references in {len(changed)} file(s)" + "".join(f"\n  {c}" for c in changed))
    if update_backend_known():
        print("updated tests/test_backend_boundaries.py KNOWN")
    left = leftovers()
    if left:
        raise SplitError("references the rewrite could not follow:\n  " + "\n  ".join(left))
    if verify:
        after = _snapshot_subprocess(tmp / "after.json")
        bad = _compare(before, after)
        if bad:
            raise SplitError("the split changed what it must not:\n  " + "\n  ".join(bad[:50]))
        _write_json(DIGESTS_OUT, {"owned": before["owned"], "entries": before["entries"]})
        _write_json(TASKS_OUT, _family_tasks(before))
        print(f"verified: {len(before['entries'])} descriptor digests over {len(before['owned'])} Definitions, "
              f"{len(before['tasks'])} circuit-check tasks unchanged; wrote {DIGESTS_OUT.relative_to(REPO)}, "
              f"{TASKS_OUT.relative_to(REPO)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 0 iff the tree is already split")
    ap.add_argument("--no-verify", action="store_true", help="skip the before/after snapshots")
    ap.add_argument("--snapshot", metavar="OUT", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.snapshot:
        _write_json(Path(a.snapshot), snapshot())
        return 0
    if a.check:
        st = state()
        print(json.dumps(st, indent=1))
        return 0 if all(v == "after" for k, v in st.items() if k in TWINS or k == "vocab") and not leftovers() else 1
    split(verify=not a.no_verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
