"""Split the sampling family's registry modules into Definitions, twins and vocabulary (the catalog split, one family per PR).

    uv run python integrations/vllm/tools/migrate/split_sampling.py            # split, rewrite every reference, verify
    uv run python integrations/vllm/tools/migrate/split_sampling.py --no-verify  # the same without the before/after snapshots
    uv run python integrations/vllm/tools/migrate/split_sampling.py --check      # exit 0 iff the tree is already split

The Definitions stay where they are, in `verity_vllm/program/registry/` (`sampling`, `sampling_boolean`, `gumbel_boolean`,
`topp_keep_builder`, `topp_split`, `topp_word_gates`, `topp_words`, and `sampling_topp_difftest`'s Definition with its `spec`
and `spec_registered`), the directory the catalog move maps to `catalog/`.  What moves out of the registry:

* the twin, to `verity_vllm/program/kernels/` beside the other row twins (the move map's `verity/kernels/`): `sampling_rows`
  (numpy twins of the sampler's Definitions);
* the harness, to `verity_vllm/properties/sampling_topp_admission.py` (the admission difftests' package; the move map's
  `properties/`): `sampling_topp_difftest`'s G4 adapter, its corpus, `cases` and vLLM's `production` chain, with the module
  docstring that names the `--adapter`;
* the vocabulary, to `verity_vllm/program/sampling_vocab.py`: `SHARED_GREEDY` and `select_for`, which Definition a target's
  `sampler_construction` binds for a stochastic select.

Every reference in the tree follows: imports (module-level and function-local, `from` and plain), attribute uses of the
vocabulary through an alias of `registry.sampling`, dotted names and paths in strings, comments, docs and the lint allowlists,
and `tests/test_backend_boundaries.py`'s `KNOWN` sites.  The script reruns on a later main: an already-split tree is left as it
is (its references are rewritten again, which finds nothing new unless a later commit added one), and any state it does not
recognise (a twin in both places, the vocabulary in neither) fails by name, as does a reference it cannot rewrite (an
attribute use of a harness name through the Definition module, for one).

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
DIFFTEST = f"{REG}.sampling_topp_difftest"
HARNESS_MODULE = "verity_vllm.properties.sampling_topp_admission"

DEFINITION_MODULES = ("sampling", "sampling_boolean", "gumbel_boolean", "topp_keep_builder", "topp_split", "topp_word_gates",
                      "topp_words", "sampling_topp_difftest")
#: twin module -> the package it moves to
TWINS = {"sampling_rows": KERNELS}
VOCAB = ("SHARED_GREEDY", "select_for")
#: the G4 admission adapter's own names in `sampling_topp_difftest` (its corpus and vLLM's production chain); the Definition,
#: `spec` and `spec_registered` stay, and the adapter imports `spec` back (`properties.admission` reads `mod.spec`)
HARNESS = ("SEMANTIC_ID", "VOCABS", "KINDS", "TEMPS", "TOPPS", "SEEDS", "POSS", "_f32_bits", "_real_rows", "cases", "production")
HARNESS_REEXPORT = ("spec",)
#: module -> (names that leave it, the module they go to)
MOVED_NAMES = {SAMPLING: (VOCAB, VOCAB_MODULE), DIFFTEST: (HARNESS, HARNESS_MODULE)}
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

def _defines_any(module: str, names: tuple[str, ...]) -> bool:
    tree = ast.parse(_path(module).read_text(encoding="utf-8"))
    return any(n in names for n in _top_names(tree))


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
    for key, src, names, dst in (("vocab", SAMPLING, VOCAB, VOCAB_MODULE), ("harness", DIFFTEST, HARNESS, HARNESS_MODULE)):
        if not _path(src).exists():
            continue
        old, new = _defines_any(src, names), _path(dst).exists()
        out[key] = "before" if old and not new else "after" if new and not old else (
            f"{src} defines some of {names} and {dst} exists" if old else f"{names} are in neither {src} nor {dst}")
    return out


# --- moving names out of a module --------------------------------------------------------------------------------------------------

VOCAB_DOC = '''"""Which sampler Definition a target binds: each stochastic select under the `TargetProfile.sampler_construction` it is bound
under.  The Definitions are `registry.sampling`'s; this is the integration's vocabulary over them (`sampling_event.CONSTRUCTION_OF`
maps the other way, from a family to its construction)."""
'''

DIFFTEST_DOC = '''"""`GumbelTopPTokenSelectDifftest_v1{V,S}`: `GumbelTopPTokenSelect_v1{V}` returning also the masked fp32 row it samples from, at the
static split count S a launch used; `spec` binds it and `spec_registered` binds the registered family.  The Match-time stoch
recompute (`check.replay.stoch_recompute`), the row harness (`kernels.rows`) and the G4 admission adapter
(`verity_vllm.properties.sampling_topp_admission`) evaluate these."""
'''


def _span(lines: list[str], node: ast.stmt) -> tuple[int, int]:
    """0-based [start, end) line range of a top-level statement with the `#:` comment lines right above it."""
    start = (node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno) - 1
    while start > 0 and lines[start - 1].lstrip().startswith("#:"):
        start -= 1
    return start, node.end_lineno


def _loads(tree: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _imported(tree: ast.Module) -> dict[str, tuple[str | None, ast.alias]]:
    """name -> (module of a `from` import, or None for a plain import; the alias) for each module-level import binding it."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module != "__future__":
            if node.level:
                raise SplitError(f"a relative import at line {node.lineno}: split by hand")
            out.update({a.asname or a.name: (node.module, a) for a in node.names})
        elif isinstance(node, ast.Import):
            out.update({a.asname or a.name.split(".")[0]: (None, a) for a in node.names})
    return out


def _import_block(wanted: list[tuple[str | None, ast.alias]]) -> str:
    """Import lines for (module, alias) pairs, in the given order: the standard library, then other distributions, then ours."""
    groups: list[dict[str | None, list[ast.alias]]] = [{}, {}, {}]
    for mod, a in wanted:
        top = (mod or a.name).split(".")[0]
        g = 0 if top in sys.stdlib_module_names else 2 if top.startswith("verity") else 1
        groups[g].setdefault(mod, []).append(a)
    out = []
    for g in groups:
        lines = [f"import {_alias_text(a)}" for a in g.get(None, [])] + [_from(m, al) for m, al in g.items() if m]
        if lines:
            out.append("\n".join(lines) + "\n")
    return "\n".join(out)


def move_names(src_mod: str, names: tuple[str, ...], dst_mod: str, *, dst_doc: str | None = None, src_doc: str | None = None,
               reexport: tuple[str, ...] = ()) -> None:
    """Move the top-level statements that define `names` (with their `#:` comment lines) from `src_mod` into a new module `dst_mod`.
    The new module imports what the moved code reads: the source's own imports of those names, and the source's other top-level
    names from the source (with `reexport`, names the new module offers that stay in the source).  With `dst_doc` None the
    source's module docstring moves too and `src_doc` replaces it.  Imports only the moved code read leave the source."""
    src_path = _path(src_mod)
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    top = _top_names(tree)
    missing = [n for n in names if n not in top]
    if missing:
        raise SplitError(f"{src_mod} defines {[n for n in names if n in top]} but not {missing}: split by hand")
    nodes = {id(top[n]) for n in names}
    extra = sorted(k for k, v in top.items() if id(v) in nodes and k not in names)
    if extra:
        raise SplitError(f"{src_mod}: a moved statement also binds {extra}: split by hand")
    lines = [ln.decode() for ln in src.encode().splitlines(keepends=True)]
    spans: list[tuple[int, int]] = []
    for a, b in sorted({_span(lines, top[n]) for n in names}):
        if spans and all(not ln.strip() for ln in lines[spans[-1][1]:a]):
            spans[-1] = (spans[-1][0], b)               # only blank lines between: the source's own spacing stays
        else:
            spans.append((a, b))
    moved = "".join("".join(lines[a:b]).rstrip("\n") + "\n\n\n" for a, b in spans).rstrip("\n") + "\n"
    moved_tree = ast.parse(moved)
    doc = tree.body[0] if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) \
        and isinstance(tree.body[0].value.value, str) else None
    if dst_doc is None:
        if doc is None or src_doc is None:
            raise SplitError(f"{src_mod}: moving its docstring needs both the docstring and its replacement")
        dst_doc = "".join(lines[doc.lineno - 1:doc.end_lineno])
        spans.append((doc.lineno - 1, doc.end_lineno))
    imported = _imported(tree)
    free = _loads(moved_tree) & (set(top) | set(imported)) - set(names)     # a local shadowing a module name costs one import
    wanted = [imported[n] for n in imported if n in free]
    own = [n for n in top if n in free and n not in imported and n not in names] + [n for n in reexport if n not in free]
    wanted += [(src_mod, ast.alias(n, None)) for n in own]
    keep = [i for i, _ in enumerate(lines) if not any(a <= i < b for a, b in spans)]
    rest = "".join(lines[i] for i in keep)
    if dst_doc is not None and doc is not None and (doc.lineno - 1, doc.end_lineno) in spans:
        rest = src_doc + rest
    rest = _drop_imports(rest, {n for n in imported if n in free})
    rest = re.sub(r"\n{4,}", "\n\n\n", rest)
    still = sorted(_loads(ast.parse(rest)) & set(names))
    if still:
        raise SplitError(f"{src_mod} still reads {still} after the move: what stays depends on what moves")
    future = "from __future__ import annotations\n\n" if any(
        isinstance(n, ast.ImportFrom) and n.module == "__future__" for n in tree.body) else ""
    public = [n for n in (*names, *reexport) if not n.startswith("_")]
    body = (dst_doc + future + _import_block(wanted) + "\n\n" + moved + f"\n\n__all__ = {json.dumps(public)}\n")
    _path(dst_mod).write_text(body, encoding="utf-8")
    src_path.write_text(rest, encoding="utf-8")
    _git("add", str(_path(dst_mod).relative_to(REPO)))


def _drop_imports(src: str, candidates: set[str]) -> str:
    """`src` without its module-level import aliases among `candidates` that nothing in it reads any more."""
    tree = ast.parse(src)
    unused = candidates - _loads(tree)
    E = _Edits(src)
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or getattr(node, "module", None) == "__future__":
            continue
        bound = lambda a: a.asname or (a.name if isinstance(node, ast.ImportFrom) else a.name.split(".")[0])  # noqa: E731
        stay = [a for a in node.names if bound(a) not in unused]
        if len(stay) == len(node.names):
            continue
        if stay:
            E.replace(node, _from(node.module, stay) if isinstance(node, ast.ImportFrom) else
                      "import " + ", ".join(_alias_text(a) for a in stay))
        else:
            E.edits.append((node.lineno, 0, node.end_lineno + 1, 0, ""))
    if not E.edits:
        return src
    out = E.apply()
    imports = [n for n in ast.parse(out).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not imports:
        return out
    lines = out.splitlines(keepends=True)
    a, b = imports[0].lineno - 1, imports[-1].end_lineno
    block = re.sub(r"\n{3,}", "\n\n", "".join(lines[a:b]))     # a removed import leaves no run of blank lines among the rest
    return "".join(lines[:a]) + block + "".join(lines[b:])


def move_vocab() -> None:
    move_names(SAMPLING, VOCAB, VOCAB_MODULE, dst_doc=VOCAB_DOC)
    p = _path(SAMPLING)
    p.write_text(p.read_text(encoding="utf-8").replace("(`TargetProfile.sampler_construction`, `select_for`)",
                                                       "(`TargetProfile.sampler_construction`, `sampling_vocab.select_for`)"),
                 encoding="utf-8")


def move_harness() -> None:
    move_names(DIFFTEST, HARNESS, HARNESS_MODULE, src_doc=DIFFTEST_DOC, reexport=HARNESS_REEXPORT)


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
            elif mod in MOVED_NAMES and any(a.name in MOVED_NAMES[mod][0] for a in node.names):
                names, dst = MOVED_NAMES[mod]
                stay = [a for a in node.names if a.name not in names]
                moved = [a for a in node.names if a.name in names]
                stmts = ([_from(mod, stay)] if stay else []) + [_from(dst, moved)]
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
    rules += [(re.compile(rf"(--adapter[ =]){re.escape(DIFFTEST)}\b"), rf"\g<1>{HARNESS_MODULE}")]
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
    pats = [re.compile(rf"\bregistry[./]{t}\b") for t in TWINS] + [re.compile(rf"--adapter[ =]{re.escape(DIFFTEST)}\b")]
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
                if isinstance(n, ast.ImportFrom) and n.module == DIFFTEST and any(a.name in HARNESS for a in n.names):
                    out.append(f"{rel}:{n.lineno}: imports the admission adapter's names from {DIFFTEST}")
            difftest = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == REG
                        for a in n.names if a.name == "sampling_topp_difftest"}
            difftest |= {a.asname for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
                         if a.name == DIFFTEST and a.asname}
            for n in ast.walk(tree):
                if isinstance(n, ast.Attribute) and n.attr in HARNESS and (
                        (isinstance(n.value, ast.Name) and n.value.id in difftest) or ast.unparse(n.value).endswith(DIFFTEST)):
                    out.append(f"{rel}:{n.lineno}: {ast.unparse(n)} (the admission adapter through {DIFFTEST}; it is in "
                               f"{HARNESS_MODULE})")
    return out


ALLOWLISTS = VLLM / "tests" / "lint" / "allowlists"


def move_allowlist_entries() -> list[str]:
    """Lint allowlist entries are keyed by file: an entry of a moved harness name (or of the module docstring, which moved
    with it) follows it to the harness module."""
    old, new = (_path(m).relative_to(VLLM).as_posix() for m in (DIFFTEST, HARNESS_MODULE))
    entry = re.compile(r'"file": "' + re.escape(old) + r'", "kind": "(?P<kind>[^"]*)", "symbol": "(?P<symbol>[^"]*)"')
    changed = []
    for p in sorted(ALLOWLISTS.glob("*.json")):
        src = p.read_text(encoding="utf-8")
        out = []
        for line in src.splitlines(keepends=True):
            m = entry.search(line)
            if m and (m["symbol"] in HARNESS or (m["symbol"] == "<module>" and m["kind"].startswith("doc"))):
                line = line.replace(f'"file": "{old}"', f'"file": "{new}"', 1)
            out.append(line)
        if "".join(out) != src:
            p.write_text("".join(out), encoding="utf-8")
            changed.append(p.relative_to(REPO).as_posix())
    return changed


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
    dirty = _git("status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        raise SplitError("the work tree has uncommitted changes; a split that fails part way is undone with `git reset --hard`, "
                         "so start from a clean one:\n" + dirty)
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
    if st["harness"] == "before":
        move_harness()
        print(f"moved the admission adapter {HARNESS} -> {HARNESS_MODULE}")
        for c in move_allowlist_entries():
            print(f"  moved its allowlist entries in {c}")
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
        return 0 if all(v == "after" for k, v in st.items() if k in TWINS or k in ("vocab", "harness")) and not leftovers() else 1
    split(verify=not a.no_verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
