"""relayout.py -- move the vLLM integration into the ontology-shaped `verity_vllm/` skeleton, from `tools/move_map.txt`.

    python tools/relayout.py check   [--root DIR]        every tracked file resolves through the map (coverage / drift report)
    python tools/relayout.py plan    [--root DIR]        dry run: the moves, drops, rewrites and the residual sweep, nothing touched
    python tools/relayout.py apply   [--root DIR]        git mv + rewrites; refuses (exit 2) when old paths remain referenced
    python tools/relayout.py readme  [--root DIR]        rewrite the "## Layout" section of README.md against the target layout

Run from anywhere; `--root` is the integration root (default: this file's parent's parent).  Idempotent: on a moved tree there is
nothing to move, nothing to rewrite, and the sweep is empty.

What it rewrites (text-level, longest-match-first, delimited so `a.b` never matches inside `a.b_c` or `a.bc`):

* every old dotted module name (`verity_capture.fold`, `verity_vllm.frontend.rules`, `tests.ir.fixtures_query`, ...) -> its new
  dotted name, in every text file of the repository: `import` / `from ... import`, `importlib.import_module("...")`,
  `pytest.importorskip("...")`, the dotted-path registry in `tools/research/src/research/store/tools_registry.py`, `python -m`
  invocations in shell scripts, module strings in JSON / TOML;
* every old file path and directory prefix (`verity_capture/sweep/row_pod.sh`, `verity_capture/profiles/`, ...) -> its new path:
  shell scripts, `pyproject.toml`, the by-name allowlist `file` fields, `fixtures.toml`, `.gitignore`, docstrings;
* `from <pkg> import <submodule>` where the submodule's new parent or basename differs (ast-located; `as` alias keeps the local name),
  and relative imports whose target no longer sits at the same relative position;
* `Path(__file__).resolve().parents[N]` / `os.path.join(os.path.dirname(__file__), "..", ...)` / nested `dirname(...)` anchors in
  moved modules whose directory depth changed (N shifts by the depth delta; sibling references are untouched);
* explicit `REWRITE <old file> :: <old literal> => <new literal>` lines of the map (the few split-segment paths no rule can see).

Recorded artefacts (`tests/regression/expected/`, `manifests/semantic-profiles/`, `data/`, `fixtures/`, `workloads/`, `docs/`) are
never rewritten and are excluded from the residual sweep: they carry the module names of record.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent
MAP = HERE / "move_map.txt"

PY_ROOTS = ("verity_vllm", "verity_capture", "verity_vllm_adapter", "verity_vllm_numerics", "tests")
RECORDED = ("tests/regression/expected/", "manifests/semantic-profiles/", "data/", "fixtures/", "workloads/", "docs/",
            "tests/query/gate/", "tests/program/data/", "tests/ir/data/", "tests/harness/fixtures/", "tests/tp/fixtures/",
            "verity_capture/commit/tests/fixtures/", "verity_capture/tests/fixtures/", "verity_vllm/check/golden/",
            "verity_capture/golden/", "verity_vllm/observe/profiles/hf_configs/", "verity_capture/profiles/hf_configs/")
TEXT_EXT = {".py", ".sh", ".toml", ".json", ".jsonl", ".md", ".txt", ".cfg", ".ini", ".yaml", ".yml", ".cu", ".cpp", ".h", ".rs",
            ".diff", ".patch", ".gitignore"}
SUBPACKAGE_DOC = {
    "verity_vllm/program": "Program layer: Definitions (`registry`), lowering (`frontend`), the lifted / padded and workload Programs; `program_digest`.",
    "verity_vllm/program/numerics": "Independent numerical models of the served kernels: the exact evaluators the Definitions wrap.",
    "verity_vllm/query": "Verification query: `Q_module_body_v1`, boundary / required values, the required-value manifest and its v1 encoding.",
    "verity_vllm/correspondence": "Runtime correspondence: Call <-> (module path, fx node, slot, view, rank) as data, its emitter, readers and occurrence resolution.",
    "verity_vllm/observe": "Value observation: the dispatcher observer, raw observation log, fold and kernel patterns (the Match-side capture).",
    "verity_vllm/observe/profiles": "Pinned capture profiles (kappa): pattern registrations and accepted geometry per (vLLM, torch, Triton, GPU).",
    "verity_vllm/acquire": "Value acquisition: the acquisition plan, hooks / taps / native collector, compiled and hidden sources.",
    "verity_vllm/input_provenance": "Input provenance: weights of record, prescribed constants and their basis (checkpoint / specified computation).",
    "verity_vllm/commit": "Commitment: position-identified Merkle roots, binding map, openings, padding steps.",
    "verity_vllm/check": "Checks -> verdicts: runtime match, local replay, boundary linkage, checkpoint binding, execution extent, coverage, identity.",
    "verity_vllm/tp": "Tensor parallelism: rank as a coordinate -- per-rank capture, collective linkage, cross-rank replay, partial sources.",
    "verity_vllm/harness": "Development harness: the Build / Match / Commit stage drivers, finalizer, admission, telemetry and `research` Tools.",
    "verity_vllm/harness/telemetry": "Runner telemetry consumers (empirical admission planner).",
    "verity_vllm/ops": "Pod operations: bootstrap, row_pod.sh and the other shell entrypoints of the sweep.",
}

# --------------------------------------------------------------------------------------------------------------------------- map

@dataclass
class MoveMap:
    exact: dict[str, tuple[str, str | None]] = field(default_factory=dict)    # old -> (new | 'DROP', lane)
    prefix: list[tuple[str, str, str | None]] = field(default_factory=list)   # (old_dir/, new_dir/, lane)
    rewrites: list[tuple[str, str, str]] = field(default_factory=list)        # (old file, old literal, new literal)

    def resolve(self, old: str) -> tuple[str, str | None, str] | None:
        """-> (new, lane, how) with how in {exact, prefix}; None when unmapped."""
        if old in self.exact:
            new, lane = self.exact[old]
            return new, lane, "exact"
        best = None
        for pre, new, lane in self.prefix:
            if old.startswith(pre) and (best is None or len(pre) > len(best[0])):
                best = (pre, new, lane)
        if best:
            return best[1] + old[len(best[0]):], best[2], "prefix"
        return None


def load_map(path: Path = MAP) -> MoveMap:
    mm = MoveMap()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("REWRITE") else raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("REWRITE "):
            body = line[len("REWRITE "):]
            f, rest = body.split(" :: ", 1)
            old, new = rest.split(" => ", 1)
            mm.rewrites.append((f.strip(), old, new))
            continue
        lane = None
        m = re.search(r"\s+(DELETE|ADDED)-BY:(\S+)\s*$", line)
        if m:
            lane = m.group(2) if m.group(1) == "DELETE" else "+" + m.group(2)     # '+lane': the lane ADDS this file
            line = line[: m.start()]
        old, arrow, new = line.partition(" -> ")
        if not arrow:
            raise SystemExit(f"move_map: bad line: {raw!r}")
        old, new = old.strip(), new.strip()
        if old.endswith("/"):
            mm.prefix.append((old, new, lane))
        else:
            mm.exact[old] = (new, lane)
    return mm


def git_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\n") if f]


def repo_root(root: Path) -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True, check=True).stdout
    return Path(out.strip())


@dataclass
class Plan:
    moves: list[tuple[str, str, str | None]]        # (old, new, lane) -- old exists
    drops: list[str]                                 # docstring-only __init__.py to git rm
    missing: list[tuple[str, str | None]]            # mapped old paths absent from the tree and not at their new path (a lane deleted them)
    moved: list[str]                                 # mapped old paths absent, present at their new path (already moved)
    unmapped: list[str]
    by_prefix: list[str]
    collisions: dict[str, list[str]]
    drop_refused: list[str]


def build_plan(root: Path, mm: MoveMap) -> Plan:
    files = git_files(root)
    present = set(files)
    moves, drops, unmapped, by_prefix, refused = [], [], [], [], []
    targets: dict[str, list[str]] = defaultdict(list)
    seen_old = set()
    for f in files:
        r = mm.resolve(f)
        if r is None:
            unmapped.append(f)
            continue
        new, lane, how = r
        seen_old.add(f)
        if how == "prefix":
            by_prefix.append(f)
        if new == "DROP":
            if _docstring_only(root / f):
                drops.append(f)
            else:
                refused.append(f)
            continue
        if new == f:
            continue
        targets[new].append(f)
        moves.append((f, new, lane))
    missing, moved = [], []
    for o, (n, lane) in mm.exact.items():
        if o in present or n == o or n == "DROP":        # an absent DROP target was dropped already
            continue
        if n in present:
            moved.append(o)
        else:
            missing.append((o, lane))
    collisions = {k: v for k, v in targets.items() if len(v) > 1 or (k in present and k not in seen_old)}
    return Plan(moves, drops, missing, moved, unmapped, by_prefix, collisions, refused)


def _docstring_only(p: Path) -> bool:
    try:
        tree = ast.parse(p.read_text())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return False
    return all(isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant) and isinstance(n.value.value, str)
               for n in tree.body)


# ------------------------------------------------------------------------------------------------------------------- rewriting

def _module_name(path: str) -> str | None:
    if not path.endswith(".py") or not path.startswith(PY_ROOTS):
        return None
    m = path[:-3].replace("/", ".")
    return m[:-9] if m.endswith(".__init__") else m


def _depth(path: str) -> int:
    return path.count("/")


@dataclass
class Rewriter:
    mod_map: dict[str, str]             # old dotted -> new dotted (only where they differ)
    path_map: dict[str, str]            # old path / old dir prefix -> new (only where they differ)
    dotted_re: re.Pattern | None
    path_re: re.Pattern | None
    depth_delta: dict[str, int]         # new path -> new_depth - old_depth (moved .py files)
    new_of: dict[str, str]              # old path -> new path
    explicit: dict[str, list[tuple[str, str]]]   # new path -> [(old literal, new literal)]
    dissolved: set[str]                 # dotted names of packages with no single successor
    new_names: set[str]                 # every dotted name (and package prefix) that exists after the move
    new_paths: set[str]                 # every path (and directory prefix, with '/') that exists after the move

    @classmethod
    def build(cls, mm: MoveMap, plan: Plan, all_old: dict[str, str]) -> "Rewriter":
        """all_old: every old path -> new path (moves + unchanged), used for the module map; DROPs excluded."""
        mod_map, path_map = {}, {}
        # a package whose members do not all land under one new directory DISSOLVES: its own dotted name has no
        # successor (bare mentions are left for the sweep to refuse), only its members are renamed
        dissolved = set()
        for old, new in all_old.items():
            if old == new or not old.endswith("/__init__.py"):
                continue
            if new == "DROP":
                dissolved.add(_module_name(old))     # the package's own name has no successor at all
                continue
            od, nd = old[: -len("__init__.py")], new[: -len("__init__.py")]
            # only module members count: tests leaving for tests/ and scripts leaving for ops/ do not dissolve a package
            members = [(o, n) for o, n in all_old.items()
                       if o.startswith(od) and o != old and n != "DROP" and o.endswith(".py")
                       and not n.startswith("tests/") and "/tests/" not in o and not os.path.basename(o).startswith("test_")]
            # fewer than half the module members follow the __init__ -> the name has no single successor
            if members and sum(n.startswith(nd) for _, n in members) * 2 < len(members):
                dissolved.add(_module_name(old))
        for old, new in all_old.items():
            if new == "DROP":
                continue
            if old != new:
                path_map[old] = new
            om, nm = _module_name(old), _module_name(new)
            if om and nm and om != nm and om not in dissolved:
                mod_map[om] = nm
        for pre, new, _ in mm.prefix:
            if pre != new and pre.count("/") >= 1:
                path_map.setdefault(pre, new)
        # names that exist AFTER the move: a token that is (a prefix of) one of these is never a residual, even if it also
        # happens to be an old name (verity_vllm.commit is the old commit.py module AND the new commit/ package)
        new_names = set()
        for new in all_old.values():
            nm = _module_name(new) if new != "DROP" else None
            while nm:
                new_names.add(nm)
                nm = nm.rpartition(".")[0]
        new_paths = set()
        for new in all_old.values():
            if new != "DROP":
                new_paths.add(new)
                d = os.path.dirname(new)
                while d:
                    new_paths.add(d + "/")
                    d = os.path.dirname(d)
        # One cheap scan per file for tokens rooted at a known top-level name, then longest-prefix dict lookup.
        # (A single 800-way regex alternation over ~15 MB of sources takes minutes; this takes seconds.)
        roots = sorted({k.split(".")[0] for k in mod_map} | {k.split("/")[0] for k in path_map} | dissolved, key=len, reverse=True)
        root_alt = "|".join(re.escape(r) for r in roots)
        dotted_re = re.compile(r"(?<![\w./])((?:" + root_alt + r")(?:\.[A-Za-z_]\w*)*)(?![\w/])") if mod_map else None
        # a path may follow '/' ($REPO/verity_capture/..., integrations/vllm/verity_capture/...): the root name is the anchor
        path_re = re.compile(r"(?<![\w.])((?:" + root_alt + r")[\w.\-/]*)") if path_map else None
        depth = {}
        new_of = {}
        for old, new, _ in plan.moves:
            new_of[old] = new
            if new.endswith(".py"):
                depth[new] = _depth(new) - _depth(old)
        explicit: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for f, o, n in mm.rewrites:
            explicit[all_old.get(f, f)].append((o, n))
        return cls(mod_map, path_map, dotted_re, path_re, depth, new_of, explicit, dissolved, new_names, new_paths)

    # -- text passes ---------------------------------------------------------------------------------------------------------
    _NAME = r"[A-Za-z_]\w*(?:[^\S\n]+as[^\S\n]+\w+)?"
    _FROM_LINE_RE = re.compile(r"(?<![\w.])from (?P<mod>[A-Za-z_][\w.]*) import (?P<names>" + _NAME + r"(?:[^\S\n]*,[^\S\n]*" + _NAME + r")*)(?![\w.])")

    def _fix_from_lines(self, text: str) -> str:
        """`from pkg import submodule` written on one line anywhere in text (embedded `python -c` programs, here-docs, docs)
        whose submodule moved to a different parent.  The ast pass only sees the real statements of a .py file itself."""
        def fix(m: re.Match) -> str:
            try:
                node = ast.parse(f"from {m.group('mod')} import {m.group('names')}").body[0]
            except SyntaxError:
                return m.group(0)
            groups = self._regroup(m.group("mod"), node.names)
            if groups is None:
                return m.group(0)
            return "; ".join(self._emit(mod, names, 0).rstrip("\n") for mod, names in groups)
        return self._FROM_LINE_RE.sub(fix, text)

    # Names the structural passes (ast regroup, split paths, one-line from-imports) emit are already NEW names; the text passes
    # must not see them (`verity_vllm.commit` is both the old commit.py module and the new commit/ package).  They are frozen as
    # \x00N\x00 placeholders until the text passes are done.
    def _freeze(self, name: str) -> str:
        self._frozen.append(name)
        return f"\x00{len(self._frozen) - 1}\x00"

    def _thaw(self, text: str) -> str:
        return re.sub(r"\x00(\d+)\x00", lambda m: self._frozen[int(m.group(1))], text)

    def rewrite_text(self, text: str, new_path: str, in_integration: bool = True) -> str:
        # explicit literals first (they may pre-empt the generic passes), then paths (contain '/'), then dotted names
        for o, n in self.explicit.get(new_path, ()):
            if n == "<DEL>":
                text = "".join(l for l in text.splitlines(keepends=True) if o not in l)
            else:
                text = text.replace(o, n)
        self._in_integration = in_integration
        if not hasattr(self, "_frozen"):
            self._frozen = []
        text = self._fix_from_lines(text)
        if self.path_re:
            text = self.path_re.sub(self._sub_path, text)
        if self.dotted_re:
            text = self.dotted_re.sub(self._sub_dotted, text)
        text = self._thaw(text)
        self._frozen = []
        return text

    def is_residual_dotted(self, tok: str) -> bool:
        """A token is residual when its longest recognised prefix is an OLD name and strictly longer than its longest prefix in
        the post-move namespace: `verity_vllm.observe.contract.foo` is new although `verity_vllm.observe` was also the old
        observe.py module; a bare `verity_vllm.observe` in post-rewrite text is the new package (a tie is new)."""
        parts = tok.split(".")
        best_new = best_old = 0
        for i in range(len(parts), 0, -1):
            head = ".".join(parts[:i])
            if not best_new and head in self.new_names:
                best_new = i
            if not best_old and (head in self.mod_map or head in self.dissolved):
                best_old = i
        if not best_old:
            return False
        return best_old > best_new

    def is_residual_path(self, tok: str) -> bool:
        if "/" not in tok:
            return False        # bare names are the dotted pass's (or the sweep token list's) business
        t = tok.rstrip("/")
        if tok in self.new_paths or t in self.new_paths or (t + "/") in self.new_paths:
            return False
        return self._sub_path_tok(tok) != tok

    # -- split-segment paths: REPO / "verity_capture" / "profiles", os.path.join(ROOT, "verity_capture", "sweep", "row_pod.sh") --
    _SEG = r"""(?:"[\w.\-]+"|'[\w.\-]+')"""
    _CHAIN_RE = re.compile(r"(?<![\w\"'])(" + _SEG + r"(?:\s*/\s*" + _SEG + r")+)")
    _JOIN_RE = re.compile(r"\bjoin\(([^()]*)\)")
    _RUN_RE = re.compile(r"(" + _SEG + r"(?:\s*,\s*" + _SEG + r")+)")

    def _fix_split_paths(self, src: str) -> str:
        if not self.path_re:
            return src

        def resegment(run: str, sep: str) -> str:
            lits = re.findall(self._SEG, run)
            q = lits[0][0]
            joined = "/".join(l[1:-1] for l in lits)
            new = self._sub_path_tok(joined)
            if new == joined:
                return run
            return sep.join(q + self._freeze(s) + q for s in new.rstrip("/").split("/"))

        src = self._CHAIN_RE.sub(lambda m: resegment(m.group(1), " / "), src)

        def fix_join(m: re.Match) -> str:
            args = self._RUN_RE.sub(lambda r: resegment(r.group(1), ", "), m.group(1))
            return "join(" + args + ")"

        return self._JOIN_RE.sub(fix_join, src)

    def _sub_dotted(self, m: re.Match) -> str:
        return self._sub_dotted_tok(m.group(1))

    def _sub_path(self, m: re.Match) -> str:
        tok = m.group(1)
        if "/" not in tok:
            # a bare top-level FILE of the integration (pod_bootstrap.sh) is a path only inside the integration's own files and
            # never right after '/' (backends/sp1/pod_bootstrap.sh is somebody else's); a bare package name is not a path at all
            if tok in self.path_map and "." in tok and self._in_integration and not (m.start() and m.string[m.start() - 1] == "/"):
                return self.path_map[tok]
            return tok
        return self._sub_path_tok(tok)

    # the dissolved top-level packages, named bare (prose, `import verity_capture` probes): the package that absorbed them
    BARE_FALLBACK = {"verity_capture": "verity_vllm", "verity_vllm_adapter": "verity_vllm"}

    def _sub_dotted_tok(self, tok: str) -> str:
        """Longest dotted prefix of the token that is a renamed module wins; the remainder (submodule / attribute) is kept."""
        if tok in self.BARE_FALLBACK and tok in self.dissolved:
            return self.BARE_FALLBACK[tok]
        parts = tok.split(".")
        for i in range(len(parts), 0, -1):
            head = ".".join(parts[:i])
            new = self.mod_map.get(head)
            if new is not None:
                return new + ("." + ".".join(parts[i:]) if i < len(parts) else "")
        return tok

    IV_PREFIX = "integrations/vllm/"

    def _sub_path_tok(self, tok: str) -> str:
        """Exact file path first, then the longest moved directory prefix (keys ending in '/'); repo-root-relative spellings keep
        their `integrations/vllm/` prefix."""
        if tok.startswith(self.IV_PREFIX):
            return self.IV_PREFIX + self._sub_path_tok(tok[len(self.IV_PREFIX):])
        new = self.path_map.get(tok)
        if new is not None:
            return new
        if not tok.endswith("/") and (tok + "/") in self.path_map:      # a directory named without its slash
            return self.path_map[tok + "/"].rstrip("/")
        segs = tok.split("/")
        for i in range(len(segs) - 1, 0, -1):
            head = "/".join(segs[:i]) + "/"
            new = self.path_map.get(head)
            if new is not None:
                return new + "/".join(segs[i:])
        return tok

    def rewrite_python(self, src: str, old_path: str, new_path: str) -> str:
        """ast-located import fixes that plain text substitution cannot express, then the text passes, then the depth fix."""
        self._frozen = []
        src = self._fix_from_imports(src, old_path, new_path)
        src = self._fix_split_paths(src)
        src = self.rewrite_text(src, new_path)
        d = self.depth_delta.get(new_path, 0)
        if d:
            src = self._fix_file_anchors(src, d)
        return src

    def _fix_from_imports(self, src: str, old_path: str, new_path: str) -> str:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return src
        old_mod = _module_name(old_path)
        new_mod = _module_name(new_path)
        if old_mod is None:
            return src
        old_pkg = old_mod if old_path.endswith("__init__.py") else old_mod.rpartition(".")[0]
        new_pkg = new_mod if new_path.endswith("__init__.py") else new_mod.rpartition(".")[0]
        lines = src.splitlines(keepends=True)
        edits: list[tuple[int, int, int, str]] = []   # (start line idx, end line idx exclusive, end col, replacement)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                base = old_pkg
                for _ in range(node.level - 1):
                    base = base.rpartition(".")[0]
                abs_mod = base + ("." + node.module if node.module else "")
                # would the same relative import resolve to the right place from the new location?
                nbase = new_pkg
                for _ in range(node.level - 1):
                    nbase = nbase.rpartition(".")[0]
                would = nbase + ("." + node.module if node.module else "")
            else:
                abs_mod = node.module or ""
                would = None
            groups = self._regroup(abs_mod, node.names)
            if groups is None:
                if node.level and would is not None:
                    target = self.mod_map.get(abs_mod, abs_mod)
                    if would != target:
                        edits.append((node.lineno - 1, node.end_lineno, node.end_col_offset, self._emit(target, node.names, node.col_offset)))
                continue
            edits.append((node.lineno - 1, node.end_lineno, node.end_col_offset, "".join(self._emit(mod, names, node.col_offset) for mod, names in groups)))
        for s, e, col, rep in sorted(edits, reverse=True):
            # keep a trailing comment (`# noqa: F401 -- imported here == must be listed`) on the statement's last line
            tail = lines[e - 1][col:].rstrip("\n")
            if tail.lstrip().startswith("#"):
                rep = rep.rstrip("\n") + "  " + tail.strip() + "\n"
            lines[s:e] = [rep]
        return "".join(lines)

    def _regroup(self, abs_mod: str, names: list[ast.alias]) -> list[tuple[str, list[ast.alias]]] | None:
        """`from abs_mod import a, b`: when `abs_mod.a` is a module that moved to a different parent or basename, regroup."""
        groups: dict[str, list[ast.alias]] = defaultdict(list)
        changed = False
        for a in names:
            sub = f"{abs_mod}.{a.name}"
            tgt = self.mod_map.get(sub)
            if tgt is not None:
                parent, _, base = tgt.rpartition(".")
                base_mod_new = self.mod_map.get(abs_mod, abs_mod)
                if parent != base_mod_new or base != a.name:
                    changed = True
                    groups[parent].append(ast.alias(name=base, asname=a.asname or (a.name if base != a.name else None)))
                    continue
            groups[self.mod_map.get(abs_mod, abs_mod)].append(a)
        if not changed:
            return None
        return list(groups.items())

    def _emit(self, mod: str, names: list[ast.alias], indent: int) -> str:
        parts = [n.name + (f" as {n.asname}" if n.asname else "") for n in names]
        return " " * indent + f"from {self._freeze(mod)} import {', '.join(parts)}\n"

    @staticmethod
    def _fix_file_anchors(src: str, d: int) -> str:
        out = []
        for line in src.splitlines(keepends=True):
            if "__file__" in line:
                def bump(m: re.Match) -> str:
                    n = int(m.group(1))
                    return f"parents[{max(n + d, 0)}]" if n >= 1 else m.group(0)
                line = re.sub(r"parents\[(\d+)\]", bump, line)
                m = re.search(r'(dirname\(__file__\)|abspath\(__file__\)\))((?:,\s*"\.\.")+)', line)
                if m:
                    ups = m.group(2).count('".."')
                    line = line[: m.start(2)] + ', ".."' * max(ups + d, 0) + line[m.end(2):]
                m = re.search(r"((?:os\.path\.dirname\()+)os\.path\.(?:abspath|realpath)\(__file__\)", line)
                if m and m.group(1).count("dirname(") >= 2:
                    k = m.group(1).count("dirname(")
                    new_k = max(k + d, 1)
                    line = line[: m.start(1)] + "os.path.dirname(" * new_k + line[m.end(1):]
                    line = _rebalance_parens(line, m.start(1), new_k - k)
            out.append(line)
        return Rewriter._fix_anchor_climbs("".join(out), d)

    # `HERE = dirname(abspath(__file__))` / `HERE = Path(__file__).resolve().parent` on one line, then `REPO = dirname(dirname(HERE))`
    # / `HERE.parents[1]` / `HERE.parent.parent` on another: the climb from the file's own directory grows by the depth delta.
    _ANCHOR_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?::\s*[\w\[\], .|]+)?=\s*(?:os\.path\.dirname\(os\.path\.(?:abspath|realpath)\(__file__\)\)"
                            r"|os\.path\.dirname\(__file__\)|(?:pathlib\.)?Path\(__file__\)(?:\.resolve\(\))?\.parent)\s*(?:#.*)?$")

    @staticmethod
    def _fix_anchor_climbs(src: str, d: int) -> str:
        lines = src.splitlines(keepends=True)
        anchors = {m.group(1) for line in lines if (m := Rewriter._ANCHOR_RE.match(line))}
        if not anchors:
            return src
        alt = "|".join(re.escape(a) for a in sorted(anchors, key=len, reverse=True))
        dn_re = re.compile(r"((?:os\.path\.dirname\()+)(" + alt + r")(\)+)")
        parents_re = re.compile(r"\b(" + alt + r")\.parents\[(\d+)\]")
        parent_re = re.compile(r"\b(" + alt + r")((?:\.parent)+)\b")
        out = []
        for line in lines:
            if "__file__" not in line:
                def dn(m: re.Match) -> str:
                    k = m.group(1).count("dirname(")
                    closing = m.group(3)
                    if len(closing) < k:
                        return m.group(0)
                    new_k = max(k + d, 0)
                    return "os.path.dirname(" * new_k + m.group(2) + ")" * (len(closing) - k + new_k)
                line = dn_re.sub(dn, line)
                line = parents_re.sub(lambda m: f"{m.group(1)}.parents[{max(int(m.group(2)) + d, 0)}]", line)
                line = parent_re.sub(lambda m: m.group(1) + ".parent" * max(m.group(2).count(".parent") + d, 0), line)
            out.append(line)
        return "".join(out)


def _rebalance_parens(line: str, start: int, delta: int) -> str:
    """after changing the number of `os.path.dirname(` wrappers by delta, add/remove the matching `)` at the end of the expression."""
    if delta == 0:
        return line
    # the wrappers' closing parens are the run of ')' right after `__file__)`
    m = re.search(r"__file__\)+", line[start:])
    if not m:
        return line
    run_end = start + m.end()
    if delta > 0:
        return line[:run_end] + ")" * delta + line[run_end:]
    return line[: run_end + delta] + line[run_end:]


# ---------------------------------------------------------------------------------------------------------------------- sweep

def is_text(p: Path) -> bool:
    if p.suffix.lower() in TEXT_EXT or p.name in (".gitignore",):
        try:
            return b"\0" not in p.read_bytes()[:4096]
        except OSError:
            return False
    return False


def residual_lines(rw: Rewriter, text: str, extra_re: re.Pattern | None) -> list[tuple[int, str]]:
    """Lines still holding an old dotted name / old path (i.e. the rewrite would change them) or a dissolved package token."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        hit = extra_re is not None and extra_re.search(line) is not None
        if not hit and rw.path_re:
            hit = any(rw.is_residual_path(m.group(1)) for m in rw.path_re.finditer(line))
        if not hit and rw.dotted_re:
            hit = any(rw.is_residual_dotted(m.group(1)) for m in rw.dotted_re.finditer(line))
        if hit:
            out.append((i, line.strip()[:160]))
    return out


def sweep(repo: Path, rel_iv: str, rw: Rewriter, extra_tokens: list[str], skip: set[str]) -> list[tuple[str, int, str]]:
    """Every remaining reference to an old dotted name / old path / dissolved package token, repo-wide, outside recorded dirs."""
    extra_re = re.compile(r"(?<![\w.])(?:" + "|".join(re.escape(t) for t in extra_tokens) + r")(?![\w])") if extra_tokens else None
    hits = []
    for f in git_files(repo):
        p = repo / f
        rel = f[len(rel_iv) + 1:] if f.startswith(rel_iv + "/") else None
        if rel is not None and rel.startswith(RECORDED):
            continue
        if f in skip or not p.is_file() or not is_text(p):
            continue
        try:
            for i, line in residual_lines(rw, p.read_text(errors="replace"), extra_re):
                hits.append((f, i, line))
        except OSError:
            pass
    return hits


# ---------------------------------------------------------------------------------------------------------------------- README

def readme_layout_section() -> str:
    return (
        "## Layout\n\n"
        "One distribution (`verity-vllm`), one package (`verity_vllm`), one subpackage per layer of the vocabulary below; `tests/` mirrors\n"
        "the package.  Naming (Draft 3.1): core Verity nouns get the shortest names (`program`, `query`, `commit`), integration concepts\n"
        "carry their qualifier (`input_provenance`, `runtime` correspondence), implementation artefacts have descriptive compound names\n"
        "(`native_collect`, `weights_of_record`, `poc_*` for v1 leftovers).  A subpackage exists because there is code for it, not because\n"
        "the layer table has a row; Build / Match / Commit are commands of the harness, not layers.  Every file arrived by\n"
        "`python tools/relayout.py apply` from `tools/move_map.txt` (the old -> new map of record; `check` reports coverage and drift).\n\n"
        "~~~text\n"
        "verity_vllm/\n"
        "  program/            Program            registry/ frontend/ lifting/ numerics/ workload.py global_program.py compact.py dtypes.py\n"
        "  query/              verification query module_body.py boundary.py partition.py required_values.py required_manifest.py cli.py\n"
        "  correspondence/     runtime corresp.   runtime.py emit.py reader_for_*.py resolve*.py batch_*.py chunk_attribution.py\n"
        "  observe/            value observation  observer.py events.py log.py fold.py patterns*.py resolver.py contract.py profiles/\n"
        "  acquire/            value acquisition  plan.py install.py stage.py native_*.{py,cu,cpp} native_host.py *_source.py *_tap_src/\n"
        "  input_provenance/   input provenance   weights_of_record.py root_policy.py analytic.py\n"
        "  commit/             commitment         merkle.py hashing.py identity.py binding.py padding_steps.py hidden_*.py engine_rs/\n"
        "  check/              checks -> verdicts global_match.py sampled_replay.py executed_prefix.py verdict.py gates.py census.py ...\n"
        "  tp/                 tensor parallelism worker.py capture.py match.py collective_*.py rank_match.py partial_source.py\n"
        "  harness/            dev harness        derive_step.py commit_delta.py finalize.py hot_commit.py admission_*.py research_tools.py\n"
        "  ops/                pod operations     pod_bootstrap.sh row_pod.sh run_row_v2.sh pod_*_tap.sh canary.sh known_roots.json\n"
        "tests/<subpackage>/   mirrors the package; tests/regression = the R19 reference harness (opt in with VERITY_REGRESSION=1)\n"
        "workloads/ manifests/ fixtures/ data/ docs/    records of the qualification table; never rewritten by the tool\n"
        "~~~\n\n"
        "**`program/`** implements *Program*: Definitions (`registry/`, the registered library; importing it registers the on-demand\n"
        "primitive families through `verity.ir.codec.register_lazy_family`), lowering (`frontend/`: the torch frontend, `derive`, `rules/`\n"
        "= lowering rules, the vLLM meta export, `target_profile`), lifted / padded Programs (`lifting/`), the rooted workload Program\n"
        "(`workload.py`, `global_program.py`, `compact.py`) and the independent kernel models the Definitions wrap (`numerics/`, C++ / CUDA\n"
        "models included).  Entrypoints: `program.frontend.derive`, `program.registry`.  Never: module paths as identity, hook sets,\n"
        "VU policy, model-name tables (`registry/quarantine/` is the by-name residue and only shrinks).\n\n"
        "**`query/`** implements the *verification query*: `module_body.py` (Q_module_body_v1), `program_view.py`, `boundary.py` /\n"
        "`partition.py` (boundary values, VU partitions), `required_values.py` + `required_manifest.py` (the required-value manifest and\n"
        "the serialized `Q_module_v1` id), `compiled_manifest.py`, `v1_bridge.py` / `vu_canonical.py` / `vu_query.py` (the v1 IDENTITY_KEY\n"
        "encoding, retired by lane vllm-v2-default), `cli.py`.  Entrypoint: `python -m verity_vllm.query.cli`.  Never: execution data,\n"
        "hooks, model names, anything read from a run.\n\n"
        "**`correspondence/`** implements *runtime correspondence* as data: `runtime.py` (runtime-correspondence/v1), `emit.py` (the\n"
        "frontend's emitter), `reader_for_query.py` / `reader_for_acquire.py`, `runtime_tree.py`; and *occurrence resolution*:\n"
        "`resolve.py`, `resolve_decomp.py`, `batch_decomp.py` / `batch_attribution.py`, `chunk_attribution.py`.  Never: verification\n"
        "policy, commitment positions, requiredness, verdicts.\n\n"
        "**`observe/`** implements *value observation* (the Match-side capture): `observer.py` (the TorchDispatchMode), `events.py` /\n"
        "`log.py` / `tree.py` / `memory.py` / `views.py` / `storage.py`, `resolver.py`, `fold.py` (raw log -> Program), `patterns*.py`,\n"
        "`contract.py` (the one observation contract), the engine we attach to (`vllm_adapter.py`, `engine_driver.py`,\n"
        "`engine_profile.py`, `triton_adapter.py`), `m1_capture.py`, and `profiles/` (kappa: capture profiles and workload JSON; lane\n"
        "vllm-profiles-generic leaves one generic constructor + a facts table).  Entrypoint: `python -m verity_vllm.observe.m1_capture`.\n"
        "Never: commit, decide what is required, name a model.\n\n"
        "**`acquire/`** implements *value acquisition*: `plan.py` (required values + correspondence -> acquisition plan), `install.py`,\n"
        "`stage.py`, the native collector (`native_collect.py` + `native_*.cu` / `.cpp`, `native_jit.py`, `leafhash.py`), `native_host.py`\n"
        "(the collector of record), `hidden_source.py` / `moe_source.py` / `compiled_source.py` / `compiled_kernel_source.py`,\n"
        "`fa2_plane_classes.py`, `committer_api.py`, and the shipped tap sources (`fa2_tap_src/`, `fa3_tap_src/`, `hidden_gpu_src/`).\n"
        "`gate.py` / `switch.py` / `match_compare.py` / `v1_decision.py` are the v1 decision path (lane vllm-v2-default).  Never: decide\n"
        "what is required; never a model-name list.\n\n"
        "**`input_provenance/`** implements *input provenance*: `weights_of_record.py` (checkpoint basis, `weights_pin`), `root_policy.py`,\n"
        "`analytic.py` (specified computations such as bf16(sqrt(H))).  Never: KV / block tables or schedule (execution evidence).\n\n"
        "**`commit/`** implements *commitment*: `merkle.py` (trees over canonical ids, `path_shape` / `fold_path`, openings), `hashing.py`\n"
        "(`H(tag, *parts)`), `identity.py`, `binding.py` (binding map + coverage), `padding_steps.py`, `fasttree.py` / `semantic_layout.py`\n"
        "/ `stream_merkle.py`, `hidden_stream.py` + `hidden_engine.py`, `engine_rs/` (Rust); `reference_engine/` and `fa2_prototype/` are\n"
        "the CMT-1 / FA2 prototypes (lane vllm-dead-code).  Never: runtime metadata in the identity tuple.\n\n"
        "**`check/`** implements *checks -> verdicts*: runtime match (`global_match.py`, `global_match_fast.py`, `gates.py` G1..G8), local\n"
        "replay (`sampled_replay.py`, `replay*.py`), execution extent (`executed_prefix.py`), `verdict.py` + `commit_verdict.py`, boundary\n"
        "/ compiled checks (`compiled_*_check.py`, `kernel_identity.py`, `kernel_allowlist.py`), `oracle_compare.py`, `program_compare.py` /\n"
        "`fold_compare.py`, `census.py`, `noninterference.py`, `operand_provenance.py`, `golden.py` + `golden/`, `difftest.py`,\n"
        "`relations.py`, `twins.py`, `protected.py`, `quarantine_lint.py`; `poc_*.py` are v1 leftovers (lane vllm-dead-code).  Never:\n"
        "acquire or commit anything.\n\n"
        "**`tp/`** is *tensor parallelism* with rank as a coordinate: `worker.py`, `capture.py`, `match.py` / `fold_match.py` /\n"
        "`rank_match.py`, `collective_*.py`, `xrank_collectives.py`, `partial_source.py`, `export_ops.py`, `commit_delta.py`, `analyze.py`\n"
        "(lane vllm-tp-n generalises the TP=2 spellings).  Never: a second copy of a layer above -- tp/ parameterises them by rank.\n\n"
        "**`harness/`** is the development harness (Build / Match / Commit are its commands): `derive_step.py`, `run_config.py`,\n"
        "`commit_delta.py`, `finalize.py`, `hot_commit.py`, `commit_summary.py`, `timeline.py`, `target_family.py`, `admission_bound.py`\n"
        "/ `admission_planner.py` + `telemetry/`, `research_outputs.py` / `research_result.py`, `source_identity.py`, `release_json.py`,\n"
        "`workload.py` / `coverage_workloads.py`, `experiment.py`, `research_tools.py` (the `vllm.build` / `vllm.match` / `vllm.commit`\n"
        "Tools of `tools/research`).  Never: Verity semantics -- a check that lives here is misfiled.\n\n"
        "**`ops/`** is pod operations: `pod_bootstrap.sh`, `row_pod.sh` / `row_pod_tp2.sh` / `run_row_v2.sh` (the row runner and its\n"
        "stages), `pod_env_extra.sh`, `pod_fa2_tap.sh` / `pod_fa3_tap.sh` / `pod_hidden_gpu.sh`, `canary.sh`, `cov_pod.sh`, the gates\n"
        "(`pod_gate.sh`, `pod_match_v2.sh`, `verify_lane.sh`), `known_roots.json`.  Never: Python that a layer above would import.\n\n"
        "The stdlib-only core of `verity_ir` (`types refs defs program layout evaluate query codec parts query_codec`, `gateset`\n"
        "as `query_ast`, `plan`) is `verity.ir` / `verity.verification.plan` in `packages/verity`; the on-demand primitive families\n"
        "veritor hardcoded in `codec.py` are registered by `verity_vllm/program/registry/__init__.py` through `verity.ir.codec.register_lazy_family`.\n\n"
    )


def readme_with_layout(text: str) -> str | None:
    """README text with its '## Layout' section replaced by the target layout; None when there is no such section."""
    m = re.search(r"^## Layout\n.*?(?=^## )", text, flags=re.S | re.M)
    if not m:
        return None
    return text[: m.start()] + readme_layout_section() + text[m.end():]


def rewrite_readme(root: Path, write: bool) -> bool:
    p = root / "README.md"
    text = p.read_text()
    new = readme_with_layout(text)
    if new is None:
        print("README: no '## Layout' section found", file=sys.stderr)
        return False
    if new == text:
        return False
    if write:
        p.write_text(new)
    return True


# ----------------------------------------------------------------------------------------------------------------------- main

def _all_old(root: Path, mm: MoveMap, plan: Plan) -> dict[str, str]:
    out = {}
    for f in git_files(root):
        r = mm.resolve(f)
        if r:
            out[f] = r[0]
    # Absent old paths still rewrite (a lane deleted the file; prose may still name it) -- unless the NEW path already exists,
    # which means the move already happened: rewriting again would double-apply (idempotency).
    for old, (new, _) in mm.exact.items():
        if old not in out and new != "DROP" and not (root / new).exists():
            out[old] = new
    return out


def cmd_check(root: Path, mm: MoveMap, verbose: bool) -> int:
    plan = build_plan(root, mm)
    n_files = len(git_files(root))
    lanes = defaultdict(int)
    for _, _, lane in plan.moves:
        if lane:
            lanes[lane] += 1
    print(f"tracked files: {n_files}")
    print(f"  moving: {len(plan.moves)}   unchanged: {n_files - len(plan.moves) - len(plan.drops) - len(plan.unmapped)}   "
          f"drops: {len(plan.drops)}   matched by prefix rule: {len(plan.by_prefix)}")
    print(f"  tagged DELETE-BY (still present): " + (", ".join(f"{k}={v}" for k, v in sorted(lanes.items()) if not k.startswith("+")) or "none"))
    if any(k.startswith("+") for k in lanes):
        print(f"  tagged ADDED-BY (present): " + ", ".join(f"{k[1:]}={v}" for k, v in sorted(lanes.items()) if k.startswith("+")))
    print(f"  mapped but absent: deleted by a lane (or ghost) = {len(plan.missing)}, already at the new path = {len(plan.moved)}")
    if verbose:
        for f, lane in plan.missing:
            print(f"    absent {f}" + (f"  (DELETE-BY:{lane})" if lane else "  (no lane tag!)"))
        for f in plan.by_prefix:
            print(f"    prefix {f} -> {mm.resolve(f)[0]}")
    rc = 0
    if plan.unmapped:
        rc = 1
        print(f"  UNMAPPED ({len(plan.unmapped)}):")
        for f in plan.unmapped:
            print(f"    {f}")
    if plan.collisions:
        rc = 1
        print(f"  COLLISIONS ({len(plan.collisions)}):")
        for k, v in plan.collisions.items():
            print(f"    {k} <- {v}")
    if plan.drop_refused:
        rc = 1
        print("  DROP refused (file holds code):")
        for f in plan.drop_refused:
            print(f"    {f}")
    untagged_absent = [f for f, lane in plan.missing if not lane]
    added = [f for f, lane in plan.missing if lane and lane.startswith("+")]
    if added:
        print(f"  not yet added by their lane: {len(added)}")
    if untagged_absent:
        print(f"  DRIFT: {len(untagged_absent)} mapped files are gone without a DELETE-BY tag (a lane removed them):")
        for f in untagged_absent:
            print(f"    {f}")
    return rc


def cmd_plan_or_apply(root: Path, mm: MoveMap, apply: bool, verbose: bool) -> int:
    repo = repo_root(root)
    rel_iv = os.path.relpath(root, repo)
    plan = build_plan(root, mm)
    if plan.unmapped or plan.collisions or plan.drop_refused:
        cmd_check(root, mm, verbose)
        print("refusing: the map does not cover the tree", file=sys.stderr)
        return 1
    all_old = _all_old(root, mm, plan)
    rw = Rewriter.build(mm, plan, all_old)
    self_files = {f"{rel_iv}/tools/relayout.py", f"{rel_iv}/tools/move_map.txt", f"{rel_iv}/tests/test_relayout_map.py"}

    # 1. moves + drops
    print(f"[relayout] {len(plan.moves)} moves, {len(plan.drops)} drops, {len(plan.missing)} mapped files absent, {len(plan.moved)} already moved")
    if verbose or not apply:
        for old, new, lane in plan.moves:
            print(f"  mv {old} -> {new}" + (f"   [DELETE-BY:{lane}]" if lane else ""))
        for f in plan.drops:
            print(f"  rm {f}   (docstring-only __init__.py)")
    if apply:
        for old, new, _ in plan.moves:
            (root / new).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "mv", "-k", old, new], cwd=root, check=True)
        for f in plan.drops:
            subprocess.run(["git", "rm", "-q", f], cwd=root, check=True)
        # git mv leaves the emptied directories behind (and __pycache__ under them)
        for d in sorted({os.path.dirname(o) for o, _, _ in plan.moves} | {os.path.dirname(f) for f in plan.drops}, key=len, reverse=True):
            while d:
                p = root / d
                if not p.is_dir():
                    d = os.path.dirname(d)
                    continue
                for pc in p.rglob("__pycache__"):
                    for f in pc.iterdir():
                        f.unlink()
                    pc.rmdir()
                if any(p.iterdir()):
                    break
                p.rmdir()
                d = os.path.dirname(d)

    # 2. package __init__.py for new directories
    created = []
    new_dirs = set()
    for _, new, _ in plan.moves:
        if new.endswith(".py"):
            d = os.path.dirname(new)
            while d and d.startswith(("verity_vllm", "tests")) and d not in ("tests",):
                new_dirs.add(d)
                d = os.path.dirname(d)
    existing_inits = {os.path.dirname(f) for f in all_old.values() if f.endswith("__init__.py") and f != "DROP"}
    old_inits = {os.path.dirname(f) for f in all_old if f.endswith("__init__.py")}
    # a directory that moved as a unit and was not a package before (C / CUDA source dirs, tap sources) stays a plain directory
    src_dirs_of = defaultdict(set)
    for old, new, _ in plan.moves:
        if new.endswith(".py"):
            src_dirs_of[os.path.dirname(new)].add(os.path.dirname(old))
    for d in sorted(new_dirs):
        if d in existing_inits:
            continue
        srcs = src_dirs_of.get(d, set())
        if len(srcs) == 1 and next(iter(srcs)) not in old_inits and not d.startswith("tests"):
            continue
        p = root / d / "__init__.py"
        if apply and p.exists():
            continue
        created.append(d + "/__init__.py")
        if apply:
            doc = SUBPACKAGE_DOC.get(d, f"{d.replace('/', '.')}: tests mirror the package" if d.startswith("tests") else d.replace("/", "."))
            p.write_text(f'"""{doc}"""\n')
            subprocess.run(["git", "add", str(p)], cwd=root, check=True)
    if created:
        print(f"[relayout] {len(created)} new __init__.py: " + ", ".join(created))

    # 3. rewrites -- every text file of the repository; in dry run the rewritten text is also what the sweep inspects
    old_of = {n: o for o, n in rw.new_of.items()}
    # top-level package names that stop existing after the move (they dissolve into verity_vllm/): a bare mention that the
    # passes could not resolve is a residual
    gone = sorted(d for d in rw.dissolved if "." not in d)
    extra_re = re.compile(r"(?<![\w.])(?:" + "|".join(re.escape(t) for t in gone) + r")(?![\w])")
    n_changed = 0
    changed_files = []
    residual: list[tuple[str, int, str]] = []
    for f in git_files(repo):
        if f in self_files:
            continue
        p = repo / f
        rel = f[len(rel_iv) + 1:] if f.startswith(rel_iv + "/") else None
        if rel is not None and (rel.startswith(RECORDED) or rel in plan.drops):
            continue
        if not p.is_file() or not is_text(p):
            continue
        text = p.read_text(errors="surrogateescape")
        if rel is not None and p.suffix == ".py":
            # after `apply` the file already sits at its new path; before, at its old one
            new_rel = rel if apply else rw.new_of.get(rel, rel)
            old_rel = old_of.get(rel, rel) if apply else rel
            new_text = rw.rewrite_python(text, old_rel, new_rel)
        else:
            new_text = rw.rewrite_text(text, rel if rel is not None else f, in_integration=rel is not None)
        if f == f"{rel_iv}/pyproject.toml":
            new_text = _fix_pyproject(new_text)
        if f == f"{rel_iv}/README.md":
            new_text = readme_with_layout(new_text) or new_text
        if new_text != text:
            n_changed += 1
            changed_files.append(f)
            if apply:
                p.write_text(new_text, errors="surrogateescape")
        if not apply:
            for i, line in residual_lines(rw, new_text, extra_re):
                residual.append((f, i, line))
    print(f"[relayout] rewrote {n_changed} files" + ("" if apply else " (dry run)"))
    if verbose:
        for f in changed_files:
            print(f"  ~ {f}")

    # 4. README
    if rewrite_readme(root, write=apply):
        print("[relayout] README.md: layout section " + ("rewritten" if apply else "would be rewritten"))

    # 5. residual sweep: on disk after apply; on the simulated text in dry run
    if apply:
        residual = sweep(repo, rel_iv, rw, gone, self_files)
    if residual:
        what = "REFUSING: residual references to old paths / modules remain" if apply else \
               "dry run: references the rewrite would NOT resolve (apply will refuse)"
        print(f"[relayout] {what}: {len(residual)}", file=sys.stderr)
        for f, i, line in residual:
            print(f"  {f}:{i}: {line}")
        return 2
    print("[relayout] residual sweep clean" if apply else "[relayout] dry run: rewrite resolves every reference")
    return 0


def _fix_pyproject(text: str) -> str:
    text = re.sub(r"packages = \[[^\]]*\]", 'packages = [\n    "verity_vllm",\n]', text, count=1)
    text = re.sub(r'testpaths = \[[^\]]*\]', 'testpaths = ["tests"]', text, count=1)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "plan", "apply", "readme"])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="integration root (integrations/vllm)")
    ap.add_argument("--map", type=Path, default=MAP)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    mm = load_map(a.map)
    if a.cmd == "check":
        return cmd_check(root, mm, a.verbose)
    if a.cmd == "readme":
        return 0 if rewrite_readme(root, write=True) else 1
    return cmd_plan_or_apply(root, mm, apply=(a.cmd == "apply"), verbose=a.verbose)


if __name__ == "__main__":
    sys.exit(main())
