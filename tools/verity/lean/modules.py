"""`modules`: the codemod that puts the in-tree Lake packages' Lean on Lean's module system, and the header reader it uses.

    python3 tools/verity/lean/modules.py [PATH...]                  port every module of each package under PATH (default: all)
    python3 tools/verity/lean/modules.py --check [PATH...]          name the files not yet ported; exit 1 when there is one
    python3 tools/verity/lean/modules.py --option NAME FILE...      add transition option NAME to each ported FILE

A package's modules are the `.lean` files of its `[[lean_lib]]`s and `[[lean_exe]]` roots; a script beside them (run with
`lean --run`) is not one and is left alone. A file whose header already starts with `module` is left as it is, so a rerun
changes nothing and a branch made before the port takes it by running this on its head. Porting a file:

- puts `module` first in its header, after the comments that open the file;
- makes each `import M` a `public import M` and each `meta import M` a `public meta import M` (`import all M` stays), so an
  importer sees what it saw before;
- opens `@[expose] public section` after the header (after the module docstring when one follows it), never closed, so every
  declaration is public and every definition's value is exposed, as without the module system; `private` still means
  private, and a theorem's proof is never exposed;
- in a file that declares a private definition, sets `backward.privateInPublic` (as `--option` does), since an exposed
  body may not otherwise read it;
- in a file that runs code while it compiles (`#eval`, `#guard`, `run_cmd`, ...), adds `meta import M` for each `M` it
  imports: a module runs that code at the `meta` phase, which reads only what it meta-imports.

Each package's `lakefile.toml` sets `backward.proofsInPublic` under `[leanOptions]` (`LAKE_OPTION`): a proof inside a public
statement or an exposed definition (a `by decide` argument, a derived `DecidableEq`'s) stays in it, as without the module
system, instead of becoming an auxiliary theorem, so every declaration elaborates to the term it did and its records in
`lean-audit.json` don't move. A proof's body is private either way. What still differs is a matcher: a derived `Repr`'s
function is never exposed, so a public definition matching as it does gets a matcher of its own instead of sharing its.

What the module system refuses beyond that is fixed by hand in the file, which this then leaves alone: a tactic's helper
gets `meta`; a proof that unfolds a core definition the toolchain doesn't expose rewrites by core's lemma for it or gets
`import all` of that definition's module (never of an in-tree module: a module's build reads only its imports' public
parts), and when that `import all` slows the file's other proofs past their budget, the proof moves to a module of its
own (`CallsFast100`); a theorem whose statement names another module's private declaration (core's `Array.qsort.sort`)
is private, beside a public theorem over the public definition for other files; an exported `:= rfl` theorem whose `rfl`
unfolds an unexposed definition is `:= (rfl)`, which isn't inferred `@[defeq]`; a tactic block in a statement gets its
type ascribed. `--option backward.privateInPublic` adds that transition option to a file whose public definition reads a
private declaration that isn't a definition, or whose private declaration another module opens with `open private … from`;
the private one keeps its name, and the warning is turned off.

It reads each header with `headers.py`, the one reading of a Lean header, and edits it where that says each import starts.
Stdlib only."""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SECTION = "@[expose] public section"
#: each transition option `--option` takes, and the `set_option` lines it adds
OPTIONS = {"backward.privateInPublic": ("backward.privateInPublic true", "backward.privateInPublic.warn false")}
#: the line each package's `[leanOptions]` holds
LAKE_OPTION = "backward.proofsInPublic = true"
#: a command that runs code while its file compiles
EVALUATES = re.compile(r"(?m)(?:^[ \t]*|\bin[ \t]+)(?:#eval!?|#guard|run_cmd|run_meta|run_elab)(?![\w!?'])")
#: a private definition, which an exposed body reads only under `backward.privateInPublic`
PRIVATE = re.compile(r"(?m)^[ \t]*(?:@\[[^\]\n]*\][ \t]*)*private[ \t]+(?:(?:noncomputable|partial|unsafe|nonrec)[ \t]+)*"
                     r"(?:def|abbrev|instance|structure|inductive|opaque|class)\b")
FIX = "python3 tools/verity/lean/modules.py"
FIXTURES = "tools/verity/lean/tests/fixtures/"


def _sibling(name: str):
    """This directory's `name.py`, loaded by path (the directory is copied into other repositories, not installed)."""
    spec = importlib.util.spec_from_file_location(f"lean_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _sibling("headers")


def is_module(src: str) -> bool:
    """Whether ``src``'s header starts with `module`."""
    return (h := H.header(src)) is not None and h.module


def _header(src: str):
    """``src``'s header (`headers.py`); ValueError when it is cut off, which leaves the file to a hand."""
    if (h := H.header(src)) is None:
        raise ValueError("its header can't be read (a comment left open, or an import without a module's name): port it "
                         "by hand")
    return h


def _comment_end(src: str, i: int) -> int:
    """Past the `/- -/` comment (or doc comment) that starts at ``i``, nesting counted; `len(src)` when it isn't closed."""
    depth, i, n = 1, i + 2, len(src)
    while i < n and depth:
        if src.startswith("/-", i):
            depth, i = depth + 1, i + 2
        elif src.startswith("-/", i):
            depth, i = depth - 1, i + 2
        else:
            i += 1
    return i


def port(src: str) -> str:
    """``src`` on the module system (the module docstring); ``src`` itself when it is a module already. Raises
    ValueError when its header can't be read or the result would not read back as that port."""
    h = _header(src)
    if h.module:
        return src
    first, body = H.skip(src, 0), H.skip(src, h.end)
    edits: list[tuple[int, str]] = []
    doc = src.startswith("/-!", body)
    if not h.end and not doc:
        edits.append((first, f"module\n\n{SECTION}\n\n"))
    else:
        edits.append((first, "module\n\n"))
        edits += [(at, "public ") for at, m in zip(H.starts(src), h.imports) if not (m.public or m.all)]
        if doc:
            edits.append((_comment_end(src, body), f"\n\n{SECTION}"))
        else:
            eol = src.find("\n", h.end)
            edits.append((len(src) if eol < 0 else eol, f"\n\n{SECTION}"))
    out, last = [], 0
    for at, text in sorted(edits, key=lambda e: e[0]):
        out += [src[last:at], text]
        last = at
    new = "".join(out) + src[last:]
    g = H.header(new)
    after = -1 if g is None else H.skip(new, g.end)
    if doc and after >= 0:
        after = H.skip(new, _comment_end(new, after))
    if not (g is not None and g.module and [m.module for m in g.imports] == [m.module for m in h.imports]
            and all(m.public or m.all for m in g.imports) and after >= 0 and new.startswith(SECTION, after)):
        raise ValueError("its header doesn't read back as ported: port it by hand")
    new = with_meta_imports(new)
    return with_option(new, "backward.privateInPublic") if PRIVATE.search(src) else new


def with_meta_imports(src: str) -> str:
    """The module ``src`` with `meta import M` after its imports for each `M` it doesn't meta-import yet, when it runs
    code while it compiles (`EVALUATES`); ``src`` itself otherwise."""
    h = _header(src)
    if not EVALUATES.search(src, H.skip(src, h.end)):
        return src
    meta = {m.module for m in h.imports if m.meta}
    more = list(dict.fromkeys(m.module for m in h.imports if m.module not in meta))
    return src[:h.end] + "".join(f"\nmeta import {m}" for m in more) + src[h.end:]


def with_option(src: str, name: str) -> str:
    """The ported ``src`` with option ``name``'s `set_option` lines (`OPTIONS`) after its `@[expose] public section`, each
    once."""
    if not is_module(src) or (at := src.find(SECTION)) < 0:
        raise ValueError(f"not ported: run `{FIX}` on it first")
    eol = src.find("\n", at)
    eol = len(src) if eol < 0 else eol
    lines = [f"set_option {o}" for o in OPTIONS[name]]
    lines = [s for s in lines if not re.search(rf"^{re.escape(s)}[ \t]*$", src, re.M)]
    return src[:eol] + "".join("\n" + s for s in lines) + src[eol:]


def has_lake_option(text: str) -> bool:
    """Whether the lakefile ``text``'s `[leanOptions]` holds `LAKE_OPTION`."""
    opts = tomllib.loads(text).get("leanOptions", {})
    return isinstance(opts, dict) and isinstance(b := opts.get("backward"), dict) and b.get("proofsInPublic") is True


def with_lake_option(text: str) -> str:
    """The lakefile ``text`` with `LAKE_OPTION` after the last key of its `[leanOptions]` table, or in a table of its own at
    the end when it has none; ``text`` itself when it holds it. Raises ValueError when the result wouldn't read back as
    holding it (the option set to something else, leanOptions written as an array or inline)."""
    if has_lake_option(text):
        return text
    lines = text.splitlines(keepends=True)
    heads = [i for i, s in enumerate(lines) if re.fullmatch(r"\[leanOptions\][ \t]*(?:#.*)?", s.strip())]
    if heads:
        end = next((j for j in range(heads[0] + 1, len(lines)) if lines[j].lstrip().startswith("[")), len(lines))
        while end > heads[0] + 1 and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
            end -= 1
        before = "".join(lines[:end])
        new = before + ("" if before.endswith("\n") else "\n") + LAKE_OPTION + "\n" + "".join(lines[end:])
    else:
        new = text.rstrip("\n") + f"\n\n[leanOptions]\n{LAKE_OPTION}\n"
    why = f"its leanOptions don't take `{LAKE_OPTION}` as a line of a `[leanOptions]` table: set it by hand"
    try:
        ok = has_lake_option(new)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(why) from e
    if not ok:
        raise ValueError(why)
    return new


def lake_option_missing(files: list[str], paths: list[str] = ()) -> list[str]:
    """The lakefiles of the packages under (or holding) ``paths`` (all when there is none) without `LAKE_OPTION`."""
    pkgs = [p for p in packages(files)
            if not paths or any(p == q.rstrip("/") or p.startswith(q.rstrip("/") + "/") or q.startswith(p + "/") for q in paths)]
    return [f"{p}/lakefile.toml" for p in pkgs if not has_lake_option((ROOT / p / "lakefile.toml").read_text())]


def _lakefile(d: Path) -> dict | None:
    f = d / "lakefile.toml"
    return tomllib.loads(f.read_text()) if f.is_file() else None


def package_modules(pkg: Path, files: list[str], root: Path | None = None) -> list[str]:
    """The tracked `.lean` files (``files``: paths from ``root``, default the repository's) of package ``pkg`` that are
    modules of one of its `[[lean_lib]]`s (the library's name or under it) or a `[[lean_exe]]`'s root, those of packages
    nested in it aside."""
    root = ROOT if root is None else root
    cfg = _lakefile(root / pkg)
    if cfg is None:
        raise ValueError(f"{pkg}: no lakefile.toml")
    src = (root / pkg / cfg.get("srcDir", ".")).resolve()
    libs = [lib["name"] for lib in cfg.get("lean_lib", [])]
    exes = {exe.get("root", exe["name"]) for exe in cfg.get("lean_exe", [])}
    nested = {f.rsplit("/", 1)[0] + "/" for f in files if f.endswith("/lakefile.toml") and f.startswith(f"{pkg}/")
              and f.rsplit("/", 1)[0] != pkg}
    out = []
    for f in files:
        if not f.endswith(".lean") or not f.startswith(f"{pkg}/") or any(f.startswith(n) for n in nested):
            continue
        path = (root / f).resolve()
        if not path.is_relative_to(src):
            continue
        mod = path.relative_to(src).with_suffix("").as_posix().replace("/", ".")
        if mod in exes or any(mod == lib or mod.startswith(lib + ".") for lib in libs):
            out.append(f)
    return out


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    return [p.decode() for p in out.split(b"\0") if p]


def packages(files: list[str]) -> list[str]:
    """Every in-tree Lake package: a directory with a tracked `lakefile.toml`, the Lean check's test fixtures aside (each
    control writes the form it tests, a plain file or a module)."""
    return sorted(f.rsplit("/", 1)[0] for f in files if f.endswith("/lakefile.toml") and not f.startswith(FIXTURES))


def modules(paths: list[str], files: list[str]) -> list[str]:
    """Every package's modules that are, or are under, one of ``paths`` (all of them when there is none)."""
    mods = sorted(f for p in packages(files) for f in package_modules(p, files))
    return [f for f in mods if not paths or any(f == q or f.startswith(q.rstrip("/") + "/") for q in paths)]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--option", choices=OPTIONS)
    a = ap.parse_args(argv)
    if a.option:
        for f in a.paths:
            p = ROOT / f
            p.write_text(with_option(p.read_text(), a.option))
        return 0
    files = tracked()
    todo, lakefiles = modules(a.paths, files), lake_option_missing(files, a.paths)
    if a.check:
        left = [f for f in todo if not is_module((ROOT / f).read_text())]
        for f in left:
            print(f"{f}: not a module: run `{FIX} {f}`")
        for f in lakefiles:
            print(f"{f}: its [leanOptions] lack `{LAKE_OPTION}`: run `{FIX} {f.rsplit('/', 1)[0]}`")
        return 1 if left or lakefiles else 0
    changed = failed = optioned = 0
    for f in lakefiles:
        p = ROOT / f
        try:
            p.write_text(with_lake_option(p.read_text()))
            optioned += 1
        except ValueError as e:
            print(f"{f}: {e}", file=sys.stderr)
            failed += 1
    for f in todo:
        p = ROOT / f
        src = p.read_text()
        try:
            new = port(src)
        except ValueError as e:
            print(f"{f}: {e}", file=sys.stderr)
            failed += 1
            continue
        if new != src:
            p.write_text(new)
            changed += 1
    print(f"{changed} of {len(todo)} module(s) ported, {optioned} lakefile(s) given `{LAKE_OPTION}`, {failed} left to port "
          "by hand")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
