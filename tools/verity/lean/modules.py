"""`modules`: the codemod that puts the in-tree Lake packages' Lean on Lean's module system, and the header reader it uses.

    python3 tools/verity/lean/modules.py [PATH...]                  port every module of each package under PATH (default: all)
    python3 tools/verity/lean/modules.py --check [PATH...]          name the files not yet ported; exit 1 when there is one
    python3 tools/verity/lean/modules.py --option NAME FILE...      add `set_option NAME true` to each ported FILE

A package's modules are the `.lean` files of its `[[lean_lib]]`s and `[[lean_exe]]` roots; a script beside them (run with
`lean --run`) is not one and is left alone. A file whose header already starts with `module` is left as it is, so a rerun
changes nothing and a branch made before the port takes it by running this on its head. Porting a file:

- puts `module` first in its header, after the comments that open the file;
- makes each `import M` a `public import M` and each `meta import M` a `public meta import M` (`import all M` stays), so an
  importer sees what it saw before;
- opens `@[expose] public section` after the header (after the module docstring when one follows it), never closed, so every
  declaration is public and every definition's value is exposed, as without the module system; `private` still means
  private, and a theorem's proof is never exposed.

What the module system refuses beyond that is fixed by hand in the file, which this then leaves alone: a tactic's helper
gets `meta`, a private declaration a public statement reads becomes public. `--option` adds one of the transition options
(`backward.privateInPublic`, `backward.proofsInPublic`) to a file that needs it instead.

`header` reads a header as Lean does (`module`, then `prelude`, then each `[public] [meta] import [all] M`, with comments
anywhere between; the header ends at the first other token). Stdlib only."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[3]
SECTION = "@[expose] public section"
OPTIONS = ("backward.privateInPublic", "backward.proofsInPublic")
FIX = "python3 tools/verity/lean/modules.py"
FIXTURES = "tools/verity/lean/tests/fixtures/"


class Import(NamedTuple):
    start: int                 # its first word: `public`, `meta` or `import`
    end: int                   # past the module's name
    words: tuple[str, ...]     # `public`, `meta` and `all`, as written
    module: str


class Header(NamedTuple):
    first: int                 # the file's first token, past the comments that open it
    module: bool
    imports: tuple[Import, ...]
    end: int                   # past the header's last token (`first` when it has none)
    body: int                  # the first token after the header, `len(src)` when there is none


def _skip(src: str, i: int) -> int:
    """Past the whitespace and comments at ``i``: `--` to the line's end and `/- -/`, nested, but not a doc comment (`/--`,
    `/-!`), which is a command."""
    n = len(src)
    while i < n:
        if src[i].isspace():
            i += 1
        elif src.startswith("--", i):
            j = src.find("\n", i)
            i = n if j < 0 else j + 1
        elif src.startswith("/-", i) and not src.startswith(("/--", "/-!"), i):
            i = _comment_end(src, i)
        else:
            break
    return i


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


def _token(src: str, i: int) -> tuple[str, int]:
    j, n = i, len(src)
    while j < n and not src[j].isspace() and not src.startswith(("--", "/-"), j):
        if src[j] == "«":
            k = src.find("»", j)
            j = n if k < 0 else k + 1
        else:
            j += 1
    return src[i:j], j


def header(src: str) -> Header:
    """``src``'s header (the module docstring)."""
    first = i = _skip(src, 1 if src.startswith("\ufeff") else 0)
    end, is_module, imports = first, False, []
    tok, j = _token(src, i)
    if tok == "module":
        is_module, end = True, j
        i = _skip(src, j)
        tok, j = _token(src, i)
    if tok == "prelude":
        end = j
        i = _skip(src, j)
        tok, j = _token(src, i)
    while True:
        words, k, t, kj = [], i, tok, j
        while t in ("public", "meta") and t not in words:
            words.append(t)
            k = _skip(src, kj)
            t, kj = _token(src, k)
        if t != "import":
            break
        k = _skip(src, kj)
        t, kj = _token(src, k)
        if t == "all":
            words.append(t)
            k = _skip(src, kj)
            t, kj = _token(src, k)
        imports.append(Import(i, kj, tuple(words), t))
        end = kj
        i = _skip(src, kj)
        tok, j = _token(src, i)
    return Header(first, is_module, tuple(imports), end, i)


def port(src: str) -> str:
    """``src`` on the module system (the module docstring); ``src`` itself when it is a module already. Raises
    ValueError when the result would not read back as that port."""
    h = header(src)
    if h.module:
        return src
    edits: list[tuple[int, str]] = []
    doc = src.startswith("/-!", h.body)
    if h.end == h.first and not doc:
        edits.append((h.first, f"module\n\n{SECTION}\n\n"))
    else:
        edits.append((h.first, "module\n\n"))
        edits += [(m.start, "public ") for m in h.imports if "public" not in m.words and "all" not in m.words]
        if doc:
            edits.append((_comment_end(src, h.body), f"\n\n{SECTION}"))
        else:
            eol = src.find("\n", h.end)
            edits.append((len(src) if eol < 0 else eol, f"\n\n{SECTION}"))
    out, last = [], 0
    for at, text in sorted(edits, key=lambda e: e[0]):
        out += [src[last:at], text]
        last = at
    new = "".join(out) + src[last:]
    g = header(new)
    body = _skip(new, _comment_end(new, g.body)) if doc else g.body
    if not (g.module and [m.module for m in g.imports] == [m.module for m in h.imports]
            and all("public" in m.words or "all" in m.words for m in g.imports) and new.startswith(SECTION, body)):
        raise ValueError("its header doesn't read back as ported: port it by hand")
    return new


def with_option(src: str, name: str) -> str:
    """The ported ``src`` with `set_option NAME true` after its `@[expose] public section`, once."""
    line = f"set_option {name} true"
    if re.search(rf"^{re.escape(line)}[ \t]*$", src, re.M):
        return src
    if not header(src).module or (at := src.find(SECTION)) < 0:
        raise ValueError(f"not ported: run `{FIX}` on it first")
    eol = src.find("\n", at)
    eol = len(src) if eol < 0 else eol
    return src[:eol] + "\n" + line + src[eol:]


def _lakefile(d: Path) -> dict | None:
    f = d / "lakefile.toml"
    return tomllib.loads(f.read_text()) if f.is_file() else None


def package_modules(pkg: Path, files: list[str]) -> list[str]:
    """The tracked `.lean` files (``files``: paths from the root) of package ``pkg`` that are modules of one of its
    `[[lean_lib]]`s (the library's name or under it) or a `[[lean_exe]]`'s root, those of packages nested in it aside."""
    cfg = _lakefile(ROOT / pkg)
    if cfg is None:
        raise ValueError(f"{pkg}: no lakefile.toml")
    src = (ROOT / pkg / cfg.get("srcDir", ".")).resolve()
    libs = [lib["name"] for lib in cfg.get("lean_lib", [])]
    exes = {exe.get("root", exe["name"]) for exe in cfg.get("lean_exe", [])}
    nested = {f.rsplit("/", 1)[0] + "/" for f in files if f.endswith("/lakefile.toml") and f.startswith(f"{pkg}/")
              and f.rsplit("/", 1)[0] != pkg}
    out = []
    for f in files:
        if not f.endswith(".lean") or not f.startswith(f"{pkg}/") or any(f.startswith(n) for n in nested):
            continue
        try:
            mod = (ROOT / f).resolve().relative_to(src).with_suffix("").as_posix().replace("/", ".")
        except ValueError:
            continue
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
    todo = modules(a.paths, tracked())
    if a.check:
        left = [f for f in todo if not header((ROOT / f).read_text()).module]
        for f in left:
            print(f"{f}: not a module: run `{FIX} {f}`")
        return 1 if left else 0
    changed = failed = 0
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
    print(f"{changed} of {len(todo)} module(s) ported, {failed} left to port by hand")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
