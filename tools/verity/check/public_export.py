"""`public_export`: a dry run of the public monorepo's one-way export. It publishes nothing.

    uv run tools/verity/check/public_export.py DEST                  # the public set into DEST, as one commit of fresh history
    uv run tools/verity/check/public_export.py DEST --build          # then build it: each Lake package, and the uv workspace
    uv run tools/verity/check/public_export.py DEST --report F.json  # the report as JSON too
    uv run tools/verity/check/public_export.py DEST --root DIR       # export another checkout (a worktree of `next`)

DEST gets the files `public_boundary.py` judges public (`public_boundary.toml`), in a new repository with one commit. A
manifest's entry that names what the export doesn't copy is dropped: a lakefile's roots, executables, input files and path
`require`s (with their `lake-manifest.json` entries), and each pyproject's members, sources, conflicts, extras, dependency
groups and test paths. A `[project] dependencies` entry stays, and so does the source of a distribution one still needs. A
list's entry naming a path the export doesn't copy is dropped too (`PATH_LISTS`: the lint allowlists, and the dead-code
census's roots and keep-list), so a file made private takes its entries with it. Each rewrite must read, as TOML or JSON, as
the old file less exactly those entries. A path `require`, an extra's entry and a
dependency group's are the edges the check counts closed by the export (`EXPORT_CUTS`), and the export fails if its rewrite
kept one; every other edge it copies as it stands, so a build of DEST fails where a file imports, runs or reads one the
export doesn't copy.

The report gives:

* what the export copied, rewrote, and cut;
* per area of the tree, the leftovers a reader of the public tree shouldn't meet, by kind, over whole files: `art:` ids, and
  the lane-text lint's kinds (`tests/lint/test_repo_lane_text.py`: Linear, Slack, Notion and PR ids and links, agent ids,
  lane handles, dates, names);
* the known residue (`public_boundary.toml`'s `[[residue]]`), each entry's sites, and the change that removes it;
* the third-party code DEST pins: the Lean packages its manifests pin, the Python distributions its lock resolves, and the
  Rust upstream Flock patches;
* with `--build`, each Lake package's `lake build` and the workspace's `uv sync` (a node's work: `research run`), and the
  licenses of what they fetched and of the Rust upstream at its pin.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PB = _load("public_export_boundary", HERE / "public_boundary.py")
LANE_TEXT = _load("public_export_lane_text", PB.ROOT / "tests" / "lint" / "test_repo_lane_text.py")
#: what the lane-text lint doesn't look for: data ids (`fixtures/artifacts.json`'s), and agents' branches
EXTRA = (("art", re.compile(r"(?<![\w-])art:[0-9A-Za-z][\w.:/-]*")),
         ("branch", re.compile(r"(?<![\w/-])cursor/[a-z0-9]+(?:-[a-z0-9]+)*-[0-9a-f]{4}\b")))
SCRUB_MAX_BYTES = 4 << 20
#: a build step's limit, so one stuck step leaves the rest a report
STEP_SECONDS = 4 * 3600
ELAN = Path.home() / ".elan" / "bin"
#: a TOML table's header line: `[a.b]` or `[[a.b]]`, its keys bare or quoted
HEADER = re.compile(r"[ \t]*\[\[?[ \t]*((?:[\w-]+|\"[^\"\n]*\")(?:[ \t]*\.[ \t]*(?:[\w-]+|\"[^\"\n]*\"))*)[ \t]*\]\]?[ \t]*(?:#.*)?$")


class Table(NamedTuple):
    #: its header's key, `None` for the top-level table
    name: str | None
    #: its lines, from the comment lines just above its header
    text: str
    #: what TOML reads in it
    value: dict


def _scan(text: str, depth: int = 0, quote: str | None = None, close: bool = False) -> tuple[int, str | None, int]:
    """(bracket depth, the multi-line string left open, where it stopped) after ``text``, outside strings and comments;
    ``close``: stop just past the bracket that brings the depth back to 0."""
    i = 0
    while i < len(text):
        if quote is not None:
            if text.startswith(quote, i) and (quote[0] == "'" or (i - len(text[:i].rstrip("\\"))) % 2 == 0):
                i, quote = i + len(quote), None
            else:
                i += 1
            continue
        c = text[i]
        if c == "#":
            nl = text.find("\n", i)
            if nl < 0:
                break
            i = nl
            continue
        if (q := next((q for q in ('"""', "'''", '"', "'") if text.startswith(q, i)), None)) is not None:
            i, quote = i + len(q), q
            continue
        depth += (c in "[{") - (c in "]}")
        i += 1
        if close and depth == 0 and c in "]}":
            return depth, quote, i
    return depth, quote if quote in ('"""', "'''") else None, i


def _tables(text: str, where: str) -> list[Table]:
    """``text``'s tables: the top-level table, then one per header, each from the comment lines just above it."""
    parts: list[tuple[str | None, list[str], int]] = [(None, [], 0)]
    depth, quote = 0, None
    for line in text.splitlines(keepends=True):
        if depth == 0 and quote is None and (m := HEADER.match(line.rstrip("\r\n"))):
            lead = []
            while parts[-1][1] and parts[-1][1][-1].lstrip().startswith("#"):
                lead.insert(0, parts[-1][1].pop())
            parts.append((re.sub(r"[ \t]+", "", m.group(1)), [*lead, line], len(lead) + 1))
            continue
        parts[-1][1].append(line)
        depth, quote, _ = _scan(line, depth, quote)
    out = []
    for name, lines, start in parts:
        try:
            value = tomllib.loads("".join(lines[start:]))
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"public_export: {where}'s table {name or '(top)'} doesn't read as TOML on its own ({e}), so "
                             "its entries can't be dropped one by one") from e
        out.append(Table(name, "".join(lines), value))
    return out


def _array(items: list[str], key: str, multiline: bool, indent: str = "") -> str:
    if not multiline:
        return f"{indent}{key} = [{', '.join(map(json.dumps, items))}]"
    return f"{indent}{key} = [\n" + "".join(f"{indent}    {json.dumps(i)},\n" for i in items) + f"{indent}]"


def _set_array(text: str, key: str, items: list) -> str:
    """``text`` with its array ``key`` holding ``items``, on one line or one item a line as it was."""
    if not all(isinstance(i, str) for i in items):
        raise ValueError(f"public_export: {key} holds more than strings, which this rewrite doesn't lay out")
    m = re.search(rf"(?m)^([ \t]*){re.escape(key)}[ \t]*=[ \t]*(?=\[)", text)
    if m is None:
        return text
    _, _, n = _scan(text[m.end():], close=True)
    return text[:m.start()] + _array(items, key, "\n" in text[m.end():m.end() + n], m.group(1)) + text[m.end() + n:]


def _checked(text: str, expect: dict, where: str) -> str:
    try:
        got = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"public_export: the rewrite of {where} isn't TOML ({e})") from e
    if got != expect:
        raise ValueError(f"public_export: the rewrite of {where} reads as more than the old one less the entries it drops; "
                         "lay that table out one entry a line, or teach the rewrite its layout")
    return text


def _norm(*parts: str) -> str:
    return PB._rel(os.path.join(*parts))


#: the lakefile's arrays of tables the export prunes
LAKE_TABLES = ("require", "lean_lib", "lean_exe", "input_file", "input_dir")


def _module_path(src: str, mod: str) -> str:
    return _norm(src, PB.HEADERS.name(mod).replace(".", "/"))


def rewrite_lakefile(text: str, here: str, copied: set[str]) -> tuple[str, list[tuple[str, str]]]:
    """A lakefile at ``here`` without the targets and path requires the export doesn't copy, and (kind, name) of each
    entry it dropped. A library keeps each root the export copies a module at or below; a default target whose root file
    it doesn't copy (an aggregate module importing what isn't public) leaves `defaultTargets`, and its modules build one
    by one (`lean_targets`)."""
    where = _norm(here, "lakefile.toml")
    doc = tomllib.loads(text)
    src = _norm(here, doc.get("srcDir", "."))
    dirs = {d for c in copied if c.endswith(".lean") for d in _parents(c)}

    def has(mod: str) -> bool:
        return _module_path(src, mod) + ".lean" in copied

    def holds(mod: str) -> bool:
        return has(mod) or _module_path(src, mod) in dirs

    out, dropped, gone, kept = [], [], set(), defaultdict(list)
    for b in _tables(text, where):
        t, body = b.value, b.text
        name = str(t.get("name"))
        if b.name == "require" and isinstance(t.get("path"), str) and _norm(here, t["path"], "lakefile.toml") not in copied:
            dropped.append(("require", name))
            continue
        if b.name == "lean_lib":
            roots = t.get("roots", [t.get("name")])
            keep = [r for r in roots if holds(r)]
            if not keep:
                dropped.append(("lean_lib", name))
                gone.add(name)
                continue
            if keep != roots:
                dropped += [("root", r) for r in roots if r not in keep]
                body, t = _set_array(body, "roots", keep), t | {"roots": keep}
            if not all(map(has, keep)):
                gone.add(name)
        if b.name == "lean_exe" and not has(t.get("root", name)):
            dropped.append(("lean_exe", name))
            gone.add(name)
            continue
        if b.name in ("input_file", "input_dir"):
            at = _norm(here, t.get("path", ""))
            if at not in copied and not any(c.startswith(at + "/") for c in copied):
                dropped.append((b.name, name))
                gone.add(name)
                continue
        if b.name in LAKE_TABLES:
            kept[b.name].append(t)
        out.append(body)
    new = "".join(out)
    expect = {k: v for k, v in doc.items() if k not in LAKE_TABLES} | dict(kept)
    targets = doc.get("defaultTargets")
    if isinstance(targets, list) and gone & set(targets):
        left = [x for x in targets if x not in gone]
        dropped += [("defaultTarget", x) for x in targets if x in gone]
        tables = _tables(new, where)
        new = _set_array(tables[0].text, "defaultTargets", left) + "".join(b.text for b in tables[1:])
        expect["defaultTargets"] = left
    if not dropped:
        return text, []
    return _checked(new, expect, where), dropped


def _parents(path: str) -> list[str]:
    parts = path.split("/")[:-1]
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


def lean_targets(text: str, here: str, original: dict, copied: set[str]) -> dict:
    """What `lake build` builds in the exported package at ``here`` (its rewritten lakefile ``text``): its default targets,
    and each copied module of a library the original builds by default (all of them, when it names none), the root files
    the export lacks among them; with the root files it lacks."""
    doc = tomllib.loads(text)
    src = _norm(here, doc.get("srcDir", "."))
    wanted = original.get("defaultTargets")
    libs = [lib for lib in doc.get("lean_lib", []) if not isinstance(wanted, list) or lib.get("name") in wanted]
    roots = [r for lib in libs for r in lib.get("roots", [lib.get("name")])]
    prefix = f"{src}/" if src else ""
    mods = sorted({m for c in copied if c.endswith(".lean") and c.startswith(prefix)
                   for m in [c[len(prefix):].removesuffix(".lean").replace("/", ".")]
                   if any(m == r or m.startswith(f"{r}.") for r in map(PB.HEADERS.name, roots))})
    return {"default": list(doc.get("defaultTargets", [])), "modules": mods,
            "missing_roots": [r for r in roots if _module_path(src, r) + ".lean" not in copied]}


def rewrite_manifest(text: str, here: str, copied: set[str]) -> tuple[str, list[tuple[str, str]]]:
    """A `lake-manifest.json` at ``here`` without the path packages the export doesn't copy.

    An inherited path package's `dir` chains through each package that required it, and Lake resolves it literally,
    so a kept one gets its normalized `dir`: the chain may pass through a package the export doesn't copy."""
    doc = json.loads(text)
    keep, changed = [], []
    for p in doc.get("packages", []):
        if p.get("type") != "path":
            keep.append(p)
        elif _norm(here, p.get("dir", ""), p.get("configFile", "lakefile.toml")) not in copied:
            changed.append(("package", str(p.get("name"))))
        else:
            rel = Path(os.path.relpath(_norm(here, p.get("dir", "")) or ".", here or ".")).as_posix()
            if rel != p.get("dir"):
                changed.append(("dir", str(p.get("name"))))
                p = p | {"dir": rel}
            keep.append(p)
    if not changed:
        return text, []
    return json.dumps(doc | {"packages": keep}, indent=1, ensure_ascii=False) + "\n", changed


def _at(doc: dict, dotted: str) -> dict:
    for k in dotted.split("."):
        doc = doc.setdefault(k, {})
    return doc


def rewrite_pyproject(text: str, here: str, copied: set[str], dists: set[str], workspace: set[str],
                      required: set[str] = frozenset()) -> tuple[str, list[tuple[str, str]]]:
    """A pyproject at ``here`` without what names a member, distribution or test path the export doesn't copy: ``dists`` are
    the copied workspace distributions, ``workspace`` all of them. A `[project] dependencies` entry stays, an edge the build
    fails on, and so does the source of any distribution ``required`` names (one a kept dependency anywhere still needs):
    without its source, uv would look for it on the package index."""
    doc = tomllib.loads(text)
    expect, dropped, out = copy.deepcopy(doc), [], []

    def there(p: str) -> bool:
        p = _norm(here, p)
        return p in copied or any(c.startswith(p + "/") for c in copied)

    def foreign(name: str) -> bool:
        return PB._norm_dist(name) in workspace and PB._norm_dist(name) not in dists

    def dep(r: object) -> bool:
        return isinstance(r, str) and foreign(PB._requirement_name(r))

    def conflict(entry: object) -> bool:
        return isinstance(entry, list) and any(isinstance(x, dict) and foreign(str(x.get("package", ""))) for x in entry)

    def lines_without(body: str, drop) -> str:
        return "".join(line for line in body.splitlines(keepends=True) if not drop(line))

    for b in _tables(text, "pyproject.toml"):
        body, t = b.text, b.value
        if b.name == "tool.uv.workspace" and isinstance(t.get("members"), list):
            keep = [m for m in t["members"] if "*" in m or f"{_norm(here, m)}/pyproject.toml" in copied]
            if keep != t["members"]:
                dropped += [("member", m) for m in t["members"] if m not in keep]
                body, _at(expect, b.name)["members"] = _set_array(body, "members", keep), keep
        if b.name == "tool.uv.sources":
            gone = [k for k in t if foreign(k) and PB._norm_dist(k) not in required]
            dropped += [("source", k) for k in gone]
            for k in gone:
                del _at(expect, b.name)[k]
            body = lines_without(body, lambda line: (m := re.match(r"[ \t]*([\w.-]+)[ \t]*=", line)) and m.group(1) in gone)
        if b.name == "tool.uv" and isinstance(t.get("conflicts"), list) and any(map(conflict, t["conflicts"])):
            dropped += [("conflict", json.dumps(c)) for c in t["conflicts"] if conflict(c)]
            _at(expect, b.name)["conflicts"] = [c for c in t["conflicts"] if not conflict(c)]
            body = lines_without(body, lambda line: any(foreign(n) for n in re.findall(r'package[ \t]*=[ \t]*"([^"]+)"', line)))
        if b.name in ("dependency-groups", "project.optional-dependencies"):
            for group, reqs in t.items():
                if isinstance(reqs, list) and any(map(dep, reqs)):
                    dropped += [(f"{b.name.rpartition('.')[2]}.{group}", r) for r in reqs if dep(r)]
                    keep = [r for r in reqs if not dep(r)]
                    body, _at(expect, b.name)[group] = _set_array(body, group, keep), keep
        for table, keys in (("tool.pytest.ini_options", ("testpaths",)), ("tool.verity.tests", ("testpaths", "untraced"))):
            for key in keys if b.name == table else ():
                if isinstance(t.get(key), list) and not all(map(there, t[key])):
                    dropped += [(key, p) for p in t[key] if not there(p)]
                    keep = [p for p in t[key] if there(p)]
                    body, _at(expect, b.name)[key] = _set_array(body, key, keep), keep
        out.append(body)
    if not dropped:
        return text, []
    return _checked("".join(out), expect, _norm(here, "pyproject.toml")), dropped


#: the lists whose entries name the tree's paths, by their path from the directory holding their `tests/`: a lint
#: allowlist's `file`s, and the dead-code census's roots (a shell, a directory or a dotted module a line) and its keep-list's
#: modules
PATH_LISTS = {"tests/lint/allowlists/*.json": "allowlist", "tests/census_roots.txt": "roots",
              "tests/dead_code_keep.json": "keep"}


def path_list(f: str) -> tuple[str, str] | None:
    """(kind, base) when ``f`` is one of `PATH_LISTS`, ``base`` the directory holding its `tests/`."""
    base, sep, rest = f.rpartition("tests/")
    if sep and (not base or base.endswith("/")):
        for pattern, kind in PATH_LISTS.items():
            if fnmatch.fnmatchcase(sep + rest, pattern):
                return kind, base.rstrip("/")
    return None


def rewrite_path_list(text: str, f: str, tracked: set[str], copied: set[str]) -> tuple[str, list[tuple[str, str]]]:
    """The list of the tree's paths at ``f`` (`PATH_LISTS`) less each entry naming a tracked file or directory the export
    doesn't copy, so a file made private takes its entries with it. An entry names its path from the directory holding the
    list's `tests/` or, naming nothing there, from the root, as the lists' readers resolve it; a glob, and an entry naming
    nothing tracked, stay."""
    kind, base = path_list(f)
    tracked_dirs = {d for p in tracked for d in _parents(p)}
    copied_dirs = {d for p in copied for d in _parents(p)}

    def gone(entry: str, module: bool = False) -> bool:
        if any(c in entry for c in "*?["):
            return False
        p = entry.rstrip("/")
        names = [p.replace(".", "/") + s for s in (".py", "/__init__.py", "")] if module else [p]
        for n in ([_norm(base, n) for n in names] + names) if base else names:
            if n in tracked or n in tracked_dirs:
                return n not in copied and n not in copied_dirs
        return False

    lines = text.splitlines(keepends=True)
    if kind == "roots":
        def root(line: str) -> str:
            return line.split("#", 1)[0].strip()
        gone_roots = {e for line in lines if (e := root(line)) and gone(e, module=not e.endswith((".sh", "/")))}
        if not gone_roots:
            return text, []
        return "".join(line for line in lines if root(line) not in gone_roots), [("root", e) for e in sorted(gone_roots)]

    doc = json.loads(text)
    if kind == "allowlist":
        drop = [e for e in doc["entries"] if isinstance(e, dict) and isinstance(e.get("file"), str) and gone(e["file"])]
        expect = doc | {"entries": [e for e in doc["entries"] if e not in drop]}
        dropped = [("file", p) for p in dict.fromkeys(e["file"] for e in drop)]
    else:
        drop = [{m: why} for m, why in doc["modules"].items() if gone(m, module=True)]
        expect = doc | {"modules": {m: why for m, why in doc["modules"].items() if {m: why} not in drop}}
        dropped = [("module", next(iter(d))) for d in drop]
    if not dropped:
        return text, []

    def item(line: str) -> object:
        s = line.strip().rstrip(",")
        try:
            return json.loads(s if kind == "allowlist" else "{" + s + "}")
        except json.JSONDecodeError:
            return None

    new = re.sub(r",(\s*\n\s*[\]}])", r"\1", "".join(line for line in lines if item(line) not in drop))
    try:
        same = json.loads(new) == expect
    except json.JSONDecodeError:
        same = False
    if not same:
        raise ValueError(f"public_export: the rewrite of {f} reads as more than the old list less the entries it drops; "
                         "lay that list out one entry a line, or teach the rewrite its layout")
    return new, dropped


def export(tree, boundary, judgment, dest: Path) -> dict:
    """Copy ``judgment``'s public set into ``dest``, rewrite its manifests and its lists of the tree's paths, and commit it
    as a new repository's one commit."""
    copied = set(judgment.public)
    for f in sorted(copied):
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tree.root / f, dest / f)
    _, dists, _, _ = tree.python
    exported = {d for d, p in dists.items() if p in copied}
    required = {n for f in copied if f.rpartition("/")[2] == "pyproject.toml"
                for r in tomllib.loads((dest / f).read_text(encoding="utf-8")).get("project", {}).get("dependencies", [])
                if (n := PB._norm_dist(PB._requirement_name(r))) in dists and n not in exported}
    rewrites: dict[str, list[tuple[str, str]]] = {}
    targets: dict[str, dict] = {}
    tracked = set(tree.files)
    for f in sorted(copied):
        name, here = f.rpartition("/")[2], os.path.dirname(f)
        listed = path_list(f) is not None
        if name not in ("lakefile.toml", "lake-manifest.json", "pyproject.toml") and not listed:
            continue
        old = (dest / f).read_text(encoding="utf-8")
        new, dropped = (rewrite_path_list(old, f, tracked, copied) if listed
                        else rewrite_lakefile(old, here, copied) if name == "lakefile.toml"
                        else rewrite_manifest(old, here, copied) if name == "lake-manifest.json"
                        else rewrite_pyproject(old, here, copied, exported, set(dists), required))
        if dropped:
            (dest / f).write_text(new, encoding="utf-8")
            rewrites[f] = dropped
        if name == "lakefile.toml":
            targets[here or "."] = lean_targets(new, here, tomllib.loads(old), copied)
    def dropped(e) -> bool:
        if e.kind == "lake require":
            return ("require", e.what) in rewrites.get(e.source, [])
        return any(k.startswith(("dependency-groups.", "optional-dependencies.")) and v == e.what
                   for k, v in rewrites.get(e.source, []))

    if kept := [e for e in judgment.cut if not dropped(e)]:
        raise ValueError("public_export: the boundary counts these edges cut by the export, but its rewrite kept them: "
                         + "; ".join(map(str, kept)))
    cut = judgment.cut
    symlinks = sorted(f for f in PB._git_files(tree.root) if (tree.root / f).is_symlink()
                      and boundary.group(f) is not None and not boundary.is_private(f))
    git = ["git", "-c", "user.name=verity-export", "-c", "user.email=export@invalid", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=dest, check=True)
    subprocess.run(["git", "add", "-A"], cwd=dest, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "The public monorepo"], cwd=dest, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True, check=True).stdout.strip()
    return {"copied": len(copied), "rewrites": {f: [f"{k} {v}" for k, v in d] for f, d in rewrites.items()},
            "edges_cut": [str(e) for e in cut], "lean_targets": targets, "symlinks_not_copied": symlinks, "commit": head}


def scrub(dest: Path, files: list[str]) -> dict:
    """Per area and per kind, the leftovers in ``files`` of ``dest``, with the files holding the most."""
    by_area: dict[str, Counter] = defaultdict(Counter)
    per_file: Counter = Counter()
    unread = []
    for f in files:
        p = dest / f
        raw = p.read_bytes() if p.stat().st_size <= SCRUB_MAX_BYTES else None
        if raw is None or b"\0" in raw[:8192]:
            unread.append(f)
            continue
        text = raw.decode("utf-8", errors="replace")
        kinds = [what.partition(":")[0] for _, what in LANE_TEXT.found_in([(1, text)])]
        kinds += [kind for kind, pattern in EXTRA for _ in pattern.finditer(text)]
        by_area[PB._area(f)].update(kinds)
        per_file[f] += len(kinds)
    total = Counter()
    for c in by_area.values():
        total.update(c)
    return {"by_kind": dict(total.most_common()),
            "by_area": {a: dict(c.most_common()) for a, c in sorted(by_area.items(), key=lambda kv: -sum(kv[1].values()))
                        if c},
            "files": dict(per_file.most_common(40)), "not_read_binary_or_large": unread}


def residue(dest: Path, files: list[str], boundary) -> list[dict]:
    """Each known residue's (`public_boundary.toml` `[[residue]]`) sites in ``files`` of ``dest``: the lines under its
    paths that its pattern matches."""
    out = []
    for r in boundary.residue:
        sites = Counter()
        for f in files:
            if r.paths.fullmatch(f) and (dest / f).stat().st_size <= SCRUB_MAX_BYTES:
                text = (dest / f).read_text(encoding="utf-8", errors="replace")
                sites[f] = sum(1 for line in text.splitlines() if r.pattern.search(line))
        out.append({"what": r.what, "until": r.until, "sites": sum(sites.values()),
                    "files": dict(sorted(((f, n) for f, n in sites.items() if n), key=lambda kv: -kv[1]))})
    return out


def licenses_static(dest: Path, files: list[str]) -> dict:
    """The third-party code DEST pins, before anything is fetched."""
    lean: dict[tuple, dict] = {}
    for f in files:
        if f.rpartition("/")[2] == "lake-manifest.json":
            for p in json.loads((dest / f).read_text(encoding="utf-8")).get("packages", []):
                if p.get("type") == "git":
                    lean.setdefault((p.get("url"), p.get("rev")), {"name": p.get("name"), "url": p.get("url"),
                                                                   "rev": p.get("rev"), "manifests": []})["manifests"].append(f)
    python = []
    if (dest / "uv.lock").is_file():
        lock = tomllib.loads((dest / "uv.lock").read_text(encoding="utf-8"))
        python = sorted({f"{p['name']}=={p.get('version', '')}" for p in lock.get("package", [])
                         if not {"registry", "git", "url"}.isdisjoint(p.get("source", {}))})
    upstream: dict[tuple, dict] = {}
    for f in files:
        if f.endswith((".sh", ".bash")):
            text = (dest / f).read_text(encoding="utf-8", errors="replace")
            rev = re.search(r"git checkout -q ([0-9a-f]{7,40})\b", text)
            for url in re.findall(r"git clone -q (https://github\.com/[\w.-]+/[\w-]+)(?:\.git)?", text):
                key = (url, rev.group(1) if rev else None)
                upstream.setdefault(key, {"url": key[0], "rev": key[1], "named_in": []})["named_in"].append(f)
    return {"lean": sorted(lean.values(), key=lambda p: (p["name"] or "", p["rev"] or "")), "python_locked": python,
            "upstream": list(upstream.values())}


def _license_of(d: Path) -> str | None:
    for p in sorted(d.glob("LICEN[CS]E*")) + sorted(d.glob("COPYING*")):
        head = p.read_text(encoding="utf-8", errors="replace")[:3000]
        for name, needle in (("Apache-2.0", "Apache License"), ("MIT", "Permission is hereby granted"),
                             ("BSD", "Redistribution and use in source and binary forms"), ("MPL-2.0", "Mozilla Public"),
                             ("LGPL", "GNU LESSER GENERAL"), ("GPL", "GNU GENERAL PUBLIC"), ("ISC", "ISC License")):
            if needle in head:
                return f"{name} ({p.name})"
        return f"unrecognized ({p.name})"
    return None


def _tool(name: str) -> str:
    """elan's ``name`` (elan, lake), where elan-init puts it, else the one on the path."""
    return str(ELAN / name) if (ELAN / name).is_file() else shutil.which(name) or name


def _step(cmd: list[str], cwd: Path, log: Path) -> dict:
    t0 = time.monotonic()
    shown = " ".join([Path(cmd[0]).name, *cmd[1:]]) if len(cmd) <= 8 else \
        f"{' '.join([Path(cmd[0]).name, *cmd[1:6]])} ... ({len(cmd) - 2} targets)"
    with log.open("a", encoding="utf-8") as out:
        out.write(f"\n$ {shown}  (in {cwd})\n")
        out.flush()
        try:
            code: int | str = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                                             timeout=STEP_SECONDS).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        except FileNotFoundError as e:
            out.write(f"{e}\n")
            code = "not found"
    return {"cmd": shown, "exit": code, "seconds": round(time.monotonic() - t0)}


def _failures(log: Path) -> list[str]:
    text = log.read_text(encoding="utf-8", errors="replace")
    return list(dict.fromkeys(m.group(0).strip()[:300] for m in re.finditer(r"(?m)^(?:✖ .*|error: .*)$", text)))[:60]


def _rel_to(d: Path, dest: Path) -> str:
    return d.relative_to(dest).as_posix() or "."


def _pins(d: Path) -> dict[str, str]:
    m = d / "lake-manifest.json"
    doc = json.loads(m.read_text(encoding="utf-8")) if m.is_file() else {}
    return {p["name"]: p.get("rev") for p in doc.get("packages", []) if p.get("type") == "git"}


def build(dest: Path, logs: Path, targets: dict[str, dict]) -> dict:
    """`lake build` of ``targets`` (`lean_targets`) in each Lake package of ``dest`` and `uv sync` of its workspace, with the
    licenses of what they fetched. The package that pins the most dependencies goes first; every later one whose pins all
    agree with its pins shares its `.lake/packages` (one Mathlib, one prebuilt cache)."""
    logs.mkdir(parents=True, exist_ok=True)
    if not (ELAN / "elan").is_file() and shutil.which("elan") is None:
        subprocess.run("curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y "
                       "--default-toolchain none", shell=True, check=False)
    elan, lake = _tool("elan"), _tool("lake")
    pkgs = sorted((dest / rel for rel in targets), key=lambda d: (-len(_pins(d)), -len(targets[_rel_to(d, dest)]["modules"]),
                                                                  d.as_posix()))
    lean, fetched, shared, toolchains = [], {}, None, set()
    for d in pkgs:
        rel = _rel_to(d, dest)
        want = targets[rel]["default"] + [f"+{m}" for m in targets[rel]["modules"]]
        log = logs / f"lake-{rel.replace('/', '_')}.log"
        pins, steps = _pins(d), []
        toolchain = (d / "lean-toolchain").read_text(encoding="utf-8").strip()
        if toolchain not in toolchains:
            steps.append(_step([elan, "toolchain", "install", toolchain], d, log))
            toolchains.add(toolchain)
        shares = shared is not None and pins and all(shared[1].get(n) == r for n, r in pins.items())
        if shares:
            (d / ".lake").mkdir(parents=True, exist_ok=True)
            (d / ".lake" / "packages").symlink_to(shared[0], target_is_directory=True)
        steps.append(fetch := _step([lake, "env", "true"], d, log))
        if shared is None and pins and fetch["exit"] == 0:
            shared = (d / ".lake" / "packages", pins)
        if "mathlib" in pins and not shares:
            steps.append(_step([lake, "exe", "cache", "get"], d, log))
        if want:
            steps.append(_step([lake, "build", *want], d, log))
        for p in sorted((d / ".lake" / "packages").glob("*")) if (d / ".lake" / "packages").is_dir() else []:
            fetched.setdefault(p.name, _license_of(p))
        lean.append({"package": rel, "ok": all(s["exit"] == 0 for s in steps), "targets": len(want),
                     "shares_packages": bool(shares), "steps": steps, "failures": _failures(log), "log": log.name})
    log = logs / "uv-sync.log"
    uv: dict = {"steps": [_step(["uv", "sync", "--all-packages", "--extra", "torch-cpu"], dest, log)], "log": log.name}
    uv["ok"] = all(s["exit"] == 0 for s in uv["steps"])
    uv["failures"] = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                      if line.lstrip().startswith(("error", "×", "╰", "help:"))][:40]
    python: dict = {}
    py = dest / ".venv" / "bin" / "python"
    if py.is_file():
        dump = ("import importlib.metadata as m, json\n"
                "print(json.dumps({d.metadata['Name']: d.metadata.get('License-Expression') or "
                "next((c.split(' :: ')[-1] for c in d.metadata.get_all('Classifier') or [] if c.startswith('License ::')), None)"
                " or (d.metadata.get('License') or '')[:80] or None for d in m.distributions()}))")
        r = subprocess.run([str(py), "-c", dump], capture_output=True, text=True)
        python = json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-2000:]}
    return {"lean": lean, "uv": uv, "licenses": {"lean": fetched, "python": python}}


def upstream_licenses(upstream: list[dict], scratch: Path) -> list[dict]:
    """The license of each Rust upstream at its pinned revision, and its crates' licenses where `cargo` is on the path."""
    out = []
    for u in upstream:
        d = scratch / re.sub(r"\W", "_", u["url"])
        got = {"url": u["url"], "rev": u["rev"]}
        r = subprocess.run(["git", "clone", "-q", "--filter=blob:none", u["url"], str(d)], capture_output=True, text=True)
        if r.returncode == 0 and u["rev"]:
            r = subprocess.run(["git", "checkout", "-q", u["rev"]], cwd=d, capture_output=True, text=True)
        if r.returncode != 0:
            out.append(got | {"error": r.stderr[-500:]})
            continue
        got["license"] = _license_of(d)
        if shutil.which("cargo") is None or not (d / "Cargo.toml").is_file():
            got["crates"] = "cargo is not on the path" if (d / "Cargo.toml").is_file() else "no Cargo.toml"
        elif (m := subprocess.run(["cargo", "metadata", "--format-version", "1"], cwd=d, capture_output=True,
                                  text=True)).returncode == 0:
            got["crates"] = dict(Counter(p.get("license") or "none" for p in json.loads(m.stdout)["packages"]).most_common())
        else:
            got["crates"] = {"error": m.stderr[-500:]}
        out.append(got)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="public_export", description=__doc__.split("\n\n")[0])
    p.add_argument("dest", type=Path, help="an absent or empty directory, outside the checkout")
    p.add_argument("--root", type=Path, default=PB.ROOT, help="the checkout to export (default: this one)")
    p.add_argument("--build", action="store_true", help="build the export: a node's work (`research run`)")
    p.add_argument("--report", type=Path, help="write the report as JSON here; with --build, the logs go beside it")
    ns = p.parse_args(argv)
    dest, root = ns.dest.resolve(), ns.root.resolve()
    if dest.exists() and any(dest.iterdir()):
        p.error(f"{dest} isn't empty")
    if dest == root or root in dest.parents:
        p.error("DEST is inside the checkout it exports")
    dest.mkdir(parents=True, exist_ok=True)
    boundary = PB.Boundary.load()
    tree = PB.Tree(root)
    judgment = PB.judge(tree, boundary)
    files = sorted(judgment.public)
    source = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    report = {"source": {"root": str(root), "commit": source}, "dest": str(dest), "edges": len(judgment.edges)}
    report |= export(tree, boundary, judgment, dest)
    report["scrub"] = scrub(dest, files)
    report["residue"] = residue(dest, files, boundary)
    report["third_party"] = licenses_static(dest, files)
    if ns.build:
        report["build"] = build(dest, (ns.report.parent if ns.report else dest.parent) / "export-logs", report["lean_targets"])
        with tempfile.TemporaryDirectory() as scratch:
            report["third_party"]["upstream_licenses"] = upstream_licenses(report["third_party"]["upstream"], Path(scratch))
    if ns.report:
        ns.report.parent.mkdir(parents=True, exist_ok=True)
        ns.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"public_export: {report['copied']} files from {source[:10]} into {dest} ({report['commit'][:10]}); "
          f"{report['edges']} open edges, {len(report['edges_cut'])} more cut by the rewrites")
    for f, dropped in report["rewrites"].items():
        print(f"  {f}: dropped {len(dropped)} ({', '.join(dropped[:6])}{', ...' if len(dropped) > 6 else ''})")
    lt = report["lean_targets"]
    print(f"Lake packages: {len(lt)}, {sum(len(t['modules']) for t in lt.values())} modules to build; root files the export "
          f"lacks: {[f'{p}: {r}' for p, t in lt.items() for r in t['missing_roots']]}")
    print(f"leftovers by kind: {report['scrub']['by_kind']}")
    for area, kinds in list(report["scrub"]["by_area"].items())[:25]:
        print(f"  {area}: {kinds}")
    for r in report["residue"]:
        print(f"known residue, until {r['until']}: {r['sites']} sites in {len(r['files'])} files: {r['what']}")
    tp = report["third_party"]
    print(f"third party: {len(tp['lean'])} Lean packages ({', '.join(sorted({str(x['name']) for x in tp['lean']}))}); "
          f"{len(tp['python_locked'])} locked Python distributions; upstream {[(u['url'], u['rev']) for u in tp['upstream']]}")
    if "build" in report:
        for b in report["build"]["lean"]:
            print(f"  lake {b['package']}: {'ok' if b['ok'] else 'FAILED'} ("
                  + ", ".join(f"{s['cmd'].split()[1]} {s['exit']} {s['seconds']}s" for s in b["steps"]) + ")")
        print(f"  uv sync: {'ok' if report['build']['uv']['ok'] else 'FAILED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
