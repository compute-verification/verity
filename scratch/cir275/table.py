"""Join the import scan (`scan.py`) with the sides (`sides.py`) into `files.json`, `files.md` and `size.md`.

For each file of `adapters/vllm/verity_vllm/program/` at one commit: its lines, its side, whether it loads torch or vllm and
by which chain, and, for a file or part on circuits' side (`kernel-circuits`, `cut`), every name it uses that a file of the
other side (`vllm`, `pytorch-adapter`) defines, classed as data, code or type. Reads files with `git cat-file` at the
commit and parses them with `ast`; imports nothing of the repository.

A name's use is found by `ast`: a Name bound by an import, or an attribute chain on an imported module, anywhere in the
file (local rebinding of an imported name is not tracked). Its class is its definition's, followed through re-exports:
a class is `type`, a function `code`, an assignment `data` (`type` for a typing alias or a namedtuple/TypeVar/NewType
call), a module `module`.

Usage: python3 scratch/cir275/table.py [COMMIT]   (default: origin/main)
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scan as S  # noqa: E402
import sides as D  # noqa: E402

CIRCUITS = ("kernel-circuits", "cut")
OTHER = ("vllm", "pytorch-adapter")
SIDE_ORDER = ("kernel-circuits", "cut", "vllm", "pytorch-adapter")
TYPE_CALLS = {"namedtuple", "NamedTuple", "TypeVar", "NewType", "make_dataclass", "TypedDict"}
MOVE_DIR = "tools/move/"


def sources(sha: str, paths: list[str]) -> dict[str, str]:
    """path -> text at `sha`, through one `git cat-file --batch`."""
    p = subprocess.run(["git", "-C", str(S.REPO), "cat-file", "--batch"], input="".join(f"{sha}:{x}\n" for x in paths).encode(),
                       capture_output=True, check=True)
    out, buf, i = {}, p.stdout, 0
    for x in paths:
        nl = buf.index(b"\n", i)
        head = buf[i:nl].split()
        if head[-1] == b"missing":
            i = nl + 1
            continue
        n = int(head[2])
        out[x] = buf[nl + 1:nl + 1 + n].decode("utf-8", "replace")
        i = nl + 1 + n + 1
    return out


class File:
    """One parsed file: its top-level units, its import bindings and the names it uses."""

    def __init__(self, path: str, mod: str, src: str, known: dict[str, str]):
        self.path, self.mod, self.src, self.known = path, mod, src, known
        self.n = len(src.splitlines())
        self.tree = ast.parse(src)
        self.units: dict[str, tuple[str, int, int]] = {}
        self.defs: set[str] = set()
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.units[node.name] = ("code", node.lineno, node.end_lineno)
                self.defs.add(node.name)
            elif isinstance(node, ast.ClassDef):
                self.units[node.name] = ("type", node.lineno, node.end_lineno)
                self.defs.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                kind = self._assign_kind(node.value)
                for t in targets:
                    for nm in ([t] if isinstance(t, ast.Name) else getattr(t, "elts", [])):
                        # a rebinding of a def (`f = wrap(f)`) keeps the def's range and class
                        if isinstance(nm, ast.Name) and nm.id not in self.defs:
                            self.units[nm.id] = (kind, node.lineno, node.end_lineno)
        self.bindings = self._bindings()

    @staticmethod
    def _assign_kind(value) -> str:
        if isinstance(value, ast.Call):
            f = value.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in TYPE_CALLS:
                return "type"
        if isinstance(value, ast.Subscript) or (isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr)):
            return "type"
        return "data"

    def _bindings(self) -> dict[str, list[dict]]:
        """local name -> [{module, attr (None for a module), line, ctx}], every import statement that binds it."""
        is_pkg = self.path.endswith("/__init__.py")
        pkg = self.mod if is_pkg else self.mod.rpartition(".")[0]
        out: dict[str, list[dict]] = defaultdict(list)
        for node, ctx, sym in S.import_nodes(self.tree):
            # an import inside a def binds only there: the top-level unit's lines
            top = sym.split(".")[0]
            scope = self.units[top][1:] if sym != "<module>" and top in self.units else None
            ctx = {"ctx": ctx, "scope": scope}
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        out[a.asname].append({"module": a.name, "attr": None, "line": node.lineno, "ctx": ctx})
                    else:
                        top = a.name.split(".")[0]
                        out[top].append({"module": top, "attr": None, "line": node.lineno, "ctx": ctx})
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = pkg.split(".")
                    base = ".".join([*parts[: len(parts) - node.level + 1], *([base] if base else [])])
                for a in node.names:
                    if a.name == "*":
                        continue
                    sub = f"{base}.{a.name}"
                    rec = {"module": sub, "attr": None} if sub in self.known else {"module": base, "attr": a.name}
                    out[a.asname or a.name].append({**rec, "line": node.lineno, "ctx": ctx})
        return out

    def uses(self):
        """(line, local name, module, attr or '*') for every use of an imported name."""
        parent = {}
        for node in ast.walk(self.tree):
            for ch in ast.iter_child_nodes(node):
                parent[ch] = node
        consumed = set()
        out = []

        def live(name: str, line: int) -> list[dict]:
            return [b for b in self.bindings.get(name, ()) if b["ctx"]["scope"] is None
                    or b["ctx"]["scope"][0] <= line <= b["ctx"]["scope"][1]]

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute) and not isinstance(parent.get(node), ast.Attribute):
                chain, cur = [], node
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                if not isinstance(cur, ast.Name) or not live(cur.id, node.lineno):
                    continue
                chain.reverse()
                for b in live(cur.id, node.lineno):
                    consumed.add(id(cur))
                    if b["attr"] is not None:
                        out.append((node.lineno, cur.id, b["module"], b["attr"], b["line"]))
                        continue
                    m = b["module"]
                    attr = "*"
                    for comp in chain:
                        if f"{m}.{comp}" in self.known:
                            m = f"{m}.{comp}"
                            continue
                        attr = comp
                        break
                    out.append((node.lineno, cur.id, m, attr, b["line"]))
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and id(node) not in consumed:
                for b in live(node.id, node.lineno):
                    out.append((node.lineno, node.id, b["module"], b["attr"] or "*", b["line"]))
        return out


class Sides:
    """Each scope file's side by line."""

    def __init__(self, files: dict[str, File], scope_rel):
        self.ranges: dict[str, list[tuple[int, int, str]]] = {}
        self.header: dict[str, str] = {}
        self.split_units: dict[str, dict[str, list[tuple[str, int, int]]]] = {}
        errors = []
        for path, f in files.items():
            rel = scope_rel(path)
            side = D.FILES[rel][0]
            if side != "split":
                self.ranges[path] = [(1, max(f.n, 1), side)]
                continue
            spec = D.SPLITS[rel]
            self.header[path] = spec["header"]
            rs, named = [], defaultdict(list)
            for s, units in spec["parts"].items():
                for u in units:
                    name, _, span = u.partition(":")
                    if name not in f.units:
                        errors.append(f"{rel}: no top-level {name!r}")
                        continue
                    _k, lo, hi = f.units[name]
                    if span:
                        a, b = (int(x) for x in span.split("-"))
                        if not (lo <= a <= b <= hi):
                            errors.append(f"{rel}: {u} outside {name} {lo}-{hi}")
                        lo, hi = a, b
                    rs.append((lo, hi, s))
                    named[s].append((u if span else name, lo, hi))
            rs.sort()
            for (a1, b1, _), (a2, b2, _) in zip(rs, rs[1:]):
                if a2 <= b1:
                    errors.append(f"{rel}: parts overlap at {a2}")
            for name, (kind, lo, hi) in f.units.items():
                covered = sum(max(0, min(hi, b) - max(lo, a) + 1) for a, b, _ in rs)
                if name in f.defs or hi - lo > 3:
                    if covered != hi - lo + 1:
                        errors.append(f"{rel}: {name} {lo}-{hi} not wholly in a part ({covered} of {hi - lo + 1} lines)")
            self.ranges[path] = rs
            self.split_units[path] = dict(named)
        if errors:
            raise SystemExit("sides.py does not cover the files:\n  " + "\n  ".join(errors))

    def part_at(self, path: str, line: int) -> str | None:
        """the side of the named part holding `line`, None in a split file's header"""
        for a, b, s in self.ranges[path]:
            if a <= line <= b:
                return s
        return None

    def at(self, path: str, line: int) -> str:
        return self.part_at(path, line) or self.header[path]

    def lines(self, path: str, n: int) -> dict[str, int]:
        out = defaultdict(int)
        if path not in self.header:
            out[self.ranges[path][0][2]] = n
            return dict(out)
        inside = 0
        for a, b, s in self.ranges[path]:
            out[s] += b - a + 1
            inside += b - a + 1
        out["header"] = n - inside
        return dict(out)


def render_chain(c) -> str | None:
    if not c:
        return None
    parts = [Path(c["steps"][0]["from"]).as_posix().split("verity_vllm/", 1)[-1]] if c["steps"] else []
    for s in c["steps"]:
        tag = "parent" if s["ctx"] == "parent" else f"{s['line']} {s['ctx']}"
        parts.append(f"({tag}) {s['to'].split('verity_vllm/', 1)[-1] if 'verity_vllm/' in s['to'] else s['to']}")
    h = c["heavy"]
    parts.append(f":{h['line']} {h['ctx']} `{h['text']}`")
    return " → ".join(parts[:-1]) + " " + parts[-1] if parts[:-1] else parts[-1]


def main() -> None:
    commit = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    sha = S.git("rev-parse", commit).strip()
    raw = S.scan(sha)
    known, path_to_mod, scope_files = S.modules(sha)
    rel = lambda p: p[len(S.SCOPE):]  # noqa: E731

    missing = sorted(set(map(rel, scope_files)) - set(D.FILES))
    extra = sorted(set(D.FILES) - set(map(rel, scope_files)))
    if missing or extra:
        raise SystemExit(f"sides.py: files without a side {missing}; sides for no file {extra}")
    bad = sorted(r for r, v in D.FILES.items() if (v[0] == "split") != (r in D.SPLITS) or v[0] not in (*SIDE_ORDER, "split")
                 or v[1] not in ("docstring", "code", "inferred"))
    if bad:
        raise SystemExit(f"sides.py: malformed entries {bad}")

    # every repository .py file, for its importers of program/
    all_py = [f for f in S.git("ls-tree", "-r", "--name-only", sha).splitlines() if f.endswith(".py")]
    texts = sources(sha, all_py)
    files: dict[str, File] = {p: File(p, path_to_mod[p], texts[p], known) for p in scope_files}
    sides = Sides(files, rel)
    parsed: dict[str, File | None] = dict(files)

    def file_of(mod: str) -> File | None:
        path = known.get(mod)
        if path is None:
            return None
        if path not in parsed:
            try:
                parsed[path] = File(path, mod, texts[path], known)
            except (SyntaxError, KeyError):
                parsed[path] = None
        return parsed[path]

    def define(mod: str, attr: str, depth: int = 0):
        """(path, name, kind, line) of attr's definition, following re-exports; kind 'unresolved' where not found."""
        if attr == "*" or f"{mod}.{attr}" in known:
            m = mod if attr == "*" else f"{mod}.{attr}"
            return known.get(m), m, "module", 0
        f = file_of(mod)
        if f is None:
            return None, f"{mod}.{attr}", "external", 0
        if attr in f.units:
            k, lo, _ = f.units[attr]
            return f.path, attr, k, lo
        if attr in f.bindings and depth < 6:
            b = f.bindings[attr][0]
            return define(b["module"], b["attr"] or "*", depth + 1)
        return f.path, attr, "unresolved", 0

    def bound_line(mod: str, attr: str) -> int:
        """the line in mod's file that binds attr: its unit's first line, or the import that binds it."""
        f = file_of(mod)
        if f is None or attr == "*":
            return 0
        if attr in f.units:
            return f.units[attr][1]
        if attr in f.bindings:
            return f.bindings[attr][0]["line"]
        return 0

    def target_side(mod: str, attr: str) -> str | None:
        path = known.get(mod)
        if path not in sides.ranges:
            return None
        if attr == "*" and path in sides.header:
            return "split (module object)"
        line = bound_line(mod, attr)
        return sides.at(path, line) if line else sides.ranges[path][0][2] if path not in sides.header else sides.header[path]

    # uses inside the scope: item 4, the part graph and the cross-line import lines
    cross: dict[str, list[dict]] = defaultdict(list)
    outside: dict[str, list[dict]] = defaultdict(list)
    edges: dict[tuple[str, str], dict[tuple[str, str], set]] = defaultdict(lambda: defaultdict(set))
    cross_lines: set[tuple[str, int]] = set()
    for path, f in files.items():
        for line, local, mod, attr, imp_line in f.uses():
            here = sides.at(path, line)
            tpath = known.get(mod)
            if tpath in sides.ranges and tpath != path:
                ts = target_side(mod, attr)
                edges[(path, here)][(tpath, ts)].add(attr)
                if here in CIRCUITS and ts not in CIRCUITS:
                    dpath, dname, kind, dline = define(mod, attr)
                    cross[path].append({"name": attr, "from": tpath, "from_side": ts, "kind": kind, "defined": dpath,
                                        "defined_line": dline, "used_at": line, "used_by_side": here, "import_line": imp_line})
                    cross_lines.add((path, imp_line))
            elif here in CIRCUITS and (mod.startswith("verity_vllm.") or mod == "verity_vllm") and tpath not in sides.ranges:
                dpath, dname, kind, dline = define(mod, attr)
                outside[path].append({"name": attr, "from": tpath or mod, "kind": kind, "used_at": line, "used_by_side": here,
                                      "import_line": imp_line})
            elif here in CIRCUITS and mod.split(".")[0] in S.HEAVY:
                outside[path].append({"name": attr, "from": mod, "kind": "external", "used_at": line, "used_by_side": here,
                                      "import_line": imp_line})

    # inside a split file, a part that names a unit of a part on the other side of the line: an import once split
    for path in sides.header:
        f = files[path]
        for node in ast.walk(f.tree):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in f.units
                    and node.id not in f.bindings):
                continue
            here, there = sides.part_at(path, node.lineno), sides.part_at(path, f.units[node.id][1])
            if here is None or there is None or here == there or (here in CIRCUITS) == (there in CIRCUITS):
                continue
            edges[(path, here)][(path, there)].add(node.id)
            if here in CIRCUITS:
                cross[path].append({"name": node.id, "from": path, "from_side": there, "kind": f.units[node.id][0],
                                    "defined": path, "defined_line": f.units[node.id][1], "used_at": node.lineno,
                                    "used_by_side": here, "import_line": None})

    def group(recs):
        g: dict[tuple, dict] = {}
        for r in recs:
            k = (r["name"], r["from"], r["kind"])
            e = g.setdefault(k, {**{x: r[x] for x in r if x not in ("used_at",)}, "used_at": []})
            e["used_at"] = sorted(set(e["used_at"]) | {r["used_at"]})
        return sorted(g.values(), key=lambda r: (r["from"], r["name"]))

    # repository-wide import lines naming a file (or a split file's part) that moves to a circuits directory
    moves_whole = {p for p in scope_files if D.FILES[rel(p)][0] in CIRCUITS}
    moves_part = {p for p in scope_files if D.FILES[rel(p)][0] == "split"
                  and any(s in CIRCUITS for s in D.SPLITS[rel(p)]["parts"])}
    importer_lines: dict[str, set] = defaultdict(set)
    module_object_lines: dict[str, set] = defaultdict(set)
    for p in all_py:
        if p not in texts or (p not in files and "program" not in texts[p]):
            continue
        f = file_of(path_to_mod[p]) if p in path_to_mod else None
        if f is None:
            try:
                f = File(p, "_outside_", texts[p], known)
            except SyntaxError:
                continue
        for line, local, mod, attr, imp_line in f.uses():
            tpath = known.get(mod)
            if tpath == p:
                continue
            if tpath in moves_whole:
                importer_lines[tpath].add((p, imp_line))
            elif tpath in moves_part:
                ts = target_side(mod, attr)
                if ts in CIRCUITS:
                    importer_lines[tpath].add((p, imp_line))
                elif ts == "split (module object)":
                    module_object_lines[tpath].add((p, imp_line))

    # the move maps: does any name a new path for a program/ file?
    move_files = [x for x in all_py + [f for f in S.git("ls-tree", "-r", "--name-only", sha, MOVE_DIR).splitlines()]
                  if x.startswith(MOVE_DIR)]
    move_texts = sources(sha, sorted(set(move_files)))
    mapped = {}
    for p in scope_files:
        hits = sorted(m for m, t in move_texts.items() if p in t or path_to_mod[p] in t)
        if hits:
            mapped[p] = hits

    rows = []
    for r in raw["files"]:
        p = r["path"]
        side, basis, reason, *doubt = D.FILES[rel(p)]
        f = files[p]
        row = {
            "path": p, "rel": rel(p), "lines": r["lines"], "side": side,
            "lines_by_side": sides.lines(p, r["lines"]),
            "loads": r["loads"], "loads_own": r["loads_own"],
            "direct": r["direct"],
            "chain": r["chain_at_import"] or r["chain_at_call"] or r["chain_type_checking"],
            "chain_kind": "at-import" if r["chain_at_import"] else "at-call" if r["chain_at_call"] else
                          "type-checking" if r["chain_type_checking"] else None,
            "chain_own": r["chain_at_import_own"] or r["chain_at_call_own"],
            "reason": reason, "reason_basis": basis, "doubt": doubt[0] if doubt else None,
            "split": None, "imports_from_other_side": group(cross.get(p, [])),
            "imports_from_outside_table": group(outside.get(p, [])),
            "new_path": None, "new_path_note": "no map in tools/move/ names one" if p not in mapped else f"named in {mapped[p]}",
        }
        if side == "split":
            row["split"] = {"header": sides.header[p],
                            "parts": {s: [{"unit": u, "lines": [a, b]} for u, a, b in units]
                                      for s, units in sides.split_units[p].items()}}
        if p in moves_whole or p in moves_part:
            row["importer_import_lines"] = len(importer_lines.get(p, ()))
            row["importer_files"] = len({x for x, _ in importer_lines.get(p, ())})
        rows.append(row)

    # counts
    by_dir = defaultdict(int)
    for r in rows:
        d = r["rel"].split("/")[0] if "/" in r["rel"] else "top"
        by_dir[d] += 1
    side_files = defaultdict(int)
    side_lines = defaultdict(int)
    for r in rows:
        side_files[r["side"]] += 1
        for s, n in r["lines_by_side"].items():
            side_lines[s] += n
    split_lines = defaultdict(int)
    for r in rows:
        if r["side"] == "split":
            for s, n in r["lines_by_side"].items():
                split_lines[s] += n
    imports_heavy = [r for r in rows if r["direct"]]
    counts = {
        "files": len(rows), "lines": sum(r["lines"] for r in rows), "files_by_dir": dict(by_dir),
        "files_by_side": dict(side_files), "lines_by_side_all": dict(side_lines),
        "lines_in_split_files_by_side": dict(split_lines),
        "files_importing_torch_or_vllm": len(imports_heavy),
        "files_importing_torch_or_vllm_by_dir": {d: sum(1 for r in imports_heavy if (r["rel"].split("/")[0] if "/" in r["rel"] else "top") == d)
                                                  for d in by_dir},
        "loads": {k: sum(1 for r in rows if r["loads"] == k) for k in sorted({r["loads"] for r in rows})},
        "loads_own": {k: sum(1 for r in rows if r["loads_own"] == k) for k in sorted({r["loads_own"] for r in rows})},
    }

    # part graph: mutual imports across the line
    pairs = []
    for (pa, sa), outs in edges.items():
        if sa not in CIRCUITS:
            continue
        for (pb, sb), names in outs.items():
            if sb not in OTHER:
                continue
            back = edges.get((pb, sb), {}).get((pa, sa))
            if back:
                pairs.append({"circuits": {"file": pa, "side": sa}, "other": {"file": pb, "side": sb},
                              "circuits_uses": sorted(names), "other_uses": sorted(back)})
    pairs.sort(key=lambda p: (p["circuits"]["file"], p["other"]["file"]))

    edited = {
        "cross_line_import_lines_in_circuits_parts": len(cross_lines),
        "cross_line_import_files": len({p for p, _ in cross_lines}),
        "importer_import_lines_of_moving_files": sum(len(v) for v in importer_lines.values()),
        "importer_files_of_moving_files": len({x for v in importer_lines.values() for x, _ in v}),
        "module_object_import_lines_of_split_files": sum(len(v) for v in module_object_lines.values()),
        "split_parts_naming_the_other_side": sum(1 for v in cross.values() for r in v if r["import_line"] is None),
        "one_way_edges_circuits_to_other": sum(1 for (pa, sa), outs in edges.items() if sa in CIRCUITS
                                               for (pb, sb) in outs if sb in OTHER),
        "one_way_edges_other_to_circuits": sum(1 for (pa, sa), outs in edges.items() if sa in OTHER
                                               for (pb, sb) in outs if sb in CIRCUITS),
    }
    out = {"schema": "cir275/files/v1", "commit": sha, "scope": S.SCOPE,
           "sides": {"kernel-circuits": "circuits: each served kernel's circuit", "cut": "circuits: the cut of a period's circuit",
                     "vllm": "vllm: a model and the advice into kernel calls, reading vLLM", "pytorch-adapter":
                     "the PyTorch adapter: the same, reading torch alone", "split": "crosses the line; `split` names each part"},
           "counts": counts, "edited": edited, "mutual_pairs": pairs, "rows": rows,
           "not_placed": [{"files": a, "why": b} for a, b in D.NOT_PLACED]}
    (HERE / "files.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    (HERE / "files.md").write_text(render_md(out))
    (HERE / "size.md").write_text(render_size(out))
    print(sha, counts["files"], "files", counts["lines"], "lines;", dict(side_files))


def short(p: str) -> str:
    return p[len(S.SCOPE):] if p and p.startswith(S.SCOPE) else (p or "")


def render_md(out: dict) -> str:
    rows, c = out["rows"], out["counts"]
    L = [f"# `program/` by side of the list of kernel calls", "",
         f"Generated by `table.py` from `sides.py` and `scan.py` at `{out['commit']}`; `files.json` (schema `{out['schema']}`) is the "
         f"record and this its rendering. {c['files']} files, {c['lines']} lines. Paths are under `{out['scope']}`.", "",
         "Columns: lines; side; *loads* is whether importing the file loads torch or vllm (`direct-at-import`, "
         "`direct-at-import-optional`, `chain-at-import`, `direct-at-call`, `chain-at-call`, `type-checking-only`, `no`), and "
         "*own* the same with the file's own package `__init__` files taken to load nothing; *chain* is the first chain found "
         "(each step's line and context, then the heavy import); *reason* the side's reason, its basis in brackets. No map in "
         "`tools/move/` names a new path for any file, so the new-path column is omitted (each row's `new_path_note` in "
         "`files.json`).", ""]
    L += ["| file | lines | side | loads | own | chain | reason |", "|---|---:|---|---|---|---|---|"]
    for r in rows:
        ch = render_chain(r["chain"]) or ""
        reason = f"{r['reason']} [{r['reason_basis']}]" + (f" Doubt: {r['doubt']}." if r["doubt"] else "")
        L.append(f"| `{r['rel']}` | {r['lines']} | {r['side']} | {r['loads']} | {r['loads_own']} | {ch.replace('|', '/')} | "
                 f"{reason.replace('|', '/')} |")
    L += ["", "## Split files", ""]
    for r in rows:
        if r["side"] != "split":
            continue
        lb = ", ".join(f"{s} {n}" for s, n in sorted(r["lines_by_side"].items()))
        L.append(f"### `{r['rel']}` ({r['lines']} lines: {lb}; header to {r['split']['header']})")
        L.append("")
        for s, units in r["split"]["parts"].items():
            L.append(f"- **{s}**: " + ", ".join(f"`{u['unit']}` {u['lines'][0]}–{u['lines'][1]}" for u in units))
        L.append("")
    L += ["## Names circuits' side takes from the other side", "",
          "For each file or part on `kernel-circuits` or `cut`: every name it uses that a `vllm` or `pytorch-adapter` file or part "
          "binds, with its class (data, code, type, module) and the lines that use it. Then the names it takes from `verity_vllm` "
          "outside `program/` (no side in this table) and from torch or vllm directly.", ""]
    for r in rows:
        x, y = r["imports_from_other_side"], r["imports_from_outside_table"]
        if not (x or y):
            continue
        L.append(f"### `{r['rel']}` ({r['side']})")
        L.append("")
        for e in x:
            where = (f"import line {e['import_line']}" if e["import_line"] else "this file's own part (an import once split)")
            L.append(f"- `{e['name']}` ({e['kind']}) from `{short(e['from'])}` [{e['from_side']}], {where}, "
                     f"used by {e['used_by_side']} at {', '.join(map(str, e['used_at'][:8]))}{' …' if len(e['used_at']) > 8 else ''}")
        for e in y:
            L.append(f"- outside the table: `{e['name']}` ({e['kind']}) from `{short(e['from'])}`, import line {e['import_line']}, "
                     f"used by {e['used_by_side']} at {', '.join(map(str, e['used_at'][:8]))}{' …' if len(e['used_at']) > 8 else ''}")
        L.append("")
    return "\n".join(L) + "\n"


def render_size(out: dict) -> str:
    rows, c, e = out["rows"], out["counts"], out["edited"]
    whole = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["side"] != "split":
            whole[r["side"]][0] += 1
            whole[r["side"]][1] += r["lines"]
    split = [r for r in rows if r["side"] == "split"]
    L = ["# The size of the split (sized, not done)", "",
         f"At `{out['commit']}`, over `{out['scope']}`: {c['files']} files, {c['lines']} lines. Generated by `table.py`; the prose "
         "below the tables is the author's.", "", "## Files whole to each side", "", "| side | files | lines |", "|---|---:|---:|"]
    for s in SIDE_ORDER:
        L.append(f"| {s} | {whole[s][0]} | {whole[s][1]} |")
    sl = c["lines_in_split_files_by_side"]
    L += ["", f"## Split files: {len(split)}, {sum(r['lines'] for r in split)} lines", "",
          "| file | lines | " + " | ".join(SIDE_ORDER) + " | header |", "|---|---:|" + "---:|" * (len(SIDE_ORDER) + 1)]
    for r in split:
        lb = r["lines_by_side"]
        L.append(f"| `{r['rel']}` | {r['lines']} | " + " | ".join(str(lb.get(s, 0)) for s in SIDE_ORDER) + f" | {lb.get('header', 0)} |")
    L.append("| total | " + str(sum(r["lines"] for r in split)) + " | " + " | ".join(str(sl.get(s, 0)) for s in SIDE_ORDER)
             + f" | {sl.get('header', 0)} |")
    tot = c["lines_by_side_all"]
    L += ["", "## Lines by side, whole files and split parts together", "", "| side | lines |", "|---|---:|"]
    for s in (*SIDE_ORDER, "header"):
        L.append(f"| {s} | {tot.get(s, 0)} |")
    L += ["", "`header` is a split file's lines outside its named parts (docstring, imports, small constants): each side's "
          "copy would need some of them.", "",
          "## Import lines that would be edited", "",
          f"- {e['cross_line_import_lines_in_circuits_parts']} import lines in {e['cross_line_import_files']} files bring a name "
          "from the other side into a circuits file or part: each must go or be replaced by what the list of kernel calls carries.",
          f"- {e['importer_import_lines_of_moving_files']} import lines in {e['importer_files_of_moving_files']} files of the "
          "repository (tests included) name a file that moves whole to a circuits directory, or a name a split file's circuits part "
          "binds: each changes path when the file moves.",
          f"- {e['module_object_import_lines_of_split_files']} import lines take a split file as a module object; which part "
          "they need is not resolved.",
          f"- Inside the split files, {e['split_parts_naming_the_other_side']} uses in a circuits part name a unit of the same "
          "file's other-side part: each becomes an import, or goes, once the file is split.", "",
          f"Edges between files or parts across the line, one way: {e['one_way_edges_circuits_to_other']} from circuits' side "
          f"to the other, {e['one_way_edges_other_to_circuits']} from the other side to circuits'.", "",
          "Per moving file: `importer_import_lines` in `files.json`.", "",
          "## Pairs that import each other across the line", "",
          "A pair is a file or split part on circuits' side and one on the other side, each using a name the other binds.", ""]
    if out["mutual_pairs"]:
        L += ["| circuits | other | circuits uses | other uses |", "|---|---|---|---|"]
        for p in out["mutual_pairs"]:
            L.append(f"| `{short(p['circuits']['file'])}` [{p['circuits']['side']}] | `{short(p['other']['file'])}` "
                     f"[{p['other']['side']}] | {', '.join(f'`{x}`' for x in p['circuits_uses'])} | "
                     f"{', '.join(f'`{x}`' for x in p['other_uses'])} |")
    else:
        L.append("None.")
    L += ["", "## What could not be placed", ""]
    for x in out["not_placed"]:
        L.append(f"- `{x['files']}`: {x['why']}.")
    L += ["", "Doubts on placed files (each row's `doubt` in `files.json`):", ""]
    for r in rows:
        if r["doubt"]:
            L.append(f"- `{r['rel']}` ({r['side']}): {r['doubt']}.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
