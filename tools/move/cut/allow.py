"""Record in the counted lints' lists what a cut moved from one file to another.

    python allow.py ROOT BASE [--table TABLE.json] [--dry]

Run after ``cut.py`` on the same tree, with BASE the commit the cut was applied on and the interpreter the one that has
the tree's pinned ruff.  The ratchet gate lets a file's count rise, or a new file's appear, only when the same list's other
files fall by at least as much in the same change, so a definition that moves with a finding (a ruff rule's, a direct
environment read, a silent except handler, lane text, a retired term) needs its count moved with it.  For each list
under ``tests/lint/allowlists/`` this counts, with the tree's own lint, every file the working tree changes against
BASE, before and after, and adds each file's difference to its entry; a file that moved whole takes its entries with
it first, as the lint's ``--move`` does.

It writes nothing and exits 1 when the files hold more of a lint's sites after the cut than before, beyond what the
table's ``own`` allows that lint with its reason (a split writes an import line of the same names into each new
module): the cut then added a finding of its own, and the lines it prints name the files that gained more than the
others lost.  Nothing here reads history beyond BASE's files.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

LINTS = ("repo_ruff", "repo_env_reads", "repo_silent_excepts", "repo_lane_text", "repo_retired_terms")


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout.decode("utf-8", "replace")


def lint(root: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"allow_{name}", root / "tests" / "lint" / f"test_{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def changes(root: Path, base: str) -> tuple[dict[str, str | None], dict[str, str]]:
    """(each changed file's path at BASE, None for a new one; old -> new of the files that moved whole)."""
    was: dict[str, str | None] = {}
    renames: dict[str, str] = {}
    fields = git(root, "diff", "--name-status", "-M", "-z", base).split("\0")
    i = 0
    while i < len(fields) and fields[i]:
        status = fields[i]
        if status.startswith("R"):
            old, new = fields[i + 1], fields[i + 2]
            was[new], renames[old] = old, new
            i += 3
        else:
            path = fields[i + 1]
            if status.startswith("D"):
                was.setdefault(path, path)
            else:
                was[path] = None if status.startswith("A") else path
            i += 2
    for path in git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0"):
        if path:
            was[path] = None
    return was, renames


def ruff_counts(root: Path, module: ModuleType, path: str, text: str | None) -> Counter:
    if text is None or not path.endswith((".py", ".pyi")) or path.startswith(module.LEFT_OUT):
        return Counter()
    cmd = [sys.executable, "-m", "ruff", "check", "--config", str(root / "ruff.toml"), "--no-cache", "--exit-zero",
           "--output-format", "json", "--stdin-filename", str(root / path), "-"]
    r = subprocess.run(cmd, cwd=root, input=text, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"allow: ruff exited {r.returncode} on {path}: run this with the interpreter that has the tree's pinned "
                 f"ruff (`uv run python allow.py ...`)\n{r.stderr.strip()[-400:]}")
    return Counter((path, f["code"] or "syntax-error") for f in json.loads(r.stdout))


def counts(root: Path, name: str, module: ModuleType, path: str, text: str | None) -> Counter:
    if name == "repo_ruff":
        return ruff_counts(root, module, path, text)
    if text is None:
        return Counter()
    if name in ("repo_env_reads", "repo_silent_excepts"):
        if not path.endswith((".py", ".pyi")) or path.startswith(module.LEFT_OUT):
            return Counter()
        n = len(module.reads(text) if name == "repo_env_reads" else module.handlers(text))
    else:
        n = len(module.sites(text, path))
    return Counter({path: n}) if n else Counter()


ROW = re.compile(r"^(\s*)(\{.*\})\s*,?\s*$")


def edit_list(path: Path, renames: dict[str, str], delta: Counter) -> Counter:
    """Write ``path`` with its moved files' rows renamed and ``delta`` added, every other row as it stands and where it
    stands (the lint's own ``dump`` sorts the list, which would rewrite the rows of every change in flight); the counts
    it holds after, by key."""
    lines = path.read_text(encoding="utf-8").split("\n")
    at = [i for i, line in enumerate(lines) if ROW.match(line) and "file" in line]
    first, last = at[0], at[-1]
    assert at == list(range(first, last + 1)), f"{path.name}: its entries are not one run of rows"
    rows: list[tuple[dict, str | None]] = []  # each entry, and its text while nothing changed it
    for i in at:
        indent, body = ROW.match(lines[i]).groups()
        rows.append((json.loads(body), indent + body))

    def key(e: dict) -> object:
        return (e["file"], e["rule"]) if "rule" in e else e["file"]

    def moved(f: str) -> str:
        for old, new in renames.items():
            if f == old or f.startswith(f"{old}/"):
                return new + f[len(old):]
        return f

    for n, (e, _text) in enumerate(rows):
        if moved(e["file"]) != e["file"]:
            e["file"] = moved(e["file"])
            rows[n] = (e, None)
    seen: dict[object, int] = {}
    for n, (e, _text) in enumerate(list(rows)):  # a moved row that meets one already there adds to it
        if "count" not in e:
            continue
        if key(e) in seen:
            rows[seen[key(e)]][0]["count"] += e["count"]
            rows[seen[key(e)]] = (rows[seen[key(e)]][0], None)
            rows[n] = (e, "")
        else:
            seen[key(e)] = n
    rows = [r for r in rows if r[1] != ""]
    for k, n in sorted(delta.items(), key=str):
        where = next((i for i, (e, _t) in enumerate(rows) if "count" in e and key(e) == k), None)
        if where is not None:
            e = rows[where][0]
            e["count"] = max(0, e["count"] + n)
            rows[where] = (e, None)
        elif n > 0:
            new = dict(zip(("file", "rule"), k)) if isinstance(k, tuple) else {"file": k}
            new["count"] = n
            folder = new["file"].rsplit("/", 1)[0]
            beside = [i for i, (e, _t) in enumerate(rows) if e["file"].rsplit("/", 1)[0] == folder and e["file"] <= new["file"]]
            beside = beside or [i for i, (e, _t) in enumerate(rows) if e["file"].rsplit("/", 1)[0] == folder][:1]
            rows.insert(beside[-1] + 1 if beside and rows[beside[-1]][0]["file"] <= new["file"] else
                        (beside[0] if beside else len(rows)), (new, None))
    rows = [(e, t) for e, t in rows if e.get("count", 1) > 0]
    indent = ROW.match(lines[first]).group(1)
    body = [t if t is not None else indent + json.dumps(e, ensure_ascii=False) for e, t in rows]
    lines[first:last + 1] = [b + "," for b in body[:-1]] + body[-1:]
    path.write_text("\n".join(lines), encoding="utf-8")
    return Counter({key(e): e["count"] for e, _t in rows if "count" in e})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("base")
    ap.add_argument("--table", type=Path, help="the cut's table: its `own` says what the cut may add of its own")
    ap.add_argument("--dry", action="store_true", help="print what would be written and write nothing")
    a = ap.parse_args()
    own = {k: v["count"] for k, v in (json.loads(a.table.read_text()).get("own", {}) if a.table else {}).items()}
    root = a.root.resolve()
    was, renames = changes(root, a.base)
    before_text = {p: (git(root, "show", f"{a.base}:{old}") if old else None) for p, old in was.items()}
    after_text = {p: ((root / p).read_text(encoding="utf-8", errors="replace") if (root / p).is_file() else None)
                  for p in was}
    out, grown, writes = {}, [], []
    for name in LINTS:
        module = lint(root, name)
        have = module.allowed()
        marks = None
        if isinstance(have, tuple):
            have, marks = have
        have = Counter(have)
        for old, new in renames.items():
            if marks is not None:
                have, marks = module.moved(have, marks, old, new)
            else:
                have = module.moved(have, old, new)
        delta: Counter = Counter()
        for path in sorted(was):
            if marks is not None and module.is_registry(path, marks):
                continue
            before = counts(root, name, module, path, before_text[path])
            if was[path] and was[path] != path:  # a file that moved whole: its count at BASE, under its new path
                before = Counter({(path, *k[1:]) if isinstance(k, tuple) else path: n for k, n in before.items()})
            after = counts(root, name, module, path, after_text[path])
            for key in set(before) | set(after):
                if after[key] != before[key]:
                    delta[key] = after[key] - before[key]
        total = sum(delta.values())
        rows = {str(k): n for k, n in sorted(delta.items(), key=str)}
        out[name] = {"total": total, "own": own.get(name, 0), "files": rows}
        if total > own.get(name, 0):
            grown.append(f"{name}: the cut adds {total} of its own, and the table's `own` allows {own.get(name, 0)} "
                         f"({', '.join(f'{k} {n:+d}' for k, n in rows.items())}): mend them in the script's table "
                         f"(a module's `doc`, an `edits` row) or in the code before the cut, or say in `own` why the "
                         f"cut must add them; never in the list alone")
            continue
        want = Counter(have)
        for key, n in delta.items():
            want[key] = max(0, want[key] + n)
        writes.append((module, delta, +want))
    print(json.dumps(out, indent=1))
    if grown:
        print("\n".join(grown), file=sys.stderr)
        return 1
    if not a.dry:
        for module, delta, want in writes:
            if not delta and not renames:
                continue
            kept = module.LIST.read_text(encoding="utf-8")
            if edit_list(module.LIST, renames, delta) != want:
                module.LIST.write_text(kept, encoding="utf-8")
                sys.exit(f"allow: {module.LIST.name} would not hold what its lint's own `--move` and the counts give: "
                         f"left as it was; read its rows for the files the cut changes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
