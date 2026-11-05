"""`unread_gate`: drive's landing gate on the seeds of the Lean nothing reads (`unread.py`'s ratchet). A change that takes the
last mention outside Lean of a module of the security analysis, or of a declaration of it, out of the files that keep it
fails here, before it lands, when that takes the module's area above its cap in `unread-caps.json`.

    python3 tools/verity/lean/unread_gate.py MAIN HEAD [--json]

Those seeds (code that names a declaration or a module, `unread.cited_kind`, and the documents a lock's text names,
`unread.lock_docs`) are in no audit's key, so a change to them alone reruns no Lean check and lands green, and the next check's
ratchet fails on it. The change judged is ``HEAD``'s against the commit below it (``LO``): the first parent of drive's merge of
a PR (a ``Queue: #N`` merge), else its merge base with ``MAIN``, as `markdown_caps.py` judges it.

- A change to what a Lean check reads (drive's `lean_reads`: Lean sources, Lake files, the toolchain, a lock,
  `tools/verity/lean/`) passes with a note: the Lean check runs the ratchet on it (`lean_audit.unread_ratchet`).
- Otherwise, from git alone: what each changed seed file names at ``LO`` and no longer at ``HEAD`` (`seeds`), less what any
  seed file still names at ``HEAD``. It passes when nothing went that could name the analysis's Lean: a declaration whose last
  component no Lean file of `Security/` declares (`declared`; an instance's name aside, which Lean may make up), a module or
  a path no file there has.
- Otherwise it runs the ratchet on both trees, ``LO``'s and ``HEAD``'s, from the facts the audits of `Security` and
  `Security/Proofs` wrote (`lean_audit.facts_file`, by each package's key): ``LO``'s, else ``MAIN``'s, as this machine keeps them
  (a node's passes, or this gate's cache) or as one of the `SCAN` newest recorded checks published them in the evidence
  store. Each area whose wholly unread modules, or their declarations, ``HEAD`` takes above both its cap and ``LO``'s count
  fails, naming each module that went, the mentions that kept it (each at its file and line at ``LO``) and the fix.

It runs ``MAIN``'s `unread.py`, `lean_audit.py` and their helpers (`git archive`, `CODE`), so the change it judges can't change
how it is judged, and it builds nothing. Before marking a branch, run ``python3 tools/verity/lean/unread_gate.py origin/main
HEAD``. Exit 0 when it passes, 1 when it fails (``--json``: `{"failures": [{"why", "fix"}]}`, `KIND`), 2 when it can't judge: a
git error, `MAIN`'s code doesn't load, or neither tree's facts are kept here or stored.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

KIND = "lean-unread-gate/v1"
#: the subject of the queue's merge of one PR onto `next`
QUEUE_SUBJECT = "Queue: #"
LEAN_TOOL = "tools/verity/lean/"
#: what a Lean check reads (drive's `lean_reads`): a change to any of it is the Lean check's to judge
LEAN_READS = (".lean", "lakefile.toml", "lakefile.lean", "lean-toolchain", "lake-manifest.json", "lean-audit.json")
#: ``MAIN``'s code the gate runs: the Lean tools (not their tests) and what `lean_audit.py` loads
CODE = (f":(glob){LEAN_TOOL}*", "tools/verity/check/common.py", "tools/verity/graph/verity_graph/contract.py")
#: the analysis's Lean, all of whose packages (`unread.ANALYSIS`) are under it
ANALYSIS_DIR = "Security/"
CAPS = f"{LEAN_TOOL}unread-caps.json"
#: how many of the newest recorded checks the store lookup lists the files of
SCAN = 64
#: how many facts files this gate keeps in its cache, the last used first
KEPT = 16
#: a recorded check's facts of the package with key K: `unread/K.jsonl.gz` under its lean-audit output
STORED = re.compile(r"(?:.*/)?lean-audit/unread/([0-9a-f]{64})\.jsonl\.gz")
#: a name component's start that is a word (its `'`, `!` or `?` aside)
WORD = re.compile(r"[A-Za-z0-9_]+")
#: a declaring keyword and the name it declares, whatever modifiers and attributes precede it on its line, or a namespace
#: (`git grep -E`: one pattern, since two `-e` take it ten times as long)
DECLARED = (r"\b(theorem|lemma|def|abbrev|structure|class|inductive|instance|opaque|axiom|alias|irreducible_def|rec|namespace)"
            r"[[:space:]]+[^[:space:]({:⟨]+")
#: the line that opens a definition's helpers (`where`), each first on one of the lines after it
WHERE = r"^[[:space:]]*where[[:space:]]*($|--)"
#: the statuses of a wholly unread module (`unread.analyse`)
GONE = ("unread", "unread aggregator")


class GitError(RuntimeError):
    pass


class CantJudge(RuntimeError):
    pass


def git(*args: str, input: bytes | None = None, ok: tuple[int, ...] = (0,)) -> bytes:
    r = subprocess.run(["git", *args], capture_output=True, input=input)
    if r.returncode not in ok:
        raise GitError(f"git {' '.join(args[:4])}: {r.stderr.decode(errors='replace').strip()[-300:]}")
    return r.stdout


def below(base: str, top: str) -> str:
    """The commit ``top``'s change is judged against: the tip it merged onto when it is the queue's merge of a PR, else its
    merge base with ``base``."""
    parents, _, subject = git("log", "-1", "--format=%P%x00%s", top).decode().strip().partition("\0")
    if len(parents.split()) == 2 and subject.startswith(QUEUE_SUBJECT):
        return parents.split()[0]
    return git("merge-base", base, top).decode().strip()


def tracked(rev: str, *paths: str) -> list[str]:
    return [p for p in git("ls-tree", "-r", "-z", "--name-only", rev, "--", *paths).decode().split("\0") if p]


def blobs(rev: str, paths: list[str]) -> dict[str, bytes | None]:
    """Each of ``paths``' bytes at ``rev``, None where it has no such file."""
    paths = [p for p in paths if "\n" not in p]
    out, got, at = git("cat-file", "--batch", input="".join(f"{rev}:{p}\n" for p in paths).encode()), {}, 0
    for p in paths:
        nl = out.index(b"\n", at)
        head, at = out[at:nl], nl + 1
        if head.endswith(b" missing") or head.endswith(b" ambiguous"):
            got[p] = None
            continue
        _, kind, size = head.rsplit(b" ", 2)
        got[p] = out[at:at + int(size)] if kind == b"blob" else None
        at += int(size) + 1
    return got


def archive(rev: str, dest: Path, *paths: str) -> Path:
    """``rev``'s tree (its ``paths``, as git pathspecs, when given) written under ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(["git", "archive", "--format=tar", rev, *(["--", *paths] if paths else [])], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    with tarfile.open(fileobj=p.stdout, mode="r|") as t:
        t.extractall(dest, filter="data")
    if p.wait():
        raise GitError(f"git archive {rev[:12]}: {p.stderr.read().decode(errors='replace').strip()[-300:]}")
    return dest


def overlay(tree: Path, rev: str, paths: list[str]) -> None:
    """``tree`` with ``paths`` as they are at ``rev``: written, or removed where ``rev`` has none."""
    got = blobs(rev, paths)
    for p in sorted(got, key=lambda p: got[p] is not None):
        if got[p] is None:
            (tree / p).unlink(missing_ok=True)
        else:
            (tree / p).parent.mkdir(parents=True, exist_ok=True)
            (tree / p).write_bytes(got[p])


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def seeds(U, path: str, data: bytes | None, docs: set[str]) -> set[tuple[str, str]]:
    """What the file at ``path`` keeps by naming it, as `unread.citations` and `unread.mentioned` read it, each (role, name):
    "decl", a dotted name, which keeps the declaration of that name; "module", one that keeps the module of that name; "path",
    a Lean file's path, which keeps the modules it names (`unread.path_namer`). A document keeps only where a lock's text names
    it (``docs``)."""
    kind = U.cited_kind(path)
    if data is None or kind is None or (kind == "doc" and path not in docs) or len(data) > U.CITE_MAX_BYTES:
        return set()
    text = data.decode(errors="replace")
    dotted = U.names(text)
    out = {("decl", t) for t in dotted}
    if kind != "code":
        out |= {("module", t) for t in dotted}
        if ".lean" in text:
            out |= {("path", p) for p in U.lean_paths(text)}
    return out


class Docs:
    """The documents a lock's text names (`unread.lock_docs`) at each revision asked, read once: the locks are the same at
    both, since a change to one is the Lean check's, but the tracked documents a bare name matches may not be."""

    def __init__(self, U, lo: str):
        self.U, self.lo, self.texts, self.at = U, lo, None, {}

    def __call__(self, rev: str) -> set[str]:
        if rev not in self.at:
            files = tracked(rev)
            if self.texts is None:
                locks = [f"{p}/lean-audit.json" for p in self.U.MODULES.packages(files) if f"{p}/lean-audit.json" in set(files)]
                self.texts = self.U.lock_seeds({f.rsplit("/", 1)[0]: json.loads(b)
                                                for f, b in blobs(self.lo, locks).items() if b is not None})[2]
            self.at[rev] = set(self.U.lock_docs(files, self.texts))
        return self.at[rev]


def line_of(data: bytes | None, name: str) -> int | None:
    """The line ``name`` (a path: its file's name) first appears on in ``data``."""
    i = -1 if data is None else data.find(name.encode())
    if i < 0 and data is not None and "/" in name:
        i = data.find(name.rsplit("/", 1)[-1].encode())
    return None if i < 0 else data.count(b"\n", 0, i) + 1


def removed(U, lo: str, top: str, changed: list[str], seeded: list[str], docs: Docs) -> dict[tuple[str, str], list[str]]:
    """{(role, name): the files that name it at ``lo``} for each name a file names at ``lo`` and no longer at ``top``
    (`seeds`): one of ``seeded`` (the changed seed files), or a document a lock's text names at ``lo`` only, which a
    document added or deleted can do (a lock's bare `PROTOCOL.md` names every one until a path from the root has one)."""
    docs_lo, docs_top = (docs(lo), docs(top)) if any(p.endswith(U.DOCS) for p in changed) else (set(), set())
    files = sorted({*seeded, *(docs_lo - docs_top)})
    was, now = blobs(lo, files), blobs(top, files)
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in files:
        for s in seeds(U, p, was[p], docs_lo) - seeds(U, p, now[p], docs_top):
            out[s].append(p)
    return dict(out)


def unnamed(U, top: str, gone: dict[tuple[str, str], list[str]], docs: Docs, work: Path) -> dict[tuple[str, str], list[str]]:
    """``gone`` less what a seed file still names at ``top`` (another file, or a changed one it moved to): the files git finds
    holding a name (a path: its file's name), each read as `seeds` reads it."""
    if not gone:
        return {}
    words = sorted({n.rsplit("/", 1)[-1] if r == "path" else n for r, n in gone})
    (work / "words").write_text("\n".join(words) + "\n")
    hits = [p.split(":", 1)[1] for p in git("grep", "-l", "-z", "-F", "-f", str(work / "words"), top, "--",
                                            *(f"*{s}" for s in U.CITED_BY + U.DOCS), ok=(0, 1)).decode().split("\0") if p]
    hits = [p for p in hits if U.cited_kind(p) is not None]
    in_docs = docs(top) if any(U.cited_kind(p) == "doc" for p in hits) else set()
    left = dict(gone)
    for p, b in blobs(top, hits).items():
        for s in seeds(U, p, b, in_docs) & left.keys():
            del left[s]
    return left


def declared(lo: str) -> tuple[set[str], set[str]]:
    """What the names the Lean under `ANALYSIS_DIR` declares at ``lo`` can start and end with (each a `WORD`): the first
    component of a `namespace` or of a name a keyword of `DECLARED` declares, and a private name's `_private`; the last
    component of such a name, or of a helper first on a line of a definition's `WHERE` block."""
    heads, lasts = {"_private"}, set()
    for line in git("grep", "-h", "-o", "-E", DECLARED, lo, "--", f"{ANALYSIS_DIR}*.lean", ok=(0, 1)).decode().splitlines():
        word, name = (line.split(None, 1) + [""])[:2]
        parts = name.strip().removeprefix("_root_.").split(".")
        if (m := WORD.match(parts[0])):
            heads.add(m.group(0))
        if word != "namespace" and (m := WORD.match(parts[-1])):
            lasts.add(m.group(0))
    for line in git("grep", "-h", "-A", "3", "-E", WHERE, lo, "--", f"{ANALYSIS_DIR}*.lean", ok=(0, 1)).decode().splitlines():
        if (m := re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*)", line)):
            lasts.add(m.group(1))
    return heads, lasts


def plausible(lo: str, gone: dict[tuple[str, str], list[str]]) -> dict[tuple[str, str], list[str]]:
    """The names of ``gone`` that could name the analysis's Lean at ``lo``: a declaration's full name (all a file outside
    Lean keeps it by) whose first component a namespace or a declaration there starts with and whose last component a Lean
    file under `ANALYSIS_DIR` declares, or is an instance's (`inst...`, which Lean may make up) (`declared`); a module or a
    path a Lean file there ends with. A constant Lean makes up otherwise (a constructor, a matcher, an equation lemma) has
    its owner's range or none, so it is no declaration a file outside Lean keeps (`unread.owners`)."""
    if not gone:
        return {}
    lean = [f for f in tracked(lo, ANALYSIS_DIR) if f.endswith(".lean")]
    tails = {"/".join(f.split("/")[k:]) for f in lean for k in range(f.count("/") + 1)}
    ends = {n: (WORD.match(n.split(".", 1)[0]), WORD.match(n.rsplit(".", 1)[-1])) for r, n in gone if r == "decl"}
    heads, lasts = declared(lo) if ends else (set(), set())

    def could(role: str, name: str) -> bool:
        if role == "decl":
            head, last = ends[name]
            return (head is None or head.group(0) in heads) and (
                last is None or last.group(0).startswith("inst") or last.group(0) in lasts)
        if role == "module":
            return name in tails or name.replace(".", "/") + ".lean" in tails
        parts = [x for x in name.split("/") if x not in ("", ".", "..")]
        return any("/".join(parts[k:]) in tails for k in range(len(parts)))

    return {s: fs for s, fs in gone.items() if could(*s)}


def stored(keys: dict[str, str], cache: Path, recorded: str, store=None, scan: int = SCAN) -> tuple[dict[str, Path], list[str]]:
    """({package: its facts file in ``cache``} for each of ``keys`` ({package: key}) that one of the ``scan`` newest recorded
    checks (``recorded``, `common.RECORDED_CHECKS`) published (`STORED`), checked against the sha256 its listing gives; what
    kept the store from being read, if anything)."""
    want = {k: p for p, k in keys.items()}
    got: dict[str, Path] = {}
    errors: list[str] = []
    try:
        if store is None:
            from research.store.local import LocalStore
            store = LocalStore()
        if store.remote is not None:
            store.refresh(attempts_only=True)
        rows = store.index.sql(f"SELECT id FROM attempts WHERE {recorded} AND state IN ('done', 'failed', 'cancelled') "
                               "ORDER BY start_utc DESC LIMIT ?", (scan,))
    except Exception as e:  # noqa: BLE001  (no store here: the gate says it can't judge, naming why)
        return got, [f"the evidence store can't be read here ({type(e).__name__}: {e})"]
    for (rid,) in rows:
        try:
            art = ((store.get_attempt(rid) or {}).get("outputs") or {}).get("run_files")
            for f in store.listing(art) if art else []:
                m = STORED.fullmatch(f["path"])
                if not m or want.get(m.group(1)) in (None, *got) or not (f.get("remote") or f.get("local")):
                    continue
                with tempfile.TemporaryDirectory(prefix="unread-gate-", dir=cache) as d:
                    data = (Path(store.fetch(art, Path(d), paths=[f["path"]])) / f["path"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != f["sha256"]:
                    errors.append(f"{rid}: {f['path']} isn't the bytes its listing names")
                    continue
                dst = cache / f"{m.group(1)}.jsonl.gz"
                tmp = dst.with_name(f".{dst.name}.{os.getpid()}")
                tmp.write_bytes(data)
                os.replace(tmp, dst)
                got[want[m.group(1)]] = dst
        except Exception as e:  # noqa: BLE001  (one run's files unreadable: the next run's may hold the facts)
            errors.append(f"{rid}: {type(e).__name__}: {e}")
        if len(got) == len(keys):
            break
    return got, errors


def facts_of(LA, tree: Path, files: list[str], cache: Path, work: Path) -> tuple[dict[str, str], dict[str, Path], list[str]]:
    """The keys of the packages the ratchet reads the facts of (`lean_audit.UNREAD_FACTS`) on ``tree``, and the facts of each
    that this gate's cache, this machine's Lean check (a node's kept passes) or the store holds, and why the store gave none."""
    LA.ROOT = tree
    keys = {p: LA.key(p, files, LA.FLAGS) for p in LA.UNREAD_FACTS}
    got: dict[str, Path] = {}
    for p, k in keys.items():
        if (own := cache / f"{k}.jsonl.gz").is_file():
            os.utime(own)
            got[p] = own
        elif (f := LA.kept_facts(LA.COMMON.cache_dir() / "lean-audit", p, k, work, log=lambda s: None)) is not None:
            got[p] = f
    errors: list[str] = []
    if missing := {p: k for p, k in keys.items() if p not in got}:
        more, errors = stored(missing, cache, LA.COMMON.RECORDED_CHECKS)
        got |= more
    return keys, got, errors


def judge(U, LA, base: str, lo: str, top: str, gone: dict[tuple[str, str], list[str]], work: Path, cache: Path) -> dict:
    """The ratchet on ``lo``'s and ``top``'s trees from the facts of ``lo``'s Lean, else of ``base``'s: {"failures", "notes",
    "facts"}."""
    files_lo, files_top = tracked(lo), tracked(top)
    tree = archive(lo, work / "tree")
    keys, facts, errors = facts_of(LA, tree, files_lo, cache, work)
    on_main, notes = False, []
    if len(facts) < len(keys):
        keys_base, facts_base, errors_base = (keys, facts, []) if base == lo else \
            facts_of(LA, archive(base, work / "base"), tracked(base), cache, work)
        if len(facts_base) < len(keys_base):
            listed = "; ".join(f"{r[:12]}: " + ", ".join(f"{p} {k[:12]}" for p, k in ks.items())
                               for r, ks in {lo: keys, base: keys_base}.items())
            raise CantJudge(f"the facts of the Lean of {lo[:12]} and of `main` {base[:12]} aren't all kept here or published "
                            f"by the {SCAN} newest recorded checks ({listed})"
                            + (f": {'; '.join((errors + errors_base)[:3])}" if errors + errors_base else ""))
        keys, facts, on_main = keys_base, facts_base, True
        notes.append(f"judged on the facts of `main` {base[:12]}'s Lean: not all of {lo[:12]}'s are kept or stored")
    F = U.Facts(*(facts[p] for p in LA.UNREAD_FACTS))
    t_lo = U.Tree(tree, files_lo)
    rep_lo = U.analyse(t_lo, F, full=False)
    named = U.module_names(t_lo)
    namer = U.path_namer(named)
    changed = sorted({f for fs in gone.values() for f in fs})
    was = blobs(lo, changed)
    overlay(tree, top, [p for p in git("diff", "--no-renames", "--name-only", "-z", lo, top).decode().split("\0") if p])
    rep_top = U.analyse(U.Tree(tree, files_top), F, full=False)
    caps = U.load_caps(tree / CAPS) if (tree / CAPS).is_file() else {}
    kept_by: dict[str, list[str]] = defaultdict(list)
    for (role, name), fs in sorted(gone.items()):
        if role == "decl":
            mods = {F.modules[F.module[i]]["module"]} if (i := F.index.get(name)) is not None else set()
        else:
            mods = ({named[name]} if name in named else set()) if role == "module" else namer(name)
        for m in mods:
            kept_by[m] += [f"`{name}` at {f}:{line_of(was.get(f), name) or '?'}" for f in fs]
    now, before, zero = U.counts(rep_top), U.counts(rep_lo), dict.fromkeys(U.CAP_KEYS, 0)
    failures = []
    for area in sorted(now):
        got, cap, had = now[area], zero | caps.get(area, {}), before.get(area, zero)
        new = sorted((m for m, r in rep_top["modules"].items() if r["area"] == area and r["status"] in GONE
                      and rep_lo["modules"].get(m, {}).get("status") not in GONE), key=lambda m: (not kept_by.get(m), m))
        if not any(got[k] > cap[k] and got[k] > had[k] for k in U.CAP_KEYS):
            notes += [f"{m} is wholly unread at {top[:12]}, within {area}'s cap" for m in new]
            continue
        each = [f"{m} ({rep_top['modules'][m]['unread']} declarations, {rep_top['modules'][m]['file']}) "
                + (f"loses the last mention outside Lean that kept it: {', '.join(kept_by[m][:4])}" if kept_by.get(m)
                   else "was kept only through what went") for m in new[:6]]
        failures.append({
            "area": area, "modules": new,
            "why": f"{area} would have {got['modules']} wholly unread modules ({got['declarations']} declarations), above its cap "
                   f"of {cap['modules']} ({cap['declarations']} declarations) in {CAPS} and up from {had['modules']} "
                   f"({had['declarations']}) at {lo[:12]}: " + "; ".join(each) + (f"; and {len(new) - 6} more" if len(new) > 6 else "")
                   + (f" (on the facts of `main` {base[:12]}'s Lean)" if on_main else ""),
            "fix": [f"delete {' and '.join(new[:3])}{' ...' if len(new) > 3 else ''} (its file, and its import where another "
                    "module imports it): nothing reads it",
                    "or seed it again: keep a mention of it outside Lean in code that reads it (its dotted name or its file's "
                    f"path in a {', '.join(U.MODULE_CITED_BY)} file, or the full name of a declaration of it in a "
                    f"{', '.join(U.CITED_BY)} file), or name it in a lock's text"]})
    return {"failures": failures, "notes": notes, "facts": {p: k for p, k in keys.items()}}


def gate(base: str, top: str) -> dict:
    base, top = (git("rev-parse", "--verify", f"{r}^{{commit}}").decode().strip() for r in (base, top))
    lo = below(base, top)
    doc = {"kind": KIND, "base": base, "top": top, "below": lo, "stage": "lean", "seed_files": [], "went": [],
           "failures": [], "notes": []}
    changed = [p for p in git("diff", "--no-renames", "--name-only", "-z", lo, top).decode().split("\0") if p]
    if lean := [p for p in changed if p.endswith(LEAN_READS) or p.startswith(LEAN_TOOL)]:
        doc["notes"].append(f"it changes what a Lean check reads ({', '.join(lean[:3])}{' ...' if len(lean) > 3 else ''}): "
                            "the Lean check's ratchet judges it")
        return doc
    with tempfile.TemporaryDirectory(prefix="unread-gate-") as d:
        code = archive(base, Path(d) / "code", *CODE)
        U = load(code / LEAN_TOOL / "unread.py", "unread_for_unread_gate")
        doc["stage"], doc["seed_files"] = "git", (seeded := [p for p in changed if U.cited_kind(p) is not None])
        docs = Docs(U, lo)
        gone = removed(U, lo, top, changed, seeded, docs)
        could = unnamed(U, top, plausible(lo, gone), docs, Path(d))
        doc["went"] = sorted(f"{r}:{n}" for r, n in could)[:40]
        if not could:
            doc["notes"].append(f"{len(seeded)} seed file(s) changed, and of the {len(gone)} name(s) they no longer hold, "
                                "none that the analysis's Lean could hold is gone from every seed file")
            return doc
        LA = load(code / LEAN_TOOL / "lean_audit.py", "lean_audit_for_unread_gate")
        cache = LA.COMMON.cache_dir() / "unread-gate"
        cache.mkdir(parents=True, exist_ok=True)
        for old in cache.glob("work-*"):
            if not Path("/proc", old.name.split("-")[1]).exists():
                shutil.rmtree(old, ignore_errors=True)
        work = Path(tempfile.mkdtemp(prefix=f"work-{os.getpid()}-", dir=cache))
        try:
            doc |= {"stage": "facts", **judge(U, LA, base, lo, top, could, work, cache)}
        finally:
            shutil.rmtree(work, ignore_errors=True)
            for old in sorted(cache.glob("*.jsonl.gz"), key=lambda f: f.stat().st_mtime, reverse=True)[KEPT:]:
                old.unlink(missing_ok=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="unread_gate", description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("top")
    ap.add_argument("--json", action="store_true", help=f"print {KIND}")
    a = ap.parse_args(argv)
    try:
        doc = gate(a.base, a.top)
    except (GitError, CantJudge, OSError, ValueError, LookupError, TypeError, AttributeError, ImportError, SyntaxError,
            tarfile.TarError) as e:
        print(f"unread gate: can't judge: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(doc, indent=1))
    else:
        print(f"unread gate {doc['below'][:12]}..{doc['top'][:12]} (base {doc['base'][:12]}): {doc['stage']}, "
              f"{len(doc['seed_files'])} seed file(s) changed, {len(doc['failures'])} area(s) above the cap")
        print("".join(f"  {f['why']}\n" + "".join(f"    fix: {x}\n" for x in f["fix"]) for f in doc["failures"])
              + "".join(f"  note: {n}\n" for n in doc["notes"]), end="")
    return 1 if doc["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
