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
  component no Lean file of `security/` declares (`declared`; an instance's name aside, which Lean may make up), a module or
  a path no file there has.
- Otherwise it runs the ratchet on both trees, ``LO``'s and ``HEAD``'s, from the facts the audits of `security` and
  `security/Proofs` wrote (`lean_audit.facts_file`, by each package's key): ``LO``'s, else ``MAIN``'s, as this machine keeps them
  (a node's passes, or this gate's cache) or as a recorded check published them in the evidence store: one of that commit
  (drive runs a commit's Lean check before it lands it), else one of the `SCAN` newest. When neither's are (a Lean landing's are
  published once its lock run lands, about an hour after it), it reads stale ones: those of the newest commit below ``LO``
  along first parents, within `ANCESTORS`, that a recorded check ran at and whose facts are kept or published
  (`ancestor_facts`). The verdict says so (``--json``: `facts_from`, `stale`) and isn't kept, and the Lean check's ratchet
  after landing judges what such facts miss. Each area whose wholly unread modules, or their declarations, ``HEAD`` takes above both its cap and ``LO``'s count
  fails, naming each module that went, the mentions that kept it (each at its file and line at ``LO``) and the fix.

It runs ``MAIN``'s `unread.py`, `lean_audit.py` and their helpers (`git archive`, `CODE`), so the change it judges can't change
how it is judged, and it builds nothing; it runs git under `LOCALE`, so the host's locale can't either. Drive judges every PR
of `next` again whenever `main` moves, so the gate keeps each verdict by the trees it judged and the code that judged it
(`gate`), and what the Lean declares by the Lean's tree (`declared`). Before marking a branch, run ``python3 tools/verity/lean/unread_gate.py origin/main
HEAD``. Exit 0 when it passes, 1 when it fails (``--json``: `{"failures": [{"why", "fix"}]}`, `KIND`), 2 when it can't judge: a
git error, `MAIN`'s code doesn't load, or neither tree's facts nor an ancestor's are kept here or stored.
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
COMMON = "tools/verity/check/common.py"
CODE = (f":(glob){LEAN_TOOL}*", COMMON, "tools/verity/graph/verity_graph/contract.py")
#: what a kept verdict's key holds of ``MAIN``: `CODE`'s trees (the Lean tools with their tests: a superset)
CODE_TREES = (LEAN_TOOL.rstrip("/"), *CODE[1:])
#: the analysis's Lean, all of whose packages (`unread.ANALYSIS`) are under it
ANALYSIS_DIR = "security/"
CAPS = f"{LEAN_TOOL}unread-caps.json"
#: how many of the newest recorded checks the store lookup lists the files of
SCAN = 64
#: how far below ``LO`` along first parents the gate looks for a commit whose Lean's facts it holds when it holds neither
#: ``LO``'s nor ``MAIN``'s (a Lean landing's are published once its lock run lands, about an hour after it), and how many of
#: the commits there a recorded check ran at it tries, the newest first: each costs an archive of the tree (about a second)
ANCESTORS, ANCESTOR_CHECKS = 300, 8
#: a recorded check that ended
ENDED = "state IN ('done', 'failed', 'cancelled')"
#: how many facts files this gate keeps in its cache, the last used first; how many `declared` sets; how many verdicts (drive
#: judges each PR of `next` again on every landing)
KEPT, DECLARED_KEPT, VERDICTS_KEPT = 16, 16, 1024
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
#: the locale git runs under: its patterns match characters, whatever the host's default (`C`'s match bytes, and cut a name
#: inside a character)
LOCALE = "C.UTF-8"


class GitError(RuntimeError):
    pass


class CantJudge(RuntimeError):
    pass


def git_env() -> dict[str, str]:
    return os.environ.copy() | {"LC_ALL": LOCALE}


def git(*args: str, input: bytes | None = None, ok: tuple[int, ...] = (0,)) -> bytes:
    r = subprocess.run(["git", *args], capture_output=True, input=input, env=git_env())
    if r.returncode not in ok:
        raise GitError(f"git {' '.join(args[:4])}: {r.stderr.decode(errors='replace').strip()[-300:]}")
    return r.stdout


def git_text(*args: str, ok: tuple[int, ...] = (0,)) -> str:
    """git's output as text, a byte that is no UTF-8 replaced (a host without `LOCALE` gives git `C`'s)."""
    return git(*args, ok=ok).decode(errors="replace")


def below(base: str, top: str) -> str:
    """The commit ``top``'s change is judged against: the tip it merged onto when it is the queue's merge of a PR, else its
    merge base with ``base``."""
    parents, _, subject = git_text("log", "-1", "--format=%P%x00%s", top).strip().partition("\0")
    if len(parents.split()) == 2 and subject.startswith(QUEUE_SUBJECT):
        return parents.split()[0]
    return git_text("merge-base", base, top).strip()


def tracked(rev: str, *paths: str) -> list[str]:
    return [p for p in git_text("ls-tree", "-r", "-z", "--name-only", rev, "--", *paths).split("\0") if p]


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
                         stderr=subprocess.PIPE, env=git_env())
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
    hits = [p.split(":", 1)[1] for p in git_text("grep", "-l", "-z", "-F", "-f", str(work / "words"), top, "--",
                                                 *(f"*{s}" for s in U.CITED_BY + U.DOCS), ok=(0, 1)).split("\0") if p]
    hits = [p for p in hits if U.cited_kind(p) is not None]
    in_docs = docs(top) if any(U.cited_kind(p) == "doc" for p in hits) else set()
    left = dict(gone)
    for p, b in blobs(top, hits).items():
        for s in seeds(U, p, b, in_docs) & left.keys():
            del left[s]
    return left


def own() -> str:
    """This script's sha256: the caches' keys hold it, since it decides what they keep."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def digest(*parts: str) -> str:
    return hashlib.sha256("\0".join([KIND, *parts]).encode()).hexdigest()


def kept(f: Path) -> dict | None:
    """The entry kept at ``f``, touched as the last used, or None: there is none, or it doesn't read (then it is deleted, and
    whatever it held is worked out again)."""
    if not f.is_file():
        return None
    try:
        doc = json.loads(f.read_text())
    except (OSError, ValueError):
        f.unlink(missing_ok=True)
        return None
    os.utime(f)
    return doc


def keep(f: Path, doc: dict, most: int) -> None:
    """``doc`` at ``f``, written whole, with at most ``most`` entries beside it, the last used first."""
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{os.getpid()}")
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, f)
    for old in sorted(f.parent.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[most:]:
        old.unlink(missing_ok=True)


def lean_tree(rev: str) -> str:
    return git_text("rev-parse", f"{rev}:{ANALYSIS_DIR.rstrip('/')}").strip()


def declared(lo: str, cache: Path | None = None) -> tuple[set[str], set[str]]:
    """What the names the Lean under `ANALYSIS_DIR` declares at ``lo`` can start and end with (each a `WORD`): the first
    component of a `namespace` or of a name a keyword of `DECLARED` declares, and a private name's `_private`; the last
    component of such a name, or of a helper first on a line of a definition's `WHERE` block.  Kept in ``cache`` by the tree
    of `ANALYSIS_DIR` (`declared/K.json`): every PR of `next` above the same Lean reads the same."""
    f = cache / "declared" / f"{digest(own(), lean_tree(lo))}.json" if cache is not None else None
    if f is not None and (got := kept(f)) is not None:
        return set(got["heads"]), set(got["lasts"])
    heads, lasts = {"_private"}, set()
    for line in git_text("grep", "-h", "-o", "-E", DECLARED, lo, "--", f"{ANALYSIS_DIR}*.lean", ok=(0, 1)).splitlines():
        word, name = (line.split(None, 1) + [""])[:2]
        parts = name.strip().removeprefix("_root_.").split(".")
        if (m := WORD.match(parts[0])):
            heads.add(m.group(0))
        if word != "namespace" and (m := WORD.match(parts[-1])):
            lasts.add(m.group(0))
    for line in git_text("grep", "-h", "-A", "3", "-E", WHERE, lo, "--", f"{ANALYSIS_DIR}*.lean", ok=(0, 1)).splitlines():
        if (m := re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*)", line)):
            lasts.add(m.group(1))
    if f is not None:
        keep(f, {"heads": sorted(heads), "lasts": sorted(lasts)}, DECLARED_KEPT)
    return heads, lasts


def plausible(lo: str, gone: dict[tuple[str, str], list[str]], cache: Path | None = None) -> dict[tuple[str, str], list[str]]:
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
    heads, lasts = declared(lo, cache) if ends else (set(), set())

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


_STORE: list = []


def evidence_store():
    """This machine's evidence store, its attempts refreshed once a run: every lookup of one gate's run reads it."""
    if not _STORE:
        from research.store.local import LocalStore
        store = LocalStore()
        if store.remote is not None:
            store.refresh(attempts_only=True)
        _STORE.append(store)
    return _STORE[0]


def checked_commits(line: list[str], recorded: str, store=None) -> tuple[list[str], list[str]]:
    """The commits of ``line`` a recorded check (``recorded``, `common.RECORDED_CHECKS`) ran at, in ``line``'s order; what
    kept the store from being read, if anything."""
    if not line:
        return [], []
    try:
        store = evidence_store() if store is None else store
        rows = store.index.sql(f"SELECT DISTINCT source_commit FROM attempts WHERE {recorded} AND {ENDED} "
                               f"AND source_commit IN ({', '.join('?' * len(line))})", tuple(line))
    except Exception as e:  # noqa: BLE001  (no store here: the gate says it can't judge, naming why)
        return [], [f"the evidence store can't be read here ({type(e).__name__}: {e})"]
    ran = {c for (c,) in rows}
    return [c for c in line if c in ran], []


def stored(keys: dict[str, str], cache: Path, recorded: str, store=None, scan: int = SCAN,
           commits: tuple[str, ...] = ()) -> tuple[dict[str, Path], list[str]]:
    """({package: its facts file in ``cache``} for each of ``keys`` ({package: key}) that a recorded check (``recorded``,
    `common.RECORDED_CHECKS`) published (`STORED`), checked against the sha256 its listing gives; what kept the store from
    being read, if anything).  The checks of ``commits`` come first, however old (a `main` that stays put for days outlives
    the ``scan`` newest), then the ``scan`` newest."""
    want = {k: p for p, k in keys.items()}
    got: dict[str, Path] = {}
    errors: list[str] = []
    ended = f"{recorded} AND {ENDED}"
    try:
        if store is None:
            store = evidence_store()
        elif store.remote is not None:
            store.refresh(attempts_only=True)
        rows = [*(store.index.sql(f"SELECT id FROM attempts WHERE {ended} AND source_commit IN ({', '.join('?' * len(commits))}) "
                                  "ORDER BY start_utc DESC", commits) if commits else []),
                *store.index.sql(f"SELECT id FROM attempts WHERE {ended} ORDER BY start_utc DESC LIMIT ?", (scan,))]
    except Exception as e:  # noqa: BLE001  (no store here: the gate says it can't judge, naming why)
        return got, [f"the evidence store can't be read here ({type(e).__name__}: {e})"]
    for rid in dict.fromkeys(r for (r,) in rows):
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


def facts_of(LA, rev: str, tree: Path, files: list[str], cache: Path, work: Path,
             scan: int = SCAN) -> tuple[dict[str, str], dict[str, Path], list[str]]:
    """The keys of the packages the ratchet reads the facts of (`lean_audit.UNREAD_FACTS`) on ``tree`` (``rev``'s), and the
    facts of each that this gate's cache, this machine's Lean check (a node's kept passes) or the store holds (a check of
    ``rev``'s, or of the ``scan`` newest), and why the store gave none."""
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
        more, errors = stored(missing, cache, LA.COMMON.RECORDED_CHECKS, scan=scan, commits=(rev,))
        got |= more
    return keys, got, errors


def ancestor_facts(LA, lo: str, cache: Path, work: Path) -> tuple[str | None, dict[str, str], dict[str, Path], list[str]]:
    """The newest of the commits within `ANCESTORS` below ``lo`` along first parents that a recorded check ran at (a commit
    none ran at has no facts the store holds), of the `ANCESTOR_CHECKS` newest, whose Lean's facts are all kept here or
    published: (it, its keys, its facts, why the others gave none); None for it when none is."""
    line = git_text("rev-list", "--first-parent", f"--max-count={ANCESTORS + 1}", lo).split()[1:]
    ran, errors = checked_commits(line, LA.COMMON.RECORDED_CHECKS)
    for c in ran[:ANCESTOR_CHECKS]:
        tree = work / "ancestor"
        try:
            keys, facts, more = facts_of(LA, c, archive(c, tree), tracked(c), cache, work, scan=0)
        finally:
            shutil.rmtree(tree, ignore_errors=True)
        if len(facts) == len(keys):
            return c, keys, facts, errors
        errors.append(f"{c[:12]}: " + ("; ".join(more[:2]) or "not all of its Lean's facts are kept or published"))
    if not ran and not errors:
        errors.append(f"no recorded check ran at any of the {len(line)} commits below {lo[:12]} along first parents")
    return None, {}, {}, errors


def judge(U, LA, base: str, lo: str, top: str, gone: dict[tuple[str, str], list[str]], work: Path, cache: Path) -> dict:
    """The ratchet on ``lo``'s and ``top``'s trees from the facts of ``lo``'s Lean, else of ``base``'s, else of an ancestor's of
    ``lo`` (`ancestor_facts`): {"failures", "notes", "facts", "facts_from": the commit whose Lean's facts it read, "stale":
    whether that is the ancestor's}."""
    files_lo, files_top = tracked(lo), tracked(top)
    tree = archive(lo, work / "tree")
    keys, facts, errors = facts_of(LA, lo, tree, files_lo, cache, work)
    source, notes = lo, []
    if len(facts) < len(keys):
        keys_base, facts_base, errors_base = (keys, facts, []) if base == lo else \
            facts_of(LA, base, archive(base, work / "base"), tracked(base), cache, work)
        if len(facts_base) == len(keys_base):
            keys, facts, source = keys_base, facts_base, base
            notes.append(f"judged on the facts of `main` {base[:12]}'s Lean: not all of {lo[:12]}'s are kept or stored")
        else:
            at, keys_at, facts_at, errors_at = ancestor_facts(LA, lo, cache, work)
            if at is None:
                listed = "; ".join(f"{r[:12]}: " + ", ".join(f"{p} {k[:12]}" for p, k in ks.items())
                                   for r, ks in {lo: keys, base: keys_base}.items())
                why = errors + errors_base + errors_at
                raise CantJudge(f"the facts of the Lean of {lo[:12]} and of `main` {base[:12]} aren't all kept here or "
                                f"published by a recorded check of either or by the {SCAN} newest ({listed}), nor those of "
                                f"any of the {ANCESTOR_CHECKS} newest commits a recorded check ran at within {ANCESTORS} "
                                f"below {lo[:12]} along first parents" + (f": {'; '.join(why[:3])}" if why else ""))
            keys, facts, source = keys_at, facts_at, at
            notes.append(f"judged on the stale facts of {at[:12]}'s Lean, below {lo[:12]} along first parents: neither "
                         f"{lo[:12]}'s nor `main` {base[:12]}'s are kept or published yet (a Lean landing's are once its "
                         "lock run lands), and the ratchet of the Lean check after landing judges what they miss")
    F = U.Facts(*(facts[p] for p in LA.UNREAD_FACTS))
    t_lo = U.Tree(tree, files_lo)
    rep_lo = U.analyse(t_lo, F, full=False)
    named = U.module_names(t_lo)
    namer = U.path_namer(named)
    changed = sorted({f for fs in gone.values() for f in fs})
    was = blobs(lo, changed)
    overlay(tree, top, [p for p in git_text("diff", "--no-renames", "--name-only", "-z", lo, top).split("\0") if p])
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
                   + (f" (on the facts of `main` {base[:12]}'s Lean)" if source == base != lo else
                      f" (on the stale facts of {source[:12]}'s Lean)" if source not in (lo, base) else ""),
            "fix": [f"delete {' and '.join(new[:3])}{' ...' if len(new) > 3 else ''} (its file, and its import where another "
                    "module imports it): nothing reads it",
                    "or seed it again: keep a mention of it outside Lean in code that reads it (its dotted name or its file's "
                    f"path in a {', '.join(U.MODULE_CITED_BY)} file, or the full name of a declaration of it in a "
                    f"{', '.join(U.CITED_BY)} file), or name it in a lock's text"]})
    return {"failures": failures, "notes": notes, "facts": {p: k for p, k in keys.items()}, "facts_from": source,
            "stale": source not in (lo, base)}


def gate(base: str, top: str) -> dict:
    """The verdict on ``top``'s change: one kept for the same trees (``LO``'s and ``top``'s) by the same code (this script,
    ``base``'s `CODE_TREES`), and for one on `main`'s facts the same `main` Lean, else `judged` and kept (`verdicts/K.json`).
    Drive judges every PR of `next` again when `main` moves, and a PR whose members below are the same has the same trees.
    A verdict on stale facts isn't kept, so a later run reads the facts of the Lean it judges once they are published; a gate
    that can't judge raises, and keeps nothing."""
    base, top = (git_text("rev-parse", "--verify", f"{r}^{{commit}}").strip() for r in (base, top))
    lo = below(base, top)
    with tempfile.TemporaryDirectory(prefix="unread-gate-") as d:
        code = archive(base, Path(d) / "code", *CODE)
        cache = load(code / COMMON, "common_for_unread_gate").cache_dir() / "unread-gate"
        trees = git_text("rev-parse", f"{lo}^{{tree}}", f"{top}^{{tree}}", *(f"{base}:{p}" for p in CODE_TREES)).split()
        f = cache / "verdicts" / f"{digest(own(), *trees)}.json"
        if (got := kept(f)) is not None and got.get("main_lean") in (None, lean_tree(base)):
            was = got["doc"]
            return was | {"base": base, "top": top, "below": lo, "notes": [
                *was["notes"], f"kept: the verdict of {was['below'][:12]}..{was['top'][:12]}, the same trees, by the same code"]}
        doc = judged(base, lo, top, code, cache, Path(d))
    if not doc.get("stale"):
        keep(f, {"doc": doc, "main_lean": lean_tree(base) if doc.get("facts_from") not in (None, lo) else None}, VERDICTS_KEPT)
    return doc


def judged(base: str, lo: str, top: str, code: Path, cache: Path, d: Path) -> dict:
    doc = {"kind": KIND, "base": base, "top": top, "below": lo, "stage": "lean", "seed_files": [], "went": [],
           "failures": [], "notes": []}
    changed = [p for p in git_text("diff", "--no-renames", "--name-only", "-z", lo, top).split("\0") if p]
    if lean := [p for p in changed if p.endswith(LEAN_READS) or p.startswith(LEAN_TOOL)]:
        doc["notes"].append(f"it changes what a Lean check reads ({', '.join(lean[:3])}{' ...' if len(lean) > 3 else ''}): "
                            "the Lean check's ratchet judges it")
        return doc
    U = load(code / LEAN_TOOL / "unread.py", "unread_for_unread_gate")
    doc["stage"], doc["seed_files"] = "git", (seeded := [p for p in changed if U.cited_kind(p) is not None])
    docs = Docs(U, lo)
    gone = removed(U, lo, top, changed, seeded, docs)
    could = unnamed(U, top, plausible(lo, gone, cache), docs, d)
    doc["went"] = sorted(f"{r}:{n}" for r, n in could)[:40]
    if not could:
        doc["notes"].append(f"{len(seeded)} seed file(s) changed, and of the {len(gone)} name(s) they no longer hold, "
                            "none that the analysis's Lean could hold is gone from every seed file")
        return doc
    LA = load(code / LEAN_TOOL / "lean_audit.py", "lean_audit_for_unread_gate")
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
