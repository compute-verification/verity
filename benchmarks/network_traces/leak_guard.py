"""The record's leak guard: before custody publishes a run dir, every file in it that holds a gate run's witness is moved
out of it, so no run record takes one. chain_record.sh runs it on its way out, whatever the exit.

    python3 leak_guard.py RUN_DIR QUARANTINE [--gates DIR]... [--declared FILE] [--report FILE] [--dry]

A gate run is a directory holding witness/ under a --gates directory (chain_record's unpacked gates), and its public files
are the rest of it. Its pieces are each witness file's aligned pieces of at least `DISTINCT` distinct bytes, `TEXT` bytes
of a UTF-8 file and `BINARY` bytes of any other, searched raw and, a binary file's, as hex in either case; less the pieces
its public files hold, as they are and with their JSON written again (`reprinted`), and less a witness file equal to a
public one of its name.

Every regular file under RUN_DIR (symlinks not followed, as custody's `run_files`) is read, an archive through its members
(tar, zip, and gzip, bzip2 or xz streams), and it is a find when
* its path, or an archive member's, is a witness's: under a ``witness`` directory, or a name the declared data lists;
* it holds a piece at any offset that no entry of the allowlist excuses; or
* it is a file the guard can't read through (`UNREAD`, a damaged archive, an archive inside another).
The declared data (`DECLARED`, leak_guard.json beside this file) holds the witness file names (``names``: the gates'
witness files whose names no other file has) and the allowlist (``allow``): each entry a piece's sha256, the name of the
witness file it is cut from, the record paths it may lie in (fnmatch globs on the run-dir-relative path, an archive
member's as ``ARCHIVE/MEMBER``) and why its bytes are public. An entry's sha256 ``*`` excuses every piece of its witness
file in those records: for a record that observes what the witness file also records, in the same format, so that
which pieces match changes with the values (the runner's resources.jsonl and the stall probe's cgroup counters). An entry
is written only after its find was read by counts and digests (QUARANTINE/leak-guard.json), never by its bytes.

Each find is moved to QUARANTINE/files/ under its run-dir path. QUARANTINE must lie outside RUN_DIR and is kept 0700;
QUARANTINE/leak-guard.json names each find's pieces (sha256, the witness file and offset, distinct and printable bytes, the
JSON key path at the find). The --report file and stdout hold counts and paths only (what each entry excused, by its
index), never a piece's bytes or digest. --dry moves nothing (a published run read again). Exit 0 when nothing was found, 3 when something
was, 2 when the guard refuses to run.
"""
from __future__ import annotations

import argparse
import bz2
import fnmatch
import gzip
import hashlib
import importlib.util
import json
import lzma
import os
import shutil
import sys
import tarfile
import zipfile
import zlib
from pathlib import Path

#: the system python3 of an exit before chain_record's setup has no numpy: `Finder` then scans in pure Python
np = importlib.import_module("numpy") if importlib.util.find_spec("numpy") else None

DECLARED = Path(__file__).resolve().with_name("leak_guard.json")
SCHEMA = "network-traces/leak-guard/v1"
DECLARED_SCHEMA = "network-traces/leak-guard-declared/v1"
#: A UTF-8 witness file's pieces are 32 bytes, as the guard's first runs cut them; a binary
#: file's are 16, a salt's size, so one salt copied raw or as hex is found.
TEXT, BINARY = 32, 16
#: A piece of fewer distinct bytes isn't searched for: the runs of zeros and padding every record holds.
DISTINCT = 6
CHUNK = 64 << 20
#: A find's pieces the quarantine's report explains one by one; the rest it counts.
EXPLAINED = 64
#: Compressed formats this interpreter has no reader for, by magic: a file in one is a find.
UNREAD = {b"\x28\xb5\x2f\xfd": "zstd", b"\x04\x22\x4d\x18": "lz4", b"7z\xbc\xaf\x27\x1c": "7z", b"Rar!": "rar"}
STREAMS = {b"\x1f\x8b": gzip.open, b"BZh": bz2.open, b"\xfd7zXZ\x00": lzma.open}
ZIP = (b"PK\x03\x04", b"PK\x05\x06")
BROKEN = (OSError, EOFError, ValueError, tarfile.TarError, zipfile.BadZipFile, lzma.LZMAError, zlib.error,
          NotImplementedError, RuntimeError)


class Unread(Exception):
    """A file the guard can't read through, which is a find."""


class Finder:
    """Needles (each at least 8 bytes) found at any offset of the bytes given, keyed on their first 8 bytes as `a8.find`
    keys them, prefiltered by a table of the keys' low `BITS` bits."""

    BITS = 26

    def __init__(self, needles):
        self.keys: dict[int, list[bytes]] = {}
        for nd in needles:
            self.keys.setdefault(int.from_bytes(nd[:8], "little"), []).append(nd)
        self.span = max((len(nd) for nd in needles), default=8)
        if np is not None and self.keys:
            self.sorted = np.array(sorted(self.keys), dtype=np.uint64)
            self.mask = np.uint64((1 << self.BITS) - 1)
            self.table = np.zeros(1 << self.BITS, dtype=bool)
            self.table[(self.sorted & self.mask).astype(np.intp)] = True

    def find(self, data: bytes) -> set[bytes]:
        hits: set[bytes] = set()
        if not self.keys or len(data) < 8:
            return hits
        if np is None:
            for off in range(len(data) - 7):
                for nd in self.keys.get(int.from_bytes(data[off:off + 8], "little"), ()):
                    if data[off:off + len(nd)] == nd:
                        hits.add(nd)
            return hits
        a = np.frombuffer(data, dtype=np.uint8)
        for s in range(8):
            k = (len(a) - s) // 8
            if k <= 0:
                continue
            w = a[s:s + 8 * k].view("<u8")
            c = np.flatnonzero(self.table[(w & self.mask).astype(np.intp)])
            if not len(c):
                continue
            wc = w[c]
            i = np.minimum(np.searchsorted(self.sorted, wc), len(self.sorted) - 1)
            for at in c[self.sorted[i] == wc]:
                off = s + 8 * int(at)
                for nd in self.keys[int(w[at])]:
                    if data[off:off + len(nd)] == nd:
                        hits.add(nd)
        return hits

    def stream(self, f, head: bytes = b"") -> set[bytes]:
        """The needles in all of file object `f` (after `head`, bytes already read from it), read `CHUNK` at a time."""
        hits, tail, chunk = set(), b"", head or f.read(CHUNK)
        while chunk:
            data = tail + chunk
            hits |= self.find(data)
            tail = data[-(self.span - 1):]
            chunk = f.read(CHUNK)
        return hits


def _files(root: Path):
    """Every regular file under `root`, in path order, symlinks not followed (custody's `run_files`)."""
    for base, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            p = Path(base) / f
            if not p.is_symlink() and p.is_file():
                yield p


def gate_runs(roots: list[Path]) -> list[Path]:
    """The gate runs under `roots`: each directory holding a witness/ directory."""
    out = set()
    for r in roots:
        for base, dirs, _ in os.walk(r):
            if "witness" in dirs:
                out.add(Path(base))
                dirs.remove("witness")
    return sorted(out)


def _is_text(b: bytes) -> bool:
    try:
        b.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return b"\0" not in b


def forms(piece: bytes) -> list[bytes]:
    """The needles a piece is searched as: raw, and a binary file's piece as hex in either case."""
    return [piece] + ([piece.hex().encode(), piece.hex().upper().encode()] if len(piece) == BINARY else [])


def _dicts(doc):
    if isinstance(doc, dict):
        yield doc
        for v in doc.values():
            yield from _dicts(v)
    elif isinstance(doc, list):
        for v in doc:
            yield from _dicts(v)


def reprinted(p: Path) -> tuple[bytes, int]:
    """A public JSON file's dicts written again as a verifier's files write them (indented, keys sorted or not), so a piece
    that differs from the public file only in its layout reads public; and how many of its documents (a .jsonl's lines)
    aren't JSON, whose pieces then read public only as they are."""
    if p.suffix not in (".json", ".jsonl"):
        return b"", 0
    text = p.read_text(errors="replace")
    out, unparsed = [], 0
    for line in text.splitlines() if p.suffix == ".jsonl" else [text]:
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            unparsed += 1
            continue
        for d in _dicts(doc):
            for s in (False, True):
                out += [json.dumps([d], indent=1, sort_keys=s), json.dumps(d, indent=1, sort_keys=s),
                        json.dumps(d, indent=2, sort_keys=s)]
    return "\n".join(out).encode(), unparsed


def pieces(gates: list[Path]) -> tuple[dict[bytes, tuple[str, int]], int, int]:
    """The gate runs' witness pieces, each to the witness file (under its gate run's name) and offset it is first cut
    from, less the pieces any gate run's public files hold; how many pieces those dropped; and how many of the public JSON
    files' documents weren't JSON, so weren't written again (`reprinted`)."""
    got: dict[bytes, tuple[str, int]] = {}
    public: list[Path] = []
    for g in gates:
        pub = [p for p in _files(g) if "witness" not in p.relative_to(g).parts]
        twins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in pub}
        public += pub
        for f in _files(g / "witness"):
            b = f.read_bytes()
            if twins.get(f.name) == hashlib.sha256(b).hexdigest():
                continue
            n = TEXT if _is_text(b) else BINARY
            for o in range(0, len(b) - n + 1, n):
                if len(set(x := b[o:o + n])) >= DISTINCT:
                    got.setdefault(x, (f"{g.name}/{f.relative_to(g).as_posix()}", o))
    raw = {nd: x for x in got for nd in forms(x)}
    finder, out, unparsed = Finder(raw), set(), 0
    for p in public:
        with open(p, "rb") as f:
            out |= {raw[nd] for nd in finder.stream(f)}
        again, n = reprinted(p)
        out |= {raw[nd] for nd in finder.find(again)}
        unparsed += n
    return {x: o for x, o in got.items() if x not in out}, len(out), unparsed


def load_declared(path: Path) -> tuple[set[str], list[dict]]:
    """The declared witness names and allowlist, each entry refused unless it names a sha256 (or ``*``), a witness
    file's name, record globs and why."""
    doc = json.loads(path.read_text())
    if doc.get("schema") != DECLARED_SCHEMA:
        raise SystemExit(f"leak_guard: {path} is not {DECLARED_SCHEMA}")
    for i, e in enumerate(doc["allow"]):
        sha = e.get("sha256", "")
        if not ((sha == "*" or (len(sha) == 64 and set(sha) <= set("0123456789abcdef")))
                and e.get("witness") and e.get("records") and e.get("why")):
            raise SystemExit(f"leak_guard: {path} entry {i} needs a sha256 (lower-case hex, or * for any piece of the "
                             "witness file), the witness file's name, its records and why")
    return set(doc["names"]), doc["allow"]


def _is_archive(head: bytes) -> bool:
    return head.startswith((*ZIP, *STREAMS, *UNREAD)) or head[257:262] == b"ustar"


def _contents(p: Path):
    """(member, file object) for each content of `p`: `p` itself (member None), a compressed stream's contents (member
    ""), or an archive's members (a member that isn't a regular file with file object None); an archive the guard
    can't read raises `Unread`."""
    with open(p, "rb") as f:
        head = f.read(512)
    for magic, kind in UNREAD.items():
        if head.startswith(magic):
            raise Unread(f"a {kind} file, which the guard can't read")
    if head.startswith(ZIP):
        with zipfile.ZipFile(p) as z:
            for i in z.infolist():
                if not i.is_dir():
                    with z.open(i) as m:
                        yield i.filename, m
        return
    opener = next((o for magic, o in STREAMS.items() if head.startswith(magic)), None)
    if opener is not None:
        with opener(p, "rb") as s:
            head = s.read(512)
    if head[257:262] == b"ustar":
        with tarfile.open(p, mode="r|*") as t:
            for m in t:
                yield m.name, t.extractfile(m) if m.isfile() else None
        return
    if opener is None:
        with open(p, "rb") as s:
            yield None, s
    else:
        with opener(p, "rb") as s:
            yield "", s


def _named(parts: tuple[str, ...], names: set[str]) -> bool:
    return "witness" in parts[:-1] or (bool(parts) and parts[-1] in names)


def read(p: Path, finder: Finder, names: set[str]) -> dict:
    """What `p` holds: its first witness-named member, the needles in it with the member each is in, and why it can't be
    read through (None when it can)."""
    named, hits, unread = None, {}, None
    try:
        for member, f in _contents(p):
            if member and _named(Path(member).parts, names):
                named = member
                break
            if f is None:
                continue
            head = f.read(512)
            if member is not None and _is_archive(head):
                unread = f"an archive inside an archive ({member or 'its stream'})"
                break
            for nd in finder.stream(f, head):
                hits.setdefault(nd, member)
    except Unread as e:
        unread = str(e)
    except BROKEN as e:
        unread = f"a file the guard can't read through ({type(e).__name__})"
    return {"named": named, "hits": hits, "unread": unread}


def keypath(doc, at: str, path=()):
    """The key path of the JSON value of `doc` holding text `at` (``<key>`` for a key), or None when no one value does."""
    if isinstance(doc, dict):
        for k, v in doc.items():
            if at in k:
                return [*path, "<key>"]
            if (r := keypath(v, at, (*path, k))) is not None:
                return r
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            if (r := keypath(v, at, (*path, i))) is not None:
                return r
    elif isinstance(doc, str) and at in doc:
        return list(path)
    return None


def explain(b: bytes | None, nd: bytes, piece: bytes, origin: tuple[str, int], member: str | None) -> dict:
    """A find's piece by counts and digests: its sha256, the witness file and offset it is cut from, whether it was found
    raw or as hex, its distinct and printable bytes, and in a plain file `b` the offset, line and JSON key path of the
    find (``"not JSON"`` when its line isn't)."""
    out = {"sha256": hashlib.sha256(piece).hexdigest(), "witness": origin[0], "witness_at": origin[1],
           "form": "raw" if nd == piece else "hex", "distinct": len(set(piece)),
           "printable": sum(32 <= c < 127 for c in piece), "member": member}
    if b is not None:
        at = b.find(nd)
        start, end = b.rfind(b"\n", 0, at) + 1, b.find(b"\n", at)
        try:
            kp = keypath(json.loads(b[start:end if end >= 0 else len(b)]), nd.decode("utf-8", "replace"))
        except ValueError:
            kp = "not JSON"
        out |= {"at": at, "line": b.count(b"\n", 0, at), "keypath": kp}
    return out


def excused(entry: dict, sha: str, origin: tuple[str, int], where: str) -> bool:
    return (entry["sha256"] in ("*", sha) and entry["witness"] == Path(origin[0]).name
            and any(fnmatch.fnmatchcase(where, g) for g in entry["records"]))


def guard(run: Path, quarantine: Path, gates: list[Path], names: set[str], allow: list[dict],
          dry: bool = False) -> tuple[dict, dict]:
    """The guard on `run`, its finds moved under `quarantine` (unless `dry`): (the report, counts and paths only; the
    quarantine's, with each find's pieces)."""
    runs = gate_runs(gates)
    got, dropped, unparsed = pieces(runs)
    raw = {nd: x for x in got for nd in forms(x)}
    finder = Finder(raw)
    files = list(_files(run))
    size = {p: p.stat().st_size for p in files}
    found, held, used = [], [], {}
    for p in files:
        rel = p.relative_to(run).as_posix()
        r = ({"named": None, "hits": {}, "unread": None, "path": True} if _named(Path(rel).parts, names)
             else read(p, finder, names))
        bad = []
        for nd, member in r["hits"].items():
            piece = raw[nd]
            sha, where = hashlib.sha256(piece).hexdigest(), f"{rel}/{member}" if member else rel
            if (i := next((i for i, e in enumerate(allow) if excused(e, sha, got[piece], where)), None)) is not None:
                u = used.setdefault(i, (set(), set()))
                u[0].add(piece)
                u[1].add(where)
            else:
                bad.append((nd, piece, member))
        n = len({x for _, x, _ in bad})
        why = (["a witness path"] * bool(r.get("path")) + [f"a witness-named member, {r['named']}"] * bool(r["named"])
               + [r["unread"]] * bool(r["unread"]) + [f"{n} witness pieces"] * bool(n))
        if not why:
            continue
        doc = {"path": rel, "bytes": size[p], "why": why, "pieces": n}
        found.append(doc)
        b = p.read_bytes() if bad and bad[0][2] is None else None
        held.append({**doc, "pieces_found": [explain(b if m is None else None, nd, x, got[x], m)
                                             for nd, x, m in bad[:EXPLAINED]]})
    for doc in [] if dry else found:
        dest = quarantine / "files" / doc["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run / doc["path"]), dest)
    report = {"schema": SCHEMA, "run": str(run), "quarantine": str(quarantine), "gate_runs": [g.name for g in runs],
              "pieces": len(got), "public_dropped": dropped, "public_unparsed": unparsed, "files_scanned": len(files),
              "bytes_scanned": sum(size.values()),
              "excused": [{"entry": i, "witness": allow[i]["witness"], "pieces": len(x), "records": sorted(w)}
                          for i, (x, w) in sorted(used.items())],
              "found": found, "dry": dry, "verdict": "refused" if found else "clean"}
    return report, {**report, "found": held}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("quarantine", type=Path)
    ap.add_argument("--gates", type=Path, action="append", default=[])
    ap.add_argument("--declared", type=Path, default=DECLARED)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    run, q = a.run.resolve(), a.quarantine.resolve()
    if not run.is_dir():
        print(f"leak_guard: refused: no run dir {run}", file=sys.stderr)
        return 2
    if q == run or q.is_relative_to(run):
        print(f"leak_guard: refused: the quarantine {q} lies inside the run dir {run}, which custody publishes",
              file=sys.stderr)
        return 2
    if missing := [str(g) for g in a.gates if not g.is_dir()]:
        print(f"leak_guard: refused: no gates directory {missing}", file=sys.stderr)
        return 2
    names, allow = load_declared(a.declared)
    q.mkdir(parents=True, exist_ok=True)
    q.chmod(0o700)
    report, held = guard(run, q, [g.resolve() for g in a.gates], names, allow, a.dry)
    (q / "leak-guard.json").write_text(json.dumps(held, indent=1) + "\n")
    if a.report is not None:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=1) + "\n")
    print(f"leak guard: {report['verdict']}: {report['files_scanned']} files ({report['bytes_scanned']} bytes) scanned "
          f"for {report['pieces']} pieces of {len(report['gate_runs'])} gate runs, "
          f"{sum(e['pieces'] for e in report['excused'])} excused by the allowlist; {len(report['found'])} {'found (--dry)' if a.dry else f'moved to {q}/files'}"
          + "".join(f"\n  {d['path']}: {'; '.join(d['why'])}" for d in report["found"]), flush=True)
    return 3 if report["found"] else 0


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
