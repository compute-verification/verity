"""The record's leak guard: before custody publishes a run dir, every file in it that holds a gate run's witness is moved
out of it, so no run record takes one; and a job's outputs, held outside the run dir, reach it only once the guard has
passed them. Every job that publishes after a gate runs it on its way out, whatever the exit.

    python3 leak_guard.py RUN_DIR QUARANTINE [--gates DIR|TARBALL]... [--hold DIR [--release-to OUT]] [--budget-s S]
        [--public GLOB]... [--declared FILE] [--report FILE] [--dry]

A gate run is a directory holding witness/ under a --gates directory (a tarball, a gate's fetch, is unpacked under
QUARANTINE for the guard alone and removed after), and its public files are the rest of it. Its pieces are each witness
file's aligned pieces of at least `DISTINCT` distinct bytes, `TEXT` bytes of a UTF-8 file and `BINARY` bytes of any other,
searched raw and, a binary file's, as hex in either case, less a witness file equal to a public one of its name. A
piece the records hold that the gate runs' public files also hold, as they are or, a public JSON file of at most
`REPRINT` bytes, with its JSON written again (`reprinted`), is public.

Every regular file under RUN_DIR and HOLD (symlinks not followed, as custody's `run_files`) is read, an archive through its
members (tar, zip, and gzip, bzip2 or xz streams), except a file or member equal to one of the gate runs' public files
(their sha256s, the bundle a gate publishes to any key), which is public as it is and isn't scanned. It is a find when
* its path, or an archive member's, is a witness's: under a ``witness`` directory, or a name the declared data lists;
* it holds a piece at any offset that no entry of the allowlist excuses;
* it is a file the guard can't read through (`UNREAD`, a damaged archive, an archive inside another); or
* with ``--public``, its record path matches no ``--public`` glob (fnmatch): a record that declares what is public holds
  nothing else, whatever it holds, so the counts and lengths no witness piece shows (a window's requests, steps and
  outputs, its advice's bits) stay out by being undeclared.
The declared data (`DECLARED`, leak_guard.json beside this file) holds the witness file names (``names``: the gates'
witness files whose names no other file has) and the allowlist (``allow``): each entry a piece's sha256, the name of the
witness file it is cut from, the record paths it may lie in (fnmatch globs on the run-dir-relative path, an archive
member's as ``ARCHIVE/MEMBER``) and why its bytes are public. An entry's sha256 ``*`` excuses every piece of its witness
file in those records: for a record that observes what the witness file also records, in the same format, so that
which pieces match changes with the values (the runner's resources.jsonl and the stall probe's cgroup counters). An entry
is written only after its find was read by counts and digests (QUARANTINE/leak-guard.json), never by its bytes.

Each find is moved to QUARANTINE/files/ under its record path. QUARANTINE must lie outside RUN_DIR and is kept 0700;
QUARANTINE/leak-guard.json names each find's pieces (sha256, the witness file and offset, distinct and printable bytes, the
JSON key path at the find). The --report file and stdout hold counts and paths only (what each entry excused, by its
index), never a piece's bytes or digest. --dry moves nothing (a published run read again).

**Holding.** With --hold, HOLD (outside RUN_DIR and OUT) holds the job's outputs until the guard has read them, and a held
file's record path is its path under OUT (by default RUN_DIR/out). Once the guard has read every file, each held file that
isn't a find is moved to OUT, and only then (a held log onto a log of its path already there; any other held file whose
path is taken refuses the release). A refusal, the budget, an error or SIGTERM moves nothing out of HOLD, so a
job that keeps its outputs there leaks none of them however it ends, a SIGKILL included: HOLD is their quarantine.

**Cost.** Linear in the bytes read: the witness once, the run dir's and HOLD's files less the public ones, then the
public files for the pieces those held alone. Each 8-byte word, at every alignment, is looked up in a table of the
needles' low bits sized to be at most 1/8 full, and each hit is compared whole, in numpy (without numpy, in pure Python,
far slower). Its memory is linear in the witness: the needles are one matrix of 32-byte rows. --budget-s (`BUDGET_S`
by default) bounds the time: past it the guard stops.

Exit 0 when nothing was found (with --hold, everything released), 3 when something was (the finds moved to the
quarantine, the rest released), 2 when the guard refuses to run or stops past its budget, 143 on SIGTERM.
"""
from __future__ import annotations

import argparse
import bz2
import fnmatch
import gzip
import hashlib
import importlib.util
import io
import json
import lzma
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import zipfile
import zlib
from pathlib import Path

#: the system python3 of an exit before chain_record's setup may have no numpy: `Finder` then scans in pure Python.
#: numpy's import starts its BLAS threads, which start with SIGTERM blocked: a SIGTERM the kernel gave one of them would
#: reach the guard only once the main thread's read returned, however long that read blocks.
_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
try:
    np = importlib.import_module("numpy") if importlib.util.find_spec("numpy") else None
finally:
    signal.pthread_sigmask(signal.SIG_SETMASK, _mask)

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
#: The largest public JSON file written again (`reprinted`): every dict of a document written six ways is many times its
#: size, so a larger one's pieces read public only as they are, which can only add finds.
REPRINT = 4 << 20
#: Seconds the guard runs before it stops (exit 2), inside a job's time limit.
BUDGET_S = 1800
#: The guard's clock, for its budget alone.
clock = time.monotonic
#: Compressed formats this interpreter has no reader for, by magic: a file in one is a find.
UNREAD = {b"\x28\xb5\x2f\xfd": "zstd", b"\x04\x22\x4d\x18": "lz4", b"7z\xbc\xaf\x27\x1c": "7z", b"Rar!": "rar"}
STREAMS = {b"\x1f\x8b": gzip.open, b"BZh": bz2.open, b"\xfd7zXZ\x00": lzma.open}
ZIP = (b"PK\x03\x04", b"PK\x05\x06")
BROKEN = (OSError, EOFError, ValueError, tarfile.TarError, zipfile.BadZipFile, lzma.LZMAError, zlib.error,
          NotImplementedError, RuntimeError)


class Unread(Exception):
    """A file the guard can't read through, which is a find."""


class Refused(Exception):
    """The guard won't run or release, and why."""


class OverBudget(Refused):
    """The guard ran past its budget."""


class Ended(BaseException):
    """SIGTERM reached the guard."""


class Budget:
    """`seconds` from now on `clock`; None for no limit."""

    def __init__(self, seconds: float | None = None):
        self.seconds, self.start = seconds, clock()

    def check(self, what: str) -> None:
        if self.seconds is not None and clock() - self.start > self.seconds:
            raise OverBudget(f"past its budget of {self.seconds:g} s, {what}")


class Finder:
    """Needles (each 8 to 32 bytes) found at any offset of the bytes given: each 8-byte word of them, at every alignment,
    looked up among the needles' first 8 bytes (in numpy through a table of their low bits at most 1/8 full, so the cost
    is linear in the bytes whatever the needles' count), and each hit compared whole. `hits` and `stream` give the
    indices of the needles found: of `items`, or of the rows of a needle matrix given as ``rows`` (each needle
    zero-padded, ``lens`` its length), which holds the needles without a Python object each."""

    def __init__(self, needles=(), budget: Budget | None = None, *, rows=None, lens=None):
        self.budget = budget or Budget()
        if rows is None:
            self.items: list[bytes] | None = list(dict.fromkeys(needles))
            self.span = max((len(nd) for nd in self.items), default=8)
            self.size = len(self.items)
            if np is None or not self.items:
                self.keys: dict[int, list[int]] = {}
                for i, nd in enumerate(self.items):
                    self.keys.setdefault(int.from_bytes(nd[:8], "little"), []).append(i)
                return
            rows = np.frombuffer(b"".join(nd.ljust(self.span, b"\0") for nd in self.items), np.uint8)
            rows = rows.reshape(len(self.items), self.span)
            lens = np.fromiter((len(nd) for nd in self.items), np.int8, len(self.items))
        else:
            self.items, self.span = None, rows.shape[1]
        self.size = n = len(rows)
        if not n:
            return
        keys = np.ascontiguousarray(rows[:, :8]).view("<u8").ravel()
        self.order = np.argsort(keys, kind="stable")
        self.base, self.sorted, self.lens = rows, keys[self.order], lens[self.order].astype(np.int64)
        del keys
        bits = min(30, max(16, (n - 1).bit_length() + 3))
        self.mask = np.uint64((1 << bits) - 1)
        self.table = np.zeros(1 << bits, dtype=bool)
        self.table[(self.sorted & self.mask).astype(np.intp)] = True

    def hits(self, data: bytes) -> set[int]:
        n = len(data)
        if not self.size or n < 8:
            return set()
        if np is None:
            got = set()
            for off in range(n - 7):
                if not off & 0xFFFFF:
                    self.budget.check("scanning in pure Python (no numpy)")
                for i in self.keys.get(int.from_bytes(data[off:off + 8], "little"), ()):
                    if data[off:off + len(self.items[i])] == self.items[i]:
                        got.add(i)
            return got
        a = np.frombuffer(data, np.uint8)
        pad = np.zeros(n + self.span, np.uint8)
        pad[:n] = a
        span, last, found = np.arange(self.span), len(self.sorted) - 1, []
        for s in range(8):
            k = (n - s) // 8
            if k <= 0:
                continue
            w = a[s:s + 8 * k].view("<u8")
            c = np.flatnonzero(self.table[(w & self.mask).astype(np.intp)])
            if not c.size:
                continue
            wc = w[c]
            j = np.minimum(np.searchsorted(self.sorted, wc), last)
            ok = self.sorted[j] == wc
            off, j = s + 8 * c[ok], j[ok]
            while off.size:
                lens = self.lens[j]
                same = ((pad[off[:, None] + span] == self.base[self.order[j]]) | (span >= lens[:, None])).all(axis=1)
                found.append(j[same & (off + lens <= n)])
                more = j < last
                off, j = off[more], j[more] + 1
                more = self.sorted[j] == self.sorted[j - 1]
                off, j = off[more], j[more]
        return set(self.order[np.concatenate(found)].tolist()) if found else set()

    def stream(self, f, head: bytes = b"") -> set[int]:
        """The needles in all of file object `f` (after `head`, bytes already read from it), read `CHUNK` at a time."""
        hits, tail, chunk = set(), b"", head or f.read(CHUNK)
        while chunk:
            self.budget.check("scanning")
            data = tail + chunk
            hits |= self.hits(data)
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


def cut(b: bytes, n: int):
    """The offsets of `b`'s aligned `n`-byte pieces of at least `DISTINCT` distinct bytes (an array in numpy)."""
    k = len(b) // n
    if np is None:
        return [o for o in range(0, k * n, n) if len(set(b[o:o + n])) >= DISTINCT]
    rows, out = np.frombuffer(b, np.uint8, k * n).reshape(k, n), [np.zeros(0, np.int64)]
    for i in range(0, k, 1 << 20):
        s = np.sort(rows[i:i + (1 << 20)], axis=1)
        out.append((i + np.flatnonzero(np.count_nonzero(np.diff(s, axis=1), axis=1) + 1 >= DISTINCT)) * n)
    return np.concatenate(out)


def _hex(rows, upper: bool):
    """Each row of bytes as its hex digits."""
    digits = np.frombuffer(b"0123456789ABCDEF" if upper else b"0123456789abcdef", np.uint8)
    out = np.empty((len(rows), 2 * rows.shape[1]), np.uint8)
    out[:, 0::2], out[:, 1::2] = digits[rows >> 4], digits[rows & 15]
    return out


class Pieces:
    """The gate runs' witness pieces as the needles they are searched as (`forms`), each with the witness file and offset
    it is first cut from; in numpy one zero-padded row a needle, with no Python object a piece."""

    SPAN = TEXT

    def __init__(self):
        self.labels: list[str] = []
        self._parts: list[tuple] = []
        self.count = 0

    def add(self, label: str, b: bytes, n: int) -> None:
        offs = cut(b, n)
        if not len(offs):
            return
        src = len(self.labels)
        self.labels.append(label)
        if np is None:
            self._parts.append((label, [(nd, src, o, f) for o in offs for f, nd in enumerate(forms(b[o:o + n]))]))
            return
        rows = np.frombuffer(b, np.uint8, len(b) // n * n).reshape(-1, n)[offs // n]
        for f, r in enumerate([rows, _hex(rows, False), _hex(rows, True)] if n == BINARY else [rows]):
            pad = np.zeros((len(r), self.SPAN), np.uint8)
            pad[:, :r.shape[1]] = r
            self._parts.append((pad, np.full(len(r), r.shape[1], np.int8), np.full(len(r), src, np.int32), offs,
                                np.full(len(r), f, np.int8)))

    def finish(self, budget: Budget) -> Finder:
        """The needles, each once (its first cut kept), and their `Finder`."""
        if np is None:
            first: dict[bytes, tuple[int, int, int]] = {}
            for _, part in self._parts:
                for nd, src, o, f in part:
                    first.setdefault(nd, (src, o, f))
            self._parts = []
            self.needles, self.meta = list(first), list(first.values())
            self.count = sum(f == 0 for _, _, f in self.meta)
            return Finder(self.needles, budget)
        if not self._parts:
            self.rows = np.zeros((0, self.SPAN), np.uint8)
            self.lens = self.src = self.off = self.form = np.zeros(0, np.int64)
            return Finder((), budget)
        cols = [np.concatenate(c) for c in zip(*self._parts)]
        self._parts = []
        rows, lens, src, off, form = cols
        budget.check("ordering the witness pieces")
        w = rows.view("<u8")
        order = np.lexsort((lens, w[:, 3], w[:, 2], w[:, 1], w[:, 0]))
        sw, sl = w[order], lens[order]
        first = np.ones(len(order), bool)
        first[1:] = (sw[1:] != sw[:-1]).any(axis=1) | (sl[1:] != sl[:-1])
        keep = np.sort(order[first])
        del w, sw, sl, order, first
        self.rows, self.lens, self.src, self.off, self.form = rows[keep], lens[keep], src[keep], off[keep], form[keep]
        self.count = int(np.count_nonzero(self.form == 0))
        return Finder(budget=budget, rows=self.rows, lens=self.lens)

    def needle(self, i: int) -> bytes:
        return self.needles[i] if np is None else self.rows[i, :self.lens[i]].tobytes()

    def piece(self, i: int) -> bytes:
        """The piece needle `i` is a form of."""
        form = self.meta[i][2] if np is None else self.form[i]
        nd = self.needle(i)
        return nd if form == 0 else bytes.fromhex(nd.decode())

    def origin(self, i: int) -> tuple[str, int]:
        src, off = (self.meta[i][0], self.meta[i][1]) if np is None else (self.src[i], self.off[i])
        return self.labels[int(src)], int(off)


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


class Public:
    """The gate runs' public files by sha256 (and their sizes, to read only a file that could be one)."""

    def __init__(self):
        self.digests: set[str] = set()
        self.sizes: set[int] = set()

    def add_file(self, f) -> str:
        h, size = hashlib.sha256(), 0
        while chunk := f.read(CHUNK):
            h.update(chunk)
            size += len(chunk)
        self.digests.add(h.hexdigest())
        self.sizes.add(size)
        return h.hexdigest()

    def holds(self, b: bytes) -> bool:
        return len(b) in self.sizes and hashlib.sha256(b).hexdigest() in self.digests


def pieces(gates: list[Path], budget: Budget | None = None) -> tuple[Pieces, Public, list[Path]]:
    """The gate runs' witness pieces, each with the witness file (under its gate run's name) and offset it is first cut
    from; the public files' digests; and the public files."""
    budget = budget or Budget()
    got, public, pub_files = Pieces(), Public(), []
    for g in gates:
        pub = [p for p in _files(g) if "witness" not in p.relative_to(g).parts]
        twins = {}
        for p in pub:
            with open(p, "rb") as f:
                twins[p.name] = public.add_file(f)
        pub_files += pub
        for f in _files(g / "witness"):
            budget.check(f"cutting {g.name}'s witness")
            b = f.read_bytes()
            if twins.get(f.name) == hashlib.sha256(b).hexdigest():
                continue
            got.add(f"{g.name}/{f.relative_to(g).as_posix()}", b, TEXT if _is_text(b) else BINARY)
    return got, public, pub_files


def public_pieces(found: set[bytes], pub_files: list[Path], budget: Budget | None = None) -> tuple[set[bytes], dict]:
    """The pieces of `found` that the gate runs' public files hold, as they are and, each public JSON file of at most
    `REPRINT` bytes, written again (`reprinted`); and counts: ``unparsed``, the public JSON documents that weren't JSON,
    so weren't written again, and ``unreprinted``, the public JSON files past `REPRINT`, read only as they are. Only the
    pieces a record holds are looked for, so this costs the public files' bytes once, whatever the witness's size."""
    budget = budget or Budget()
    counts = {"unparsed": 0, "unreprinted": 0}
    if not found:
        return set(), counts
    raw = {nd: x for x in found for nd in forms(x)}
    finder, out = Finder(raw, budget), set()
    for p in pub_files:
        with open(p, "rb") as f:
            out |= {raw[finder.items[i]] for i in finder.stream(f)}
        if p.suffix in (".json", ".jsonl") and p.stat().st_size > REPRINT:
            counts["unreprinted"] += 1
            continue
        again, n = reprinted(p)
        out |= {raw[finder.items[i]] for i in finder.hits(again)}
        counts["unparsed"] += n
    return out, counts


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
    """(member, file object, size) for each content of `p`: `p` itself (member None), a compressed stream's contents
    (member "", size None), or an archive's members (a member that isn't a regular file with file object None); an
    archive the guard can't read raises `Unread`."""
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
                        yield i.filename, m, i.file_size
        return
    opener = next((o for magic, o in STREAMS.items() if head.startswith(magic)), None)
    if opener is not None:
        with opener(p, "rb") as s:
            head = s.read(512)
    if head[257:262] == b"ustar":
        with tarfile.open(p, mode="r|*") as t:
            for m in t:
                yield m.name, t.extractfile(m) if m.isfile() else None, m.size
        return
    if opener is None:
        with open(p, "rb") as s:
            yield None, s, p.stat().st_size
    else:
        with opener(p, "rb") as s:
            yield "", s, None


def _named(parts: tuple[str, ...], names: set[str]) -> bool:
    return "witness" in parts[:-1] or (bool(parts) and parts[-1] in names)


def read(p: Path, finder: Finder, names: set[str], public: Public) -> dict:
    """What `p` holds: its first witness-named member, the needles in it with the member each is in, why it can't be read
    through (None when it can), and how many of its contents are public files, which aren't scanned."""
    named, hits, unread, skipped = None, {}, None, 0
    try:
        for member, f, size in _contents(p):
            if member and _named(Path(member).parts, names):
                named = member
                break
            if f is None:
                continue
            head = f.read(512)
            if member is not None and _is_archive(head):
                unread = f"an archive inside an archive ({member or 'its stream'})"
                break
            if size in public.sizes:
                data = head + f.read()
                if public.holds(data):
                    skipped += 1
                    continue
                found = finder.stream(io.BytesIO(data))
            else:
                found = finder.stream(f, head)
            for i in found:
                hits.setdefault(i, member)
    except Unread as e:
        unread = str(e)
    except BROKEN as e:
        unread = f"a file the guard can't read through ({type(e).__name__})"
    return {"named": named, "hits": hits, "unread": unread, "public": skipped}


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


def record_path(dest: Path, run: Path, hold: Path, p: Path) -> str:
    """A held file's record path: where it is released to, relative to the run dir (else under ``held/``)."""
    return dest.relative_to(run).as_posix() if dest.is_relative_to(run) else f"held/{p.relative_to(hold).as_posix()}"


def guard(run: Path, quarantine: Path, gates: list[Path], names: set[str], allow: list[dict], dry: bool = False,
          hold: Path | None = None, release: Path | None = None, budget: Budget | None = None,
          declared: list[str] | None = None) -> tuple[dict, dict]:
    """The guard on `run` (and `hold`, its files bound for `release`), its finds moved under `quarantine` and the rest of
    `hold` released only after every file was read (unless `dry`): (the report, counts and paths only; the quarantine's,
    with each find's pieces). With `declared` (``--public``), a file whose record path matches none of its globs is a
    find."""
    budget = budget or Budget()
    runs = gate_runs(gates)
    got, public, pub_files = pieces(runs, budget)
    finder = got.finish(budget)
    files = [(p.relative_to(run).as_posix(), p, False) for p in _files(run)]
    held = [(record_path(release / p.relative_to(hold), run, hold, p), p, True) for p in _files(hold)] if hold else []
    there = {rel for rel, _, _ in files}
    if clash := [rel for rel, _, _ in held if rel in there and not rel.endswith(".log")]:
        raise Refused(f"held files would replace {clash[:4]} in the record")
    size = {p: p.stat().st_size for _, p, _ in files + held}
    read_ = []
    for rel, p, is_held in sorted(files + held):
        budget.check(f"at {rel}")
        read_.append((rel, p, is_held, {"named": None, "hits": {}, "unread": None, "public": 0, "path": True}
                      if _named(Path(rel).parts, names) else read(p, finder, names, public)))
    dropped, counts = public_pieces({got.piece(i) for *_, r in read_ for i in r["hits"]}, pub_files, budget)
    found, explained, used, skipped = [], [], {}, 0
    for rel, p, is_held, r in read_:
        skipped += r["public"]
        bad = []
        for i, member in r["hits"].items():
            piece = got.piece(i)
            if piece in dropped:
                continue
            sha, where, origin = hashlib.sha256(piece).hexdigest(), f"{rel}/{member}" if member else rel, got.origin(i)
            if (k := next((k for k, e in enumerate(allow) if excused(e, sha, origin, where)), None)) is not None:
                u = used.setdefault(k, (set(), set()))
                u[0].add(piece)
                u[1].add(where)
            else:
                bad.append((got.needle(i), piece, member, origin))
        n = len({x for _, x, _, _ in bad})
        undeclared = declared is not None and not any(fnmatch.fnmatchcase(rel, g) for g in declared)
        why = (["a witness path"] * bool(r.get("path")) + [f"a witness-named member, {r['named']}"] * bool(r["named"])
               + [r["unread"]] * bool(r["unread"]) + [f"{n} witness pieces"] * bool(n)
               + ["not declared public (--public)"] * undeclared)
        if not why:
            continue
        doc = {"path": rel, "bytes": size[p], "why": why, "pieces": n, **({"held": True} if is_held else {})}
        found.append((doc, p))
        b = p.read_bytes() if bad and bad[0][2] is None else None
        explained.append({**doc, "pieces_found": [explain(b if m is None else None, nd, x, o, m)
                                                  for nd, x, m, o in bad[:EXPLAINED]]})
    released = 0
    if not dry:
        for doc, p in found:
            dest = quarantine / "files" / doc["path"]
            if dest.exists():
                dest = dest.with_name(dest.name + ".held")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), dest)
        for _, p, _ in held:
            if not p.is_file():
                continue
            dest = release / p.relative_to(hold)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                with open(dest, "ab") as w, open(p, "rb") as r:
                    shutil.copyfileobj(r, w)
                p.unlink()
            else:
                shutil.move(str(p), dest)
            released += 1
    found = [doc for doc, _ in found]
    report = {"schema": SCHEMA, "run": str(run), "quarantine": str(quarantine), "gate_runs": [g.name for g in runs],
              "pieces": got.count, "public_dropped": len(dropped), "public_unparsed": counts["unparsed"],
              "public_unreprinted": counts["unreprinted"], "files_scanned": len(files) + len(held),
              "bytes_scanned": sum(size.values()), "public_files": skipped,
              **({"hold": str(hold), "release": str(release), "held": len(held), "released": released}
                 if hold is not None else {}),
              "excused": [{"entry": i, "witness": allow[i]["witness"], "pieces": len(x), "records": sorted(w)}
                          for i, (x, w) in sorted(used.items())],
              "found": found, "dry": dry, "verdict": "refused" if found else "clean"}
    return report, {**report, "found": explained}


def unpack(tarball: Path, dest: Path) -> Path:
    """A gate's fetch unpacked into `dest` (0700) for the guard alone."""
    dest.mkdir(mode=0o700, parents=True)
    with tarfile.open(tarball) as t:
        t.extractall(dest, filter="data")
    return dest


def _ended(signum, _frame):
    raise Ended(signal.Signals(signum).name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("quarantine", type=Path)
    ap.add_argument("--gates", type=Path, action="append", default=[])
    ap.add_argument("--hold", type=Path)
    ap.add_argument("--release-to", type=Path)
    ap.add_argument("--budget-s", type=float, default=BUDGET_S)
    ap.add_argument("--public", action="append", metavar="GLOB")
    ap.add_argument("--declared", type=Path, default=DECLARED)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    run, q = a.run.resolve(), a.quarantine.resolve()
    hold = a.hold.resolve() if a.hold else None
    release = (a.release_to or a.run / "out").resolve() if hold else None

    def refuse(why: str) -> int:
        print(f"leak_guard: refused: {why}", file=sys.stderr)
        return 2

    if not run.is_dir():
        return refuse(f"no run dir {run}")
    if q == run or q.is_relative_to(run):
        return refuse(f"the quarantine {q} lies inside the run dir {run}, which custody publishes")
    if missing := [str(g) for g in a.gates if not g.exists()]:
        return refuse(f"no gates directory {missing}")
    if hold is not None:
        if not hold.is_dir():
            return refuse(f"no hold {hold}")
        if hold.is_relative_to(run) or hold.is_relative_to(release) or release.is_relative_to(hold):
            return refuse(f"the hold {hold} lies inside the run dir {run} or {release}, or {release} inside it: "
                          "custody would take what it holds before the guard read it")
    names, allow = load_declared(a.declared)
    q.mkdir(parents=True, exist_ok=True)
    q.chmod(0o700)
    was = signal.signal(signal.SIGTERM, _ended)
    unpacked = []
    try:
        budget = Budget(a.budget_s)
        gates = []
        for i, g in enumerate(a.gates):
            if g.is_dir():
                gates.append(g.resolve())
            else:
                unpacked.append(unpack(g, q / f"gates-{os.getpid()}-{i}"))
                gates.append(unpacked[-1])
        report, held = guard(run, q, gates, names, allow, a.dry, hold, release, budget, a.public)
    except Ended as e:
        print(f"leak_guard: {e} ended the guard: nothing more released"
              + (f", what is left in {hold} stays out of the record" if hold else ""), file=sys.stderr, flush=True)
        return 143
    except (Refused, OSError, tarfile.TarError, MemoryError) as e:
        return refuse(f"{type(e).__name__}: {e}" + (f": nothing released, {hold} stays out of the record" if hold else ""))
    finally:
        signal.signal(signal.SIGTERM, was)
        if unpacked:
            subprocess.run(["rm", "-rf", "--", *map(str, unpacked)], check=False)
    (q / "leak-guard.json").write_text(json.dumps(held, indent=1) + "\n")
    if a.report is not None:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=1) + "\n")
    moved = "found (--dry)" if a.dry else f"moved to {q}/files"
    print(f"leak guard: {report['verdict']}: {report['files_scanned']} files ({report['bytes_scanned']} bytes) read, "
          f"{report['public_files']} of them public, for {report['pieces']} pieces of {len(report['gate_runs'])} gate "
          f"runs, {sum(e['pieces'] for e in report['excused'])} excused by the allowlist; {len(report['found'])} {moved}"
          + (f"; {report['released']} of {report['held']} held released to {release}" if hold else "")
          + "".join(f"\n  {d['path']}: {'; '.join(d['why'])}" for d in report["found"]), flush=True)
    return 3 if report["found"] else 0


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
