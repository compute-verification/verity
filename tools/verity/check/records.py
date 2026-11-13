"""`records`: circuit-check's binding records, carried between checks through the store, each once under its name.

A binding's record is an entry of circuit-check's cache (`circuit_check.cache`, the step's `--cache`: `check.py`'s
`cache_dir()/circuit-check`, which is the suite's records directory too) whose key's request is a catalog unit
(`circuit_check.units`' `REQUEST`), with the read records it needs (`NEEDS`, followed through each record).  A record's name is its
path in that cache: `<version>/<its key's digest>/<sha512 of its bytes>.json`, or `<version>/reads/<part>/<sha512 of its bytes>.json`
for a read record, so two trees' records under one key have two names and a name says which bytes it holds.  The store's remote
holds each once, under `PREFIX` and its name.

A recorded check that runs circuit-check's step or the suite (`check.builds_records`) does two things with them:

- Before its steps, it fetches the records that the verdict packs it imported name (`named_in`) and its cache lacks (`fetch`), in
  parallel, at most `FETCH_BYTES` of them in at most `FETCH_S`.  A record is held to its name: one the store lacks, whose bytes
  aren't its name's, or past either bound is left out, as is an entry any record it needs is missing for, so circuit-check
  builds the binding.  A record is a cache, never a verdict: nothing a fetch does fails the check or enters a step's key.
- Each time it writes its verdict pack (`check.Packs`), it names the binding records its circuit-check runs took or built, as
  their manifests in its out dir list them (`run_records`: the step's `--records-manifest`, a session's `SESSION_ENV`), or,
  from a circuit-check that writes none, those whose files changed since its steps began (`Publisher`), with what they need;
  puts each one the store doesn't hold yet (`ON_STORE` remembers, for the node, what is there already, so nothing is uploaded
  twice); lists every one on the store in the pack's `MANIFEST`; and leaves those out of the pack, which then carries only
  records the store lacks.
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import stat
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

#: where the store's remote holds the records, each under its name
PREFIX = "circuit-records/"
#: the verdict pack's member listing the records the check names, `{"records": {name: bytes}}`: under a part of the check cache
#: `verdicts.SKIP` leaves out, so no import takes it for a verdict
MANIFEST = "check/circuit-records/manifest.json"
#: the request of a binding's record (`circuit_check.units.REQUEST`)
REQUEST = "catalog-unit"
#: the records' directory in a version's (`circuit_check.cache.READS`)
READS = "reads"
#: a record's name: a version, a key's directory or a read record's part, and the sha512 of its bytes
NAME = re.compile(r"[A-Za-z0-9._-]+/(?:[0-9a-f]{32}|reads/[A-Za-z0-9_-]+)/[0-9a-f]{128}\.json")
#: the node's list of records the store holds, a name a line, beside the check's caches (hidden: no pack carries it)
ON_STORE = ".circuit-records-on-store"
#: the most bytes of records one check fetches before its steps, and the most seconds it spends on them (beside each request's own
#: timeout): what is left is built, as it would be without the store
FETCH_BYTES = 512 << 20
FETCH_S = 300.0
#: the most seconds one pack write spends putting records on the store; the rest stay in the pack until the next write puts them
PUBLISH_S = 300.0
#: the requests in flight at once, fetching or putting
JOBS = 32
#: what circuit-check writes in a check's out dir once its catalog is made, one file a run (the step, each of its shards, and a
#: session, through `SESSION_ENV`): every binding record the catalog took or built (`records[].record`) and the read records
#: they reach (`needs`)
RUN_MANIFESTS = "circuit-records.*.json"
RUN_SCHEMA = "circuit-check-records/v1"
SESSION_ENV = "CIRCUIT_CHECK_RECORDS_MANIFEST"


def run_records(out: Path) -> list[str] | None:
    """The binding records the manifests circuit-check wrote in ``out`` list (`RUN_MANIFESTS`); None when it wrote none."""
    files = sorted(out.glob(RUN_MANIFESTS))
    if not files:
        return None
    got: list[str] = []
    for f in files:
        try:
            doc = json.loads(f.read_text())
        except (OSError, ValueError) as e:
            print(f"check: circuit records: {f.name} can't be read ({type(e).__name__}); none it lists is named", flush=True)
            continue
        if not isinstance(doc, dict) or doc.get("schema") != RUN_SCHEMA or not isinstance(doc.get("records"), list):
            print(f"check: circuit records: {f.name} is no {RUN_SCHEMA} manifest; none it lists is named", flush=True)
            continue
        got += [r["record"] for r in doc["records"] if isinstance(r, dict) and isinstance(r.get("record"), str)
                and named(r["record"])]
    return got


def named(rel: str) -> bool:
    return NAME.fullmatch(rel) is not None


def intact(rel: str, data: bytes) -> bool:
    """Whether ``data`` are the bytes the record named ``rel`` holds."""
    return hashlib.sha512(data).hexdigest() == posixpath.basename(rel)[:-len(".json")]


def needs(rel: str, data: bytes) -> list[str]:
    """The names of the records the record named ``rel``, of bytes ``data``, needs; ValueError when its bytes aren't JSON or a
    need leaves its version."""
    doc = json.loads(data)
    got = doc.get("needs", []) if isinstance(doc, dict) else []
    if not isinstance(got, list):
        raise ValueError(f"{rel}: its needs are no list")
    out, top = [], rel.split("/", 1)[0] + "/"
    for n in got:
        p = posixpath.normpath(posixpath.join(posixpath.dirname(rel), n)) if isinstance(n, str) else ""
        if not p.startswith(top):
            raise ValueError(f"{rel}: {n!r} names no record of its version")
        out.append(p)
    return out


def binding(data: bytes) -> bool:
    """Whether the entry of bytes ``data`` is a binding's record."""
    if f'"{REQUEST}"'.encode() not in data:
        return False
    try:
        target = json.loads(data)["key"]["request"]["target"]
    except (ValueError, KeyError, TypeError):
        return False
    return isinstance(target, list) and target[:1] == [REQUEST]


def named_in(pack: Path) -> dict[str, int]:
    """`{name: bytes}` of the records the verdict pack ``pack`` lists (`MANIFEST`); empty when it lists none or its list can't be read."""
    try:
        with tarfile.open(pack, "r:*") as tar:
            if MANIFEST not in tar.getnames():
                return {}
            doc = json.loads(tar.extractfile(MANIFEST).read())
    except (OSError, ValueError, tarfile.TarError, EOFError) as e:
        print(f"check: circuit records: {pack.name}'s list can't be read ({type(e).__name__}: {e}); none it names is fetched",
              flush=True)
        return {}
    got = doc.get("records") if isinstance(doc, dict) else None
    return {k: v for k, v in got.items() if isinstance(k, str) and named(k) and type(v) is int and v >= 0} \
        if isinstance(got, dict) else {}


def _on_store(base: Path) -> set[str]:
    if not (p := base / ON_STORE).is_file():
        return set()
    try:
        return set(p.read_text().split())
    except OSError as e:
        print(f"check: circuit records: {p} can't be read ({type(e).__name__}); the store is asked for each", flush=True)
        return set()


def _remember(base: Path, names) -> None:
    if names := sorted(names):
        try:                                   # a line a write, appended: slots on one node append at once
            with open(base / ON_STORE, "a") as f:
                f.write("".join(f"{n}\n" for n in names))
        except OSError as e:
            print(f"check: circuit records: {base / ON_STORE} can't be written ({type(e).__name__}); the next check asks the store "
                  "for these again", flush=True)


def _listed(d) -> list[os.DirEntry]:
    """``d``'s entries by name; none once it is gone (circuit-check's expiry may remove a key's directory meanwhile)."""
    if not os.path.isdir(d):
        return []
    try:
        with os.scandir(d) as it:
            return sorted(it, key=lambda e: e.name)
    except OSError as e:
        print(f"check: circuit records: {d} can't be listed ({type(e).__name__}); none in it is named", flush=True)
        return []


def _stat(p) -> os.stat_result | str:
    """The file's stat, not following a link, else why not."""
    try:
        return os.lstat(p)
    except OSError as e:
        return f"gone ({type(e).__name__})"


def _place(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.stem}.{os.getpid()}.{threading.get_ident()}.fetch")
    tmp.write_bytes(data)
    os.replace(tmp, p)


def getter(remote) -> Callable[[str], bytes]:
    """A `get(key) -> bytes` from a `Remote`."""
    def get(key: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="check-records-") as d:
            remote.get(key, p := Path(d) / "record")
            return p.read_bytes()
    return get


def putter(remote) -> Callable[[str, Path], str]:
    """A `put(key, file) -> action` from a `Remote` (`created`, `skipped` when the store held it, `exists`)."""
    return lambda key, path: remote.put(key, path)["action"]


def fetch(names: dict[str, int], d: Path, get: Callable[[str], bytes], max_bytes: int = FETCH_BYTES, max_s: float = FETCH_S,
          jobs: int = JOBS, clock: Callable[[], float] = time.monotonic, base: Path | None = None) -> dict:
    """Fetch into ``d`` each record ``names`` (`{name: bytes}`) lists that ``d`` lacks, with ``get``, in ``jobs`` threads, at most
    ``max_bytes`` of them (by the sizes listed) and none once ``max_s`` have gone by.  A record is placed only once its bytes are its
    name's, and an entry only once every record it needs is here; the read records go in first, each whole (a rename), so a
    circuit-check reading ``d`` meanwhile finds each record whole or not at all.  ``base`` (the check cache) remembers what came
    from the store (`ON_STORE`).  The counts: named, here already, fetched and its bytes, missing on the store, corrupt, left out
    for a missing need, past the bytes bound and past the time bound."""
    t0 = clock()
    out = {"named": len(names), "here": 0, "fetched": 0, "bytes": 0, "missing": 0, "corrupt": 0, "incomplete": 0, "over": 0,
           "late": 0}
    todo, room = [], max_bytes
    for rel in sorted(names, key=lambda r: (READS not in r.split("/")[1:2], r)):
        if (d / rel).is_file():
            out["here"] += 1
        elif names[rel] > room:
            out["over"] += 1
        else:
            todo.append(rel)
            room -= names[rel]
    got: dict[str, bytes] = {}
    lock = threading.Lock()

    def one(rel: str) -> None:
        if clock() - t0 > max_s:
            why = "late"
        else:
            try:
                data = get(PREFIX + rel)
            except Exception:  # noqa: BLE001  (a record the store can't give is built)
                why = "missing"
            else:
                why = None if len(data) == names[rel] and intact(rel, data) else "corrupt"
                if why is None:
                    try:
                        needs(rel, data)
                    except ValueError:
                        why = "corrupt"
        with lock:
            if why is None:
                got[rel] = data
            else:
                out[why] += 1

    with ThreadPoolExecutor(max_workers=max(1, jobs), thread_name_prefix="records-fetch") as pool:
        list(pool.map(one, todo))
    placed: set[str] = set()

    def place(rel: str, seen: frozenset = frozenset()) -> bool:
        if rel in placed or (rel not in got and (d / rel).is_file()):
            return True
        if rel not in got or rel in seen:
            return False
        if not all(place(n, seen | {rel}) for n in needs(rel, got[rel])):
            return False
        _place(d / rel, got[rel])
        placed.add(rel)
        out["fetched"] += 1
        out["bytes"] += len(got[rel])
        return True

    for rel in sorted(got, key=lambda r: (READS not in r.split("/")[1:2], r)):
        if not place(rel):
            out["incomplete"] += 1
    if base is not None:
        _remember(base, placed)
    out["seconds"] = round(clock() - t0, 1)
    return out


def fetch_line(got: dict) -> str:
    left = {k: got[k] for k in ("missing", "corrupt", "incomplete", "over", "late") if got.get(k)}
    return (f"check: circuit records: {got['named']} named by the packs imported, {got['here']} here already, {got['fetched']} "
            f"fetched ({got['bytes'] / 2**20:.1f} MiB) in {got['seconds']:.0f} s"
            + (f"; left to circuit-check to build: {', '.join(f'{v} {k}' for k, v in left.items())}" if left else ""))


class Publisher:
    """The records one check names in its verdict pack: each binding record in ``d`` (circuit-check's cache, of pack path ``at``)
    that the manifests circuit-check wrote in ``out`` list (`run_records`), else, while it wrote none, each whose file changed at
    or after ``since_ns`` (the start of its steps); each whose bytes are its name's, with every record it needs, put on the store
    with ``put`` unless ``base``'s `ON_STORE` lists it.  An entry a record it needs is missing or corrupt for is named by none: it
    can't count anywhere.  A file is read once while its size and mtime stay."""

    def __init__(self, d: Path, at: str, since_ns: int, put: Callable[[str, Path], str] | None, base: Path,
                 clock: Callable[[], float] = time.monotonic, jobs: int = JOBS, out: Path | None = None):
        self.d, self.at, self.since_ns, self.put, self.base, self.clock, self.jobs = d, at, since_ns, put, base, clock, jobs
        self.out = out
        self.on_store = _on_store(base)
        self.read: dict[str, tuple] = {}
        self.lock = threading.Lock()
        self.stats = {"put": 0, "already": 0, "failed": 0, "late": 0}

    def _file(self, rel: str, st: os.stat_result) -> tuple[bool, list[str] | str]:
        """(whether it is a binding's record, the records it needs, or why it counts nowhere)."""
        k = (st.st_size, st.st_mtime_ns)
        if (m := self.read.get(rel)) is None or m[:2] != k:
            try:
                data = (self.d / rel).read_bytes()
            except OSError as e:
                return False, f"unreadable ({type(e).__name__})"
            if not intact(rel, data):
                need = "not the bytes its name digests"
            else:
                try:
                    need = needs(rel, data)
                except ValueError as e:
                    need = f"no record ({e})"
            m = (*k, binding(data), need)
            self.read[rel] = m
        return m[2], m[3]

    def _closure(self, rel: str, sizes: dict[str, int], memo: dict[str, set[str] | None]) -> set[str] | None:
        if rel in memo:
            return memo[rel]
        memo[rel] = None
        if not named(rel) or isinstance(st := _stat(self.d / rel), str) or isinstance(need := self._file(rel, st)[1], str):
            return None
        out = {rel}
        for n in need:
            if (c := self._closure(n, sizes, memo)) is None:
                return None
            out |= c
        sizes[rel] = st.st_size
        memo[rel] = out
        return out

    def _changed(self) -> list[str]:
        roots: list[str] = []
        for version in _listed(self.d):
            if version.name.startswith(".") or not version.is_dir(follow_symlinks=False):
                continue
            for key in _listed(version.path):
                if key.name == READS or key.name.startswith(".") or not key.is_dir(follow_symlinks=False):
                    continue
                for f in _listed(key.path):
                    rel = f"{version.name}/{key.name}/{f.name}"
                    if named(rel) and not isinstance(st := _stat(f.path), str) and stat.S_ISREG(st.st_mode) \
                            and st.st_mtime_ns >= self.since_ns and self._file(rel, st)[0]:
                        roots.append(rel)
        return roots

    def produced(self) -> dict[str, int]:
        """`{name: bytes}` of the records named now."""
        roots = listed if self.out is not None and (listed := run_records(self.out)) is not None else self._changed()
        sizes: dict[str, int] = {}
        memo: dict[str, set[str] | None] = {}
        for rel in roots:
            self._closure(rel, sizes, memo)
        return {r: sizes[r] for r in sorted({x for rel in roots for x in memo.get(rel) or ()})}

    def publish(self, names: dict[str, int], max_s: float = PUBLISH_S) -> set[str]:
        """Put on the store each of ``names`` it may not hold yet, the read records before the entries, none once ``max_s`` have
        gone by; the names the store holds now."""
        held = {r for r in names if r in self.on_store}
        if self.put is None:
            return held
        t0, done = self.clock(), set()

        def one(rel: str) -> None:
            if self.clock() - t0 > max_s:
                with self.lock:
                    self.stats["late"] += 1
                return
            try:
                action = self.put(PREFIX + rel, self.d / rel)
            except Exception:  # noqa: BLE001  (one the store didn't take stays in the pack)
                with self.lock:
                    self.stats["failed"] += 1
                return
            with self.lock:
                self.stats["put" if action == "created" else "already"] += 1
                done.add(rel)

        left = sorted(set(names) - held)
        with ThreadPoolExecutor(max_workers=max(1, self.jobs), thread_name_prefix="records-put") as pool:
            for wave in ([r for r in left if r.split("/")[1] == READS], [r for r in left if r.split("/")[1] != READS]):
                list(pool.map(one, wave))
        _remember(self.base, done)
        self.on_store |= done
        return held | done

    def pack(self, publish: bool = True) -> tuple[set[str], tuple[str, bytes] | None]:
        """The pack paths the pack leaves out and its `MANIFEST` member (None when it names none): every record named whose
        closure the store holds, after putting what it lacks (``publish``)."""
        names = self.produced()
        before = dict(self.stats)
        held = self.publish(names) if publish else {r for r in names if r in self.on_store}
        m = {k: self.stats[k] - before[k] for k in self.stats}
        if any(m.values()):
            print(f"check: circuit records: {len(names)} named, {m['put']} put on the store, {m['already']} held there already, "
                  f"{m['failed']} not taken, {m['late']} left past {PUBLISH_S:.0f} s (PUBLISH_S)", flush=True)
        memo: dict[str, set[str] | None] = {}
        listed = {x for r in names if r.split("/")[1] != READS and (c := self._closure(r, {}, memo)) is not None and c <= held
                  for x in c}
        if not listed:
            return set(), None
        doc = {"records": {r: names[r] for r in sorted(listed)}}
        return {self.at + r for r in listed}, (MANIFEST, json.dumps(doc, sort_keys=True, separators=(",", ":")).encode())
