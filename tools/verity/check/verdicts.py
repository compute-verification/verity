"""`verdicts`: the passes `check`'s steps store, carried between machines as one pack.

    python tools/verity/check/verdicts.py list                                  each cache and how many verdicts it holds
    python tools/verity/check/verdicts.py snapshot OUT.json                     every verdict entry here, by path and sha256
    python tools/verity/check/verdicts.py export OUT.tar.gz [--since SNAP.json] the entries (those new or changed since SNAP) as a pack
    python tools/verity/check/verdicts.py import PACK.tar.gz                    add a pack's entries here, never replacing one
    python tools/verity/check/verdicts.py combine OUT.tar.gz PACK.tar.gz...     one pack of several, the first's entry where two hold one

A verdict is a pass that one step stored under a key of everything it read: a suite's pass and its tests' (`suites.py`, under `$VERITY_TEST_CACHE`),
and under `$CHECK_CACHE` a circuit-check target's report, a Lean package's audit (`lean_audit.py`) and the agreement's pass
(`check.py`).  A step looks an entry up by the key it computes from this machine's own inputs, so a pack can answer only for inputs
identical to the ones that earned it.  It is trusted as far as its source: `check.py --record` gathers packs only from recorded
runs (`gather_verdicts`).
Never packed, since they aren't verdicts: `lean-deps/` (fetched dependencies), `lean-builds/` (the Lean audit's kept builds,
`lean_audit.WarmBuilds`, whose `ir/*.setup.json` run to 9 MB each and grew a combined pack past 15 GB), `lean-artifacts/`
(Lake's cache of built modules, `lean_audit.LakeArtifacts`, whose `outputs/` entries name files no pack carries), the trees steps build in, not even walked since a
tree holds a package's dependencies, some 170k files (`lean_changed.py`'s `lean-changed/` for check's early Lean step and
`lean-records/` for the Lean fast path, and `lean_audit.py`'s `lean-audit-scratch-*/`, which another slot may remove while this one
reads it), the suites' `logs/` and `durations.json`, and any file that isn't a finished `.json` entry. An entry that goes away while
it is read (`suites.py` keeps only a module's newest passes) is left out.

Each root keeps a hidden index, `.verdicts-index.json`, memoizing its entries' sha256 by (size, mtime_ns), so `snapshot` and a
check's closing `export` stat the cache instead of re-reading it (hundreds of GB once grown); only new or changed entries are
read.  Hidden files are never listed and never packed, and a missing or garbled index only costs the re-read.

A pack is a tar of regular files under `check/` and `tests/`, gzipped, or xz-compressed when `export_paths` writes a recorded
check's reused passes (`check.Reused`), whose circuit-check entries share most of their bytes. `import` refuses a pack holding anything else (a link, an
absolute or `..` path, a device, another prefix, a hidden file), and it never overwrites an entry that exists here. A member in a
part of the caches that is never packed is left out, not refused, so a pack written before that part was is still taken.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import lzma
import os
import sys
import tarfile
from pathlib import Path

SKIP = {"check": {"lean-deps", "lean-builds", "lean-changed", "lean-records", "lean-artifacts"}, "tests": {"logs", "durations.json"}}
SCRATCH = {"check": ("lean-audit-scratch-",), "tests": ()}


def _skipped(prefix: str, top: str) -> bool:
    return top in SKIP[prefix] or top.startswith(SCRATCH[prefix])


def roots() -> dict[str, Path]:
    """The caches, where `check.py` (`cache_dir()`) and `suites.py` (`CACHE`) keep them; `tests/test_verdicts.py` pins the two."""
    env = os.environ
    home = Path.home()
    return {"check": Path(env.get("CHECK_CACHE") or home / ".cache" / "verity-check"),
            "tests": Path(env.get("VERITY_TEST_CACHE") or Path(env.get("XDG_CACHE_HOME") or home / ".cache") / "verity" / "tests")}


def entries() -> dict[str, Path]:
    """`{pack path: file}` for every verdict entry here, a pack path being `check/...` or `tests/...`."""
    out = {}
    for prefix, base in roots().items():
        if not base.is_dir():
            continue
        for d, dirs, names in os.walk(base):
            top = Path(d) == base
            dirs[:] = [x for x in dirs if not x.startswith(".") and not (top and _skipped(prefix, x))]
            for name in names:
                p = Path(d) / name
                rel = p.relative_to(base)
                if (name.endswith(".json") and not name.startswith(".") and not _skipped(prefix, rel.parts[0]) and p.is_file()
                        and not p.is_symlink()):
                    out[f"{prefix}/{rel.as_posix()}"] = p
    return out


def _read(p: Path) -> bytes | None:
    try:
        return p.read_bytes()
    except FileNotFoundError:
        return None


#: each root's memo of its entries' digests, `{rel path: [size, mtime_ns, sha256]}`: hidden, so `entries()` never lists it and
#: no pack carries it
INDEX = ".verdicts-index.json"


def _index_load(base: Path) -> dict:
    try:
        d = json.loads((base / INDEX).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _digests(found: dict[str, Path] | None = None) -> dict[str, str]:
    """`{pack path: sha256}` of every entry, reading only the files the root's index doesn't know by (size, mtime_ns).  The
    caches run to hundreds of GB, and re-reading every entry at both ends of a check held a finished run for an hour of disk
    reads once another job had evicted the page cache (14 Oct, r20261014-032950-9e49).  A stale or garbled index only costs the
    re-read; an entry rewritten in place with its size and mtime_ns unchanged is the one change it would miss, and every writer
    replaces entries atomically."""
    found = entries() if found is None else found
    out: dict[str, str] = {}
    for prefix, base in roots().items():
        index = _index_load(base)
        fresh: dict[str, list] = {}
        dirty = False
        for k, p in found.items():
            if not k.startswith(prefix + "/"):
                continue
            rel = k.split("/", 1)[1]
            try:
                st = p.stat()
            except OSError:
                continue
            known = index.get(rel)
            if isinstance(known, list) and len(known) == 3 and known[0] == st.st_size and known[1] == st.st_mtime_ns:
                sha = known[2]
            else:
                data = _read(p)
                if data is None:
                    continue
                sha = hashlib.sha256(data).hexdigest()
                dirty = True
            fresh[rel] = [st.st_size, st.st_mtime_ns, sha]
            out[k] = sha
        if dirty or set(fresh) != set(index):
            try:                                   # concurrent slots race here; a lost update only costs a later re-read
                tmp = base / f"{INDEX}.{os.getpid()}.tmp"
                tmp.write_text(json.dumps(fresh))
                os.replace(tmp, base / INDEX)
            except OSError:
                pass
    return out


def snapshot() -> dict[str, str]:
    return _digests()


#: an xz pack's filter: a dictionary spanning many entries, since one tree's circuit-check entries share most of what they record
#: (300 of them, 68 MB: 11 MB gzipped, 0.35 MB this way, in less time; vy-nebius-cpu-1, 19 Oct)
XZ_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 1, "dict_size": 64 << 20}]


def _compressed(raw, xz: bool):
    # the same entries always make the same pack: no file name or mtime in the gzip header (xz has neither)
    return lzma.LZMAFile(raw, "wb", format=lzma.FORMAT_XZ, filters=XZ_FILTERS) if xz else \
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)


def _write(out: Path, items, xz: bool = False) -> int:
    """Write ``items`` ((pack path, bytes or a file), in order) as a pack at `out`, gzipped, or xz-compressed (``xz``); the number
    written."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    n = 0
    with open(tmp, "wb") as raw, _compressed(raw, xz) as z, tarfile.open(fileobj=z, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for k, data in items:
            info = tarfile.TarInfo(k)
            info.mode, info.mtime = 0o644, 0
            if isinstance(data, Path):
                try:
                    f = open(data, "rb")
                except FileNotFoundError:
                    continue
                with f:
                    info.size = os.fstat(f.fileno()).st_size
                    tar.addfile(info, f)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            n += 1
    os.replace(tmp, out)
    return n


def changed(since: dict[str, str]) -> dict[str, tuple[str, Path]]:
    """`{pack path: (sha256, file)}` of the entries new or changed since `since` (a `snapshot`)."""
    found = entries()
    return {k: (d, found[k]) for k, d in _digests(found).items() if since.get(k) != d}


def export(out: Path, since: dict[str, str] | None = None, new: dict[str, tuple[str, Path]] | None = None) -> int:
    """Write the entries new or changed since `since` (all when None; ``new``, `changed(since)`, when given) as a pack at `out`;
    the number written.  With `since`, only the new or changed entries are read (`_digests`), so a check's closing export costs
    what the run stored, not the whole cache."""
    if since is None and new is None:
        found = entries()

        def items():
            for k, p in sorted(found.items()):
                data = _read(p)
                if data is not None:
                    yield k, data
    else:
        new = changed(since) if new is None else new

        def items():
            for k in sorted(new):
                data = _read(new[k][1])
                if data is not None:
                    yield k, data
    return _write(out, items())


def export_paths(out: Path, paths, xz: bool = False) -> int:
    """Write the entries here that ``paths`` (pack paths) name as a pack at `out`, in path order, each once; one that isn't here
    (or is no verdict entry) is left out.  The number written."""
    base = roots()
    files = {}
    for k in paths:
        prefix, _, rel = k.partition("/")
        parts = Path(rel).parts
        if prefix in base and k.endswith(".json") and parts and ".." not in parts and not any(x.startswith(".") for x in parts) \
                and not _skipped(prefix, parts[0]) and (base[prefix] / rel).is_file():
            files[k] = base[prefix] / rel
    return _write(out, sorted(files.items()), xz)


class BadPack(Exception):
    pass


def _members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    out = []
    for m in tar.getmembers():
        parts = Path(m.name).parts
        if not m.isfile() or m.name.startswith("/") or ".." in parts or len(parts) < 2 or parts[0] not in SKIP or not m.name.endswith(".json"):
            raise BadPack(f"member {m.name!r} is not a verdict entry (a regular .json file under check/ or tests/)")
        if any(x.startswith(".") for x in parts):
            raise BadPack(f"member {m.name!r} is a hidden file, which no pack holds")
        if not _skipped(parts[0], parts[1]):
            out.append(m)
    return out


def refusal(pack: Path) -> str | None:
    """Why `import` and `combine` would refuse the pack (not a gzipped or xz-compressed tar, or a member that isn't a verdict entry); None when
    they take it."""
    try:
        with tarfile.open(pack, "r:*") as tar:
            _members(tar)
    except (BadPack, tarfile.TarError, OSError, EOFError) as e:
        return f"{type(e).__name__}: {e}"
    return None


def import_(pack: Path, only: str | None = None) -> tuple[int, int]:
    """Add the pack's entries that don't exist here, with ``only`` just those whose pack path starts with it; (added, already
    here).  Nothing is written when the pack is refused."""
    base = roots()
    added = kept = 0
    with tarfile.open(pack, "r:*") as tar:
        members = _members(tar)
        for m in members:
            if only is not None and not m.name.startswith(only):
                continue
            prefix, rel = m.name.split("/", 1)
            dst = base[prefix] / rel
            if dst.exists():
                kept += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            # this process's own name: two checks on one machine can import the same entry at once
            tmp = dst.with_name(f".{dst.name}.{os.getpid()}.import")
            tmp.write_bytes(tar.extractfile(m).read())
            os.replace(tmp, dst)
            added += 1
    return added, kept


def combine(packs: list[Path], out: Path) -> int:
    """Write one pack at `out` holding every entry of ``packs``, the first pack's where two hold one path (an entry's path is its
    key, so both say the same); the number written.  A pack `import` would refuse is refused here too, and nothing is written.
    Each entry is spilled to a file beside `out` as it is read and streamed back in path order, so memory holds one entry at a
    time: the newest recorded checks' packs, held whole, outgrew a 15 GB VM (`check.py --record`'s `gather_verdicts`)."""
    import shutil
    import tempfile

    out.parent.mkdir(parents=True, exist_ok=True)
    spill = Path(tempfile.mkdtemp(prefix=".combine-", dir=out.parent))
    try:
        found: dict[str, Path] = {}
        for pack in packs:
            with tarfile.open(pack, "r:*") as tar:
                for m in _members(tar):
                    if m.name not in found:
                        p = found[m.name] = spill / str(len(found))
                        with tar.extractfile(m) as src, open(p, "wb") as dst:
                            shutil.copyfileobj(src, dst)
        return _write(out, sorted(found.items()))
    finally:
        shutil.rmtree(spill, ignore_errors=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="verdicts", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("snapshot")
    s.add_argument("out", type=Path)
    e = sub.add_parser("export")
    e.add_argument("out", type=Path)
    e.add_argument("--since", type=Path)
    i = sub.add_parser("import")
    i.add_argument("pack", type=Path)
    c = sub.add_parser("combine")
    c.add_argument("out", type=Path)
    c.add_argument("packs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        found = entries()
        for prefix, base in roots().items():
            print(f"{prefix:6} {sum(k.startswith(prefix + '/') for k in found):6}  {base}")
        return 0
    if a.cmd == "snapshot":
        a.out.write_text(json.dumps(snapshot(), indent=1, sort_keys=True) + "\n")
        return 0
    if a.cmd == "export":
        n = export(a.out, json.loads(a.since.read_text()) if a.since else None)
        print(f"verdicts: {n} entries -> {a.out}")
        return 0
    if a.cmd == "combine":
        try:
            n = combine(a.packs, a.out)
        except (BadPack, tarfile.TarError) as err:
            print(f"verdicts: refused: {err}", file=sys.stderr)
            return 1
        print(f"verdicts: {n} entries from {len(a.packs)} packs -> {a.out}")
        return 0
    try:
        added, kept = import_(a.pack)
    except (BadPack, tarfile.TarError) as err:
        print(f"verdicts: refused {a.pack}: {err}", file=sys.stderr)
        return 1
    print(f"verdicts: {added} added, {kept} already here, from {a.pack}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
