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
Never packed, since they aren't verdicts: `lean-deps/` (fetched dependencies), `lean-artifacts/` (Lake's cache of built
modules, `lean_audit.LakeArtifacts`, whose `outputs/` entries name files no pack carries), the trees steps build in, not even walked since a
tree holds a package's dependencies, some 170k files (`lean_changed.py`'s `lean-changed/` for check's early Lean step and
`lean-records/` for the Lean fast path, and `lean_audit.py`'s `lean-audit-scratch-*/`, which another slot may remove while this one
reads it), the suites' `logs/` and `durations.json`, and any file that isn't a finished `.json` entry. An entry that goes away while
it is read (`suites.py` keeps only a module's newest passes) is left out.

A pack is a gzipped tar of regular files under `check/` and `tests/`. `import` refuses a pack holding anything else (a link, an
absolute or `..` path, a device, another prefix), and it never overwrites an entry that exists here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

SKIP = {"check": {"lean-deps", "lean-changed", "lean-records", "lean-artifacts"}, "tests": {"logs", "durations.json"}}
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


def snapshot() -> dict[str, str]:
    return {k: hashlib.sha256(data).hexdigest() for k, p in entries().items() if (data := _read(p)) is not None}


def _write(out: Path, items) -> int:
    """Write ``items`` ((pack path, bytes), in order) as a pack at `out`; the number written."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    n = 0
    # the same entries always make the same pack: no mtimes in the tar, and no file name or mtime in the gzip header
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for k, data in items:
            info = tarfile.TarInfo(k)
            info.mode, info.mtime = 0o644, 0
            if isinstance(data, Path):
                info.size = data.stat().st_size
                with open(data, "rb") as f:
                    tar.addfile(info, f)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            n += 1
    os.replace(tmp, out)
    return n


def export(out: Path, since: dict[str, str] | None = None) -> int:
    """Write the entries new or changed since `since` (all when None) as a pack at `out`; the number written."""
    def items():
        for k, p in sorted(entries().items()):
            data = _read(p)
            if data is not None and (since is None or since.get(k) != hashlib.sha256(data).hexdigest()):
                yield k, data
    return _write(out, items())


class BadPack(Exception):
    pass


def _members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    out = []
    for m in tar.getmembers():
        parts = Path(m.name).parts
        if not m.isfile() or m.name.startswith("/") or ".." in parts or len(parts) < 2 or parts[0] not in SKIP or not m.name.endswith(".json"):
            raise BadPack(f"member {m.name!r} is not a verdict entry (a regular .json file under check/ or tests/)")
        if _skipped(parts[0], parts[1]) or any(x.startswith(".") for x in parts):
            raise BadPack(f"member {m.name!r} is in a part of the caches that holds no verdicts")
        out.append(m)
    return out


def refusal(pack: Path) -> str | None:
    """Why `import` and `combine` would refuse the pack (not a gzipped tar, or a member that isn't a verdict entry); None when
    they take it."""
    try:
        with tarfile.open(pack, "r:gz") as tar:
            _members(tar)
    except (BadPack, tarfile.TarError, OSError, EOFError) as e:
        return f"{type(e).__name__}: {e}"
    return None


def import_(pack: Path) -> tuple[int, int]:
    """Add the pack's entries that don't exist here; (added, already here).  Nothing is written when the pack is refused."""
    base = roots()
    added = kept = 0
    with tarfile.open(pack, "r:gz") as tar:
        members = _members(tar)
        for m in members:
            prefix, rel = m.name.split("/", 1)
            dst = base[prefix] / rel
            if dst.exists():
                kept += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.import")
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
            with tarfile.open(pack, "r:gz") as tar:
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
