"""Which finished entries of this machine's check cache `verdicts.entries` leaves out, and why: a measuring script, not part of check.

    python tools/verity/check/cache_probe.py [--no-store] [--target ID]      writes $OUT/cache-probe.json

It reads the caches a run on this machine sees (`verdicts.roots`), lists what is at the check cache's top (names, link status and
device numbers only), counts the entries `verdicts.entries` finds against a walk that follows links under the same skip rules, and
says for the ones left out which rule left them out.  Then it stores one circuit-check target (`F32Add_v1`) through circuit-check
itself, once into this machine's cache and once into a temporary cache whose top has this machine's shape (a link where it has a
link), and says whether `verdicts.changed` and a pack written from it hold the stored entry, and how long the snapshot took.
"""
from __future__ import annotations

import collections
import getpass
import json
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdicts

TARGET = "F32Add_v1"


def family(key: str) -> str:
    return "/".join(key.split("/")[:2]) if key.count("/") > 1 else key.split("/")[0] + "/(top)"


def where(p: Path) -> dict:
    d: dict = {"path": str(p), "link": p.is_symlink(), "dir": p.is_dir()}
    if p.is_symlink():
        d["target"], d["resolved"] = os.readlink(p), str(p.resolve())
    if p.exists():
        d["dev"], d["mount"] = p.stat().st_dev, os.path.ismount(p)
    return d


def followed() -> dict[str, Path]:
    """`verdicts.entries` with links followed: the same roots, the same skipped names, every finished `.json` under them."""
    out: dict[str, Path] = {}
    for prefix, base in verdicts.roots().items():
        if not base.is_dir():
            continue
        seen: set[str] = set()
        for d, dirs, names in os.walk(base, followlinks=True):
            real = os.path.realpath(d)
            if real in seen:
                dirs[:] = []
                continue
            seen.add(real)
            top = Path(d) == base
            dirs[:] = [x for x in dirs if not x.startswith(".") and not (top and verdicts._skipped(prefix, x))]
            for name in names:
                p = Path(d) / name
                rel = p.relative_to(base)
                if name.endswith(".json") and not name.startswith(".") and not verdicts._skipped(prefix, rel.parts[0]) and p.is_file():
                    out[f"{prefix}/{rel.as_posix()}"] = p
    return out


def why_left_out(key: str, p: Path) -> str:
    prefix, _, rel = key.partition("/")
    at = verdicts.roots()[prefix]
    for part in Path(rel).parts[:-1]:
        at = at / part
        if at.is_symlink():
            return f"under a linked directory: {prefix}/{at.relative_to(verdicts.roots()[prefix]).as_posix()} -> {os.readlink(at)}"
    return "the file is a link" if p.is_symlink() else "not a regular file" if not p.is_file() else "unexplained"


def census() -> dict:
    found, every = verdicts.entries(), followed()
    out_of = collections.Counter(family(k) for k in every)
    seen = collections.Counter(family(k) for k in found)
    missed = sorted(set(every) - set(found))
    reasons = collections.Counter(why_left_out(k, every[k]) for k in missed)
    newest = max((every[k].stat().st_mtime for k in missed), default=None)
    return {"entries": len(found), "with_links_followed": len(every), "left_out": len(missed),
            "by_family": {f: {"entries": seen.get(f, 0), "with_links_followed": n} for f, n in sorted(out_of.items())},
            "left_out_why": dict(reasons.most_common(8)), "left_out_bytes": sum(every[k].stat().st_size for k in missed),
            "left_out_newest_mtime": newest, "left_out_sample": missed[:3]}


def store(cache: Path) -> tuple[dict, dict]:
    """circuit-check's own store of one target into ``cache``, whether `verdicts.changed` sees it, what changed, and the pack
    `verdicts.export` writes of that (its members by family, the pack itself not kept)."""
    t0 = time.monotonic()
    before = verdicts.snapshot()
    t1 = time.monotonic()
    files = set(followed())
    r = subprocess.run([sys.executable, "-P", "-m", "circuit_check", TARGET, "--cache", str(cache), "-j", "1", "-q"],
                       capture_output=True, text=True)
    new = sorted(set(followed()) - files)
    t2 = time.monotonic()
    changed = verdicts.changed(before)
    t3 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cache-probe-pack-") as t:
        pack = Path(t) / "verdicts.tar.gz"
        n = verdicts.export(pack, new=changed)
        with tarfile.open(pack) as tar:
            names = tar.getnames()
        size = pack.stat().st_size
    return {"cache": str(cache), "target": TARGET, "rc": r.returncode, "tail": (r.stdout + r.stderr).strip().splitlines()[-3:],
            "snapshot_entries": len(before), "snapshot_s": round(t1 - t0, 1), "changed_s": round(t3 - t2, 1),
            "stored": new, "changed": len(changed), "stored_in_changed": [k for k in new if k in changed],
            "changed_by_family": dict(collections.Counter(family(k) for k in changed)),
            "pack_entries": n, "pack_bytes": size, "pack_by_family": dict(collections.Counter(family(k) for k in names)),
            "stored_in_pack": [k for k in new if k in names]}, changed


def replica(shape: dict[str, bool], top_link: bool) -> dict:
    """A temporary check cache whose top has ``shape`` (a name's directory a link where True), one target stored in it by
    circuit-check, and the pack `verdicts.export` writes of what changed."""
    with tempfile.TemporaryDirectory(prefix="cache-probe-") as t:
        root = Path(t) / "cache"
        (Path(t) / "other").mkdir()
        if top_link:
            (Path(t) / "real").mkdir()
            root.symlink_to(Path(t) / "real")
        else:
            root.mkdir()
        for name, link in shape.items():
            if link:
                (Path(t) / "other" / name).mkdir()
                (root / name).symlink_to(Path(t) / "other" / name)
            else:
                (root / name).mkdir()
        keep = {k: os.environ.get(k) for k in ("CHECK_CACHE", "VERITY_TEST_CACHE")}
        os.environ.update(CHECK_CACHE=str(root), VERITY_TEST_CACHE=str(Path(t) / "tests"))
        try:
            got, changed = store(root / "circuit-check")
            pack = Path(t) / "pack.tar.gz"
            n = verdicts.export(pack, new=changed)
            with tarfile.open(pack) as tar:
                names = tar.getnames()
            size = pack.stat().st_size
        finally:
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return {"shape": shape, "top_link": top_link, "rc": got["rc"], "tail": got["tail"], "stored": got["stored"],
                "stored_in_changed": got["stored_in_changed"], "pack_entries": n, "pack_bytes": size, "pack_members": names[:5]}


def main() -> int:
    global TARGET
    if "--target" in sys.argv:
        TARGET = sys.argv[sys.argv.index("--target") + 1]
    out = Path(os.environ.get("OUT") or ".")
    out.mkdir(parents=True, exist_ok=True)
    roots = verdicts.roots()
    check = roots["check"]
    doc: dict = {"host": socket.gethostname(), "user": getpass.getuser(), "home": str(Path.home()),
                 "set": {k: k in os.environ for k in ("CHECK_CACHE", "VERITY_TEST_CACHE", "XDG_CACHE_HOME", "CHECK_SLOT", "CHECK_SLOTS")},
                 "slot": os.environ.get("CHECK_SLOT"),
                 "roots": {k: where(p) for k, p in roots.items()}, "top": {}}
    shape: dict[str, bool] = {}
    if check.is_dir():
        for p in sorted(check.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                doc["top"][p.name] = {**{k: v for k, v in where(p).items() if k != "path"},
                                      "packed": not verdicts._skipped("check", p.name), "children": sum(1 for _ in p.iterdir())}
                if not verdicts._skipped("check", p.name):
                    shape[p.name] = p.is_symlink()
    shape.setdefault("circuit-check", False)
    print(f"cache-probe: {json.dumps({k: doc[k] for k in ('host', 'user', 'home', 'set', 'slot', 'roots')})}", flush=True)
    print(f"cache-probe: top {json.dumps(doc['top'])}", flush=True)
    doc["census"] = census()
    print(f"cache-probe: census {json.dumps(doc['census'])}", flush=True)
    (out / "cache-probe.json").write_text(json.dumps(doc, indent=1) + "\n")
    if "--no-store" not in sys.argv:
        doc["store_here"], _ = store(check / "circuit-check")
        print(f"cache-probe: store here {json.dumps(doc['store_here'])}", flush=True)
        doc["replica"] = replica(shape, check.is_symlink())
        print(f"cache-probe: replica {json.dumps(doc['replica'])}", flush=True)
        doc["replica_plain"] = replica(dict.fromkeys(shape, False), False)
        print(f"cache-probe: replica, no links {json.dumps(doc['replica_plain'])}", flush=True)
    (out / "cache-probe.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
