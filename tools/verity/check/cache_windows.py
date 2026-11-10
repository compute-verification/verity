"""What a node's check cache gained in given time windows, by family: a measuring script that reads only, not part of check.

    python3 tools/verity/check/cache_windows.py --window LABEL:START:END [...] [--hours N]     writes $OUT/cache-windows.json

A check's verdict pack holds what the node's check cache gained while it ran (`verdicts.changed`), whoever stored it.  For each
window (epoch seconds, a recorded check's start and end) this counts the verdict entries under `~/.cache/verity-check` whose
mtime falls in it, by family, a family behind a link at the cache's top included and marked, and writes the window's entries as a
gzipped tar to a counter (never to disk) to say what a pack of them weighs: all of them, and those a walk that follows no link
would list.  Then the circuit-check family's entries and bytes per hour over the last N hours (96).  An entry rewritten or
removed since its window is not counted, so each number is a floor.  Stdlib only; nothing is written outside $OUT.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

SKIP = {"lean-deps", "lean-builds", "lean-changed", "lean-records", "lean-artifacts"}
SCRATCH = "lean-audit-scratch-"
GZ_CAP = 8 << 30


class Count:
    def __init__(self):
        self.n = 0

    def write(self, b):
        self.n += len(b)
        return len(b)

    def flush(self):
        pass


def walk(base: Path):
    """(family, linked, pack path, file, mtime, size) for every verdict entry under ``base``, following a link at its top only."""
    for top in sorted(os.scandir(base), key=lambda e: e.name):
        if top.name.startswith(".") or top.name in SKIP or top.name.startswith(SCRATCH) or not top.is_dir():
            continue
        linked = top.is_symlink()
        stack = [(top.path, top.name)]
        while stack:
            d, rel = stack.pop()
            try:
                it = list(os.scandir(d))
            except OSError:
                continue
            for e in it:
                if e.name.startswith("."):
                    continue
                if e.is_dir(follow_symlinks=False):
                    stack.append((e.path, f"{rel}/{e.name}"))
                elif e.name.endswith(".json") and e.is_file(follow_symlinks=False):
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    yield top.name, linked, f"check/{rel}/{e.name}", e.path, st.st_mtime, st.st_size


def gz(rows) -> dict:
    """The bytes a gzipped tar of ``rows`` (pack path, file, size) has, written to a counter; stops at GZ_CAP raw bytes."""
    sink, raw, n = Count(), 0, 0
    t0 = time.monotonic()
    with gzip.GzipFile(filename="", mode="wb", fileobj=sink, mtime=0) as z, tarfile.open(fileobj=z, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for k, p, size in sorted(rows):
            if raw + size > GZ_CAP:
                break
            try:
                data = Path(p).read_bytes()
            except OSError:
                continue
            info = tarfile.TarInfo(k)
            info.mode, info.mtime, info.size = 0o644, 0, len(data)
            tar.addfile(info, io.BytesIO(data))
            raw += len(data)
            n += 1
    return {"entries": n, "of": len(rows), "raw_bytes": raw, "gz_bytes": sink.n, "seconds": round(time.monotonic() - t0, 1)}


def main() -> int:
    argv = sys.argv[1:]
    windows = []
    for i, a in enumerate(argv):
        if a == "--window":
            label, start, end = argv[i + 1].rsplit(":", 2)
            windows.append((label, float(start), float(end)))
    hours = int(argv[argv.index("--hours") + 1]) if "--hours" in argv else 96
    out = Path(os.environ.get("OUT") or ".")
    out.mkdir(parents=True, exist_ok=True)
    base = Path.home() / ".cache" / "verity-check"
    now = time.time()
    t0 = time.monotonic()
    fam: dict[str, dict] = {}
    inwin: dict[str, list] = {w[0]: [] for w in windows}
    hourly: dict[int, list] = {}
    for name, linked, k, p, mtime, size in walk(base):
        f = fam.setdefault(name, {"link": linked, "entries": 0, "bytes": 0})
        f["entries"] += 1
        f["bytes"] += size
        for label, start, end in windows:
            if start <= mtime <= end:
                inwin[label].append((name, linked, k, p, size))
        if name == "circuit-check" and now - mtime < hours * 3600:
            h = hourly.setdefault(int(mtime // 3600), [0, 0])
            h[0] += 1
            h[1] += size
    doc: dict = {"cache": str(base), "top_is_link": base.is_symlink(), "walk_seconds": round(time.monotonic() - t0, 1), "families": fam,
                 "windows": {}, "circuit_check_by_hour": {time.strftime("%m-%dT%HZ", time.gmtime(h * 3600)): v for h, v in sorted(hourly.items())}}
    print(f"cache-windows: {sum(f['entries'] for f in fam.values())} entries walked in {doc['walk_seconds']} s; behind a link: "
          f"{sorted(n for n, f in fam.items() if f['link'])}", flush=True)
    for label, start, end in windows:
        rows = inwin[label]
        by: dict[str, dict] = {}
        for name, linked, _k, _p, size in rows:
            b = by.setdefault(name, {"link": linked, "entries": 0, "bytes": 0})
            b["entries"] += 1
            b["bytes"] += size
        w = {"start": start, "end": end, "by_family": by,
             "pack_of_all": gz([(k, p, size) for _n, _l, k, p, size in rows]),
             "pack_of_unlinked": gz([(k, p, size) for _n, linked, k, p, size in rows if not linked])}
        doc["windows"][label] = w
        print(f"cache-windows: {label} {json.dumps(w)}", flush=True)
        (out / "cache-windows.json").write_text(json.dumps(doc, indent=1) + "\n")
    hs = doc["circuit_check_by_hour"]
    print(f"cache-windows: circuit-check by hour over {hours} h: {len(hs)} hours with a store, "
          f"{sum(v[0] for v in hs.values())} entries, {sum(v[1] for v in hs.values()) / 2**30:.1f} GiB; the largest hours "
          f"{sorted(((v[0], round(v[1] / 2**20), h) for h, v in hs.items()), reverse=True)[:6]}", flush=True)
    (out / "cache-windows.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
