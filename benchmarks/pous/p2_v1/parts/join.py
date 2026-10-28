"""join.py STORE PART...: check STORE's segments and each fetched PART dir against their seg/<s>.json digests, then move
the PARTs' segments into STORE (.c and .rows first, .json last, so a half-moved segment is never counted done).
A PART whose setup.json has another commitment or salt than STORE's is refused. Prints one JSON line."""
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha512(p: Path) -> str:
    h = hashlib.sha512()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check(root: Path, s: str) -> str | None:
    meta = json.loads((root / "seg" / f"{s}.json").read_text())
    if sha512(root / "seg" / f"{s}.c") != meta["sha512"]:
        return f"{root}/seg/{s}.c"
    if sha512(root / "rows" / f"{s}.rows") != meta["rows_sha512"]:
        return f"{root}/rows/{s}.rows"
    return None


def segs(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "seg").glob("*.json"))


store, parts = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
setup = json.loads((store / "setup.json").read_text())
for part in parts:
    ps = json.loads((part / "setup.json").read_text())
    for k in ("commitment", "salt", "layout_sha512", "manifest"):
        if ps.get(k) != setup.get(k):
            raise SystemExit(f"{part}/setup.json: {k} differs from {store}/setup.json")
jobs = [(store, s) for s in segs(store)] + [(p, s) for p in parts for s in segs(p)]
with ThreadPoolExecutor(16) as ex:
    bad = [b for b in ex.map(lambda j: check(*j), jobs) if b]
if bad:
    raise SystemExit(f"{len(bad)} files differ from their digests, the first {bad[:5]}")
have, moved = set(segs(store)), 0
for part in parts:
    for s in segs(part):
        if s in have:
            continue
        os.replace(part / "seg" / f"{s}.c", store / "seg" / f"{s}.c")
        os.replace(part / "rows" / f"{s}.rows", store / "rows" / f"{s}.rows")
        os.replace(part / "seg" / f"{s}.json", store / "seg" / f"{s}.json")
        have.add(s)
        moved += 1
print(json.dumps({"checked": len(jobs), "moved": moved, "segments": len(have)}))
