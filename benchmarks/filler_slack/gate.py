"""Check `filler gate` ("G") and `filler sweep`'s sampled ("V") lines against `twin.thread_words`.

    filler gate | python3 gate.py > gate.json         # exit 0 only if every kernel has lines and none differs
    python3 gate.py < verify.txt > verify.json

Each line is "G|V KERNEL SEED KSEED ITERS FTID w0..w19". The report counts lines and mismatches per kernel, and, for the
first mismatching line of each kernel, which words differ (receipts 0-3, accumulators 4-19).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import twin  # noqa: E402


def main() -> int:
    seen: dict[str, dict] = {}
    cache: dict[tuple, list[int]] = {}
    for line in sys.stdin:
        f = line.split()
        if len(f) != 26 or f[0] not in ("G", "V"):
            continue
        kern, seed, kseed, iters, ftid = f[1], int(f[2]), int(f[3]), int(f[4]), int(f[5])
        key = (seed, kseed, iters, ftid)
        if key not in cache:
            cache[key] = twin.thread_words(*key)
        got = [int(w, 16) for w in f[6:]]
        tally = seen.setdefault(kern, {"lines": 0, "bad": 0})
        tally["lines"] += 1
        if got != cache[key]:
            tally["bad"] += 1
            if "first" not in tally:
                tally["first"] = {"key": key, "words": [i for i in range(20) if got[i] != cache[key][i]]}
    ok = bool(seen) and all(t["lines"] and not t["bad"] for t in seen.values())
    print(json.dumps({"kernels": seen, "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
