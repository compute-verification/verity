"""Check `phi gate` output: every line "KIND ITERS SEED KSEED TID U WORD" is `twins.pure_chains`'s word.

    phi gate | python3 gate.py > gate.json      # exit 0 only if every kind has lines and none differs
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import twins  # noqa: E402


def main() -> int:
    seen: dict[str, list[int]] = {}
    cache: dict[tuple, list[int]] = {}
    for line in sys.stdin:
        kind, iters, seed, kseed, tid, u, word = line.split()
        key = (kind, int(iters), int(seed), int(kseed), int(tid))
        if key not in cache:
            cache[key] = twins.pure_chains(*key)
        tally = seen.setdefault(kind, [0, 0])
        tally[0] += 1
        tally[1] += int(cache[key][int(u)] != int(word, 16))
    ok = set(seen) == set(twins.KINDS) and all(n and not bad for n, bad in seen.values())
    print(json.dumps({"kinds": {k: {"words": n, "bad": b} for k, (n, b) in seen.items()}, "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
