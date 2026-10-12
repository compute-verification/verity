"""Check `sqfloor gate` output: every line "FORM K x0 xK" (hex, canonical) has xK = x0^(2^K) mod 2^768 - 825."""

import json
import sys

P = 2**768 - 825


def main() -> int:
    bad = total = 0
    for line in sys.stdin:
        form, k, x0, xk = line.split()
        total += 1
        if pow(int(x0, 16) % P, 1 << int(k), P) != int(xk, 16):
            bad += 1
    print(json.dumps({"words": total, "bad": bad, "ok": total > 0 and bad == 0}))
    return 0 if total and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
