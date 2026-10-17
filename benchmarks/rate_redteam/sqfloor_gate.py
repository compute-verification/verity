"""Check `sqfloor gate` output: every line "FORM K BITS C x0 xK" (hex, canonical) has xK = x0^(2^K) mod 2^BITS - C."""

import json
import sys

def main() -> int:
    bad = total = 0
    for line in sys.stdin:
        form, k, bits, c, x0, xk = line.split()
        p = 2 ** int(bits) - int(c)
        total += 1
        if pow(int(x0, 16) % p, 1 << int(k), p) != int(xk, 16):
            bad += 1
    print(json.dumps({"words": total, "bad": bad, "ok": total > 0 and bad == 0}))
    return 0 if total and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
