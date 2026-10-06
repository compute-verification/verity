"""Runs the squaring-chain binaries (`sqchain`, `sqchain2`, `sqchain3`) over configurations and checks every result.

For each configuration: random x < p, N squarings on the device, the output digits read back as an integer and compared
with Python's `pow(x, 2**N, p)`. Then the root identity at the real width: for y = z^2 mod p, the device's 16,446
squarings give y^(2^16446), the host multiplies by (y^5266)^-1 (off the critical path: it depends on y alone), and
r = y^((p+1)/4) must satisfy r^2 = y. The device's own timing (globaltimer and clock64 around the chain) is what is
reported; the host times nothing.

    python driver.py --bin NAME=PATH [--bin ...] --out DIR [--configs NAME:C:TPB[:X],...] [--n N] [--reps K] [--roots K]

X is the binary's last argument: MODE for sqchain (1 = st.async), WPB (warps per column block) for sqchain2, P (lanes per
column) for sqchain3. For sqchain3 the C field is PER, the digits per CTA (C = ceil(633 / PER) CTAs).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

W = 16448
C0 = 21065
P = (1 << W) - C0
ND = 633
RB = 26
E_SQ = W - 2                      # (p + 1) / 4 = 2^16446 - 5266
E_TAIL = ((1 << E_SQ) - (P + 1) // 4)
assert E_TAIL == 5266


def to_digits(x: int) -> list[int]:
    return [(x >> (RB * m)) & ((1 << RB) - 1) for m in range(ND)]


def from_digits(d: list[int]) -> int:
    return sum(v << (RB * m) for m, v in enumerate(d)) % P


def run(binary: str, c: int, tpb: int, mode: int, n: int, reps: int, x: int, tmp: Path) -> tuple[dict, int, list[int]]:
    xin, xout = tmp / "in.bin", tmp / "out.bin"
    xin.write_bytes(b"".join(v.to_bytes(4, "little") for v in to_digits(x)))
    pr = subprocess.run([binary, str(c), str(tpb), str(n), str(reps), str(xin), str(xout), str(mode)], capture_output=True,
                        text=True)
    if pr.returncode:
        raise SystemExit(f"{binary} C={c} tpb={tpb} mode={mode}: exit {pr.returncode}: {pr.stderr.strip()}")
    out = pr.stdout
    rec = json.loads(out.strip().splitlines()[-1])
    raw = xout.read_bytes()
    d = [int.from_bytes(raw[4 * m:4 * m + 4], "little") for m in range(ND)]
    return rec, from_digits(d), d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", required=True, help="NAME:C:TPB[:X],...")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--roots", type=int, default=2, help="root identities per configuration, at N = 16,446")
    ap.add_argument("--seed", type=int, default=20261006)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    bins = dict(b.split("=", 1) for b in a.bin)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for cfg in a.configs.split(","):
            name, *nums = cfg.split(":")
            c, tpb, mode = ([int(v) for v in nums] + [0])[:3]
            if name.startswith("sqchain2") and mode == 0:
                mode = 1
            x = rng.randrange(P)
            rec, got, d = run(bins[name], c, tpb, mode, a.n, a.reps, x, tmp)
            want = pow(x, 1 << a.n, P)
            rec.update(binary=name, check_pow=got == want, max_digit_bits=max(d).bit_length())
            roots = []
            for _ in range(a.roots):
                z = rng.randrange(1, P)
                y = z * z % P
                rr, s, _ = run(bins[name], c, tpb, mode, E_SQ, 1, y, tmp)
                r = s * pow(pow(y, E_TAIL, P), -1, P) % P
                roots.append({"identity": r * r % P == y, "ns_per_sq": rr["ns_per_sq_min"],
                              "chain_us": rr["ns_per_sq_min"] * E_SQ / 1e3})
            rec["roots"] = roots
            rec["root_identity"] = all(q["identity"] for q in roots)
            rec["root_us_at_best"] = rec["ns_per_sq_min"] * E_SQ / 1e3
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    bad = [r for r in rows if not (r["check_pow"] and r["root_identity"])]
    if bad:
        raise SystemExit(f"{len(bad)} configuration(s) failed their checks")


if __name__ == "__main__":
    main()
