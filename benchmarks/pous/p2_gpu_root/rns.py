"""Runs the RNS squaring chain (`rnschain`, explicit CRT) over configurations and checks every result, as driver.py does.

For J squarings per round the moduli are the K = C R largest primes below 2^26, and K must cover the range: a round's
input y < K 2^26 p + K p, so M = prod m_j must exceed y^(2^J) with 20 bits of slack (the fixed-point alpha needs it).
The output residues are an integer y = x^(2^N) mod p up to a multiple of p; the CRT recovers it and reduces mod p.

    python rns.py --bin NAME=PATH [--bin ...] --out DIR --configs NAME:C:J,... [--n N] [--reps K] [--roots K]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver import E_SQ, E_TAIL, P  # noqa: E402

SLACK = 20


def primes_below(top: int, count: int) -> list[int]:
    lo = top - 64 * count - 4096
    sieve = bytearray([1]) * (top - lo)
    for q in range(2, int(top ** 0.5) + 1):
        start = max(q * q, (lo + q - 1) // q * q)
        sieve[start - lo::q] = bytearray(len(range(start, top, q)))
    ps = [lo + i for i in range(top - lo - 1, -1, -1) if sieve[i]]
    assert len(ps) >= count
    return ps[:count]


class Base:
    def __init__(self, k: int, j: int):
        self.k, self.j = k, j
        self.m = primes_below(1 << 26, k)
        self.M = 1
        for v in self.m:
            self.M *= v
        ybound = k * (1 << 26) * P + k * P
        if self.M <= ybound ** (1 << j) << SLACK:
            raise SystemExit(f"K = {k} moduli do not cover J = {j}: need {(ybound ** (1 << j)).bit_length() + SLACK} bits, "
                             f"have {self.M.bit_length() - 1}")
        mi = [self.M // v for v in self.m]
        self.inv = [pow(a % v, -1, v) for a, v in zip(mi, self.m)]
        self.mi = mi
        c = [a % P for a in mi]
        d = self.M % P
        self.e = [(P - d) % v for v in self.m]
        rows = [b"".join((ci % mj).to_bytes(4, "little") for ci in c) for mj in self.m]
        self.params = (b"".join(v.to_bytes(4, "little") for v in self.m + self.inv + self.e) + b"".join(rows))

    def residues(self, x: int) -> bytes:
        return b"".join((x % v).to_bytes(4, "little") for v in self.m)

    def value(self, raw: bytes) -> int:
        r = [int.from_bytes(raw[4 * i:4 * i + 4], "little") for i in range(self.k)]
        return sum(ri * inv % mj * a for ri, inv, mj, a in zip(r, self.inv, self.m, self.mi)) % self.M


def run(binary: str, c: int, n: int, reps: int, base: Base, x: int, tmp: Path) -> tuple[dict, int]:
    pp, xin, xout = tmp / "params.bin", tmp / "in.bin", tmp / "out.bin"
    xin.write_bytes(base.residues(x))
    pr = subprocess.run([binary, str(c), str(n), str(reps), str(pp), str(xin), str(xout)], capture_output=True, text=True)
    if pr.returncode:
        raise SystemExit(f"{binary} C={c}: exit {pr.returncode}: {pr.stderr.strip()}")
    rec = json.loads(pr.stdout.strip().splitlines()[-1])
    return rec, base.value(xout.read_bytes()) % P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", action="append", required=True, help="NAME=PATH (built with -DRR=R -DJJ=J)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", required=True, help="NAME:C:R:J,...")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--roots", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261007)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    bins = dict(b.split("=", 1) for b in a.bin)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for cfg in a.configs.split(","):
            name, c, r, j = cfg.split(":")
            c, r, j = int(c), int(r), int(j)
            base = Base(c * r, j)
            (tmp / "params.bin").write_bytes(base.params)
            n = a.n - a.n % j
            x = rng.randrange(P)
            rec, got = run(bins[name], c, n, a.reps, base, x, tmp)
            if (rec["R"], rec["J"]) != (r, j):
                raise SystemExit(f"{name} was built with R = {rec['R']}, J = {rec['J']}, not {r}, {j}")
            rec.update(binary=name, check_pow=got == pow(x, 1 << n, P), range_bits=base.M.bit_length() - 1)
            roots = []
            for _ in range(a.roots):
                z = rng.randrange(1, P)
                y = z * z % P
                rr, s = run(bins[name], c, E_SQ, 1, base, y, tmp)
                root = s * pow(pow(y, E_TAIL, P), -1, P) % P
                roots.append({"identity": root * root % P == y, "ns_per_sq": rr["ns_per_sq_min"],
                              "chain_us": rr["ns_per_sq_min"] * E_SQ / 1e3})
            rec["roots"] = roots
            rec["root_identity"] = all(q["identity"] for q in roots)
            rec["root_us_at_best"] = rec["ns_per_sq_min"] * E_SQ / 1e3
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    with open(out / "results.jsonl", "a") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    bad = [r for r in rows if not (r["check_pow"] and r["root_identity"])]
    if bad:
        raise SystemExit(f"{len(bad)} configuration(s) failed their checks")


if __name__ == "__main__":
    main()
