"""The trusted third party's RW-SMS encoder and the public decoder's reference, for mont.cu's gates (stdlib only).

A Williams modulus N = p q (p = 3, q = 7 mod 8, both n/2 bits with their top two bits set, so 2^(n-1) < N < 2^n); in
the half-fixed form (`--form 1`) N = 5 mod 2^(32 KF), KF = n/64 - 1: q's low 32 KF bits are 5 p^-1, its top limb is
searched. A payload m < N of record j is encoded as y = m + t + j mod N, (a1, r) = RW root of y R, z = r xor K,
h = z div N, v = z - h N, (a2, c) = RW root of v R, with R = 2^n so that the decoder's Montgomery squares
x -> x^2 R^-1 land on y and v. The record is (c, a1 | a2 << 2 | h << 4). A tweak (e, f) multiplies by (-1)^e 2^f.

  rw.py keygen BITS FORM SEED > params.json
  rw.py params params.json > params.txt          N, n', K and t in hex, one a line (mont's format)
  rw.py encode params.json COUNT SEED > records.txt   "c_hex tag m_hex", one record a line
  rw.py check-decode params.json records.txt < decoded.txt   JSON {"records", "bad", "ok"}
  rw.py check-chain params.json K < chain.txt          lines "x0_hex xK_hex"; JSON {"words", "bad", "ok"}
"""

import json
import random
import sys

SMALL = [p for p in range(3, 2000) if all(p % d for d in range(2, int(p**0.5) + 1))]


def is_prime(n, rng):
    if n < 2 or any(n % p == 0 for p in SMALL if p < n):
        return n in SMALL
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for _ in range(24):
        x = pow(rng.randrange(2, n - 1), d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def prime_mod8(bits, res, rng):
    while True:
        x = rng.getrandbits(bits) | (3 << (bits - 2))
        x += (res - x) % 8
        if x.bit_length() == bits and is_prime(x, rng):
            return x


def keygen(bits, form, seed):
    rng = random.Random(seed)
    nl = bits // 32
    kf = nl // 2 - 1
    half = bits // 2
    while True:
        p = prime_mod8(half, 3, rng)
        if form % 2 == 0:
            q = prime_mod8(half, 7, rng)
        else:
            low = 32 * kf
            q0 = 5 * pow(p, -1, 1 << low) % (1 << low)
            for _ in range(20000):
                s = rng.getrandbits(half - low) | (3 << (half - low - 2))
                q = q0 + (s << low)
                if q % 8 == 7 and is_prime(q, rng):
                    break
            else:
                continue
        n = p * q
        if p != q and n.bit_length() == bits:
            break
    if form % 2 == 1:
        assert n % (1 << (32 * kf)) == 5
    assert p % 8 == 3 and q % 8 == 7
    return {"bits": bits, "form": form, "N": n, "p": p, "q": q, "np": (-pow(n, -1, 1 << 32)) % (1 << 32),
            "K": rng.getrandbits(bits), "t": rng.randrange(n // 2)}


def tweak(x, tw, n):
    e, f = tw & 1, (tw >> 1) & 1
    x = x * (2 if f else 1) % n
    return (n - x) % n if e else x


def is_qr(u, p):
    return pow(u, (p - 1) // 2, p) == 1


def rw_root(y, k):
    """(tw, r) with tweak(r^2 R^-1, tw) = y, r the principal root (itself a square mod p and mod q)."""
    n, p, q, bits = k["N"], k["p"], k["q"], k["bits"]
    big_r = pow(2, bits, n)
    for tw in range(4):
        e, f = tw & 1, (tw >> 1) & 1
        u = y * big_r % n
        if e:
            u = (n - u) % n
        if f:
            u = u * pow(2, -1, n) % n
        if is_qr(u % p, p) and is_qr(u % q, q):
            rp, rq = pow(u, (p + 1) // 4, p), pow(u, (q + 1) // 4, q)
            if not is_qr(rp, p):
                rp = p - rp
            if not is_qr(rq, q):
                rq = q - rq
            r = (rp * q * pow(q, -1, p) + rq * p * pow(p, -1, q)) % n
            assert tweak(r * r * pow(big_r, -1, n) % n, tw, n) == y
            return tw, r
    raise ValueError("y shares a factor with N")


def encode(k, m, j):
    n = k["N"]
    a1, r = rw_root((m + k["t"] + j) % n, k)
    z = r ^ k["K"]
    h, v = divmod(z, n)
    assert h in (0, 1)
    a2, c = rw_root(v, k)
    return c, a1 | a2 << 2 | h << 4


def decode(k, c, tag, j):
    n, rinv = k["N"], pow(2, -k["bits"], k["N"])
    v = tweak(c * c * rinv % n, (tag >> 2) & 3, n)
    r = (v + ((tag >> 4) & 1) * n) ^ k["K"]
    y = tweak(r * r * rinv % n, tag & 3, n)
    return (y - k["t"] - j) % n


def load(path):
    with open(path) as f:
        return {key: int(v) if not isinstance(v, int) else v for key, v in json.load(f).items()}


def main(argv):
    cmd = argv[0]
    if cmd == "keygen":
        k = keygen(int(argv[1]), int(argv[2]), int(argv[3]))
        print(json.dumps({key: str(v) if v > 1 << 62 else v for key, v in k.items()}))
    elif cmd == "params":
        k = load(argv[1])
        for key in ("N", "np", "K", "t"):
            print(f"{k[key]:x}")
    elif cmd == "encode":
        k = load(argv[1])
        rng = random.Random(int(argv[3]))
        for j in range(int(argv[2])):
            m = rng.randrange(1 << (k["bits"] - 8))
            c, tag = encode(k, m, j)
            assert decode(k, c, tag, j) == m
            print(f"{c:x} {tag} {m:x}")
    elif cmd == "check-decode":
        k = load(argv[1])
        with open(argv[2]) as f:
            want = [int(line.split()[2], 16) for line in f if line.strip()]
        got = [int(line, 16) for line in sys.stdin if line.strip()]
        bad = sum(g != w for g, w in zip(got, want)) + abs(len(got) - len(want))
        print(json.dumps({"bits": k["bits"], "records": len(want), "bad": bad, "ok": bad == 0}))
        return 0 if bad == 0 else 1
    elif cmd == "check-chain":
        k = load(argv[1])
        n, rinv, kk = k["N"], pow(2, -k["bits"], k["N"]), int(argv[2])
        words = bad = 0
        for line in sys.stdin:
            if not line.strip():
                continue
            x0, xk = (int(s, 16) for s in line.split())
            x = x0
            for _ in range(kk):
                x = x * x * rinv % n
            words += 1
            bad += x != xk
        print(json.dumps({"bits": k["bits"], "K": kk, "words": words, "bad": bad, "ok": words > 0 and bad == 0}))
        return 0 if words and not bad else 1
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
