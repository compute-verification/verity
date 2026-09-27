"""M1 (lane flock-zk): real against simulated transcripts of ``flock-circuit --zk`` on the checked fields.

Input: the ``ZKREAL``, ``ZKSIM`` and ``ZKCONTROL`` lines of ``flock-circuit zkstat --zk --n N`` (one verifier coin seed; the
real sessions draw fresh prover seeds; the simulator sees only the public statement and those coins; the control is the same
prover with zero pads and a zero mask slot, i.e. M0's values in M1's format). For each field class it runs four tests:

* ``bytes``: two-sample chi-square on the byte histogram (256 bins), real against simulated;
* ``bits``: chi-square over the 128 bit positions against one half, per kind;
* ``xor``: consecutive runs XORed at the same position, byte histogram against uniform, per kind (a value that depends on
  the witness and the coins but not on the prover's randomness XORs to zero);
* ``constant``: the fraction of positions whose value is the same in every run, per kind.

Positions holding one value in every real and simulated transcript are public structure (an F256 message's second
coordinate after Ligerito's code switch is always 0) and are excluded and counted (``structural_constants``).

Real and simulated must be indistinguishable (every ``p`` above ``ALPHA``); the control must be distinguished on the classes
the masks cover (its ``xor`` test fails and its positions are constant), which shows the tests have power.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

CLASSES = ["exposed", "s_ab", "s_c", "inner_scalars", "inner_y", "inner_open_h", "inner_open_mu",
           "lig_exposed", "lig_tprime_ybar", "lig_rows0", "lig_deep"]
MASKED = ["exposed", "s_ab", "s_c", "lig_exposed", "lig_tprime_ybar", "lig_rows0", "lig_deep"]
ALPHA = 1e-3


def load(lines):
    by = defaultdict(list)
    for line in lines:
        tag, _, body = line.partition("\t")
        if tag in ("ZKREAL", "ZKSIM", "ZKCONTROL"):
            by[tag[2:].lower()].append(json.loads(body))
    return by


def values(rows, cls):
    """(transcripts, positions, 16) uint8, transcripts ordered by (rep, run)."""
    rows = sorted(rows, key=lambda r: (r["rep"], r["run"]))
    return np.array([[np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in r[cls]] for r in rows])


def byte_hist(a):
    return np.bincount(a.reshape(-1), minlength=256)


def bits_p(a):
    bits = np.unpackbits(a.reshape(-1, 16), axis=1, bitorder="little")
    n = bits.shape[0]
    ones = bits.sum(axis=0)
    chi = (((ones - n / 2) ** 2) / (n / 4)).sum()
    return float(stats.chi2.sf(chi, 128))


def xor_p(a, reps):
    per = a.shape[0] // reps
    xs = [a[r * per + i] ^ a[r * per + i + 1] for r in range(reps) for i in range(per - 1)]
    h = byte_hist(np.array(xs))
    return float(stats.chisquare(h).pvalue)


def constant_frac(a, reps):
    per = a.shape[0] // reps
    fr = []
    for r in range(reps):
        blk = a[r * per:(r + 1) * per]
        fr.append(float(np.all(blk == blk[0], axis=0).all(axis=1).mean()))
    return max(fr)


def structural(a):
    """Positions holding one value in every real and simulated transcript (public structure, e.g. an F256 message's
    second coordinate after the code switch, always 0): excluded from the tests and counted."""
    both = np.concatenate([a["real"], a["sim"]])
    return np.all(both == both[0], axis=0).all(axis=1)


def analyse(by):
    reps = len({r["rep"] for r in by["real"]})
    out = {"n": {k: len(v) for k, v in by.items()}, "alpha": ALPHA, "classes": {}}
    ok = True
    for cls in [c for c in CLASSES if all(c in r for r in by["real"][:1])]:
        a = {k: values(by[k], cls) for k in ("real", "sim", "control")}
        fixed = structural(a)
        a = {k: v[:, ~fixed] for k, v in a.items()}
        tab = np.array([byte_hist(a["real"]), byte_hist(a["sim"])])
        tab = tab[:, tab.sum(axis=0) > 0]
        p_bytes = float(stats.chi2_contingency(tab)[1])
        row = {"values_per_transcript": int(a["real"].shape[1]), "structural_constants": int(fixed.sum()), "bytes_real_vs_sim": p_bytes}
        for k in ("real", "sim", "control"):
            row[f"bits_{k}"] = bits_p(a[k])
            row[f"xor_{k}"] = xor_p(a[k], reps)
            row[f"constant_{k}"] = constant_frac(a[k], reps)
        indist = min(p_bytes, row["bits_real"], row["bits_sim"], row["xor_real"], row["xor_sim"]) > ALPHA \
            and row["constant_real"] == 0 and row["constant_sim"] == 0
        row["real_indistinguishable_from_sim"] = bool(indist)
        if cls in MASKED:
            row["control_distinguished"] = bool(row["xor_control"] < ALPHA or row["constant_control"] > 0.5)
            ok &= row["control_distinguished"]
        ok &= indist
        out["classes"][cls] = row
    out["pass"] = bool(ok)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with open(argv[0]) as f:
        res = analyse(load(f))
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
