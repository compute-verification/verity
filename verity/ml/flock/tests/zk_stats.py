"""Flock ZK (lane flock-zk): real against simulated transcripts of ``flock-circuit --zk`` on the checked fields.

Input: the ``ZKREAL``, ``ZKSIM`` and ``ZKCONTROL`` lines of ``flock-circuit zkstat --zk --n N`` (one verifier coin seed; the
real sessions draw fresh prover seeds; the simulator sees only the public statement and those coins; the control is the same
prover with zero pads and a zero mask slot, i.e. the non-ZK session's values in Flock ZK's format). For each field
class it runs four tests:

* ``bytes``: two-sample chi-square on the byte histogram (256 bins), real against simulated;
* ``bits``: chi-square over the 128 bit positions against one half, per kind;
* ``xor``: consecutive runs XORed at the same position, byte histogram against uniform, per kind (a value that depends on
  the witness and the coins but not on the prover's randomness XORs to zero);
* ``constant``: the fraction of positions whose value is the same in every run, per kind.

Positions holding one value in every real and simulated transcript are public structure (an F256 message's second
coordinate after Ligerito's code switch is always 0) and are excluded and counted (``structural_constants``).

Rep 1's value equal to rep 0's in every run is a reuse of randomness across the reps, which fails the class
(``reused_across_reps``), with one exception: the level-0 rows (``lig_rows0``) at a sample slot where both reps opened the same
level-0 position (``lig_rows0_q``, from the verifier's record). Both reps open the one level-0 codeword, so such a row is one
observation of the session, not two; it is excluded and counted (``shared_by_the_reps``). Likewise the inner proof draws its
168 query positions with replacement: a column whose position (``inner_open_q``, from the coins) repeats an earlier one in the
same rep's list is one observation, excluded and counted (``repeated_positions``).

Relations across fields and reps: the ``ZKREL`` lines hold, per session, two linear functionals of the public transcript that
combine both reps' ``ybar``, their opened level-0 rows and their shared level-1 positions (``flock-circuit``'s
``session_relations``: witness functionals, constant across prover seeds, if the reps shared level 0's extra lanes or their
padding). Each is a class of one value per session, tested the same way (``rel_level0``, ``rel_level1``).

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
RELATIONS = ["rel_level0", "rel_level1"]
ALPHA = 1e-3


def load(lines):
    """Per-rep rows by kind (``real``, ``sim``, ``control``), and per-session relation rows under ``rel:<kind>``."""
    by = defaultdict(list)
    for line in lines:
        tag, _, body = line.partition("\t")
        if tag in ("ZKREAL", "ZKSIM", "ZKCONTROL"):
            by[tag[2:].lower()].append(json.loads(body))
        elif tag == "ZKREL":
            v = json.loads(body)
            by["rel:" + v["kind"]].append(v)
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


def equal_across_reps(a, reps):
    """Positions whose rep-1 value equals rep 0's in every real run or in every simulated run."""
    if reps != 2:
        return np.zeros(a["real"].shape[1], dtype=bool)
    same = [np.all(v[: len(v) // 2] == v[len(v) // 2:], axis=(0, 2)) for v in (a["real"], a["sim"])]
    return same[0] | same[1]


def level0_shared(rows, cls, width):
    """``lig_rows0``'s positions at the sample slots where both reps opened the same level-0 position (``lig_rows0_q``)."""
    out = np.zeros(width, dtype=bool)
    q = {rep: next((r.get("lig_rows0_q") for r in rows["real"] if r["rep"] == rep), None) for rep in (0, 1)}
    if cls != "lig_rows0" or not q[0] or not q[1]:
        return out
    lanes = width // len(q[0])
    for k, (a, b) in enumerate(zip(q[0], q[1])):
        if a == b:
            out[k * lanes:(k + 1) * lanes] = True
    return out


INNER_OPENED = ("inner_open_h", "inner_open_mu")


def repeated_positions(rows, cls, width):
    """The inner opened columns whose declared position (``inner_open_q``) repeats an earlier one in the same rep's list."""
    out = np.zeros(width, dtype=bool)
    if cls not in INNER_OPENED:
        return out
    for rep in (0, 1):
        q = next((r.get("inner_open_q") for r in rows["real"] if r["rep"] == rep), None) or []
        seen = set()
        for j, x in enumerate(q[:width]):
            out[j] |= x in seen
            seen.add(x)
    return out


def test_class(rows, cls, reps, masked):
    """The four tests on one class, and reuse across reps; ``rows``: kind -> rows."""
    a = {k: values(rows[k], cls) for k in ("real", "sim", "control")}
    fixed = structural(a)
    equal = equal_across_reps(a, reps) & ~fixed
    allowed = level0_shared(rows, cls, a["real"].shape[1])
    shared, reused = equal & allowed, equal & ~allowed
    repeated = repeated_positions(rows, cls, a["real"].shape[1]) & ~fixed
    a = {k: v[:, ~(fixed | shared | repeated)] for k, v in a.items()}
    tab = np.array([byte_hist(a["real"]), byte_hist(a["sim"])])
    tab = tab[:, tab.sum(axis=0) > 0]
    p_bytes = float(stats.chi2_contingency(tab)[1])
    row = {"values_per_transcript": int(a["real"].shape[1]), "structural_constants": int(fixed.sum()), "shared_by_the_reps": int(shared.sum()),
           "reused_across_reps": int(reused.sum()), "repeated_positions": int(repeated.sum()), "bytes_real_vs_sim": p_bytes}
    for k in ("real", "sim", "control"):
        row[f"bits_{k}"] = bits_p(a[k])
        row[f"xor_{k}"] = xor_p(a[k], reps)
        row[f"constant_{k}"] = constant_frac(a[k], reps)
    indist = min(p_bytes, row["bits_real"], row["bits_sim"], row["xor_real"], row["xor_sim"]) > ALPHA \
        and row["constant_real"] == 0 and row["constant_sim"] == 0 and row["reused_across_reps"] == 0
    row["real_indistinguishable_from_sim"] = bool(indist)
    ok = indist
    if masked:
        row["control_distinguished"] = bool(row["xor_control"] < ALPHA or row["constant_control"] > 0.5)
        ok &= row["control_distinguished"]
    return row, ok


def analyse(by):
    reps = len({r["rep"] for r in by["real"]})
    out = {"n": {k: len(v) for k, v in by.items()}, "alpha": ALPHA, "classes": {}}
    ok = True
    for cls in [c for c in CLASSES if all(c in r for r in by["real"][:1])]:
        out["classes"][cls], o = test_class(by, cls, reps, cls in MASKED)
        ok &= o
    rel = {k: by["rel:" + k] for k in ("real", "sim", "control")}
    if all(rel.values()):
        for cls in RELATIONS:
            out["classes"][cls], o = test_class(rel, cls, 1, True)
            ok &= o
    else:
        out["relations"] = "no ZKREL lines"
        ok = False
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
