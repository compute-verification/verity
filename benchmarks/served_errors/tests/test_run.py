"""`verity bench served_errors` (`run.py`): the error terms it evaluates at the served sizes.

- Pearl+'s gap and ε_a on each of the four linears are the Lean lemma's own literals
  (`security/Proofs/Pouw/PearlC/ServedH100Rev1Gap.lean`'s table), so the script and the lemma price the same credit;
- one Flock session's knowledge error is NumbersDeploy's 2^-150.0 at one table and 2^80 calls;
- the outer proof's κ_S at 2^80 is KappaEpsCap's κ + t·ε, 2^-94.95;
- PoUW's δ_ks holds no binding term (no SHA-512 collision), δ_bind is always at κ = 0, and both it and ε_a are tagged
  assumed;
- the joint error's cryptographic sum takes every term at one budget, and says which;
- every row names its guarantee and formula, and the script runs with `--json`."""
from __future__ import annotations

import json
import re
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import run as R  # noqa: E402

LEMMA = HERE.parents[2] / "security" / "Proofs" / "Pouw" / "PearlC" / "ServedH100Rev1Gap.lean"


def test_pearl_gap_is_the_lemmas():
    table = {m[0]: (Fraction(m[1]), Fraction(m[2])) for m in
             re.findall(r"^\s*\| (\w+) \| \(\d+, \d+\) \| [\d/]+ \| (\d+/\d+) \(.*?\) \| (\d+/\d+) \(", LEMMA.read_text(), re.M)}
    assert set(table) == set(R.LINEARS)
    for name, (n, k) in R.LINEARS.items():
        g = R.pearl_gap(n, k)
        assert (g, g / (1 - g)) == table[name]


def test_flock_session_and_kappa_s():
    assert round(R.session_ks(80, 1), 2) == -150.00
    assert round(R.l2sum(R.REC_KAPPA, 80 + R.REC_EPS), 2) == -94.95


def test_rows_run(capsys):
    assert R.main(["--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows and all(r["guarantee"] and r["formula"] and r["log2"] != "" for r in rows)
    assert R.main([]) == 0
    assert all(r["log2"] > -300 for r in rows if r["term"] == "pouw δ_ks")
    assert all("κ = 0" in r["at"] and r["tag"] == "assumed" for r in rows if r["term"] == "pouw δ_bind")
    assert {r["tag"] for r in rows if r["term"] == "pearl+ ε_a"} == {"assumed"}
    sums = [r for r in rows if r["term"] == "joint, cryptographic" and "one budget" in r["at"]]
    assert len(sums) == 2 and all(r["at"].count("2^") >= 4 for r in sums)
