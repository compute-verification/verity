"""RT2-LOGIC-9: sm_120's measured headline reading on VF3r-16 (the statement's vector) and on VF3s-16 (RT-FACET's refit).

Runs #2482's own exporter (`benchmarks/pouw/lean_headline.py`'s `measured()`, every check it makes) with only the
vector changed, and prints π*, w*, `B_♭` and the Lean bound's two branches on each. Light (a few seconds, one LP a
vector). Usage, from the repository root of a tree carrying #2482:

    PYTHONPATH=benchmarks/pouw/machine_model python benchmarks/rt2_logic/vf3s_reading.py [ROOT] [VECTOR ...]
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
vectors = sys.argv[2:] or ["VF3r-16", "VF3s-16"]
sys.path.insert(0, str(root / "benchmarks" / "pouw"))
import lean_headline as L  # noqa: E402

i = L.instance()
out = {"stated": float(L.EPS_M), "vectors": {}}
for vec in vectors:
    L.VECTOR, L.EPS_M = vec, Fraction(1)
    try:
        m = L.measured(i)
    except SystemExit as e:
        out["vectors"][vec] = {"refused": str(e)}
        continue
    out["vectors"][vec] = {
        "pistar": m.pistar, "wstar": m.wstar, "B": float(m.B), "kappa": float(m.kappa),
        "folded": float(m.folded), "unfolded": float(m.unfolded), "bound": float(m.bound),
        "below_stated": m.bound < Fraction(591, 100000), "exact_logs": L.exact_logs(m),
    }
print(json.dumps(out, indent=1))
