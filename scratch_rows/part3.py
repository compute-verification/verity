"""Scratch, never lands: the console rule and the registry's fixed policies in the unit TS_ROWS selects."""
import json
import sys

from training_slowdown import calculator, gates, policy, starts

rows = []
for name in calculator.policies():
    p = calculator.policy(name)
    c = p["gamma_flop"] / 2 * gates.per_mac("gemm/bf16")
    rows.append({"policy": name, "kind": "fixed", "gamma_flop": p["gamma_flop"], "c": c, "k": p["k_bits"],
                 "k_over_c": p["k_bits"] / c})
reg = starts.registry()
for sname in reg["starts"]:
    shape = starts.start(sname, reg).shape
    c, k = policy.own_session(shape)
    rows.append({"policy": f"console@{sname}", "kind": "rule", "params": shape["parameters"], "c": c, "k": k,
                 "k_over_c": k / c})
c, k = policy.CONSOLE
rows.append({"policy": "console@reference-45M", "kind": "rule", "c": c, "k": k, "k_over_c": k / c})
out = {"rows_unit": gates.ROWS, "unit": gates.UNIT, "policies": rows}
json.dump(out, open(sys.argv[1], "w") if len(sys.argv) > 1 else sys.stdout, indent=1)
print(json.dumps(out, indent=1)) if len(sys.argv) > 1 else None
