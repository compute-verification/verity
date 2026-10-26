"""`registration.sh`'s inputs: per N, DIR/reg-N.json, the `registration` piece's `{"params", "record"}` for a valid v1
audit with N more ports (`test_difftest._many_ports`, a MoE window's shape: 8 tiles to an expert, roots shuffled), which
R3 accepts.

  python registration.py DIR N...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "verity/core/service/sampled_proofs/tests"))

import test_difftest as T  # noqa: E402


def main(out: Path, ns: list[int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    base = T._REGISTRATION_OF(T._v1_audit("subset"))
    for n in ns:
        a = T._json(T._many_ports(base, n))
        (out / f"reg-{n}.json").write_text(json.dumps(a, sort_keys=True, separators=(",", ":")))
        print(json.dumps({"n": n, "ports": len(a["params"]["ports"]), "roots": len(a["record"]["roots"])}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), [int(n) for n in sys.argv[2:]])
