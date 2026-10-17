"""prove.py BINARY DIR N [--selftest]: one loopback session of DIR's statement (`class_statement.prove`: the prover's peak
memory, its phase timings, the verifier's), or the selftest build's negative cases (``--selftest``). A form the soundness proof
does not cover is refused by the verifier (its verdict ``refused: ...``); the verdict it would have given is the record's
``without_refusal``, written here as ``verdict_without_refusal``. Writes DIR/prove.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from verity_flock import class_statement as CS


def main(argv: list[str]) -> int:
    binary, d, n = argv[0], Path(argv[1]), int(argv[2])
    if "--selftest" in argv:
        res = CS.prove(binary, d, n, 0, selftest=True, timeout=7200)
    else:
        res = CS.prove(binary, d, n, 7611, timeout=7200)
        live = [json.loads(ln[5:]) for ln in (d / "prove.out").read_text().splitlines() if ln.startswith("LIVE")]
        v = live[-1] if live else {}
        w = v.get("without_refusal") or (v.get("verdict") or {}).get("without_refusal") or {}
        res |= {"verdict_why": v.get("why"), "verdict_without_refusal": {k: w.get(k) for k in ("accepted", "why") if k in w}}
    (d / ("selftest.json" if "--selftest" in argv else "prove.json")).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps({k: res.get(k) for k in ("accepted", "all_pass", "failed", "why", "verdict_why", "verdict_without_refusal", "prove_s",
                                              "prove_total_s", "prover_peak_gb", "verifier_peak_gb", "verify_s", "k_log", "nbl",
                                              "dense_m", "proof_bytes", "wall_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
