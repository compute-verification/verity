"""run.sh's SERVE=node prover, memory accounting's `benchmarks/remote_session/node.py` unchanged, with each session's
passes at a rate of its own, for the curve of serving lost against passes a second.

~~~text
--pass-rates R0,R1,...   session k's passes a second (0: unpaced, as node.py runs them; sessions past the list too)
~~~

A paced session launches a pass no sooner than 1/R after the one before it, the prover's host waiting before the launch
(`device.Device.launch`, which `prover.ProverSide.session` calls for each of an honest session's passes). Nothing of the
prover's is queued on a GPU in the gap, so the engines have it alone: the engines and the prover are processes of their
own, whose kernels time-slice, and a spin kernel in the gap would keep the prover's slices. The wait holds the previous
pass's root by up to 1/R - τ (τ the pass), which stays under the window's allowance A (118 ms on one machine, about 145
ms across two) only for 1/R - τ well under A: rates of 15 a second and up at τ ≈ 15 ms. Calibration (`Device.calibrate`,
the store's own chain) and a session's warmup passes are never paced. Each session's record in prover.json gets its
`pace_passes_per_s`.

    python node_paced.py --pass-rates 0,30,15 --connect 127.0.0.1:PORT --out OUT --plan honest,honest,honest ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RS = Path(__file__).resolve().parents[2] / "remote_session"
sys.path.insert(0, str(RS))

import device
import node
from prover import ProverSide


class Pace:
    rate = 0.0
    next_at = 0.0


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pass-rates", default="")
    mine, rest = ap.parse_known_args()
    rates = [float(x) for x in mine.pass_rates.split(",") if x.strip()]
    if any(r < 0 for r in rates):
        raise SystemExit(f"--pass-rates are passes a second, 0 or more: {mine.pass_rates}")
    sys.argv = [str(RS / "node.py"), *rest]
    launch, session = device.Device.launch, ProverSide.session

    def paced(self, i, after):
        if Pace.rate:
            now = time.perf_counter()
            if now < Pace.next_at:
                time.sleep(Pace.next_at - now)
                now = time.perf_counter()
            Pace.next_at = max(now, Pace.next_at) + 1 / Pace.rate
        return launch(self, i, after)

    def paced_session(self, head: dict, mode: str, fraction: float) -> dict:
        k = head["k"]
        Pace.rate, Pace.next_at = (rates[k] if k < len(rates) else 0.0), 0.0
        print(f"node_paced: session {k} at {Pace.rate:g} passes a second (0: unpaced)", flush=True)
        try:
            r = session(self, head, mode, fraction)
        finally:
            rate, Pace.rate = Pace.rate, 0.0
        r["pace_passes_per_s"] = rate
        return r

    device.Device.launch = paced
    ProverSide.session = paced_session
    return node.main()


if __name__ == "__main__":
    sys.exit(main())
