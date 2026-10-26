"""`serve.py` as its own process, with the Lean verifier's coins stubbed: a test doesn't build `sampled-proofs-verifier`, so
`one_stage.verifier.coins` answers seeded bytes and appends each request's size to $COINS_LOG, or refuses as a Lean that
can't answer when $COINS_REFUSE is set.  The stub stands in for the Lean, so it carries the Lean's claim instance
(`one_stage.verifier.CLAIM`), unless $COINS_UNNAMED is set.

    python stub_serve.py SERVE_PY [serve.py's arguments]
"""
import os
import random
import runpy
import sys
from pathlib import Path

serve = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(serve.parents[2]))

from verity.core.protocols.sampled_proofs.one_stage import verifier as V  # noqa: E402

rng = random.Random(" ".join(sys.argv[2:]))


def coins(n: int) -> bytes:
    if os.environ.get("COINS_REFUSE"):
        raise V.LeanError(f"{V.EXE} refused {n} coin bytes")
    with open(os.environ["COINS_LOG"], "a") as f:
        f.write(f"{n}\n")
    return rng.randbytes(n)


if not os.environ.get("COINS_UNNAMED"):
    coins.claim = V.CLAIM
V.coins = coins
sys.argv = sys.argv[1:]
runpy.run_path(str(serve), run_name="__main__")
