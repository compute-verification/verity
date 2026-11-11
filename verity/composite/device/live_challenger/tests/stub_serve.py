"""`serve.py` as its own process, with the Lean verifier's coins stubbed: a test doesn't build `sampled-proofs-verifier`, so
`one_stage.verifier.coins` answers seeded bytes and appends each request's size to $COINS_LOG, or refuses as a Lean that
can't answer when $COINS_REFUSE is set.  The stub stands in for the Lean, so it carries the Lean's claim instance
(`one_stage.verifier.CLAIM`), unless $COINS_UNNAMED is set.  `--max-hellos N` lowers the cap on a session's refused
hellos (`challenger.MAX_HELLOS`, which `remote.log_hello` reads) to N, so a test reaches it in a few connections.

    python stub_serve.py [--max-hellos N] SERVE_PY [serve.py's arguments]
"""
import os
import random
import runpy
import sys
from pathlib import Path

args, max_hellos = sys.argv[1:], None
if args[:1] == ["--max-hellos"]:
    max_hellos, args = int(args[1]), args[2:]
serve = Path(args[0]).resolve()
sys.path.insert(0, str(serve.parents[2]))

from verity.protocols.spot.device import challenger as C  # noqa: E402
from verity.protocols.spot.verifier.one_stage import verifier as V  # noqa: E402

rng = random.Random(" ".join(args[1:]))


def coins(n: int) -> bytes:
    if os.environ.get("COINS_REFUSE"):
        raise V.LeanError(f"{V.EXE} refused {n} coin bytes")
    with open(os.environ["COINS_LOG"], "a") as f:
        f.write(f"{n}\n")
    return rng.randbytes(n)


if not os.environ.get("COINS_UNNAMED"):
    coins.claim = V.CLAIM
if max_hellos is not None:
    C.MAX_HELLOS = max_hellos
V.coins = coins
sys.argv = args
runpy.run_path(str(serve), run_name="__main__")
