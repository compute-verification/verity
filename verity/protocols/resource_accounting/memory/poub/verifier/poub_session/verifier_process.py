"""PoUB's session verifier as a process: `verity/protocols/resource_accounting/memory/poub/verifier/session/verifier.py`'s `main`, on
this command line, handed what a verifier imports none of (it reads no `verity.ml` code beyond its frozen edges): the
node's figures (`timing.RTX_PRO_6000_NODE`, the measured node's), the LMS verify a gate session's signatures are checked
with (`lms_verify`) and a served session's (`verify`, which asks the Lean's where the Python's holds), and the
challenger's own process `--challenger serve` starts for a session (`SERVE`). Its usage is `verifier.py`'s.
live_verifier's host runs it under `python3 -I`, which puts neither the repository nor a venv on the path, so it puts
the repository's root there first."""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))

from verity.commitments.crypto import lms
from verity.commitments.scheme import DIGEST_BYTES
from verity.protocols.resource_accounting.memory.poub.verifier import coins, timing
from verity.protocols.resource_accounting.memory.poub.verifier.session import verifier
from verity.protocols.proofs.spot.verifier.challenger import challenger as Ch

ROOT = Path(__file__).resolve().parents[7]
SERVE = ROOT / "verity" / "protocols" / "compose" / "verifier" / "live_challenger" / "serve.py"


def lms_verify(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool:
    """`challenger.Verify`: LMS over SHA-512 at the key's (64, W, H)."""
    p = lms.Params(DIGEST_BYTES["sha512"], key.W, key.H)
    try:
        return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))
    except ValueError:
        return False


def lms_answer(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool:
    """The Lean's LMS verify (`verity/commitments/lms`) of a served session's signature at the key's (64, W, H), the
    one `verify` acts on where `lms_verify` holds. Until it is written the signature is undecided."""
    raise NotImplementedError(coins.UNWRITTEN)


def verify(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool:
    """`challenger.Verify` for a served session (`challengers.Serve`): `lms_verify` may refuse the signature, and where
    it doesn't the Lean's (`lms_answer`) decides; raises `coins.Undecided` while that isn't written."""
    if not lms_verify(key, message, signature):
        return False
    try:
        return lms_answer(key, message, signature)
    except NotImplementedError:
        raise coins.Undecided(coins.undecided("served session's LMS signatures")) from None


def main(argv: Sequence[str] | None = None) -> int:
    return verifier.main(argv, figures=timing.RTX_PRO_6000_NODE, verify=lms_verify, served_verify=verify,
                         serve=(sys.executable, str(SERVE)))


__all__ = ["lms_answer", "lms_verify", "main", "verify"]

if __name__ == "__main__":
    sys.exit(main())
