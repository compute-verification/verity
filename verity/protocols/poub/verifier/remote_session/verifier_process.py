"""PoUB's session verifier as a process: `verity/protocols/poub/verifier/session/verifier.py`'s `main`, on this command line, handed
what a verifier imports none of (it reads no `verity.ml` code beyond its frozen edges): the LMS verify a served or gate
session's signatures are checked with (`lms_verify`), and the challenger's own process `--challenger serve` starts for a
session (`SERVE`). Its usage is `verifier.py`'s. live_verifier's host runs it under `python3 -I`, which puts neither
the repository nor a venv on the path, so it puts the repository's root there first."""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from verity.protocols.poub.verifier.session import verifier
from verity.protocols.spot.device import challenger as Ch
from verity.commitments.scheme import DIGEST_BYTES
from verity.commitments.crypto import lms

ROOT = Path(__file__).resolve().parents[5]
SERVE = ROOT / "verity" / "composite" / "device" / "live_challenger" / "serve.py"


def lms_verify(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool:
    """`challenger.Verify`: LMS over SHA-512 at the key's (64, W, H)."""
    p = lms.Params(DIGEST_BYTES["sha512"], key.W, key.H)
    try:
        return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))
    except ValueError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    return verifier.main(argv, verify=lms_verify, serve=(sys.executable, str(SERVE)))


if __name__ == "__main__":
    sys.exit(main())
