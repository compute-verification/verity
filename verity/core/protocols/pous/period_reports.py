"""The memory challenger's sessions reported under its party's per-period root (`verity.core.attest`).

As an auditor party with a device identity (`lms.Identity`, the lower key it issued the challenger, `lms.PartyKey`),
the challenger signs no session alone. `PeriodSigner` is its `memory_challenger.Signer`: each session it closes is the
next sampling block of the party's certification period, with the session's record (`memory_challenger.message`, 64
bytes, bound to the session's id and header) the block's one record, and `sign` returns the block's public liveness
record, the session's one output in the clear, signed before the session's record joins the period. The period's root
is signed once, at its close (`PeriodSigner.close`), and the closed period (its records, root and the root's signature)
goes to the prover only, as witness; the signature's lower key and certificate are public, as every liveness record
shows them. The period starts its attachment on the party's key (`lms.PartyKey.start`), so a challenger restarted within
the attachment signs no more of its sessions, under no-state-rollback.

`period_verify` is `PERIOD`'s verification of a session against its closed period, in the clear: the session's record
opens to the period's root, whose signature verifies under the device's identity for the party, and its liveness record
is its block's. A period's proof runs the root's signature and the openings in zero knowledge (`Attest.check`), and
the verifier the certificate in the clear (`Attest.accepts` is both), which `Properties.Attest.PeriodSigned` is about.
`PERIOD` is not in `memory_challenger.SCHEMES`: the run protocol's Lean
(`PousCheck.RunProtocol`) still reads S1's per-session signatures.
"""
from __future__ import annotations

from verity.core import attest
from verity.ml.crypto import lms

from .memory_challenger import Verify

__all__ = ["PERIOD", "PeriodSigner", "period_verify"]

PERIOD = "hss-sha512/n64-w1-h20/period-root/v1"
"""A session reported under its party's per-period root, two LMS levels at `lms.DEPLOYED`."""


class PeriodSigner:
    """The memory challenger's signer as an auditor party: its sessions are the sampling blocks of `period` (under the
    party's lower key, `lms.PartyKey`), each with its record the block's one record, and `sign` returns the block's
    liveness record. `public_key` is the device's identity, the key a session's verifier trusts."""

    def __init__(self, period: attest.Period, identity: bytes, scheme: str = PERIOD) -> None:
        self.period, self.public_key, self.scheme, self.blocks = period, bytes(identity), scheme, 0

    def sign(self, session: bytes, record: bytes) -> bytes:
        """The session's liveness record, as the period's next block, whose one record is ``record``: signed before the
        record joins the period, and a refused signature takes no block, so each block's index is its record's."""
        live = self.period.block(self.blocks, [record])
        self.blocks += 1
        return live

    def close(self) -> attest.SignedPeriod:
        """Sign the period's root, once; what goes to the prover."""
        return self.period.close()


def period_verify(closed: attest.SignedPeriod, deployment: bytes, party: str,
                  params: lms.Params = lms.DEPLOYED) -> Verify:
    """`PERIOD`'s `verify(public_key, session, record, signature)` against the closed period `closed`, for `party` in the
    deployment whose hash is `deployment`: `record` is the period's record at some block, which opens to the root
    (`attest.verify_opening`), the root's signature verifies under the identity `public_key` (`lms.hss_verify`, both
    levels), and `signature` is that block's liveness record (`lms.verify_liveness`)."""

    def verify(public_key: bytes, session: bytes, record: bytes, signature: bytes) -> bool:
        if len(public_key) != 16 + params.N or record not in closed.records:
            return False
        block = closed.records.index(record)
        read = closed.read(block)
        sig = lms.HssSignature.decode(params, closed.signature)
        message = lms.period_root_message(sig.lower[:16], deployment, party, closed.attachment, closed.link,
                                          closed.period, closed.count, closed.root)
        return (attest.verify_opening(closed.root, closed.count, block, read.record, read.path)
                and lms.hss_verify(params, public_key, deployment, party, sig, message)
                and lms.verify_liveness(params, public_key, deployment, party, closed.attachment, closed.link,
                                        closed.period, block, lms.HssSignature.decode(params, signature)))

    return verify
