"""The public beacon: drand's quicknet, pinned and verified (Daniel named it; `beacon-unpredictability`, the assessor's A
for the verifier path, `audit.Epoch.start` and `pearl_c_work.audit`, on its three use conditions below: research store
internal/pouw/red-team/ratings.md 21:48Z).

- **Verify each round.** A `Round` carries drand's threshold BLS signature on the round number, and `verify` checks it
  against the pinned group key (`QUICKNET`). No relay's word is trusted. A round's value is SHA-256 of its signature,
  so once the key is fixed nobody can bias it.
  - The scheme is `bls-unchained-g1-rfc9380`: the message is SHA-256 of the round as 8 bytes big-endian, hashed to G1
    under RFC 9380's DST, and the key is in G2 (`bls12_381`).
  - Predicting a round before it is published needs t of the group's n operators' partial signatures.
- **The round rule, read from the clock.** A round follows an event only if it was not due until at least `MARGIN`
  rounds after the round due when the event was recorded. That round is computed from the chain's genesis and period
  at the recording time (`round_at`), never from "the latest round seen", which can lag the schedule under relay
  latency. The margin of 2 rounds (6 s) covers clock skew and early publication.
  - `check_after` states the rule. `audit.Epoch.start` applies it to the salt (after the weight roots),
    `pearl_c_work.audit` to the draw (after the commitment), and the fork's rotation key takes the same rule after its
    registration.
- **Not covered:** liveness (if quicknet stalls, audits wait) and quantum adversaries.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from . import bls12_381 as bls

DST = b"BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_"
MARGIN = 2


@dataclass(frozen=True)
class Chain:
    """A drand chain's pins: its chain and group hashes, scheme, period (s), genesis (UNIX s), group key, and the group's
    threshold t of n operators."""

    name: str
    chain_hash: str
    group_hash: str
    scheme: str
    period: int
    genesis: int
    public_key: bytes
    key: tuple                      # the group key decompressed into G2, checked against `public_key` by the tests
    threshold: int                  # t, the trust statement's parameter: predicting a round needs t colluding operators
    nodes: str                      # n, as the League of Entropy has published it (a reshare can change it)


#: drand quicknet, as published at api.drand.sh (chain info, fetched 1:45 PM PDT on 30 Sep 2026 by the assessor, again at
#: 1:52 PM PDT, and confirmed against the live /info by compute-accounting at 2:32 PM PDT). The public API doesn't serve
#: the group file.
#: - t = 12 is the League of Entropy's published threshold, and only t enters the trust statement.
#: - n is 22 at launch (drand's "quicknet is live", 16 Oct 2023) and 23 per Filecoin FIP-0063.
#: - A reshare keeps the key but can change t and n, so both need re-checking after one.
QUICKNET = Chain(
    name="quicknet",
    chain_hash="52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971",
    group_hash="f477d5c89f21a17c863a7f937c6a6d15859414d2be09cd448d4279af331c5d3e",
    scheme="bls-unchained-g1-rfc9380",
    period=3,
    genesis=1692803367,
    public_key=bytes.fromhex(
        "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921c"
        "c22b0e17e63aaf4bcb5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"),
    key=((2020076495541918814736776030432697997716141464538799718886374996015782362070437455929656164150586936230283253179482,
          586231829158603972936263795113906716025771067144631327612230935308837823978471744132589153452744931590357767971921),
         (2211131032793566610140590394248723125180008851249985816060296934936346156689749396593077297304404280871596812489331,
          254368178680493348045912140930876774076610928954039097415743904368045981295538168002596692724553813124605269104267)),
    threshold=12,
    nodes="22 at launch, 23 per FIP-0063",
)


@dataclass(frozen=True)
class Round:
    """One published round: its number and its signature (48 bytes, compressed G1)."""

    round: int
    signature: bytes

    @property
    def value(self) -> bytes:
        return hashlib.sha256(self.signature).digest()


def message(round: int) -> bytes:
    return hashlib.sha256(round.to_bytes(8, "big")).digest()


def verify(chain: Chain, rnd: Round) -> bool:
    """Whether `rnd` is the chain's round: its signature verifies on the round number under the pinned key."""
    return rnd.round >= 1 and bls.verify_g1(chain.key, message(rnd.round), bytes(rnd.signature), DST)


def round_at(chain: Chain, unix_time: float) -> int:
    """The round due at `unix_time` by the chain's schedule: round 1 at genesis, round r at genesis + (r − 1)·period."""
    if unix_time < chain.genesis:
        return 0
    return int((unix_time - chain.genesis) // chain.period) + 1


def check_after(chain: Chain, rnd: Round, recorded_at: float) -> str | None:
    """None if `rnd` is a verified round of `chain` at least `MARGIN` rounds after the one due at `recorded_at` (the
    verifier's clock when it recorded what the round must follow), else the reason."""
    due = round_at(chain, recorded_at)
    if rnd.round < due + MARGIN:
        return (f"beacon round {rnd.round} is not at least {MARGIN} rounds after round {due}, the round due when what "
                f"it follows was recorded")
    if not verify(chain, rnd):
        return f"beacon round {rnd.round}'s signature does not verify under {chain.name}'s key"
    return None
