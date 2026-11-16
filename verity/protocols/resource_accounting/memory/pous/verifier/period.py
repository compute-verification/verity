"""The verifier's side of one PoUS period on one node: the run's check, with the answer of PoUS's component of onsite
Verity (`verity/pous/p2/v4/run`, `PousCheck.RunProtocol` in `check/`). The prover's side, the setup, the session and the
period's subprotocol, is `verity/protocols/resource_accounting/memory/pous/prover/period_prover.py`, which ends each period here.

The verifier judges the run, setup's acceptance of `C*` publishing exactly the run's commitment (`tie`) and then
`run.check_at`, and answers in the core interface's canonical form (`answer`); the case (`case`) carries setup's own
inputs (`setup_case`, read back by `setup_inputs`).

What an accepted run certifies is `certify`'s, as `PousCheck.RunProtocol.certify` computes it: `s_bits = ⌊ρ·blocks·w⌋`
at `eps = δ + 2^eps_log2` = 1/100 + 2^-128 when a certificate covers the point (`certificate`: P2's registered one at
`run.DEPLOYED`, a 1 GiB `C*`, else P2's proved family `p2.family_certificate`, any `2^8` to `2^23` blocks at the
registered width and prime, else the per-period guarantee `p2.wide_certificate`, `periodHeldAt`, to `2^27` blocks, which
holds Demo B's `C*` of 1038 segments, 1,060,008,348,186 bits), else 0 and 0 with the profile naming each gap. More
challenges than a certificate's `k` keep its `eps`: at `k ≥ 320` it stays `1% + 2^-128`, and
`p2.wide320_certificate`'s `2^-20` (`ObligationWide320`, from the open conjecture `P2RingRoundSecureWide`) is no
run's certificate.
`point_for` gives the smallest such point holding a store, 256 blocks (526 KB) at least; the small point
(`point_for(size, W, C0)`, 1,024-bit blocks) certifies nothing.

`answer_signed` is the run on the memory attester's signed setup (`verity/pous/p2/v4/signed-run`,
`PousCheck.SignedRun`): `run.check_signed_at` in place of `tie` and `run.check_at`, its case (`case_signed`) the
session, the setup record the session's key signed (`memory_attester.sign_setup_record`: the published outputs the
memory attester's device was given) and the prover's openings (`openings_of`, from its table), a few megabytes at a
served store where `case` carries the table and `C*`. It certifies nothing, and its profile says why (`SIGNED_GAPS`);
setup's acceptance is the open premise `SETUP_ACCEPTED`, among its certificate's hypotheses.

`trusted-device`'s premise, that the memory attester runs on a machine the developer doesn't control, holds only
under a committed configuration, for a session signed under a key its `[[pin]]` entries give the memory attester
(`period.json`'s `deployment`), whatever the host names say; the subprotocol says which (`challenger_session.placement`).
The setup's salt is still sampled on the node, and its check runs there too: every verdict rests on setup checked by
the developer, not the auditor, which the subprotocol says beside the premise.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.commitments.identity import canonical_json_bytes

from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc

from . import protocol as pous_protocol
from . import run as R
from .schemes import p2 as P2
from .verifier import DIGEST_BYTES, MerkleTree, row_leaf

PROTOCOL = "verity/pous/p2/v4/run"
"""`PousCheck.RunProtocol.ID`; a test signer's runs answer under `memory_attester.TEST_PREFIX` + it."""
SIGNED_PROTOCOL = "verity/pous/p2/v4/signed-run"
"""`PousCheck.SignedRun.ID`: the run on the memory attester's signed setup (`answer_signed`)."""
SIGNED_GAPS = ("the signed run's guarantee is not proved: this check takes the memory attester's signed setup where "
               "the run's check reads the store",
               "setup's check is not yet a proof, and it ran on the developer's node")
"""`PousCheck.SignedRun.GAPS`: why the signed run certifies nothing, after the point's own gaps."""
SETUP_ACCEPTED = ("the store at the published root is the encoding of the registered weights under the published "
                  "commitment and salt (SetupAccepted)")
"""`PousCheck.SignedRun.SETUP_ACCEPTED`: setup's acceptance, an open premise of the signed run, among its certificate's
hypotheses."""
SESSION_DOMAIN = "verity/pous-period/session/v1"
W, C0 = 1024, 105
"""The small point's width: a 1,024-bit prime `2^w − c0` of P2's ladder, 126 payload bytes a 128-byte block."""
K = P2.CERT_K
LIMIT_NS = P2.DELTA_NS + P2.RTT_ALLOWANCE_CAP_NS


def point_for(size: int, w: int = P2.POINT[0], c0: int = P2.POINT[1], k: int = K, limit_ns: int = LIMIT_NS) -> R.Point:
    """The smallest point at width `w` whose payload holds `size` bytes: one segment of up to 2^16 blocks, then more. At
    the registered width and prime, the default, it has at least `2^8` blocks (`p2.CERT_FAMILY_BLOCKS`), so with `k =
    105` and `Δ + RTT` at the cap `certificate` covers it to `2^27` blocks; `point_for(size, W, C0)` is the small point."""
    payload = (w - 16) // 8
    least = P2.CERT_FAMILY_BLOCKS[0] if (w, c0) == P2.POINT[:2] else 1
    blocks = max(least, -(-size // payload))
    segments = -(-blocks // (1 << 16))
    return R.Point(P2.P2Params(w, c0, -(-blocks // segments), segments), k, limit_ns)


def certificate(q: P2.P2Params) -> Any:
    """`PousCheck.RunProtocol.certificate`: P2's registered certificate when `q` is its point, segment count and all,
    else P2's proved family's (`p2.family_certificate`, `2^8` to `2^23` blocks), else the per-period guarantee's
    (`p2.wide_certificate`, to `2^27` blocks), else none."""
    registered = P2.P2(P2.P2Params(q.w, q.c0, q.blocks_per_segment, q.segments)).certificate
    if registered is not None and registered.segments == q.segments:
        return registered
    return P2.family_certificate(q) or P2.wide_certificate(q)


def gaps(point: R.Point) -> tuple[Any, list[str]]:
    """`PousCheck.RunProtocol.gaps`: the certificate at the point, and why it doesn't cover the run (none when it does)."""
    q = point.params
    cert = certificate(q)
    if cert is None:
        return None, [f"P2 has no certificate at (w, c0, blocks) = ({q.w}, {q.c0}, {q.blocks})"]
    return cert, list(cert.refusals(point.k, point.limit_ns))


def certify(point: R.Point) -> dict[str, Any]:
    """`PousCheck.RunProtocol.certify`'s public outputs: `{"certified", "eps", "s_bits"}`, eps an exact ratio."""
    cert, why = gaps(point)
    if cert is None or why:
        return {"certified": False, "eps": "0", "s_bits": 0}
    q = point.params
    eps = cert.delta + Fraction(1, 2 ** -cert.eps_log2)
    return {"certified": True, "eps": str(eps), "s_bits": q.blocks * q.w * cert.rho.numerator // cert.rho.denominator}


def profile(point: R.Point) -> dict[str, Any]:
    """`PousCheck.RunProtocol.profileJson`: the store and the audit the run was checked at, the certificate, every gap."""
    cert, why = gaps(point)
    q = point.params
    return {"block_bits": q.w, "blocks": q.blocks, "certificate": None if cert is None else cert.to_json(),
            "experimental": P2.EXPERIMENTAL, "gaps": why, "k": point.k, "limit_ns": point.limit_ns}


def point_json(point: R.Point) -> dict[str, Any]:
    q = point.params
    return {"k": point.k, "limit_ns": point.limit_ns,
            "params": {"blocks_per_segment": q.blocks_per_segment, "c0": q.c0, "segments": q.segments, "w": q.w}}


def point_of(doc: Mapping[str, Any]) -> R.Point:
    q = doc["params"]
    return R.Point(P2.P2Params(q["w"], q["c0"], q["blocks_per_segment"], q["segments"]), doc["k"], doc["limit_ns"])


def case(point: R.Point, session_id: bytes, trusted: Sequence[bytes], published: R.Setup, digests: bytes,
         session: R.ChallengePhaseRecord, setup_of: Mapping[str, Any]) -> dict[str, Any]:
    """The run protocol's case, `{"commitment": hex, "evidence": hex, "params": JSON}` (`PousCheck.RunProtocol`): its
    evidence carries the setup's own commitment and evidence, from `setup_of`, the setup protocol's case."""
    params = {"point": point_json(point), "session_id": session_id.hex(), "trusted": [t.hex() for t in trusted]}
    evidence = {"digests": digests.hex(), "session": session.to_json(),
                "setup": {"commitment": setup_of["commitment"], "evidence": setup_of["evidence"]}}
    return {"commitment": canonical_json_bytes(published.to_json()).hex(),
            "evidence": canonical_json_bytes(evidence).hex(), "params": params}


SetupInputs = tuple[bytes, bytes, Sequence[Sequence[bytes]]]
"""The setup's own inputs (`PousCheck.RunProtocol.SetupEvidence`): the server's commitment to `W`, the verifier's salt,
and `C*` segment by segment."""


def tie(point: R.Point, published: R.Setup, inputs: SetupInputs) -> None:
    """`PousCheck.RunProtocol.tie`: setup accepts `C*` against the commitment and salt at the point's parameters (the
    public-encoder path, `protocol.verifier_key`) and publishes exactly `published`, the run's commitment; else raises
    `run.Refused`."""
    commitment, salt, store = inputs
    q = point.params
    try:
        vk = pous_protocol.verifier_key(P2.P2(q), commitment, salt, mode=pous_protocol.PUBLIC, encoded=store)
    except ValueError as e:
        raise R.Refused(f"setup: {e}") from None
    if R.Setup(q.blocks, commitment, vk.root, salt) != published:
        raise R.Refused("the run's commitment is not what its setup published")


def case_signed(point: R.Point, session_id: bytes, trusted: Sequence[bytes], published: R.Setup,
                session: R.ChallengePhaseRecord, signed_setup: mc.SignedSetupRecord,
                openings: Sequence[R.Opening]) -> dict[str, Any]:
    """The signed run protocol's case (`PousCheck.SignedRun`): `case`'s params and commitment, and its evidence the
    session, the memory attester's signed setup record and one opening per round, in place of the table and the
    setup's own inputs."""
    params = {"point": point_json(point), "session_id": session_id.hex(), "trusted": [t.hex() for t in trusted]}
    evidence = {"paths": [o.to_json() for o in openings], "session": session.to_json(), "setup": signed_setup.to_json()}
    return {"commitment": canonical_json_bytes(published.to_json()).hex(),
            "evidence": canonical_json_bytes(evidence).hex(), "params": params}


def openings_of(point: R.Point, salt: bytes, digests: bytes, rounds: Sequence[mc.Round]) -> list[R.Opening]:
    """The prover's openings for `check_signed`, from its table of `C*` (`digests`, one row digest a block): for each
    round, its index's row and that row's leaf's path in the tree of the table's leaves under `salt`."""
    q = point.params
    rows = [digests[DIGEST_BYTES * i:DIGEST_BYTES * (i + 1)] for i in range(q.blocks)]
    tree = MerkleTree([row_leaf(q, salt, i, r) for i, r in enumerate(rows)])
    return [R.Opening(rows[r.index], tree.path(r.index)) for r in rounds]


def profile_signed(point: R.Point) -> dict[str, Any]:
    """`PousCheck.SignedRun.profileJson`: `profile`, its gaps the point's and then `SIGNED_GAPS`, and its certificate's
    hypotheses, when it has a certificate, followed by `SETUP_ACCEPTED`."""
    cert, why = gaps(point)
    cert = None if cert is None else replace(cert, hypotheses=(*cert.hypotheses, SETUP_ACCEPTED))
    return profile(point) | {"certificate": None if cert is None else cert.to_json(), "gaps": why + list(SIGNED_GAPS)}


def answer_signed(point: R.Point, schemes: Mapping[str, mc.Verify], trusted: Collection[bytes], published: R.Setup,
                  session: R.ChallengePhaseRecord, signed_setup: mc.SignedSetupRecord, openings: Sequence[R.Opening],
                  session_id: bytes, protocol: str = SIGNED_PROTOCOL) -> dict[str, Any]:
    """The signed run protocol's answer (`Core.Answer.canonical`) on decoded inputs: `run.check_signed_at`, then nothing
    certified (`eps` 0, `s_bits` 0) and `profile_signed`."""
    digest = hashlib.sha512(canonical_json_bytes(published.to_json())).hexdigest()
    try:
        R.check_signed_at(point, schemes, trusted, published, session, signed_setup, openings, session_id)
    except R.Refused as e:
        return {"commitment": digest, "profile": None, "protocol": protocol, "public": None, "reason": str(e),
                "verdict": "reject"}
    return {"commitment": digest, "profile": profile_signed(point), "protocol": protocol,
            "public": {"certified": False, "eps": "0", "s_bits": 0}, "reason": None, "verdict": "accept"}


def answer(point: R.Point, schemes: Mapping[str, mc.Verify], trusted: Collection[bytes], published: R.Setup,
           digests: bytes, session: R.ChallengePhaseRecord, session_id: bytes, protocol: str = PROTOCOL, *,
           setup_inputs: SetupInputs) -> dict[str, Any]:
    """The run protocol's answer (`Core.Answer.canonical`) on decoded inputs: `tie` on the setup's own inputs, then
    `run.check_at`, then `certify`."""
    digest = hashlib.sha512(canonical_json_bytes(published.to_json())).hexdigest()
    try:
        tie(point, published, setup_inputs)
        R.check_at(point, schemes, trusted, published, digests, session, session_id)
    except R.Refused as e:
        return {"commitment": digest, "profile": None, "protocol": protocol, "public": None, "reason": str(e),
                "verdict": "reject"}
    return {"commitment": digest, "profile": profile(point), "protocol": protocol, "public": certify(point),
            "reason": None, "verdict": "accept"}


def setup_case(q: P2.P2Params, commitment: bytes, salt: bytes, encoded: Sequence[Sequence[bytes]]) -> dict[str, Any]:
    """The setup protocol's case (`PousCheck.SetupProtocol`): the manifest, `commitment ‖ salt`, and `C*` framed."""
    u32 = lambda n: n.to_bytes(4, "big")  # noqa: E731
    evidence = u32(len(encoded)) + b"".join(u32(len(seg)) + b"".join(u32(len(b)) + b for b in seg) for seg in encoded)
    return {"commitment": (commitment + salt).hex(), "evidence": evidence.hex(), "params": q.manifest()}


def setup_inputs(setup_of: Mapping[str, Any]) -> SetupInputs:
    """The setup's own inputs from its case's `commitment` and `evidence`, read as `SetupProtocol`'s decoders read them
    (each count at most a quarter of the bytes left, nothing after the last segment); `ValueError` on bytes they refuse."""
    c, b = bytes.fromhex(setup_of["commitment"]), bytes.fromhex(setup_of["evidence"])
    if len(c) < DIGEST_BYTES:
        raise ValueError(f"the commitment to W is 64 bytes, then the salt; {len(c)} bytes is too short")
    pos = 0

    def take(n: int) -> bytes:
        nonlocal pos
        if pos + n > len(b):
            raise ValueError(f"C* is truncated at byte {pos}")
        pos += n
        return b[pos - n:pos]

    def count() -> int:
        n = int.from_bytes(take(4), "big")
        if 4 * n > len(b) - pos:
            raise ValueError(f"a count {n} exceeds the bytes left")
        return n

    store = [[take(int.from_bytes(take(4), "big")) for _ in range(count())] for _ in range(count())]
    if pos != len(b):
        raise ValueError(f"{len(b) - pos} bytes follow the last segment")
    return c[:DIGEST_BYTES], c[DIGEST_BYTES:], store


@dataclass(frozen=True)
class Run:
    """A subprotocol's run over one monitoring period, PoUS's or the bandwidth audit's: its answer in the core
    interface's canonical form, the core case its Lean replays to that answer, for a run on a setup (PoUS's), the
    setup protocol's case that published its commitment, and for PoUS's, the SHA-512 of the key its memory attester
    signed the session under (`challenger_session.placement`'s ``key_sha512``)."""

    answer: dict[str, Any]
    case: dict[str, Any]
    setup_case: dict[str, Any] | None = None
    key_sha512: str | None = None

    @property
    def verdict(self) -> str:
        return self.answer["verdict"]

    def public(self, key: str) -> int:
        """What the run certifies under `key` (PoUS's `s_bits`, the bandwidth audit's `q_bytes`): 0 unless accepted."""
        return self.answer["public"][key] if self.verdict == "accept" else 0

    @property
    def eps(self) -> Fraction:
        """The error what it certifies holds at: 0 (vacuous) when nothing is certified."""
        return Fraction(self.answer["public"]["eps"]) if self.verdict == "accept" else Fraction(0)


def serve(exe: str | os.PathLike, cases: Sequence[Mapping[str, Any]], *args: str) -> list[dict[str, Any]]:
    """`EXE [ARGS] serve` on `cases`, one canonical JSON line each: the Lean's answers (`Core.Protocol` CLI). The lines go
    as bytes, so a served case (gigabytes) is held once beside its own JSON, not again as text."""
    lines = b"".join(canonical_json_bytes(dict(c)) + b"\n" for c in cases)
    p = subprocess.run([str(exe), *args, "serve"], input=lines, capture_output=True, check=True)
    return [json.loads(x) for x in p.stdout.splitlines()]


def decided_by(exe: str | os.PathLike | None) -> dict[str, str]:
    """Who decided a case, as a piece's summary names it under `decided_by`: the program, `exe`'s file name (a
    component's Lean, as `serve` runs it) or `python` with none, and the host it ran on, this one."""
    return {"program": "python" if exe is None else Path(exe).name, "host": socket.gethostname()}


#: why a PoUS answer no `pous-run-protocol` gave is undecided where the Python accepted (`unjudged`)
NO_JUDGE = "no pous-run-protocol answered the case: the Lean decides PoUS, and the Python's accept decides nothing alone"


def unjudged(answer_: Mapping[str, Any], reason: str = NO_JUDGE) -> dict[str, Any]:
    """A Python answer with no Lean to ask: its reject or undecided, which the Python may give alone, and undecided for
    its accept (`reason`), with no public outputs and no profile."""
    if answer_["verdict"] != "accept":
        return dict(answer_)
    return dict(answer_) | {"profile": None, "public": None, "reason": reason, "verdict": "undecided"}


DEPLOYMENT = ("configuration", "pins")
PIN_FIELDS = ("key-sha512", "party", "protocol", "scheme")


def deployment(doc: Mapping[str, Any]) -> dict[str, Any]:
    """`period.json`'s `deployment`, what `trusted-device`'s premise reads (`memory_attester.trusted_device`):
    `configuration`, the hash of the configuration the run is under (`verity config hash`), and `pins`, that
    configuration's `[[pin]]` entries, with `committed`, the hashes `tools/verity/cli/verity_cli/configurations.toml` commits. With none, the
    premise reads rehearsal; an unknown key or a malformed pin raises."""
    got = doc.get("deployment") or {}
    if unknown := set(got) - set(DEPLOYMENT):
        raise ValueError(f"period.json's deployment has unknown keys {sorted(unknown)}: it takes {sorted(DEPLOYMENT)}")
    pins, configuration = got.get("pins") or [], got.get("configuration")
    if not (isinstance(pins, list) and all(isinstance(e, dict) and sorted(e) == sorted(PIN_FIELDS)
                                           and all(type(v) is str for v in e.values()) for e in pins)
            and (configuration is None or type(configuration) is str)):
        raise ValueError(f"period.json's deployment: configuration a hash and pins a list of pins, each "
                         f"{list(PIN_FIELDS)} as strings, not {got!r}")
    return {"pins": pins, "configuration": configuration, "committed": mc.committed()}


SETUP_WORKERS = 32
"""The processes the subprotocol encodes a node's `C*` in, and the memory attester's side makes its key in, at most, within the
CPUs each may run on (`period_prover.setup_store`; `onsite.device_key`, from `challenger_party.main`)."""


__all__ = ["DEPLOYMENT", "NO_JUDGE", "PIN_FIELDS", "PROTOCOL", "SETUP_ACCEPTED", "SETUP_WORKERS", "SIGNED_GAPS",
           "SIGNED_PROTOCOL", "Run", "SetupInputs", "answer", "answer_signed", "case", "case_signed", "certificate",
           "certify", "deployment", "gaps", "openings_of", "point_for", "point_of", "profile", "profile_signed",
           "serve", "setup_case", "setup_inputs", "tie", "unjudged"]
