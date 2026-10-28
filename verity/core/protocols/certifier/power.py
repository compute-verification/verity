"""Power certificates: the power certifier's signed reading of one channel's energy over one block's span
(record v1).

The power certifier is onsite's power sub-protocol's auditor-trusted device: an inline meter or a metered PDU
outlet on a machine the prover doesn't control, never the GPU's own sensor or the prover's BMC. Trust in it and its host
is the named ``trusted-nebius-host`` assumption, as for the network certifier, whose certificate machinery it shares: the
same hiding signed digest (`certificate.certificate_digest`, the certifier's fresh salt), LMS over SHA-512 (n = 64) and the
fail-closed signer (`certificate.Signer`). Its span is one block's span of the run, and once per span it signs one **power certificate**
over the hiding digest of the span's record. The record and the salt are witness: they enter proofs and never reach the auditor in the clear.

**The record (v1).**

    R = u16be len(channel) ‖ channel ‖ u64be interval ‖ u32be Δ_ms ‖ u8 status ‖ parameters (64 bytes)
      ‖ u64be E_µJ ‖ u32be samples
    record digest = tagged_sha512("verity-power-certifier/record/v1", R)
    signed digest = SHA-512("verity/net-cert/v1" ‖ salt_c ‖ record digest)      (`certificate.certificate_digest`)

`channel` names the metered outlet, `interval` the span's index and Δ_ms its length, `status` is `PowerStatus`'s code,
`parameters` is the meter's manifest digest (`Meter.digest`: its source, sampling rate and accuracy class, and whether it
is a stand-in), `E_µJ` the energy the meter integrated over the span and `samples` the readings it integrated.

**The certificate** is the device's (`certificate.Issuer.sign_on`) on a channel of kind ``power``, the outlet
(`Outlet`). Its message is LMS's power-certificate role digest (`verity.ml.crypto.lms.power_certificate_message`):

    message = SHA-512( [tag] ‖ I ‖ [ [session] ‖ [outlet] ‖ u64be span ] ‖ [signed digest] ),
              tag = "verity/power-certificate-message/v1", [x] = u64be len(x) ‖ x

so a power certificate binds its key, certifier session, outlet and span, and never verifies as a link's, whose tag
differs. **The issuer** (`PowerIssuer`) integrates the meter's samples over the span (`integrate`), builds the head
under the meter's manifest, draws the salt from the device's coins and signs; a refused signature leaves the span
without a certificate (fail-closed). The prover gets the head and the salt as witness (`CertifiedPower`).

**The threshold.** Span i passes when the certified energy, less the public floors of the work it accounts and a
public tolerance, leaves at most the bar's share of hidden capacity:

    E − Σ_j e_j·N_j − τ₀·Δ − ρ·E − P₀·Δ ≤ β·e_h·θ·Δ

over the span's committed counts N_j (MACs per declared operand class, compressions, forming units), with the floors
e_j, idle P₀, the hidden-MAC floor e_h, the capacity θ, the bar β and the tolerance (τ₀, ρ) public (`Threshold`). It is the
served bound's U = (E − floors)/(e_h·θ·wall) ≤ β, per span and with the tolerance. `power_check` proves it in
integers: energies in 2⁻⁴⁰ µJ and the gain 1 − ρ in 2⁻²⁰,

    gain · E_µJ · 2^(40 − 20) ≤ Σ_j floor_j · N_j + slack,

each quantity rounded against the prover (`Threshold.quantized`): the floors and the slack down, the gain up.
"""
from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar

from verity.ml.commitments.identity import canonical_json_bytes, tagged_sha512
from verity.ml.crypto import lms
from verity.core.protocols.certifier.certificate import (DIGEST_BYTES, SALT_BYTES, ChannelCertificate, Issuer,
                                                         PublicKey, certificate_digest)

RECORD_TAG = "verity-power-certifier/record/v1"
METER_TAG = "verity-power-certifier/meter/v1"
#: the fixed-point scales: energies in 2^-ENERGY_BITS µJ, the gain in 2^-GAIN_BITS
ENERGY_BITS = 40
GAIN_BITS = 20
#: the widths `power_check` reads the public threshold in: a floor, the gain and the slack
FLOOR_WIDTH = 48
GAIN_WIDTH = 24
SLACK_WIDTH = 128
#: µJ in a joule, and ms in a second
UJ = 10 ** 6
MS = 1000


class PowerStatus(enum.Enum):
    COMPLETE = 0
    METER_UNAVAILABLE = 1
    SAMPLE_GAP = 2


def _u(value: int, width: int, name: str) -> bytes:
    if type(value) is not int or not 0 <= value < 1 << (8 * width):
        raise ValueError(f"{name} is an unsigned {8 * width}-bit integer, not {value!r}")
    return value.to_bytes(width, "big")


@dataclass(frozen=True, slots=True)
class Meter:
    """The meter's manifest, fixed before the first span: what it reads (`source`; a stand-in names itself
    `stand-in/...` and sets `stand_in`), how often (`rate_hz`) and how well (`accuracy_ppm` of the reading, and
    `offset_mw`)."""

    source: str
    rate_hz: int
    accuracy_ppm: int
    offset_mw: int
    stand_in: bool

    def __post_init__(self) -> None:
        if self.stand_in != self.source.startswith("stand-in/"):
            raise ValueError(f"a stand-in meter's source starts with 'stand-in/', and only a stand-in's: {self.source!r}")
        if type(self.rate_hz) is not int or self.rate_hz <= 0:
            raise ValueError(f"a meter samples at a positive integer rate, not {self.rate_hz!r} Hz")

    def digest(self) -> bytes:
        return tagged_sha512(METER_TAG, canonical_json_bytes({
            "source": self.source, "rate_hz": self.rate_hz, "accuracy_ppm": self.accuracy_ppm,
            "offset_mw": self.offset_mw, "stand_in": self.stand_in}))


@dataclass(frozen=True, slots=True)
class PowerHead:
    """A v1 power record: everything the record digest reads. Like the certificate's salt, it is witness."""

    channel: str
    interval: int
    interval_ms: int
    status: PowerStatus
    parameters: bytes
    energy_uj: int
    samples: int

    def __post_init__(self) -> None:
        if len(self.parameters) != DIGEST_BYTES:
            raise ValueError(f"the parameters digest is {DIGEST_BYTES} bytes")

    def encode(self) -> bytes:
        channel = self.channel.encode("utf-8")
        return b"".join([
            _u(len(channel), 2, "len(channel)"), channel, _u(self.interval, 8, "interval"),
            _u(self.interval_ms, 4, "Δ_ms"), _u(self.status.value, 1, "status"), self.parameters,
            _u(self.energy_uj, 8, "E_µJ"), _u(self.samples, 4, "samples"),
        ])

    def digest(self) -> bytes:
        return tagged_sha512(RECORD_TAG, self.encode())

    def signed_digest(self, salt_c: bytes) -> bytes:
        """The digest the span's certificate signs: the record digest under the certifier's salt."""
        if type(salt_c) is not bytes or len(salt_c) != SALT_BYTES:
            raise ValueError(f"a power certificate's salt is {SALT_BYTES} bytes")
        return certificate_digest(salt_c, self.digest())


@dataclass(frozen=True, slots=True)
class Outlet:
    """The power certifier's channel (`certificate.Channel`): one metered outlet, kind ``power``. Its message is
    `lms.power_certificate_message`, and the device's `window` argument is the span's index."""

    name: str

    kind: ClassVar[str] = "power"

    @property
    def id(self) -> tuple[str]:
        return (self.name,)

    def message(self, key_id: bytes, session: bytes, window: int, digest: bytes) -> bytes:
        return lms.power_certificate_message(key_id, session, self.name, window, digest)


def integrate(samples: Sequence[tuple[int, int]], span_ms: int, meter: Meter) -> tuple[PowerStatus, int, int]:
    """(status, E_µJ, readings) of one span from the meter's readings, each (t_ms, mW) with t_ms strictly increasing
    in 0 … `span_ms`, the span's start at 0. E is the upper sum: each stretch between two readings, and from the start
    to the first and the last to the end, at the larger reading at its ends, exact in mW·ms = µJ and so never below
    the energy of any power curve through the readings. No reading: `METER_UNAVAILABLE`. A stretch longer than
    2000/`rate_hz` ms, two of the meter's sampling steps: `SAMPLE_GAP`."""
    times = [t for t, _ in samples]
    if any(type(v) is not int or v < 0 for s in samples for v in s) or times != sorted(set(times)) or (
            times and times[-1] > span_ms):
        raise ValueError(f"readings are (t_ms, mW) of non-negative integers, t_ms increasing in 0 … {span_ms}")
    if not samples:
        return PowerStatus.METER_UNAVAILABLE, 0, 0
    max_gap = math.ceil(2 * MS / meter.rate_hz)
    edges = [(0, samples[0][1]), *samples, (span_ms, samples[-1][1])]
    energy = sum(max(pa, pb) * (tb - ta) for (ta, pa), (tb, pb) in zip(edges, edges[1:]))
    gap = any(tb - ta > max_gap for (ta, _), (tb, _) in zip(edges, edges[1:]))
    return (PowerStatus.SAMPLE_GAP if gap else PowerStatus.COMPLETE), energy, len(samples)


@dataclass(frozen=True, slots=True)
class CertifiedPower:
    """What the power certifier hands over per span: the head and the salt it drew for it (`salt_c`), both witness,
    and the certificate, the one public part, which signs the head's salted digest (`PowerHead.signed_digest`)."""

    head: PowerHead
    salt: bytes
    certificate: ChannelCertificate

    def refusal(self, key: PublicKey, session: bytes) -> str | None:
        """Why the host check fails under `key` in certifier session `session`, or None when it holds: the
        certificate names the head's outlet and span, is `key`'s, signs the head's digest under the salt, and its
        signature verifies (`ChannelCertificate.verify`)."""
        c, h = self.certificate, self.head
        if c.channel != Outlet(h.channel) or c.window != h.interval:
            return "the certificate names another outlet or span than the head"
        if c.key_id != key.key_id:
            return f"another key's, not {key.key_id.hex()[:16]}…"
        try:
            opened = h.signed_digest(self.salt)
        except ValueError:
            opened = None
        if c.digest != opened:
            return "the head and salt don't open the salted digest it signs"
        if not c.verify(key, session):
            return "its signature doesn't verify under the key in its session for the outlet and span it names"
        return None

    def verify(self, key: PublicKey, session: bytes) -> bool:
        return self.refusal(key, session) is None


@dataclass
class PowerIssuer:
    """The power certifier's signing step on the device (`device`, the certifier's `certificate.Issuer`: its signer,
    session and coins) under its meter's manifest (`meter`): one certificate per span."""

    device: Issuer
    meter: Meter

    def certify(self, outlet: str, span: int, span_ms: int,
                samples: Sequence[tuple[int, int]]) -> CertifiedPower | None:
        """The span's certified head, its status and energy `integrate`'s over `samples`, or None when the signer
        refuses (`(outlet, span)` is then in the device's `refused`). A span the meter didn't cover is signed with its
        status, which `power_check` refuses."""
        status, energy_uj, readings = integrate(samples, span_ms, self.meter)
        head = PowerHead(outlet, span, span_ms, status, self.meter.digest(), energy_uj, readings)
        salt = self.device.draw_salt()
        certificate = self.device.sign_on(Outlet(outlet), span, head.signed_digest(salt))
        return None if certificate is None else CertifiedPower(head, salt, certificate)


def counts_row(counts: tuple[int, ...]) -> bytes:
    """A span's committed counts as `power_check` opens them: each a u64 big-endian, in the threshold's order."""
    return b"".join(_u(n, 8, "a count") for n in counts)


@dataclass(frozen=True, slots=True)
class Threshold:
    """The public verifier parameters of the power threshold, in joules and seconds: a floor per counted kind (J per
    unit: a MAC of a declared operand class, a compression, a forming unit), the idle power P₀ (W), the hidden-MAC floor
    e_h (J per MAC), the capacity θ (MAC/s), the bar β, and the tolerance: an offset τ₀ (W) and a gain error ρ. Each is a
    `Fraction`, quoted under a named reading."""

    reading: str
    floors: tuple[Fraction, ...]
    idle_w: Fraction
    hidden_j: Fraction
    capacity: Fraction
    bar: Fraction
    offset_w: Fraction
    gain_error: Fraction

    def __post_init__(self) -> None:
        values = (*self.floors, self.idle_w, self.hidden_j, self.capacity, self.bar, self.offset_w, self.gain_error)
        if not self.floors or any(type(v) is not Fraction or v < 0 for v in values) or self.gain_error >= 1:
            raise ValueError("a threshold has a floor per counted kind and every parameter a Fraction >= 0, ρ < 1")

    def slack_j(self, interval_ms: int) -> Fraction:
        """(P₀ + τ₀ + β·e_h·θ)·Δ: the energy a span of Δ_ms may spend beyond its work's floors and the gain error."""
        return (self.idle_w + self.offset_w + self.bar * self.hidden_j * self.capacity) * Fraction(interval_ms, MS)

    def quantized(self, interval_ms: int) -> tuple[tuple[int, ...], int, int]:
        """(floors, gain, slack) as `power_check` reads them, each rounded against the prover: floors and slack down
        in 2⁻⁴⁰ µJ, the gain 1 − ρ up in 2⁻²⁰."""
        floors = tuple(math.floor(f * UJ * 2 ** ENERGY_BITS) for f in self.floors)
        gain = math.ceil((1 - self.gain_error) * 2 ** GAIN_BITS)
        slack = math.floor(self.slack_j(interval_ms) * UJ * 2 ** ENERGY_BITS)
        if any(f >= 1 << FLOOR_WIDTH for f in floors) or gain >= 1 << GAIN_WIDTH or slack >= 1 << SLACK_WIDTH:
            raise ValueError("a quantized floor, gain or slack exceeds its width")
        return floors, gain, slack

    def passes(self, head: PowerHead, counts: tuple[int, ...]) -> bool:
        """The threshold in integers, as `power_check` decides it."""
        if len(counts) != len(self.floors):
            raise ValueError(f"the threshold counts {len(self.floors)} kinds, not {len(counts)}")
        return passes_quantized(head.energy_uj, counts, *self.quantized(head.interval_ms))


def passes_quantized(energy_uj: int, counts: tuple[int, ...], floors: tuple[int, ...], gain: int, slack: int) -> bool:
    """gain · E_µJ · 2^(ENERGY_BITS − GAIN_BITS) ≤ Σ_j floor_j · N_j + slack: (1 − ρ)·E ≤ Σ_j e_j·N_j + slack in
    2⁻⁴⁰ µJ."""
    return gain * energy_uj << (ENERGY_BITS - GAIN_BITS) <= sum(f * n for f, n in zip(floors, counts, strict=True)) + slack
