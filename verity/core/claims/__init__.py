"""The vocabulary of security claims, each with a stable id: what a configuration provides (guarantees: soundness, ZK,
non-interactive or designated-verifier, transparent setup), what that rests on (assumptions) and the setting it is proven
in (models).

Id format (shared with the docs site's assumptions registry): an assumption over a primitive is ``<property>/<instance>``,
composing a :data:`PROPERTIES` entry with an :data:`INSTANCES` entry -- ``cr/sha-512`` is collision resistance of SHA-512,
labelled ``CR · SHA-512``, and an algebraic instance makes it ``algebraic``; every other claim is a bare kebab-case id
(``logup``, ``live-verifier``) in :data:`ENTRIES`, and never contains ``/``.  This module is the source of the ids a render
cites; ``bench.views --format entities`` carries each cited entry, which is what the site imports.  :func:`get` resolves either.  Only what a render cites is here; a new claim is a new
entry, and an id, once cited, is never renamed.  Stdlib only; imports no other core module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["ENTRIES", "INSTANCES", "PROPERTIES", "Entry", "Instance", "Property", "assumption", "get"]

Kind = Literal["guarantee", "assumption", "model"]
# ``conjectured`` is for mathematics only; ``measured``: an assumption about a device, falsified by a measurement (Daniel, 8 Oct)
Standing = Literal["standard", "algebraic", "proven", "conjectured", "measured"]


@dataclass(frozen=True, slots=True)
class Entry:
    """One claim: ``kind``, full ``name``, short ``label``, and for an assumption its ``standing``."""

    id: str
    kind: Kind
    name: str
    label: str
    standing: Standing | None = None


@dataclass(frozen=True, slots=True)
class Property:
    """A security property of a primitive (collision resistance, ...), defined once for any instance."""

    id: str
    name: str
    label: str


@dataclass(frozen=True, slots=True)
class Instance:
    """A primitive a property is assumed of; ``algebraic``: an algebraic hash (Poseidon2, Ajtai digests)."""

    id: str
    name: str
    algebraic: bool = False


PROPERTIES: dict[str, Property] = {p.id: p for p in (
    Property("cr", "Collision resistance", "CR"),
    # success linear in the finder's expected time (Chiesa-Dall'Agnol-Guan-Spooner-Yogev, ePrint 2024/1434, Thm 2); the constant
    # is the assumption's statement (A2 in Security/Proofs/Flock/Soundness/ASSUMPTIONS.md), not restated here
    Property("ecr", "Expected-time collision resistance", "ECR"),
    Property("xof", "Extendable-output-function security", "XOF"),
    Property("sis", "Ring-SIS binding", "SIS"),
    # A3 in Security/Proofs/Flock/Soundness/ASSUMPTIONS.md, for the unit draw's bytes
    Property("uniform", "Independent uniform output", "Uniform"),
    # A4 there, for verity.ml.randomness's keyed streams (prf/sha-512): SHA-512's compression function as a PRF keyed by
    # its chaining value (the cascade over prefix-free inputs, Bellare-Canetti-Krawczyk 1996) and as a dual PRF keyed by the
    # secret bytes in its first message block; its eta is stated there (at most 2^-128 for the chain's tests, for a 32-byte key)
    Property("prf", "Pseudorandom function", "PRF"),
    # of a signature scheme: no forgery on a new message after chosen-message queries, each one-time key used once
    Property("euf-cma", "Existential unforgeability under chosen-message attack", "EUF-CMA"),
)}

INSTANCES: dict[str, Instance] = {i.id: i for i in (
    Instance("sha-256", "SHA-256"),
    Instance("sha-512", "SHA-512"),
    Instance("blake3", "BLAKE3"),
    Instance("blake2b", "BLAKE2b"),
    Instance("shake-256", "SHAKE-256"),
    # PoUW's hiding tile leaf (row pouw/tile-h0/v1): TurboSHAKE128 at L = 32 bytes, domains 0x21 (chunks) and 0x22 (chunk digests)
    Instance("turboshake128", "TurboSHAKE128"),
    Instance("poseidon2-babybear", "Poseidon2-BabyBear", algebraic=True),
    Instance("poseidon2-koalabear", "Poseidon2-KoalaBear", algebraic=True),
    Instance("ajtai-n64", "Ajtai n = 64", algebraic=True),
    Instance("ajtai-n128", "Ajtai n = 128", algebraic=True),
    Instance("io-getrandombytes", "IO.getRandomBytes"),
    # A5: the PoUW window's secret, secrets.token_bytes(32), fresh per window (#423)
    Instance("python-secrets", "Python's secrets"),
    # A6: each round's coins from Flock's Rust coin server (/dev/urandom outside seed mode), drawn after the prover's message
    Instance("flock-coin-server", "Flock's coin server"),
    # verity.ml.crypto.lms: LMS (RFC 8554) over SHA-512, its whole 64-byte output, the network certifier's and the
    # memory challenger's signatures; its EUF-CMA rests on SHA-512 itself: multi-target second-preimage resistance under RFC
    # 8554's prefixes (its §9) and, for keys derived from a seed (its Appendix A), SHA-512 as a PRF keyed by the seed
    Instance("lms-sha512-n64", "LMS over SHA-512"),
)}

ENTRIES: dict[str, Entry] = {e.id: e for e in (
    Entry("statistical-soundness", "guarantee", "Statistical soundness", "Statistical soundness"),
    Entry("computational-soundness", "guarantee", "Computational soundness (argument)", "Computational soundness"),
    Entry("malicious-verifier-zk", "guarantee", "Zero knowledge against a malicious verifier", "Malicious-verifier ZK"),
    Entry("hvzk", "guarantee", "Honest-verifier zero knowledge", "HVZK"),
    Entry("no-zk", "guarantee", "No zero knowledge", "No ZK"),
    Entry("statistical-hiding", "guarantee", "Statistically hiding commitments, given uniform salts", "Statistical hiding"),
    Entry("non-interactive", "guarantee", "Non-interactive, transferable proof", "Non-interactive"),
    Entry("designated-verifier", "guarantee", "Designated-verifier proof (convinces only the live verifier)", "Designated verifier"),
    Entry("rs-proximity-udr", "assumption", "Reed–Solomon proximity gaps, unique-decoding regime", "RS proximity · UDR", "proven"),
    Entry("rs-proximity-johnson", "assumption", "Reed–Solomon proximity gaps up to the Johnson bound", "RS proximity · Johnson", "proven"),
    Entry("logup", "assumption", "LogUp fractional-sum soundness", "LogUp", "proven"),
    Entry("live-verifier", "model", "Live verifier: the verifier's coins arrive online during the proof", "Live verifier"),
    Entry("fiat-shamir", "model", "Fiat–Shamir (non-interactive) in the random-oracle model", "Fiat–Shamir"),
    Entry("common-reference-string", "model",
          "Common reference string: a public parameter treated as a uniform draw fixed before use, such as the pinned keys of "
          "hm96-sha256/v1 and hm96-sha512/v1 (their hiding bound holds except for a 2^-64 fraction of keys)", "CRS"),

    Entry("random-oracle", "assumption", "A hash behaves as a random oracle", "Random oracle", "standard"),
    Entry("hash-derived-key", "assumption",
          "A key derived as counter-mode hash output over a public label is not in the 2^-64 fraction of keys for which the "
          "pinned-key hiding bound (N * 2^-192) fails: hm96's pinned keys are H(label || u32be i), with H the instance's hash "
          "(SHA-256 for hm96-sha256/v1, SHA-512 for hm96-sha512/v1)", "Hash-derived key", "standard"),
    Entry("honest-verifier", "assumption", "The verifier follows the protocol", "Honest verifier", "standard"),
    # `Verity.Assumptions.TrustedNebiusHost` (Security/Models/Core/Assumptions.lean) states its part about one run; no
    # attestation hardware backs it
    Entry("trusted-nebius-host", "assumption",
          "An auditor-trusted device (a network certifier, the memory challenger) is the auditor's hardware on a machine "
          "its host provider runs and the prover doesn't control; it runs its stated code on its Nebius host, and its host "
          "never restores an earlier copy of the state the device keeps there", "Trusted Nebius host", "standard"),
    Entry("transparent-setup", "guarantee", "Transparent setup (no trusted setup)", "Transparent setup"),
    # `Pous.EndToEnd.EndToEndRing`'s steps to the deployed system (Security/Properties/Pous/EndToEnd.lean), beside
    # prf/sha-512 for the memory challenger's index streams: the model, the numbers and the instantiation
    Entry("pous-ring-device", "model",
          "PoUS ring model of the server: the deployed server is a ring-model server whose state is bounded between "
          "setup, the session and its rounds, with one session per audit and the memory challenger's coins drawn online, "
          "after its state is fixed", "PoUS ring device"),
    # its falsification test is the measurement behind ROOT_CALL_NS (verity/core/protocols/pous/PROTOCOL.md, P2's SeqRoot):
    # the timing red team's floor, note:20261004T2102Z-report-relay-p2-redteam-timing
    # (art:23a323f313d2193873c59ee8ab59a0f1197167a0b4c6d008346d771c3e72de57), and the 8-core cooperation run
    # r20260929-060834-0ec4 (art:b92b6d56fa12cc8e04b3f6843991c2e76f49366adc89140a96c33842f70b41ed); a faster squaring on the
    # stated device falsifies it
    Entry("pous-ring-speed", "assumption",
          "PoUS ring speed: on the stated device (up to 8 cores of one chiplet cooperating, each at least as slow as a "
          "Zen 5 core at 5.7 GHz with full-rate IFMA), within Δ + RTT (at most 854 µs) no server finishes a 16,447th "
          "sequential multiplication round (one root's depth, so D = 16,446), more than Q = 2^20 key queries or more than "
          "L = 2^64 ring operations per answer; measured: a cooperating 8-core 16,448-bit squaring takes at least 103.6 ns, "
          "so 16,447 rounds take 1.995× Δ + RTT, and 1.64× and 1.25× under the timing red team's sync model",
          "PoUS ring speed", "measured"),
    Entry("pous-ring-instantiation", "assumption",
          "P2 with the identity labeling, SHA-512-CTR keys and its published Merkle root of C* is no easier to answer "
          "than p2ring with a uniformly random odd labeling and uniformly random keys", "PoUS ring instantiation",
          "conjectured"),
)}


def assumption(prop: str, instance: str) -> Entry:
    """The composed assumption ``<prop>/<instance>``; ``KeyError`` for an unknown property or instance."""
    p, i = PROPERTIES[prop], INSTANCES[instance]
    return Entry(f"{p.id}/{i.id}", "assumption", f"{p.name} of {i.name}", f"{p.label} · {i.name}", "algebraic" if i.algebraic else "standard")


def get(id: str) -> Entry:
    """The entry with this id, composed or not; ``KeyError`` when there is none."""
    if id in ENTRIES:
        return ENTRIES[id]
    prop, sep, instance = id.partition("/")
    if not sep:
        raise KeyError(id)
    return assumption(prop, instance)
