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
Standing = Literal["standard", "algebraic", "proven", "conjectured"]


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
    Property("xof", "Extendable-output-function security", "XOF"),
    Property("sis", "Ring-SIS binding", "SIS"),
)}

INSTANCES: dict[str, Instance] = {i.id: i for i in (
    Instance("sha-256", "SHA-256"),
    Instance("sha-512", "SHA-512"),
    Instance("blake3", "BLAKE3"),
    Instance("blake2b", "BLAKE2b"),
    Instance("shake-256", "SHAKE-256"),
    Instance("poseidon2-babybear", "Poseidon2-BabyBear", algebraic=True),
    Instance("poseidon2-koalabear", "Poseidon2-KoalaBear", algebraic=True),
    Instance("ajtai-n64", "Ajtai n = 64", algebraic=True),
    Instance("ajtai-n128", "Ajtai n = 128", algebraic=True),
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
          "Common reference string: a public parameter treated as a uniform draw fixed before use, such as hm96-sha256/v1's pinned key "
          "(its hiding bound holds except for a 2^-64 fraction of keys)", "CRS"),

    Entry("random-oracle", "assumption", "A hash behaves as a random oracle", "Random oracle", "standard"),
    Entry("honest-verifier", "assumption", "The verifier follows the protocol", "Honest verifier", "standard"),
    Entry("transparent-setup", "guarantee", "Transparent setup (no trusted setup)", "Transparent setup"),
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
