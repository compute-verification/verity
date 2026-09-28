"""The library: the hardware and protocol objects core owns, the only objects the verifier holds (Project doc
`constant-api-public.md` §2.9, as Daniel ruled on 1 Oct 2026: an operation's own definition is gates, and everything a program
or a request brings is a value).

Every circuit type and every table a statement names carries a label. It is ``library`` exactly when the object is on this
list, and ``program`` otherwise:

* the verifier holds library objects and folds them itself; they are the public track, and it holds nothing else;
* a program table or value is a registered input (`verity.ml.circuits.program`'s `RegisteredInput<w>_v1`, bound at registration), a
  value committed under a hiding leaf and always hidden; a program's circuit types stay committed in c. There is one build,
  the private one.

The list is fixed and versioned. Adding or removing an entry is a new `VERSION`, and so a new `digest()`.

**Tables** are named by the SHA-512 of their raw little-endian 32-bit words: the key under which the Lean verifier's archive
holds them (`Flock/Lookup.lean`'s `MUFU_TABLES`). Their SHA-256 over the same bytes is the pin the circuit files' META names.
The library holds the measured MUFU tables only. A table derived from a model's architecture (an activation such as
``gelu_tanh_bf16``, or a software function such as ``tanh_rn``) stays ``program``, a registered input: it names the
architecture, and a library table is folded in every session whether a unit reads it or not.

**Definitions** are the IR primitives whose circuits are library types: the tensor-core steps and the casts that core
registers. The row-hashing circuits (``sha512x3``, ``hm96``) join by their type digests once the hierarchical circuit format
defines them.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from verity.ml.circuits.codec import canonical_json

LIBRARY = "library"
PROGRAM = "program"
LABELS = (LIBRARY, PROGRAM)

TAG = "verity/library"
VERSION = 1


@dataclass(frozen=True)
class Table:
    """A measured table: ``2^index_bits`` little-endian u32 words, of which the low ``value_bits`` bits vary."""

    name: str
    index_bits: int
    value_bits: int
    sha256: str
    sha512: str

    @property
    def words(self) -> int:
        return 1 << self.index_bits


TABLES: tuple[Table, ...] = (
    Table("rsq", 24, 31, "218fc04d271773269367dedf4cff9ff719081b1218c3d67d248d8d6126040246",
          "39f7b912bc25b913618d926725a18974af9eb246f6f36ce06ef63ffdb293d80bc222623c6d182d2c23e16f5ada2b8db9e7e6a53b3e42046763f7794ac3cef378"),
    Table("sqrt", 24, 31, "4e017678de306e0be8bed66193c2ebf09e9c60a9600cad56cd05bb7b6c3f266f",
          "33d076106efaa955e6452200d45651a799628e76d1d1cc682168f464a03100b937bee08533761d9fa8f8a114cee2a32d322888d5f08fd320bfeb17805a5c2536"),
    Table("rcp", 23, 31, "c408381446132e07fa5581e930dc33fb790bea7ec4276ef3621e322df2492a55",
          "8a317adbcc078bf70ae01959b679d49f97cd9c8a93d6f39b7325c93d0fd550fa4396a4c3a0568d2387c2ff2f33eb76453b33a7b8890d97752d86cc1d971aeacc"),
    Table("ex2", 23, 31, "b2a42c4a626469c986e33e43f16f41bde9d84de94347cd9ceaa1bd68d36bcdf0",
          "5cb768c1725369ad905e725437463729d5b6054ca9adc498e755247e2ba848b510e3325e1c3fecad1242832aca5fa0afd463b16002a7f34050a7b377ed107877"),
    # MUFU.TANH as the gate circuit reads it (`verity_flock.tail_pieces`): the measured word at every positive word of the table
    # region, by its low 27 bits
    Table("tanh_mufu", 27, 31, "4c25bacba332fda09d215e64dc5ca3ec9a7dfdc98a3ea7971f5422054b2943e0",
          "f83e87737a231de2ec109f5a7ea9f8dac78089ef9978ac19ce540c03cefd356ce542948ba8ffb11952b68a2d0f63fe4c6e2e84d218739a262a9b4bb07dd66985"),
)

#: The IR primitives whose circuits are library types: the tensor-core steps and the casts `verity.ml.circuits.ops.prims` registers.
DEFINITIONS: frozenset[str] = frozenset({
    "AmpereBF16TcDot16_v2", "HopperBF16WgmmaDot16_v1", "HopperE4m3QgmmaDot32_v1",
    "Bf16ToF32_v1", "F32ToBf16Rn_v1", "F2fpBf16_v1", "F32ToE4m3Sat_v1",
})

_BY_SHA512 = {t.sha512: t for t in TABLES}
_BY_NAME = {t.name: t for t in TABLES}


def table(key: str) -> Table:
    """A library table by its SHA-512 or its name."""
    t = _BY_SHA512.get(key) or _BY_NAME.get(key)
    if t is None:
        raise KeyError(f"no library table {key!r}")
    return t


def table_label(sha512: str) -> str:
    """``library`` for a table on the list, by the SHA-512 of its words; ``program`` otherwise."""
    return LIBRARY if sha512 in _BY_SHA512 else PROGRAM


def definition_label(definition_id: str) -> str:
    """``library`` for a Definition whose circuit is a library type, by its IR id; ``program`` otherwise."""
    return LIBRARY if definition_id in DEFINITIONS else PROGRAM


def spec() -> dict:
    """The list as canonical data: what `digest` hashes."""
    return {"format": TAG, "version": VERSION, "tables": [asdict(t) for t in TABLES], "definitions": sorted(DEFINITIONS)}


def digest() -> str:
    """``SHA-512(TAG || "/v" || version || 0 || canon(spec))``: the library version a statement can pin."""
    return hashlib.sha512(f"{TAG}/v{VERSION}".encode() + b"\0" + canonical_json(spec())).hexdigest()


__all__ = ["DEFINITIONS", "LABELS", "LIBRARY", "PROGRAM", "TABLES", "TAG", "VERSION", "Table", "definition_label", "digest",
           "spec", "table", "table_label"]
