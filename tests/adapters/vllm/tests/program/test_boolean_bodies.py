"""Every word primitive registered once every module of `verity`, both adapters and Flock's prover is imported has one
Boolean body (`versions.boolean_version`), since a Program over one without is refused at build (`E_NO_BODY`). `BODYLESS` is
the list of those that have none yet, each with why. It is held exactly: a new primitive without a body fails, and so does
one listed that has gained one, until its entry is removed, so the list only shrinks. A module is left out only when it
needs a package outside the workspace that this environment lacks (a GPU runtime)."""
from __future__ import annotations

import importlib
import pkgutil

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import REGISTRY, PrimitiveDefinition
from verity.circuits.ops.boolean import versions as V
from verity.circuits.program import is_word_primitive

ROOTS = ("verity", "verity_vllm", "verity_torch", "verity_flock")

_ADAPTER = "the adapter's registry: no Boolean body written yet"
_BYTE = ("byte `at` of a SHA-512 compression, its word form one gate a byte (pearl_c_noise): the compression's Boolean "
         "version is its whole `_v2`, and a body a byte would be a whole compression a byte")

#: the word primitives without a Boolean body: id -> why; only shrinks
BODYLESS: dict[str, str] = {
    **{f"{n}_v1": _ADAPTER + f" ({m})" for n, m in (
        ("DsaButterflySet", "dsa_topk_mask"), ("DsaMaskCount", "dsa_topk_mask"), ("DsaMaskTag", "dsa_topk_mask"),
        ("DsaTopKAtLeastBelow", "dsa_topk_mask"), ("DsaTopKAtLeast", "dsa_topk_mask"), ("E2m1ToBf16", "marlin_moe"),
        ("E2m1x2Hi", "marlin_moe"), ("E2m1x2Lo", "marlin_moe"), ("E8m0DequantBf16", "marlin_moe"), ("F32ErfRn", "ref_prims"),
        ("F32TanhRn", "ref_prims"), ("GeluErfBf16", "act"), ("I32Mul", "marlin_moe"), ("NcpAbsMax", "pouw_rows"),
        ("NcpDot16I7", "pouw_rows"), ("NcpRneClampI7", "pouw_rows"))},
    "GumbelNoiseLane_v2": _ADAPTER + " (sampling)",
    **{f"PearlC{n}Byte_v1": _BYTE for n in ("LineKey", "LineRaw", "RowBlock", "RowSeedHead", "RowSeedTail", "SeedB", "ShaBlock")},
}


def _word_primitives() -> dict[str, PrimitiveDefinition]:
    for root in ROOTS:
        pkg = importlib.import_module(root)
        for m in pkgutil.walk_packages(pkg.__path__, root + "."):
            try:
                importlib.import_module(m.name)
            except ModuleNotFoundError as e:
                if (e.name or "").split(".")[0] in ROOTS:
                    raise
    assert not V.load(), "a catalog module failed to import"
    return {descriptor_id(d): d for d in list(REGISTRY.defs.values())
            if isinstance(d, PrimitiveDefinition) and is_word_primitive(d)}


def test_every_word_primitive_has_a_boolean_body_but_the_listed() -> None:
    prims = _word_primitives()
    missing = {}
    for fid, p in prims.items():
        try:
            V.boolean_version(p)
        except V.NoBooleanVersion as e:
            missing[fid] = str(e)
    new = sorted(set(missing) - set(BODYLESS))
    assert not new, f"word primitives without a Boolean body, not listed: {new}; write each one's body"
    gained = sorted(set(BODYLESS) - set(missing))
    assert not gained, f"listed in BODYLESS but now with a Boolean body, or gone: {gained}; remove them"
