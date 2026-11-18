"""Every word primitive the catalogs register, `verity.circuits.ops`' and this adapter's (`verity_vllm.definitions`), has
one Boolean body (`versions.boolean_version`), since a Program over one without is refused at build (`E_NO_BODY`). `BODYLESS`
is the list of those that have none yet, each with why. It is held exactly: a new primitive without a body fails, and so does
one listed that has gained one, until its entry is removed, so the list only shrinks."""
from __future__ import annotations

import importlib
import pkgutil

import verity.circuits.ops as OPS
import verity_vllm.definitions  # noqa: F401  (registers the adapter's catalog with the lookup)
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import REGISTRY, PrimitiveDefinition
from verity.circuits.ops.boolean import versions as V
from verity.circuits.program import is_word_primitive

CATALOGS = ("verity.circuits.ops", "verity_vllm.definitions")

_POUW = "PoUW's word primitive (verity/circuits/ops/words.py): no Boolean body written yet"
_ADAPTER = "the adapter's registry: no Boolean body written yet"

#: the word primitives without a Boolean body: id -> why; only shrinks
BODYLESS: dict[str, str] = {
    **{f"Pouw{n}_v1": _POUW for n in (
        "Add64Hi", "Add64Lo", "Add8", "AddSub8", "Ch32", "Dot16U8S8", "FormX3", "FormX", "Maj32", "NegBetaSum16", "NoiseByte",
        "PackBE4", "Sha512BigSigma0Hi", "Sha512BigSigma0Lo", "Sha512BigSigma1Hi", "Sha512BigSigma1Lo", "Sha512SmallSigma0Hi",
        "Sha512SmallSigma0Lo", "Sha512SmallSigma1Hi", "Sha512SmallSigma1Lo", "Sub8", "I32ToF32Rn")},
    **{f"{n}_v1": _ADAPTER + f" ({m})" for n, m in (
        ("DsaButterflySet", "dsa_topk_mask"), ("DsaMaskCount", "dsa_topk_mask"), ("DsaMaskTag", "dsa_topk_mask"),
        ("DsaTopKAtLeastBelow", "dsa_topk_mask"), ("DsaTopKAtLeast", "dsa_topk_mask"), ("E2m1ToBf16", "marlin_moe"),
        ("E2m1x2Hi", "marlin_moe"), ("E2m1x2Lo", "marlin_moe"), ("E8m0DequantBf16", "marlin_moe"), ("F32ErfRn", "ref_prims"),
        ("F32TanhRn", "ref_prims"), ("GeluErfBf16", "act"), ("I32Mul", "marlin_moe"), ("NcpAbsMax", "pouw_rows"),
        ("NcpDot16I7", "pouw_rows"), ("NcpRneClampI7", "pouw_rows"))},
    "GumbelNoiseLane_v2": _ADAPTER + " (sampling)",
}


def _catalog_primitives() -> dict[str, PrimitiveDefinition]:
    for m in pkgutil.walk_packages(OPS.__path__, "verity.circuits.ops."):
        importlib.import_module(m.name)
    assert not V.load(), "a catalog module failed to import"
    return {descriptor_id(d): d for d in list(REGISTRY.defs.values())
            if isinstance(d, PrimitiveDefinition) and is_word_primitive(d)
            and getattr(d.evaluate, "__module__", "").startswith(CATALOGS)}


def test_every_catalog_word_primitive_has_a_boolean_body_but_the_listed() -> None:
    prims = _catalog_primitives()
    missing = {}
    for fid, p in prims.items():
        try:
            V.boolean_version(p)
        except V.NoBooleanVersion as e:
            missing[fid] = str(e)
    new = sorted(set(missing) - set(BODYLESS))
    assert not new, f"word primitives without a Boolean body, not listed: {new}; write each one's body"
    gained = sorted(set(BODYLESS) - set(missing))
    assert not gained, f"listed in BODYLESS but now with a Boolean body, or gone from the catalogs: {gained}; remove them"
