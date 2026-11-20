"""Every word primitive registered once every module of `verity`, both adapters and Flock's prover is imported has one
Boolean body (`versions.boolean_version`), since a Program over one without is refused at build (`E_NO_BODY`). `BODYLESS` is
the list of those that have none yet, each with why. It is held exactly: a new primitive without a body fails, and so does
one listed that has gained one, until its entry is removed, so the list only shrinks. A family made on demand
(`codec.register_lazy_family`) is seen at the one member `ON_DEMAND` names for it, and every member of it is read as that one,
whichever members other tests made first; a primitive a test module defines is not counted. A module is left out only when it needs a package outside the workspace that this
environment lacks (a GPU runtime)."""
from __future__ import annotations

import importlib
import pkgutil

from verity.circuits import codec
from verity.circuits.codec import descriptor_id, lookup_primitive
from verity.circuits.defs import REGISTRY, PrimitiveDefinition
from verity.circuits.ops.boolean import versions as V
from verity.circuits.program import is_word_primitive
from verity_vllm.definitions.registry import RETIRED

ROOTS = ("verity", "verity_vllm", "verity_torch", "verity_flock")

#: one member of each family made on demand, at a served binding where one is served
ON_DEMAND = (
    "And2_v1", "Const1[0x1]_v1", "Not_v1", "Const32[0x00000000]_v1", "BitOf32x0_v1", "WordOf32_v1", "MoeSelectBf16x128_v1",
    "GatherBf16x154880_v1", "GatherE4m3x128_v1", "GatherF32x154880_v1",
    "DsaTopKLongRowCheck131072x2048_v1", "DsaTopKThresholdKey131072x2048_v1",
    "DsaTopKCountedThresholdKey131072x2048_v1", "DsaTopKSlot131072x2048_v1",
    "TopPMaskWordx154880_v1", "BitAtx154880_v1", "BitXor_v1", "U32Xor_v1", "SelectBit_v1", "I32ToF32_v1", "NvLogf_v1",
    "TopPNumkeep_v1", "GatherI32x154880_v1", "TableRead8x8_v1")

#: the word primitives without a Boolean body, a family by its `ON_DEMAND` member: id -> why; only shrinks
BODYLESS: dict[str, str] = {}


def _families() -> dict[str, str]:
    """Each family made on demand, by its pattern, to its `ON_DEMAND` member; a family with none fails here."""
    out = {}
    for pat, _ in codec._LAZY_PRIMITIVE_FAMILIES:
        if pat.pattern == RETIRED:
            continue  # refuses every member, so it registers no primitive
        named = [f for f in ON_DEMAND if pat.fullmatch(f)]
        assert named, f"no member of the family {pat.pattern} in ON_DEMAND: name one at a served binding"
        out[pat.pattern] = named[0]
    return out


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
    for fid in ON_DEMAND:
        lookup_primitive(REGISTRY, fid)
    fams = _families()
    out = {}
    for d in list(REGISTRY.defs.values()):
        if isinstance(d, PrimitiveDefinition) and is_word_primitive(d) and not getattr(d.evaluate, "__module__", "").startswith("tests."):
            fid = descriptor_id(d)
            rep = next((fams[p.pattern] for p, _ in codec._LAZY_PRIMITIVE_FAMILIES if p.fullmatch(fid)), fid)
            out[rep] = d if rep == fid else lookup_primitive(REGISTRY, rep)
    return out


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
