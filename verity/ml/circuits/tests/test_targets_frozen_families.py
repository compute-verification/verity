"""No new gate family under the retired ``veritor.*`` prefix (CIR-12, circuits, 14 Oct 2026; README.md "Older names stay...").

The existing ids are grandfathered: their spellings are baked into the ``*_FAMILY_DIGEST`` pins, fixtures and recorded
runs, so they stay exactly as declared in ``targets/target.py``. A new family takes a fresh id outside ``veritor.*``;
this test fails when ``target.py`` declares one inside it."""
from __future__ import annotations

from verity.ml.circuits.targets import target as T

GRANDFATHERED = {
    "veritor.tc-ampere-bf16",                                # TC_AMPERE_BF16_NAME
    "veritor.tc-bf16.",                                      # TC_BF16_NAME_PREFIX
    "veritor.tc-ampere-bf16@1",                              # FAMILY
    "veritor.tc-bf16.hopper_bf16_m16n8k16@1",                # BF16_HOPPER_FAMILY
    "veritor.tensor-core.hopper_e4m3_wgmma_k32@2",           # FP8_HOPPER_FAMILY
    "veritor.tensor-core.ada_e4m3_m16n8k32@2",               # FP8_ADA_FAMILY
    "veritor.tensor-core.blackwell_sm120_nvf4_m16n8k64@2",   # NVFP4_FAMILY
    "veritor.tensor-core.blackwell_sm120_e4m3_m16n8k32@2",   # FP8_SM120_FAMILY
}


def test_no_new_veritor_family() -> None:
    declared = {v for v in vars(T).values() if isinstance(v, str) and v.startswith("veritor.")}
    declared |= {t.family for t in T.TARGETS.values() if t.family.startswith("veritor.")}
    new = declared - GRANDFATHERED
    assert not new, (
        f"new veritor.* gate-family id(s) {sorted(new)}: the prefix is frozen (CIR-12) - "
        "give the new family a fresh id outside veritor.*, or, for a truly pre-existing id, "
        "add it to GRANDFATHERED here with the pin that bakes it in"
    )
    gone = GRANDFATHERED - declared
    assert not gone, (
        f"grandfathered veritor.* id(s) {sorted(gone)} no longer declared in targets/target.py: "
        "if the family was retired on purpose, remove it here and from README.md's paragraph in the same PR"
    )
