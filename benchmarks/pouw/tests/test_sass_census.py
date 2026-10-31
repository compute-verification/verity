"""sass_census: the SASS opcode census diff_menu.py diffs against A7's menu (`sass_census/`).

On a synthetic two-architecture listing (`fixtures/sass_census_sample.sass`, sm_90a and sm_100a): `sass_count` parses
the `cuobjdump -sass` text, counts each dotted opcode and folds the minor versions into one device family; `sass_semantics`
classifies each base opcode (can it head a product, does it write a word, which menu route prices it); and `diff_menu`
buckets the family's census against the menu, so the red-team gaps are the GAP-* and UNDERPRICED buckets. It also pins the
NVIDIA Instruction Set Reference tables and the PTX index the diff reads for the opcodes the corpus never used.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

POUW = Path(__file__).resolve().parents[1]
SC = POUW / "sass_census"
FIX = Path(__file__).resolve().parent / "fixtures" / "sass_census_sample.sass"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# diff_menu imports `loads` and `sass_semantics` by bare name off the paths it inserts; load those first under the same
# names so every module is the one diff_menu uses.
sass_semantics = _load("sass_semantics", SC / "sass_semantics.py")
loads = _load("loads", POUW / "facet_break" / "loads.py")
sass_count = _load("sass_count", SC / "sass_count.py")
diff_menu = _load("diff_menu", SC / "diff_menu.py")


def census() -> dict:
    with FIX.open() as f:
        obj = sass_count.count(f, "libtest")
    return sass_count.merge([obj])


# ---- the census -----------------------------------------------------------------------------------------------------------


def test_family_folds_the_minor_versions():
    assert sass_count.family("sm_90a") == "sm_90"
    assert sass_count.family("sm_100a") == "sm_100" == sass_count.family("sm_100f")
    assert sass_count.family("sm_120") == "sm_120" == sass_count.family("sm_120a")


def test_count_reads_each_architectures_opcodes():
    with FIX.open() as f:
        obj = sass_count.count(f, "libtest")
    assert set(obj["archs"]) == {"sm_90a", "sm_100a"}
    a90 = obj["archs"]["sm_90a"]
    assert a90["funcs"] == 1 and "IMMA.16832.S8.S8" in a90["ops"] and "DMMA.8" in a90["ops"]
    # the guard predicate `@!P0` is stripped, so BRA is counted by its opcode, not the predicate
    assert "BRA" in a90["ops"] and a90["ops"]["BRA"][0] == 1


def test_merge_folds_to_families_and_bases():
    c = census()
    assert set(c) == {"sm_90", "sm_100"}
    sm90 = c["sm_90"]
    assert "DMMA" in sm90["bases"] and "IMMA" in sm90["bases"] and "HGMMA" in sm90["bases"]
    assert sm90["ops"]["IMMA.16832.S8.S8"]["base"] == "IMMA"
    # UTCIMMA/UTCQMMA head words only on the Blackwell side of the fixture
    assert "UTCIMMA" not in sm90["bases"] and "UTCIMMA" in c["sm_100"]["bases"]


# ---- the classification ---------------------------------------------------------------------------------------------------


def test_classify_the_routes():
    cl = sass_semantics.classify
    # tensor: every MMA family heads products on A3's tensor route
    for base in ("DMMA", "IMMA", "HGMMA", "IGMMA", "UTCIMMA", "UTCQMMA"):
        assert cl(base)["mul"] is True and cl(base)["route"] == "tensor", base
    # the honest lane MAC, and a paired lane multiply (two FMAs at two issue slots) flagged paired (UNDERPRICED)
    assert cl("IMAD")["mul"] is True and cl("IMAD")["route"] == "lane"
    assert cl("FFMA2")["mul"] is True and cl("FFMA2").get("paired") is True
    assert cl("FMUL2").get("paired") is True
    # words with no product: a cross-lane fold on no route, a packed add on the lane route, a priced side reduction
    assert cl("REDUX") == {"mul": False, "word": True, "route": None, "reduce": True,
                           "note": sass_semantics.CROSSLANE_REDUCE["REDUX"]}
    assert cl("VIADD")["mul"] is False and cl("VIADD")["word"] is True and cl("VIADD")["route"] == "lane"
    assert cl("REDG")["route"] == "sidefloor" and cl("REDG").get("reduce") is True
    # a mover copies an existing word; control and a compare write no committed word
    assert cl("MOV")["mul"] is False and cl("MOV")["route"] == "mover"
    assert cl("LDGSTS")["route"] == "mover"
    assert cl("BRA")["word"] is False and cl("BRA")["route"] == "control"
    assert cl("ISETP")["word"] is False and cl("ISETP")["route"] == "control"
    # a base the table doesn't know is surfaced, not silently priced
    assert cl("ZZZNEW") == {"mul": None, "word": None, "route": None,
                            "note": "unclassified base opcode (new to the table)"}


# ---- the diff against the menu --------------------------------------------------------------------------------------------


def buckets(family: str) -> dict[str, set[str]]:
    d = diff_menu.diff(family, census(), with_unused=False)
    out: dict[str, set[str]] = {}
    for r in d["rows"]:
        out.setdefault(r["bucket"], set()).add(r["op"])
    return out


def test_diff_sm90_gaps():
    """On the Hopper side: DMMA f64 has no tensor peak in the device census, so A3 prices it at 0 (GAP-TENSOR); the
    priced MMA forms (HGMMA/IGMMA/IMMA) land in PRICED-TENSOR; a warp reduce is census class REDUX, priced on issue."""
    b = buckets("sm_90")
    assert "DMMA.8" in b["GAP-TENSOR"]
    assert {"HGMMA.64x128x16.F32", "IGMMA.64x256x32.S8", "IMMA.16832.S8.S8"} <= b["PRICED-TENSOR"]
    assert "REDUX.ADD.S32" in b["PRICED-LANE"]


def test_diff_sm100_gaps_and_underpricing():
    """On the Blackwell side: DMMA is a tensor gap again; the paired FFMA2/FMUL2 (two FMAs at two issue slots) are native
    but the census counts each as one FMA (UNDERPRICED); VIADD is census class VIADD, priced on issue."""
    b = buckets("sm_100")
    assert "DMMA.8" in b["GAP-TENSOR"]
    assert {"FFMA2", "FMUL2"} <= b["UNDERPRICED"]
    assert "VIADD" in b["PRICED-LANE"]


def test_diff_flags_the_lane_classifiers_uniform_mis_map():
    """`loads.class_of`'s UNIFORM regex `U[A-Z].*` swallows tcgen05 (UTC*) into UNI; the diff flags that disagreement so a
    classifier run on B200 SASS can't silently misprice the Blackwell tensor route."""
    d = diff_menu.diff("sm_100", census(), with_unused=False)
    flagged = {x["op"] for x in d["disagreements"]}
    assert {"UTCIMMA.MXF8F6F4", "UTCQMMA.E4M3"} <= flagged
    assert all(x["class_of"] == "UNI" for x in d["disagreements"])


def test_ds_tensor_types_read_each_devices_sheet():
    """A3 prices a tensor form by its operand type's `DS_tensor` peak, read from each device's own rows file: no device
    lists f64, sm_90/sm_100 list no e2m1 (fp4), sm_120 does; sm_90's "no b1 figure" (after the `;`) doesn't count b1."""
    assert diff_menu.ds_tensor_types("sm_90") == {"s8", "e4m3", "e5m2", "f16", "bf16", "tf32"}
    assert diff_menu.ds_tensor_types("sm_100") == {"s8", "e4m3", "e5m2", "f16", "bf16", "tf32"}
    assert diff_menu.ds_tensor_types("sm_120") == {"s8", "e4m3", "e5m2", "e2m1", "f16", "bf16", "tf32"}


def test_mma_type_reads_the_operand_type():
    t = diff_menu.mma_type
    assert (t("DMMA.8x8x4"), t("HMMA.16816.F32"), t("HMMA.16816.F32.BF16"), t("HMMA.1688.F32.TF32")) == \
        ("f64", "f16", "bf16", "tf32")
    # tcgen05 names no type (it's in the descriptor): UTCOMMA goes by base, the block-scaled fp4 family
    assert t("UTCOMMA.4X") == t("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X") == "e2m1"
    assert (t("IMMA.16832.S8.S8"), t("UTCQMMA"), t("BGMMA")) == ("s8", "e4m3", "b1")


def test_a_tensor_form_is_priced_by_its_type_not_only_a_census_row():
    """sm_100's census has no mma.sync HMMA row, but f16/bf16/tf32 have a `DS_tensor` peak, so HMMA is priced; UTCOMMA's
    e2m1 has neither, so it's a gap there, while sm_120 (which lists e2m1) prices its OMMA fp4."""
    def b(family, op):
        census = {family: {"ops": {op: {"count": 1, "example": "", "base": op.split(".")[0], "sources": {}}}}}
        return next(r["bucket"] for r in diff_menu.diff(family, census, with_unused=False)["rows"] if r["op"] == op)
    assert b("sm_100", "HMMA.16816.F32") == "PRICED-TENSOR"
    assert b("sm_100", "HMMA.1688.F32.TF32") == "PRICED-TENSOR"
    assert b("sm_100", "UTCOMMA.4X") == "GAP-TENSOR"
    assert b("sm_120", "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X") == "PRICED-TENSOR"
    assert b("sm_90", "DMMA.8x8x4") == b("sm_100", "DMMA.8x8x4") == b("sm_120", "DMMA.8x8x4") == "GAP-TENSOR"


def test_a_gap_row_carries_the_cheapest_escape():
    d = diff_menu.diff("sm_90", census(), with_unused=False)
    dmma = next(r for r in d["rows"] if r["op"] == "DMMA.8")
    assert dmma["escape"] and "int8" in dmma["escape"]
    priced = next(r for r in d["rows"] if r["op"] == "IMMA.16832.S8.S8")
    assert priced["escape"] is None  # a priced row names no escape


# ---- the reference tables -------------------------------------------------------------------------------------------------


def test_the_isa_reference_tables_are_whole():
    p = SC / "data" / "sass_isa_tables.json"
    assert p.is_file(), f"{p} isn't fetched: `uv run research data fetch-fixtures benchmarks/pouw/sass_census/data`"
    t = json.loads(p.read_text())["tables"]
    assert {k: len(v) for k, v in t.items()} == {"hopper": 199, "blackwell": 253, "ampere": 166, "turing": 160}
    assert "DMMA" in t["hopper"] and "UTCIMMA" in t["blackwell"]
    assert diff_menu.isa_vocab("sm_90") is t["hopper"] or diff_menu.ISA_TABLE["sm_90"] == "hopper"


def test_the_ptx_index_is_whole():
    p = json.loads((SC / "data" / "ptx_index.json").read_text())
    assert len(p["instructions"]) == 156
