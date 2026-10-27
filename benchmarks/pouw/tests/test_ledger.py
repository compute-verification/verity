"""The attempt ledger (`harness/ledger.py`, written by `harness/verify.py`): every arm row of a verified run becomes one
``kernel-attempt/v1`` record named by its kernel variant, with its parent, tier, gate version, metrics, step times, γ at
both FP32 prices and the verifier's lines; a verifier's negative control is judged; records publish into the evidence
store once (content-addressed, labels not repeated); and the store's records come back as `panel.py` rows that follow the
panel's rules (a screen never plots, a measurement stands only with every verifier field)."""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "harness"))

import arm as A  # noqa: E402
import ledger as L  # noqa: E402
import price_twins as P  # noqa: E402
import verify as V  # noqa: E402

RUN, COMMIT = "r20260930-170000-abcd", "0123456789abcdef0123456789abcdef01234567"
PREFILL, DECODE = "m8192-n8192-k8192", "m32-n8192-k8192"
SHAPES = {PREFILL: ("prefill", 8192), DECODE: ("decode", 32)}

#: prints one line per transcript and control under the run: ACCEPT the transcripts, REJECT the controls (exit 0, as a
#: verifier does when it rejected what it was meant to), and ACCEPT a control that says "accept me"
VERIFIER = """
import hashlib, sys
from pathlib import Path
run = Path(sys.argv[1])
for m in sorted(run.glob("*/*/*/*.json")):
    raw = m.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    ok = m.parts[-4] == "transcripts" or b"accept me" in raw
    print(("ACCEPT" if ok else "REJECT") + f" {run.name}/{m.relative_to(run)} sha256:{sha}")
"""


@pytest.fixture(autouse=True)
def store_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "absent-store.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path))
    for k in ("RESEARCH_RUN_ID", "RESEARCH_SOURCE_SHA", "GPU_LEASE_WHO", "POUW_TIER", "POUW_LEDGER_PUBLISH"):
        monkeypatch.delenv(k, raising=False)


def kernel_dir(tmp_path: Path, text: str = "kernel v1") -> Path:
    d = tmp_path / "kernel"
    d.mkdir(exist_ok=True)
    (d / "run.py").write_text("# the pipeline\n")
    (d / "k.cu").write_text(text)
    (d / "k.cubin").write_bytes(hashlib.sha256(text.encode()).digest())
    return d


def variant(tmp_path: Path, text: str = "kernel v1") -> dict:
    d = kernel_dir(tmp_path, text)
    return L.variant_record(family="pearl-c", device="sm120", schemes={"pearl-c-sm120-v2-hot-h3"}, sources=["run.py", "k.cu"],
                            root=d, binaries=[d / "k.cubin"], prices="cast-8.72")


#: an arm's own `Work.gamma` (GPU 1's `pearlc_arm.gamma`: its device record's readings at one price), which the ledger keeps
#: as the arm's claim and never publishes
ARM_GAMMA = {"cap": "3/1000", "floor": "1/1000", "basis": "Derived at the device record's prices"}
#: γ of v2-hot on the wide-store cast at 8,192³, the larger of its two twins (the loop price's), from the pins
V2_HOT_872 = P.lookup(P.load(), "pearl-c-sm120/v2-hot", "cast-8.72", 8192, 8192)


def result(run_dir: Path, var: dict | None, *, gamma=ARM_GAMMA, control: bool = True, lineage: dict | None = None,
           poisoned_by: str = "harness") -> Path:
    """A bench.py result for one arm `pc` at both headlines, with gates, controls, the SASS gate and a decode split."""
    shapes = []
    for label, (phase, m) in SHAPES.items():
        p = {"phase": phase, "shape": label, "slowdown": 1.21 if phase == "prefill" else 1.25, "hash_free": 1.1,
             "baseline": "fp8-e4m3", "baseline_kernel": "lt13_algo_5", "baseline_ms": 1.446,
             "sm_clock_arm": "2090,2092", "sm_clock_base": "2091,2093"}
        row = {"graph": {"total_x": p["slowdown"], "hash_free_x": 1.1, "total_ms": 1.75, "hash_free_ms": 1.59,
                         "hashing_ms": 0.16, "baseline": "fp8-e4m3", "baseline_ms": 1.446},
               "gamma": gamma, "w_ref": "100", "credited": "99", "unit": "fp8-mac", "panel": p}
        shape = {"label": label, "m": m, "k": 8192, "n": 8192, "phase": phase, "arms": {"pc": row}}
        if phase == "decode":
            p |= {"decode_method": "dependent-chain", "chain_length": 32}
            shape["chain"] = {"arms": {"pc": {"split": {"graph": {"before_ms": 0.004, "during_ms": 0.05, "after_ms": 0.012,
                                                                  "derive_ms": 0.004, "side_ms": 0.058,
                                                                  "hidden_by_side_ms": 0.008}}}}}
        shapes.append(shape)
    decl = {"per": "run", "transcript": "transcripts/{arm}/{shape}/manifest.json",
            "commands": [["{python}", str(run_dir.parent / "v.py"), "{run_dir}"]]}
    if control:
        decl["control"] = "negative/{arm}/{shape}/manifest.json"
    build = {"binaries": [], "panel": {"line": "pearl-c-sm120", "version": "v2-hot"}, "scheme": "pearl-c-sm120-v2-hot-h3"}
    if var is not None:
        build["variant"] = var
    build["lineage"] = lineage if lineage is not None else {
        "parent": "art:" + "1" * 64, "approach": "pouw/pearl-c-sm120", "mechanism": ["hash:side-stream", "cast:packed-sts64"],
        "hypothesis": "tile hashing on the side lane hides it under the next call",
        "phase_groups": {"before": ["a_tree", "forming"], "gemm": ["gemm", "cleanup"], "after": ["tile_hash", "screen"]}}
    arms = {"pc": {"stream": "fp8", "precision": "fp8-e4m3", "build": build, "verifier": A.check_verifier("pc", {"verifier": decl})}}
    gates = [{"key": ["arm", label, "pc", role], "ok": True, "detail": {}} for label in SHAPES for role in ("hash", "nohash")]
    negatives = [{"key": ["known-bad", PREFILL, "pc", "hash", "flip-output-byte"], "rejected": True}]
    sass = {"version": "sass-gate/v4", "ok": True,
            "binaries": [{"source": "arm:pc", "ok": True, "cubins": [{"toolkit": "12.9.86", "functions": 3}]}]}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir.parent / "v.py").write_text(VERIFIER)
    for label in SHAPES:
        for top, text in (("transcripts", f'{{"shape": "{label}"}}'), ("negative", f'{{"wrote": "nothing", "at": "{label}"}}')):
            f = run_dir / top / "pc" / label / "manifest.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
    path = run_dir / "bench.json"
    poisoning = [{"arm": "pc", "shape": label, "by": poisoned_by, "poison": "0xA5", "written": {"root": 32}} for label in SHAPES]
    path.write_text(json.dumps({"harness": {"interface": A.VERSION}, "shapes": shapes, "arms": arms, "gates": gates,
                                "negative_controls": negatives, "sass_gate": sass, "clocks": {"label": "locked-2100"},
                                "poisoning": poisoning,
                                "versions": {"driver": "580.173.02", "nvcc": "12.9.86"}}))
    return path


def verify(path: Path, *extra: str) -> tuple[int, dict]:
    code = V.main([str(path), "--run-id", RUN, "--commit", COMMIT, "--by", "bc-test", *extra])
    return code, json.loads(path.read_text())


def records(run_dir: Path) -> dict[str, dict]:
    return {r["meta"]["shape"]: r for r in L.read(run_dir)}


# ---- the variant ---------------------------------------------------------------------------------------------------------------


def test_a_variant_is_named_by_its_sources_digest_in_any_order_and_a_changed_source_is_a_new_variant(tmp_path):
    v = variant(tmp_path)
    assert v["id"] == f"pearl-c-sm120/{v['source_sha256'][:12]}"
    d = tmp_path / "kernel"
    assert L.source_sha256(["k.cu", "run.py"], d) == L.source_sha256([d / "run.py", "k.cu"], d) == v["source_sha256"]
    assert v["binary_sha256"] == hashlib.sha256((d / "k.cubin").read_bytes()).hexdigest()
    assert variant(tmp_path, "kernel v1, unroll 2")["id"] != v["id"]
    with pytest.raises(L.LedgerError, match="twice"):
        L.source_sha256(["k.cu", "k.cu"], d)
    with pytest.raises(L.LedgerError, match="at least one source"):
        L.source_sha256([], d)


def test_the_variant_and_the_groups_are_verity_pouw_serving_s_when_it_is_importable(tmp_path):
    S = L.serving()
    if S is None:
        pytest.skip("verity.core.protocols.pouw.serving (#567) isn't on this tree yet")
    v = variant(tmp_path)
    kv = S.KernelVariant(family="pearl-c", device="sm120", schemes=frozenset(v["schemes"]), schedules=frozenset(v["schedules"]),
                         shapes=S.ShapeRule(), source_sha256=v["source_sha256"], binary_sha256=v["binary_sha256"],
                         loader=v["loader"], prices=v["prices"])
    assert kv.id == v["id"]
    assert tuple(S.GROUPS) == L.GROUPS
    gate = records_of(tmp_path)[PREFILL]["meta"]["gate"]
    assert set(gate) == set(S.GateRecord.__dataclass_fields__)


def records_of(tmp_path: Path) -> dict[str, dict]:
    run_dir = tmp_path / RUN
    verify(result(run_dir, variant(tmp_path)), "--tier", "2b")
    return records(run_dir)


# ---- γ at both FP32 prices, from the price twins ---------------------------------------------------------------------------

#: pins as the Lean store and the staged twins print them (price-twins-lean, 30 Sep): v2 at the statement's cast and 8.00
#: (in the Lean store), v2 on the wide-store cast at the loop price, a rounded twin (the loop price at the record's own cast),
#: v2-hot's value lemma, and Pearl-C4's chain-only value lemma
PINS = {
    "Pouw.PearlC.pearlCGammaSm120v2Cap1000_8192": {"type_hash": "0000000064832e60", "signature": [
        "Pouw.PearlC.pearlCGammaSm120v2Cap1000_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]",
        "  (CM : Pouw.PearlC.CostModel Q R S) (sem : Pouw.PearlC.PearlCSem Q R S)",
        "  (hTT : Pouw.PearlC.Assumptions.TTOutPearlCDev CM (Pouw.PearlC.devSm120v2 Pouw.PearlC.Prices.sm120) sem (1 / 1000)) :",
        "  Pouw.PearlC.Gγ CM (Pouw.PearlC.pearlCProtocolDev (Pouw.PearlC.devSm120v2 Pouw.PearlC.Prices.sm120) sem (1 / 1000))",
        "    (Pouw.PearlC.pearlCDomainDevAt (Pouw.PearlC.devSm120v2 Pouw.PearlC.Prices.sm120) Pouw.PearlC.sh8192)",
        "    (↑(216793 / 59950000)) Pouw.PearlC.εPearlC"]},
    "Pouw.PearlC.pearlCGammaSm120v2LoopCast8p72Cap1000_8192": {"type_hash": "x", "signature": [
        "Pouw.PearlC.pearlCGammaSm120v2LoopCast8p72Cap1000_8192 {Q R S : Type} (CM : Pouw.PearlC.CostModel Q R S)",
        "  (hTT : Pouw.PearlC.Assumptions.TTOutPearlCDev CM (Pouw.PearlC.devSm120v2 Pouw.PearlC.Prices.sm120Loop) sem (1 / 1000)) :",
        "  Pouw.PearlC.Gγ CM",
        "    (Pouw.PearlC.pearlCProtocolDevK (Pouw.PearlC.devSm120v2 Pouw.PearlC.Prices.sm120Loop) sem (1 / 1000)",
        "      (instHSub.hSub (218 / 25) (8953 / 1000)))",
        "    (Pouw.PearlC.pearlCDomainDevAt (Pouw.PearlC.devSm120v2 Pouw.PearlC.Prices.sm120Loop) Pouw.PearlC.sh8192)",
        "    (1555901 / 419688350).cast Pouw.PearlC.εPearlC"]},
    "Pouw.PearlC.pearlCGammaSm120v2LoopCap1000_8192": {"type_hash": "y", "signature": [
        "Pouw.PearlC.pearlCGammaSm120v2LoopCap1000_8192 (hp : Eq d.prices Pouw.PearlC.Prices.sm120Loop)",
        "  Pouw.PearlC.Gγ CM (Pouw.PearlC.pearlCProtocolDev (Pouw.PearlC.devSm120v2 Pouw.PearlC.Prices.sm120Loop) sem (1 / 1000))",
        "    (522517 / 139900000).cast Pouw.PearlC.εPearlC"]},
    "Pouw.PearlC.gammaHot_sm120v2HotLoopCast8_publicConst64_8192": {"type_hash": "z", "signature": [
        "Pouw.PearlC.gammaHot_sm120v2HotLoopCast8_publicConst64_8192 (d : Pouw.PearlC.PearlCDevice) (hG : Eq d.G 0)",
        "  (hp : Eq d.prices Pouw.PearlC.Prices.sm120Loop) :",
        "  Eq",
        "    (Pouw.PearlC.gammaHot d (Pouw.PearlC.HotSizing.publicConst 64) (instHSub.hSub 8 (8953 / 1000)) (1 / 1000)",
        "      Pouw.PearlC.sh8192)",
        "    (1521365753 / 420071150000)"]},
    "Pouw.PearlC.gammaFp4ChainOnly_sm120Loop_lut256_8192": {"type_hash": "w", "signature": [
        "Pouw.PearlC.gammaFp4ChainOnly_sm120Loop_lut256_8192 (d : Pouw.PearlC.Fp4Device)",
        "  (hp : Eq d.prices (Pouw.PearlC.Fp4Prices.sm120At (1047 / 125) Pouw.PearlC.Fp4ScalePath.lut256)) :",
        "  And",
        "    (Rat.instLT.lt 0",
        "      (instHSub.hSub (Pouw.PearlC.creditFp4ChainOnly d Pouw.PearlC.sh8192)",
        "        (instHMul.hMul (1 / 400) (Pouw.PearlC.creditFp4 d Pouw.PearlC.sh8192))))",
        "    (Eq (Pouw.PearlC.gammaFp4ChainOnly d (1 / 400) Pouw.PearlC.sh8192) (26680734301 / 1376663200000))"]},
    "Pouw.PearlC.pearlCGammaUOnlySm120v2Cap1000_8192": {"type_hash": "u", "signature": ["U-only messages"]},
    "Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192": {"type_hash": "v", "signature": [
        "Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192 {Q R S : Type} (CM : Pouw.PearlC.CostModel Q R S)",
        "  (hTT : Pouw.PearlC.Assumptions.TTOutPearlCDevRev1 CM (Pouw.PearlC.devSm120v1 Pouw.PearlC.Prices.sm120Loop) sem",
        "    (1 / 1000)) :",
        "  Pouw.PearlC.Gγ CM",
        "    (Pouw.PearlC.pearlCProtocolDevRev1K (Pouw.PearlC.devSm120v1 Pouw.PearlC.Prices.sm120Loop) sem (1 / 1000)",
        "      (instHSub.hSub (218 / 25) (8953 / 1000)))",
        "    (Pouw.PearlC.pearlCDomainDevAt (Pouw.PearlC.devSm120v1 Pouw.PearlC.Prices.sm120Loop) Pouw.PearlC.sh8192)",
        "    (1648180439 / 446072750000).cast Pouw.PearlC.εPearlC"]},
    "Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192_of_cap400": {"type_hash": "o", "signature": ["the same γ"]},
    "Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_8192": {"type_hash": "i", "signature": [
        "Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap1000_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R]",
        "  [Fintype S] (CM : Pouw.PearlC.CostModel Q R S) (sem : Pouw.PearlC.PearlCSem Q R S)",
        "  (hTT :",
        "    Pouw.PearlC.Assumptions.TTOutPearlCDevRev1 CM (Pouw.PearlC.devSm120v1 Pouw.PearlC.Prices.sm120) sem (1 / 1000)) :",
        "  Pouw.PearlC.Gγ CM",
        "    (Pouw.PearlC.pearlCProtocolDevRev1K (Pouw.PearlC.devSm120v1 Pouw.PearlC.Prices.sm120) sem (1 / 1000)",
        "      (instHSub.hSub (218 / 25) 8))",
        "    (Pouw.PearlC.pearlCDomainDevAt (Pouw.PearlC.devSm120v1 Pouw.PearlC.Prices.sm120) Pouw.PearlC.sh8192)",
        "    (205211 / 55610750).cast Pouw.PearlC.εPearlC"]},
    "Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap_8192": {"type_hash": "g", "signature": [
        "Pouw.PearlC.pearlCGammaSm120v1Cast8p72Rev1Cap_8192 {Q R S : Type} (ρ : Rat) (hρ : Rat.instLT.lt ρ 1) :",
        "    (instHSub.hSub 1 (instHMul.hMul (instHSub.hSub 1 ρ) (221844 / 222443))).cast Pouw.PearlC.εPearlC"]},
}


def test_price_twins_read_each_pin_s_statement_path_price_and_value_and_skip_what_has_no_priced_path():
    rows = {r["pin"].removeprefix("Pouw.PearlC."): r
            for r in P.extract(PINS, "t", {"Pouw.PearlC.pearlCGammaSm120v2Cap1000_8192": "0000000064832e60"})}
    assert set(rows) == {"pearlCGammaSm120v2Cap1000_8192", "pearlCGammaSm120v2LoopCast8p72Cap1000_8192",
                         "gammaHot_sm120v2HotLoopCast8_publicConst64_8192", "gammaFp4ChainOnly_sm120Loop_lut256_8192",
                         "pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192", "pearlCGammaSm120v1Cast8p72Rev1Cap1000_8192"}
    stmt = rows["pearlCGammaSm120v2Cap1000_8192"]
    assert (stmt["statement"], stmt["path"], stmt["fadd"], stmt["reading"], stmt["kind"], stmt["shape"], stmt["status"]) == \
        ("pearl-c-sm120/v2", "cast-8", Fraction(8), "forming-credited", "unit", 8192, "pinned")
    v1 = rows["pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192"]
    assert (v1["statement"], v1["path"], v1["fadd"], v1["cap"], v1["gamma"]) == (
        "pearl-c-sm120/v1", "cast-8.72", Fraction(1047, 125), Fraction(1, 1000), Fraction(1648180439, 446072750000))
    issue = rows["pearlCGammaSm120v1Cast8p72Rev1Cap1000_8192"]
    assert (issue["path"], issue["fadd"], issue["cap"], issue["gamma"]) == (
        "cast-8.72", Fraction(8), Fraction(1, 1000), Fraction(205211, 55610750))
    assert stmt["cap"] == Fraction(1, 1000) and rows["gammaHot_sm120v2HotLoopCast8_publicConst64_8192"]["cap"] is None
    assert stmt["gamma"] == Fraction(216793, 59950000)
    wide = rows["pearlCGammaSm120v2LoopCast8p72Cap1000_8192"]
    assert (wide["path"], wide["fadd"], wide["gamma"], wide["status"]) == ("cast-8.72", Fraction(1047, 125),
                                                                           Fraction(1555901, 419688350), "staged")
    hot = rows["gammaHot_sm120v2HotLoopCast8_publicConst64_8192"]
    assert (hot["statement"], hot["sizing"], hot["kind"], hot["gamma"]) == ("pearl-c-sm120/v2-hot", "publicConst64", "value",
                                                                           Fraction(1521365753, 420071150000))
    fp4 = rows["gammaFp4ChainOnly_sm120Loop_lut256_8192"]
    assert (fp4["statement"], fp4["path"], fp4["reading"], fp4["gamma"]) == ("pearl-c-fp4/v1", "lut256", "chain-only",
                                                                             Fraction(26680734301, 1376663200000))


LIFTED = "Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192"
LIFTED_G = "Pouw.Guarantees.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_8192"
LIFTED_SOURCE = """namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions

def GammaSm120v1LoopCast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : Pouw.PearlC.CostModel Q R S)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((1648180439 / 446072750000 : ℚ) : ℝ) εPearlC

end Pouw.Guarantees.PearlC
"""


def lifted_audit(tmp_path: Path, source: str = LIFTED_SOURCE, recorded: bool = True) -> tuple[dict, Path]:
    lean = tmp_path / "Security"
    (lean / "Definitions" / "Pouw" / "Guarantees").mkdir(parents=True)
    (lean / "Definitions" / "Pouw" / "Guarantees" / "PearlC.lean").write_text(source)
    audit = {"guarantees": {LIFTED: {"type_hash": "l", "signature": [f"{LIFTED} :", f"  {LIFTED_G}"]}},
             "reads": {"Definitions.Pouw.Guarantees.PearlC": {"definitions": {LIFTED_G: "h"} if recorded else {}, "pins": [LIFTED]}}}
    return audit, lean


def test_a_lifted_guarantee_s_twin_is_read_from_its_recorded_definition(tmp_path):
    """A twin pinned as `X : G` is the twin its old theorem was: γ, cast, prices and cap read off `G`'s body."""
    audit, lean = lifted_audit(tmp_path)
    (row,) = P.extract(P.guarantees(audit), "t", {LIFTED: "l"}, P.lifted_definitions(audit, lean))
    old = P.parse("Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192",
                  PINS["Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192"])
    assert row == old | {"pin": LIFTED, "type_hash": "l", "source": "t", "status": "pinned"}


@pytest.mark.parametrize("source, recorded, match", [
    (LIFTED_SOURCE, False, "no Guarantees definition"),
    (LIFTED_SOURCE.replace("def GammaSm120", "def OtherSm120"), True, "no Guarantees definition"),
    (LIFTED_SOURCE.replace("namespace Pouw.Guarantees.PearlC\n", "namespace Pouw.Guarantees\n"), True, "no Guarantees definition"),
    (LIFTED_SOURCE.replace("((1648180439 / 446072750000 : ℚ) : ℝ)", "γ"), True, "states no literal γ"),
    (LIFTED_SOURCE.replace("(218 / 25 - ", "(16 - "), True, "prices the cast"),
])
def test_a_lifted_twin_without_a_recorded_definition_stating_its_gamma_fails_closed(tmp_path, source, recorded, match):
    audit, lean = lifted_audit(tmp_path, source, recorded)
    with pytest.raises(P.TwinError, match=match):
        P.extract(P.guarantees(audit), "t", {}, P.lifted_definitions(audit, lean))
    with pytest.raises(P.TwinError, match="no Guarantees definition"):
        P.parse(LIFTED, P.guarantees(audit)[LIFTED])


def test_a_store_pin_the_repository_holds_lifted_is_one_twin_the_store_s(tmp_path):
    """The store's pins keep their names from before the lift; the repository's lifted copy of one is the same pin, not
    a second twin stating it."""
    old = "Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_8192"
    audit, lean = lifted_audit(tmp_path)
    (lean / "lean-audit.json").write_text(json.dumps(audit))
    store = tmp_path / "store"
    for rel in (P.LEAN_STORE, *P.STAGED):
        (store / rel).parent.mkdir(parents=True, exist_ok=True)
        (store / rel).write_text(json.dumps({"pins": {old: PINS[old]} if rel == P.LEAN_STORE else {}}))
    (twin,) = P.build(store, lean / "lean-audit.json")["twins"]
    assert (twin["pin"], twin["source"], twin["status"]) == (old, P.LEAN_STORE, "pinned")


def test_every_pinned_twin_of_the_repository_is_its_audit_s_pin_and_states_its_gamma():
    """The table's twins from the repository's audit name pins it holds, at their type hashes, and reading those pins now,
    lifted guarantees included, gives the table's rows: a pin that moves or goes fails here until the table follows it."""
    audit = json.loads(P.REPO_AUDIT.read_text())
    defs = P.lifted_definitions(audit, P.REPO_AUDIT.parent)
    table = [t for t in P.load() if t["source"] == P.REPO_SOURCE]
    assert table and all(t["status"] == "pinned" for t in table)
    assert LIFTED in {t["pin"] for t in table}
    pins = P.guarantees(audit)
    for t in table:
        assert pins[t["pin"]]["type_hash"] == t["type_hash"], t["pin"]
        assert P._jsonable(P.parse(t["pin"], pins[t["pin"]], defs)) | {"source": t["source"], "status": "pinned"} == t


def test_a_pin_whose_signature_isn_t_its_name_is_an_error():
    bad = {k: v for k, v in PINS.items() if "v2LoopCast8p72" in k}
    (name, pin), = bad.items()
    with pytest.raises(P.TwinError, match="device record and prices"):
        P.parse(name, {"signature": [line.replace("sm120Loop", "sm120") for line in pin["signature"]]})
    with pytest.raises(P.TwinError, match="prices the cast"):
        P.parse(name, {"signature": [line.replace("(218 / 25)", "16") for line in pin["signature"]]})
    with pytest.raises(P.TwinError, match="name's cap 1/1000"):
        P.parse(name, {"signature": [line.replace("(1 / 1000)", "(1 / 400)") for line in pin["signature"]]})


@pytest.mark.parametrize("statement, path, reading, published, at", [
    # the panel's published figures (30 Sep 16:00Z), each from its pin, the larger of the two prices;
    # v1's at its cap then, 1/400
    ("pearl-c-sm120/v1", "cast-32.06", "forming-credited", "0.77888%", "8"),
    ("pearl-c-sm120/v1", "cast-16", "forming-credited", "0.60019%", "1047/125"),
    ("pearl-c-sm120/v1", "cast-8", "forming-credited", "0.51105%", "1047/125"),
    ("pearl-c-sm120/v1", "cast-8.72", "forming-credited", "0.51908%", "1047/125"),
    ("pearl-c-sm120/v1", "cast-32.06", "chain-only", "1.22614%", "8"),
    ("pearl-c-sm120/v2", "cast-8", "forming-credited", "0.36218%", "1047/125"),
    ("pearl-c-sm120/v2", "cast-8.72", "forming-credited", "0.37073%", "1047/125"),
    ("pearl-c-sm120/v2", "cast-16", "forming-credited", "0.45706%", "1047/125"),
    ("pearl-c-sm120/v2", "cast-8.72", "chain-only", "0.84608%", "1047/125"),
    ("pearl-c-sm120/v2-hot", "cast-32.06", "forming-credited", "0.64670%", "1047/125"),
    ("pearl-c-sm120/v1-hot", "cast-8", "forming-credited", "0.51176%", "1047/125"),
    ("pearl-c-fp4/v1", "lut256", "forming-credited", "0.71732%", "1047/125"),
    ("pearl-c-fp4/v1", "lut256", "chain-only", "1.93807%", "1047/125"),
])
def test_the_checked_in_table_gives_the_panel_s_published_gamma(statement, path, reading, published, at):
    g = P.lookup(P.load(), statement, path, 8192, 8192, reading=reading,
                 cap=Fraction(1, 400) if statement == "pearl-c-sm120/v1" else None)
    assert g["missing"] == [] and f"{float(Fraction(g['published'])) * 100:.5f}%" == published and g["published_at"] == at
    assert set(g["by_price"]) == {"8", "1047/125"} and all(v["pin"].startswith("Pouw.PearlC.") for v in g["by_price"].values())


@pytest.mark.parametrize("n, issue, loop", [(8192, "205211/55610750", "1648180439/446072750000"),
                                             (16384, "5273/1466810", "1058166877/294158650000")])
def test_v1_publishes_at_its_cap_1_1000_from_both_price_twins(n, issue, loop):
    """v1's statement is at 1/1,000 now: its twins at the cast 8.72 are DeviceSm120CapRev1Gamma's at FADD 8.00 (0.36901% at
    8,192³, 0.35949% at 16,384³) and #838's at the loop price (0.36949%, 0.35973%), the larger published; its 1/400 twins
    are history."""
    t = P.load()
    g = P.lookup(t, "pearl-c-sm120/v1", "cast-8.72", n, n)
    assert g["cap"] == "1/1000" and g["missing"] == [] and g["pins"] == ["pinned"]
    assert {p: v["gamma"] for p, v in g["by_price"].items()} == {"8": issue, "1047/125": loop}
    assert (g["published"], g["published_at"]) == (loop, "1047/125")
    assert g["by_price"]["8"]["pin"] == f"Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_{n}"
    assert all(P.lookup(t, "pearl-c-sm120/v1", path, n, n)["by_price"] == {} for path in ("cast-8", "cast-16", "cast-32.06"))
    assert P.lookup(t, "pearl-c-sm120/v2", "cast-8.72", n, n)["published"] is not None


LLAMA = {(6144, 4096): ("508661/135634400", "851373343/226641700000"),
         (4096, 4096): ("1359/350750", "295006781/75957050000"),
         (28672, 4096): ("928813/261481000", "1863165467/524322650000"),
         (4096, 14336): ("106307/27386000", "3045273083/782596025000")}


@pytest.mark.parametrize("nk", list(LLAMA), ids=lambda nk: f"n{nk[0]}k{nk[1]}")
def test_v1_publishes_at_llama_3_1_8b_s_linears_from_both_price_twins(nk):
    """A linear (n, k) that isn't a pinned cube reads its own twins, at any batch m: Llama-3.1-8B's four at the cast 8.72
    and the cap 1/1,000 (`ServedFadd8Rev1Gamma` at FADD 8.00, `ServedRev1Gamma` at the loop price), the larger published."""
    (n, k), (issue, loop) = nk, LLAMA[nk]
    g = P.lookup(P.load(), "pearl-c-sm120/v1", "cast-8.72", n, k)
    assert g["missing"] == [] and g["pins"] == ["pinned"] and (g["shape"], g["layout"]) == (None, None)
    assert {p: v["gamma"] for p, v in g["by_price"].items()} == {"8": issue, "1047/125": loop}
    assert (g["published"], g["published_at"]) == (loop, "1047/125")
    assert g["by_price"]["8"]["pin"] == f"Pouw.SecurityProofs.PearlC.GammaSm120v1Cast8p72Rev1Cap1000_n{n}k{k}"


def test_a_layout_reads_its_own_twins_whose_gamma_is_its_worst_linear_s():
    t = P.load()
    g = P.lookup(t, "pearl-c-sm120/v1", "cast-8.72", None, None, layout="llama31_8b")
    assert g["missing"] == [] and (g["published"], g["published_at"]) == (LLAMA[(4096, 14336)][1], "1047/125")
    assert g["by_price"]["1047/125"]["pin"] == "Pouw.SecurityProofs.PearlC.GammaSm120v1LoopCast8p72Rev1Cap1000_llama31_8b"
    assert P.lookup(t, "pearl-c-sm120/v1", "cast-8.72", 6144, 4096, layout="qwen3_8b")["missing"] == [
        f"a forming-credited twin of pearl-c-sm120/v1 on cast-8.72 at layout qwen3_8b, FADD {f}, cap 1/1000"
        for f in ("8", "1047/125")]


def test_a_twin_whose_domain_isn_t_its_name_s_linear_or_layout_is_an_error():
    sig = ("Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 : Pouw.PearlC.Gγ CM "
           "(Pouw.PearlC.pearlCProtocolDevRev1K (Pouw.PearlC.devSm120v1 Pouw.PearlC.Prices.sm120Loop) sem (1 / 1000) "
           "(instHSub.hSub (218 / 25) (8953 / 1000))) (Pouw.PearlC.pearlCDomainDevNK "
           "(Pouw.PearlC.devSm120v1 Pouw.PearlC.Prices.sm120Loop) 6144 4096) (851373343 / 226641700000).cast Pouw.PearlC.εPearlC")
    name = "Pouw.PearlC.pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n6144k4096"
    assert (P.parse(name, {"signature": [sig]}) or {}).get("n") == 6144
    with pytest.raises(P.TwinError, match="isn't the linear"):
        P.parse(name, {"signature": [sig.replace(" 6144 4096)", " 4096 4096)")]})
    with pytest.raises(P.TwinError, match="isn't the layout llama31_8bLinears"):
        P.parse(name.replace("n6144k4096", "llama31_8b"), {"signature": [sig]})


def test_a_repository_only_refresh_keeps_the_store_s_rows_and_rereads_the_repository_s(tmp_path):
    """`--repository-only` is `build` when only the repository's audit changed: the store's rows as checked in, the
    repository's re-read."""
    doc = json.loads(P.TABLE.read_text())
    assert P.refresh(doc) == doc, "price_twins.json is stale: run python benchmarks/pouw/harness/price_twins.py --repository-only"
    audit = json.loads(P.REPO_AUDIT.read_text())
    gone = next(t["pin"] for t in doc["twins"] if t["source"] == P.REPO_SOURCE)
    del audit["guarantees"][gone]
    (tmp_path / "lean-audit.json").write_text(json.dumps(audit))
    for root in audit["roots"]:
        (tmp_path / root).symlink_to(P.REPO_AUDIT.parent / root)
    assert gone not in {t["pin"] for t in P.refresh(doc, tmp_path / "lean-audit.json")["twins"]}


def test_a_lookup_names_what_it_lacks_and_the_table_stays_small():
    t = P.load()
    assert P.lookup(t, "pearl-c-sm120/v2", None, 8192, 8192)["missing"] == [
        "the priced path (GateRecord.prices, from the variant's prices)"]
    assert "(n, k) = (3584, 18944), FADD 8" in P.lookup(t, "pearl-c-sm120/v2", "cast-8.72", 3584, 18944)["missing"][0]
    assert P.lookup(t, "pearl-c-sm120/v1", "cast-8.72", None, 4096)["missing"] == ["the linear's (n, k), here (None, 4096)"]
    assert P.lookup(t, "pearl-c-sm120/v2", "cast-9", 8192, 8192)["published"] is None
    assert P.statement("pearl-c-sm120", "v2-hot-h3") == "pearl-c-sm120/v2-hot" and P.statement("x", None) is None
    assert P.TABLE.stat().st_size < 256 * 1024


# ---- verify.py writes the records ----------------------------------------------------------------------------------------------


def test_verify_writes_one_record_per_arm_and_shape_with_everything_the_ledger_keeps(tmp_path, capsys):
    run_dir = tmp_path / RUN
    var = variant(tmp_path)
    code, res = verify(result(run_dir, var), "--tier", "2b")
    assert code == V.EXIT_OK and res["verification"]["ok"], res["verification"]
    assert sorted(res["ledger"]["records"]) == [f"ledger/pc/{s}.json" for s in sorted(SHAPES)]
    recs = records(run_dir)
    m = recs[DECODE]["meta"]
    assert m["variant"] == var["id"] and m["variant_record"] == var and m["parent"] == "art:" + "1" * 64
    assert recs[DECODE]["refs"] == {"parent": "art:" + "1" * 64}
    assert m["tier"] == "2b" and m["run"] == RUN and m["harness"] == COMMIT and m["by"] == "bc-test"
    assert m["gates"] == {"version": L.GATES, "arm_interface": A.VERSION, "verify": V.VERSION, "sass_gate": "sass-gate/v4"}
    assert m["gate"] == {"variant": var["id"], "run": RUN, "gates": L.GATES, "shapes": [[8192, 8192]], "exact": True,
                         "poisoned_accepted": True, "controls_rejected": True, "schedules_equal": True, "sass_ok": True,
                         "toolkit": "12.9.86", "prices": "cast-8.72"}
    assert m["metrics"]["slowdown"] == 1.25 and m["metrics"]["decode_method"] == "dependent-chain"
    assert m["metrics"]["sm_clock"] == {"arm": [2090.0, 2092.0], "base": [2091.0, 2093.0]}
    assert m["steps"]["phases"]["gemm"] == 0.05 and m["steps"]["hashing_ms"] == 0.16
    assert m["steps"]["phase_groups"]["after"] == ["tile_hash", "screen"]
    assert m["gamma"]["published"] == V2_HOT_872["published"] and m["gamma"]["path"] == "cast-8.72"
    assert m["gamma"]["by_price"] == V2_HOT_872["by_price"] and m["gamma"]["arm_claim"] == ARM_GAMMA
    assert m["hypothesis"] == "tile hashing on the side lane hides it under the next call"
    assert m["mechanism"] == ["cast:packed-sts64", "hash:side-stream"] and m["approach"] == "pouw/pearl-c-sm120"
    assert m["verify"]["accepted"] and RUN in m["verify"]["transcript"]
    assert m["verify"]["negative_control"].startswith(f"REJECT {RUN}/negative/pc/{DECODE}/")
    assert m["missing"] == []
    assert recs[DECODE]["labels"] == [[L.KERNEL_OUTCOME, "screened"]]
    assert "phase" not in recs[PREFILL]["meta"]["steps"] or recs[PREFILL]["meta"]["steps"].get("phases") is None
    args = res["shapes"][1]["arms"]["pc"]["panel"]["panel_args"]
    assert args[args.index("--negative-control") + 1].startswith("REJECT")
    assert capsys.readouterr().out.count("--negative-control") == 2


def test_the_run_s_flags_win_over_the_arm_s_lineage_and_a_bare_arm_still_gets_a_record(tmp_path):
    run_dir = tmp_path / RUN
    code, _ = verify(result(run_dir, None, gamma=None, lineage={}), "--hypothesis", "  a   run's own\nline ",
                     "--parent", "pearl-c-sm120/0123456789ab", "--outcome", "dropped", "--reason", "slower than its parent")
    assert code == V.EXIT_OK
    r = records(run_dir)[PREFILL]
    m = r["meta"]
    assert m["hypothesis"] == "a run's own line" and m["parent"] == "pearl-c-sm120/0123456789ab" and r["refs"] == {}
    assert m["tier"] == "2a" and r["labels"] == [[L.KERNEL_OUTCOME, "dropped"], [L.KERNEL_REASON, "slower than its parent"]]
    assert m["missing"] == ["variant", "the priced path (GateRecord.prices, from the variant's prices)"]
    assert m["gamma"]["published"] is None and m["gamma"]["arm_claim"] is None
    assert m["gate"]["variant"] is None and m["gate"]["controls_rejected"]


def test_bad_run_flags_are_usage_errors_and_nothing_is_written(tmp_path):
    run_dir = tmp_path / RUN
    path = result(run_dir, None)
    assert V.main([str(path), "--run-id", RUN, "--commit", COMMIT, "--outcome", "dropped"]) == V.EXIT_USAGE
    assert V.main([str(path), "--run-id", RUN, "--commit", COMMIT, "--hypothesis", "x" * 241]) == V.EXIT_USAGE
    assert not (run_dir / "ledger").exists()
    assert V.main([str(path), "--run-id", RUN, "--commit", COMMIT, "--no-ledger"]) == V.EXIT_OK
    assert not (run_dir / "ledger").exists()


def test_a_control_the_verifier_accepts_rejects_its_row(tmp_path):
    run_dir = tmp_path / RUN
    path = result(run_dir, None)
    (run_dir / "negative" / "pc" / PREFILL / "manifest.json").write_text('{"accept me": 1}')
    code, res = verify(path)
    assert code == V.EXIT_REJECTED
    p = res["shapes"][0]["arms"]["pc"]["panel"]
    assert p["verification"] == {"accepted": False, "why": "the verifier accepted the negative control"}
    rows = {r["shape"]: r for r in res["verification"]["rows"]}
    assert rows[DECODE]["accepted"] and rows[DECODE]["control"]["rejected"]
    assert not records(run_dir)[PREFILL]["meta"]["verify"]["accepted"]


def test_a_missing_control_leaves_the_row_accepted_without_a_negative_control(tmp_path):
    run_dir = tmp_path / RUN
    path = result(run_dir, variant(tmp_path))
    (run_dir / "negative" / "pc" / DECODE / "manifest.json").unlink()
    code, res = verify(path)
    assert code == V.EXIT_OK
    p = res["shapes"][1]["arms"]["pc"]["panel"]
    assert p["verification"] == {"accepted": True} and "negative_control" not in p
    assert "--negative-control" not in p["panel_args"]
    assert records(run_dir)[DECODE]["meta"]["missing"] == ["negative control"]


def test_per_shape_control_runs_don_t_fail_the_step(tmp_path):
    run_dir = tmp_path / RUN
    path = result(run_dir, None)
    script = tmp_path / "s.py"
    script.write_text("import hashlib, sys\nraw = open(sys.argv[1], 'rb').read()\nsha = hashlib.sha256(raw).hexdigest()\n"
                      "neg = b'nothing' in raw\nprint(('REJECT' if neg else 'ACCEPT') + ' ' + sys.argv[2] + ' sha256:' + sha)\n"
                      "sys.exit(1 if neg else 0)\n")
    res = json.loads(path.read_text())
    res["arms"]["pc"]["verifier"] = A.check_verifier("pc", {"verifier": {
        "transcript": "transcripts/{arm}/{shape}/manifest.json", "control": "negative/{arm}/{shape}/manifest.json",
        "commands": [["{python}", str(script), "{transcript}", "{run_id}"]]}})
    path.write_text(json.dumps(res))
    code, res = verify(path)
    assert code == V.EXIT_OK and res["verification"]["ok"]
    cmds = res["verification"]["commands"]
    assert [c["control"] for c in cmds] == [False, True, False, True] and [c["exit"] for c in cmds] == [0, 1, 0, 1]
    control = (run_dir / "negative" / "pc" / PREFILL / "manifest.json").read_bytes()
    assert res["shapes"][0]["arms"]["pc"]["panel"]["negative_control"] == f"REJECT {RUN} sha256:" + hashlib.sha256(
        control).hexdigest()


def test_check_verifier_refuses_a_bad_control():
    good = {"transcript": "t/{arm}/{shape}.json", "commands": [["v"]]}
    assert A.check_verifier("x", {"verifier": good | {"control": "n/{arm}/{shape}.json"}})["control"] == "n/{arm}/{shape}.json"
    with pytest.raises(A.ArmError, match="control must be another file"):
        A.check_verifier("x", {"verifier": good | {"control": good["transcript"]}})
    with pytest.raises(A.ArmError, match="control .* must name"):
        A.check_verifier("x", {"verifier": good | {"control": "n/{arm}.json"}})


# ---- the store -----------------------------------------------------------------------------------------------------------------


def test_publishing_twice_is_one_record_each_and_the_labels_are_not_repeated(tmp_path):
    run_dir = tmp_path / RUN
    verify(result(run_dir, variant(tmp_path)), "--tier", "2b", "--publish")
    res = json.loads((run_dir / "bench.json").read_text())
    first = res["ledger"]["published"]
    assert len(first) == 2 and all(x["labels"] == [L.KERNEL_OUTCOME] for x in first)
    assert L.main(["publish", str(run_dir), "--by", "bc-test"]) == 0
    st = L.open_store()
    again = L.publish(L.read(run_dir), store=st, by="bc-test")
    assert sorted(x["art"] for x in again) == sorted(x["art"] for x in first) and all(x["labels"] == [] for x in again)
    art = first[0]["art"]
    assert st.get_manifest(art).kind == L.KIND
    assert [lab.value for lab in st.labels(art, key=L.KERNEL_OUTCOME)] == ["screened"]
    assert len(L.attempts(st)) == 2


def test_an_outcome_label_is_refused_off_vocabulary_and_changes_the_panel_row(tmp_path):
    run_dir = tmp_path / RUN
    verify(result(run_dir, variant(tmp_path)), "--tier", "2b", "--publish")
    st = L.open_store()
    art = next(a for a, m, _ in L.attempts(st) if m["shape"] == PREFILL)
    with pytest.raises(L.LedgerError, match="needs a one-line reason"):
        L.outcome(art, "voided", reason=None, by="bc-test", store=st)
    assert L.outcome(art, "voided", reason="timed another commitment", by="bc-test", store=st) == [L.KERNEL_OUTCOME,
                                                                                                    L.KERNEL_REASON]
    assert L.outcome(art, "voided", reason="timed another commitment", by="bc-test", store=st) == []
    row = next(r for r in L.panel_rows(L.attempts(st)) if r["record"] == art)
    assert row["voided"] == "timed another commitment" and "outcome voided: timed another commitment" in row["description"]


def test_store_records_come_back_as_panel_rows_under_the_panel_s_rules(tmp_path):
    run_dir = tmp_path / RUN
    var = variant(tmp_path)
    verify(result(run_dir, var), "--tier", "2b", "--publish")
    rows = {r["shape"]: r for r in L.panel_rows(L.attempts())}
    r = rows[DECODE]
    assert r["kind"] == "measured" and r["line"] == "pearl-c-sm120" and r["version"] == "v2-hot" and r["attempt"] is None
    assert r["verifier"]["negative_control"].startswith("REJECT") and r["sm_clock_mhz"]["baseline"] == [2091.0, 2093.0]
    assert r["gamma"] == float(Fraction(V2_HOT_872["published"])) and r["decode_method"] == "dependent-chain"
    assert r["precision"] == "fp8" and "γ cast-8.72 at FADD 1047/125, the larger, from staged price twins" in r["description"]
    assert r["source"].endswith("(kernel-attempt/v1)") and var["id"] in r["description"] and "store_refused" not in r

    screen = tmp_path / "screen" / RUN
    verify(result(screen, var), "--publish")
    shapes = {x["shape"] for x in L.panel_rows(L.attempts())}
    assert {f"{PREFILL}:screen", f"{DECODE}:screen"} <= shapes

    bare = tmp_path / "bare" / RUN
    path = result(bare, var)
    (bare / "negative" / "pc" / PREFILL / "manifest.json").unlink()
    verify(path, "--tier", "2b", "--publish")
    refused = [x for x in L.panel_rows(L.attempts()) if x["run"] == RUN and x.get("store_refused")]
    assert refused and refused[0]["store_refused"] == "no verifier reject line, naming the run, on the negative control"
    assert "verifier" not in refused[0]


def test_measured_refusal_follows_panel_append():
    meta = {"run": RUN, "gates": {"version": L.GATES},
            "verify": {"accepted": True, "accept": f"ACCEPT {RUN}", "transcript": f"{RUN}:t sha256:x",
                       "negative_control": f"REJECT {RUN}/negative"},
            "metrics": {"sm_clock": {"arm": [2090.0], "base": [2091.0]}}}
    assert L.measured_refusal(meta) is None
    assert "predates" in L.measured_refusal(meta | {"gates": {"version": "pouw-gates/1"}})
    assert "naming the run" in L.measured_refusal(meta | {"verify": meta["verify"] | {"negative_control": "REJECT other"}})
    assert "doesn't name the run" in L.measured_refusal(meta | {"verify": meta["verify"] | {"transcript": "elsewhere"}})
    assert "SM clocks" in L.measured_refusal(meta | {"metrics": {"sm_clock": {"arm": [2090.0], "base": []}}})


def test_the_ledger_s_words_are_the_store_s():
    _, vocab = L.store_module()
    from research.store import kinds

    assert L.OUTCOMES == vocab.KERNEL_OUTCOMES
    assert (L.KERNEL_OUTCOME, L.KERNEL_REASON) == (vocab.KERNEL_OUTCOME, vocab.KERNEL_REASON)
    assert vocab.check(L.KERNEL_OUTCOME, "frontier") is None and vocab.check(L.KERNEL_OUTCOME, "maybe")
    assert L.KIND in kinds.KINDS


def test_an_arm_that_types_gamma_in_gets_no_record_and_the_run_says_so(tmp_path, capsys):
    run_dir = tmp_path / RUN
    lineage = {"hypothesis": "x", "gamma": {"8": "1/1000", "1047/125": "1/1000"}}
    code, res = verify(result(run_dir, variant(tmp_path), lineage=lineage), "--tier", "2b", "--publish")
    assert code == V.EXIT_OK and res["verification"]["ok"]
    assert res["ledger"]["records"] == [] and res["ledger"]["published"] == []
    assert "typed into its lineage" in res["ledger"]["refused"]["pc"]
    assert "REFUSED pc's records" in capsys.readouterr().err
    assert L.records(json.loads((run_dir / "bench.json").read_text()), run_id=RUN, commit=COMMIT) == []
    assert L.attempts() == []


def test_a_scheme_that_isn_t_the_panel_version_s_device_record_publishes_no_gamma(tmp_path):
    run_dir = tmp_path / RUN
    lineage = {"hypothesis": "x", "scheme": "pearl-c-sm120-v1-h3"}
    verify(result(run_dir, variant(tmp_path), lineage=lineage), "--tier", "2b")
    g = records(run_dir)[PREFILL]["meta"]["gamma"]
    assert g["published"] is None and "scheme pearl-c-sm120-v1-h3 isn't pearl-c-sm120/v2-hot's device record" in g["missing"]
    assert L._scheme_conflict("pearl-c-sm120-unpromoted-v1-h1", "pearl-c-sm120/v2") is None
    assert L._scheme_conflict("pearl-c-sm120-unpromoted-v1-h1", "pearl-c-sm120/v1")


def test_a_harness_poisoned_dump_s_control_is_judged_without_a_declared_one(tmp_path):
    run_dir = tmp_path / RUN
    code, res = verify(result(run_dir, variant(tmp_path), control=False), "--tier", "2b")
    assert code == V.EXIT_OK
    rows = {r["shape"]: r for r in res["verification"]["rows"]}
    assert rows[PREFILL]["control"]["path"] == f"negative/pc/{PREFILL}/manifest.json" and rows[PREFILL]["control"]["rejected"]
    m = records(run_dir)[PREFILL]["meta"]
    assert m["verify"]["negative_control"].startswith(f"REJECT {RUN}/negative/pc/{PREFILL}/") and m["gate"]["poisoned_accepted"]


def test_an_arm_that_dumps_its_own_transcripts_has_no_harness_poisoning_to_its_name(tmp_path):
    run_dir = tmp_path / RUN
    code, res = verify(result(run_dir, variant(tmp_path), control=False, poisoned_by="arm"), "--tier", "2b")
    assert code == V.EXIT_OK and "control" not in res["verification"]["rows"][0]
    m = records(run_dir)[PREFILL]["meta"]
    assert m["verify"]["accepted"] and not m["gate"]["poisoned_accepted"] and "negative control" in m["missing"]


def test_a_run_time_choice_of_code_is_part_of_the_variant_s_id(tmp_path):
    """GPU 2's pilot finding (18:00Z): PEARLC_A_FORMS picks kernels from one binary at run time, so the forms shared an id."""
    d = kernel_dir(tmp_path)
    base = dict(family="pearl-c", device="sm120", schemes={"s"}, sources=["run.py", "k.cu"], root=d, binaries=[d / "k.cubin"])
    plain, fold, nodes = (L.variant_record(**base, config=c) for c in (None, {"PEARLC_A_FORMS": "p,fold"},
                                                                          {"PEARLC_A_FORMS": "w,nodes"}))
    assert len({plain["id"], fold["id"], nodes["id"]}) == 3 and fold["config"] == {"PEARLC_A_FORMS": "p,fold"}
    assert L.variant_record(**base)["id"] == plain["id"] == variant(tmp_path)["id"] and plain["config"] is None
    two = {"PEARLC_A_FORMS": "p,fold", "PEARLC_EPILOGUE": "b"}
    assert L.variant_record(**base, config=two)["id"] == L.variant_record(**base, config=dict(reversed(two.items())))["id"]
    (d / L.CONFIG_PATH).write_text("x")
    with pytest.raises(L.LedgerError, match="may not be named"):
        L.source_sha256([L.CONFIG_PATH], d)
    with pytest.raises(L.LedgerError, match="names to strings"):
        L.source_sha256(["k.cu"], d, {"PEARLC_A_FORMS": 1})


def test_the_runner_publishes_the_records_verify_declares_with_the_same_ids(tmp_path):
    """The pilot's gap (18:26Z): under `research run --on` the store key is the runner's, not the workload's. verify.py
    declares its records in outputs.json, and the runner's typed_outputs puts the same manifests `ledger.publish` would."""
    from research.store.attempt import typed_outputs

    run_dir = tmp_path / RUN
    path = result(run_dir, variant(tmp_path))
    (run_dir / "outputs.json").write_text(json.dumps({"schema": "research/outputs/v0.1", "outputs": [
        {"name": "earlier", "kind": "evidence/v1", "meta": {"x": 1}}]}))
    code, res = verify(path, "--tier", "2b")
    assert code == V.EXIT_OK and res["ledger"]["outputs"] == "outputs.json"
    doc = json.loads((run_dir / "outputs.json").read_text())
    assert [e["name"] for e in doc["outputs"]] == ["earlier", f"kernel-attempt/pc/{PREFILL}", f"kernel-attempt/pc/{DECODE}"]
    st = L.open_store()
    by_runner = typed_outputs(run_dir, st, RUN)
    by_lane = L.publish(L.read(run_dir), store=st, by="bc-test")
    assert {by_runner[f"kernel-attempt/pc/{x['shape']}"] for x in by_lane} == {x["art"] for x in by_lane}
    verify(path, "--tier", "2b")
    assert [e["name"] for e in json.loads((run_dir / "outputs.json").read_text())["outputs"]].count(
        f"kernel-attempt/pc/{PREFILL}") == 1
