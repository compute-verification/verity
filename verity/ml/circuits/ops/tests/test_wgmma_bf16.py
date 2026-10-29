"""``sm90.wgmma.m64n8k16.bf16`` against the H100 capture (``fixtures/tc-hopper-wgmma-bf16-2026-09-26``, see the
fixtures README): every recorded word reproduces through the registered model's total step, one wgmma per record or four
chained on one accumulator, on both A operand paths; the Ampere pipeline does not.  A run of the legacy ``mma.sync.m16n8k16``
on a second H100 against the same model shows the two instructions compute the same words, non-finite ones included."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from verity.errors import InvalidArtifact
from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
from verity.ml.circuits.silicon.models import HOPPER_BF16_M16N8K16, HOPPER_BF16_WGMMA_K16, MODELS, tc_dot_chain
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16
from verity.ml.circuits.silicon.total import tc_dot_total

CAPTURE = Path(__file__).parent / "fixtures" / "tc-hopper-wgmma-bf16-2026-09-26" / "records.json.gz"
DOC = json.loads(gzip.open(CAPTURE, "rt").read())


def _words(hex_text: str) -> list[int]:
    return [int(hex_text[i : i + 4], 16) for i in range(0, len(hex_text), 4)]


def _step_total(pipeline, acc: int, a: list[int], b: list[int]) -> int:
    for s in range(0, len(a), 16):
        acc = tc_dot_total(pipeline, acc, a[s : s + 16], b[s : s + 16])
    return acc


def test_the_instruction_is_registered_with_its_own_model() -> None:
    ins = INSTRUCTIONS["sm90.wgmma.m64n8k16.bf16"]
    assert ins.model is HOPPER_BF16_WGMMA_K16 is MODELS["hopper_bf16_wgmma_k16"]
    assert ins.ptx == DOC["ptx"] and ins.shape == (64, 8, 16) and ins.status == "pinned"
    assert (ins.model.groups, ins.model.width, ins.model.zero_exponent) == (
        HOPPER_BF16_M16N8K16.groups, HOPPER_BF16_M16N8K16.width, HOPPER_BF16_M16N8K16.zero_exponent)


def test_every_captured_word_reproduces() -> None:
    by_path: dict[tuple[str, str], int] = {}
    for r in DOC["records"]:
        a, b, acc, d = _words(r["a"]), _words(r["b"]), int(r["acc"], 16), int(r["d"], 16)
        assert len(a) == len(b) == 16 * r["steps"]
        assert _step_total(HOPPER_BF16_WGMMA_K16, acc, a, b) == d, r
        run = DOC["runs"][r["run"]]
        by_path[run["instruction"], run["a_source"]] = by_path.get((run["instruction"], run["a_source"]), 0) + 1
    assert set(by_path) == {("sm90.wgmma.m64n8k16.bf16", "smem"), ("sm90.wgmma.m64n8k16.bf16", "registers"), ("sm90.mma.m16n8k16.bf16", "n/a")}
    assert min(by_path.values()) > 6_000


def test_finite_chains_agree_with_tc_dot_chain() -> None:
    """Four back-to-back wgmma on one accumulator are the chain of four single steps (no wider carry between instructions)."""
    chains = [r for r in DOC["records"] if r["steps"] == 4 and r["family"] != "chain4/specials"]
    assert len(chains) >= 2 * 13 * 30
    for r in chains:
        acc = int(r["acc"], 16)
        if (acc >> 23) & 0xFF == 0xFF:
            continue
        a, b = _words(r["a"]), _words(r["b"])
        if any((w >> 7) & 0xFF == 0xFF for w in a + b):
            continue
        try:
            assert tc_dot_chain(HOPPER_BF16_WGMMA_K16, acc, a, b) == int(r["d"], 16), r
        except InvalidArtifact as exc:  # the finite model refuses an intermediate infinity; the total step covers those
            assert _step_total(HOPPER_BF16_WGMMA_K16, acc, a, b) == int(r["d"], 16), (r, exc)


def test_the_capture_separates_hopper_from_ampere() -> None:
    wrong = sum(_step_total(AMPERE_BF16_M16N8K16, int(r["acc"], 16), _words(r["a"]), _words(r["b"])) != int(r["d"], 16)
                for r in DOC["records"])
    assert wrong > 100


def test_mma_and_wgmma_write_the_same_words_on_the_h100() -> None:
    """The specials family has the same inputs on every BF16 k16 instruction: its words, NaN and infinity outputs included, are the
    same on mma.sync (one H100) and on wgmma with A from shared memory or registers (another H100)."""
    words: dict[str, dict[tuple, int]] = {}
    for r in DOC["records"]:
        if r["family"] == "specials/all":
            run = DOC["runs"][r["run"]]
            words.setdefault(f"{run['instruction']}/{run['a_source']}", {})[r["acc"], r["a"], r["b"]] = int(r["d"], 16)
    mma = words.pop("sm90.mma.m16n8k16.bf16/n/a")
    assert set(words) == {"sm90.wgmma.m64n8k16.bf16/smem", "sm90.wgmma.m64n8k16.bf16/registers"}
    assert sum(((d >> 23) & 0xFF) == 0xFF for d in mma.values()) > 1_000
    for other in words.values():
        assert other == mma


def test_the_mma_run_agrees_with_the_wgmma_model_everywhere() -> None:
    (run,) = [v for v in DOC["runs"].values() if v["instruction"] == "sm90.mma.m16n8k16.bf16"]
    hyp = HOPPER_BF16_WGMMA_K16.name + "[HYPOTHESIS]"
    assert run["sass"] == ["HMMA.16816.F32.BF16"] and run["sass_count"] == 1 and run["failures"] == [] and run["chain"] is None
    assert run["declared_model"] == HOPPER_BF16_M16N8K16.name and run["device"] == "NVIDIA H100 80GB HBM3"
    assert sum(f["elements"] for f in run["sweep"].values()) >= 25_000_000
    assert all(f["mismatches"][hyp] == 0 for f in run["sweep"].values())   # specials included (the finite model raises there)
    assert run["sweep"]["specials"]["mismatches"][HOPPER_BF16_M16N8K16.name] > 0
    cap = run["capture_replay"]   # veritor's wgmma-recorded words, re-executed through mma.sync
    assert cap["records"] == 44_046 and cap["model"] == hyp and cap["mismatches"] == cap["model_mismatches"] == cap["other_positions_nonzero"] == 0


def test_the_runs_behind_the_fixture_saw_no_mismatch() -> None:
    """The fixture is a sample; the runs' own counts (from probe_results.json) cover every element."""
    wgmma = {r: v for r, v in DOC["runs"].items() if v["instruction"] == "sm90.wgmma.m64n8k16.bf16"}
    assert {v["a_source"] for v in wgmma.values()} == {"smem", "registers"}
    for run, v in wgmma.items():
        assert v["device"] == "NVIDIA H100 80GB HBM3" and v["arch"] == "sm_90a" and v["failures"] == [], run
        assert v["sass"] == ["HGMMA.64x8x16.F32.BF16"] and v["sass_count"] == 1
        model = v["declared_model"]
        assert model == HOPPER_BF16_WGMMA_K16.name
        sweep = sum(f["elements"] for f in v["sweep"].values())
        assert sweep >= 25_000_000 and all(f["mismatches"][model] == 0 for f in v["sweep"].values())
        assert "specials" in v["sweep"] and "adder_width_cancellation" in v["sweep"] and "acc_only_lsb_walk" in v["sweep"]
        assert v["chain"]["steps"] == 4 and all(f["mismatches"][model] == 0 for f in v["chain"]["families"].values())
        cap = v["capture_replay"]
        assert cap["records"] == 44_046 and cap["mismatches"] == cap["model_mismatches"] == cap["other_positions_nonzero"] == 0
