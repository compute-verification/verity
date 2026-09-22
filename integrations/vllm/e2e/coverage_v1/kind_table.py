"""The SMALL kind table: builder / checker / framing / status per VU kind, and the unsupported-check exposure (R18).

Every row below is a reading of code, cited ``file:line`` against mainline ``346aa4ac`` (the worktree
``contract/a56-coverage`` is off it).  The statuses use the contract's vocabulary only:

* ``supported (approved)`` -- an obligation builder turns the selected VU into a statement AND an approved
  guest evaluates the specified relation over authenticated openings of the VU's own boundary;
* ``supported only under an unapproved or PROPOSED guest`` -- as above, but the only checker is not approved;
* ``builder missing`` -- an approved guest evaluates the relation (a gate set names every primitive of the
  kind program at the right widths) but nothing on the Python side turns a *selected VU of this kind* into a
  statement over committed leaves;
* ``checker missing`` -- no guest in the scenario evaluates the relation;
* ``no path`` -- neither.

"exempt (charged)" is a legacy label of D4's ``expected_work.json`` and is *not* a status here.

The exposure is the probability of selecting at least one unsupported check -- **completion risk**, never a
family-level exemption (PROTOCOL correction, §Reporting requirements).  It is computed with ``decimal`` at 60
significant digits from ``1 - exp(n * ln(1 - s))``; the expected number of unsupported selected checks
``s * n`` is exact (``fractions.Fraction``).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

getcontext().prec = 60

SCHEMA = "veritor/a56-coverage/v1"
BASED_ON = "346aa4ac"

#: Guests named in the brief.
GUESTS = {
    "9fd00e60": {
        "elf_sha256": "9fd00e60fb6a74762de987cde211aca9de4576de844569da779acac6bf3863db",
        "approval": "reproduced (e2e/approved_guests.json)",
        "source": "zk/sp1 (tree 416482656ffad8a45b4589d933da04e6f9076dff; guest+common match mainline 346aa4ac)",
        "families": [
            "veritor.toy-isa@1", "veritor.word-arithmetic@2", "veritor.int8-q7@1",
            "veritor.tensor-core.ampere_bf16_m16n8k16@2", "veritor.tc-ampere-bf16@1", "veritor.tc-ampere-bf16@2",
            "veritor.pad-serve@1", "veritor.reduce-mufu@1", "veritor.elem-bf16@1", "veritor.elem-f32@1",
            "veritor.pad-inactive@1", "veritor.embed-gather@1",
        ],
        "families_evidence": "zk/sp1/common/src/gateset.rs:470-560 (GateSet::resolve); kernel/mod.rs:168-171 (gate_set from the statement's id)",
        "framings": [
            "veritor/protocol/merkle/frame/v3 per-word leaves (statement format 2; witness formats 1/2/3): kernel/openings.rs:95,391,611",
            "verity-vllm/leaf/v1 packed leaves (statement format 3 rows, format 3 witness): packed.rs:39, kernel/openings.rs:611-660,787-830",
        ],
    },
    "2389b9a8": {
        "approval": "pricing-approved; promotion PROPOSED (out/auth-promote/dossier.md), pending the user",
        "source": "guest-range/sp1 -- common/src/{gateset,kernel/relations,tc,reduce,elem,embed,pad,frame}.rs byte-identical to zk/sp1/common; only kernel/openings.rs, packed.rs, codec.rs differ (range openings, frame-v3 packed leaves)",
        "families": "same as 9fd00e60 (identical gateset.rs)",
        "framings": [
            "the two above",
            "veritor-frame-v3 packed leaves with range openings (guest-range/sp1/common/src/packed.rs:47,128-132) -- 'no store commits frame-v3 leaves; a record cannot use range openings today' (out/auth-promote/store_framing.md)",
        ],
        "adds_kinds": False,
    },
    "33cba058": {
        "approval": "unapproved (fa2-attn)",
        "source": "zk/sp1-attn/common/src/fa2.rs",
        "families": ["veritor.fa2-attn@<tables> (fa2.rs:24-36: f32_add_ftz, f32_fma_ftz, f32_fma_sub_ftz, f32_max, f32_mul_ftz, f32_sub_ftz, f32_to_bf16, fa2_inv_sum, guard_neginf_zero, in, mufu_ex2_ftz, tc_dot16, tc_dot16_0)"],
        "source_width": "`in` is 16 bits only (fa2.rs:89-93): no FP32 running value (score, max, sum, accumulator) can be opened -- the family checks a whole attention head as one relation (attn_gates.py:184-307 fold), never one interior primitive",
        "framings": ["as the shared kernel's (the attn guest reuses codec/openings)"],
    },
}

APPROVED_ONLY = "approved"
PLUS_RANGE = "plus_range"
PLUS_FA2 = "plus_fa2"
SCENARIOS = (APPROVED_ONLY, PLUS_RANGE, PLUS_FA2)


@dataclass(frozen=True)
class Checker:
    exists: bool
    guests: tuple[str, ...]
    gate_set: str
    ops: tuple[str, ...]
    evidence: tuple[str, ...]
    granularity: str = "same"  # the checker takes exactly this VU ("same"), or only a larger unit
    domain_caveat: str = ""


@dataclass(frozen=True)
class Builder:
    exists: bool
    evidence: tuple[str, ...]
    granularity: str = "same"
    note: str = ""
    fallback: str = ""  # A5/6 fallback builder (this lane), not yet wired into the record's selection -> proving pipeline


@dataclass(frozen=True)
class KindRow:
    kind: str
    circuit: str
    checker: Checker
    builder: Builder
    framings: tuple[str, ...]
    legacy: bool = False
    aliases: tuple[str, ...] = ()
    note: str = ""

    @staticmethod
    def guests_of(scenario: str) -> set[str]:
        approved = {"9fd00e60"}
        if scenario in (PLUS_RANGE, PLUS_FA2):
            approved.add("2389b9a8")
        if scenario == PLUS_FA2:
            approved.add("33cba058")
        return approved

    def checker_in_scenario(self, scenario: str, *, allow_larger_unit: bool = False) -> bool:
        """Some guest of the scenario evaluates this VU's relation over openings of the VU's own boundary
        (``allow_larger_unit``: or of a larger unit containing it, e.g. the whole attention head)."""

        if not self.checker.exists or not (set(self.checker.guests) & self.guests_of(scenario)):
            return False
        return self.checker.granularity == "same" or allow_larger_unit

    def builder_ok(self, *, with_fallbacks: bool = False) -> bool:
        if self.builder.exists and self.builder.granularity == "same":
            return True
        return with_fallbacks and bool(self.builder.fallback)

    def supported_with_fallbacks(self, scenario: str) -> bool:
        return self.checker_in_scenario(scenario) and self.builder_ok(with_fallbacks=True)

    def status(self, scenario: str) -> str:
        if self.checker_in_scenario(scenario):
            return "supported (approved)" if self.builder_ok() else "builder missing"
        # no checker at this VU's granularity in the scenario
        elsewhere = self.checker.exists and self.checker.granularity == "same" and bool(self.checker.guests)
        if self.builder_ok():
            return "supported only under an unapproved or PROPOSED guest" if elsewhere else "checker missing"
        return "no path"

    def supported(self, scenario: str) -> bool:
        return self.status(scenario) == "supported (approved)"


G = ("9fd00e60", "2389b9a8")
GS = "zk/sp1/common/src/gateset.rs"
RL = "zk/sp1/common/src/kernel/relations.rs"
OP = "zk/sp1/common/src/kernel/openings.rs"
REDUCE_IN64 = f"{GS}:607 (`in` is 64 bits in reduce-mufu: an FP32 operand word opens as a source), {GS}:742-747 (argument range check at the consuming gate's width)"
TC_CAVEAT = ("tc_dot returns None on any non-finite accumulator/operand (zk/sp1/common/src/tc.rs:77,98,239,246) -> CheckError::Argument "
             f"({RL}:110-116): the proof is REJECTED, while the registered primitive is total (verity_vllm/registry/prims.py:359-395 `_tc_dot16_total`: NaN/inf extension). "
             "A saturated group returns +-inf (tc.rs:249-251) and the NEXT chained step rejects it. Honest-completion (HC) mismatch on records with a non-finite word in a selected coordinate.")
FR_PERWORD = "veritor/protocol/merkle/frame/v3 per-word leaf (format 2)"
FR_PACKED = "verity-vllm/leaf/v1 packed row (format 3)"
NO_STORE = "none of these words is committed by today's store (operator boundaries only: e2e/predicates.py:9-13, 170-173; A2's scope)"

ROWS: tuple[KindRow, ...] = (
    KindRow(
        "GemmCoordinate_v1{K=576}|batch",
        "verity_vllm/registry/b1.py:99-113 (DotBf16 K/16 AmpereBF16TcDot16 steps from acc 0, then F2fpBf16)",
        Checker(True, G, "veritor.tc-ampere-bf16@2", ("tc_dot16_0", "tc_dot16", "f32_to_bf16 (hw NaN 0x7FFF)"),
                (f"{GS}:328-335 (TC_AMPERE_BF16_HW_GATES)", f"{GS}:750-762 (evaluate)", "zk/sp1/common/src/tc.rs:234-255 (tc_dot), 280-285 (f32_to_bf16_hw)",
                 f"{RL}:76-133 (check_relations: source gates opened, every other gate recomputed, opened outputs compared bit-exactly at :171-173)"),
                domain_caveat=TC_CAVEAT),
        Builder(True, ("verity_vllm/typed_obligation_export.py:433-560 (export_composed: GemmCoordinate<K> obligations)",
                       "src/veritor/interface/programs.py:77-107 (gemm_coordinate(K): the pinned body)",
                       "verity_vllm_adapter/obligations.py:66,135 (sampled_gemm_rows / select_coordinate_units; A2)",
                       "e2e/predicates.py:136-140 (CHECKERS 'gemm-coordinate', unit='same')")),
        (FR_PACKED, FR_PERWORD),
        note="the family is fixed to veritor.tc-ampere-bf16@2 by the exporter (typed_obligation_export.py:121); K must be a multiple of 16 (programs.py:92).",
    ),
    KindRow(
        "GemmCoordinate_v1{K=1536}|batch",
        "verity_vllm/registry/b1.py:99-113",
        Checker(True, G, "veritor.tc-ampere-bf16@2", ("tc_dot16_0", "tc_dot16", "f32_to_bf16 (hw)"),
                (f"{GS}:328-335", "zk/sp1/common/src/tc.rs:234-255", f"{RL}:76-133"), domain_caveat=TC_CAVEAT),
        Builder(True, ("verity_vllm/typed_obligation_export.py:433-560", "src/veritor/interface/programs.py:77-107"),
                note="predicates.py:139-140: K=1536 was priced on the frozen ELF 08298f55 (not approved); the approved ELF's K-generality is by code reading (programs.py:92 accepts any multiple of 16; the guest has no K limit in relations.rs)"),
        (FR_PACKED, FR_PERWORD),
    ),
    # [E7 SCALE 2026-09-12 21:10Z, D46 QWEN] Qwen2.5-0.5B's two GEMM shapes, named by E6 (night 05:25Z kind-diff; out/night/e6/qwen05/coverage_table.json):
    # K = 896 (hidden: q/k/v/o and gate_up projections) and K = 4864 (intermediate: down projection).  Same function, same checker, same exporter
    # path as K = 576 / 1536 -- a TABLE ROW, not code: `row_of` already resolved them parametrically (E6), these rows make the paper's table
    # explicit and carry the measurement status.  Not a registry kind (the GEMM coordinate is A9's exporter/checker path, not `BUILDERS`), so
    # `tests/contract/a3_sequence/registry_pin.py` is unchanged by design (D22 binds builders, and none is added here).
    KindRow(
        "GemmCoordinate_v1{K=896}|batch",
        "verity_vllm/registry/b1.py:99-113 (DotBf16 K/16 = 56 AmpereBF16TcDot16 steps from acc 0, then F2fpBf16)",
        Checker(True, G, "veritor.tc-ampere-bf16@2", ("tc_dot16_0", "tc_dot16", "f32_to_bf16 (hw)"),
                (f"{GS}:328-335", "zk/sp1/common/src/tc.rs:234-255", f"{RL}:76-133"), domain_caveat=TC_CAVEAT),
        Builder(True, ("verity_vllm/typed_obligation_export.py:433-560", "src/veritor/interface/programs.py:77-107"),
                note="K = 896 = 56 x 16 (programs.py:92); Qwen2.5-0.5B hidden size (q/k/v/o, gate_up). MEASURED with proofs: pending E10's QWEN05 live run (SCALE wave)"),
        (FR_PACKED, FR_PERWORD),
        note="Qwen2.5-0.5B (E6 kind-diff 05:25Z): K = 896 GEMM output rows are committed operator boundaries like K = 576.",
    ),
    KindRow(
        "GemmCoordinate_v1{K=4864}|batch",
        "verity_vllm/registry/b1.py:99-113 (DotBf16 K/16 = 304 AmpereBF16TcDot16 steps from acc 0, then F2fpBf16)",
        Checker(True, G, "veritor.tc-ampere-bf16@2", ("tc_dot16_0", "tc_dot16", "f32_to_bf16 (hw)"),
                (f"{GS}:328-335", "zk/sp1/common/src/tc.rs:234-255", f"{RL}:76-133"), domain_caveat=TC_CAVEAT),
        Builder(True, ("verity_vllm/typed_obligation_export.py:433-560", "src/veritor/interface/programs.py:77-107"),
                note="K = 4864 = 304 x 16 (programs.py:92); Qwen2.5-0.5B intermediate size (down projection) -- the longest chained tc_dot16 body of the three models. MEASURED with proofs: pending E10's QWEN05 live run"),
        (FR_PACKED, FR_PERWORD),
        note="Qwen2.5-0.5B (E6 kind-diff 05:25Z): K = 4864 down-proj GEMM output row is a committed operator boundary like K = 1536.",
    ),
    KindRow(
        "Input|batch",
        "IR §E prescribed input: the committed word equals its authorized source (weights: checkpoint word; prompt: the request's token)",
        Checker(True, G, "any family (pinned `in` = public input)", ("in with `expected`",),
                (f"{OP}:152-156 (format 2), :642-646 (packed), :818-822 (ranged): the decoded word must equal the statement's pinned bytes",
                 "e2e/predicates.py:141-145 (CHECKERS 'input-equation', proof='verifier' for prompt/boundary words)")),
        Builder(True, ("verity_vllm/typed_obligation_export.py:423-430 (input_equation_subcircuit: one Input<16> gate pinned)",
                       "verity_vllm_adapter/obligations.py:227,254 (input_equations / boundary_input_equations; A2)",
                       "verifier_v2/reverify.py:330-333,373 (V5.inputs: pinned weight words are Input equations; prompt words verifier-compared)")),
        (FR_PACKED, FR_PERWORD),
        legacy=True,
        aliases=("Input[prompt]", "Input[weights]|out=0", "Input[weights]|out=16"),
        note="legacy kind (D4/D2 population; PROTOCOL: inputs prescribed). A weight word pinned in-guest is a zk check; a prompt word is a verifier comparison (predicates.py:144). Derived tables (rope cos/sin) are opening-only (reverify.py:374).",
    ),
    KindRow(
        "AmpereBF16TcDot16_v1|batch",
        "verity_vllm/registry/prims.py:398-402 (one mma.sync m16n8k16 accumulator step; total, prims.py:359-395); attention QK^T / PV steps (b1.py:335-388)",
        Checker(True, G, "veritor.tensor-core.ampere_bf16_m16n8k16@2 (`in` 32 bits: the FP32 accumulator opens as a source)", ("tc_dot16",),
                (f"{GS}:319 (TENSOR_CORE_GATES)", f"{GS}:603 (`in` width = 32 for TensorCore)", f"{GS}:620-626 (arg widths [32, 16x16, 16x16])", "zk/sp1/common/src/tc.rs:234-255"),
                domain_caveat=TC_CAVEAT),
        Builder(False, ("no exporter: typed_obligation_export.py builds GemmCoordinate/InputEq only (:433-560); e2e/attn_relation.py is a verification-stage identity check, not a builder; "
                        "e2e/predicates.py:165 OPERATOR_PENDING['Attention'] refuses (S55)",),
                note=f"the accumulator and P^T operands are attention-interior: {NO_STORE}"),
        (FR_PERWORD, FR_PACKED),
        note="fa2-attn (33cba058) also evaluates tc_dot16 (fa2.rs:216-226) but with a *total* NaN extension (fa2.rs:254-298) -- the approved guest's tc_dot and the registered primitive disagree only off the finite domain.",
    ),
    KindRow(
        "DotBf16_v1{K=64}|call",
        "verity_vllm/registry/b1.py:99-106 (K/16 = 4 AmpereBF16TcDot16 steps from acc 0 -> F32): the per-key attention score",
        Checker(True, G, "veritor.tc-ampere-bf16@2 (or @1; tensor-core@2)", ("tc_dot16_0", "tc_dot16 x3"),
                (f"{GS}:320-335", "src/veritor/interface/programs.py:77-104 (gemm_coordinate(k, opened_accumulator=True) = exactly this body, outgoing FP32)",
                 "zk/sp1/common/src/tc.rs:234-255"), domain_caveat=TC_CAVEAT),
        Builder(False, ("the program exists (programs.py:77-104, `GemmAccumulator`) but no exporter emits it for attention (typed_obligation_export.py:433-560 covers GemmCoordinate/InputEq only); predicates.py:165 refuses Attention (S55)",),
                note=f"q row (qkv-proj output) and K row are operator boundaries; the FP32 score is attention-interior: {NO_STORE}"),
        (FR_PERWORD, FR_PACKED),
    ),
    KindRow(
        "F2fpBf16_v1|primitive",
        "verity_vllm/registry/prims.py:202-206 (SASS F2FP.BF16.F32: RNE, every NaN -> 0x7FFF); attention P cast (b1.py:372)",
        Checker(True, G, "veritor.elem-f32@1 (`f32_to_bf16_hw`, `in` 32) or veritor.reduce-mufu@1 (`f2fp_bf16`)", ("f32_to_bf16_hw", "f2fp_bf16"),
                (f"{GS}:377 (ELEM_F32_GATES)", f"{GS}:603 (`in` = width 32 for ElemF32)", "zk/sp1/common/src/tc.rs:280-285 (f32_to_bf16_hw)", "zk/sp1/common/src/reduce.rs:311-325 (f2fp_bf16, NaN -> 0x7FFF)")),
        Builder(False, ("no record-wired builder; zk/bench/proof_elem_batches.py:243-250 (map1_program) builds format-2 F2fpBf16 statements from B1 rows (bench, not a selected VU); predicates.py:146-156 registers only the RMSNorm 'cone' ADAPTER (a different unit, not this VU)",),
                note=f"the FP32 operand is interior: {NO_STORE}"),
        (FR_PERWORD, FR_PACKED),
        note="semantics match on every word: prims.f32_to_bf16_rn_bits(u, 0x7FFF) (prims.py:60-65) == tc.rs:280-285 == reduce.rs:312-318.",
    ),
    KindRow(
        "F2fpBf16_v1|batch",
        "verity_vllm/registry/prims.py:202-206; the RMSNorm output cast (b1.py:187,220), attention output cast (b1.py:424)",
        Checker(True, G, "veritor.elem-f32@1 / veritor.reduce-mufu@1", ("f32_to_bf16_hw", "f2fp_bf16"),
                (f"{GS}:377", "zk/sp1/common/src/tc.rs:280-285", "zk/sp1/common/src/reduce.rs:321-325")),
        Builder(False, ("predicates.py:146-156: 'rmsnorm-cone' ADAPTER proves the CONE of the output coordinate (F32Mul/F32Add reduction + rsqrt + scale), unit='cone' -- not this VU's own relation; no per-VU builder",),
                granularity="cone", note="the output word is committed (norm output row); its FP32 input is interior."),
        (FR_PERWORD, FR_PACKED),
    ),
    KindRow(
        "Bf16ToF32_v1|batch",
        "verity_vllm/registry/prims.py (Bf16ToF32: exact widening, `u << 16`)",
        Checker(True, G, "veritor.tc-ampere-bf16@2 / elem-bf16@1 / reduce-mufu@1", ("bf16_to_f32",),
                (f"{GS}:321,371,350", f"{GS}:763", "zk/sp1/common/src/reduce.rs:307-309")),
        Builder(False, ("no record-wired builder (bench: zk/bench/proof_elem_batches.py:110-116 eval_f2fp / elem rows)",), note=f"output FP32 word interior: {NO_STORE}"),
        (FR_PERWORD, FR_PACKED),
    ),
    KindRow(
        "F32Mul_v1|batch",
        "verity_vllm/registry/prims.py (F32Mul: IEEE RN-even f32 multiply; NaN rule = host numpy's)",
        Checker(True, G, "veritor.reduce-mufu@1", ("f32_mul",),
                (f"{GS}:355", f"{GS}:771", "zk/sp1/common/src/reduce.rs:182-207 (RN-even; SSE NaN rule: first NaN operand quieted, 0*inf -> 0xFFC00000)", REDUCE_IN64),
                domain_caveat="the registered primitive's NaN word is the evaluating host's (prims.py uses np.float32); the guest pins SSE's rule (reduce.rs:15-18). Equal on x86 hosts; a non-x86 reference host may produce a different NaN payload -> the circuit is under-specified off the finite domain (EA: the statement's NaN semantics are the guest's, not the registry's)."),
        Builder(False, ("no record-wired builder; zk/bench/reduce_relations.py:120-215 (Translated) translates whole registered composites (RMSNormTriton, RoPEHead, TokenSelect) gate-for-gate into reduce-mufu programs -- an operator-level batch builder, not a selected-VU builder",),
                note=f"operands interior: {NO_STORE}"),
        (FR_PERWORD, FR_PACKED),
    ),
    KindRow(
        "F32Mul_v1|primitive", "verity_vllm/registry/prims.py (F32Mul)",
        Checker(True, G, "veritor.reduce-mufu@1", ("f32_mul",), (f"{GS}:355,771", "zk/sp1/common/src/reduce.rs:182-207", REDUCE_IN64),
                domain_caveat="NaN rule pinned to SSE in the guest; host-dependent in the registry (see F32Mul_v1|batch)"),
        Builder(False, ("no record-wired builder (bench: zk/bench/reduce_relations.py:120-215, whole composites)",), note=NO_STORE),
        (FR_PERWORD, FR_PACKED),
    ),
    KindRow(
        "F32Add_v1|primitive", "verity_vllm/registry/prims.py (F32Add)",
        Checker(True, G, "veritor.reduce-mufu@1", ("f32_add",), (f"{GS}:354,770", "zk/sp1/common/src/reduce.rs:131-180 (RN-even, SSE NaN rule)", REDUCE_IN64),
                domain_caveat="NaN rule pinned to SSE in the guest; host-dependent in the registry"),
        Builder(False, ("no record-wired builder (bench: zk/bench/reduce_relations.py:120-215)",), note=NO_STORE),
        (FR_PERWORD, FR_PACKED), aliases=("F32Add_v1|primitive|out=0", "F32Add_v1|primitive|out=32"),
    ),
    KindRow(
        "F32Fma_v1|primitive", "verity_vllm/registry/prims.py (F32Fma = f32_fma_bits: one rounding of a*b+c, NaN -> 0x7FC00000)",
        Checker(True, G, "veritor.reduce-mufu@1", ("f32_fma",), (f"{GS}:356,773", "zk/sp1/common/src/reduce.rs:240-305 (canonical NaN 0x7FC00000, exact-zero sign rule)", REDUCE_IN64)),
        Builder(False, ("no record-wired builder (bench: zk/bench/reduce_relations.py:120-215)",), note=NO_STORE),
        (FR_PERWORD, FR_PACKED),
    ),
    KindRow(
        "F32Div_v1|primitive", "verity_vllm/registry/prims.py (F32Div: div.rn.f32)",
        Checker(True, G, "veritor.reduce-mufu@1", ("f32_div",), (f"{GS}:355,772", "zk/sp1/common/src/reduce.rs:209-238 (RN-even; SSE NaN rule)", REDUCE_IN64),
                domain_caveat="NaN rule pinned to SSE in the guest; host-dependent in the registry"),
        Builder(False, ("no record-wired builder",), note=NO_STORE),
        (FR_PERWORD, FR_PACKED),
    ),
    KindRow(
        "SiluMulBf16_v1|batch",
        "verity_vllm/registry/prims.py (SiluMulBf16: bf16_rn(f32(bf16_rn(silu_f32(f32(g)))) * f32(u)), NaN -> 0x7FC0)",
        Checker(True, G, "veritor.elem-bf16@1", ("silu_mul_bf16",),
                (f"{GS}:370-376,794", "zk/sp1/common/src/elem.rs:43-52 (SiLU as an embedded 2^16-entry table, elem_silu_table.bin), :247-260 (silu_mul_bf16)"),
                domain_caveat="the transcendental silu_f32 is a TABLE generated from the registered evaluator (zk/bench/gen_elem_silu_table.py) -- the table bytes are the checker's semantics; equality with the registry holds by construction of the table, not by code (elem.rs:26-34)."),
        Builder(False, ("no record-wired builder; zk/bench/proof_elem_batches.py:237-241 (map2_program 'silu_mul_bf16') builds format-2 statements from B1 rows (bench)",),
                note="gate/up operands are the gate_up GEMM output row (committed boundary); the product word is the down-proj input (committed boundary) -- the ONLY non-GEMM kind whose whole boundary is committed today",
                fallback="A5/6 e2e/coverage_v1/elem_builder.py (build_statement: raw format-3 statement over the record's packed rows, one silu_mul_bf16 gate per selected word; "
                         "tests/contract/a56_test_elem_builder.py: REAL B0 rows, transparent backend accepts, flipped output word rejected, coverage accounted). "
                         "Not wired into selection -> proving (A3) and no SP1 proof run in this lane."),
        (FR_PERWORD, FR_PACKED),
    ),
    KindRow(
        "Bf16AddF2fp_v1|batch",
        "verity_vllm/registry/prims.py:291-293 (fused_add_rms_norm residual add: bf16_rn(f32(a)+f32(b)), NaN -> 0x7FFF)",
        Checker(True, G, "veritor.elem-bf16@1 / veritor.reduce-mufu@1", ("bf16_add_f2fp",),
                (f"{GS}:371,348,793,775", "zk/sp1/common/src/elem.rs:236-243", "zk/sp1/common/src/reduce.rs:327-332")),
        Builder(False, ("no record-wired builder (bench: zk/bench/proof_elem_batches.py:237-241 map2_program 'bf16_add_f2fp')",),
                note="operands: previous residual row + o-proj/down-proj output rows (boundaries, committed); output residual row committed (fused norm) -- boundary fully committed for the residual add of the fused kernel",
                fallback="A5/6 e2e/coverage_v1/elem_builder.py (KINDS['Bf16AddF2fp_v1']: one bf16_add_f2fp gate per selected word; same builder path as SiluMulBf16; "
                         "UNTESTED on record rows in this lane -- no committed residual-row pair fixture reachable; the gate's Python mirror is the guest's pinned evaluator, elem.rs:361-363)."),
        (FR_PERWORD, FR_PACKED),
    ),
    KindRow(
        # [E6 NIGHT 2026-09-12] Qwen2's QKV bias add (b1.BiasAdd = aten.add bf16): NOT a kind of the Llama population; no single guest gate for it --
        # E7's `composed-body/v1` (wave 2, 155171c0; PROPOSED rule) proves it by a two-gate body of approved reduce-mufu@1 gates reading one structural literal
        "Bf16Add_v1|batch",
        "verity_vllm/registry/prims.py:283-286 (Bf16Add: bf16_rn(f32(a) + f32(b)), NaN -> 0x7FC0 (torch canonical NaN); b1.py:123-127 BiasAdd, qkv_proj + qkv_b)",
        Checker(True, G, "veritor.reduce-mufu@1", ("f2fp_bf16", "rope_out_add"),
                ("e2e/coverage_v1/elem_builders.py COMPOSED_KINDS['Bf16Add_v1|batch'] (composed-body/v1): one = f2fp_bf16(Const32[0x3f800000]); out = rope_out_add(a, b, one, one) "
                 "= bf16_rn_torch(RN32(b*1 + a*1)) -- MEASURED bit-identical to prims.Bf16Add on 500,400 pairs incl. every NaN/inf/denormal edge (E7)",
                 "zk/sp1/common/src/gateset.rs (bf16_add_f2fp only: NaN 0x7FFF -- NOT this kind: 2,811 of 500,400 measured pairs differ, every NaN result)"),
                domain_caveat="the body reads the structural literal Const32[0x3f800000]_v1 from the pinned literal tree (D36 commitments.literals); a program without that "
                              "literal refuses LITERAL_UNPINNED -- Qwen2.5-0.5B 4x8 has it (12 units)"),
        Builder(True, ("e2e/coverage_v1/elem_builders.py COMPOSED_KINDS (registered, family veritor.reduce-mufu@1; tests/contract/a3_sequence/registry_pin.py pins the kind)",)),
        (FR_PERWORD, FR_PACKED),
        note="operands = the qkv GEMM output row (committed boundary) + the bias weight row (prescribed input); output = the qkv_proj.add node row (committed boundary, "
             "qwen2 profile: rope reads q/k, attention v from it). composed-body/v1 is E7's PROPOSED rule (integrator ruling pending); before wave 2 this row was "
             "support none (E6-QWEN05-4x8-r1-5b58b141: 331,776 VUs without a checker)",
    ),
    KindRow(
        "RopeOut_v1|batch", "verity_vllm/registry/prims.py (RopeOut: bf16_rn(fma(x, c, -(y*s))), NaN -> 0x7FC0)",
        Checker(True, G, "veritor.reduce-mufu@1", ("rope_out",), (f"{GS}:364,776", "zk/sp1/common/src/reduce.rs:334-340")),
        Builder(False, ("no record-wired builder; zk/bench/reduce_relations.py:511-543 build_rope (whole RoPEHead_v1{D}, seeded or B0 rows)",),
                note="x,y from the qkv-proj output row (committed), cos/sin from the registered derived table (opening-only, reverify.py:374); the rotated word is the attention q/k input -- committed as the K cache row, interior for q"),
        (FR_PERWORD, FR_PACKED),
    ),
    KindRow(
        "RopeOutAdd_v1|batch", "verity_vllm/registry/prims.py (RopeOutAdd)",
        Checker(True, G, "veritor.reduce-mufu@1", ("rope_out_add",), (f"{GS}:365,777-779", "zk/sp1/common/src/reduce.rs:342-346")),
        Builder(False, ("no record-wired builder (bench: zk/bench/reduce_relations.py:511-543)",), note="as RopeOut"),
        (FR_PERWORD, FR_PACKED),
    ),
    KindRow(
        "GatherBf16x49152_v1|batch",
        "verity_vllm/registry/prims.py:554-564 (GatherBf16xV: table[tok][h] -> bf16 word; out-of-range token -> 0x7FFF, a documented total extension)",
        Checker(True, G, "veritor.embed-gather@1", ("i32_mul", "i32_add", "gather_word (rejects a row whose embedded index is not the recomputed flat index)"),
                (f"{GS}:386-392,798-803", "zk/sp1/common/src/embed.rs:28-50")),
        Builder(False, ("row-level bench builder zk/bench/embed_relations.py:86-130,237-458 (Embedding_v1{V,H} over the record bundle's rows) attached by e2e/cov_attach.py; predicates.py:157-161 'embedding-row' is a VERIFIER row comparison (proof='verifier', unit='row'); no per-selected-VU builder",),
                granularity="row",
                note="historical proofs of this family were under ELF 0eec2c18 (approved_guests.json 'provisional-legacy', not build-verified); 9fd00e60 resolves the same family id"),
        (FR_PERWORD, FR_PACKED),
        note="the guest's `gather_word` check (embed.rs:43-50) does not implement the registry's out-of-range extension (0x7FFF): an out-of-range token is a rejection, not the extension word -- unreachable for a committed prompt id < V.",
    ),
    KindRow(
        "Const32[0x00000000]_v1|primitive", "verity.ir/codec.py:334 (Const<w>[0x..]_v1: a 0-ary gate whose value is the literal)",
        Checker(True, G, "any family: a source gate pinned with `expected` (public input) -- the same mechanism as Input", ("in with `expected`",),
                (f"{OP}:152-156,642-646,818-822", "zk/bench/reduce_relations.py:10-12 (every Const gate of a body becomes a pinned `in` port)")),
        Builder(False, ("input_equation_subcircuit (typed_obligation_export.py:423-430) pins a 16-bit word only; no 32-bit pin builder and no selected-VU wiring",),
                note=f"a Const VU's output is interior: {NO_STORE}"),
        (FR_PERWORD, FR_PACKED), aliases=("Const32[0x00000000]_v1|primitive|out=0", "Const32[0x00000000]_v1|primitive|out=32"),
    ),
    KindRow("Const32[0x3e38aa3b]_v1|primitive", "verity.ir/codec.py:334", Checker(True, G, "pinned `in`", ("in with `expected`",), (f"{OP}:152-156,642-646,818-822",)),
            Builder(False, ("as Const32[0x00000000]",), note=NO_STORE), (FR_PERWORD, FR_PACKED)),
    KindRow("Const32[0x3727c5ac]_v1|primitive", "verity.ir/codec.py:334", Checker(True, G, "pinned `in`", ("in with `expected`",), (f"{OP}:152-156,642-646,818-822",)),
            Builder(False, ("as Const32[0x00000000]",), note=NO_STORE), (FR_PERWORD, FR_PACKED)),
    KindRow("Const32[0x44100000]_v1|primitive", "verity.ir/codec.py:334", Checker(True, G, "pinned `in`", ("in with `expected`",), (f"{OP}:152-156,642-646,818-822",)),
            Builder(False, ("as Const32[0x00000000]",), note=NO_STORE), (FR_PERWORD, FR_PACKED)),
    KindRow("Const32[0x3f800000]_v1|primitive", "verity.ir/codec.py:334", Checker(True, G, "pinned `in`", ("in with `expected`",), (f"{OP}:152-156,642-646,818-822",)),
            Builder(False, ("as Const32[0x00000000]",), note=NO_STORE), (FR_PERWORD, FR_PACKED)),
    KindRow("Const32[0x00000001]_v1|primitive", "verity.ir/codec.py:334 (the ArgmaxStep increment, b1.py:466)", Checker(True, G, "pinned `in`", ("in with `expected`",), (f"{OP}:152-156,642-646,818-822",)),
            Builder(False, ("as Const32[0x00000000]",), note=NO_STORE), (FR_PERWORD, FR_PACKED)),
    KindRow("Const16[0x0000]_v1|primitive", "verity.ir/codec.py:334", Checker(True, G, "pinned `in`", ("in with `expected`",), (f"{OP}:152-156,642-646,818-822",)),
            Builder(False, ("input_equation_subcircuit (typed_obligation_export.py:423-430) pins exactly a 16-bit word -- the builder primitive exists, the selected-VU wiring does not",), note=NO_STORE),
            (FR_PERWORD, FR_PACKED), aliases=("Const16[0x0000]_v1|primitive|out=0", "Const16[0x0000]_v1|primitive|out=16")),
    KindRow(
        "DivFullRcp_v1|primitive", "verity_vllm/registry/prims.py (DivFullRcp: MUFU.RCP-based full-range reciprocal step; table-measured)",
        Checker(True, G, "veritor.reduce-mufu@1 (table gate: operand + one committed table row `(index << 32) | word`)", ("div_full_rcp",),
                (f"{GS}:351,642-648,790", "zk/sp1/common/src/reduce.rs:422-450 (recomputes the MUFU index from the operand; a row with another index rejects)")),
        Builder(False, ("no record-wired builder; zk/bench/reduce_relations.py:13-18 (table ports) over a committed table tree (reduce_tables.py) that today's record does not commit",),
                note="requires a committed MUFU table tree in the record (not in commitments.json today)"),
        (FR_PERWORD, FR_PACKED),
    ),
    KindRow("DivFullScaleA_v1|primitive", "verity_vllm/registry/prims.py (DivFullScaleA)",
            Checker(True, G, "veritor.reduce-mufu@1", ("div_full_scale_a",), (f"{GS}:352,791", "zk/sp1/common/src/reduce.rs (div_full_scale_a)")),
            Builder(False, ("no record-wired builder (bench: reduce_relations.py)",), note=NO_STORE), (FR_PERWORD, FR_PACKED)),
    KindRow("RsqrtApprox_v1|primitive", "verity_vllm/registry/prims.py (RsqrtApprox: MUFU.RSQ, table-measured)",
            Checker(True, G, "veritor.reduce-mufu@1 (table gate)", ("rsqrt_approx",), (f"{GS}:366,792", "zk/sp1/common/src/reduce.rs (rsqrt_approx: index recomputed, row rejected otherwise)")),
            Builder(False, ("no record-wired builder; needs the committed table tree",), note=NO_STORE), (FR_PERWORD, FR_PACKED)),
    KindRow("MufuSqrtFtz_v1|primitive", "verity_vllm/registry/prims.py (MufuSqrtFtz: MUFU.SQRT ftz, table-measured)",
            Checker(True, G, "veritor.reduce-mufu@1 (table gate)", ("mufu_sqrt_tab",), (f"{GS}:363,789", "zk/sp1/common/src/reduce.rs:452-475")),
            Builder(False, ("no record-wired builder; needs the committed table tree",), note=NO_STORE), (FR_PERWORD, FR_PACKED)),
    KindRow(
        "ArgmaxStep_v1|scan",
        "verity_vllm/registry/b1.py:458-468 (Bf16GtStrict, SelectBf16, SelectI32, I32Add, Const32[1]): one greedy-argmax step",
        Checker(True, G, "veritor.reduce-mufu@1", ("bf16_gt", "select_bf16", "select_i32", "i32_add", "pinned in (Const)"),
                (f"{GS}:349,367-368,357,780-786", "zk/sp1/common/src/reduce.rs:348-360 (bf16_gt: NaN compares false), :384-404")),
        Builder(False, ("zk/bench/reduce_relations.py:545-583 build_token_select translates the WHOLE TokenSelect_v1{V=49152} scan (all steps) from the B0 store -- operator-level; no per-step builder",),
                granularity="whole scan", note=f"the carry tuple (best, best_index, i) is interior: {NO_STORE}"),
        (FR_PERWORD, FR_PACKED), legacy=True, aliases=("ArgmaxStep_v1|scan|out=32", "ArgmaxStep_v1|scan|out=80"),
        note="legacy kind: D2's P1v2 replaces the per-step scan VU by TokenSelect (one VU per position); see TokenSelect_v1{V=49152}|call.",
    ),
    KindRow(
        "TokenSelect_v1{V=49152}|call",
        "verity_vllm/registry/b1.py:471-476 (argmax over the bf16 logits row, lowest index on ties; conformance='model-untested')",
        Checker(True, G, "veritor.reduce-mufu@1 (the whole scan as one relation of ~5V gates)", ("bf16_gt", "select_bf16", "select_i32", "i32_add", "pinned in"),
                (f"{GS}:347-369", "zk/bench/reduce_relations.py:545-583 (build_token_select: validated against the IR oracle on B0 store rows)")),
        Builder(False, ("zk/bench/reduce_relations.py:545-583 builds it from the B0 commit store (bench batches), not from a selected VU of a record; predicates.py has no entry",),
                note="logits row (lm_head GEMM output) and the served token id are committed boundaries -- the whole boundary is committed; only the selected-VU wiring is missing. Per D2's P1v2 this is the sampling-step VU (one per output position); not in D4's expected_work.json (which counts the legacy ArgmaxStep scan)."),
        (FR_PERWORD, FR_PACKED), aliases=("TokenSelect_v1|whole-scan",),
        note="count per RU under P1v2 = output positions (1,600 at 6500/1599; 16 at 32x15) -- taken from the legacy Const32[0x00000001] count, which is one per TokenSelect (b1.py:473). "
             "OUTPUT LEAF (A2 M2): the B1 mapping / W3 harness point the relation at the INTERIOR step-3 copy `.../l=sampler/op=sampler.select/port=0` "
             "(verity_vllm_sampler/registry.py:17-22, profiles/b1-hopper/capture_mapping.py:43); under one-value-per-gate the checker's outgoing value must be the step-1 "
             "boundary leaf `boundary/<rid>` `.../output_token/t=<t>` (or A2's output_copy_checks agreement runs before Q and the interior leaf is deleted). "
             "NaN/TIE RULE: from the served kernel (select.py:13: fp32 argmax, first index wins ties, per 1024-block then across blocks), NOT from relations.py:277-278/:377 "
             "or sampler registry.py:99-100 (NaN bans) -- kernel NaN behaviour unmeasured (EA); the guest's bf16_gt (NaN compares false) implements b1.py:461 'NaN never wins'.",
    ),
    # --- attention interior (FTZ family): no approved checker -------------------------------------------------------
    *[
        KindRow(
            kind, circuit,
            Checker(True, ("33cba058",), "veritor.fa2-attn (unapproved) -- and ONLY as part of a whole-head fold (`in` is 16 bits, fa2.rs:89-93)", ops,
                    ("zk/sp1-attn/common/src/fa2.rs:24-36,229-241", "zk/sp1-attn/common/src/ftz.rs", "zk/sp1-attn/common/src/mufu.rs",
                     f"approved guest: no FTZ gate in any family ({GS}:300-392)"),
                    granularity="whole attention head",
                    domain_caveat="the head fold is a different unit from this VU: even with 33cba058 approved, this per-primitive VU has no per-VU checker unless S55 redefines the attention VU as the head (predicates.py:165)"),
            Builder(False, ("zk/bench/attn_gates.py:184-307 (attn_block_subcircuit / epilogue) + zk/bench/pb_attn_batches.py build head-level statements for the fa2 guest from synthetic/B0 rows; no record-wired, selected-VU builder; predicates.py:165 refuses Attention (S55)",),
                    granularity="whole attention head", note=NO_STORE),
            (FR_PERWORD,),
        )
        for kind, circuit, ops in (
            ("F32MulFtz_v1|primitive", "verity_vllm/registry/prims.py (F32MulFtz: mul.ftz.f32)", ("f32_mul_ftz",)),
            ("F32MulFtz_v1|batch", "verity_vllm/registry/prims.py (F32MulFtz)", ("f32_mul_ftz",)),
            ("MufuEx2Ftz_v1|primitive", "verity_vllm/registry/prims.py (MufuEx2Ftz: MUFU.EX2 ftz, table-measured)", ("mufu_ex2_ftz",)),
            ("F32FmaSubFtz_v1|primitive", "verity_vllm/registry/prims.py (F32FmaSubFtz)", ("f32_fma_sub_ftz",)),
            ("F32Max_v1|primitive", "verity_vllm/registry/prims.py (F32Max: fmax, NaN rule)", ("f32_max",)),
            ("F32AddFtz_v1|primitive", "verity_vllm/registry/prims.py (F32AddFtz)", ("f32_add_ftz",)),
            ("F32FmaFtz_v1|primitive", "verity_vllm/registry/prims.py (F32FmaFtz)", ("f32_fma_ftz",)),
            ("F32SubFtz_v1|primitive", "verity_vllm/registry/prims.py (F32SubFtz)", ("f32_sub_ftz",)),
            ("GuardNegInfZero_v1|primitive", "verity_vllm/registry/prims.py (GuardNegInfZero)", ("guard_neginf_zero",)),
            ("Fa2InvSum_v1|primitive", "verity_vllm/registry/prims.py (Fa2InvSum: MUFU.RCP-based 1/l, table-measured)", ("fa2_inv_sum",)),
        )
    ],
)

FTZ_KINDS = tuple(r.kind for r in ROWS if "33cba058" in r.checker.guests)
ROW_BY_KIND = {r.kind: r for r in ROWS}
for _r in ROWS:
    for _a in _r.aliases:
        ROW_BY_KIND[_a] = _r


# --- counts -------------------------------------------------------------------------------------------------------------


def counts_from_expected_work(doc: dict[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
    """Per-kind VU counts per RU at P = 8,099 (D4's ``expected_work.json``, ``population.kinds_table``)."""

    rows = doc["population"]["kinds_table"]
    counts = {r["kind"]: int(r["count_per_RU_8099"]) for r in rows}
    meta = {
        "source": "lane/record-6k:out/record-6k/expected_work.json (D4 RECORD-6K; MODELED by census_v2.small_scaling.terms(6500, 1599))",
        "N_R": int(doc["population"]["N_R"]),
        "N_V_per_RU": int(doc["population"]["N_V_per_RU"]),
        "legacy_classes": doc["population"]["by_class_per_RU"],
        "kinds": len(rows),
    }
    assert sum(counts.values()) == meta["N_V_per_RU"], (sum(counts.values()), meta["N_V_per_RU"])
    return counts, meta


def counts_from_census(doc: dict[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
    """Per-kind VU counts per request (= per RU, one RU per request) at 32x15 (D2's ``census_small.json``)."""

    counts: dict[str, int] = {}
    for f in doc["families"]:
        name = f["name"]
        if name.startswith("Input["):
            base = "Input|batch"
        elif name in ROW_BY_KIND:
            base = ROW_BY_KIND[name].kind
        else:
            # the census splits a kind by its measured output width (`|out=<bits>`); the kind is the prefix
            stem = name.split("|out=")[0]
            if stem not in ROW_BY_KIND:
                raise KeyError(f"census family {name!r} is not in the kind table")
            base = ROW_BY_KIND[stem].kind
        counts[base] = counts.get(base, 0) + int(f["count_per_request"])
    meta = {
        "source": "lane/partition:data/census/census_small.json (D2 census_v2, SMALL B0 Serve_v1{LP=32,STEPS=15}, partition 56ebcdf5...)",
        "N_R": 4,
        "N_V_per_RU": int(doc["N_V"]["N_V_population"]),
        "kinds": len(doc["families"]),
    }
    assert sum(counts.values()) == meta["N_V_per_RU"], (sum(counts.values()), meta["N_V_per_RU"])
    return counts, meta


# --- exposure -----------------------------------------------------------------------------------------------------------

S_VALUES = {"2^-14": Fraction(1, 2**14), "2^-19": Fraction(1, 2**19), "2^-30": Fraction(1, 2**30)}


def exposure(n: int, s: Fraction) -> dict[str, str]:
    """``1 - (1 - s)^n``: probability of selecting at least one unsupported check (completion risk)."""

    if n == 0:
        return {"probability": "0", "complement_log10": "0", "expected_unsupported_selected": "0"}
    sd = Decimal(s.numerator) / Decimal(s.denominator)
    ln_c = Decimal(n) * (Decimal(1) - sd).ln()
    complement = ln_c.exp()
    p = Decimal(1) - complement
    expected = s * n
    return {
        "probability": format(p, ".40g"),
        "complement_log10": format(ln_c / Decimal(10).ln(), ".12g"),
        "expected_unsupported_selected": f"{expected.numerator}/{expected.denominator}" if expected.denominator != 1 else str(expected.numerator),
        "expected_unsupported_selected_decimal": format(Decimal(expected.numerator) / Decimal(expected.denominator), ".12g"),
    }


def unsupported_sets(counts: dict[str, int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scenario in SCENARIOS:
        unsupported = {k: c for k, c in counts.items() if not ROW_BY_KIND[k].supported(scenario)}
        checker_missing = {k: c for k, c in counts.items() if not ROW_BY_KIND[k].checker_in_scenario(scenario)}
        checker_missing_s55 = {k: c for k, c in counts.items() if not ROW_BY_KIND[k].checker_in_scenario(scenario, allow_larger_unit=True)}
        with_fb = {k: c for k, c in counts.items() if not ROW_BY_KIND[k].supported_with_fallbacks(scenario)}
        n_r = sum(unsupported.values())
        n_cm = sum(checker_missing.values())
        n_s55 = sum(checker_missing_s55.values())
        n_fb = sum(with_fb.values())
        out[scenario] = {
            "unsupported_kinds": sorted(unsupported),
            "n_R": n_r,
            "N_4RU": 4 * n_r,
            "unsupported_kinds_with_a56_fallbacks": sorted(with_fb),
            "n_R_with_a56_fallbacks": n_fb,
            "N_4RU_with_a56_fallbacks": 4 * n_fb,
            "exposure_per_RU_with_a56_fallbacks": {name: exposure(n_fb, s) for name, s in S_VALUES.items()},
            "exposure_4RU_with_a56_fallbacks": {name: exposure(4 * n_fb, s) for name, s in S_VALUES.items()},
            "checker_missing_kinds": sorted(checker_missing),
            "n_R_checker_missing_only": n_cm,
            "N_4RU_checker_missing_only": 4 * n_cm,
            "checker_missing_kinds_if_larger_units_allowed": sorted(checker_missing_s55),
            "n_R_checker_missing_if_larger_units_allowed": n_s55,
            "exposure_per_RU": {name: exposure(n_r, s) for name, s in S_VALUES.items()},
            "exposure_4RU": {name: exposure(4 * n_r, s) for name, s in S_VALUES.items()},
            "exposure_per_RU_if_every_builder_existed": {name: exposure(n_cm, s) for name, s in S_VALUES.items()},
            "exposure_4RU_if_every_builder_existed": {name: exposure(4 * n_cm, s) for name, s in S_VALUES.items()},
        }
    return out


def build(expected_work: dict[str, Any] | None, census: dict[str, Any] | None) -> dict[str, Any]:
    kinds = []
    for r in ROWS:
        d = asdict(r)
        d["status"] = {sc: r.status(sc) for sc in SCENARIOS}
        kinds.append(d)
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "based_on": BASED_ON,
        "label": "probability of selecting an unsupported check -- completion risk (never a family-level exemption)",
        "status_vocabulary": ["supported (approved)", "supported only under an unapproved or PROPOSED guest", "builder missing", "checker missing", "no path"],
        "scenarios": {
            APPROVED_ONLY: "approved set only: 9fd00e60 (approved_guests.json 'reproduced'); the other listed ELFs are older builds of the same zk/sp1 lineage kept for existing records",
            PLUS_RANGE: "+ range guest 2389b9a8 promoted (identical gate sets; adds frame-v3/range openings only -> the unsupported set does not change)",
            PLUS_FA2: "+ fa2-attn guest 33cba058 approved (adds the FTZ attention family, whole-head granularity only; per-VU attention kinds remain without a per-VU checker until S55 redefines the attention VU)",
        },
        "guests": GUESTS,
        "kinds": kinds,
        "configurations": {},
    }
    if expected_work is not None:
        counts, meta = counts_from_expected_work(expected_work)
        doc["configurations"]["6500x1599_P8099"] = {**meta, "counts_per_RU": counts, "by_scenario": unsupported_sets(counts)}
    if census is not None:
        counts, meta = counts_from_census(census)
        doc["configurations"]["32x15"] = {**meta, "counts_per_RU": counts, "by_scenario": unsupported_sets(counts)}
    return doc


# --- [E6 2026-09-11] family inventory: support owner / producer / position for every family of a population ---------------
#
# Brief EOD-2026-09-11 interface 1: family inventory = ``population_v3.expected_population`` kind table (verifier-derived counts, no
# sampled input gates) x this table, with, per family: class (VU | prescribed input), support (approved checker | E4 | E5 | none, the owner
# named), producer (boundary | interior-interpreter | none) and position (boundary tree | interior tree | refused).  The support reading:
#   * ``approved checker``: the approved guest evaluates the kind's relation at VU granularity (this table's rows); the statement builder is
#     E7's (``coverage_v1.builders.BUILDERS``): ``provable`` when a registered builder derives the expectation today, else ``builder missing``;
#   * ``E4``: no approved checker at VU granularity and the kind is FTZ/scalar attention interior (the fa2 guest is the only checker) --
#     E4's ck_ftz + guest program;
#   * ``E5``: token selection (ArgmaxStep -> TokenSelect: E5's ck_select) and every Const literal (E5 decides compile-vs-retain; retained ->
#     covered by E5's pinned-`in` builder, compiled -> leaves the population);
#   * ``none``: a VU kind none of the above claims -- reported by name (the brief: every family must have an owner).
# The producer reading is per kind (E3 resolves each gate's position; a mixed kind -- some instances at an operator boundary, the rest
# interior -- is listed as interior-interpreter with the boundary share noted: whatever is not at a boundary must come from E1's interpreter).

SUPPORTS = ("approved checker", "E4", "E5", "none")
PRODUCERS = ("boundary", "interior-interpreter", "none")
POSITIONS = ("boundary tree", "interior tree", "refused", "program literal")
CLASSES = ("VU", "prescribed input", "prescribed literal")

# E5's Const decision (out/eod/e5/const32-consequence.md, 2026-09-11 20:20Z): every literal kind of the TINY partition COMPILES into its
# consumer's prescribed function -- the consumer's checker pins the value as a public input (`in ... expected`).  kind -> (support, owner,
# consumers) of the compiled literal = the consuming checker's.
LITERAL_CONSUMERS: dict[str, tuple[str, str, str]] = {
    "Const32[0x00000000]_v1|primitive": ("approved checker", "E7 (tc_dot16_0 zero accumulator; F32Add reduce init) + E4 (F32AddFtz lane-sum init) + E5 (TokenSelect best_i init)",
                                         "AmpereBF16TcDot16_v1|batch 210,600 (incl. 1/head dead z) / F32AddFtz_v1|primitive 3,240 / F32Add_v1|primitive 684 / TokenSelect_v1|whole-scan 9"),
    "Const32[0x3e38aa3b]_v1|primitive": ("E4", "E4 (F32MulFtz / F32FmaSubFtz pin D^-0.5*log2e)", "F32MulFtz_v1|primitive, F32FmaSubFtz_v1|primitive (one per AttnBlock)"),
    "Const32[0x44100000]_v1|primitive": ("approved checker", "E7 (reduce: F32Div / DivFullRcp pin N = 576.0)", "F32Div_v1|primitive 717 / DivFullRcp_v1|primitive 12"),
    "Const32[0x3727c5ac]_v1|primitive": ("approved checker", "E7 (reduce: F32Add / F32Fma pin eps = 1e-5)", "F32Add_v1|primitive 717 / F32Fma_v1|primitive 12"),
    "Const32[0x3f800000]_v1|primitive": ("approved checker", "E7 (reduce: DivFullScaleA pins 1.0)", "DivFullScaleA_v1|primitive 12"),
    "Const32[0x00000001]_v1|primitive": ("E5", "E5 (TokenSelect i init = 1; ck_select expectation derives (0, 1) from the registry)", "TokenSelect_v1|whole-scan 9"),
    "Const16[0x0000]_v1|primitive": ("approved checker", "E7 (tc_dot16 / DotBf16 padding operands pinned `in` 0x0000)", "AmpereBF16TcDot16_v1|batch 3,240 / DotBf16_v1{K=64}|call 3,240"),
    # [E6 NIGHT 2026-09-12] the other checkpoints' literals: the rmsnorm divisor N = H and eps change by value, the consumers are the same functions
    "Const32[0x44700000]_v1|primitive": ("approved checker", "E7 (reduce: F32Div / DivFullRcp pin N = 960.0 -- SmolLM2-360M's H)", "F32Div_v1|primitive / DivFullRcp_v1|primitive (SMOL360)"),
    "Const32[0x44600000]_v1|primitive": ("approved checker", "E7 (reduce: F32Div / DivFullRcp pin N = 896.0 -- Qwen2.5-0.5B's H)", "F32Div_v1|primitive / DivFullRcp_v1|primitive (QWEN05)"),
    "Const32[0x358637bd]_v1|primitive": ("approved checker", "E7 (reduce: F32Add / F32Fma pin eps = 1e-6 -- Qwen2's rms_norm_eps)", "F32Add_v1|primitive / F32Fma_v1|primitive (QWEN05)"),
}
INVENTORY_SCHEMA = "veritor/e6/family-inventory/v1"
COVERAGE_TABLE_SCHEMA = "veritor/e6/coverage-table/v1"

#: kinds whose output words are committed operator-boundary rows today (e2e/predicates.py boundaries; A2's store)
BOUNDARY_KINDS: dict[str, str] = {
    "GemmCoordinate_v1{K=576}|batch": "GEMM output rows (qkv/o/gate_up/down/lm_head projections) are committed operator boundaries",
    "GemmCoordinate_v1{K=1536}|batch": "down-proj GEMM output row: committed operator boundary",
    # [E7 SCALE D46] Qwen2.5-0.5B's shapes (E6 kind-diff): the same boundaries at K = 896 (hidden) / 4864 (intermediate)
    "GemmCoordinate_v1{K=896}|batch": "Qwen2.5-0.5B: GEMM output rows (q/k/v/o, gate_up projections, K = hidden 896) are committed operator boundaries",
    "GemmCoordinate_v1{K=4864}|batch": "Qwen2.5-0.5B: down-proj GEMM output row (K = intermediate 4864): committed operator boundary",
    "SiluMulBf16_v1|batch": "the product word is the down-proj input row (committed boundary); operands are the gate_up output row",
    "Bf16AddF2fp_v1|batch": "the residual row of the fused add+norm is a committed boundary",
    "F2fpBf16_v1|batch": "the RMSNorm / attention-output cast lands in a committed output row",
    "GatherBf16x49152_v1|batch": "the embedding output row is a committed boundary (predicates.py 'embedding-row')",
    "TokenSelect_v1|whole-scan": "the selected token id is the request's output token leaf (boundary/<rid>/.../output_token/t=<t>)",
    "TokenSelect_v1{V=49152}|call": "as TokenSelect_v1|whole-scan",
    # E1 MEASURED 2026-09-11 (INTEGRATOR D9): 4,185,216 boundary rows per RU at 4x8 = the seven kinds above (retained form, 3,908,736) + RopeOut +
    # RopeOutAdd (2 x 138,240): every rotated q and k word has a boundary position (the committed q / K-cache rows); nothing of Rope is interior
    "RopeOut_v1|batch": "rotated q / k words are committed rows (E1 boundary oracle, MEASURED 2026-09-11: all 138,240 have a position)",
    "RopeOutAdd_v1|batch": "as RopeOut_v1|batch (E1 MEASURED: all 138,240 have a boundary position)",
}
#: kinds with some instances at a boundary and the rest interior (listed as interior-interpreter; the boundary share noted).  Empty since
#: E1's measurement moved RopeOut / RopeOutAdd to the boundary set; kept as the hook for a future mixed kind.
MIXED_KINDS: dict[str, str] = {}
_CONST_RE = __import__("re").compile(r"^Const\d+\[0x[0-9a-fA-F]+\]_v\d+\|primitive$")
E5_SELECT_KINDS = ("ArgmaxStep_v1|scan", "TokenSelect_v1|whole-scan", "TokenSelect_v1{V=49152}|call")


_GEMM_COORD_RE = re.compile(r"^GemmCoordinate_v1\{K=(?P<k>\d+)\}\|batch$")
_GATHER_RE = re.compile(r"^GatherBf16x(?P<v>\d+)_v1\|batch$")
_PARAMETRIC_ROWS: dict[str, KindRow] = {}


def row_of(kind: str) -> KindRow | None:
    """The coverage row of a kind.  [E6 NIGHT 2026-09-12] shape-parametric kinds (another checkpoint changes K of the GEMM coordinates and V of
    the embedding gather, never the function) resolve to the function's row with the parameter noted: GemmCoordinate<K> for any K multiple of 16
    (the approved checker is K-generic: programs.py:92, no K limit in relations.rs), GatherBf16x<V> for any V (the Gather family has no builder at
    any V -- G1 (b))."""

    r = ROW_BY_KIND.get(kind)
    if r is None:
        stem = kind.split("|out=")[0]
        r = ROW_BY_KIND.get(stem)
    if r is not None:
        return r
    if kind in _PARAMETRIC_ROWS:
        return _PARAMETRIC_ROWS[kind]
    from dataclasses import replace

    m = _GEMM_COORD_RE.match(kind)
    if m and int(m.group("k")) % 16 == 0:
        base = ROW_BY_KIND["GemmCoordinate_v1{K=576}|batch"]
        r = replace(base, kind=kind, note=f"K = {m.group('k')} (shape-parametric row of GemmCoordinate<K>; approved checker K-generic by code reading: programs.py:92 any multiple of 16, "
                                          f"relations.rs no K limit; MEASURED with proofs at K = 576 and 1536 only). {base.note}")
        _PARAMETRIC_ROWS[kind] = r
        return r
    m = _GATHER_RE.match(kind)
    if m:
        base = ROW_BY_KIND["GatherBf16x49152_v1|batch"]
        r = replace(base, kind=kind, note=f"V = {m.group('v')} (shape-parametric row of GatherBf16x<V>: the family veritor.embed-gather@1 has no record-wired builder at any V -- G1 (b)). {base.note}")
        _PARAMETRIC_ROWS[kind] = r
        return r
    return None


def registered_expectation_families() -> dict[str, str]:
    """kind -> interface family for every kind whose registered builder derives the verifier's expectation today (E7's registry)."""

    try:
        from e2e.coverage_v1 import builders as _b

        return {k: str(v.family) for k, v in _b.expectation_builders().items()}
    except Exception:  # pragma: no cover - registry unavailable
        return {}


def interior_refuser_kinds() -> set[str]:
    try:
        from e2e.coverage_v1.interior_layout import INTERIOR_KINDS

        return set(INTERIOR_KINDS)
    except Exception:  # pragma: no cover
        return set()


def approved_families() -> dict[str, str]:
    """family -> ELF sha256 (first 8) of an APPROVED guest (``approval == "reproduced"`` in ``e2e/approved_guests.json``) that carries it.  The
    verifier-side policy: a registered builder whose family is not here is NOT an approved checker, whatever its author says."""

    try:
        from e2e.approved_guests import load_policy

        out: dict[str, str] = {}
        for g in load_policy().get("guests", []):
            if g.get("approval") != "reproduced":
                continue
            fams = g.get("families") or {}
            for fam in (fams if isinstance(fams, dict) else fams or []):
                out.setdefault(str(fam), str(g["elf_sha256"])[:8])
        return out
    except Exception:  # pragma: no cover - policy unavailable
        return {}


#: unapproved families under construction today and who owns them (EOD 2026-09-11)
UNAPPROVED_FAMILY_OWNERS: dict[str, tuple[str, str]] = {
    "veritor.ftz-f32@1": ("E4", "E4 (veritor.ftz-f32@1: consolidated FTZ/scalar guest zk/sp1/programs/ftz, ELF e379de41..., 131+34 vectors, 2 CPU core proofs; "
                                "NOT yet approved -- E7 approval pending: pod reproduction + 12-negative suite over committed f32 leaves)"),
    "veritor.select-argmax@1": ("E5", "E5 (veritor.select-argmax@1: dedicated select guest zk/sp1/programs/select, ELF 8f30cf2c..., 61/61 vectors on the ELF, 2 REAL core "
                                      "proofs incl. the D7 request t=3; UNAPPROVED -- approval is E7's; 16 pod proofs pending)"),
}


#: [E6 NIGHT 2026-09-12] kinds another architecture exposes that have NO checker and NO builder today, with the owner named (the family is not
#: covered -- `support = none` -- but it is not unowned: the builder it needs is E7's)
NEW_KIND_OWNERS: dict[str, tuple[str, str]] = {
    # fallback owner if E7's composed-body/v1 registration (elem_builders.COMPOSED_KINDS) is absent from a checkout: the kind is then E7's, not unowned
    "Bf16Add_v1|batch": ("E7 (composed-body/v1 builder: elem_builders.COMPOSED_KINDS['Bf16Add_v1|batch'], wave 2 155171c0; Qwen2 QKV bias add, 331,776 VUs at 4x8)",
                         "no single guest gate (approved set has bf16_add_f2fp, NaN 0x7FFF, only); provable only through E7's composed-body/v1 registration"),
}


def support_of(kind: str, *, registered: dict[str, str], refusers: set[str], scenario: str = APPROVED_ONLY, approved: dict[str, str] | None = None) -> dict[str, Any]:
    """``{"support", "owner", "status", "checker", "builder", "note", "family", "guest"}`` of a VU kind.  Support is decided by the VERIFIER'S policy:
    a registered builder counts as ``approved checker`` only when its family is carried by an approved (reproduced) guest; a builder of an
    unapproved family is that family's owner (E4 / E5) and says so; a kind with neither is ``none`` -- named."""

    approved = approved_families() if approved is None else approved
    row = row_of(kind)
    if _CONST_RE.match(kind):
        # only reached when the population RETAINS literals (literals=retained): the pre-decision identity, kept for re-deriving attempt 6
        return {"support": "E5", "owner": "E5 (select/const)",
                "status": "RETAINED form (pre-decision identity, attempt 6); E5 ruled COMPILE for every literal kind (out/eod/e5/const32-consequence.md 20:20Z)",
                "checker": "approved guest: source gate pinned with `expected` (public input), same mechanism as Input" if row else None,
                "builder": "none registered", "note": "handoff fixed decision applied with E5; identities/counts updated once (population_v3.expected_population literals=compiled)"}
    if kind in E5_SELECT_KINDS:
        whole = kind.endswith("|whole-scan")
        fam = registered.get(kind, "veritor.select-argmax@1")
        sup, owner = UNAPPROVED_FAMILY_OWNERS["veritor.select-argmax@1"]
        return {"support": sup, "owner": owner, "family": fam, "guest": approved.get(fam),
                "status": (("builder registered; " if kind in registered else "builder NOT yet registered in coverage_v1.builders (E7 to fold select_builders in); ")
                           + "checker ck_select + guest veritor.select-argmax@1 (ELF 8f30cf2c...) UNAPPROVED -- 2 real proofs, 16 pod rows pending; approval E7's"
                           if whole else "LEGACY form (argmax-scan, attempt 6): replaced by TokenSelect_v1|whole-scan under E5's ruling"),
                "checker": (row.checker.gate_set if row else None), "builder": f"coverage_v1.builders.BUILDERS[{kind!r}]" if kind in registered else "none registered",
                "note": "E5 FINAL 21:05Z: one VU per forward pass; operands = the committed logits row (RU tree, boundary) + program literals 0/1; output = boundary output_token leaf; "
                        "exact semantics: strict IEEE > on bf16 widened, lowest index wins ties, NaN never wins; approved path today = 9fd00e60 reduce-mufu@1 gate-for-gate (205 s/VU MODELED)"}
    if row is None and kind not in registered:
        return {"support": "none", "owner": "none", "family": None, "guest": None, "status": "kind not in the coverage table and no registered builder",
                "checker": None, "builder": None, "note": "REPORT: no row, no owner"}
    if kind in registered:
        fam = registered[kind]
        if fam in approved:
            return {"support": "approved checker", "owner": f"E7 (registered expectation builder, family {fam}, approved guest {approved[fam]})", "family": fam, "guest": approved[fam],
                    "status": "provable", "checker": (row.checker.gate_set if row else fam), "builder": f"coverage_v1.builders.BUILDERS[{kind!r}] ({fam})",
                    "note": (row.builder.fallback or row.note) if row else "builder registered by E7; family carried by an approved guest"}
        if fam in UNAPPROVED_FAMILY_OWNERS:
            sup, owner = UNAPPROVED_FAMILY_OWNERS[fam]
            return {"support": sup, "owner": owner, "family": fam, "guest": None, "status": f"builder registered ({fam}); family NOT approved -- not provable under the verifier's policy today",
                    "checker": (row.checker.gate_set if row else fam), "builder": f"coverage_v1.builders.BUILDERS[{kind!r}] ({fam})",
                    "note": (row.checker.domain_caveat if row and row.checker.domain_caveat else "") or "proves the moment E7 adds the family's guest to approved_guests.json"}
        return {"support": "none", "owner": "none", "family": fam, "guest": None, "status": f"builder registered under family {fam!r}: neither approved nor owned by name",
                "checker": None, "builder": f"coverage_v1.builders.BUILDERS[{kind!r}]", "note": "REPORT: unowned family"}
    st = row.status(scenario)
    if st == "supported (approved)":
        return {"support": "approved checker", "owner": "E7 (builder: legacy exporter path typed_obligation_export / predicates.py)", "family": row.checker.gate_set, "guest": "9fd00e60",
                "status": "provable", "checker": row.checker.gate_set, "builder": row.builder.evidence[0], "note": row.note}
    if row.checker_in_scenario(scenario):
        how = "interior refuser today (coverage_v1.interior_layout.InteriorRefuser): builds once the interior tree exists (E1/E2)" if kind in refusers else "no registered builder"
        return {"support": "approved checker", "owner": "E7 (approved checker gate set; builder missing)", "family": row.checker.gate_set, "guest": "9fd00e60",
                "status": f"builder missing -- {how}", "checker": row.checker.gate_set, "builder": row.builder.evidence[0], "note": row.builder.note or row.note}
    if kind in FTZ_KINDS or "Ftz" in kind or kind.startswith(("F32Max_", "GuardNegInfZero_", "Fa2InvSum_")):
        sup, owner = UNAPPROVED_FAMILY_OWNERS["veritor.ftz-f32@1"]
        return {"support": sup, "owner": owner, "family": "veritor.ftz-f32@1", "guest": None, "status": f"{st}: no approved checker at VU granularity; builder not registered",
                "checker": row.checker.gate_set, "builder": row.builder.evidence[0], "note": row.checker.domain_caveat}
    if kind in NEW_KIND_OWNERS:
        owner, what = NEW_KIND_OWNERS[kind]
        return {"support": "none", "owner": owner, "family": None, "guest": None, "status": f"{st}: {what}", "checker": row.checker.gate_set, "builder": row.builder.evidence[0],
                "note": row.note}
    return {"support": "none", "owner": "none", "family": None, "guest": None, "status": st, "checker": row.checker.gate_set, "builder": row.builder.evidence[0],
            "note": "REPORT: no owner claims this kind"}


def exposure_at(n: int, s_log2: int) -> float:
    """``1 - (1 - s)^n`` at ``s = 2^-s_log2`` as a float: the probability that at least one of a family's ``n`` VUs is selected -- what an
    unsupported (or refused) family costs the record when its VUs are drawn (an uncovered selected VU is a named refusal, never a pass).
    (``exposure`` above is the Decimal form over ``S_VALUES`` used by the scenario tables.)"""

    import math

    s = 2.0 ** -s_log2
    return 1.0 - math.exp(n * math.log1p(-s)) if n > 0 else 0.0


def producer_of(kind: str, *, ru_output_kinds: set[str] | None = None) -> tuple[str, str]:
    """``(producer, note)``: for Serve_v1 by the boundary/mixed tables; for a generic program by whether the kind's gates are RU outputs
    (``ru_output_kinds`` from ``population_v3.expected_population_generic``)."""

    if ru_output_kinds is not None:
        if kind in ru_output_kinds:
            return "boundary", "the kind's gates are the RU's declared output tuple (committed at step 1 by the program's record)"
        return "interior-interpreter", "not an RU output: produced by the program's reference interpreter into the interior tree"
    if kind in BOUNDARY_KINDS:
        return "boundary", BOUNDARY_KINDS[kind]
    if _GEMM_COORD_RE.match(kind):
        return "boundary", "GEMM output rows (qkv/o/gate_up/down/lm_head projections) are committed operator boundaries (shape-parametric K)"
    if _GATHER_RE.match(kind):
        return "boundary", "the embedding output row is a committed boundary (predicates.py 'embedding-row'; shape-parametric V)"
    if kind == "Bf16Add_v1|batch":
        return "boundary", "the qkv_proj.add node's output row is a committed operator boundary (qwen2-eager-batch-invariant-v2: rope reads q/k and attention reads v from the add node)"
    if kind in MIXED_KINDS:
        return "interior-interpreter", "MIXED: " + MIXED_KINDS[kind]
    row = row_of(kind)
    return "interior-interpreter", (row.builder.note if row and row.builder.note else "output word is operator-interior: produced by E1's interpreter (brief interface 2)")


EXPOSURE_S_LOG2 = (14, 20)  # s = 2^-14 (the conformance-scale s of the TINY instance) and s = 2^-20


# [D23 2026-09-12] the kinds whose approved checker opens a row of a committed MUFU table (E2 table-layout/v1; E2 report 00:05Z "what the checkers
# open"): provable once the table roots are pinned in commitments.tables AND E7's builder shapes the 64-bit row operand
TABLE_READER_KINDS: dict[str, str] = {"RsqrtApprox_v1|primitive": "mufu/rsq", "DivFullRcp_v1|primitive": "mufu/rcp", "DivFullScaleA_v1|primitive": "mufu/rcp",
                                      "MufuSqrtFtz_v1|primitive": "mufu/sqrt"}


def inventory(expected: Any, *, population: str, layout_pin: str, ru_output_kinds: set[str] | None = None, scenario: str = APPROVED_ONLY, n_r: int = 1,
              measured_positions: dict[str, Any] | None = None, count_basis: str = "per RU", interior_committed: bool = False,
              tables_pinned: bool = False) -> dict[str, Any]:
    """The family inventory of one population (``population_v3.ExpectedPopulation``): one row per kind + one per prescribed-input family
    (+ one per compiled literal kind).  ``layout_pin`` is ``e2e.protocol_instance.interior_layout_status(...)[0]`` (pinned | placeholder |
    absent): an interior position is ``interior tree`` only when the layout is pinned or named (placeholder); ``refused`` when the pin is
    absent.  ``n_r`` = the instance's RU count; ``count_basis`` = ``"per RU"`` (Serve_v1: ``count`` per RU, ``count_total = count x n_r``) or
    ``"per instance"`` (generic program: ``count`` = total over the ``n_r`` RUs, ``count_per_ru = count / n_r`` when it divides).
    ``interior_committed`` = whether the instance's record commits its interior tree TODAY (CORE: E8's record does; TINY: not until E1/E2's
    pod run) -- decides ``provable_today`` for approved interior kinds; ``provable_with_interior_tree`` does not depend on it.
    ``measured_positions`` = E3's ``verifier_v3.locator.inventory(resolver)`` of the record (streamed; per kind ``boundary`` / ``interior``
    words, ``out_gates_per_vu``): folded per family as MEASURED evidence and reconciled in ``reconciliation``."""

    registered = registered_expectation_families()
    refusers = interior_refuser_kinds()
    approved = approved_families()
    families: list[dict[str, Any]] = []
    generic = expected.get("schema", "").endswith("generic/v1")
    meas_kinds = (measured_positions or {}).get("kinds") or {}
    assert count_basis in ("per RU", "per instance"), count_basis
    per_ru = count_basis == "per RU"

    def _tot(c: int) -> int:
        return c * n_r if per_ru else c

    def _pru(c: int) -> int | None:
        return c if per_ru else (c // n_r if n_r and c % n_r == 0 else None)

    for k in expected["kinds"]:
        sup = support_of(k["kind"], registered=registered, refusers=refusers, scenario=scenario, approved=approved)
        if generic and sup["support"] == "none":
            # a generic program's kinds are E8's choice: both DemoCore kinds are elem-bf16 builders; anything else stays unowned (reported)
            sup["note"] = "generic program kind with no Serve_v1 owner; " + sup["note"]
        producer, pnote = producer_of(k["kind"], ru_output_kinds=ru_output_kinds)
        if producer == "boundary":
            position, posnote = "boundary tree", "operator boundary leaf (E3 RecordPartition.locate)"
        elif layout_pin == "absent":
            position, posnote = "refused", "interior word but the instance carries no commitments.interior_layout pin (V3.interior_layout fails closed)"
        else:
            position, posnote = "interior tree", f"interior/<req>/<step> tree (layout pin {layout_pin}; E2 interior-layout/chunks-4096 rule)"
        cnt = int(k["count"])
        # provable TODAY = an approved checker with a registered builder whose words are committed today (boundary); an approved interior kind is
        # provable once E1/E2's interior tree exists (counted separately); anything else is not provable under the verifier's policy
        provable_now = sup["status"] == "provable" and (producer == "boundary" or interior_committed)
        provable_with_interior = sup["status"] == "provable"
        m = meas_kinds.get(k["kind"])
        measured = ({"boundary_words": int(m["boundary"]), "interior_words": int(m["interior"]), "vus": int(m["vus"]), "out_gates_per_vu": int(m["out_gates_per_vu"]),
                     "producer": m["producer"], "widths": list(m.get("widths") or [])} if m else None)
        families.append({"kind": k["kind"], "class": "VU", "count": cnt, "count_per_ru": _pru(cnt), "count_total": _tot(cnt), "width": int(k["declared_bits"]), **({"index_bits": k["index_bits"]} if "index_bits" in k else {}),
                         "support": sup["support"], "owner": sup["owner"], "family": sup.get("family"), "guest": sup.get("guest"), "status": sup["status"], "checker": sup["checker"],
                         "builder": sup["builder"], "provable_today": provable_now, "provable_with_interior_tree": provable_with_interior,
                         "producer": producer, "producer_note": pnote, "position": position, "position_note": posnote,
                         "exposure": {f"s=2^-{b}": exposure_at(cnt, b) for b in EXPOSURE_S_LOG2}, "measured_positions": measured, "note": sup["note"]})
    pres = expected["prescribed_inputs"]
    if generic:
        for c in pres["collections"]:
            families.append({"kind": c["kind"], "name": c["name"], "class": "prescribed input", "count": int(c["gates"]), "width": int(c["width"]), "support": "approved checker",
                             "owner": "verifier (declared input check, contract step 2; never sampled)", "status": "checked against the declaration", "checker": "verifier comparison",
                             "builder": None, "producer": "none", "producer_note": "declared by the auditor; not produced", "position": "boundary tree",
                             "position_note": "the declared input collection (record's input tree); not a VU", "note": "prescribed: in no VU and no RU"})
    else:
        for name, cnt, w, note in (("Input|batch|prompt", pres["prompt"], 32, "prompt token ids: LP int32 leaves, compared to epoch.inputs (declared) by the verifier"),
                                   ("Input|batch|weights|out=16", pres["weights_consumed"], 16, "consumed weight words: equal to the authorized checkpoint (weights root, V3.weights_root)"),
                                   ("Input|batch|weights|out=0", pres["weights_unconsumed_rope_rows"], 16, "rope cos/sin rows at positions >= P: read by nothing (derived table)")):
            families.append({"kind": name, "class": "prescribed input", "count": int(cnt), "width": w, "support": "approved checker",
                             "owner": "verifier (unconditional input check; weights: authorized commitment)", "status": "checked against the declaration / authorized root; never sampled",
                             "checker": "verifier comparison (prompt) / weights root recomputation (weights)", "builder": None, "producer": "none",
                             "producer_note": "declared / authorized; not produced by the prover", "position": "boundary tree",
                             "position_note": "weights tree root (weights_authorized) / declared prompt (epoch.inputs); not a VU", "note": note})
    lits = expected.get("literals") or {}
    if lits.get("treatment") == "compiled":
        # E5's ruling applied: the literal gates stay in the program (digest unchanged) but are prescribed by its text -- no VU, no committed word
        for name, cnt in sorted(lits["kinds"].items()):
            w = int(re.match(r"Const(\d+)\[", name).group(1))
            sup, owner, consumers = LITERAL_CONSUMERS.get(name, ("none", "none", "REPORT: literal kind with no consumer row (kind_table.LITERAL_CONSUMERS)"))
            families.append({"kind": name, "class": "prescribed literal", "count": int(cnt), "width": w, "support": sup, "owner": owner,
                             "status": "compiled into the consumer's expected statement (E5 ruling 2026-09-11; PROTOCOL-CONFORMANCE OR1); value verifier-derived from the pinned program",
                             "checker": "the consumer's checker pins the literal as a public input (`in ... expected`); a statement pinning another value is refused by name (LITERAL_MISMATCH)",
                             "builder": None, "producer": "none", "producer_note": "program literal: no produced value, no committed word (E1 must not count it as required; E2 commits no leaf)",
                             "position": "program literal", "position_note": "E3 RecordPartition.locate(g) returns a literal operand (kind=literal, width, value) from program.circuit.gate(g).prim",
                             "note": f"consumers: {consumers}"})
    for f in families:
        f.setdefault("count_per_ru", _pru(int(f["count"])))
        f.setdefault("count_total", _tot(int(f["count"])))
    vus = [f for f in families if f["class"] == "VU"]
    literal_fams = [f for f in families if f["class"] == "prescribed literal"]
    unowned = sorted(f["kind"] for f in vus + literal_fams if f["owner"] == "none")  # a named owner without a checker is NOT COVERED (below), not unowned
    no_producer = sorted(f["kind"] for f in vus if f["producer"] == "none")
    refused = sorted(f["kind"] for f in vus if f["position"] == "refused")
    not_covered = [{"kind": f["kind"], "count_per_ru": f["count_per_ru"], "count_total": f["count_total"],
                    "reason": (("no owner" if f["owner"] == "none" else f"no checker / builder (owner {f['owner'].split(' (')[0]})") if f["support"] == "none" else "") +
                    (" / " if f["support"] == "none" and f["producer"] == "none" else "") + ("no producer" if f["producer"] == "none" else ""),
                    "exposure": f["exposure"]} for f in vus if f["support"] == "none" or f["producer"] == "none"]
    by_support = {s: sorted(f["kind"] for f in vus if f["support"] == s) for s in SUPPORTS}
    by_producer = {p: sorted(f["kind"] for f in vus if f["producer"] == p) for p in PRODUCERS}
    n_v = sum(f["count"] for f in vus)
    provable = sum(f["count"] for f in vus if f["provable_today"])
    provable_int = sum(f["count"] for f in vus if f["provable_with_interior_tree"])
    not_provable = [f for f in vus if not f["provable_with_interior_tree"]]
    n_np = sum(f["count"] for f in not_provable)
    # reconciliation of the rule's words with E3's measured positions (when given): the measured record carries its own identity (e.g. attempt 6 /
    # D7 record = retained literals + argmax-scan); the rule's identity may compile literals and regroup the scan -- the deltas are named here
    recon = None
    if measured_positions:
        mk = meas_kinds
        meas_b = sum(int(v["boundary"]) for v in mk.values())
        meas_i = sum(int(v["interior"]) for v in mk.values())
        rule_b = sum(f["count"] for f in vus if f["producer"] == "boundary")
        rule_i = sum(f["count"] for f in vus if f["producer"] == "interior-interpreter")
        only_meas = sorted(set(mk) - {f["kind"] for f in vus})
        only_rule = sorted({f["kind"] for f in vus} - set(mk))
        deltas = []
        for kname in only_meas:  # kinds the measured record has and the rule's identity does not (literals compiled; ArgmaxStep regrouped)
            v = mk[kname]
            deltas.append({"kind": kname, "measured": {"boundary": int(v["boundary"]), "interior": int(v["interior"]), "vus": int(v["vus"]), "out_gates_per_vu": int(v["out_gates_per_vu"])},
                           "rule": "not a VU under this identity", "why": ("compiled literal (E5 ruling): no VU, no required word" if _CONST_RE.match(kname) else
                                                                           "regrouped: TokenSelect_v1|whole-scan is the VU (E5 ruling); the scan's per-step words are not VU outputs" if kname.startswith("ArgmaxStep")
                                                                           else "REPORT: kind absent from the rule's identity")})
        for f in vus:
            m = f.get("measured_positions")
            if m and (m["boundary_words"] + m["interior_words"] != f["count"] * m["out_gates_per_vu"] or m["vus"] != f["count"]):
                deltas.append({"kind": f["kind"], "measured": m, "rule": f["count"], "why": "REPORT: measured VU/word count differs from the rule"})
        recon = {"measured": {"boundary_words": meas_b, "interior_words": meas_i, "vus": sum(int(v["vus"]) for v in mk.values()), "kinds": len(mk),
                              "source": (measured_positions.get("_meta") or {})},
                 "rule": {"boundary_vus": rule_b, "interior_vus": rule_i, "n_v": n_v, "kinds": len(vus), "one_word_per_vu": True},
                 "kinds_only_in_measured": only_meas, "kinds_only_in_rule": only_rule,
                 "interior_words_under_rule": rule_i, "boundary_words_under_rule": rule_b,
                 "delta_interior_words": rule_i - meas_i, "delta_boundary_words": rule_b - meas_b, "deltas": deltas,
                 "agree_on_common_kinds": not any(d["why"].startswith("REPORT") for d in deltas)}
    return {"schema": INVENTORY_SCHEMA, "population": population, "based_on": BASED_ON, "scenario": scenario, "layout_pin": layout_pin, "n_r": n_r, "count_basis": count_basis,
            "interior_committed": interior_committed,
            "program": expected["program_identity"], "kind_table_digest": expected["kind_table_digest"], "kind_table_digest_kind": expected["kind_table_digest_kind"],
            "counts_label": "verifier-derived by rule (population_v3.expected_population); no sampled input gates; `count` " + count_basis
                            + ("; `count_total` = count x N_R" if per_ru else " (total over the instance's N_R RUs); `count_per_ru` = count / N_R when it divides"),
            "literals": lits.get("treatment"), "token_select": expected.get("token_select"), "approved_families": approved,
            "summary": {"families": len(families), "vu_kinds": len(vus), "prescribed_input_families": len(families) - len(vus) - len(literal_fams),
                        "prescribed_literal_families": len(literal_fams), "literal_gates": sum(f["count"] for f in literal_fams), "n_v": n_v, "n_r": n_r, "n_v_total": _tot(n_v), "n_v_per_ru": _pru(n_v),
                        "provable_vus": provable, "provable_share": (provable / n_v if n_v else None),
                        "provable_with_interior_tree_vus": provable_int, "provable_with_interior_tree_share": (provable_int / n_v if n_v else None),
                        "not_provable_today_vus": n_np, "not_provable_today_kinds": sorted(f["kind"] for f in not_provable),
                        "not_provable_today_exposure": {f"s=2^-{b}": exposure_at(n_np, b) for b in EXPOSURE_S_LOG2},
                        # [D23] of the not-provable VUs, those whose checker reads a committed MUFU table: unlocked by commitments.tables (+ E7's builder)
                        "table_reader_vus_not_provable": sum(f["count"] for f in not_provable if f["kind"] in TABLE_READER_KINDS),
                        "table_reader_kinds_not_provable": {f["kind"]: TABLE_READER_KINDS[f["kind"]] for f in not_provable if f["kind"] in TABLE_READER_KINDS},
                        "tables_pinned": tables_pinned,
                        "owned": len(vus) - len([k for k in unowned if k in {f["kind"] for f in vus}]),
                        "unowned": unowned, "no_producer": no_producer, "refused_position": refused, "not_covered": not_covered,
                        "by_support": {s: len(v) for s, v in by_support.items()}, "by_producer": {p: len(v) for p, v in by_producer.items()},
                        "vus_by_producer": {p: sum(f["count"] for f in vus if f["producer"] == p) for p in PRODUCERS},
                        "vus_by_support": {s: sum(f["count"] for f in vus if f["support"] == s) for s in SUPPORTS}},
            "reconciliation": recon, "by_support": by_support, "by_producer": by_producer, "families": families,
            "registered_expectation_builders": registered_expectation_families(), "interior_refusers": sorted(interior_refuser_kinds())}


def coverage_table(inv: dict[str, Any]) -> dict[str, Any]:
    """The coverage table the population identity is frozen against (``a9/legacy-minus-inputs/v2`` pins its sha): kind -> support / status /
    producer / position, nothing else (no counts: the table describes kinds, the population describes the instance)."""

    rows = {f["kind"]: {"class": f["class"], "support": f["support"], "owner": f["owner"], "status": f["status"], "producer": f["producer"], "position": f["position"]}
            for f in inv["families"]}
    return {"schema": COVERAGE_TABLE_SCHEMA, "based_on": BASED_ON, "scenario": inv["scenario"], "population": inv["population"], "layout_pin": inv["layout_pin"],
            "registered_expectation_builders": inv["registered_expectation_builders"], "kinds": rows}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else (f"{x * 100:.4g}%" if x < 0.9999 else f"{x * 100:.6f}%")


def inventory_markdown(inv: dict[str, Any]) -> str:
    s = inv["summary"]
    prog = inv["program"]
    n_r = inv.get("n_r", 1)
    per_ru = inv.get("count_basis", "per RU") == "per RU"
    exp_keys = [f"s=2^-{b}" for b in EXPOSURE_S_LOG2]
    lines = [f"# Family inventory -- {inv['population']}\n",
             f"Program `{prog.get('function')}`; counts verifier-derived by rule (`{inv['kind_table_digest_kind']}` digest `{inv['kind_table_digest'][:16]}...`); "
             + (f"no sampled input gates; `count` per RU, `total` = count x N_R = {n_r}. Layout pin:" if per_ru else
                f"no sampled input gates; `count` = total over the instance's N_R = {n_r} RUs, `count/RU` = count / N_R. Layout pin:") + f" `{inv['layout_pin']}`. Scenario `{inv['scenario']}`; "
             f"table based on `{inv['based_on']}`"
             + (f"; literals `{inv['literals']}`" if inv.get("literals") else "") + (f"; token select `{inv['token_select']}`" if inv.get("token_select") else "")
             + f". Approved families (verifier policy `e2e/approved_guests.json`, approval = reproduced): {inv.get('approved_families')}.\n"]
    nc = s.get("not_covered") or []
    lines.append("## NOT COVERED (no owner or no producer)\n")
    if nc:
        lines += [f"| kind | reason | count/RU | total | exposure {exp_keys[0]} | exposure {exp_keys[1]} |", "|---|---|---:|---:|---:|---:|"]
        lines += [f"| `{r['kind']}` | {r['reason']} | {r['count_per_ru'] if r['count_per_ru'] is not None else 'uneven'} | {r['count_total']:,} | {_pct(r['exposure'][exp_keys[0]])} | {_pct(r['exposure'][exp_keys[1]])} |" for r in nc]
        lines.append("")
    else:
        lines.append("**none** -- every VU family has a named owner and a producer.\n")
    lines += [f"**Summary:** {s['families']} families = {s['vu_kinds']} VU kinds + {s['prescribed_input_families']} prescribed-input families"
              + (f" + {s['prescribed_literal_families']} prescribed-literal families ({s['literal_gates']:,} compiled literal gates)" if s.get("prescribed_literal_families") else "")
              + (f"; N_V = {s['n_v']:,} per RU x N_R = {n_r} = {s['n_v_total']:,} total; " if per_ru else
                 f"; N_V = {s['n_v']:,} total over N_R = {n_r} RUs ({s.get('n_v_per_ru') if s.get('n_v_per_ru') is not None else 'uneven'} per RU); ")
              + f"provable today (approved checker, registered builder, words committed today{'' if inv.get('interior_committed') else ' = boundary only; interior tree pending'}) "
              f"{s['provable_vus']:,} VUs ({_pct(s['provable_share'])}); "
              f"provable once the interior tree exists (approved checker + builder, any position) {s['provable_with_interior_tree_vus']:,} VUs ({_pct(s['provable_with_interior_tree_share'])}); "
              f"NOT provable under today's policy {s['not_provable_today_vus']:,} VUs in {len(s['not_provable_today_kinds'])} kinds "
              f"(exposure 1-(1-s)^n: {exp_keys[0]} {_pct(s['not_provable_today_exposure'][exp_keys[0]])}, {exp_keys[1]} {_pct(s['not_provable_today_exposure'][exp_keys[1]])}); "
              + (f"of these {s['table_reader_vus_not_provable']:,} VUs in {len(s['table_reader_kinds_not_provable'])} kinds read a committed MUFU table "
                 f"({'roots pinned in commitments.tables, D23 -- provable when E7 registers the row-operand builder' if s.get('tables_pinned') else 'table roots NOT pinned'}); "
                 if s.get("table_reader_vus_not_provable") else "")
              + f"owned {s['owned']}/{s['vu_kinds']} VU kinds; unowned: {', '.join(f'`{k}`' for k in s['unowned']) or 'none'}; no producer: {', '.join(s['no_producer']) or 'none'}; "
              f"refused position: {', '.join(s['refused_position']) or 'none'}. "
              f"Kinds by support: {s['by_support']}; VUs by support: { {k: f'{v:,}' for k, v in s.get('vus_by_support', {}).items()} }. "
              f"Kinds by producer: {s['by_producer']}; VUs by producer: { {k: f'{v:,}' for k, v in s.get('vus_by_producer', {}).items()} }.\n"]
    rc = inv.get("reconciliation")
    if rc:
        m, r = rc["measured"], rc["rule"]
        src = m.get("source") or {}
        lines += ["## Reconciliation with the measured positions (E3 `verifier_v3.locator.inventory`, streamed)\n",
                  f"Measured on `{src.get('archive', '?')}` request `{src.get('request_id', '?')}` ({src.get('seconds', '?')} s; population `{str(src.get('population_kind'))}` "
                  f"{str(src.get('population_partition_digest'))[:16]}...): **boundary words {m['boundary_words']:,}, interior words {m['interior_words']:,}**, {m['vus']:,} VUs in {m['kinds']} kinds "
                  f"(the record's own identity). Under THIS instance's identity: boundary VUs = words {r['boundary_vus']:,}; interior VUs = required words {r['interior_vus']:,} "
                  f"(one word per VU); N_V {r['n_v']:,} in {r['kinds']} kinds. Delta interior words {rc['delta_interior_words']:+,}; delta boundary words {rc['delta_boundary_words']:+,}. "
                  f"Kinds only in the measured record: {', '.join(f'`{k}`' for k in rc['kinds_only_in_measured']) or 'none'}; only in the rule: "
                  f"{', '.join(f'`{k}`' for k in rc['kinds_only_in_rule']) or 'none'}. Common kinds agree VU-for-VU: **{rc['agree_on_common_kinds']}**.\n"]
        if rc["deltas"]:
            lines += ["| kind | measured (boundary / interior words, VUs, words per VU) | under this identity | why |", "|---|---|---|---|"]
            for d in rc["deltas"]:
                mm = d["measured"]
                mtxt = (f"{mm.get('boundary', mm.get('boundary_words', 0)):,} / {mm.get('interior', mm.get('interior_words', 0)):,}, {mm.get('vus', 0):,} VUs, {mm.get('out_gates_per_vu', '?')} w/VU")
                lines.append(f"| `{d['kind']}` | {mtxt} | {d['rule']} | {d['why']} |")
            lines.append("")
    lines += [f"| kind | class | count/RU | total | width | support | owner | family / guest | status | producer | position | exposure {exp_keys[0]} | exposure {exp_keys[1]} | measured b/i words (w/VU) |",
              "|---|---|---:|---:|---:|---|---|---|---|---|---|---:|---:|---|"]
    for f in inv["families"]:
        ex = f.get("exposure") or {}
        mp = f.get("measured_positions")
        mtxt = f"{mp['boundary_words']:,} / {mp['interior_words']:,} ({mp['out_gates_per_vu']})" if mp else ("--" if f["class"] == "VU" else "n/a")
        fam = f"{f.get('family') or '--'} / {f.get('guest') or '--'}" if f["class"] == "VU" else "--"
        cpr = f.get("count_per_ru")
        label = f["kind"] + (f" ({f['name']})" if f.get("name") else "")
        lines.append(f"| `{label}` | {f['class']} | {cpr if cpr is not None else 'uneven'} | {f.get('count_total', f['count']):,} | {f['width']} | {f['support']} | {f['owner']} | {fam} | {f['status']} | "
                     f"{f['producer']} | {f['position']} | {_pct(ex.get(exp_keys[0])) if ex else 'n/a'} | {_pct(ex.get(exp_keys[1])) if ex else 'n/a'} | {mtxt} |")
    lines.append("\n## Notes per family\n")
    for f in inv["families"]:
        notes = [t for t in (f.get("producer_note"), f.get("position_note"), f.get("note")) if t]
        lines.append(f"- `{f['kind']}{' (' + f['name'] + ')' if f.get('name') else ''}`: " + " | ".join(notes))
    return "\n".join(lines) + "\n"


# --- markdown -----------------------------------------------------------------------------------------------------------


def _md_table(doc: dict[str, Any]) -> str:
    cfg = doc["configurations"]
    c8099 = cfg.get("6500x1599_P8099", {}).get("counts_per_RU", {})
    c3215 = cfg.get("32x15", {}).get("counts_per_RU", {})
    lines = [
        "| kind | builder (selected VU -> statement) | checker: guest / gate set / ops | framings | status: approved | + range | + fa2 | VUs/RU @6500x1599 | VUs/RU @32x15 |",
        "|---|---|---|---|---|---|---|---:|---:|",
    ]
    for k in doc["kinds"]:
        b = k["builder"]
        c = k["checker"]
        bt = ("yes" if b["exists"] and b["granularity"] == "same" else ("larger unit only (" + b["granularity"] + ")" if b["exists"] else "no")) + "; " + b["evidence"][0]
        ct = (("yes" if c["granularity"] == "same" else f"only as {c['granularity']}") if c["exists"] else "no") + f" -- {'/'.join(c['guests'])}: {c['gate_set']} [{', '.join(c['ops'])}]; " + "; ".join(c["evidence"][:2])
        name = k["kind"] + (" (legacy)" if k["legacy"] else "")
        st = k["status"]

        def n(c: dict[str, int]) -> str:
            v = c.get(k["kind"])
            return "--" if v is None else f"{v:,}"

        lines.append(f"| `{name}` | {bt} | {ct} | {'; '.join(k['framings'])} | {st[APPROVED_ONLY]} | {st[PLUS_RANGE]} | {st[PLUS_FA2]} | {n(c8099)} | {n(c3215)} |")
    return "\n".join(lines)


def _md_exposure(name: str, cfg: dict[str, Any]) -> str:
    out = [f"### {name}\n", f"Source: {cfg['source']}. N_V per RU = {cfg['N_V_per_RU']:,}; retained set of {cfg['N_R']} RUs.\n"]
    for sc in SCENARIOS:
        d = cfg["by_scenario"][sc]
        out.append(f"**Scenario `{sc}`** -- unsupported set per RU n_R = {d['n_R']:,} ({len(d['unsupported_kinds'])} kinds); N over 4 RUs = {d['N_4RU']:,}. "
                   f"With A5/6's fallback builders (SiluMulBf16, Bf16AddF2fp; `e2e/coverage_v1/elem_builder.py`, transparent-verified on B0 rows, not yet pipeline-wired, no SP1 proof run): "
                   f"n_R = {d['n_R_with_a56_fallbacks']:,} ({len(d['unsupported_kinds_with_a56_fallbacks'])} kinds), "
                   f"1-(1-s)^n_R at 2^-30 = 1 - 10^({d['exposure_per_RU_with_a56_fallbacks']['2^-30']['complement_log10']}). "
                   f"Checker-missing-only floor (if every missing builder existed): n_R = {d['n_R_checker_missing_only']:,} ({len(d['checker_missing_kinds'])} kinds).\n")
        out.append("| s | 1-(1-s)^n_R | E[unsupported selected]/RU | 1-(1-s)^N | E[...]/4 RUs | floor: 1-(1-s)^n_R | floor: 1-(1-s)^N |")
        out.append("|---|---|---:|---|---:|---|---|")
        for sname in S_VALUES:
            a, b_ = d["exposure_per_RU"][sname], d["exposure_4RU"][sname]
            fa, fb = d["exposure_per_RU_if_every_builder_existed"][sname], d["exposure_4RU_if_every_builder_existed"][sname]

            def p(x: dict[str, str]) -> str:
                return x["probability"] if x["probability"] in ("0",) else f"1 - 10^({x['complement_log10']})"

            out.append(f"| {sname} | {p(a)} | {a.get('expected_unsupported_selected_decimal', '0')} | {p(b_)} | {b_.get('expected_unsupported_selected_decimal', '0')} | {p(fa)} | {p(fb)} |")
        out.append("")
    return "\n".join(out)


def to_markdown(doc: dict[str, Any]) -> str:
    parts = [
        "# A5/6 coverage table -- VU kinds: builder, checker, framing, status; unsupported-check exposure (R18 input)\n",
        f"Based on mainline `{doc['based_on']}`; schema `{doc['schema']}`; generated by `python -m e2e.coverage_v1.kind_table`. "
        "Every cell is a code reading with file:line (mainline). Statuses use the contract vocabulary only; the legacy label \"exempt (charged)\" is not a status.\n",
        "**Label for every probability below: probability of selecting an unsupported check -- completion risk.** It is not an exemption of any family, RU or kind.\n",
        "## Guests\n",
    ]
    for gid, g in doc["guests"].items():
        parts.append(f"- `{gid}`: {g['approval']}. Source: {g['source']}. Families: {g['families'] if isinstance(g['families'], str) else ', '.join(g['families'])}. Framings: {'; '.join(g['framings'])}."
                     + (f" {g['source_width']}." if "source_width" in g else ""))
    parts.append("\n## Scenarios\n")
    for sc, text in doc["scenarios"].items():
        parts.append(f"- `{sc}`: {text}")
    parts.append("\n## Kind table\n")
    parts.append(_md_table(doc))
    parts.append("\n### Domain caveats and notes (R10 input)\n")
    for k in doc["kinds"]:
        notes = [t for t in (k["checker"].get("domain_caveat"), k["builder"].get("note"), k.get("note")) if t]
        if notes:
            parts.append(f"- `{k['kind']}`: " + " | ".join(notes))
    parts.append("\n## Unsupported set and exposure\n")
    parts.append("n_R = VUs per RU whose kind is not `supported (approved)` in the scenario; N = 4 n_R (retained set of 4 RUs, one RU per request). "
                 "`floor` = the same with every kind whose approved checker exists counted as supported (i.e. if every missing builder were implemented and its interface committed): "
                 "the completion risk that approval scenarios alone cannot remove. Probabilities via decimal(60) `1 - exp(n ln(1-s))`; expectations exact.\n")
    for name, cfg in doc["configurations"].items():
        parts.append(_md_exposure(name, cfg))
    parts.append("\n## Reading\n")
    parts.append("- Under every scenario the as-built unsupported set is every kind but GemmCoordinate (K=576, K=1536) and Input: the approval scenarios move NO kind, "
                 "because the range guest has the same gate sets and the fa2 guest opens 16-bit sources only. At s = 2^-30 and one RU the expected number of unsupported selected checks is already ~95; "
                 "the probability that a run selects at least one is 1 to within 10^-41. Completion of a P1/P1v2 record therefore depends on builders and commitments, not on guest approvals.\n"
                 "- If every kind the approved guest can already evaluate had a builder and a committed interface, the remaining (checker-missing) set is exactly the FTZ attention interior "
                 "(10 kinds), whose only checker is the unapproved fa2 guest and only at whole-head granularity (S55).\n")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--expected-work", type=Path, help="D4's expected_work.json (lane/record-6k:out/record-6k/expected_work.json)")
    ap.add_argument("--census", type=Path, help="D2's census_small.json (lane/partition:data/census/census_small.json)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    ew = json.loads(a.expected_work.read_text()) if a.expected_work else None
    cs = json.loads(a.census.read_text()) if a.census else None
    doc = build(ew, cs)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "coverage.json").write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    (a.out / "coverage.md").write_text(to_markdown(doc))
    for name, cfg in doc["configurations"].items():
        for sc in SCENARIOS:
            d = cfg["by_scenario"][sc]
            print(f"{name} {sc}: n_R={d['n_R']:,} N={d['N_4RU']:,} checker-missing-only n_R={d['n_R_checker_missing_only']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
