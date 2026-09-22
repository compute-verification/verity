"""Build W11's result rows (Table 2 / Table 4 / manifest) from the raw check reports.

    cd vllm-poc && python -m verity_vllm_numerics.attn_rows results/W11-40f0cebeb670-20260907T1800Z

Reads ``<results>/raw/*.json`` (written on the pod by attn_check.py, attn_stage_gpu.py,
attn_mutation_check.py, mufu_probe verify) and the fixture indices, writes ``table2.jsonl``
(7 rows: the four stages, then full decode and full prefill attention, then the MUFU tables as
background characterisation), ``table4.jsonl`` (checker cost), ``manifest.json`` and
``dataflow_variants.json``.  Every number is copied from the raw files; nothing is extrapolated.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POC = HERE.parent
sys.path.insert(0, str(POC))

from verity_vllm import rows as ROWS  # noqa: E402
from verity_vllm_numerics import fa2_model as FA  # noqa: E402

EXP = "W11-40f0cebeb670-20260907T1800Z"
PROFILE = "4090-vllm-bi1-fa2"
H, KVH, D = 12, 2, 128


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=str(POC), check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def main(results: Path) -> None:
    raw = results / "raw"
    fx = POC / "fixtures" / EXP
    nat_idx = _load(fx / "natural" / "index.json")
    st_idx = _load(fx / "stress" / "index.json")
    check_all = _load(raw / "check_all.json")
    check_stress = _load(raw / "check_stress.json")
    muts = _load(raw / "mutations_check.json")
    stage_nat = _load(raw / "stage_natural_600.json")
    stage_st = _load(raw / "stage_stress_300.json")
    dec1 = _load(raw / "check_decode_1thread.json")
    pre1 = _load(raw / "check_prefill_1thread.json")
    mufu = _load(raw / "mufu_verify_frac4.json")
    sass = _load(raw / "production_sass_summary.json")

    common = dict(experiment_id=EXP, case="B1", profile_id=PROFILE, worker="W11", adapter_revision=f"{_git_rev()} (vllm-poc-w11-attention-model)")
    runtime_impl = ("vllm.vllm_flash_attn.flash_attn_varlen_func -> flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, false, false, "
                    "cutlass::bfloat16_t>, Is_causal, Is_local=false, Has_alibi=false, Is_even_MN=false, Is_even_K=true, Is_softcap=false, Split=false, "
                    "Append_KV=false> (vllm-flash-attn 506341a, --use_fast_math, sm_80 cubin on sm_89; _vllm_fa2_C.abi3.so sha256 "
                    f"{nat_idx['vllm_flash_attn_so_sha256']['_vllm_fa2_C.abi3.so'][:16]}...; num_splits=1 under VLLM_BATCH_INVARIANT)")
    domain_covered = ("sm_89 (RTX 4090), head_dim 128, (12, 2) heads, block_size 16, FA2, num_splits 1, causal, no window/softcap/alibi, BF16 Q/K/V/O, "
                      "1 <= seqlen_q <= seqlen_k <= 4096, finite operands and finite FP32 intermediates; everything else `unresolved`")

    # ---- stage rows (model intermediates vs the same SASS instructions re-executed on the GPU on the natural / stress operands)
    def stage(nat_key: str, st_key: str, excl_st: int = 0):
        a = stage_nat["stages"][nat_key]
        b = stage_st["stages"][st_key]
        return (a["compared"], a["exact"]), (b["compared"] - excl_st, b["exact"])

    # the one sampled stress row with a NaN operand (extremes/k_nan_n64, seq 24 head 7) is outside the domain: excluded from the stage counts
    nan_row = [r for r in stage_st["rows"] if r["seq"] == 24 and r["head"] == 7]  # the head that reads the KV head holding the NaN
    assert len(nan_row) == 1 and nan_row[0]["S_mismatch"] == 1
    assert all(r["S_mismatch"] == 0 for r in stage_st["rows"] if not (r["seq"] == 24 and r["head"] == 7))
    excl = {"A_qk_mma_S": 1, "B_softmax_P": 64, "B_rescale_factor": 0, "B_row_sum": 4, "C_pv_mma_O": 128, "D_inv": 0, "D_out_word": 97}
    for k, v in excl.items():
        assert stage_st["stages"][k]["compared"] - stage_st["stages"][k]["exact"] == v, k
    stage_note = (f"reference for the intermediates = the same SASS instructions (HMMA.16816.F32.BF16 via mma.sync, FFMA/FMUL/FADD .FTZ, MUFU.EX2, MUFU.RCP) re-executed on the "
                  f"RTX 4090 by cuda/fa2_softmax_probe.cu + mma_tiles on the fixture operands ({len(stage_nat['rows'])} natural (row, head) samples across 1..33 blocks, "
                  f"{len(stage_st['rows']) - 1} stress samples; production's intermediates are not observable, its output is the end-to-end rows); the stress sample from "
                  "extremes/k_nan_n64 (NaN operand, outside the domain) is excluded from these counts (its S/P/O and the hardware chain itself differ from production there)")

    def stage_row(op_id, decomposition, semantics, nat, st, flops, extra_lim=""):
        (nc, ne), (sc, se) = nat, st
        row = dict(common, op_id=op_id, phase="decode+prefill (natural + stress operands)", shape={"head_dim": D, "kBlockN": 64, "kBlockM": 64},
                   dtype="bf16 in, fp32 accumulate", layout_or_quant="Q row x K rows (n_block from last to first), P bf16 x V rows",
                   runtime_impl=runtime_impl, primitive_semantics=semantics, decomposition=decomposition,
                   interface={"input_bytes": 2 * D + 2 * 64 * D, "output_bits": 32 * 64, "range": None}, origin="derived",
                   binding={"present": 0, "unresolved": 0}, checker="independent",
                   natural_checks={"exact": ne, "attempted": nc}, stress_checks={"exact": se, "attempted": sc, "domain": domain_covered},
                   mutations={"rejected": 0, "attempted": 0},
                   checker_work={"seconds": 0.0, "flops_or_bytes": float(flops), "includes_full_op": False},
                   status="exact" if (ne == nc and se == sc) else "mismatch",
                   limitation=(f"stage of the attention model (contract 15.9 background for the two full `attention` rows); cost not measured separately; {stage_note}" + extra_lim))
        return ROWS.require_valid(2, row)

    rows = []
    rows.append(stage_row(
        "attention.qk_mma", "direct QK^T MMA: S = Q K^T over one 64-key block, 8 mma.sync m16n8k16 BF16 k-steps, FP32 accumulator from 0",
        "ampere_bf16_m16n8k16 chain (veritor.core.silicon.AMPERE_BF16_M16N8K16: groups (8,8), width 25, exponent floor -132), k ascending, accumulator 0; per S element",
        *stage("A_qk_mma_S", "A_qk_mma_S", excl["A_qk_mma_S"]), flops=2 * D))
    rows.append(stage_row(
        "attention.online_softmax_step", "online-softmax block step: running max, P = ex2(fma(S, scale_log2, -m*scale_log2)), rescale factor ex2((m_prev-m_new)*scale_log2), per-lane row sums",
        "FTZ FP32; MUFU.EX2 = measured 2^23-entry table + FRAC_MODE-4 index rule (exhaustive 2^32, 0 mismatches); scale_log2 = f32(f32(scale) * M_LOG2E); "
        "P via FFMA (one rounding); row sums per quad lane in element order, the block's first add contracted with the rescale into FFMA(scale, l, p0) as compiled",
        (stage_nat["stages"]["B_softmax_P"]["compared"] + stage_nat["stages"]["B_rescale_factor"]["compared"] + stage_nat["stages"]["B_row_sum"]["compared"],
         stage_nat["stages"]["B_softmax_P"]["exact"] + stage_nat["stages"]["B_rescale_factor"]["exact"] + stage_nat["stages"]["B_row_sum"]["exact"]),
        (stage_st["stages"]["B_softmax_P"]["compared"] - excl["B_softmax_P"] + stage_st["stages"]["B_rescale_factor"]["compared"] + stage_st["stages"]["B_row_sum"]["compared"] - excl["B_row_sum"],
         stage_st["stages"]["B_softmax_P"]["exact"] + stage_st["stages"]["B_rescale_factor"]["exact"] + stage_st["stages"]["B_row_sum"]["exact"]),
        flops=64 * 4 + 4,
        extra_lim="; counts = P values + rescale factors + final per-lane sums (4 per row)"))
    rows.append(stage_row(
        "attention.pv_mma", "PV MMA: O += bf16(P) V over the 64-key block, 4 mma.sync m16n8k16 k-steps, accumulator = the rescaled O",
        "P rounded RNE to BF16 before the MMA (F2FP.BF16.PACK_AB); ampere_bf16_m16n8k16 chain with the FP32 O as the accumulator; V rows past seqlen_k zero; per O element after the last block",
        *stage("C_pv_mma_O", "C_pv_mma_O", excl["C_pv_mma_O"]), flops=2 * 64))
    rows.append(stage_row(
        "attention.normalize_cast", "epilogue: sum = (l0+l2)+(l1+l3), inv = MUFU.RCP(sum) (1.0 if sum == 0 or NaN), out = bf16_rne(O * inv)",
        "quad allreduce order (l0+l2)+(l1+l3) FTZ; MUFU.RCP = measured 2^23-entry mantissa table (exhaustive 2^32, 0 mismatches); FMUL.FTZ; cvt RNE to BF16 (NaN word 0x7fff); per output word",
        (stage_nat["stages"]["D_out_word"]["compared"] + stage_nat["stages"]["D_inv"]["compared"], stage_nat["stages"]["D_out_word"]["exact"] + stage_nat["stages"]["D_inv"]["exact"]),
        (stage_st["stages"]["D_out_word"]["compared"] - excl["D_out_word"] + stage_st["stages"]["D_inv"]["compared"], stage_st["stages"]["D_out_word"]["exact"] + stage_st["stages"]["D_inv"]["exact"]),
        flops=2, extra_lim="; counts = output words + the inv scalar per row"))

    # ---- full attention rows (model vs the production kernel's output words)
    mut_phase = {}
    st_phase = {c["seq"]: c["phase"] for c in st_idx["calls"]}
    nat_phase = {c["seq"]: c["phase"] for c in nat_idx["calls"]}
    for m in muts["mutations"]:
        ph = (nat_phase if m.get("base_fixture") else st_phase)[m["seq"]]
        d = mut_phase.setdefault(ph, {"flips": 0, "changed": 0, "rejected": 0, "tracks": 0, "declined": 0, "undetected": 0})
        d["flips"] += 1
        if m["production_output_words_changed"]:
            d["changed"] += 1
            d["rejected"] += int(bool(m["rejected"]))
            d["undetected"] += int(not m["rejected"])
        if m["model_tracks_mutation"] is None:
            d["declined"] += 1
        elif m["model_tracks_mutation"]:
            d["tracks"] += 1
    assert all(v["undetected"] == 0 for v in mut_phase.values())
    assert muts["counts"]["tracks_mismatch"] == 0

    def full_row(phase_key, phase_text, shape, cost_seconds, flops, unresolved_text, extra):
        nat = check_all["stats"][f"phase={phase_key}"]
        st = check_stress["stats"][f"phase={phase_key}"]
        mp = mut_phase[phase_key]
        fam = sorted(k.split("=", 1)[1] for k in check_stress["stats"] if k.startswith("family="))
        row = dict(common, op_id="attention", phase=phase_text, shape=shape, dtype="bfloat16", layout_or_quant="paged BF16 KV cache (block 16) read as canonical K/V rows; GQA 6:1 (kv head = h // 6)",
                   runtime_impl=runtime_impl, primitive_semantics="attention.bf16.v1 = " + FA.KERNEL["kernel"].split("<", 1)[0] + " modelled instruction by instruction (fa2_model.cpp; see docs/vllm-poc/attention-model.md)",
                   decomposition="one output row (one query position x 12 heads x 128) from q and the authenticated K/V rows 0..t",
                   interface={"input_bytes": 2 * H * D + 2 * 2 * KVH * D, "output_bits": 16 * H * D, "range": None}, origin="captured",
                   binding={"present": 0, "unresolved": 0}, checker="independent",
                   natural_checks={"exact": nat["exact"], "attempted": nat["outputs"]},
                   stress_checks={"exact": st["exact"], "attempted": st["outputs"], "domain": "families " + ", ".join(fam) + "; " + domain_covered},
                   mutations={"rejected": mp["rejected"], "attempted": mp["changed"]},
                   checker_work={"seconds": cost_seconds, "flops_or_bytes": float(flops), "includes_full_op": True},
                   status="exact" if nat["mismatch"] == 0 and st["mismatch"] == 0 else "mismatch",
                   limitation=(f"{nat['calls']} natural kernel calls ({nat['rows']} (row, head) outputs) and {st['calls']} stress calls ({st['rows']}) checked, 0 mismatching words; "
                               f"mutations: {mp['flips']} one-bit flips of Q/K/V (sign, exponent and mantissa bits) re-run through the production kernel, {mp['changed']} changed its output and "
                               f"all {mp['rejected']} of those are rejected by the model of the unmutated operands ({mp['flips'] - mp['changed']} flip(s) left the output bit-identical: nothing to reject); "
                               f"the model follows the mutated operands bit-exactly in {mp['tracks']} cases and declines {mp['declined']} (exponent-bit flips that overflow FP32 -> unresolved); "
                               f"{unresolved_text}{extra}"))
        return ROWS.require_valid(2, row)

    # single-thread cost per output row (one position x 12 heads), from the OMP_NUM_THREADS=1 runs
    dpc = dec1["per_call"]
    lk = np.array([c["Lk"] for c in dpc], float)
    sec = np.array([c["seconds"] for c in dpc])
    nrow = np.array([c["rows"] for c in dpc], float)  # (row, head) units
    dec_cost = float(sec.sum() / nrow.sum() * H)
    coef = np.polyfit(lk, sec / nrow, 1)
    ppc = pre1["per_call"]
    pre_cost = float(sum(c["seconds"] for c in ppc) / sum(c["rows"] for c in ppc) * H)
    nat_dec_lk = np.mean([c["Lk"] for c in nat_idx["calls"] if c["phase"] == "decode"])
    # mean visible keys of the checked prefill rows
    vis = []
    for c in nat_idx["calls"]:
        if c["phase"] == "prefill":
            vis += [r + 1 for r in c["rows"]]
    pre_vis = float(np.mean(vis))

    rows.append(full_row(
        "decode", f"decode (Lq=1, Lk 6..2071 = every step of 3 requests, layers 0/13/27; stress Lk 1..4096)", [H, D],
        dec_cost, 4 * H * D * nat_dec_lk,
        "unresolved on the stress calls with non-finite operands or FP32 overflow (extremes: q_inf_n64, k_nan_n64, v_near_max_n64, v_pm_max_n64) -- the model declines, production writes 0x7fff words",
        f"; cost: single-thread CPU {1e3 * dec_cost:.2f} ms per output row (12 heads) at the natural mean Lk {nat_dec_lk:.0f} "
        f"(fit over {len(dpc)} calls: {1e6 * coef[0]:.2f} us x Lk + {1e6 * coef[1]:.0f} us per (row, head)); with OpenMP on 64 cores the full natural fixture "
        f"({check_all['stats']['all=all']['outputs']} words) took {check_all['checker_seconds_total']:.1f} s wall"))
    rows.append(full_row(
        "prefill", "prefill (causal Lq=Lk 5/700/2000: 5 + 154 + 337 sampled rows per call x 3 layers; stress Lq 1..700, Lk to 4096)", [H, D],
        pre_cost, 4 * H * D * pre_vis,
        "no unresolved prefill calls",
        f"; a prefill row is checked exactly like a decode row (single query, keys 0..t): the kernel's extra fully-masked blocks inside a 64-row M block are arithmetic "
        f"no-ops, verified on the fixture (tests/test_attention_model.py::test_prefill_row_equals_single_query_call); cost: single-thread CPU {1e3 * pre_cost:.2f} ms per output row "
        f"(12 heads) at the sampled rows' mean t+1 = {pre_vis:.0f}"))

    # ---- MUFU tables (background characterisation)
    rows.append(ROWS.require_valid(2, dict(
        common, op_id="mufu.ex2+rcp.approx.ftz.f32", phase="instruction (exhaustive 2^32 inputs each)", shape={"table_entries": 1 << 23}, dtype="fp32 -> fp32",
        layout_or_quant=None, runtime_impl="MUFU.EX2 (ex2.approx.ftz.f32 = exp2f under --use_fast_math), MUFU.RCP (rcp.approx.ftz.f32 = 1.f/x) on sm_89, nvcc 12.8 (cuda/mufu_probe.cu)",
        primitive_semantics=("ex2: T[j] = hw(j/2^23) for the 2^23 fraction indices, result T[j] * 2^n with K = floor(|x| 2^23), x >= 0: n = K >> 23, j = K & (2^23-1); x < 0: r == 0 ? "
                             "(n = -q, j = 0) : (n = -(q+1), j = 2^23-1-r) [ones' complement of the truncated magnitude, carry only when the fraction bits are zero]; ftz in/out; "
                             "rcp: R[m] = hw(1.m), exponent exact, ftz"),
        decomposition="direct SFU instruction", interface={"input_bytes": 4, "output_bits": 32, "range": None}, origin="derived", binding={"present": 0, "unresolved": 0},
        checker="independent", natural_checks={"exact": 0, "attempted": 0},
        stress_checks={"exact": 2 * mufu["inputs"] - mufu["ex2_mismatches"] - mufu["rcp_mismatches"], "attempted": 2 * mufu["inputs"],
                       "domain": "all 2^32 fp32 bit patterns for each instruction (inf/NaN/subnormal included)"},
        mutations={"rejected": 0, "attempted": 0}, checker_work={"seconds": 0.0, "flops_or_bytes": 1.0, "includes_full_op": True},
        status="exact" if mufu["ex2_mismatches"] == 0 and mufu["rcp_mismatches"] == 0 else "mismatch",
        limitation=("background characterisation (contract 15.9), not a production relation; the tables are fixture data "
                    "(fixtures/W11-40f0cebeb670-20260907T1800Z/mufu, int8 deltas, sha256 in the fixture README); the naive 'truncate the fraction' index rule mismatches on "
                    "925,796,320 negative inputs (raw/mufu_verify_frac0 in the lane log), the ones'-complement rule on 0; NaN in -> 0x7fffffff; per-architecture: measured on sm_89 only"))))

    (results / "table2.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    # ---- Table 4: checker cost of the attention relation on this fixture
    t4 = dict(common, partition_id="attention-independent-model (fixture, no replay)",
              instrumentation_mode="fixture capture: rebind vllm.v1.attention.backends.flash_attn.flash_attn_varlen_func in the eager engine (no trace replay, no commitments)",
              production_overhead_ratio=None, replay_commit_overhead_ratio=None, replay_vs_production_gpu_work=None,
              transparent_check_cost={"cpu_s": float(sec.sum() + sum(c["seconds"] for c in ppc)), "gpu_s": 0.0},
              peak_replay_memory={"device_gb": None, "host_gb": None, "delta_vs_reference": None},
              committed_logical_bytes=None, processed_bytes={"hashed": None, "copied": None, "retained": None, "emitted": None},
              description_size_bytes=None, constructor_inputs_bytes=None, logical_size={"gates": None, "vus": None, "rus": None},
              lookup_cost_s=None, validation_cost_s=float(check_all["checker_seconds_total"] + check_stress["checker_seconds_total"]),
              moe_padding={"inactive_units": None, "inactive_checks": None, "inactive_work_s": None, "inactive_bytes": None},
              restrictions=[
                  f"transparent_check_cost.cpu_s = single-thread CPU seconds of attention.bf16.v1 on {int(nrow.sum())} decode (row, head) outputs (189 calls, every 7th decode call) + "
                  f"{sum(c['rows'] for c in ppc)} prefill (row, head) outputs (3 calls) = {dec_cost * 1e3:.2f} ms per decode output row (12 heads, mean Lk {nat_dec_lk:.0f}), "
                  f"{pre_cost * 1e3:.2f} ms per prefill output row (mean t+1 {pre_vis:.0f}); AMD EPYC 7282, g++ -O2",
                  f"validation_cost_s = wall seconds of the full natural + stress checks with OpenMP (64 threads): {check_all['stats']['all=all']['outputs']} + "
                  f"{check_stress['stats']['all=all']['outputs']} output words",
                  "cost scales linearly in the context length (one mma.sync chain per (key, head-dim k-step) and one per (key, output d)); per token of Qwen2.5-1.5B (28 layers) at Lk 2048: "
                  f"~{28 * H * (coef[0] * 2048 + coef[1]) * 1e3:.0f} ms single-thread",
                  "no production overhead: the operands are what the eager engine already holds (q, paged K/V, out); the fixture capture copied them after cuda.synchronize",
              ],
              spend={"gpu_hours": None, "usd": None})
    (results / "table4.jsonl").write_text(json.dumps(ROWS.require_valid(4, t4), sort_keys=True) + "\n")

    # ---- dataflow variants (hypothesis tests on the natural fixture, all 4,317,696 words), measured BEFORE the FFMA fix
    variants = {}
    for v in (0, 1, 2, 4, 8, 16, 32, 64, 128):
        r = _load(raw / f"var_{v}.json")["stats"]["all=all"]
        variants[str(v)] = {"mismatch_words": r["mismatch"], "outputs": r["outputs"]}
    nudges = {str(n): _load(raw / f"nudge_{n}.json")["stats"]["all=all"]["mismatch"] for n in (-1, 1)}
    (results / "dataflow_variants.json").write_text(json.dumps({
        "note": ("attn_check.py --variant sweep on the natural fixture, run with the PRE-FIX bit assignment: base (0) = source-literal row-sum rescale "
                 "(FMUL then FADD), bit 128 = the FFMA contraction.  After the fix the meanings of 0 and 128 are swapped in fa2_model.cpp (0 = compiled FFMA, "
                 "the production dataflow).  Bits 1-64 were measured on top of the source-literal base (4 baseline mismatches)."),
        "bits": {"1": "lane-sum order nj-fastest", "2": "PV k-steps reversed", "4": "QK k-steps reversed", "8": "single sequential row sum (no lanes)",
                 "16": "row sum over BF16-rounded P", "32": "FMUL+FSUB instead of FFMA for the exp2 argument", "64": "blocks ascending",
                 "128": "(pre-fix) FFMA contraction of the row-sum rescale = the production dataflow"},
        "mismatching_words_of_4317696": variants,
        "scale_log2_nudged_by_ulps": nudges,
        "production_sass": sass,
    }, indent=1))

    # ---- manifest
    man = {
        "schema": "verity-vllm/numerics-manifest/v1",
        "experiment_id": EXP, "worker": "W11", "case": "B1", "revision": common["adapter_revision"],
        "gpu": {"name": nat_idx["gpu"], "sm": "sm_89", "driver": "580.65.06 (RunPod RTX 4090, container bdccd52933e7)", "cuda_toolkit": "12.8 (nvcc 12.8.93)", "cuda_runtime": nat_idx["cuda"]},
        "software": {"torch": nat_idx["torch"], "vllm_pinned": nat_idx["vllm"], "python": "3.11.11 (/workspace/venv)",
                     "vllm_flash_attn_so_sha256": nat_idx["vllm_flash_attn_so_sha256"],
                     "flash_attention_source": "github.com/vllm-project/flash-attention @ 506341a143fcabd4bb79052a7605ada727d6b3f5 (vllm cmake/external_projects/vllm_flash_attn.cmake GIT_TAG), csrc/flash_attn, built with --use_fast_math"},
        "checkpoint": {"repo": nat_idx["model"], "revision": nat_idx["model_local_path"].rsplit("/", 1)[-1]},
        "profiles": {PROFILE: ("RTX 4090 sm_89, the B1 profile's engine configuration (dtype bf16, max_model_len 4096, enforce_eager, prefix caching off, VLLM_BATCH_INVARIANT=1 -> "
                               "FlashAttention-2 with num_splits=1) re-run on this lane's pod for fixture capture; not W1's B1 bundle itself")},
        "engine_flags_observed": nat_idx["env"], "engine_kwargs": nat_idx["engine_kwargs"],
        "kernel": FA.KERNEL, "kernel_consts": nat_idx["consts"],
        "fixtures": {"dir": f"fixtures/{EXP}", "sha256sums": f"fixtures/{EXP}/SHA256SUMS", "natural_calls": nat_idx["n_calls"], "stress_calls": st_idx["n_calls"], "mutations": st_idx["n_mutations"]},
        "results": {"natural": check_all["stats"]["all=all"], "stress": check_stress["stats"]["all=all"], "stress_unresolved": check_stress["unresolved"],
                    "mutations": muts["counts"], "stage_natural": stage_nat["stages"], "stage_stress": stage_st["stages"],
                    "mufu_exhaustive": {k: mufu[k] for k in ("inputs", "frac_mode", "ex2_mismatches", "rcp_mismatches")}},
        "raw": sorted(p.name for p in raw.iterdir()),
        "docs": "docs/vllm-poc/attention-model.md",
        "code": ["verity_vllm_numerics/fa2_model.py", "verity_vllm_numerics/cpp/fa2_model.cpp", "verity_vllm_numerics/fa2_relation.py", "verity_vllm_numerics/attn_capture.py",
                 "verity_vllm_numerics/attn_check.py", "verity_vllm_numerics/attn_stress.py", "verity_vllm_numerics/attn_mutation_check.py", "verity_vllm_numerics/attn_stage_gpu.py",
                 "verity_vllm_numerics/attn_diagnose.py", "verity_vllm_numerics/cuda/mufu_probe.cu", "verity_vllm_numerics/cuda/fa2_softmax_probe.cu", "tests/test_attention_model.py"],
    }
    (results / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {len(rows)} table2 rows, 1 table4 row, manifest, dataflow_variants -> {results}")
    for r in rows:
        print(f"  {r['op_id']:32s} {r['status']:8s} natural {r['natural_checks']['exact']}/{r['natural_checks']['attempted']}  stress {r['stress_checks']['exact']}/{r['stress_checks']['attempted']}  "
              f"mut {r['mutations']['rejected']}/{r['mutations']['attempted']}  cost/row {r['checker_work']['seconds']:.4f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else POC / "results" / EXP)
