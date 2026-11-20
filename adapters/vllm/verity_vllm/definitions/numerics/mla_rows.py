"""The replay row kernels of FlashMLA's sparse FP8 decode, its prefill and the prefill's workspace gather (`flashmla_twin`,
`flashmla_prefill_twin`) and of the q / kv norm over a masked block (`dsa_norm_rows.block_rmsnorm_rows`), one per Definition family,
each the numpy twin its Definition is held to.  A row is one token: the twins' leading token axis is 1, and each Tuple's fields are
ports "0", "1", ... in the Definition's return order.  A FlashMLA row whose DOT is not the Hopper k16 step the twins model is
declined by name.  The row kernels register from `rows` (kernel "rows").
"""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

import numpy as np

from verity_vllm.definitions.evidence import EVIDENCE
from verity_vllm.definitions.numerics.ir_rows import f64_statics
from verity_vllm.definitions.numerics.kernel_registry import Declined, prim_static

EVALUATORS = {
    "FlashMlaDecode_v1": "verity_vllm.definitions.numerics.flashmla_twin.decode (FlashMlaDecode_v1{NH,SPLITS,NW,DN,DR,QT,BN,DOT,SCALE}: "
                         "each row's FlashMlaRowValid_v1 and FlashMlaKvRow_v1, then per head FlashMlaSplit_v1 over each split's 64-row "
                         "blocks and FlashMlaCombine_v1 with the sink; DOT the Hopper k16 step) -> NH x (DN + DR) bf16",
    "FlashMlaPrefill_v1": "verity_vllm.definitions.numerics.flashmla_prefill_twin.prefill (FlashMlaPrefill_v1{NH,NP,D,BN,DOT,SCALE}: each "
                          "row's FlashMlaPrefillValid_v1 and masking, then per head FlashMlaPrefillHead_v1's block pairs and the sink; "
                          "DOT the Hopper k16 step) -> NH x D bf16",
    "FlashMlaGatherRow_v1": "verity_vllm.definitions.numerics.flashmla_prefill_twin.gather_rows (FlashMlaGatherRow_v1{DN,DR,QT}: "
                            "Bf16MulBf16(F2fpBf16(E4m3ToF32(nope[d])), E8m0ShlBf16(scale[d / QT])), then the RoPE words) -> DN + DR bf16",
    "TritonBlockRMSNorm_v1": "verity_vllm.definitions.numerics.dsa_norm_rows.block_rmsnorm_rows (TritonBlockRMSNorm_v1{N,BLOCK,WARPS,EPS}: per "
                             "thread of 32 WARPS an FMA chain over its dsa_norm.layout registers of the BLOCK, the elements from N +0, "
                             "each warp's xor-butterfly, the warp sums' halving tree, r = RsqrtApprox(fma(DivFullScaleA(sum, N), "
                             "DivFullRcp(N), eps)), y = F2fpBf16((r * x) * w)) -> N bf16",
}

_FLASHMLA_IDS = EVIDENCE["dsv4_flashmla_capture"]
_FLASHMLA_PROVENANCE = {
    "property": "dsv4_flashmla_capture (verity-vllm/dsv4-flashmla-capture/v1), tests/properties/dsv4_flashmla_capture_gpu.py",
    "run_record": _FLASHMLA_IDS["run_record"],
    "art": _FLASHMLA_IDS["art"],
    "device": "cc 9.0 with 132 SMs (the capture's gate), vLLM d9105ea80 vendoring FlashMLA 6bc49418",
    "captured_utc": _FLASHMLA_IDS["captured_utc"],
    "cases": "decode swa, c4, c128, lone, batch and edge at the token's SPLITS; every gathered workspace row; the prefill cases at "
             "the token's NP (16 heads padded to 64, the 128-token window, the sink)",
    "tests": ["tests/program/test_flashmla.py, tests/program/test_flashmla_prefill.py (the twins against registry.flashmla and "
              "registry.flashmla_prefill)",
              "tests/program/test_kernel_self_check.py (each row kernel against its Definition)",
              "tests/check/test_sampled_replay_mla_rows.py (each row kernel on spec-text statics against its Definition)"],
    "definitions": "adapters/vllm/verity_vllm/definitions/registry/flashmla.py, flashmla_prefill.py (the reference; the twins are checked against them, "
                   "never the other way round)",
}

_Q_NORM_IDS = EVIDENCE["dsv4_q_norm_commit"]
_BLOCK_NORM_PROVENANCE = {
    "property": None,
    "run_record": _Q_NORM_IDS["run_record"],
    "art": None,
    "device": "cc 9.0, the eight devices of the row's attempt-3 Commit (the run above), vLLM d9105ea80, Triton 3.7.1",
    "captured_utc": _Q_NORM_IDS["captured_utc"],
    "cases": "the q norm (N 1536 of a 2048 block on 8 warps, eps 1e-6): 1,230 committed rows (308 distinct) on 8 ranks, read by "
             f"{_Q_NORM_IDS['rows']} (out/rows.npz) from the Commit's bundles and the checkpoint's q_norm weights, every committed "
             "output the Definition's; eps + DivFullScaleA(sum, N) * DivFullRcp(N) in two roundings misses 5 of the 308 by 1-4 "
             "bf16 words, which tests/properties/fixtures/dsv4_q_norm_commit_20261021.json holds with 4 controls",
    "sass": f"{_Q_NORM_IDS['sass']}: the served kernel's cc 9.0 SASS (Triton 3.7.1, 8 warps at BLOCK 2048) is MUFU.RCP of the "
            "scaled SIZE, FFMA rcp, scaled sum, eps, then MUFU.RSQ",
    "gap": "no probe launch of _fused_q_kv_rmsnorm_kernel: the reduction tree is the PTX's, pinned on the device only by committed "
           "rows; the kv norm (N 512, a power of two, so either epilogue) only by the re-judge's sampled kv rows (306 a rank, all "
           f"equal, {_Q_NORM_IDS['kv_rows'][0]} and {_Q_NORM_IDS['kv_rows'][1]})",
    "model": "tests/program/test_dsa_norm.py::test_block_definition_matches_the_ptx_model (the Definition against a model of that "
             "PTX and SASS at the served (N, BLOCK, WARPS) = (1536, 2048, 8) and (512, 2048, 8), and four other blocks)",
    "related": f"glm53_fused_rmsnorm_capture ({EVIDENCE['glm53_fused_rmsnorm_capture']['art']}): the same "
               "tl.sum lowering at the unmasked four-warp block (DsaRMSNorm_v1, TritonBlockRMSNorm_v1{N, N, 4, EPS}) in GLM-5.3's "
               "_fused_norm_rope_kernel",
    "tests": ["tests/properties/test_dsv4_q_norm_commit.py (the Definition and the row kernel against the committed rows)",
              "tests/program/test_dsa_norm.py (the Definition against the PTX model)",
              "tests/program/test_kernel_self_check.py (each row kernel against its Definition)",
              "tests/check/test_sampled_replay_mla_rows.py (each row kernel on spec-text statics against its Definition)"],
    "definitions": "adapters/vllm/verity_vllm/definitions/registry/dsa_norm.py (the reference; the twin is checked against it, never the other way round)",
}

#: family -> what pinned its twin to the H200
PROVENANCE = {
    **{f: _FLASHMLA_PROVENANCE for f in ("FlashMlaDecode_v1", "FlashMlaPrefill_v1", "FlashMlaGatherRow_v1")},
    "TritonBlockRMSNorm_v1": _BLOCK_NORM_PROVENANCE,
}


def _words(a: np.ndarray, n: int, dtype) -> np.ndarray:
    a = np.asarray(a)
    if a.size != n:
        raise Declined(f"an operand of {a.size} words, the signature has {n}")
    return a.astype(dtype, copy=False).reshape(n)


def _bf16(a: np.ndarray, *shape: int) -> np.ndarray:
    return _words(a, int(np.prod(shape)), np.uint16).reshape(shape)


def _f32(a: np.ndarray, *shape: int) -> np.ndarray:
    return _words(a, int(np.prod(shape)), np.uint32).reshape(shape)


def _ports(*arrays: np.ndarray) -> dict[str, np.ndarray]:
    return {str(j): np.asarray(a).reshape(-1) for j, a in enumerate(arrays)}


def _i32(a: np.ndarray, n: int) -> np.ndarray:
    """n committed i32 words, whether they arrive as int32 or as their u32 bits, as u32 bits."""
    return (_words(a, n, np.int64) & 0xFFFFFFFF).astype(np.uint32)


def _hopper_dot(family: str, statics: dict[str, Any]) -> None:
    from verity_vllm.definitions.registry.hopper import HopperBF16WgmmaDot16
    dot = prim_static(statics, "DOT", family)
    if dot != HopperBF16WgmmaDot16.id:
        raise Declined(f"{family}: no twin for DOT={dot} (the twins model {HopperBF16WgmmaDot16.id})")


def _splits(v: Any) -> tuple[int, ...]:
    """SPLITS as the spec text spells it (`[0,1,2,3,4]`) or as a binding holds it."""
    return tuple(int(s) for s in (json.loads(v) if isinstance(v, str) else v))


def flashmla_decode(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`FlashMlaDecode_v1{NH,SPLITS,NW,DN,DR,QT,BN,DOT,SCALE}(q, nope, scale, rope, idx, swa_len[, extra_len], sink)` -> NH x (DN +
    DR) bf16 words, head-major."""
    from verity_vllm.definitions.numerics import flashmla_twin as FT
    f = "FlashMlaDecode_v1"
    _hopper_dot(f, statics)
    S = f64_statics(statics)
    NH, NW, DN, DR, QT, BN = (int(S[k]) for k in ("NH", "NW", "DN", "DR", "QT", "BN"))
    sp = _splits(S["SPLITS"])
    N = sp[-1] * BN
    lens = 2 if N > NW else 1
    if len(ops) != 6 + lens:
        raise Declined(f"{f}: the row wires {len(ops)} args, the Definition takes {6 + lens}")
    out = FT.decode(_bf16(ops[0], NH, DN + DR), _words(ops[1], N * DN, np.uint8).reshape(N, DN),
                    _words(ops[2], N * (DN // QT), np.uint8).reshape(N, DN // QT), _bf16(ops[3], N, DR), _i32(ops[4], N),
                    [_i32(ops[5 + j], 1) for j in range(lens)], _f32(ops[5 + lens], NH), SPLITS=sp, NW=NW, QT=QT, BN=BN,
                    SCALE=float(S["SCALE"]))
    return _ports(np.asarray(out).astype(np.uint16))


def flashmla_prefill(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`FlashMlaPrefill_v1{NH,NP,D,BN,DOT,SCALE}(q, kv, idx, len, skv, sink)` -> NH x D bf16 words, head-major."""
    from verity_vllm.definitions.numerics import flashmla_prefill_twin as FP
    _hopper_dot("FlashMlaPrefill_v1", statics)
    S = f64_statics(statics)
    NH, NP, D, BN = (int(S[k]) for k in ("NH", "NP", "D", "BN"))
    N = 2 * NP * BN
    out = FP.prefill(_bf16(ops[0], NH, D), _bf16(ops[1], N, D), _i32(ops[2], N), _i32(ops[3], 1), _i32(ops[4], 1), _f32(ops[5], NH),
                     BN=BN, SCALE=float(S["SCALE"]))
    return _ports(np.asarray(out).astype(np.uint16))


def flashmla_gather_row(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`FlashMlaGatherRow_v1{DN,DR,QT}(nope, scale, rope)` -> DN + DR bf16 words."""
    from verity_vllm.definitions.numerics import flashmla_prefill_twin as FP
    DN, DR, QT = (int(statics[k]) for k in ("DN", "DR", "QT"))
    out = FP.gather_rows(_words(ops[0], DN, np.uint8).reshape(1, DN), _words(ops[1], DN // QT, np.uint8).reshape(1, DN // QT),
                         _bf16(ops[2], 1, DR), QT)
    return _ports(np.asarray(out).astype(np.uint16))


def triton_block_rmsnorm(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`TritonBlockRMSNorm_v1{N,BLOCK,WARPS,EPS}(x, w)` -> N bf16 words."""
    from verity_vllm.definitions.numerics.dsa_norm_rows import block_rmsnorm_rows
    S = f64_statics(statics)
    N = int(S["N"])
    r = block_rmsnorm_rows(_bf16(ops[0], 1, N), _bf16(ops[1], N), float(S["EPS"]), int(S["BLOCK"]), int(S["WARPS"]))
    return _ports(r["out"].astype(np.uint16))


#: family -> (row kernel, the ports it returns in the Definition's return order)
ROWS: dict[str, tuple[Callable[..., dict[str, np.ndarray]], tuple[str, ...]]] = {
    "FlashMlaDecode_v1": (flashmla_decode, ("0",)),
    "FlashMlaPrefill_v1": (flashmla_prefill, ("0",)),
    "FlashMlaGatherRow_v1": (flashmla_gather_row, ("0",)),
    "TritonBlockRMSNorm_v1": (triton_block_rmsnorm, ("0",)),
}


def _sample_flashmla(family: str) -> Callable[..., list[np.ndarray]]:
    from verity_vllm.definitions.numerics import flashmla_prefill_twin as FP
    from verity_vllm.definitions.numerics import flashmla_twin as FT
    return {"FlashMlaDecode_v1": FT._sample_decode, "FlashMlaPrefill_v1": FP._sample_prefill,
            "FlashMlaGatherRow_v1": FP._sample_gather}[family]


SAMPLES: dict[str, Callable[[], Callable[..., list[np.ndarray]]]] = {
    **{f: (lambda _f=f: _sample_flashmla(_f)) for f in ("FlashMlaDecode_v1", "FlashMlaPrefill_v1", "FlashMlaGatherRow_v1")},
}


def check_targets(family: str) -> list:
    """The small specialisations `family`'s self-check binds (the served statics are too wide for the reference)."""
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import dsa_norm as DN
    from verity_vllm.definitions.registry import flashmla as FM
    from verity_vllm.definitions.registry import flashmla_prefill as FP
    from verity_vllm.definitions.registry.hopper import HopperBF16WgmmaDot16
    eps = 1e-6
    attn = dict(DOT=HopperBF16WgmmaDot16, SCALE=32 ** -0.5)
    decode = dict(NH=2, DN=16, DR=16, QT=16, BN=16, **attn)
    return {
        "FlashMlaDecode_v1": [bind(FM.FlashMlaDecode, SPLITS=(0, 1, 2), NW=16, **decode),
                              bind(FM.FlashMlaDecode, SPLITS=(0, 2), NW=32, **decode)],
        "FlashMlaPrefill_v1": [bind(FP.FlashMlaPrefill, NH=2, NP=1, D=32, BN=16, **attn),
                               bind(FP.FlashMlaPrefill, NH=1, NP=2, D=32, BN=16, **attn)],
        "FlashMlaGatherRow_v1": [bind(FP.FlashMlaGatherRow, DN=32, DR=16, QT=16)],
        "TritonBlockRMSNorm_v1": [bind(DN.TritonBlockRMSNorm, N=n, BLOCK=b, WARPS=w, EPS=eps)
                                  for n, b, w in ((48, 64, 1), (136, 256, 2), (264, 512, 1), (512, 2048, 8))],
    }[family]


def register_rows(evaluators: dict, row: Callable, provenance: dict) -> None:
    """Register this module's row evaluators without growing the shared rows catalog."""
    evaluators |= EVALUATORS
    provenance |= PROVENANCE
    for family, (fn, returns) in ROWS.items():
        sample = SAMPLES.get(family)
        row(family, fn, returns, sample=sample() if sample else None, check_targets=lambda rng, f=family: check_targets(f))


__all__ = ["EVALUATORS", "PROVENANCE", "ROWS", "check_targets", "register_rows"]
