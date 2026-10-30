"""The device a served Build vouches for: the bindings its compute capability registers, and each attention launch's Definitions there.

A Build vouches for its own device only.  What the served Program binds is the device's own:

  * every cuBLAS GEMM's k16 step: its GEMM target's (`frontend.gemm_targets.gemm_target_for`), `HopperBF16WgmmaDot16_v1` on cc 9.0
    (wgmma), on cc 10.0 (tcgen05.mma kind::f16, silicon `sm100.tcgen05.mma.bf16`) and on cc 12.0 (mma.sync m16n8k16, silicon
    `sm120.mma.m16n8k16.bf16`): the same pipeline;
  * the attention kernel, FLASH_ATTN at the version `ATTENTION` names for the step family: vLLM's selection at its defaults on cc 9.0
    (FA3, `fa3_launch`) and cc 12.0 (FA2, `fa2_launch`); on cc 10.0, where vLLM's default is FLASHINFER, the workload declares FLASH_ATTN
    FA2 (`engine.args`, with its `engine.reasons`), whose mma.sync m16n8k16 there is the same pipeline (`sm100.mma.m16n8k16.bf16`);
  * the Triton kernels' compile target (`inductor_templates.FLAGS`).

A device without one of them is refused with every missing binding by name (`device_of`), never bound to another device's.  A served
target is held to its card file first (`card_of`: `engine.profiles.devices.CARD`, `verity.hardware`): a device whose compute
capability or SM count is not its card's is refused, naming each field, both values and the card file.  Only a target
`engine.profiles.devices.NO_CARD` lists is held to none, and any other is refused.

FA2 (vllm-flash-attn 506341a1 `mha_varlen_fwd`, at the `num_splits = 0` vLLM's FA2 metadata passes) runs a launch whose every
sequence has one query row as a non-causal swapped decode (a GQA model's query heads as rows), its split count `fa2_num_splits` at the
launch's batch and max_seqlen_k: under a FULL graph the capture's (the padded batch, `max_model_len`, vLLM's capture metadata), the
step's own otherwise.  A count of 1 runs the align kernel (`Attention_v5`, kBlockN `fa2_align_kblock_n`), more runs
`Fa2AttentionSplit_v1` (kBlockN `fa2_split_kblock_n`); either guards each split's first block only.  Any other launch runs the split
kernel's tile unsplit (`Attention_v5`, kBlockN `fa2_split_kblock_n`, FA2's causal masking steps `attention_vocab.fa2_masked_from`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from verity_vllm.acquire import inductor_templates as T
from verity_vllm.acquire.inductor_templates import Unmodelled, bound
from verity_vllm.engine.profiles.devices import CARD, NO_CARD
from verity_vllm.program.frontend import targets as TG
from verity_vllm.program.frontend.gemm_targets import UnregisteredGemmTarget, gemm_target_for
from verity.hardware import load
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics import fa3_model as FA3
from verity.ml.definitions.registry import attention_vocab
from verity.ml.definitions.registry import fa2_check_inf as F2
from verity.ml.definitions.registry import fa2_split as W
from verity.ml.definitions.registry import fa3_check_inf as F3
from verity.ml.definitions.registry import inductor as IR
from verity.ml.definitions.registry.hopper import Fa3InvSum
from verity.ml.definitions.registry.prims import Fa2InvSum

#: the FLASH_ATTN version each step family's served attention Definitions model
ATTENTION = {"hopper": 3, "blackwell_consumer": 2, "blackwell": 2}
BACKEND = "FLASH_ATTN"


@dataclass(frozen=True)
class DeviceBindings:
    """A served Build's device bindings: its compute capability and SM count, its GEMMs' k16 step, its attention kernel
    ("FA3" | "FA2") and, under FA2, the max_model_len a FULL graph's split count reads."""
    cc: tuple[int, int]
    sms: int
    dot: Any
    attention: str
    max_model_len: int | None = None

    def dispatch(self, capture_sizes: list[int], max_num_seqs: int, split_cap: int) -> dict[str, Any]:
        """The Program's dispatch record (cc 9.0's keeps the fields its Builds of record were digested with)."""
        if self.attention == "FA3":
            return {"capture_sizes": capture_sizes, "max_num_seqs": max_num_seqs, "split_cap": split_cap, "sms": self.sms}
        return {"capture_sizes": capture_sizes, "max_num_seqs": max_num_seqs, "sms": self.sms, "cc": list(self.cc),
                "attention": self.attention, "max_model_len": self.max_model_len}


def device_of(cc: Any, sms: int, *, card: str | None = None, attention: list[dict[str, Any]] | None = None,
              max_model_len: int | None = None) -> DeviceBindings:
    """The device bindings of compute capability `cc`; `card` the SKU of the card the served target is registered on (`card_of`), whose
    compute capability and SM count the device must have; `attention` the attention kernels
    the serving engine selected (`served_match.facts_of`: [{"backend", "fa_version"}]), held to the device's when a process reports
    them.  Raises `Unmodelled` naming each field that is not the card's, else every binding the device lacks."""
    cc = (int(cc[0]), int(cc[1]))
    if card is not None:
        check_card(cc, int(sms), card)
    missing, dot, fam = [], None, None
    try:
        gt = gemm_target_for(cc, batch_invariant=False)
        fam, dot = gt.key, TG.DOTS[gt.dot]
    except UnregisteredGemmTarget:
        missing.append(f"the cuBLAS GEMMs' k16 step (no registered GEMM target at cc {cc[0]}.{cc[1]})")
    want = ATTENTION.get(fam)
    got = [(a.get("backend"), a.get("fa_version")) for a in attention] if attention is not None else [(BACKEND, want)]
    if want is None or got != [(BACKEND, want)]:
        missing.append(f"attention {', '.join(f'{b} (FA{v})' if v else str(b) for b, v in got)}: the served attention Definitions are "
                       f"{', '.join(f'{BACKEND} FA{v} on {f}' for f, v in ATTENTION.items())}")
    if cc not in T.FLAGS:
        missing.append(f"Inductor's Triton kernels for cc={cc[0]}{cc[1]} (inductor_templates.FLAGS: cc {sorted(T.FLAGS)})")
    if want == 2 and max_model_len is None:
        missing.append("max_model_len (FA2's full-graph split count reads it)")
    if missing:
        raise Unmodelled(f"cc {cc[0]}.{cc[1]}: no binding for " + "; ".join(missing))
    return DeviceBindings(cc, int(sms), dot, f"FA{want}", max_model_len)


def card_of(name: str) -> str | None:
    """The SKU of the card the served target `name` (the name its device reports) is registered on; None for a target `NO_CARD`
    lists.  Any other name is refused."""
    if name in CARD:
        return CARD[name]
    if name in NO_CARD:
        return None
    raise Unmodelled(f"device {name!r} is in neither engine.profiles.devices.CARD ({sorted(CARD)}) nor NO_CARD ({sorted(NO_CARD)}): "
                     "add its card under verity/hardware/cards/ and map it in CARD")


def check_card(cc: tuple[int, int], sms: int, sku: str) -> None:
    """Refuse a device whose compute capability or SM count is not that of card `sku` (its `arch` and `sm.count`)."""
    c = load(sku)
    arch = re.fullmatch(r"sm_(\d+)(\d)[af]?", c["arch"])
    if arch is None:
        raise Unmodelled(f"verity/hardware/cards/{sku}.toml: arch {c['arch']!r} names no compute capability")
    fields = (("compute capability", f"{cc[0]}.{cc[1]}", f"{arch[1]}.{arch[2]}"),
              ("SM count", str(sms), str(c["sm"]["count"]["value"])))
    off = [f"{name} {got}, the card's {want}" for name, got, want in fields if got != want]
    if off:
        raise Unmodelled(f"the device is not its registered card verity/hardware/cards/{sku}.toml: " + "; ".join(off))


# ------------------------------------------------------------------------------------------------------------- FA3 (cc 9.0)
def fa3_splits(qlens: list[int], klens: list[int], *, NH: int, KVH: int, D: int, split_cap: int, sms: int) -> dict[str, Any]:
    """FA3's tile and per-request split count for one varlen launch (vllm-flash-attn 506341a1: `tile_size_fwd_sm90`, the
    static `num_splits_heuristic` (or the static cap a full graph fixes) and `prepare_varlen_num_blocks`' dynamic split,
    blocks per split = ceil(total m x n blocks x 1.1 x KVH / SMs) in f32 with div.full)."""
    g = NH // KVH
    max_q, max_k = max(qlens), max(klens)
    BM, BN = FA3.modelled_tile(D, max_q, g)
    nmb = [-(-(q * g) // BM) for q in qlens]
    nnb = [-(-k // BN) for k in klens]
    if split_cap > 0:
        static = split_cap
    else:
        nmb_s, nnb_s = -(-(max_q * g) // BM), -(-max_k // BN)
        static = FA3.num_splits_heuristic(KVH * nmb_s, sms, nnb_s, nmb_s, max_k * 2 * D * 2, True, 128)
    if static <= 1:
        return {"BM": BM, "BN": BN, "static": static, "NS": [1] * len(qlens)}
    f = np.float32
    total = sum(m * n for m, n in zip(nmb, nnb))
    a = np.array([f(f(f(total) * f(1.1)) * f(KVH))], f).view(np.uint32)
    b = np.array([f(sms)], f).view(np.uint32)
    bps = int(np.ceil(DR.as_f32(DR.mul_ftz(DR.div_full_scale_a(a, b), DR.div_full_rcp(b)))[0]))
    return {"BM": BM, "BN": BN, "static": static, "NS": [max(min(-(-n // bps), static), 1) for n in nnb]}


def attention_rows(qlen: int, klen: int, NS: int, *, NH: int, KVH: int, D: int, BM: int, BN: int, dot: Any) -> list[dict[str, Any]]:
    """The FA3 Definition of each of a request's query rows in one launch."""
    g = NH // KVH
    if NS == 1:
        return [bound(F3.AttentionV4, T=T_, NH=NH, KVH=KVH, D=D, BN=BN, DOT=dot, INV=Fa3InvSum,
                      MASKED_FROM=attention_vocab.fa3_masked_from(T_, BN, i, BM, g))
                for i, T_ in enumerate(range(klen - qlen + 1, klen + 1))]
    if qlen != 1:
        raise Unmodelled(f"FA3 splits a prefill chunk of {qlen} rows {NS} ways: a split prefill is not modelled")
    return [bound(IR.Fa3AttentionSplit, T=klen, NS=NS, NH=NH, KVH=KVH, D=D, BN=BN, DOT=dot, INV=Fa3InvSum,
                  MASKED_FROM=attention_vocab.fa3_masked_from(klen, BN, 0, BM, g))]


# ---------------------------------------------------------------------------------------------------- FA2 (cc 10.0 and 12.0)
def fa2_launch(qlens: list[int], *, NH: int, KVH: int, D: int, sms: int, batch: int, max_seqlen_k: int) -> dict[str, Any]:
    """FA2's launch of one varlen step (module docstring): swapped or not, its kBlockN and its split count."""
    if max(qlens) == 1 and NH > KVH and D % 8 == 0:
        ns = W.fa2_num_splits(batch, KVH, NH // KVH, max_seqlen_k, D, sms)
        return {"swapped": True, "BN": W.fa2_align_kblock_n(D) if ns == 1 else W.fa2_split_kblock_n(D), "NS": ns,
                "batch": batch, "max_seqlen_k": max_seqlen_k}
    return {"swapped": False, "BN": W.fa2_split_kblock_n(D), "NS": 1, "causal": max(qlens) > 1}


def fa2_attention_rows(qlen: int, klen: int, launch: dict[str, Any], *, NH: int, KVH: int, D: int, dot: Any) -> list[dict[str, Any]]:
    """The FA2 Definition of each of a request's query rows in `launch` (`fa2_launch`)."""
    BN, NS = launch["BN"], launch["NS"]
    if not launch["swapped"]:
        return [bound(F2.AttentionV5, T=T_, NH=NH, KVH=KVH, D=D, BN=BN, DOT=dot, INV=Fa2InvSum,
                      MASKED_FROM=attention_vocab.fa2_masked_from(klen, qlen, i, BN, launch["causal"]))
                for i, T_ in enumerate(range(klen - qlen + 1, klen + 1))]
    mf = attention_vocab.fa2_masked_from(klen, 1, 0, BN, False)
    if NS == 1:
        return [bound(F2.AttentionV5, T=klen, NH=NH, KVH=KVH, D=D, BN=BN, DOT=dot, INV=Fa2InvSum, MASKED_FROM=mf)]
    return [bound(W.Fa2AttentionSplit, T=klen, NS=NS, NH=NH, KVH=KVH, D=D, BN=BN, DOT=dot, INV=Fa2InvSum, MASKED_FROM=mf)]


def launch(device: DeviceBindings, qlens: list[int], klens: list[int], heads: tuple[int, int, int], *, mode: str, padded: int, split_cap: int,
           capture_sizes: list[int]) -> tuple[list[int], list[list[dict[str, Any]]], dict[str, Any]]:
    """One step's attention launch on `device`: each request's split count and row Definitions, and the plan's record of the launch
    (its key "fa3" or "fa2")."""
    NH, KVH, D = heads
    if device.attention == "FA3":
        sp = fa3_splits(qlens, klens, NH=NH, KVH=KVH, D=D, split_cap=split_cap if capture_sizes and sum(qlens) <= max(capture_sizes) else 0,
                        sms=device.sms)
        rows = [attention_rows(q, k, ns, NH=NH, KVH=KVH, D=D, BM=sp["BM"], BN=sp["BN"], dot=device.dot)
                for q, k, ns in zip(qlens, klens, sp["NS"])]
        return sp["NS"], rows, {"fa3": {"BM": sp["BM"], "BN": sp["BN"], "static_splits": sp["static"]}}
    full = mode == "FULL"
    la = fa2_launch(qlens, NH=NH, KVH=KVH, D=D, sms=device.sms, batch=padded if full else len(qlens),
                    max_seqlen_k=device.max_model_len if full else max(klens))
    rows = [fa2_attention_rows(q, k, la, NH=NH, KVH=KVH, D=D, dot=device.dot) for q, k in zip(qlens, klens)]
    return [la["NS"]] * len(qlens), rows, {"fa2": la}
