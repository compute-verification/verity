"""GLM-5.3's dense MHA prefill on an H200: the FA3 kernel `run_prefill_new_tokens` launches (vllm-flash-attn 506341a1, qk head 256 =
192 nope + 64 rope, v head 256, 8 heads a TP8 rank, causal varlen self-attention) against `Attention_v4{T, NH=8, KVH=8, D=256, BN=80,
DOT=HopperBF16WgmmaDot16_v1, INV=Fa3InvSum_v1, MASKED_FROM}`, the row the target binds (`targets.attention_spec`), and, for a
sequence FA3 splits, against `AttentionSplit_v1{T, NBM, NS, ...}` (`fa3_split.split_of` of that binding).

One GPU process writes `<out>/glm53_fa3_prefill_capture.json`, the property record `glm53_fa3_prefill_capture` (`properties.record`).
The device gate runs first and nothing else runs when it fails (exit 3):

  gate     `fa2_target_capture_gpu.gate`: driver, compute capability 9.0, 132 SMs.
  select   vLLM's `get_flash_attn_version(head_size=256)` on this GPU is 3.
  mufu     `fa2_target_capture_gpu.mufu`: the ex2 / rcp tables (FA3's `--use_fast_math` exp2f and 1/x, the combine's too) on all
           2^32 inputs.
  launch   every launch's kernels (torch.profiler): one `FlashAttnFwdSm90` instantiation at TileShape 128x80x256, kHeadDimV 256,
           causal, varlen, not paged, MmaPV_is_RS, IntraWGOverlap, Split exactly when `fa3_model.split_plan` says the launch splits;
           then, exactly when it splits, one `FlashAttnFwdCombine` instantiation, `WANT_COMBINE` (`SparseMlaCombine_v1`'s: 8 x 128
           tiles, 2^5 = 32 split lanes, which holds the plan's split count; vllm-flash-attn's `run_mha_fwd_combine_` picks it for
           dv > 64 and num_splits <= 32), and, counted in the launch's profiler window (`kernel_counts`), one forward kernel a call and
           one combine a call exactly then; each launch run `CALLS` times, the words equal.
  rows     each case (one sequence at each of `LENGTHS`, the `VARLEN` batches, a `wide` batch with every operand word scaled by
           2^U(-8, 8)) at `num_splits = 1` (pinned) and at the served `num_splits = 0`.  q and k are contiguous [tokens, 8, 256] (k =
           k_nope ++ k_pe broadcast over heads), v the strided view of kv_b_proj's [tokens, 8, 192 + 256] output, as vLLM passes them.
           Every output row against the C++ twin: `fa3_model.attention_call` (kBlockN 80) for a sequence run as one split, and
           `fa3_split_twins.split_call` at the plan's split count for one the plan splits.  On `SAMPLE` rows also the vectorised
           twin (`derived_rows.attention_rows`, `fa3_split_twins.head_split_rows`), the reference evaluator on one head
           (`AttentionHead_v4`, `AttentionHeadSplit_v1`) and the target's binding; on a split row, which split counts up to the
           launch's give the row's words, every head (``ns_unique``: rows where only the plan's cut does; recorded, not a check:
           a split's combine and the unsplit epilogue mostly round to the same bf16 words).  A head the
           twin flags non-finite takes the reference evaluator's words; one whose split sum's MUFU.LG2 the model leaves open
           (`sparse_mla_split_twins.undetermined_words`) is unresolved (``lg2_open``).

GO = the gate, the selection, MUFU, every launch, and every row of the pinned launches, of the served sequences run as one split
and of the served sequences the plan splits (`classes`: pinned, served, served_split) equal to the twin, with every sampled check
and nothing unresolved.  MUFU.LG2 on sm_90 is measured by `glm53_fa3_split_capture_gpu.py` (GLM-SPLIT's); here it is read through
the combined words.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_fa3_prefill_capture_gpu.py --out DIR --scratch /tmp --want-cc 9.0 --want-sms 132 --min-driver 575
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-fa3-prefill-capture/v2"
RECORD = "glm53_fa3_prefill_capture.json"
NH, NOPE, ROPE, DV = 8, 192, 64, 256
D = NOPE + ROPE
SCALE = D ** -0.5
BM, BN, NUM_SMS = 128, 80, 132
CMAP = Path(__file__).resolve().parents[2] / "manifests" / "capture-maps" / "glm-moe-dsa-eager-hopper-tp8-v0.json"
LENGTHS = (1, 2, 79, 80, 81, 127, 128, 129, 160, 161, 255, 256, 257, 320, 321, 512, 1024, 1664, 1665, 2048)
VARLEN = {"varlen_mixed": [512, 512, 64], "varlen_long_short": [1000, 30, 30], "varlen_blocks": [1, 80, 81, 300],
          "varlen_2048": [2048, 1, 17]}
QUICK = ("len1", "len81", "len161", "len321", "len512", "varlen_mixed")
MODES = (("pinned", 1), ("served", 0))
CALLS = 2                       # calls a launch window makes, their words compared for determinism
#: rows (of their sequence) the vectorised twin, the reference evaluator and the binding also check: both sides of every 80-key
#: block and 128-row tile boundary the lengths reach, and each sequence's last row
SAMPLE = (0, 1, 79, 80, 81, 127, 128, 129, 159, 160, 161, 255, 256, 319, 320, 321, 511, 1023, 1663, 1664, 2047)
CHUNK = 256
TOTALS = ("rows", "heads", "mismatch", "nonfinite", "lg2_open", "reference", "twin_mismatch", "binding_mismatch", "sampled_rows",
          "ns_sampled", "ns_unique")
CLASSES = ("pinned", "served", "served_split")
FLAGS = ("Is_causal", "Is_local", "Has_softcap", "Varlen", "PagedKVNonTMA", "AppendKV", "HasQv", "MmaPV_is_RS", "IntraWGOverlap",
         "PackGQA", "Split", "V_colmajor")
WANT_FLAGS = {"Is_causal": True, "Is_local": False, "Has_softcap": False, "Varlen": True, "PagedKVNonTMA": False, "AppendKV": False,
              "HasQv": False, "MmaPV_is_RS": True, "IntraWGOverlap": True, "V_colmajor": False}
_TUPLE = r"cute::tuple<\s*cute::C<(\d+)>\s*,\s*cute::C<(\d+)>\s*,\s*cute::C<(\d+)>\s*>"
_FWD = re.compile(r"CollectiveMainloopFwdSm90<\s*(\d+)\s*,\s*" + _TUPLE + r"\s*,\s*" + _TUPLE + r"\s*,\s*(\d+)\s*,\s*cutlass::bfloat16_t\s*,"
                  r"\s*float\s*,\s*cutlass::arch::Sm90\s*,\s*((?:true|false)(?:\s*,\s*(?:true|false)){11})")
#: `FlashAttnFwdCombine<TileShape_MK, kLogMaxSplits, kNThreads, AlignmentLSE, Is_even_K, Varlen, Element, ElementPartial, ArchTag>`
_COMBINE = re.compile(r"FlashAttnFwdCombine<\s*cute::tuple<\s*cute::C<(\d+)>\s*,\s*cute::C<(\d+)>\s*>\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,"
                      r"\s*(true|false)\s*,\s*(true|false)\s*,\s*cutlass::bfloat16_t\s*,\s*float\s*,")
COMBINE_FIELDS = ("tile", "log_max_splits", "threads", "alignment_lse", "is_even_k", "varlen")
#: the combine `SparseMlaCombine_v1` models (hopper/flash_fwd_combine_launch_template.h at dv > 64: kBlockK 128, kBlockM 8, so at
#: least 2^5 split lanes, IsEvenK false; vLLM's varlen call), the one the H200 ran for GLM-SPLIT's sparse decode (r20261013-001656-57d6)
WANT_COMBINE = {"tile": [8, 128], "log_max_splits": 5, "threads": 256, "alignment_lse": 1, "is_even_k": False, "varlen": True}


def _error(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()[-600:] + " | " + traceback.format_exc(limit=-3)[-900:]


def case_list(quick: bool = False) -> list[dict[str, Any]]:
    out = [{"name": f"len{n}", "seqlens": [n]} for n in LENGTHS]
    out += [{"name": k, "seqlens": v} for k, v in VARLEN.items()]
    out.append({"name": "wide", "seqlens": [161, 300], "wide": True})
    return [c for c in out if c["name"] in QUICK] if quick else out


def plan(seqlens: list[int], num_splits: int) -> dict[str, Any]:
    """The launch's split count and each sequence's (`fa3_model.split_plan` for the served `num_splits = 0`; 1 for a pinned launch)."""
    from verity_vllm.program.kernels import fa3_model as FM

    if num_splits == 1:
        return {"static": 1, "dynamic": [1] * len(seqlens)}
    static, dyn = FM.split_plan(seqlens, NH, NH, D, num_sms=NUM_SMS, Dv=DV)
    return {"static": static, "dynamic": dyn}


def launch_geometry(names: list[str], static: int, counts: dict[str, int] | None = None) -> dict[str, Any]:
    """The `FlashAttnFwdSm90` and `FlashAttnFwdCombine` instantiations among a launch's kernel names, against the 128 x 80 x 256 tile
    and the flags vLLM's call sets, with `Split` and exactly `WANT_COMBINE` where the plan splits the launch (and no combine where it
    doesn't); with ``counts`` (`kernel_counts`), the window's `CALLS` calls each ran one forward kernel, and one combine exactly when
    the launch splits."""
    split = static > 1
    fwd = sorted({(int(m[0]), (int(m[4]), int(m[5]), int(m[6])), int(m[7]), tuple(x.strip() == "true" for x in m[8].split(",")))
                  for n in names for m in _FWD.findall(n)})
    combines = [dict(zip(COMBINE_FIELDS, ([int(m[0]), int(m[1])], int(m[2]), int(m[3]), int(m[4]), m[5] == "true", m[6] == "true")))
                for m in sorted({m for n in names for m in _COMBINE.findall(n)})]
    parsed = all(_COMBINE.search(n) for n in names if "FlashAttnFwdCombine" in n)
    want = dict(WANT_FLAGS, Split=split)
    want_counts = {"fwd": CALLS, "combine": CALLS if split else 0}
    rec: dict[str, Any] = {"instantiations": [{"stages": s, "tile": list(t), "headdim_v": dv, "flags": dict(zip(FLAGS, f))}
                                              for s, t, dv, f in fwd],
                           "combine": combines, "counts": counts,
                           "want": {"tile": [BM, BN, D], "headdim_v": DV, "flags": want, "combine": [WANT_COMBINE] if split else [],
                                    "counts": want_counts},
                           "names": [n[:600] for n in names]}
    one = rec["instantiations"][0] if len(fwd) == 1 else None
    rec["ok"] = bool(one and one["tile"] == [BM, BN, D] and one["headdim_v"] == DV
                     and all(one["flags"][k] == v for k, v in want.items()) and parsed and combines == rec["want"]["combine"]
                     and static <= 1 << WANT_COMBINE["log_max_splits"] and (counts is None or counts == want_counts))
    return rec


def kernel_counts(prof) -> dict[int, dict[str, int]]:
    """How many `FlashAttnFwdSm90` and `FlashAttnFwdCombine` kernels each launch window ran, in `fa2_target_capture_gpu.kernels_by_launch`'s
    windows (one profiler session; a device kernel belongs to the window its start falls in)."""
    from torch.autograd import DeviceType

    ev = list(prof.events())
    windows = {int(e.name[len(C.LABEL):]): e.time_range for e in ev if e.name.startswith(C.LABEL)}
    out = {i: {"fwd": 0, "combine": 0} for i in windows}
    for e in ev:
        kind = "combine" if "FlashAttnFwdCombine" in e.name else "fwd" if "FlashAttnFwdSm90" in e.name else None
        if e.device_type == DeviceType.CUDA and kind:
            for i, w in windows.items():
                if w.start <= e.time_range.start <= w.end:
                    out[i][kind] += 1
    return out


# ---------------------------------------------------------------------------------------------------------------------------- gpu
def selection() -> dict[str, Any]:
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported

    v = get_flash_attn_version(head_size=D)
    return {"fa_version": v, "fa3_supported": bool(is_fa_version_supported(3)), "ok": v == 3}


def operands(c: dict[str, Any], seed: int) -> dict[str, Any]:
    """q [N, 8, 256]; k = k_nope ++ k_pe broadcast over the heads; v the strided view kv[..., 192:] of kv_b_proj's output."""
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    N = sum(c["seqlens"])

    def randn(*shape):
        x = torch.randn(*shape, generator=g, device="cuda")
        if c.get("wide"):
            x = x * torch.exp2(torch.randint(-8, 9, shape, generator=g, device="cuda").to(torch.float32))
        return x.to(torch.bfloat16)

    q, kv, k_pe = randn(N, NH, D), randn(N, NH, NOPE + DV), randn(N, 1, ROPE)
    k = torch.cat((kv[..., :NOPE], k_pe.expand(N, NH, ROPE)), dim=-1)
    return {"q": q, "k": k, "v": kv[..., NOPE:]}


def fa3_call(t: dict[str, Any], seqlens: list[int], num_splits: int):
    import torch
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    cu = torch.tensor([0, *np.cumsum(seqlens).tolist()], dtype=torch.int32, device="cuda")
    m = max(seqlens)
    return flash_attn_varlen_func(q=t["q"], k=t["k"], v=t["v"], max_seqlen_q=m, cu_seqlens_q=cu, max_seqlen_k=m, cu_seqlens_k=cu,
                                  softmax_scale=SCALE, causal=True, return_softmax_lse=False, num_splits=num_splits, fa_version=3)


def gpu(work: Path, outdir: Path, quick: bool, only: list[str] | None, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Selection, MUFU and every launch (doc["selection"], doc["mufu"], doc["launches"]); the comparison jobs, their words saved
    under ``work``."""
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    doc["selection"] = selection()
    print(f"[glm53-fa3] selection {json.dumps(doc['selection'])}", flush=True)
    p = torch.cuda.get_device_properties(0)
    doc["mufu"] = C.mufu(work, f"sm_{p.major}{p.minor}")
    for b in doc["mufu"]["builds"].values():
        if b.get("tables_kept"):
            kept = outdir / Path(b["tables_kept"]).name
            os.replace(b["tables_kept"], kept)
            b["tables_kept"] = str(kept)
    print(f"[glm53-fa3] mufu ok={doc['mufu']['ok']}", flush=True)
    jobs: list[dict[str, Any]] = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i, c in enumerate(case_list(quick)):
            if only and c["name"] not in only:
                continue
            t = operands(c, 20261013 + i)
            d = work / c["name"]
            d.mkdir(parents=True, exist_ok=True)
            words = {k: str(d / f"{k}.npy") for k in ("q", "k", "v")}
            for k, path in words.items():
                np.save(path, G._words(t[k]))
            for mode, ns in MODES:
                with record_function(f"{C.LABEL}{len(doc['launches'])}"):
                    outs = [fa3_call(t, c["seqlens"], ns) for _ in range(CALLS)]
                    torch.cuda.synchronize()
                pl = plan(c["seqlens"], ns)
                doc["launches"].append({"case": c["name"], "mode": mode, "seqlens": c["seqlens"], "num_splits_arg": ns, "plan": pl,
                                        "deterministic": all(G._same(outs[0], o) for o in outs[1:])})
                out = str(d / f"out_{mode}.npy")
                np.save(out, G._words(outs[0]))
                jobs += jobs_for(c, mode, pl, {**words, "out": out})
                del outs
            del t
    names, counts = C.kernels_by_launch(prof), kernel_counts(prof)
    for i, launch in enumerate(doc["launches"]):
        launch.update(launch_geometry(names.get(i, []), launch["plan"]["static"], counts.get(i, {"fwd": 0, "combine": 0})))
        print(f"[glm53-fa3] launch {launch['case']:18s} {launch['mode']:6s} plan={launch['plan']['static']}/{launch['plan']['dynamic']} "
              f"ok={launch['ok']} det={launch['deterministic']} tiles={[x['tile'] for x in launch['instantiations']]} "
              f"combine={launch['combine']} counts={launch['counts']}", flush=True)
    torch.cuda.empty_cache()
    return jobs


# ------------------------------------------------------------------------------------------------------------------------ compare
def jobs_for(c: dict[str, Any], mode: str, pl: dict[str, Any], paths: dict[str, str]) -> list[dict[str, Any]]:
    """One launch's comparison jobs: each sequence's rows in chunks of ``CHUNK``, over the words saved at ``paths``."""
    jobs, start = [], 0
    for s, L in enumerate(c["seqlens"]):
        for r0 in range(0, L, CHUNK):
            jobs.append({"case": c["name"], "mode": mode, "seq": s, "start": start, "L": L, "span": [r0, min(L, r0 + CHUNK)],
                         "launch_max": max(c["seqlens"]), "static": pl["static"], "split": pl["dynamic"][s], "paths": paths})
        start += L
    return jobs


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"


def compare(job: dict[str, Any]) -> dict[str, Any]:
    """One chunk of one sequence's rows against the twin, with the sampled checks (a worker process: saved words in, a record out)."""
    try:
        return _compare(job)
    except Exception as e:  # noqa: BLE001
        return {k: job[k] for k in ("case", "mode", "seq", "span")} | {"error": _error(e)}


def matching_ns(q: np.ndarray, k: np.ndarray, v: np.ndarray, T: int, nbm: int, mf: int, static: int, got: np.ndarray) -> list[int]:
    """The split counts up to the launch's ``static`` at which the twin gives one row's words ``got`` (q (NH, D), k / v (T, NH, D),
    got (NH, D)): the least NS of each distinct cut, 1 being the unsplit epilogue and NS >= 2 the combine over
    `fa3_split.split_ranges`."""
    from verity_vllm.program.kernels import derived_rows as DR
    from verity_vllm.program.kernels import fa3_split_twins as FT
    from verity_vllm.program.registry import fa3_split as FS

    seen, out = set(), []
    for ns in range(1, int(static) + 1):
        key = (ns,) if ns == 1 else tuple(FS.split_ranges(T, nbm, ns, BN))
        if key in seen:
            continue
        seen.add(key)
        if ns == 1:
            w = DR.attention_rows(q, k, v, BN, masked_from=mf, dot_prim="HopperBF16WgmmaDot16_v1", inv_prim="Fa3InvSum_v1")[0]
        else:
            w = FT.head_split_rows(q, k.transpose(1, 0, 2), v.transpose(1, 0, 2), T=T, NBM=nbm, NS=ns, BN=BN, MASKED_FROM=mf)[0]
        if np.array_equal(w, got):
            out.append(ns)
    return out


def _compare(job: dict[str, Any]) -> dict[str, Any]:
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.reference import evaluate
    from verity_vllm.program.frontend import targets as TG
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.program.kernels import derived_rows as DR
    from verity_vllm.program.kernels import fa2_relation, fa3_model as FM
    from verity_vllm.program.kernels import fa3_split_twins as FT
    from verity_vllm.program.registry import attention_vocab, hopper
    from verity_vllm.program.registry import fa3_check_inf as F3
    from verity_vllm.program.registry import fa3_split as FS

    fa3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
    twin = {"dot_prim": "HopperBF16WgmmaDot16_v1", "inv_prim": "Fa3InvSum_v1"}
    profile = TargetProfile.from_json(json.loads(CMAP.read_text())["target"]["profile"])
    a, L, (r0, r1), NS = job["start"], job["L"], job["span"], int(job["split"])
    P = job["paths"]
    q = np.load(P["q"], mmap_mode="r")[a:a + L]
    k = np.ascontiguousarray(np.load(P["k"], mmap_mode="r")[a:a + L])
    v = np.ascontiguousarray(np.load(P["v"], mmap_mode="r")[a:a + L])
    got = np.load(P["out"], mmap_mode="r")[a + r0:a + r1]
    rows = np.arange(r0, r1)
    if NS > 1:
        want, nf, opn = FT.split_call(np.ascontiguousarray(q[r0:r1]), k, v, rows, L, NS, fa2_relation.tables(), kblock_n=BN, kblock_m=BM)
        nbm = FM.kernel_geometry(L, L, rows, BN, kblock_m=BM, qh_per_kvh=1)[1]
    else:
        res = FM.attention_call(np.ascontiguousarray(q[r0:r1]), k, v, rows, L, L, SCALE, fa2_relation.tables(), kblock_n=BN)
        want, nf, opn = res.out.copy(), res.nonfinite, np.zeros_like(res.nonfinite)
    rec: dict[str, Any] = {k_: 0 for k_ in TOTALS} | {"first_mismatch": None, "ns_matches": []}
    for i, r in enumerate(rows):
        T = int(r) + 1
        mf = attention_vocab.fa3_masked_from(T, BN, int(r), BM, 1)
        nb = int(nbm[i]) if NS > 1 else 0
        head = (bind(FS.AttentionHeadSplit, T=T, NBM=nb, NS=NS, D=D, BN=BN, MASKED_FROM=mf, **fa3) if NS > 1
                else bind(F3.AttentionHeadV4, T=T, D=D, BN=BN, MASKED_FROM=mf, **fa3))

        def reference(h: int) -> list[int]:
            rec["reference"] += 1
            return [int(x) for x in evaluate(head, q[r, h].tolist(), k[:T, h].reshape(-1).tolist(), v[:T, h].reshape(-1).tolist())]

        nonfinite = np.flatnonzero(nf[i])
        for h in nonfinite:
            want[i, h] = reference(int(h))
        if int(r) in SAMPLE or int(r) == L - 1:
            rec["sampled_rows"] += 1
            if NS > 1:
                vec, vnf, vop = FT.head_split_rows(np.ascontiguousarray(q[r]), k[:T].transpose(1, 0, 2), v[:T].transpose(1, 0, 2), T=T,
                                                   NBM=nb, NS=NS, BN=BN, MASKED_FROM=mf)
                fin = ~(vnf | nf[i] | vop | opn[i])
            else:
                vec, vnf = DR.attention_rows(np.ascontiguousarray(q[r]), k[:T], v[:T], BN, masked_from=mf, **twin)
                fin = ~(vnf | nf[i])
            rec["twin_mismatch"] += int((vec[fin] != want[i][fin]).any(axis=1).sum())
            h = int(r) % NH
            if h not in nonfinite:
                rec["twin_mismatch"] += int(reference(h) != [int(x) for x in want[i, h]])
            spec = TG.attention_spec(profile, T, NH, NH, D, job["launch_max"], row=int(r))
            bound = bind(F3.AttentionV4, T=T, NH=NH, KVH=NH, D=D, BN=BN, MASKED_FROM=mf, **fa3)
            if NS > 1:
                spec, bound = FS.split_of(spec, nb, NS), bind(FS.AttentionSplit, T=T, NBM=nb, NS=NS, NH=NH, KVH=NH, D=D, BN=BN,
                                                              MASKED_FROM=mf, **fa3)
                ns = matching_ns(np.ascontiguousarray(q[r]), k[:T], v[:T], T, nb, mf, job["static"], np.asarray(got[i]))
                plan_cut = tuple(FS.split_ranges(T, nb, NS, BN))
                rec["ns_sampled"] += 1
                rec["ns_unique"] += int(len(ns) == 1 and ns[0] > 1 and tuple(FS.split_ranges(T, nb, ns[0], BN)) == plan_cut)
                rec["ns_matches"].append({"row": int(r), "matches": ns})
            rec["binding_mismatch"] += spec.id != bound.id
        bad = np.flatnonzero((want[i] != got[i]).any(axis=1) & ~opn[i])
        rec["rows"] += 1
        rec["heads"] += NH
        rec["nonfinite"] += int(nonfinite.size)
        rec["lg2_open"] += int(opn[i].sum())
        rec["mismatch"] += int(bad.size)
        if bad.size and rec["first_mismatch"] is None:
            h = int(bad[0])
            col = int(np.flatnonzero(want[i, h] != got[i, h])[0])
            rec["first_mismatch"] = {"row": int(r), "head": h, "T": T, "col": col, "want": int(want[i, h, col]), "kernel": int(got[i, h, col])}
    return {k_: job[k_] for k_ in ("case", "mode", "seq", "span", "split")} | rec


# ---------------------------------------------------------------------------------------------------------------------------- run
def klass(r: dict[str, Any]) -> str:
    return "pinned" if r["mode"] == "pinned" else ("served_split" if r["split"] > 1 else "served")


def summary(comparisons: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for c in CLASSES:
        rs = [r for r in comparisons if klass(r) == c]
        out[c] = {k: sum(r[k] for r in rs) for k in TOTALS} | {"first_mismatch": next((r["first_mismatch"] for r in rs if r["first_mismatch"]), None),
                                                                 "sequences": len({(r["case"], r["seq"]) for r in rs})}
    return out


def _class_ok(t: dict[str, Any]) -> bool:
    return (t["rows"] > 0 and t["mismatch"] == 0 and t["lg2_open"] == 0 and t["twin_mismatch"] == 0 and t["binding_mismatch"] == 0
            and t["sampled_rows"] > 0 and t["reference"] > 0)


def _verdict(doc: dict[str, Any]) -> bool:
    cl = doc["classes"]
    return bool(doc["gate"]["ok"] and not doc["errors"] and doc["selection"].get("ok") is True and doc["mufu"].get("ok") is True
                and doc["launches"] and all(x["ok"] and x["deterministic"] for x in doc["launches"])
                and all(_class_ok(cl[c]) for c in CLASSES))


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int, quick: bool,
        only: list[str] | None) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    import torch  # noqa: F401

    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "quick": quick, "environment": G._environment(),
                           "gate": C.gate(min_driver, want_cc, want_sms), "errors": {}, "launches": [],
                           "selection": {"ok": False}, "mufu": {"ok": False}}
    print(f"[glm53-fa3] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("GLM53-FA3-PREFILL GATE-FAIL", flush=True)
        raise SystemExit(3)
    try:
        jobs = gpu(work, outdir, quick, only, doc)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["gpu"] = _error(e)
        print(f"[glm53-fa3] gpu ERROR {doc['errors']['gpu']}", flush=True)
        jobs = []
    print(f"[glm53-fa3] comparing {len(jobs)} jobs on {workers} workers", flush=True)
    doc["comparisons"] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as ex:
        for res in ex.map(compare, jobs):
            if "error" in res:
                doc["errors"][f"compare/{res['case']}/{res['mode']}/{res['seq']}/{res['span'][0]}"] = res["error"]
                continue
            doc["comparisons"].append(res)
    doc["classes"] = summary(doc["comparisons"])
    for c, t in doc["classes"].items():
        print(f"[glm53-fa3] rows {c:12s} sequences={t['sequences']} rows={t['rows']} heads={t['heads']} !={t['mismatch']} "
              f"nonfinite={t['nonfinite']} lg2_open={t['lg2_open']} ref={t['reference']} twin!={t['twin_mismatch']} "
              f"bind!={t['binding_mismatch']} ns_unique={t['ns_unique']}/{t['ns_sampled']} first={json.dumps(t['first_mismatch'])}",
              flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    cl, ls = doc["classes"], doc["launches"]
    split = [x for x in ls if x["plan"]["static"] > 1]
    print(f"GLM53-FA3-PREFILL {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} "
          + " ".join(f"{c}={json.dumps({k: cl[c][k] for k in TOTALS})}" for c in CLASSES)
          + f" launches_ok={sum(bool(x['ok']) for x in ls)}/{len(ls)} split_launches_ok={sum(bool(x['ok']) for x in split)}/{len(split)} "
          f"deterministic={sum(bool(x['deterministic']) for x in ls)}/{len(ls)} selection={doc['selection'].get('ok')} "
          f"mufu={doc['mufu'].get('ok')} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's FA3 dense MHA prefill (head 256) on this Hopper GPU against Attention_v4 and, "
                                             "for a sequence FA3 splits, AttentionSplit_v1")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operand and output words)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true", help=f"only the cases {QUICK}: a first signal in minutes")
    ap.add_argument("--only", default=None, help="comma list of case names")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick, a.only.split(",") if a.only else None)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
