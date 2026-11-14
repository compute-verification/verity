"""GLM-5.3's IndexShare layers on an H200: vLLM d9105ea80's `_fused_norm_rope_kernel` and `_fused_q_kernel` compiled with
`HAS_INDEXER` false, against the Definitions the capture map names for the same programs with the indexer.  58 of the 78 target
layers have no indexer (`index_topk_freq` 4 at offset 3: attention.py:161-195), and the MTP draft's steps after its first reuse the
first step's top-k (`set_skip_topk`); each calls both kernels with `has_indexer=False` (attention.py:308-327), a Triton constexpr,
so each is its own specialisation.  Without the indexer `_fused_norm_rope_kernel`'s programs 0 (the index-K cache write) and 3 (the
top-k fill) return at once, and `_fused_q_kernel`'s programs 1 (the index query) and, with the bf16 query, 2 (the fp8 pack).

One GPU process writes `<out>/glm53_index_share_capture.json`, the property record `glm53_index_share_capture`
(`records.property_record`), and both kernels' PTX (and SASS where Triton's `cuobjdump` runs) under `<out>/ptx/`.  The device gate
runs first and nothing else runs when it fails (exit 3).  vLLM runs as in `glm53_dsa_capture_gpu`, at its defaults (its
`environment`), its switches read back by that capture's `served`.

  gate       `fa2_target_capture_gpu.gate`: driver, compute capability 9.0, 132 SMs.
  selection  FLASH_ATTN_MLA_SPARSE first for GLM-5.3's MLA (bf16, kv cache "auto"), and `is_fused_q_cutedsl_supported` refuses
             the bf16 query without the indexer, so `_fused_q_kernel` runs.
  norm_rope  `fused_norm_rope` as attention.py:340-366 calls it for a layer without an indexer (index_k, its LayerNorm and its
             rotary cache None, no indexer cache), twice on fresh caches over `glm53_fused_rmsnorm_capture_gpu`'s served operands
             and once over its RMSNorm probe: the words equal, every entry of the top-k buffer as it was (the layer reuses the last
             full layer's), each written MLA cache row [kv_c_out | k_pe_out], every unwritten row untouched.  Then the subnormal
             RoPE probe (`glm53_dsa_capture_gpu.rope_probe_inputs`) in k_pe on its own rotary cache.
  q          `fused_q` as attention.py:378-392 calls it without the indexer and with the bf16 query, twice over the served
             operands, then the RoPE probe in every head's q_pe.
  words      q_c's and kv_c's outputs of every row against `DsaRMSNorm_v1{N, EPS=1e-05}` (`glm53_fused_rmsnorm_capture_gpu`'s
             twin, reference sample and refuted alternatives); k_pe_out and q_pe_out of every token and of the probe: which RoPE
             contraction the words are (`glm53_dsa_capture_gpu.phase_rope`), which must be `RoPEHeadGptJ_v2`'s.
  ptx        the Triton cache holds one specialisation of each kernel (every launch here ran the form without the indexer); their
             PTX counts, the e4m3 conversions among them.
  bind       DsaRMSNorm_v1 at N 2048 and 512 and RoPEHeadGptJ_v2{D=64} bind.

GO = the gate, the switches, the selection, every launch's checks, every RMSNorm word equal to the twin with the reference equal to
it on every sampled row and every alternative refuted on at least PER_ALT words of each norm, both RoPE outputs settled on
`RoPEHeadGptJ_v2`, the PTX, the bindings, no error.

    cd adapters/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_index_share_capture_gpu.py --out DIR --scratch /tmp --want-cc 9.0 --want-sms 132 --min-driver 575
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import argparse
import json
import multiprocessing
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from verity.circuits.defs import bind
from verity.definitions.registry import boolean_rope as BR

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from tests.properties import glm53_dsa_capture_gpu as DS
from tests.properties import glm53_fused_rmsnorm_capture_gpu as FR
from verity_vllm.records.property_record import REGISTRY, seal as seal_record

SCHEMA = REGISTRY["glm53_index_share_capture"].schema
RECORD = "glm53_index_share_capture.json"
ROPE_DEFINITION = "RoPEHeadGptJ_v2"
TOPK_SENTINEL = 7
E4M3 = "cvt.rn.satfinite.e4m3x2.f32"


# ---------------------------------------------------------------------------------------------------------------------------- gpu
def norm_rope(o: dict[str, Any], c: dict[str, Any], cs_k):
    """`fused_norm_rope` as attention.py:340-366 calls it on a layer without an indexer; returns (q_c_out, kv_c_out, k_pe_out)."""
    import torch
    from vllm.models.deepseek_v32.common.kernels import fused_norm_rope

    kv_c_out, k_pe_out = torch.empty_like(o["kv_c"]), torch.empty_like(o["k_pe"])
    q_c_out = fused_norm_rope(o["pos"], o["q_c"], o["q_rms_w"], DS.RMS_EPS, o["kv_c"], o["kv_rms_w"], DS.RMS_EPS, o["k_pe"], cs_k, None,
                              None, None, 1e-6, None, c["topk"], slot_mapping=o["slots"], indexer_k_cache=None, mla_kv_cache=c["mla"],
                              mla_kv_cache_dtype="auto", mla_k_scale=o["k_scale"], has_indexer=False, index_rope_interleave=True,
                              kv_c_out=kv_c_out, k_pe_out=k_pe_out, index_k_out=None)
    return q_c_out, kv_c_out, k_pe_out


def fused_q_call(o: dict[str, Any], cs):
    """`fused_q` as attention.py:378-392 calls it without the indexer, with the bf16 query: its third output, q_pe_out."""
    from vllm.models.deepseek_v32.common.kernels import fused_q

    return fused_q(o["pos"], o["q_pe"], cs, None, None, o["ql_nope"], o["q_scale"], None, 0.0, 0.0, has_indexer=False,
                   index_rope_interleave=True, quantize_mqa=False)[2]


def selection() -> dict[str, Any]:
    import torch
    from vllm.models.deepseek_v32.nvidia.ops.fused_q_cutedsl import is_fused_q_cutedsl_supported

    rec = DS.selection()
    q_pe = torch.empty(1, DS.NH, DS.DR, dtype=torch.bfloat16, device="cuda")
    rec["fused_q_cutedsl_no_indexer"] = bool(is_fused_q_cutedsl_supported(
        q_pe, torch.empty(1, 1, 1, dtype=torch.bfloat16, device="cuda"), torch.empty(1, DS.NH, DS.DV, dtype=torch.bfloat16, device="cuda"),
        has_indexer=False, quantize_mqa=False))
    rec["ok"] = bool(rec["valid"] and rec["valid"][0] == "FLASH_ATTN_MLA_SPARSE" and rec["fused_q_cutedsl_no_indexer"] is False)
    return rec


def launch(work: Path, o: dict[str, Any], case: str) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    """`norm_rope` twice on fresh caches: the launch record, its two norms' comparison jobs, and its k_pe_out."""
    import torch

    T = o["pos"].numel()
    c1, c2 = DS.caches(DS.NB, T), DS.caches(DS.NB, T)
    out1 = norm_rope(o, c1, o["cs"])
    out2 = norm_rope(o, c2, o["cs"])
    torch.cuda.synchronize()
    slots = o["slots"].cpu().numpy()
    written = np.flatnonzero(slots >= 0)
    rows = DS.NB * DS.BLOCK
    mla = DS._u16(c1["mla"]).reshape(rows, DS.KV_LORA + DS.DR)
    unwritten = np.setdiff1d(np.arange(rows), slots[written])
    kv_out, kpe_out = DS._u16(out1[1]), DS._u16(out1[2])
    rec = {"case": case, "tokens": int(T), "slots_written": int(written.size), "checks": {
        "deterministic": bool(all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(out1, out2))
                              and torch.equal(c1["mla"].view(torch.int16), c2["mla"].view(torch.int16))),
        "topk_untouched": bool((c1["topk"] == TOPK_SENTINEL).all().item() and (c2["topk"] == TOPK_SENTINEL).all().item()),
        "mla_rows": bool(written.size and np.array_equal(mla[slots[written]], np.concatenate([kv_out[written], kpe_out[written]], axis=1))),
        "unwritten_untouched": bool((mla[unwritten] == 0x5A5A).all())}}
    saved = {norm: {"x": DS._u16(o[norm]), "w": DS._u16(o[wkey]), "out": DS._u16(out1[k])} for norm, _, wkey, k in FR.NORMS}
    return rec, FR.jobs_for(work, case, saved, o.get("probe_rows", {})), out1[2]


def rope_probes(o: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The subnormal RoPE probe through each kernel: k_pe of `rope_probe_inputs` on its own rotary cache at positions 0..n-1, then
    every head's q_pe the same; (the launch record, k_pe's probe, q_pe's probe)."""
    import torch

    n = DS.PROBE_TOKENS
    x, cs = DS.rope_probe_inputs(n)
    cs_t = torch.from_numpy(cs.view(np.int16)).to("cuda").view(torch.bfloat16).contiguous()
    p = DS.probe_operands(o, n, 4444)
    p["k_pe"].copy_(torch.from_numpy(x.view(np.int16)).to("cuda").view(torch.bfloat16))
    p["slots"] = torch.arange(n, dtype=torch.int64, device="cuda")
    c = DS.caches(1, n)
    k_pe_out = norm_rope(p, c, cs_t)[2]
    p["q_pe"].copy_(torch.from_numpy(np.repeat(x[:, None, :], DS.NH, axis=1).view(np.int16)).to("cuda").view(torch.bfloat16))
    q_pe_out = fused_q_call(p, cs_t)
    torch.cuda.synchronize()
    rec = {"case": "rope_probe", "tokens": n, "checks": {"topk_untouched": bool((c["topk"] == TOPK_SENTINEL).all().item())}}
    return rec, {"x": x, "cs": cs, "out": DS._u16(k_pe_out)}, {"x": x, "cs": cs, "out": DS._u16(q_pe_out)}


def q_served(o: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """`fused_q_call` twice over the served operands: the launch record and q_pe_out."""
    import torch

    q1, q2 = fused_q_call(o, o["cs"]), fused_q_call(o, o["cs"])
    torch.cuda.synchronize()
    T = o["pos"].numel()
    return {"case": "q_served", "tokens": int(T), "checks": {
        "deterministic": bool(torch.equal(q1.view(torch.int16), q2.view(torch.int16))),
        "shape": list(q1.shape) == [T, DS.NH, DS.DR] and q1.dtype == torch.bfloat16}}, q1


def settle_rope(o: dict[str, Any], k_pe_out, q_pe_out, kprobe: dict[str, Any], qprobe: dict[str, Any]) -> dict[str, Any]:
    """`glm53_dsa_capture_gpu.phase_rope` on both outputs, ok only where each settles on ROPE_DEFINITION."""
    rec = DS.phase_rope(o, {"k_pe_out": k_pe_out, "k_pe_probe": kprobe}, {"q_pe_out": q_pe_out, "q_pe_probe": qprobe})
    rec["ok"] = bool(rec["ok"] and all(rec[k]["definition"] == ROPE_DEFINITION for k in ("k_pe", "q_pe")))
    return rec


def gpu(work: Path, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The launches (doc["launches"]), the RMSNorm probes (doc["probes"]) and the RoPE forms (doc["rope"]); returns the norms'
    comparison jobs.  Each step's error goes to doc["errors"] and the steps that read nothing of it still run."""
    import torch
    import triton
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    def step(name: str, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = DS._error(e)
            print(f"[glm53-index-share] {name} ERROR {doc['errors'][name][:600]}", flush=True)
            return None

    doc["triton"] = triton.__version__
    jobs: list[dict[str, Any]] = []
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        doc["selection"] = step("selection", selection) or {"ok": False}
        print(f"[glm53-index-share] selection {json.dumps(doc['selection'])[:1500]}", flush=True)
        probes = step("probes", lambda: {norm: FR.probe(N, FR.EPS, 20261013 + N) for norm, N, _, _ in FR.NORMS})
        doc["probes"] = {norm: {"rows": len(p["x"]), "targets": p["targets"]} for norm, p in (probes or {}).items()}
        o = step("operands", lambda: FR.served_operands(20261005))
        if o is None:
            return jobs
        k_pe_out = q_pe_out = kprobe = qprobe = None
        cases = [("served", lambda: o)] + ([("probe", lambda: FR.probe_operands(o, probes))] if probes else [])
        for case, ops in cases:
            r = step(f"norm_rope/{case}", lambda ops=ops, case=case: launch(work, ops(), case))
            if r is not None:
                doc["launches"].append(r[0])
                jobs += r[1]
                k_pe_out = r[2] if case == "served" else k_pe_out
                print(f"[glm53-index-share] norm_rope {case:6s} tokens={r[0]['tokens']} {json.dumps(r[0]['checks'])}", flush=True)
        r = step("q_served", lambda: q_served(o))
        if r is not None:
            doc["launches"].append(r[0])
            q_pe_out = r[1]
            print(f"[glm53-index-share] fused_q served tokens={r[0]['tokens']} {json.dumps(r[0]['checks'])}", flush=True)
        r = step("rope_probe", lambda: rope_probes(o))
        if r is not None:
            doc["launches"].append(r[0])
            kprobe, qprobe = r[1], r[2]
            print(f"[glm53-index-share] rope probe tokens={r[0]['tokens']} {json.dumps(r[0]['checks'])}", flush=True)
        if k_pe_out is not None and q_pe_out is not None and kprobe is not None:
            doc["rope"] = step("rope", lambda: settle_rope(o, k_pe_out, q_pe_out, kprobe, qprobe)) or {"ok": False}
    torch.cuda.empty_cache()
    return jobs


def ptx(triton_cache: Path, outdir: Path) -> dict[str, Any]:
    """`glm53_dsa_capture_gpu.ptx` (one specialisation of each kernel), with each one's e4m3 conversions."""
    rec = DS.ptx(triton_cache, outdir)
    rec["e4m3_cvt"] = {k: [e["counts"].get(E4M3, 0) for e in v] for k, v in rec["kernels"].items()}
    return rec


def bind_check() -> dict[str, Any]:
    rec = FR.bind_check()
    name = f"{ROPE_DEFINITION}{{D={DS.DR}}}"
    try:
        rec["bindings"][name] = bool(bind(BR.RoPEHeadGptJV2, D=DS.DR).signature())
    except Exception as e:  # noqa: BLE001
        rec["bindings"][name] = DS._error(e)
    rec["ok"] = bool(rec["ok"] and rec["bindings"][name] is True)
    return rec


# ---------------------------------------------------------------------------------------------------------------------------- run
def _verdict(doc: dict[str, Any]) -> bool:
    s = doc["summary"]
    return bool(doc["gate"]["ok"] and doc["served"].get("ok") is True and not doc["errors"]
                and doc.get("selection", {}).get("ok") is True and doc.get("rope", {}).get("ok") is True
                and doc["ptx"].get("ok") is True and doc["bind"].get("ok") is True
                and {x["case"] for x in doc["launches"]} == {"served", "probe", "q_served", "rope_probe"}
                and all(x["checks"] and all(v is True for v in x["checks"].values()) for x in doc["launches"])
                and len(s["cases"]) == len(FR.CASES) * len(FR.NORMS) and all(FR._case_ok(t) for t in s["cases"].values())
                and all(n >= FR.PER_ALT for alt in s["refuted"].values() for n in alt.values()))


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    env, cleared = DS.environment(work)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "env_cleared": cleared, "served": DS.served(),
                           "environment": G._environment(), "gate": C.gate(min_driver, want_cc, want_sms), "errors": {},
                           "launches": []}
    print(f"[glm53-index-share] gate {json.dumps(doc['gate'])} served {json.dumps(doc['served'])} cleared {sorted(cleared)}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("GLM53-INDEX-SHARE GATE-FAIL", flush=True)
        raise SystemExit(3)
    jobs: list[dict[str, Any]] = []
    try:
        jobs = gpu(work, doc)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["gpu"] = DS._error(e)
        print(f"[glm53-index-share] gpu ERROR {doc['errors']['gpu']}", flush=True)
    for name, fn in (("ptx", lambda: ptx(work / "triton_cache", outdir)), ("bind", bind_check)):
        try:
            doc[name] = fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = DS._error(e)
            doc[name] = {"ok": False}
    rope = doc.get("rope", {})
    print(f"[glm53-index-share] rope {json.dumps({k: v for k, v in rope.items() if k != 'ok'})[:2000]}", flush=True)
    print(f"[glm53-index-share] ptx {json.dumps(doc['ptx'])[:2000]}", flush=True)
    print(f"[glm53-index-share] bind {json.dumps(doc['bind'])}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
    doc["comparisons"] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=FR._worker_init) as ex:
        for res in ex.map(FR.compare, jobs):
            if "error" in res:
                doc["errors"][f"compare/{res['case']}/{res['norm']}/{res['span'][0]}"] = res["error"]
                continue
            doc["comparisons"].append(res)
    doc["summary"] = FR.summary(doc["comparisons"])
    for key, t in doc["summary"]["cases"].items():
        print(f"[glm53-index-share] words {key:13s} rows={t['rows']} words={t['words']} !={t['mismatch']} ref={t['reference']} "
              f"twin!={t['twin_mismatch']} first={json.dumps(t['first_mismatch'])}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    totals = {k: sum(t[k] for t in doc["summary"]["cases"].values()) for k in FR.TOTALS}
    names = {k: rope.get(k, {}).get("definition") for k in ("k_pe", "q_pe")}
    rope_words = {k: rope.get(k, {}).get("words") for k in ("k_pe", "q_pe")}
    print(f"ROPE k_pe={names['k_pe']} q_pe={names['q_pe']} words={json.dumps(rope_words)} settled={rope.get('ok')}", flush=True)
    print(f"GLM53-INDEX-SHARE {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} served={doc['served'].get('ok')} "
          f"selection={doc.get('selection', {}).get('ok')} rmsnorm_words={json.dumps(totals)} "
          f"refuted={json.dumps(doc['summary']['refuted'])} launches_ok={sum(all(x['checks'].values()) for x in doc['launches'])}/"
          f"{len(doc['launches'])} ptx={doc['ptx'].get('ok')} bind={doc['bind'].get('ok')} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's fused_norm_rope and fused_q without the indexer on this Hopper GPU against "
                                             "DsaRMSNorm_v1 and RoPEHeadGptJ_v2")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD} and ptx/")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (the words and the Triton cache)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
