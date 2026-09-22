"""Boundary acquisition table (workstream C, experiment 2; RC 2026-09-07).

For every canonical interface family of the eager B1 profile the question is how a verifier (or a
replay) can *obtain* the canonical value at an interface, at what cost, and whether the obtained
value is bit-exact with the committed one:

(a) **direct materialised read** -- the eager capture reads the kernel's output tensor (the leaf);
    the cost is the leaf's bytes (copied device->host and hashed once at capture, once more at
    commit).  Its bit-exactness check is *cross-batch*: the same canonical id replayed in another
    batch (:mod:`verity_vllm_adapter.compare_selected`), not part of this module.
(b) **recomputation from bound neighbours with the reference function** -- the registered
    same-runtime relation re-executes the installed vLLM/torch operator on the authenticated
    openings of the node's inputs (``--mode same_runtime``, GPU, the torch venv).
(c) **alternative construction** -- an independent CPU model (W4b's GEMM chain model, W11's RMSNorm /
    FlashAttention-2 models, W3's Philox sampler re-derivation, W2's byte-slice / IEEE-add
    checkers, and three small RC transcriptions of W4b's element-wise models for the fused-add
    residual, RoPE and SiLU·mul) run on the same openings (``--mode independent``, CPU, Python 3.12
    with ``veritor`` importable).  Coverage gates are honoured: a model that was not validated on
    this GPU generation (W11's tables are sm_89) reports ``unresolved`` and is *additionally* run
    ungated as a labelled diagnostic (``diagnostic_ungated``), never as a claim.

Both modes check the *same* sample (``sample.json`` written by the first run, ``--sample`` in the
second) so per-VU witness bytes and times are comparable.  Per VU: status, checker seconds, wall
seconds (opening + authentication + checker), input/output bytes, openings.

    PYTHONPATH=. python -m verity_vllm_adapter.acquisition <bundle> --out <dir> --mode same_runtime --per-stratum 4
    PYTHONPATH=.:../src VERITY_ARCH=sm_80 python3.12 -m verity_vllm_adapter.acquisition <bundle> --out <dir> \
        --mode independent --sample <dir>/sample.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm import dtypes, identity
from verity_vllm import relations as R
from verity_vllm.capture import CaptureReader, Value
from verity_vllm.commit import TreeCache, load_commitments
from verity_vllm.description import Description, NodeInstance
from verity_vllm_adapter.memo_cache import MemoCache

# --- families -------------------------------------------------------------------------------------
# (key, label, predicate over (node json, port)) -- the interface families of the eager B1 profile.
# "GEMM in" is the producer's leaf (rmsnorm normed / attention / silu_mul / final norm), covered by
# those rows; the table names the mapping.


def _mp(nd: dict[str, Any]) -> str:
    return str(nd.get("module_path") or nd["op"])


FAMILIES: list[tuple[str, str, Any]] = [
    ("embedding", "embedding output (aten.embedding)", lambda nd, p: nd["op_type"] == "embedding"),
    ("rmsnorm_triton", "RMSNorm normed, layer-0 input norm (verity.bi_rms_norm, Triton)", lambda nd, p: nd["op_type"] == "rmsnorm" and nd.get("runtime_op") == "verity.bi_rms_norm"),
    ("rmsnorm_fused", "RMSNorm normed, residual-carrying norms (_C.fused_add_rms_norm port 0)", lambda nd, p: nd["op_type"] == "rmsnorm" and nd.get("runtime_op") == "_C.fused_add_rms_norm" and p == 0),
    ("residual", "residual stream x+residual (_C.fused_add_rms_norm port 1)", lambda nd, p: nd["op_type"] == "rmsnorm" and nd.get("runtime_op") == "_C.fused_add_rms_norm" and p == 1),
    ("gemm_qkv", "GEMM out qkv_proj [2048] (verity.bi_mm)", lambda nd, p: nd["op_type"] == "gemm" and "qkv_proj" in _mp(nd)),
    ("gemm_o", "GEMM out o_proj [1536]", lambda nd, p: nd["op_type"] == "gemm" and "o_proj" in _mp(nd)),
    ("gemm_gate_up", "GEMM out gate_up_proj [17920]", lambda nd, p: nd["op_type"] == "gemm" and "gate_up_proj" in _mp(nd)),
    ("gemm_down", "GEMM out down_proj [1536]", lambda nd, p: nd["op_type"] == "gemm" and "down_proj" in _mp(nd)),
    ("lm_head", "lm_head logits [151936] (sampled positions)", lambda nd, p: nd["op_type"] == "lm_head"),
    ("bias_add", "qkv bias add (aten.add, op_type residual_add)", lambda nd, p: nd["op_type"] == "residual_add"),
    ("rope_q", "RoPE q (_C.rotary_embedding port 0)", lambda nd, p: nd["op_type"] == "rope" and p == 0),
    ("rope_k", "RoPE k (_C.rotary_embedding port 1)", lambda nd, p: nd["op_type"] == "rope" and p == 1),
    ("attention", "attention output [12,128] (FlashAttention-2 via unified_attention_with_output)", lambda nd, p: nd["op_type"] == "attention"),
    ("silu_mul", "SiLU·mul [8960] (_C.silu_and_mul)", lambda nd, p: nd["op_type"] == "silu_mul"),
    ("sampler", "sampled token (sampler.select)", lambda nd, p: nd["op_type"] == "sampler"),
]

# the alternative construction per family (mode independent): relation id -> how to obtain it
INDEPENDENT_FOR: dict[str, str] = {
    "embedding": "embedding.row.v1",
    "rmsnorm_triton": "rmsnorm.bf16.v1",
    "rmsnorm_fused": "rc.fused_add_rms_norm.ieee_rsqrt.v1",
    "residual": "rc.residual.bf16.v1",
    "gemm_qkv": "gemm.bf16.v1", "gemm_o": "gemm.bf16.v1", "gemm_gate_up": "gemm.bf16.v1", "gemm_down": "gemm.bf16.v1",
    "lm_head": "lm_head.bf16.v1",
    "bias_add": "residual_add.bf16.v1",
    "rope_q": "rc.rope.bf16.v1", "rope_k": "rc.rope.bf16.v1",
    "attention": "attention.bf16.v1",
    "silu_mul": "rc.silu_mul.table.v1",
    "sampler": "sampler.select.v1",
}
SAME_RUNTIME_FOR: dict[str, str] = {"sampler": "sampler.select.same_runtime.v1"}  # the others use the description's relation


# --- RC transcriptions of W4b's element-wise models (numpy; docs/vllm-poc/numerics.md) ---------------


def _bf16_to_f32(w: np.ndarray) -> np.ndarray:
    return (np.asarray(w, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)
    r[nan] = 0x7FC0
    return r


def _words(v: Value, cols: tuple[int, int] | None = None) -> np.ndarray:
    if v.dtype != "bfloat16":
        raise R.RelationError(f"{v.canonical_id}: expected bfloat16, got {v.dtype}")
    w = np.frombuffer(v.data, dtype=np.uint16)
    return w[cols[0]:cols[1]] if cols else w


def _cols(inst: NodeInstance, name: str) -> tuple[int, int] | None:
    c = (inst.node.extra.get("arg_cols") or {}).get(name)
    return (int(c[0]), int(c[1])) if c else None


def rc_residual_bf16(inst: NodeInstance, inputs: R.Inputs) -> bytes:
    """port 1 of _C.fused_add_rms_norm: residual' = bf16(f32(x) + f32(residual)) (W4b's model of the
    CUDA fused_add_rms_norm_kernel; inductor_models.fused_add_rms_norm_eager)."""
    if inst.port != 1:
        raise R.RelationError("rc.residual.bf16.v1 checks port 1 (the new residual) only")
    x = _words(R.need(inputs, "x"), _cols(inst, "x"))
    r = _words(R.need(inputs, "residual"), _cols(inst, "residual"))
    return _f32_to_bf16((_bf16_to_f32(x) + _bf16_to_f32(r)).astype(np.float32)).astype("<u2").tobytes()


def rc_fused_add_rms_norm_ieee(inst: NodeInstance, inputs: R.Inputs) -> bytes:
    """port 0 of _C.fused_add_rms_norm under W4b's *fit* model with k=0: z = bf16(x+residual) (rounded
    first), s = IEEE correctly rounded 1/sqrt(mean(z^2)+eps), out = bf16((z*s)*w).  The kernel's s is
    MUFU.RSQ (approximate); W4b found the ulp offset k per row, so this is NOT a validated model --
    a `mismatch` here means the approximate rsqrt differed from IEEE on that row, not a wrong value."""
    if inst.port != 0:
        raise R.RelationError("rc.fused_add_rms_norm.ieee_rsqrt.v1 checks port 0 (the normed row) only")
    x = _bf16_to_f32(_words(R.need(inputs, "x"), _cols(inst, "x")))
    r = _bf16_to_f32(_words(R.need(inputs, "residual"), _cols(inst, "residual")))
    w = _bf16_to_f32(_words(R.need(inputs, "w")))
    eps = float((inst.node.extra.get("consts") or {}).get("eps", 1e-6))
    z = _bf16_to_f32(_f32_to_bf16((x + r).astype(np.float32)))
    mean = (z.astype(np.float64) ** 2).sum() / z.size
    s = np.float32(1.0 / np.sqrt(mean + eps))
    y = ((z * s).astype(np.float32) * w).astype(np.float32)
    return _f32_to_bf16(y).astype("<u2").tobytes()


def rc_rope_bf16(inst: NodeInstance, inputs: R.Inputs) -> bytes:
    """_C.rotary_embedding (NeoX, BF16 cos||sin cache row at position t): x' = x cos - y sin, y' = y cos + x sin,
    exact arithmetic then RNE to FP32 then RNE to BF16 (W4b beyond_gemm.rope_model; products of BF16
    operands are exact in FP32 so contraction cannot change the result)."""
    c = inst.node.extra.get("consts") or {}
    head = int(c["head_size"])
    if not bool(c.get("is_neox", True)):
        raise R.RelationError("rc.rope.bf16.v1 models the NeoX rotation only")
    name = "q" if inst.port == 0 else "k"
    x = _bf16_to_f32(_words(R.need(inputs, name), _cols(inst, name))).astype(np.float64)
    cache = R.need(inputs, "cos_sin_cache")
    if len(cache.shape) != 2 or cache.shape[1] != head:
        raise R.RelationError(f"cos_sin_cache shape {cache.shape} != [max_pos, {head}]")
    rot = head // 2
    row = _bf16_to_f32(np.frombuffer(cache.data, dtype=np.uint16).reshape(cache.shape)[inst.t]).astype(np.float64)
    cos, sin = row[:rot], row[rot:]
    nh = x.size // head
    x = x.reshape(nh, head)
    a, b = x[:, :rot], x[:, rot:]
    out = np.empty_like(x)
    out[:, :rot] = a * cos - b * sin
    out[:, rot:] = b * cos + a * sin
    return _f32_to_bf16(out.astype(np.float32)).reshape(-1).astype("<u2").tobytes()


_SILU_TABLE: np.ndarray | None = None


def _silu_table() -> np.ndarray:
    global _SILU_TABLE
    if _SILU_TABLE is None:
        # W3c's characterised table (RTX 4090, vllm._custom_ops.silu_and_mul), shipped with the synth bundle
        p = os.environ.get("VERITY_SILU_TABLE") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bundles",
                                                                "B1-b3eba618b1b5-20260907T1731Z-synth", "silu_bf16_table.npy")
        if not os.path.exists(p):
            raise R.RelationError(f"unresolved: silu BF16 table not found at {p} (set VERITY_SILU_TABLE)")
        t = np.load(p)
        if t.shape != (65536,):
            raise R.RelationError(f"silu table has shape {t.shape}, expected (65536,)")
        _SILU_TABLE = t.astype(np.uint16)
    return _SILU_TABLE


def rc_silu_mul_table(inst: NodeInstance, inputs: R.Inputs) -> bytes:
    """_C.silu_and_mul packed-BF16 path: out = bf16(f32(bf16(silu(g))) * f32(u)); silu(g) of one BF16 word is
    the 65536-entry table read off the GPU by W4b (fixtures/silu_bf16_table.npy)."""
    x = _words(R.need(inputs, "x"), _cols(inst, "x"))
    h = x.size // 2
    g, u = x[:h], x[h:]
    t = _silu_table()
    return _f32_to_bf16(_bf16_to_f32(t[g]) * _bf16_to_f32(u)).astype("<u2").tobytes()


def register_rc_models() -> list[str]:
    ids = []
    for rid, op_type, fn, doc in (
        ("rc.residual.bf16.v1", "rmsnorm", rc_residual_bf16, "bf16(f32(x)+f32(residual)) -- the new residual written by the CUDA fused_add_rms_norm_kernel (W4b's model)"),
        ("rc.fused_add_rms_norm.ieee_rsqrt.v1", "rmsnorm", rc_fused_add_rms_norm_ieee, "W4b fit model with IEEE rsqrt (k=0): NOT validated; mismatches measure the approximate MUFU.RSQ, see doc"),
        ("rc.rope.bf16.v1", "rope", rc_rope_bf16, "W4b rope_model: exact NeoX rotation with the BF16 cache row at t, RNE f32 then bf16"),
        ("rc.silu_mul.table.v1", "silu_mul", rc_silu_mul_table, "W4b silu table model: bf16(f32(table[g]) * f32(u))"),
    ):
        if rid not in R.REGISTRY:
            R.register(rid, klass=R.INDEPENDENT, op_type=op_type, includes_full_op=True, owner="RC", doc=doc)(fn)
        ids.append(rid)
    return ids


# --- sample -------------------------------------------------------------------------------------------


def family_of(nd: dict[str, Any], port: int) -> str | None:
    for key, _, pred in FAMILIES:
        if pred(nd, port):
            return key
    return None


def draw_sample(description: Description, reader: CaptureReader, per_stratum: int, seed: int) -> dict[str, Any]:
    """per_stratum VUs per (family, phase) from the bundle's request boundaries, deterministic in seed."""
    rng = random.Random(seed)
    profile_id = reader.profile_id
    pools: dict[tuple[str, str], list[str]] = defaultdict(list)
    population: dict[str, dict[str, int]] = defaultdict(lambda: {"prefill": 0, "decode": 0, "bytes": 0})
    for rid in reader.request_ids():
        b = reader.boundary(rid)
        for inst in description.instances(b):
            fam = family_of(inst.node.to_json(), inst.port)
            if fam is None:
                continue
            cid = inst.output_id(profile_id)
            pools[(fam, inst.phase)].append(cid)
            population[fam][inst.phase] += 1
            dt, shape = description.leaf_spec(inst.node, inst.port)
            population[fam]["bytes"] += int(np.prod(shape)) * (dtypes.itemsize(dt) or 0) if shape else (dtypes.itemsize(dt) or 0)
    sample: dict[str, list[str]] = {}
    for key, _, _ in FAMILIES:
        ids = []
        for phase in ("prefill", "decode"):
            pool = pools.get((key, phase), [])
            ids.extend(rng.sample(pool, min(per_stratum, len(pool))))
        sample[key] = ids
    return {"seed": seed, "per_stratum": per_stratum, "requests": reader.request_ids(), "sample": sample,
            "population": {k: dict(v) for k, v in population.items()}}


# --- checks ---------------------------------------------------------------------------------------------


def registry_for(description: Description, mode: str, fam: str) -> dict[str, R.Relation]:
    """The description's relation id -> the Relation object to run for this family in this mode."""
    reg: dict[str, R.Relation] = dict(R.REGISTRY)
    alt = (SAME_RUNTIME_FOR if mode == "same_runtime" else INDEPENDENT_FOR).get(fam)
    if alt is None:
        return reg
    if alt not in R.REGISTRY:
        raise SystemExit(f"{alt} is not registered in this process (mode {mode}, family {fam})")
    for nd in description.nodes:
        if family_of(nd.to_json(), 0) == fam or family_of(nd.to_json(), 1) == fam:
            reg[nd.relation] = R.REGISTRY[alt]
    return reg


def diagnostic_ungated(fam: str, inst: NodeInstance, inputs: dict[str, Any], committed: bytes) -> dict[str, Any] | None:
    """W11's models run *without* their sm_89 coverage gate on this GPU's rows: a labelled diagnostic."""
    try:
        if fam == "rmsnorm_triton":
            from verity_vllm_numerics import rms_relation as RR
            from verity_vllm_numerics import rms_triton_model as RM
            xw, ww, eps, _ = RR.operands(inst, inputs)
            t0 = time.perf_counter()
            out = RM.rms_norm_rows(xw[None], ww, eps, RR.tables("triton"), kernel="triton").reshape(-1).astype("<u2").tobytes()
            return {"model": "rmsnorm.bf16.v1 arithmetic (sm_89 tables) ungated", "equal": out == committed, "seconds": time.perf_counter() - t0,
                    "differing_words": int((np.frombuffer(out, "<u2") != np.frombuffer(committed, "<u2")).sum())}
        if fam == "attention":
            from verity_vllm_numerics import fa2_relation as FR
            q, k, v, consts = FR.operands(inst, inputs)
            t0 = time.perf_counter()
            res = FR.model_row(q, k, v, float(consts["scale"]))
            out = res.out.reshape(-1).astype("<u2").tobytes()
            return {"model": "attention.bf16.v1 arithmetic (sm_89 MUFU tables) ungated", "equal": out == committed, "seconds": time.perf_counter() - t0,
                    "n": int(k.shape[0]), "nonfinite_heads": int(res.nonfinite.sum()),
                    "differing_words": int((np.frombuffer(out, "<u2") != np.frombuffer(committed, "<u2")).sum())}
    except Exception as e:  # noqa: BLE001
        return {"model": "ungated diagnostic", "error": f"{type(e).__name__}: {e}"[:300]}
    return None


def open_inputs(description: Description, reader: CaptureReader, cache: TreeCache, inst: NodeInstance) -> dict[str, Any]:
    profile_id = reader.profile_id
    boundary = reader.boundary(inst.request_id)
    inputs: dict[str, Any] = {"__symbols__": description.symbols}
    for name, ref in description.resolve_args(inst, boundary, profile_id).items():
        if isinstance(ref, dict) and "const" in ref:
            inputs[name] = ref["const"]
            continue
        refs = ref if isinstance(ref, list) else [ref]
        vals = [cache.open(r).as_value() for r in refs]
        inputs[name] = vals if isinstance(ref, list) else vals[0]
    return inputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle = Path(args.bundle)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reader = CaptureReader(bundle)
    description = Description(reader.description)
    commitments = load_commitments(reader)
    profile_id = reader.profile_id
    installed: list[str] = []
    if args.mode == "same_runtime":
        for k_, v in (reader.manifest.get("engine_flags") or {}).items():
            os.environ.setdefault(k_, str(v))
        import verity_vllm_adapter.relations_gpu as gpu_rel

        installed = gpu_rel.install()
        import torch

        device = torch.cuda.get_device_name(0)
    else:
        os.environ.setdefault("VERITY_ARCH", "sm_80")
        installed += register_rc_models()
        try:
            from verity_vllm_numerics import relations as NRel  # registers gemm.bf16.v1 / lm_head.bf16.v1 (W4b)

            installed += list(NRel.REGISTERED_WITH_W2)
        except Exception as e:  # noqa: BLE001
            print(f"[acquisition] W4b GEMM relations not importable: {e!r}", flush=True)
        for modname in ("verity_vllm_numerics.rms_relation", "verity_vllm_numerics.fa2_relation"):
            try:
                mod = __import__(modname, fromlist=["register_with_w2"])
                installed += list(getattr(mod, "REGISTERED_WITH_W2", []) or mod.register_with_w2())
            except Exception as e:  # noqa: BLE001
                print(f"[acquisition] {modname} not importable: {e!r}", flush=True)
        try:
            import verity_vllm_sampler.registry as SReg

            installed += SReg.install()
        except Exception as e:  # noqa: BLE001
            print(f"[acquisition] sampler registry not installed: {e!r}", flush=True)
        device = f"cpu ({platform.processor() or platform.machine()})"
    if args.sample:
        sample_doc = json.loads(Path(args.sample).read_text())
    else:
        sample_doc = draw_sample(description, reader, args.per_stratum, args.seed)
        (out / "sample.json").write_text(json.dumps(sample_doc, indent=1) + "\n")
    cache = MemoCache(reader)  # weight tensors authenticated once (at the family's warm-up VU), reused after
    rows: list[dict[str, Any]] = []
    per_family: dict[str, Any] = {}
    for key, label, _ in FAMILIES:
        ids = sample_doc["sample"].get(key, [])
        if not ids:
            per_family[key] = {"label": label, "n": 0}
            continue
        reg = registry_for(description, args.mode, key)
        # warm-up on the first VU (first CUDA launch / first C++ build / table load), recorded apart
        t0 = time.perf_counter()
        warm = R.check(description, reader, commitments, ids[:1], registry=reg, cache=cache)
        warm_s = time.perf_counter() - t0
        fam_rows = []
        for cid in ids:
            t0 = time.perf_counter()
            rep = R.check(description, reader, commitments, [cid], registry=reg, cache=cache)
            wall = time.perf_counter() - t0
            r = rep.results[0]
            row = {**r.to_json(), "family": key, "wall_seconds": wall, "openings_attempted": rep.openings_attempted, "openings_verified": rep.openings_verified}
            row["relation_run"] = reg.get(r.relation_id).relation_id if reg.get(r.relation_id) else r.relation_id
            if args.mode == "independent" and key in ("rmsnorm_triton", "attention") and r.status in ("error", "unresolved"):
                boundary = reader.boundary(identity.parse(cid).request_id or "")
                inst = description.instance_of(cid, boundary)
                inputs = open_inputs(description, reader, cache, inst)
                row["diagnostic_ungated"] = diagnostic_ungated(key, inst, inputs, cache.open(cid).value)
            fam_rows.append(row)
            rows.append(row)
        statuses: dict[str, int] = defaultdict(int)
        for r_ in fam_rows:
            statuses[r_["status"]] += 1
        ok = [r_ for r_ in fam_rows if r_["status"] in ("exact", "mismatch")]
        diag = [r_.get("diagnostic_ungated") for r_ in fam_rows if r_.get("diagnostic_ungated")]
        per_family[key] = {
            "label": label, "n": len(fam_rows), "statuses": dict(statuses),
            "relation_run": fam_rows[0]["relation_run"], "relation_class": fam_rows[0].get("class"),
            "detail_example": next((r_["detail"] for r_ in fam_rows if r_["detail"]), None),
            "warmup_first_vu_seconds": warm_s,
            "checker_seconds_median": float(np.median([r_["checker_seconds"] for r_ in ok])) if ok else None,
            "wall_seconds_median": float(np.median([r_["wall_seconds"] for r_ in fam_rows])),
            "input_bytes_median": float(np.median([r_["input_bytes"] for r_ in fam_rows])),
            "output_bytes": fam_rows[0]["output_bytes"],
            "inputs_opened_median": float(np.median([r_["inputs_opened"] for r_ in fam_rows])),
            "diagnostic_ungated": {"n": len(diag), "equal": sum(1 for d in diag if d.get("equal")), "models": sorted({d.get("model", "") for d in diag}),
                                   "seconds_median": float(np.median([d["seconds"] for d in diag if "seconds" in d])) if any("seconds" in d for d in diag) else None,
                                   "errors": [d["error"] for d in diag if "error" in d][:2]} if diag else None,
            "population": sample_doc["population"].get(key),
        }
        print(f"[acquisition] {key:16s} {args.mode:13s} n={len(fam_rows):2d} {dict(statuses)} checker_med={per_family[key]['checker_seconds_median']} "
              f"wall_med={per_family[key]['wall_seconds_median']:.3f}s in_bytes_med={per_family[key]['input_bytes_median']:.0f} out={per_family[key]['output_bytes']}"
              + (f" diag={per_family[key]['diagnostic_ungated']['equal']}/{per_family[key]['diagnostic_ungated']['n']}" if diag else ""), flush=True)
    doc = {
        "schema": "verity-vllm/acquisition/v1", "bundle": str(bundle), "profile_id": profile_id, "mode": args.mode, "device": device,
        "python": sys.version.split()[0], "relations_installed": installed, "arch_declared": os.environ.get("VERITY_ARCH"),
        "sample": {"seed": sample_doc["seed"], "per_stratum": sample_doc["per_stratum"], "requests": len(sample_doc["requests"])},
        "families": per_family, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "identities are capture-v2 canonical ids (TA1's typed program had not landed); (a) direct read is the leaf itself, its cross-batch "
                "bit-exactness is compare_selected's; (b) = same_runtime mode; (c) = independent mode; wall = open + authenticate + checker for one VU",
    }
    (out / f"acquisition_{args.mode}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    with (out / f"acquisition_{args.mode}.jsonl").open("w") as f:
        for r_ in rows:
            f.write(json.dumps(r_, default=str) + "\n")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["same_runtime", "independent"], required=True)
    ap.add_argument("--per-stratum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--sample", default=None, help="reuse sample.json from another mode's run")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
