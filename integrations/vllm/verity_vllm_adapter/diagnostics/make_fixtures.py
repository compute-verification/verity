"""Authenticated operation fixtures for the compiled-mode kernels examined in §16.6 (CONTRACT §16.8).

    PYTHONPATH=. python -m verity_vllm_adapter.diagnostics.make_fixtures \
        --eager /workspace/diag/eager --compiled /workspace/diag/compiled --compare /workspace/diag/compare \
        --out /workspace/diag/fixtures

One directory per (layer-0) operation with, for BOTH execution modes, the captured operands and
outputs as raw words (.npy; BF16 as uint16) plus the kernel identity (compiled: generated Triton
kernel name + source sha256; eager: the vLLM kernel), and the byte-level comparison of the two
outputs. Operands are the values the kernels actually consumed in the recorded forward pass (the
compiled buffers are snapshotted right after the launch that wrote them and before any reuse; the
eager tensors are the module hook inputs/outputs), so a fixture is self-authenticating against the
capture's ``index.json`` sha256s. Weights other than norm weights are referenced by checkpoint sha.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--eager", required=True)
ap.add_argument("--compiled", required=True)
ap.add_argument("--compare", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
E, C, CMP, OUT = Path(a.eager), Path(a.compiled), Path(a.compare), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
eidx = json.load(open(E / "index.json"))
cidx = json.load(open(C / "index.json"))
kcalls = {k["index"]: k for k in json.load(open(C / "kernel_calls.json"))}
kindex = json.load(open(CMP / "kernels" / "index.json"))
params = dict(np.load(E / "small_params.npz"))
for k, v in np.load(C / "small_params.npz").items():  # buffers (rope cache) were added after the eager capture
    if k in params:
        assert np.array_equal(params[k], v), k
    params.setdefault(k, v)


def cev(kidx: int, arg: str) -> dict:
    return next(e for e in cidx["events"] if e.get("kernel_call_index") == kidx and e.get("arg") == arg and e["kind"] == "inductor")


def eev(node: str, port: int, kind: str, occ: int = 0) -> dict:
    lst = [e for e in eidx["events"] if e["node"] == node and e["port"] == port and e["kind"] == kind]
    return lst[occ]


def ksrc(kernel: str) -> dict:
    for k, v in kindex.items():
        if v["kernel"] == kernel:
            return v
    raise KeyError(kernel)


def save(dst: Path, arr: np.ndarray) -> dict:
    np.save(dst, arr)
    return {"file": dst.name, "shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}


def diff(a: np.ndarray, b: np.ndarray) -> dict:
    af, bf = a.reshape(-1), b.reshape(-1)
    d = np.nonzero(af != bf)[0]
    out = {"words": int(af.size), "n_diff": int(d.size)}
    if d.size and a.dtype == np.uint16:
        ulps = np.abs(af[d].astype(np.int64) - bf[d].astype(np.int64))
        out["ulp_hist"] = {str(int(k)): int(v) for k, v in zip(*np.unique(ulps, return_counts=True), strict=False)}
        out["first"] = {"flat": int(d[0]), "eager_bits": f"0x{int(af[d[0]]):04x}", "compiled_bits": f"0x{int(bf[d[0]]):04x}"}
    return out


FIXTURES = [
    {
        "id": "layers.0.input_layernorm+embedding",
        "compiled": {"kernel_call": 0, "operands": {"input_ids": "arg0_1"}, "outputs": {"embedding": "buf0", "normed": "buf3"},
                     "params": {"weight": "model.layers.0.input_layernorm.weight"}},
        "eager": {"operands": {"x": ("model.layers.0.input_layernorm", 0, "module_in")}, "outputs": {"normed": ("model.layers.0.input_layernorm", 0, "module_out")},
                  "kernel": "vllm.model_executor.determinism.batch_invariant._rms_norm_kernel (Triton) via RMSNorm.forward_cuda; embedding: VocabParallelEmbedding (F.embedding)"},
        "compare": [("normed", "normed")],
    },
    {
        "id": "layers.0.self_attn.rotary_emb",
        "compiled": {"kernel_call": 2, "operands": {"qkv": "buf2", "positions": "arg5_1"}, "outputs": {"k": "buf5", "q": "buf6"},
                     "params": {"cos_sin_cache": "buffer:model.layers.0.self_attn.rotary_emb.cos_sin_cache"}},
        "eager": {"operands": {"positions": ("model.layers.0.self_attn.rotary_emb", 0, "module_in"), "q": ("model.layers.0.self_attn.rotary_emb", 1, "module_in"), "k": ("model.layers.0.self_attn.rotary_emb", 2, "module_in")},
                  "outputs": {"q": ("model.layers.0.self_attn.rotary_emb", 0, "module_out"), "k": ("model.layers.0.self_attn.rotary_emb", 1, "module_out")},
                  "kernel": "vllm::rotary_embedding_kernel<bf16, bf16, true> (CUDA custom op, NeoX)"},
        "compare": [("q", "q"), ("k", "k")],
    },
    {
        "id": "layers.0.post_attention_layernorm(fused_add_rms_norm)",
        "compiled": {"kernel_call": 4, "operands": {"x": "buf0", "residual": "arg5_1"}, "outputs": {"normed": "buf4"},
                     "params": {"weight": "model.layers.0.post_attention_layernorm.weight"}},
        "eager": {"operands": {"x": ("model.layers.0.post_attention_layernorm", 0, "module_in"), "residual": ("model.layers.0.post_attention_layernorm", 1, "module_in")},
                  "outputs": {"normed": ("model.layers.0.post_attention_layernorm", 0, "module_out"), "residual_out": ("model.layers.0.post_attention_layernorm", 1, "module_out")},
                  "kernel": "vllm::fused_add_rms_norm_kernel<bf16> (CUDA custom op) via rms_norm_batch_invariant(residual=...)"},
        "compare": [("normed", "normed")],
    },
    {
        "id": "layers.0.mlp.act_fn(silu_and_mul)",
        "compiled": {"kernel_call": 6, "operands": {"gate_up": "buf3"}, "outputs": {"out": "buf7"}, "params": {}},
        "eager": {"operands": {"gate_up": ("model.layers.0.mlp.act_fn", 0, "module_in")}, "outputs": {"out": ("model.layers.0.mlp.act_fn", 0, "module_out")},
                  "kernel": "vllm::act_and_mul_kernel<bf16, silu> (CUDA custom op)"},
        "compare": [("out", "out")],
    },
    {
        "id": "layers.1.input_layernorm(fused_add_rms_norm, two outputs)",
        "compiled": {"kernel_call": 8, "operands": {"down_proj_out": "buf6", "o_proj_out": "buf0", "residual_in": "arg5_1"}, "outputs": {"residual_out": "buf9", "normed": "buf12"},
                     "params": {"weight": "model.layers.1.input_layernorm.weight"}},
        "eager": {"operands": {"x": ("model.layers.1.input_layernorm", 0, "module_in"), "residual": ("model.layers.1.input_layernorm", 1, "module_in")},
                  "outputs": {"normed": ("model.layers.1.input_layernorm", 0, "module_out"), "residual_out": ("model.layers.1.input_layernorm", 1, "module_out")},
                  "kernel": "vllm::fused_add_rms_norm_kernel<bf16> (CUDA custom op)"},
        "compare": [("normed", "normed"), ("residual_out", "residual_out")],
    },
]

index = []
for fx in FIXTURES:
    d = OUT / fx["id"].replace("(", "_").replace(")", "").replace("+", "_").replace(", ", "_").replace(" ", "_")
    (d / "compiled").mkdir(parents=True, exist_ok=True)
    (d / "eager").mkdir(parents=True, exist_ok=True)
    kc = kcalls[fx["compiled"]["kernel_call"]]
    ks = ksrc(kc["kernel"])
    meta = {
        "fixture": fx["id"],
        "request_id": cidx["result"]["request_id"],
        "prompt_len": cidx["result"]["prompt_len"],
        "compiled": {"kernel": kc["kernel"], "source_sha256": ks["sha256"], "source_file": f"{kc['kernel']}__{ks['sha256'][:8]}.py", "call_args": kc["call_args"],
                     "kernel_call_index": kc["index"], "effective": json.load(open(C / "effective.json"))["compilation"], "operands": {}, "outputs": {}, "params": {}},
        "eager": {"kernel": fx["eager"]["kernel"], "effective": json.load(open(E / "effective.json"))["compilation"], "operands": {}, "outputs": {}},
        "comparison": {},
    }
    shutil.copy2(CMP / "kernels" / meta["compiled"]["source_file"], d / "compiled" / meta["compiled"]["source_file"])
    arrs_c, arrs_e = {}, {}
    for name, arg in fx["compiled"]["operands"].items():
        ev = cev(kc["index"], arg)
        arr = np.load(C / "values" / ev["file"])
        meta["compiled"]["operands"][name] = dict(save(d / "compiled" / f"in_{name}.npy", arr), arg=arg, capture_sha256=ev["sha256"])
    for name, arg in fx["compiled"]["outputs"].items():
        ev = cev(kc["index"], arg)
        arr = np.load(C / "values" / ev["file"])
        arrs_c[name] = arr
        meta["compiled"]["outputs"][name] = dict(save(d / "compiled" / f"out_{name}.npy", arr), arg=arg, capture_sha256=ev["sha256"])
    for name, pname in fx["compiled"]["params"].items():
        arr = params[pname]
        meta["compiled"]["params"][name] = dict(save(d / f"{name}.npy", arr), param=pname)
    for name, (node, port, kind) in fx["eager"]["operands"].items():
        ev = eev(node, port, kind)
        arr = np.load(E / "values" / ev["file"])
        meta["eager"]["operands"][name] = dict(save(d / "eager" / f"in_{name}.npy", arr), node=node, port=port, kind=kind, capture_sha256=ev["sha256"])
    for name, (node, port, kind) in fx["eager"]["outputs"].items():
        ev = eev(node, port, kind)
        arr = np.load(E / "values" / ev["file"])
        arrs_e[name] = arr
        meta["eager"]["outputs"][name] = dict(save(d / "eager" / f"out_{name}.npy", arr), node=node, port=port, kind=kind, capture_sha256=ev["sha256"])
    for ce, ee in fx["compare"]:
        meta["comparison"][f"{ee}"] = diff(arrs_e[ee], arrs_c[ce])
    (d / "fixture.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    index.append({"id": fx["id"], "dir": d.name, "compiled_kernel": kc["kernel"], "compiled_source_sha256": ks["sha256"], "comparison": meta["comparison"]})

sums = []
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.name != "sha256sums.txt":
        sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}")
(OUT / "sha256sums.txt").write_text("\n".join(sums) + "\n")
(OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n")
total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
print(f"fixtures: {len(index)} dirs, {total/1e6:.1f} MB")
for i in index:
    print(i["id"], i["compiled_kernel"], json.dumps(i["comparison"])[:300])
