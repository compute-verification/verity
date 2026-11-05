"""A GPU smoke of the POUS option (`protocols.pous`, ``pous:p2-16448/v4``) over Llama-3.2-1B int8's real linears on one
GPU, timed (not collected; `pous_gpu.sh` runs it under a lease).

The model stands in for vLLM's at TP 1: each decoder layer's four linears as vLLM's Llama holds them (``qkv_proj``, q, k and v
concatenated by rows; ``o_proj``; ``gate_up_proj``, gate then up; ``down_proj``), each the int8 [N, K] weight and f32 [N, 1]
``weight_scale`` of the registered int8 pin (`engine.int8_checkpoint`, ``LLAMA32_1B_INT8``) that
`engine.int8_vllm.Int8PlainLinearMethod` holds, applied by that method's product (`engine.int8_exact.plain_linear`).  The
option is installed through `protocols.install` with `pous.linear_types` the stand-in's type, ``p2_lib`` the
`benchmarks/torch_gpu/p2_stage/build.sh` library and the salt source a public test coin (no salt of record).

    PYTHONPATH=<root>:<root>/integrations/vllm:<root>/tools/config:<root>/tools/research/src \\
      python tests/protocols/pous_gpu.py --snapshot SNAP --lib LIB --out DIR [--m 1,256] [--reps 5]

Records in ``DIR/pous_gpu.json`` and checks:

  plain     one step (every linear once, at each M of ``--m``) with the weights resident, ``--reps`` times;
  install   `protocols.install`'s seconds (phase 1's commitment, then the encode of ``C*`` on the GPU, its table and root),
            and the option's ``info()``: the encode's own timing, the blocks, what an accepted period certifies;
  released  every linear's weight is empty while installed: only ``C*`` is held (the HBM before and after);
  served    the same steps with every linear's weight decoded from ``C*`` at its use (the staging decode), each output
            equal to the plain step's to the bit;
  decode    ``weights()``, every weight decoded from ``C*`` once (one step's at-use decode, without the GEMMs), each
            tensor's SHA-256 equal to the plaintext's;
  restored  after close every weight is the plaintext again, by SHA-256.

The verdict line is ``pous_gpu: PASS`` only when every check holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from verity_vllm import config
from verity_vllm import protocols as PO
from verity_vllm.engine import int8_exact as E
from verity_vllm.protocols import pous

ENTRY = f"pous:{pous.DEFAULT_SCHEME}"
RECORD = "pous_gpu.json"
#: public, so it stands for no auditor's coin: sha512("not-a-secret: verity/vllm-pous/gpu-smoke")[:32]
TEST_SOURCE = hashlib.sha512(b"not-a-secret: verity/vllm-pous/gpu-smoke").hexdigest()[:64]
MERGED = (("self_attn.qkv_proj", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
          ("self_attn.o_proj", ("self_attn.o_proj",)),
          ("mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj")),
          ("mlp.down_proj", ("mlp.down_proj",)))


class Plain:
    """`Int8PlainLinearMethod.apply`'s product: one activation scale per row, the int8 GEMM, both scales."""

    def apply(self, layer: Any, x: torch.Tensor, bias: Any = None) -> torch.Tensor:
        return E.plain_linear(x, layer.weight, layer.weight_scale)


class Linear:
    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        self.quant_method = Plain()


class Model:
    def __init__(self, layers: dict[str, Linear]) -> None:
        self.layers = layers

    def named_modules(self):
        yield "", self
        yield from self.layers.items()

    def named_parameters(self):
        for n, m in self.layers.items():
            yield f"{n}.weight", m.weight
            yield f"{n}.weight_scale", m.weight_scale

    def named_buffers(self):
        return iter(())


def load(snapshot: Path, device: torch.device) -> tuple[Model, dict[str, Any]]:
    """vLLM's linears of the pin, on ``device``, and what of the checkpoint is not in them."""
    layers: dict[str, Linear] = {}
    with safe_open(str(snapshot / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        n = 1 + max(int(k.split(".")[2]) for k in keys if k.startswith("model.layers."))
        used: set[str] = set()
        for i in range(n):
            p = f"model.layers.{i}."
            for merged, parts in MERGED:
                names = [p + s + ".weight" for s in parts]
                w = torch.cat([f.get_tensor(k) for k in names]).contiguous()
                s = torch.cat([f.get_tensor(k + "_scale") for k in names]).contiguous()
                if w.dtype != torch.int8 or s.dtype != torch.float32 or s.shape != (w.shape[0], 1):
                    raise SystemExit(f"{p + merged}: {w.dtype} {tuple(w.shape)} with {s.dtype} {tuple(s.shape)} scales is "
                                     "not the int8 pin's linear")
                layers[p + merged] = Linear(w.to(device), s.to(device))
                used.update(names + [k + "_scale" for k in names])
        rest = {k: (str(f.get_slice(k).get_dtype()), f.get_slice(k).get_shape()) for k in sorted(keys - used)}
    dsize = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1}
    other = sum(dsize.get(d, 0) * int(torch.Size(s).numel()) for d, s in rest.values())
    return Model(layers), {"decoder_layers": n, "not_in_w": {k: [d, list(s)] for k, (d, s) in rest.items()}, "not_in_w_bytes": other}


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def timed(f: Any, reps: int) -> tuple[Any, list[float]]:
    out, secs = None, []
    for _ in range(reps):
        out = None
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = f()
        torch.cuda.synchronize()
        secs.append(round(time.perf_counter() - t, 6))
    return out, secs


def summary(secs: list[float]) -> dict[str, Any]:
    return {"s": secs, "median_s": statistics.median(secs), "min_s": min(secs)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", required=True, help="the int8 pin's snapshot directory (model.safetensors)")
    ap.add_argument("--lib", required=True, help="libp2v1.so's directory (p2_stage/build.sh's lib/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", default="1,256", help="the step's rows, comma separated")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda", torch.cuda.current_device())
    pous.linear_types = lambda: (Linear,)

    t = time.perf_counter()
    model, ckpt = load(Path(a.snapshot), dev)
    load_s = round(time.perf_counter() - t, 3)
    names = list(model.layers)
    ks = {n: m.weight.shape[1] for n, m in model.layers.items()}
    w_sha = {n: sha(m.weight) for n, m in model.layers.items()}
    w_bytes = sum(m.weight.numel() for m in model.layers.values())
    g = torch.Generator(device=dev).manual_seed(0)
    ms = [int(v) for v in a.m.split(",")]
    xs = {(m, k): torch.randn(m, k, generator=g, device=dev).to(torch.bfloat16) for m in ms for k in sorted(set(ks.values()))}

    def step(m: int) -> dict[str, torch.Tensor]:
        return {n: layer.quant_method.apply(layer, xs[(m, ks[n])]) for n, layer in model.layers.items()}

    plain = {m: timed(lambda m=m: step(m), a.reps) for m in ms}
    hbm = {"before": torch.cuda.memory_allocated(dev)}
    ctx = PO.Context(stage="serve", enabled=config.parse_protocols([ENTRY]),
                     settings={"pous": {"salt_source": TEST_SOURCE, "p2_lib": a.lib, "experimental": True}})
    t = time.perf_counter()
    comp = PO.install(model, ctx)
    install_s = round(time.perf_counter() - t, 3)
    try:
        opt = comp.get("pous")
        released = all(layer.weight.numel() == 0 for layer in model.layers.values())
        hbm["installed"] = torch.cuda.memory_allocated(dev)
        served = {m: timed(lambda m=m: step(m), a.reps) for m in ms}
        exact = {m: [n for n in names if not torch.equal(served[m][0][n], plain[m][0][n])] for m in ms}
        weights, decode_s = timed(opt.weights, a.reps)
        decoded = [n for n in names if sha(weights[n + ".weight"]) != w_sha[n]]
        weights = None
        hbm["peak"] = torch.cuda.max_memory_allocated(dev)
        info, commits = comp.info()[ENTRY], opt.commitment()
    finally:
        comp.close()
    restored = [n for n, layer in model.layers.items() if sha(layer.weight) != w_sha[n]]

    ok = released and not any(exact.values()) and not decoded and not restored
    doc = {
        "verdict": "PASS" if ok else "FAIL", "entry": ENTRY, "snapshot": str(a.snapshot), "device": torch.cuda.get_device_name(dev),
        "torch": torch.__version__, "cuda": torch.version.cuda, "salt_source": "public test coin (TEST_SOURCE)",
        "linears": len(names), "w_bytes": w_bytes, "checkpoint": ckpt, "load_s": load_s, "install_s": install_s,
        "info": info, "commits": commits, "hbm_bytes": hbm, "released": released,
        "plain_step": {str(m): summary(plain[m][1]) for m in ms},
        "served_step": {str(m): summary(served[m][1]) for m in ms},
        "decode_all": summary(decode_s), "decode_bytes_per_s": round(w_bytes / statistics.median(decode_s)),
        "not_bit_exact": {str(m): v for m, v in exact.items()}, "decoded_mismatch": decoded, "restored_mismatch": restored,
    }
    (out / RECORD).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    enc = info["encode"]
    steps = " ".join(f"M={m} plain {doc['plain_step'][str(m)]['median_s'] * 1e3:.2f} ms served "
                     f"{doc['served_step'][str(m)]['median_s'] * 1e3:.2f} ms" for m in ms)
    print(f"pous_gpu: {doc['verdict']} {doc['device']}: {len(names)} linears, W {w_bytes} B in {info['used_blocks']} of "
          f"{info['blocks']} blocks; install {install_s} s (encode wall {enc['wall_s']} s, device {enc['device_encode_s']} s, "
          f"root {enc['root_s']} s); decode of W {doc['decode_all']['median_s'] * 1e3:.2f} ms; {steps}; "
          f"certifies {info['certifies']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
