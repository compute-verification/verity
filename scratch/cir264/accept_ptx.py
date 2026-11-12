"""Compile, with no card, the Triton kernels vLLM's `rejection_sample` launches, and keep every stage of each.

The pin's own `rejection_sample` is called on CPU tensors of the dtypes and shapes the engine passes it (one request,
K draft tokens, V logits a row), with every launch turned into Triton's warmup: the binder specializes on the real
arguments (pointer alignment, an integer's divisibility by 16, an integer equal to 1, a `None`) exactly as a launch
does, the kernel is compiled for the target, and nothing runs.  No vLLM is installed: the pin's two source files are
imported unchanged under a `vllm.triton_utils` that holds Triton's modules and nothing else.

Kept under --out, a kernel: its TTIR, TTGIR, LLVM IR, PTX, cubin, the SASS `cuobjdump -sass` prints, the signature,
constexprs, attributes and options it was compiled under, and each file's sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

KERNEL_FILES = (
    "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py",
    "vllm/v1/worker/gpu/sample/gumbel.py",
)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin-src", required=True, help="a directory holding the pin's KERNEL_FILES at their paths")
    ap.add_argument("--pin", required=True, help="the pin's commit, recorded")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=154880)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--reqs", type=int, default=1)
    ap.add_argument("--max-reqs", type=int, default=1024)
    ap.add_argument("--arch", type=int, default=90)
    ap.add_argument("--logits-dtype", default="float32")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pin_src = Path(a.pin_src).resolve()

    import torch
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
    from triton import knobs
    from triton.backends.compiler import GPUTarget
    from triton.runtime import driver
    from triton.runtime.jit import JITFunction

    target = GPUTarget("cuda", a.arch, 32)

    class NoCard:
        """What the binder and the compiler ask a driver, answered for `target` with no device."""

        def get_current_target(self):
            return target

        def get_current_device(self):
            return 0

        def get_current_stream(self, device):
            return 0

    driver.set_active(NoCard())

    # the pin's two files import `vllm.triton_utils` and each other; everything else of vLLM stays out
    vllm = types.ModuleType("vllm")
    vllm.__path__ = [str(pin_src / "vllm")]
    tu = types.ModuleType("vllm.triton_utils")
    tu.HAS_TRITON, tu.tl, tu.tldevice, tu.triton = True, tl, tldevice, triton
    sys.modules["vllm"], sys.modules["vllm.triton_utils"] = vllm, tu
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as rsu

    compiled: list[dict] = []
    real_run = JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        dev = driver.active.get_current_device()
        _, _, _, backend, binder = self.device_caches[dev]
        bound, spec, opts = binder(*args, **kwargs)
        kernel = real_run(self, *args, grid=grid, warmup=True, **kwargs)
        compiled.append({"fn": self, "grid": list(grid), "kernel": kernel, "specialization": spec,
                         "names": list(bound.keys())})
        return kernel

    JITFunction.run = run

    K, V, R = a.steps, a.vocab, a.reqs
    L = R * (K + 1)
    args = dict(
        target_logits=torch.zeros(L, V, dtype=getattr(torch, a.logits_dtype)),
        draft_logits=None,
        draft_sampled=torch.zeros(L, dtype=torch.int32),
        cu_num_logits=torch.arange(0, L + 1, K + 1, dtype=torch.int32),
        pos=torch.zeros(L, dtype=torch.int64),
        idx_mapping=torch.arange(R, dtype=torch.int64),
        expanded_idx_mapping=torch.arange(R, dtype=torch.int64).repeat_interleave(K + 1),
        expanded_local_pos=torch.arange(K + 1, dtype=torch.int32).repeat(R),
        temperature=torch.ones(a.max_reqs, dtype=torch.float32),
        seed=torch.zeros(a.max_reqs, dtype=torch.int64),
        num_speculative_steps=K,
        synthetic_conditional_rates=None,
        use_fp64=False,
        use_block_verification=False,
    )
    tensors = {k: {"dtype": str(v.dtype), "shape": list(v.shape), "stride": list(v.stride()),
                   "data_ptr_mod_16": v.data_ptr() % 16}
               for k, v in args.items() if isinstance(v, torch.Tensor)}
    rsu.rejection_sample(**args)

    cuobjdump = knobs.nvidia.cuobjdump.path
    ptxas = knobs.nvidia.ptxas.path
    manifest = {
        "pin": a.pin,
        "sources": {},
        "triton": triton.__version__,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "target": {"backend": target.backend, "arch": target.arch, "warp_size": target.warp_size},
        "ptxas": {"path": str(ptxas), "version": subprocess.check_output([ptxas, "--version"]).decode().strip(),
                  "sha256": sha256(Path(ptxas).read_bytes())},
        "cuobjdump": str(cuobjdump),
        "libdevice": None,
        "call": {"vocab": V, "steps": K, "reqs": R, "max_reqs": a.max_reqs, "tensors": tensors,
                 "draft_logits": None, "synthetic_conditional_rates": None, "use_fp64": False,
                 "use_block_verification": False},
        "kernels": [],
    }
    for rel in KERNEL_FILES:
        b = (pin_src / rel).read_bytes()
        (out / "src").mkdir(exist_ok=True)
        (out / "src" / Path(rel).name).write_bytes(b)
        manifest["sources"][rel] = {"sha256": sha256(b), "git_blob": git_blob(b), "bytes": len(b)}

    for c in compiled:
        fn, k = c["fn"], c["kernel"]
        name = fn.fn.__name__
        d = out / name
        d.mkdir(exist_ok=True)
        md = k.metadata._asdict()
        files = {}
        for ext in ("ttir", "ttgir", "llir", "ptx"):
            b = k.asm[ext].encode()
            (d / f"{name}.{ext}").write_bytes(b)
            files[ext] = {"sha256": sha256(b), "bytes": len(b)}
        cubin = k.asm["cubin"]
        (d / f"{name}.cubin").write_bytes(cubin)
        files["cubin"] = {"sha256": sha256(cubin), "bytes": len(cubin)}
        sass = subprocess.run([cuobjdump, "-sass", str(d / f"{name}.cubin")], capture_output=True)
        if sass.returncode == 0:
            (d / f"{name}.sass").write_bytes(sass.stdout)
            files["sass"] = {"sha256": sha256(sass.stdout), "bytes": len(sass.stdout)}
        else:
            files["sass"] = {"error": sass.stderr.decode(errors="replace")[-2000:]}
        if manifest["libdevice"] is None:
            lib = dict(md.get("extern_libs") or ()).get("libdevice")
            if lib:
                manifest["libdevice"] = {"path": lib, "sha256": sha256(Path(lib).read_bytes())}
        manifest["kernels"].append({
            "name": name,
            "entry": md.get("name"),
            "grid": c["grid"],
            "params": [{"name": n, "specialization": [str(x) for x in s] if isinstance(s, (tuple, list)) else str(s)}
                       for n, s in zip(c["names"], c["specialization"])],
            "signature": {kk: str(v) for kk, v in k.src.signature.items()},
            "constexprs": {str(kk): str(v) for kk, v in k.src.constants.items()},
            "attrs": {str(kk): str(v) for kk, v in k.src.attrs.items()},
            "options": {kk: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v)
                        for kk, v in md.items() if kk not in ("target",)},
            "files": files,
        })
        print(f"{name}: grid {c['grid']} num_warps {md.get('num_warps')} ptx {files['ptx']['bytes']} B "
              f"sha256 {files['ptx']['sha256'][:16]} cubin {files['cubin']['bytes']} B sass "
              f"{files['sass'].get('bytes', 'FAILED')}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"kernels compiled: {len(compiled)}; triton {triton.__version__}; ptxas {manifest['ptxas']['version']!r}")
    return 0 if compiled and all("bytes" in k["files"]["sass"] for k in manifest["kernels"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
