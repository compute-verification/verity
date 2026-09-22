"""Minimal reproducer for the B1/p1 ``rmsnorm.same_runtime.v1`` mismatch (contract §11: every failure
gets a reproducer; §15.5 diagnostic order identity -> boundary/snapshot -> kernel path -> arithmetic).

Opens the three authenticated inputs of one committed ``_C.fused_add_rms_norm`` row straight from
the bundle (no live replay), re-runs ``torch.ops._C.fused_add_rms_norm`` on the row alone and
compares bytes with the committed output, once per launch configuration:

* ``VLLM_BATCH_INVARIANT`` unset (what the verifier process had at adapter e2399cd): the CUDA
  launcher picks the vectorised ``fused_add_rms_norm_kernel<bf16, 8>`` (192-thread blocks, 8
  consecutive elements per thread) -> the variance is summed in a different order -> 1-ulp flips.
* ``VLLM_BATCH_INVARIANT=1`` (the profile's declared flag, what the replay ran under): the launcher
  takes the scalar ``fused_add_rms_norm_kernel<bf16, 0>`` with 1024-thread blocks
  (csrc/libtorch_stable/layernorm_kernels.cu, ``batch_invariant_launch``) -> bit-identical.

The C++ side caches the flag, so each configuration is a fresh process: run this twice.

Run ON THE POD (from /workspace/veritor/vllm-poc, PYTHONPATH=.)::

    env -u VLLM_BATCH_INVARIANT /workspace/venv/bin/python -m verity_vllm_adapter.repro_rmsnorm \
        /workspace/results/B1/p1/commit \
        "p=c37c3f0b8363/r=11df0b3a-48c8-508c-9836-eff0876c5ee8/t=49/ph=decode/l=2/op=layers.2.post_attention_layernorm.rmsnorm/port=0"
    VLLM_BATCH_INVARIANT=1 /workspace/venv/bin/python -m verity_vllm_adapter.repro_rmsnorm <same args>

Optional ``--batch N``: also run the row inside an N-row batch (same inputs replicated, the row at
index N-1) to show the result is independent of the batch size under either launch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from verity_vllm import identity
from verity_vllm.capture import CaptureReader


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("canonical_id")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    flag = os.environ.get("VLLM_BATCH_INVARIANT")
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

    reader = CaptureReader(args.run_dir)
    cid = identity.parse(args.canonical_id)
    raw = next(n for n in reader.description["nodes"] if n["op"] == cid.op)
    args_by_name = {a["name"]: a for a in raw["args"]}
    eps = float(raw.get("consts", {}).get("eps", 1e-6))
    runtime_op = raw.get("runtime_op")
    assert runtime_op == "_C.fused_add_rms_norm", runtime_op

    def value_id(arg: dict) -> str:
        return str(identity.value_id(reader.profile_id, cid.request_id, cid.t, cid.phase, cid.layer, arg["op"], arg["port"]))

    x_v = reader.value(value_id(args_by_name["x"]))
    r_v = reader.value(value_id(args_by_name["residual"]))
    w_v = reader.weights.value(str(identity.weight_id(args_by_name["w"]["param"])))
    out_v = reader.value(args.canonical_id)

    def dev(v, shape=None):
        t = torch.frombuffer(bytearray(v.data), dtype=torch.bfloat16)
        return t.reshape(shape or (1, -1)).cuda().clone()

    x, res, w = dev(x_v), dev(r_v), dev(w_v).reshape(-1)
    committed = np.frombuffer(out_v.data, dtype=np.uint16)

    torch.ops._C.fused_add_rms_norm(x, res, w, eps)
    torch.cuda.synchronize()
    got = x[0].cpu().contiguous().view(torch.int16).numpy().astype(np.uint16)
    diff = np.nonzero(got != committed)[0]
    report = {
        "VLLM_BATCH_INVARIANT": flag,
        "canonical_id": args.canonical_id,
        "inputs": {"x": args_by_name["x"]["op"], "residual": args_by_name["residual"]["op"], "w": args_by_name["w"]["param"]},
        "hidden_size": int(committed.size),
        "eps": eps,
        "row_alone": {
            "differing_elements": int(diff.size),
            "first": [{"index": int(i), "committed_hex": f"{committed[i]:04x}", "recomputed_hex": f"{got[i]:04x}"} for i in diff[:8]],
        },
    }
    if args.batch > 1:
        xb = dev(x_v).repeat(args.batch, 1)
        rb = dev(r_v).repeat(args.batch, 1)
        torch.ops._C.fused_add_rms_norm(xb, rb, w, eps)
        torch.cuda.synchronize()
        gb = xb[args.batch - 1].cpu().contiguous().view(torch.int16).numpy().astype(np.uint16)
        report["row_in_batch"] = {"batch": args.batch, "differing_elements_vs_committed": int((gb != committed).sum()),
                                  "differing_elements_vs_row_alone": int((gb != got).sum())}
    print(json.dumps(report, indent=1))
    return 0 if diff.size == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
