"""VLLM-20 on one GPU: the int8 Llama-3.1-8B pin (as `tests/engine/int8_exact_gpu.py` finds or makes it) served on the eager
int8_exact row with PoUW's H1 + fx1 (``pouw:eps1-sm120-h1fx1``, `protocol_options.pouw_h1`), checked on the host.

One process writes `<out>/pouw_h1_gpu.json` and the option's record (`<out>/pouw-h1-calls.json`):

  served    `PROMPTS` greedy, batched (a decode batch of len(PROMPTS)) and the first alone (decode at one row), before the
            option is installed and with it: the token ids are the same, since the arm serves its own y. The engine's
            ``max_num_batched_tokens`` is T (``--rows``), with chunked prefill, so no step puts more rows in a linear.
  calls     the static window (`H1Run`, T rows a call, S = ``--steps``): every (step, linear) call in order at s·U + off(w),
            the served ones recorded and the rest empty, nothing refused; the credit (served rows) beside the padding.
  compile   every served shape compiled to `eps1-sm120-h1fx1` (`H1Run.check`): its Program digest, units, one seed group and
            only full 32 x 32 grid tiles. A shape takes minutes on the host (about 0.13 s a unit), so only `--compile` does it
            here; otherwise `--compile-from <out>/pouw-h1-calls.json` does it on a CPU node from the record, with each call's
            positions against its shape's compiled units, and writes `<out>/pouw_h1_compile.json`.
  z         the last retained calls' exact Y (`engine.int8_exact.int_mm` on the GPU) against the s8 product on the host, and
            a unit's z (`Call.z`) against its cells.
  hook      a listener (`H1Run.listen`) gets every call with the served tensors on the GPU: A int8 [m, k] and s_a f32 [m]
            (the Commit's ``Int8RowQuant_v1`` /0 and /1), the layer's own int8 weight and scales; what it returns is in the
            manifest; steps begin at linear 0, and every step's calls run over the same layers in order.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tests.engine.int8_exact_gpu import generate, pin, scratch_checkpoints

SCHEMA = "verity-vllm/pouw-h1-gpu/v1"
RECORD = "pouw_h1_gpu.json"
PROMPTS = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "The three primary colors are", "Water boils at a temperature of", "Once upon a time, in a small village,",
           "The speed of light is", "In 1492, Columbus", "The largest planet in the solar system is", "To make a cup of tea,",
           "The chemical symbol for gold is", "A haiku about autumn:", "The square root of 144 is", "Photosynthesis is the process"]


def run(a: argparse.Namespace) -> dict[str, Any]:
    from verity.core.protocols.pouw.coins import Challenger
    from verity_vllm import protocol_options as PO
    from verity_vllm.engine.build import build_engine
    from verity_vllm.program.frontend.kernel_readback import int8_exact_of
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.protocol_options import pouw
    from verity_vllm.protocol_options import pouw_h1 as PH

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    target = TargetProfile.from_device().with_overrides(int8_exact="all")
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                           "scheme": PH.SCHEME}
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = pin(a)
        doc["pin"] = {"repo": entry["repo"], "revision": entry["revision"], "source": how}
        eng = build_engine(scratch_checkpoints(a.checkpoints, entry, Path(scratch)), case=a.role, target=target,
                           max_num_seqs=len(PROMPTS), engine_args={"max_num_batched_tokens": a.rows,
                                                                   "enable_chunked_prefill": True, "enable_prefix_caching": False})
    steps_public = a.steps or 2 * (a.max_tokens + 2)
    doc["public"] = {"slots": len(PROMPTS), "rows": a.rows, "steps": steps_public, "max_tokens": a.max_tokens}
    doc["engine"] = {"int8_exact": int8_exact_of(eng.model), "enforce_eager": eng.kwargs.get("enforce_eager")}
    print(f"[h1] engine {json.dumps(doc['engine'], default=str)}", flush=True)

    def serve() -> list[list[int]]:
        return [g["token_ids"] for g in generate(eng.llm, PROMPTS, a.max_tokens) + generate(eng.llm, PROMPTS[:1], a.max_tokens)]

    plain = serve()
    entry_name = f"pouw:{PH.SCHEME}"
    ctx = PO.Context(stage="serve", enabled=(entry_name,), settings={"pouw": {"coins": Challenger(0), "retain_steps": a.retain,
                                                                              "transcript_dir": str(outdir), "rows": a.rows,
                                                                              "steps": steps_public}})
    t0 = time.monotonic()
    comp = PO.install(eng.model, ctx)
    doc["install_s"] = round(time.monotonic() - t0, 1)
    h1 = pouw.h1_run(entry_name)
    layers = PH.layers(eng.model, pouw.linear_types())
    seen: list[str] = []

    def listener(call: Any, a_: Any, s_a: Any, weight: Any, weight_scale: Any) -> dict[str, Any]:
        ok = (a_.is_cuda and a_.dtype == torch.int8 and tuple(a_.shape) == (call.m, call.k) and s_a.dtype == torch.float32
              and tuple(s_a.shape) == (call.m,) and weight is layers[call.layer].weight
              and weight_scale is layers[call.layer].weight_scale)
        if not ok:
            seen.append(f"{call.index}: {a_.device} {a_.dtype} {tuple(a_.shape)} {s_a.dtype} {tuple(s_a.shape)}")
        return {"listener": "pouw_h1_gpu"}

    h1.listen(listener)
    try:
        t0 = time.monotonic()
        with_h1 = serve()
        doc["serve_s"] = round(time.monotonic() - t0, 1)
        doc["served"] = {"same_tokens": with_h1 == plain, "requests": len(plain), "tokens": sum(map(len, plain))}
        h1.close()

        steps: dict[int, list[int]] = {}
        for c in h1.calls:
            steps.setdefault(c.step, []).append(c.layer)
        manifest = h1.manifest()
        doc["hook"] = {"bad": seen[:20], "bound": all(r.get("bound") == {"listener": "pouw_h1_gpu"} for r in manifest),
                       "steps": len(steps), "steps_in_layer_order": all(v == list(range(len(layers))) for v in steps.values()),
                       "leaves": all(r["leaves"] == [0, a.rows] for r in manifest)}
        print(f"[h1] hook {json.dumps(doc['hook'])}", flush=True)
        doc["calls"] = {"count": len(h1.calls), "units": sum(c.units for c in h1.calls), "u_step": h1.u_step,
                        "static": h1.static(), "refused": h1.refused[:20], "steps_served": h1.details["steps_served"],
                        "last_step_complete": h1.details["last_step_complete"],
                        "served_rows_per_step": [max((c.m for c in h1.calls if c.step == s), default=0) for s in range(h1.steps)],
                        "credit": h1.credit()}
        print(f"[h1] calls {json.dumps(doc['calls'])}", flush=True)

        zs = []
        for index, xq, _sa, y in h1.retained:
            call = h1.calls[index]
            w = PH.numpy_of(layers[call.layer].weight).astype(np.int64)
            exact = xq.astype(np.int64) @ w.T
            z0 = call.z(y, 0)
            u = call.compiled().units[0]
            full = np.zeros(PH.padded(call.rows, call.n), np.int64)
            full[:call.m, :call.n] = exact
            zs.append({"index": index, "shape": [call.m, call.k, call.n], "y_exact": bool(np.array_equal(y, exact)),
                       "unit0_z": bool(np.array_equal(z0, [full[u.rows[i], u.cols[j]] for i, j in u.tile.pairs]))})
        doc["z"] = zs

        doc["shapes"] = [list(x) for x in h1.shapes()]
        if a.compile:
            t0 = time.monotonic()
            doc["compile"], doc["compile_ok"] = _check(h1)
            doc["compile_s"] = round(time.monotonic() - t0, 1)
            print(f"[h1] compile {doc['compile_ok']} in {doc['compile_s']} s over {len(h1.shapes())} shapes", flush=True)
        else:
            doc["compile"], doc["compile_ok"] = "deferred to --compile-from", None
    finally:
        comp.close()
    doc["commitment"] = comp.installed[0].commitment()
    doc["ok"] = bool(doc["served"]["same_tokens"] and not doc["hook"]["bad"] and doc["hook"]["bound"] and doc["calls"]["static"]
                     and doc["hook"]["steps_in_layer_order"] and doc["hook"]["leaves"]
                     and doc["calls"]["count"] > 0 and zs and all(z["y_exact"] and z["unit0_z"] for z in zs)
                     and doc["compile_ok"] is not False and doc["engine"]["enforce_eager"] is not False)
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"POUW-H1-GPU {'OK' if doc['ok'] else 'FAIL'} calls={doc['calls']['count']} units={doc['calls']['units']} "
          f"shapes={len(doc['shapes'])} compile={doc['compile_ok']} same_tokens={doc['served']['same_tokens']}", flush=True)
    return doc


def _check(h1: Any) -> tuple[dict[str, Any], bool]:
    try:
        out = h1.check()
        return out, all(v["tiles"] == ["32x32"] and v["groups"] == [[0]] for v in out.values())
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:4000]}, False


def compile_from(calls_file: str, out: str) -> dict[str, Any]:
    """The record's calls compiled on the host: every shape by `H1Run.check`, the window static, and each call (s, w) at
    s·U + off(w) counted in its shapes' compiled units."""
    from verity_vllm.protocol_options import pouw_h1 as PH

    rec = json.loads(Path(calls_file).read_text())
    pub = rec["public"]
    h1 = PH.H1Run(rec["entry"], b"", {}, pub["shapes"], pub["rows"], pub["steps"])
    h1.calls = [PH.Call(c["index"], c["step"], c["layer"], c["module"], c["served"], c["k"], c["n"], c["base"], c["rows"])
                for c in rec["calls"]]
    h1.refused[:] = rec["refused"]
    doc: dict[str, Any] = {"schema": SCHEMA + "/compile", "argv": sys.argv, "calls_file": calls_file, "record": rec["schema"],
                           "commitment": rec["commitment"], "public": pub, "calls": len(h1.calls),
                           "shapes": [list(x) for x in h1.shapes()], "static": h1.static()}
    t0 = time.monotonic()
    doc["compile"], doc["compile_ok"] = _check(h1)
    doc["compile_s"] = round(time.monotonic() - t0, 1)
    bad, total = [], 0
    if doc["compile_ok"]:
        sizes = [len(PH.compiled(k, n, pub["rows"]).units) for n, k in pub["shapes"]]
        offs = [sum(sizes[:w]) for w in range(len(sizes))]
        bad = [c.index for c in h1.calls if c.base != c.step * sum(sizes) + offs[c.layer]]
        total = pub["steps"] * sum(sizes)
    doc["positions"] = {"public": not bad, "bad": bad[:20], "units": total, "committed_units": rec["commitment"].get("units")}
    doc["ok"] = bool(doc["compile_ok"] and doc["static"] and not bad and total == rec["commitment"].get("units"))
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "pouw_h1_compile.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"POUW-H1-COMPILE {'OK' if doc['ok'] else 'FAIL'} calls={doc['calls']} shapes={len(doc['shapes'])} units={total} "
          f"static={doc['static']} in {doc['compile_s']} s", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="int8 Llama served with PoUW's H1 + fx1 on the eager int8_exact row, checked on the host")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="LLAMA31_8B")
    ap.add_argument("--role", default="LLAMA31_8B_INT8")
    ap.add_argument("--repo", default="verity-int8/Llama-3.1-8B-INT8")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--at", help="resolved_at of the pin entry, if one is made (UTC, ISO 8601)")
    ap.add_argument("--by", help="resolved_by of the pin entry, if one is made")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--retain", type=int, default=4, help="the last calls whose operands and exact Y are kept and checked")
    ap.add_argument("--rows", type=int, default=64, help="T, the public rows a call: vLLM's max_num_batched_tokens")
    ap.add_argument("--steps", type=int, default=0, help="S, the public steps of the window (default 2 (max tokens + 2): the "
                                                         "batch and the one alone)")
    ap.add_argument("--compile", action="store_true", help="compile every served shape in this process too")
    ap.add_argument("--compile-from", metavar="CALLS_JSON", help="no GPU: compile a served run's record (pouw-h1-calls.json)")
    a = ap.parse_args(argv)
    if a.compile_from:
        return 0 if compile_from(a.compile_from, a.out)["ok"] else 1
    if not (a.at and a.by):
        ap.error("serving needs --at and --by (the pin entry's, if one is made)")
    return 0 if run(a)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
