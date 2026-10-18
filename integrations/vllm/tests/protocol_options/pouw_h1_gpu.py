"""VLLM-20 on one GPU: the int8 Llama-3.1-8B pin (as `tests/engine/int8_exact_gpu.py` finds or makes it) served on the eager
int8_exact row with PoUW's H1 + fx1 (``pouw:eps1-sm120-h1fx1``, `protocol_options.pouw_h1`), checked on the host.

One process writes `<out>/pouw_h1_gpu.json` and the option's record (`<out>/pouw-h1-calls.json`):

  served    `PROMPTS` greedy, batched (a decode batch of len(PROMPTS)) and the first alone (decode at one row), before the
            option is installed and with it: the token ids are the same, since the arm serves its own y.
  calls     every served linear call the option recorded, per shape (m, k, n), with consecutive position ranges.
  compile   every served shape compiled to `eps1-sm120-h1fx1` (`H1Run.check`): its Program digest, units, one seed group and
            only full 32 x 32 grid tiles.
  z         the last retained calls' exact Y (`engine.int8_exact.int_mm` on the GPU) against the s8 product on the host, and
            a unit's z (`Call.z`) against its cells.
  roots     each call's root(B) is its layer's weight root (`pouw_h1.weight_root`), and no two calls share a root(A).
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
                           max_num_seqs=len(PROMPTS))
    doc["engine"] = {"int8_exact": int8_exact_of(eng.model), "enforce_eager": eng.kwargs.get("enforce_eager")}
    print(f"[h1] engine {json.dumps(doc['engine'], default=str)}", flush=True)

    def serve() -> list[list[int]]:
        return [g["token_ids"] for g in generate(eng.llm, PROMPTS, a.max_tokens) + generate(eng.llm, PROMPTS[:1], a.max_tokens)]

    plain = serve()
    entry_name = f"pouw:{PH.SCHEME}"
    ctx = PO.Context(stage="serve", enabled=(entry_name,), settings={"pouw": {"coins": Challenger(0), "retain_steps": a.retain,
                                                                              "transcript_dir": str(outdir)}})
    t0 = time.monotonic()
    comp = PO.install(eng.model, ctx)
    doc["install_s"] = round(time.monotonic() - t0, 1)
    h1 = pouw.h1_run(entry_name)
    try:
        t0 = time.monotonic()
        with_h1 = serve()
        doc["serve_s"] = round(time.monotonic() - t0, 1)
        doc["served"] = {"same_tokens": with_h1 == plain, "requests": len(plain), "tokens": sum(map(len, plain))}

        layers = PH.layers(eng.model, pouw.linear_types())
        bad_b = [c.index for c in h1.calls if c.roots[1] != h1.weight_roots[c.layer]]
        roots_a = [c.roots[0] for c in h1.calls]
        doc["roots"] = {"root_b_is_the_layers": not bad_b, "distinct_root_a": len(set(roots_a)) == len(roots_a), "kind": PH.ROOTS}
        bases = [c.base for c in h1.calls]
        doc["calls"] = {"count": len(h1.calls), "units": h1.next_base, "consecutive": bases == [0, *np.cumsum([c.units for c in h1.calls])[:-1].tolist()],
                        "shapes": h1.details["shapes"], "padded_rows": h1.details["padded_rows"]}
        print(f"[h1] calls {json.dumps(doc['calls'])}", flush=True)

        zs = []
        for index, xq, _sa, y in h1.retained:
            call = h1.calls[index]
            w = PH.numpy_of(layers[call.layer].weight).astype(np.int64)
            exact = xq.astype(np.int64) @ w.T
            z0 = call.z(y, 0)
            u = call.compiled().units[0]
            full = np.zeros(PH.padded(call.m, call.n), np.int64)
            full[:call.m, :call.n] = exact
            zs.append({"index": index, "shape": [call.m, call.k, call.n], "y_exact": bool(np.array_equal(y, exact)),
                       "unit0_z": bool(np.array_equal(z0, [full[u.rows[i], u.cols[j]] for i, j in u.tile.pairs]))})
        doc["z"] = zs

        t0 = time.monotonic()
        try:
            doc["compile"] = h1.check()
            doc["compile_ok"] = all(v["tiles"] == ["32x32"] and v["groups"] == [[0]] for v in doc["compile"].values())
        except Exception as e:  # noqa: BLE001
            doc["compile"], doc["compile_ok"] = {"error": f"{type(e).__name__}: {e}"[:4000]}, False
        doc["compile_s"] = round(time.monotonic() - t0, 1)
        print(f"[h1] compile {doc['compile_ok']} in {doc['compile_s']} s over {len(h1.shapes())} shapes", flush=True)
    finally:
        comp.close()
    doc["commitment"] = comp.installed[0].commitment()
    doc["ok"] = bool(doc["served"]["same_tokens"] and doc["roots"]["root_b_is_the_layers"] and doc["roots"]["distinct_root_a"]
                     and doc["calls"]["consecutive"] and doc["calls"]["count"] > 0 and zs and all(z["y_exact"] and z["unit0_z"] for z in zs)
                     and doc["compile_ok"] and doc["engine"]["enforce_eager"] is not False)
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"POUW-H1-GPU {'OK' if doc['ok'] else 'FAIL'} calls={doc['calls']['count']} units={doc['calls']['units']} "
          f"shapes={len(doc['compile']) if doc['compile_ok'] else doc['compile']} same_tokens={doc['served']['same_tokens']}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="int8 Llama served with PoUW's H1 + fx1 on the eager int8_exact row, checked on the host")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="LLAMA31_8B")
    ap.add_argument("--role", default="LLAMA31_8B_INT8")
    ap.add_argument("--repo", default="verity-int8/Llama-3.1-8B-INT8")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--at", required=True, help="resolved_at of the pin entry, if one is made (UTC, ISO 8601)")
    ap.add_argument("--by", required=True, help="resolved_by of the pin entry, if one is made")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--retain", type=int, default=4, help="the last calls whose operands and exact Y are kept and checked")
    return 0 if run(ap.parse_args(argv))["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
