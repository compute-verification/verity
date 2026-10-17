"""VLLM-18 on one GPU: a BF16 pin's int8 checkpoint served with `TargetProfile.int8_exact = "plain"`, checked end to end.

One process writes `<out>/int8_exact_gpu.json` (and `<out>/pin_entry.json`, the int8 pin's checkpoints.json entry):

  pin       `verity-vllm int8-checkpoint pin` in-process: the int8 snapshot made from the BF16 pin into the HF cache, its entry (every
            file's sha256, the pinned revision).  The engine reads it through a scratch copy of checkpoints.json with the entry
            appended as `downloaded` (the repo's manifest is append-only and gets the entry by hand, after review).
  engine    `engine.build.build_engine(case=ROLE, target=from_device() + int8_exact="plain")`, which refuses a target mismatch
            (`int8_exact_arm_of` included).
  applied   `int8_exact.applied_record`: every linear's method, arm and calls, before and after generation, against vLLM's own
            `LinearBase` modules: every one runs `Int8PlainLinearMethod` (arm "plain") and served calls during generation.
  generate  greedy (temperature 0) on PROMPTS, batched and one by one; the texts, token ids, and which prompts' tokens differ
            between the two (the per-call activation scale makes a row's value depend on its batch).
  equal     forward hooks on the first four linears and the last one keep (x, y) of their first calls in the batched run and in
            the first solo run (prefill and decode, so the M < 17 padding too).  Each y, bit for bit, against
              cpu    `int8_exact.plain_linear` on the CPU (torch `_int_mm` on the CPU, every row)
              ref    `tests/engine/int8_reference.plain` (NumPy from egemm_ref.py, `--ref-rows` rows, the whole call's scale)
              rerun  `int8_exact.plain_linear` again on the GPU on the same weights

    cd integrations/vllm; PY=/workspace/jobs/venv312/bin/python3     # PYTHONPATH=.:../..:../../tools/research/src
    $PY tests/engine/int8_exact_gpu.py --out DIR --at 2026-10-17T00:00:00Z --by LANE
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tests.engine import int8_reference as R
from verity_vllm.engine import int8_checkpoint as I
from verity_vllm.engine import int8_exact as E

SCHEMA = "verity-vllm/int8-exact-gpu/v1"
RECORD = "int8_exact_gpu.json"
PROMPTS = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "The three primary colors are", "Water boils at a temperature of", "Once upon a time, in a small village,"]


def pin(a: argparse.Namespace) -> dict[str, Any]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = I.main(I.PinOptions(base_role=a.base_role, role=a.role, repo=a.repo, checkpoints=a.checkpoints, hf_home=a.hf_home,
                                 by=a.by, at=a.at))
    if rc:
        raise SystemExit(f"int8-checkpoint pin exited {rc}")
    return json.loads(out.getvalue())


def scratch_checkpoints(checkpoints: str, entry: dict[str, Any], scratch: Path) -> Path:
    doc = json.loads(Path(checkpoints).read_text())
    doc["checkpoints"].append({**entry, "downloaded": True})
    p = scratch / "checkpoints.json"
    p.write_text(json.dumps(doc, indent=1) + "\n")
    return p


class Taps:
    """Forward hooks keeping (x, y) of a module's first `keep` calls per phase, on the CPU."""

    def __init__(self, model: torch.nn.Module, names: list[str], keep: int):
        self.phase: str | None = None
        self.keep = keep
        self.got: dict[tuple[str, str], list[tuple[torch.Tensor, torch.Tensor]]] = {}
        mods = dict(model.named_modules())
        self.handles = [mods[n].register_forward_hook(self._hook(n)) for n in names]

    def _hook(self, name: str):
        def hook(mod, args, out):
            if self.phase is None:
                return
            calls = self.got.setdefault((self.phase, name), [])
            if len(calls) < self.keep:
                y = out[0] if isinstance(out, tuple) else out
                x = args[0]
                calls.append((x.detach().reshape(-1, x.shape[-1]).cpu(), y.detach().reshape(-1, y.shape[-1]).cpu()))
        return hook

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def _words(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def compare(mod: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, ref_rows: int) -> dict[str, Any]:
    w, s = mod.weight, mod.weight_scale
    m = x.shape[0]
    gpu = E.plain_linear(x.to(w.device), w, s).cpu()
    cpu = E.plain_linear(x, w.cpu(), s.cpu())
    rows = np.unique(np.linspace(0, m - 1, min(m, ref_rows)).round().astype(np.int64))
    ref = R.plain(_words(x), w.cpu().numpy(), s.cpu().numpy(), rows=rows)
    yw = _words(y)
    return {"M": m, "K": int(x.shape[1]), "N": int(y.shape[1]), "padded": m < E.MIN_ROWS, "dtype": str(y.dtype).removeprefix("torch."),
            "inv_s": float(ref["inv_s"]), "s_x": float(ref["s_x"]), "y32_absmax_ref_rows": int(np.abs(ref["y32"]).max(initial=0)),
            "words": int(yw.size), "cpu_mismatch": int((_words(cpu) != yw).sum()), "rerun_mismatch": int((_words(gpu) != yw).sum()),
            "ref_rows": rows.tolist(), "ref_mismatch": int((ref["y"] != yw[rows]).sum())}


def applied(model: torch.nn.Module) -> tuple[dict[str, dict[str, Any]], list[str]]:
    from vllm.model_executor.layers.linear import LinearBase
    return E.applied_record(model), [n for n, m in model.named_modules() if isinstance(m, LinearBase)]


def generate(llm: Any, prompts: list[str], max_tokens: int) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0), use_tqdm=False)
    return [{"prompt": p, "text": o.outputs[0].text, "token_ids": list(o.outputs[0].token_ids)} for p, o in zip(prompts, outs)]


def run(a: argparse.Namespace) -> dict[str, Any]:
    from verity_vllm.engine.build import build_engine, target_mismatches
    from verity_vllm.program.frontend.kernel_readback import int8_exact_arm_of
    from verity_vllm.program.frontend.target_profile import TargetProfile

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    entry = pin(a)
    doc["pin"] = {"repo": entry["repo"], "revision": entry["revision"], "recipe": entry["recipe"], "local_path": entry["local_path"]}
    (outdir / "pin_entry.json").write_text(json.dumps(entry, indent=1) + "\n")
    print(f"[int8] pin {json.dumps(doc['pin'])}", flush=True)

    target = TargetProfile.from_device().with_overrides(int8_exact=E.ARM)
    doc["target"] = target.to_json()
    with tempfile.TemporaryDirectory() as scratch:
        eng = build_engine(scratch_checkpoints(a.checkpoints, entry, Path(scratch)), case=a.role, target=target,
                           max_num_seqs=len(PROMPTS))
    doc["engine"] = {"build_s": round(eng.build_s, 1), "effective": eng.effective, "target_mismatches": target_mismatches(eng),
                     "int8_exact_arm": int8_exact_arm_of(eng.model), "quantization": eng.kwargs.get("quantization"),
                     "enforce_eager": eng.kwargs.get("enforce_eager")}
    print(f"[int8] engine {json.dumps(doc['engine'], default=str)[:800]}", flush=True)

    before, linears = applied(eng.model)
    names = list(before)
    taps = Taps(eng.model, names[:4] + names[-1:], a.keep_calls)
    taps.phase = "batched"
    batched = generate(eng.llm, PROMPTS, a.max_tokens)
    solo = []
    for i, p in enumerate(PROMPTS):
        taps.phase = "solo" if i == 0 else None
        solo += generate(eng.llm, [p], a.max_tokens)
    taps.phase = None
    taps.close()
    after, _ = applied(eng.model)
    served = {n: (after[n]["calls"] or 0) - (before[n]["calls"] or 0) for n in after}
    doc["applied"] = {"linears": len(linears), "recorded": len(after), "same_modules": sorted(linears) == sorted(after),
                      "methods": sorted({r["method"] for r in after.values()}), "arms": sorted({str(r["arm"]) for r in after.values()}),
                      "min_calls_during_generation": min(served.values(), default=0), "layers": after}
    doc["applied"]["ok"] = (doc["applied"]["same_modules"] and bool(linears) and doc["applied"]["methods"] == ["Int8PlainLinearMethod"]
                            and doc["applied"]["arms"] == [E.ARM] and doc["applied"]["min_calls_during_generation"] > 0)
    print(f"[int8] applied { {k: v for k, v in doc['applied'].items() if k != 'layers'} }", flush=True)

    doc["generations"] = {"batched": batched, "solo": solo,
                          "batch_dependent": [b["prompt"] for b, s in zip(batched, solo) if b["token_ids"] != s["token_ids"]]}
    for b, s in zip(batched, solo):
        print(f"[int8] gen {b['prompt']!r}\n  batched: {b['text']!r}\n  solo:    {s['text']!r}", flush=True)

    mods = dict(eng.model.named_modules())
    doc["equal"] = []
    for (phase, name), calls in sorted(taps.got.items()):
        for i, (x, y) in enumerate(calls):
            rec = {"phase": phase, "module": name, "call": i, **compare(mods[name], x, y, a.ref_rows)}
            doc["equal"].append(rec)
            print(f"[int8] equal {phase} {name} call {i}: M={rec['M']} K={rec['K']} N={rec['N']} padded={rec['padded']} "
                  f"cpu!={rec['cpu_mismatch']} ref!={rec['ref_mismatch']} rerun!={rec['rerun_mismatch']} inv_s={rec['inv_s']:.6g}",
                  flush=True)
    eq = doc["equal"]
    doc["equal_totals"] = {"compared": len(eq), "padded": sum(r["padded"] for r in eq), "words": sum(r["words"] for r in eq),
                           **{k: sum(r[k] for r in eq) for k in ("cpu_mismatch", "ref_mismatch", "rerun_mismatch")}}
    t = doc["equal_totals"]
    doc["ok"] = bool(doc["applied"]["ok"] and not doc["engine"]["target_mismatches"] and doc["engine"]["int8_exact_arm"] == E.ARM
                     and t["compared"] and t["padded"] and not (t["cpu_mismatch"] or t["ref_mismatch"] or t["rerun_mismatch"])
                     and all(g["token_ids"] for g in batched + solo))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"INT8-EXACT-GPU {'OK' if doc['ok'] else 'FAIL'} totals={json.dumps(t)} batch_dependent={len(doc['generations']['batch_dependent'])}",
          flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="int8 linears (TargetProfile.int8_exact) served by vLLM on this GPU, checked bit for bit")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD} and pin_entry.json")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="LLAMA31_8B")
    ap.add_argument("--role", default="LLAMA31_8B_INT8")
    ap.add_argument("--repo", default="verity-int8/Llama-3.1-8B-INT8")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--at", required=True, help="resolved_at of the pin entry (UTC, ISO 8601)")
    ap.add_argument("--by", required=True, help="resolved_by of the pin entry")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--keep-calls", type=int, default=3, help="calls kept per tapped module and phase")
    ap.add_argument("--ref-rows", type=int, default=6, help="rows per call checked against the NumPy reference")
    return 0 if run(ap.parse_args(argv))["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
