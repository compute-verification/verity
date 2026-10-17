"""VLLM-18 on one GPU: a BF16 pin's int8 checkpoint served with `TargetProfile.int8_exact = "plain"`, checked end to end.

One process writes `<out>/int8_exact_gpu.json` (and `<out>/pin_entry.json`, the int8 pin's checkpoints-int8.json entry):

  pin       `verity-vllm int8-checkpoint pin` in-process: the int8 snapshot made from the BF16 pin into the HF cache, its entry (every
            file's sha256, the pinned revision).  The engine reads it through a scratch copy of checkpoints.json and of
            checkpoints-int8.json with the entry appended as `downloaded` (the repo's manifests get the entry by hand).
  engine    `engine.build.build_engine(case=ROLE, target=from_device() + int8_exact="plain")`, which refuses a target mismatch
            (`int8_exact_arm_of` included).
  applied   `int8_exact.applied_record`: every linear's method, arm and calls, before and after generation, against vLLM's own
            `LinearBase` modules: every one runs `Int8PlainLinearMethod` (arm "plain") and served calls during generation.
  generate  greedy (temperature 0) on PROMPTS, batched and one by one; the texts, token ids, and which prompts' tokens differ
            between the two (attention stays bf16 FlashAttention, whose rows may depend on the batch).
  equal     forward hooks on the first four linears and the last one keep (x, y) of their first calls in the batched run and in
            every solo run (prefill and decode, so the M < 17 padding too).  Each served y, bit for bit, against
              oracle  `benchmarks/ppl/steps.py`'s `int8_linear` on the CPU (`tests/engine/ppl_oracle.int8_linear`) from x and the
                      BF16 base pin's own weight (q, k, v and gate, up concatenated by rows for vLLM's merged modules)
              cpu     `int8_exact.plain_linear` on the CPU on the served int8 weight and scales
              rerun   `int8_exact.plain_linear` again on the GPU
              alone   every row of the call recomputed alone on the GPU (M = 1, padded): a row's value is its own
            and the served weight against the oracle's `quant_rows(w.float(), 127)` of the base weight.
  batch     every solo-run row whose x is bitwise a row of the batched run's calls of the same module has the batched run's y
            row, bit for bit (rows past bf16 attention may differ in x, and are counted, not compared).

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

from tests.engine import ppl_oracle as O
from verity_vllm.config import checkpoint_entries
from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.engine import int8_checkpoint as I
from verity_vllm.engine import int8_exact as E

SCHEMA = "verity-vllm/int8-exact-gpu/v2"
RECORD = "int8_exact_gpu.json"
PROMPTS = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "The three primary colors are", "Water boils at a temperature of", "Once upon a time, in a small village,"]
MERGED = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj")}


def pin(a: argparse.Namespace) -> dict[str, Any]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = I.main(I.PinOptions(base_role=a.base_role, role=a.role, repo=a.repo, checkpoints=a.checkpoints, hf_home=a.hf_home,
                                 by=a.by, at=a.at))
    if rc:
        raise SystemExit(f"int8-checkpoint pin exited {rc}")
    return json.loads(out.getvalue())


def scratch_checkpoints(checkpoints: str, entry: dict[str, Any], scratch: Path) -> Path:
    p = scratch / "checkpoints.json"
    p.write_text(Path(checkpoints).read_text())
    sib = Path(checkpoints).with_name("checkpoints-int8.json")
    doc = json.loads(sib.read_text()) if sib.is_file() else {"checkpoints": []}
    doc["checkpoints"].append({**entry, "downloaded": True})
    (scratch / "checkpoints-int8.json").write_text(json.dumps(doc, indent=1) + "\n")
    return p


class BaseWeights:
    """The base pin's weight of a vLLM linear, read from its safetensors as stored (merged modules concatenated by rows)."""

    DTYPES = {"BF16": (np.int16, torch.bfloat16), "F32": (np.float32, torch.float32)}

    def __init__(self, snapshot: str):
        self.snapshot = Path(snapshot)
        self.headers = {p: F.read_header(p) for p in sorted(self.snapshot.glob("*.safetensors"))}
        self.where = {k: p for p, (_, h) in self.headers.items() for k in h if k != "__metadata__"}

    def tensor(self, name: str) -> torch.Tensor:
        p = self.where[name]
        start, header = self.headers[p]
        info = header[name]
        lo, hi = info["data_offsets"]
        np_dtype, dtype = self.DTYPES[info["dtype"]]
        raw = np.fromfile(p, dtype=np.uint8, count=hi - lo, offset=start + lo)
        return torch.from_numpy(raw.view(np_dtype).copy()).view(dtype).reshape(info["shape"])

    def weight(self, module: str) -> torch.Tensor:
        prefix, _, leaf = module.rpartition(".")
        return torch.cat([self.tensor(f"{prefix}.{p}.weight") for p in MERGED.get(leaf, (leaf,))])


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


def compare(mod: torch.nn.Module, w_base: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    w, s = mod.weight, mod.weight_scale
    m = x.shape[0]
    gpu = E.plain_linear(x.to(w.device), w, s).cpu()
    cpu = E.plain_linear(x, w.cpu(), s.cpu())
    oracle = O.int8_linear(x, w_base)
    alone = torch.cat([E.plain_linear(x[i:i + 1].to(w.device), w, s) for i in range(m)]).cpu()
    yw = _words(y)
    return {"M": m, "K": int(x.shape[1]), "N": int(y.shape[1]), "padded": m < E.MIN_ROWS, "dtype": str(y.dtype).removeprefix("torch."),
            "words": int(yw.size), "oracle_mismatch": int((_words(oracle) != yw).sum()), "cpu_mismatch": int((_words(cpu) != yw).sum()),
            "rerun_mismatch": int((_words(gpu) != yw).sum()), "alone_mismatch": int((_words(alone) != yw).sum())}


def weights_match(mod: torch.nn.Module, w_base: torch.Tensor) -> bool:
    q, s = O.int8_weights(w_base)
    return bool(torch.equal(mod.weight.cpu(), q) and torch.equal(mod.weight_scale.cpu().reshape(-1), s))


def batch_vs_solo(got: dict[tuple[str, str], list[tuple[torch.Tensor, torch.Tensor]]], names: list[str]) -> dict[str, Any]:
    """Per module and solo prompt: the solo rows whose x is bitwise a batched row of that module, and how many of their y rows
    differ from the batched y row."""
    out: dict[str, Any] = {}
    for name in names:
        batched: dict[bytes, bytes] = {}
        for x, y in got.get(("batched", name), []):
            for i in range(x.shape[0]):
                batched.setdefault(_words(x[i]).tobytes(), _words(y[i]).tobytes())
        per = []
        for phase in sorted(p for p, n in got if n == name and p.startswith("solo")):
            rows = matched = mismatch = 0
            for x, y in got[(phase, name)]:
                for i in range(x.shape[0]):
                    rows += 1
                    hit = batched.get(_words(x[i]).tobytes())
                    if hit is not None:
                        matched += 1
                        mismatch += hit != _words(y[i]).tobytes()
            per.append({"phase": phase, "rows": rows, "matched": matched, "mismatch": mismatch})
        out[name] = per
    return out


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
    base = BaseWeights(F.base_entry(entry, checkpoint_entries(a.checkpoints), I.INT8)["local_path"])

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
    tapped = names[:4] + names[-1:]
    taps = Taps(eng.model, tapped, a.keep_calls)
    taps.phase = "batched"
    batched = generate(eng.llm, PROMPTS, a.max_tokens)
    solo = []
    for i, p in enumerate(PROMPTS):
        taps.phase = f"solo{i}"
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
    w_base = {n: base.weight(n) for n in tapped}
    doc["weights_match_oracle"] = {n: weights_match(mods[n], w_base[n]) for n in tapped}
    print(f"[int8] weights_match_oracle {doc['weights_match_oracle']}", flush=True)
    doc["equal"] = []
    for (phase, name), calls in sorted(taps.got.items()):
        for i, (x, y) in enumerate(calls):
            rec = {"phase": phase, "module": name, "call": i, **compare(mods[name], w_base[name], x, y)}
            doc["equal"].append(rec)
            print(f"[int8] equal {phase} {name} call {i}: M={rec['M']} K={rec['K']} N={rec['N']} padded={rec['padded']} "
                  f"oracle!={rec['oracle_mismatch']} cpu!={rec['cpu_mismatch']} rerun!={rec['rerun_mismatch']} "
                  f"alone!={rec['alone_mismatch']}", flush=True)
    eq = doc["equal"]
    keys = ("oracle_mismatch", "cpu_mismatch", "rerun_mismatch", "alone_mismatch")
    doc["equal_totals"] = {"compared": len(eq), "padded": sum(r["padded"] for r in eq), "words": sum(r["words"] for r in eq),
                           **{k: sum(r[k] for r in eq) for k in keys}}
    doc["batch_vs_solo"] = batch_vs_solo(taps.got, tapped)
    bvs = [r for per in doc["batch_vs_solo"].values() for r in per]
    doc["batch_vs_solo_totals"] = {k: sum(r[k] for r in bvs) for k in ("rows", "matched", "mismatch")}
    t, b = doc["equal_totals"], doc["batch_vs_solo_totals"]
    doc["ok"] = bool(doc["applied"]["ok"] and not doc["engine"]["target_mismatches"] and doc["engine"]["int8_exact_arm"] == E.ARM
                     and all(doc["weights_match_oracle"].values()) and t["compared"] and t["padded"] and not any(t[k] for k in keys)
                     and b["matched"] and not b["mismatch"]
                     and all(g["token_ids"] for g in batched + solo))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"INT8-EXACT-GPU {'OK' if doc['ok'] else 'FAIL'} totals={json.dumps(t)} batch_vs_solo={json.dumps(b)} "
          f"batch_dependent_tokens={len(doc['generations']['batch_dependent'])}", flush=True)
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
    return 0 if run(ap.parse_args(argv))["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
